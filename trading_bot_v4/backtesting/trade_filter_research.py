"""Cost-aware chronological research for reducing baseline LONG turnover.

This module never changes the protected baseline.  It evaluates simple entry
gates on an earlier development segment and reports their performance on a
later validation segment so a gate cannot be promoted from full-history fit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.paper_model_comparison import predict_original_baseline_signals
from trading_bot_v4.utils.model_cache import ModelScalerCache


SUMMARY_PATH = Path("reports/v5_trade_filter_research.csv")
DETAIL_PATH = Path("reports/v5_trade_filter_validation_by_asset.csv")
CONFIG_PATH = Path("reports/v5_trade_filter_recommendation.json")
SIGNAL_CACHE_PATH = Path("data/cache/v5_original_long_research_signals.pkl")
MIN_DEVELOPMENT_TRADES = 100
MIN_VALIDATION_TRADES = 50


@dataclass(frozen=True)
class EntryGate:
    probability: float = 0.58
    trend_lookback: int = 0
    require_green: bool = False
    min_atr_pct: float = 0.0

    @property
    def name(self) -> str:
        return (
            f"p{self.probability:.2f}_ma{self.trend_lookback}_"
            f"green{int(self.require_green)}_atr{self.min_atr_pct:.2f}"
        )


def apply_entry_gate(signals: pd.DataFrame, gate: EntryGate) -> pd.DataFrame:
    result = signals.copy().sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(result["close"], errors="coerce")
    probability = pd.to_numeric(result["model_probability"], errors="coerce")
    atr_pct = pd.to_numeric(result["atr"], errors="coerce") / close.replace(0, np.nan) * 100.0
    allowed = probability.gt(gate.probability) & atr_pct.ge(gate.min_atr_pct)
    if gate.trend_lookback > 0:
        trend = close.rolling(gate.trend_lookback, min_periods=gate.trend_lookback).mean()
        allowed &= close.gt(trend)
    if gate.require_green:
        allowed &= close.gt(pd.to_numeric(result["open"], errors="coerce"))
    result["model_direction"] = np.where(allowed, "LONG", "HOLD")
    result["is_trade_candidate"] = allowed
    return result


def _aggregate(rows: list[dict[str, Any]], gate: EntryGate, segment: str) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    wins = pd.to_numeric(frame.get("net_wins", pd.Series(dtype=float)), errors="coerce").sum()
    losses = abs(pd.to_numeric(frame.get("net_losses", pd.Series(dtype=float)), errors="coerce").sum())
    return {
        "gate": gate.name,
        "segment": segment,
        **asdict(gate),
        "assets": int(len(frame)),
        "trades": int(frame["trades"].sum()) if not frame.empty else 0,
        "mean_return_pct": float(frame["return_pct"].mean()) if not frame.empty else 0.0,
        "median_return_pct": float(frame["return_pct"].median()) if not frame.empty else 0.0,
        "profitable_assets": int((frame["return_pct"] > 0).sum()) if not frame.empty else 0,
        "profit_factor": float(wins / losses) if losses else (float("inf") if wins else 0.0),
        "worst_drawdown_pct": float(frame["max_drawdown_pct"].min()) if not frame.empty else 0.0,
    }


def _evaluate(signals_by_symbol: dict[str, pd.DataFrame], gate: EntryGate,
              segment: str, starting_capital: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for symbol, all_signals in signals_by_symbol.items():
        cut = max(1, int(len(all_signals) * 0.70))
        signals = all_signals.iloc[:cut].copy() if segment == "development" else all_signals.iloc[cut:].copy()
        if len(signals) < 2:
            continue
        gated = apply_entry_gate(signals, gate)
        summary, pnl = _fast_one_candle_long(gated, symbol, starting_capital)
        rows.append({
            "symbol": symbol, "gate": gate.name, "segment": segment,
            **summary,
            "net_wins": float(pnl[pnl > 0].sum()), "net_losses": float(pnl[pnl < 0].sum()),
        })
    return _aggregate(rows, gate, segment), rows


def _fast_one_candle_long(signals: pd.DataFrame, symbol: str,
                          starting_capital: float) -> tuple[dict[str, Any], pd.Series]:
    """Equivalent fast path for the protected LONG one-candle contract."""
    frame = signals.reset_index(drop=True)
    candidate_indices = np.flatnonzero(frame["model_direction"].eq("LONG").to_numpy())
    candidate_indices = candidate_indices[candidate_indices < len(frame) - 1]
    capital, peak, drawdown = float(starting_capital), float(starting_capital), 0.0
    pnls: list[float] = []
    slip = (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) / 10000.0
    fee_rate = Config.PAPER_FEE_BPS / 10000.0
    for index in candidate_indices:
        row, nxt = frame.iloc[index], frame.iloc[index + 1]
        reference, atr = float(row["price"]), float(row["atr"])
        fill = reference * (1.0 + slip)
        stop_distance, target_distance = atr * Config.ATR_SL_MULTIPLIER, atr * Config.ATR_TP_MULTIPLIER
        if not np.isfinite(stop_distance) or stop_distance <= 0:
            continue
        notional = min(
            (capital * Config.RISK_PERCENTAGE / 100.0 / stop_distance) * reference,
            capital * Config.PAPER_MAX_POSITION_PCT / 100.0,
        )
        if notional < Config.PAPER_MIN_ORDER_USD:
            continue
        stop, target = fill - stop_distance, fill + target_distance
        if float(nxt["low"]) <= stop:
            exit_reference = stop
        elif float(nxt["high"]) >= target:
            exit_reference = target
        else:
            exit_reference = float(nxt["price"])
        exit_fill = exit_reference * (1.0 - slip)
        quantity = notional / fill
        net = (exit_fill - fill) * quantity - (notional * fee_rate * 2.0)
        capital += net
        pnls.append(net)
        peak = max(peak, capital)
        drawdown = min(drawdown, (capital / peak - 1.0) * 100.0)
    pnl = pd.Series(pnls, dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "symbol": symbol, "trades": int(len(pnl)), "return_pct": (capital / starting_capital - 1.0) * 100.0,
        "max_drawdown_pct": drawdown,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else 0.0),
    }, pnl


def candidate_gates() -> list[EntryGate]:
    return [EntryGate(*values) for values in itertools.product(
        (0.58, 0.65, 0.72, 0.80),
        (0, 50, 200),
        (False, True),
        (0.0, 0.5, 1.0),
    )]


def run_trade_filter_research(args: Any | None = None) -> pd.DataFrame:
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    capital = float(getattr(args, "capital", 100000.0))
    all_assets = bool(getattr(args, "all_assets", True))
    symbols = list_gmx_symbols(timeframe) if all_assets else [str(getattr(args, "symbol", Config.GMX_SYMBOL)).upper()]
    signals_by_symbol: dict[str, pd.DataFrame] = {}
    if SIGNAL_CACHE_PATH.exists():
        cached = pd.read_pickle(SIGNAL_CACHE_PATH)
        if cached.get("timeframe") == timeframe and set(symbols).issubset(cached.get("signals", {})):
            signals_by_symbol = {symbol: cached["signals"][symbol] for symbol in symbols}
    if not signals_by_symbol:
        model, scaler = ModelScalerCache().load()
        for symbol in symbols:
            try:
                signals_by_symbol[symbol] = predict_original_baseline_signals(
                    model, scaler, symbol, timeframe, closed_only=False
                )
            except Exception as exc:
                print(f"Skipping {symbol}: {exc}")
        SIGNAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"timeframe": timeframe, "signals": signals_by_symbol}, SIGNAL_CACHE_PATH)

    development: list[dict[str, Any]] = []
    development_detail: list[dict[str, Any]] = []
    for gate in candidate_gates():
        summary, rows = _evaluate(signals_by_symbol, gate, "development", capital)
        development.append(summary)
        development_detail.extend(rows)
    development_frame = pd.DataFrame(development)
    baseline_trades = max(1, int(development_frame.iloc[0]["trades"]))
    eligible = development_frame.loc[
        (development_frame["trades"] <= baseline_trades * 0.50)
        & (development_frame["trades"] >= MIN_DEVELOPMENT_TRADES)
        & (development_frame["profit_factor"] > 1.0)
        & (development_frame["mean_return_pct"] > 0.0)
    ].copy()
    if eligible.empty:
        # No candidate earned promotion; validate the least-bad PF candidate to
        # make the negative result explicit rather than silently loosening rules.
        selected_row = development_frame.sort_values(
            ["profit_factor", "mean_return_pct"], ascending=False
        ).iloc[0]
    else:
        selected_row = eligible.sort_values(
            ["profit_factor", "mean_return_pct", "trades"], ascending=[False, False, True]
        ).iloc[0]
    selected = EntryGate(
        float(selected_row["probability"]), int(selected_row["trend_lookback"]),
        bool(selected_row["require_green"]), float(selected_row["min_atr_pct"]),
    )
    baseline = EntryGate()
    validation_summaries, detail = [], []
    for gate in (baseline, selected):
        summary, rows = _evaluate(signals_by_symbol, gate, "validation", capital)
        validation_summaries.append(summary)
        detail.extend(rows)
    result = pd.concat([development_frame, pd.DataFrame(validation_summaries)], ignore_index=True)
    selected_validation = validation_summaries[1]
    baseline_validation = validation_summaries[0]
    promoted = bool(
        selected.name != baseline.name
        and selected_row["trades"] >= MIN_DEVELOPMENT_TRADES
        and selected_validation["trades"] >= MIN_VALIDATION_TRADES
        and selected_validation["trades"] <= baseline_validation["trades"] * 0.50
        and selected_validation["mean_return_pct"] > baseline_validation["mean_return_pct"]
        and selected_validation["profit_factor"] > max(1.0, baseline_validation["profit_factor"])
        and selected_validation["worst_drawdown_pct"] >= baseline_validation["worst_drawdown_pct"]
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(development_detail + detail).to_csv(DETAIL_PATH, index=False)
    recommendation = {
        "promoted": promoted,
        "selected_gate": asdict(selected),
        "development": selected_row.to_dict(),
        "validation_baseline": baseline_validation,
        "validation_selected": selected_validation,
        "requirements": (
            f">={MIN_DEVELOPMENT_TRADES} development and >={MIN_VALIDATION_TRADES} validation trades, "
            "<=50% baseline trades, higher validation return and PF > baseline and 1.0, no worse drawdown"
        ),
        "limitation": "Chronological signal validation, but the currently loaded model may have seen these candles during training.",
    }
    CONFIG_PATH.write_text(json.dumps(recommendation, indent=2, default=str), encoding="utf-8")
    print(json.dumps(recommendation, indent=2, default=str))
    return result
