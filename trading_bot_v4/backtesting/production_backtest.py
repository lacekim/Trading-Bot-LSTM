"""Backtest the directional model and protection rules used by the V5 scheduler.

This is intentionally separate from ``backtest_engine``.  That module is the
legacy, one-candle, upside-only research baseline and must not be presented as
a backtest of the production paper strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.bearish_model_paper import (
    combine_directional_signals,
    load_bearish_calibration,
    predict_bearish_signals,
)
from trading_bot_v4.execution.paper_model_comparison import predict_original_baseline_signals
from trading_bot_v4.ml.bearish_trainer import BEARISH_MODEL_PATH, BEARISH_SCALER_PATH
from trading_bot_v4.research.daily_research import DAILY_GO_STATUS_PATH
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache


logger = build_logger("v5_production_backtest")
SUMMARY_PATH = Path("reports/v5_production_backtest_summary.csv")
TRADES_PATH = Path("reports/v5_production_backtest_trades.csv")
PORTFOLIO_PATH = Path("reports/v5_production_backtest_portfolio.json")


@dataclass
class SimulatedPosition:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    notional: float
    entry_fee: float
    stop_price: float
    target_price: float
    entry_bar: int


def _current_long_symbols() -> set[str]:
    if not DAILY_GO_STATUS_PATH.exists():
        return set()
    status = pd.read_csv(DAILY_GO_STATUS_PATH)
    if not {"symbol", "decision"}.issubset(status.columns):
        return set()
    return set(status.loc[
        status["decision"].astype(str).str.upper().eq("GO"), "symbol"
    ].astype(str).str.upper())


def _current_short_symbols(calibration: dict[str, Any]) -> set[str]:
    if not calibration.get("promoted", False):
        return set()
    return {str(symbol).upper() for symbol in calibration.get("promoted_symbols", {})}


def _exit_reference(position: SimulatedPosition, row: pd.Series, bar: int) -> tuple[float, str] | None:
    high, low, candle_open = float(row["high"]), float(row["low"]), float(row["open"])
    if position.direction == "LONG":
        hit_stop = low <= position.stop_price
        hit_target = high >= position.target_price
    else:
        hit_stop = high >= position.stop_price
        hit_target = low <= position.target_price
    if hit_stop or hit_target:
        stop_first = Config.PAPER_STOP_TARGET_PRIORITY == "STOP_FIRST"
        use_stop = hit_stop and (not hit_target or stop_first)
        if use_stop:
            price = min(candle_open, position.stop_price) if position.direction == "LONG" else max(candle_open, position.stop_price)
            return price, "stop_loss"
        price = max(candle_open, position.target_price) if position.direction == "LONG" else min(candle_open, position.target_price)
        return price, "take_profit"
    direction = str(row["model_direction"]).upper()
    if direction in {"LONG", "SHORT"} and direction != position.direction:
        return float(row["price"]), "signal_reversal"
    max_candles = int(Config.PAPER_MAX_HOLD_CANDLES)
    if max_candles > 0 and bar - position.entry_bar >= max_candles:
        return float(row["price"]), "window_exit"
    return None


def simulate_production_symbol(signals: pd.DataFrame, symbol: str, starting_capital: float,
                               entry_eligible: bool = True) -> tuple[dict[str, Any], pd.DataFrame]:
    """Simulate persistent single-symbol positions using production costs/rules."""
    capital = float(starting_capital)
    peak = capital
    max_drawdown = 0.0
    position: SimulatedPosition | None = None
    trades: list[dict[str, Any]] = []
    last_entry_bar = -10**9
    slip = (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) / 10000.0
    fee_rate = Config.PAPER_FEE_BPS / 10000.0
    stop_factor = Config.PAPER_STOP_LOSS_PCT / 100.0
    target_factor = Config.PAPER_TAKE_PROFIT_PCT / 100.0

    ordered = signals.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for bar, row in ordered.iterrows():
        if position is not None:
            exit_event = _exit_reference(position, row, bar)
            if exit_event:
                reference, reason = exit_event
                exit_price = reference * (1 - slip if position.direction == "LONG" else 1 + slip)
                gross = (exit_price - position.entry_price) * position.quantity
                if position.direction == "SHORT":
                    gross *= -1
                exit_fee = position.notional * fee_rate
                net = gross - position.entry_fee - exit_fee
                capital += net
                trades.append({
                    "symbol": symbol, "direction": position.direction,
                    "entry_time": position.entry_time, "exit_time": row["timestamp"],
                    "entry_price": position.entry_price, "exit_price": exit_price,
                    "notional": position.notional, "gross_pnl": gross,
                    "fees": position.entry_fee + exit_fee, "net_pnl": net,
                    "return_pct": net / position.notional * 100.0 if position.notional else 0.0,
                    "exit_reason": reason,
                })
                position = None

        direction = str(row["model_direction"]).upper()
        row_entry_eligible = bool(row.get("_entry_eligible", entry_eligible))
        if (position is None and row_entry_eligible and direction in {"LONG", "SHORT"}
                and bar - last_entry_bar >= Config.PAPER_MIN_BARS_BETWEEN_TRADES):
            notional = min(
                capital * Config.PAPER_MAX_RISK_PER_TRADE / max(Config.PAPER_STOP_LOSS_PCT, 0.0001),
                capital * Config.PAPER_MAX_POSITION_PCT / 100.0,
            )
            if notional >= Config.PAPER_MIN_ORDER_USD:
                reference = float(row["price"])
                fill = reference * (1 + slip if direction == "LONG" else 1 - slip)
                fee = notional * fee_rate
                quantity = notional / fill
                position = SimulatedPosition(
                    direction=direction, entry_time=pd.Timestamp(row["timestamp"]), entry_price=fill,
                    quantity=quantity, notional=notional, entry_fee=fee,
                    stop_price=fill * (1 - stop_factor if direction == "LONG" else 1 + stop_factor),
                    target_price=fill * (1 + target_factor if direction == "LONG" else 1 - target_factor),
                    entry_bar=bar,
                )
                last_entry_bar = bar

        marked = capital
        if position is not None:
            move = (float(row["price"]) - position.entry_price) * position.quantity
            marked += move if position.direction == "LONG" else -move
        peak = max(peak, marked)
        max_drawdown = min(max_drawdown, (marked / peak - 1.0) * 100.0)

    # Close remaining exposure at the last available close so results are finite.
    if position is not None and not ordered.empty:
        row = ordered.iloc[-1]
        exit_price = float(row["price"]) * (1 - slip if position.direction == "LONG" else 1 + slip)
        gross = (exit_price - position.entry_price) * position.quantity
        if position.direction == "SHORT":
            gross *= -1
        exit_fee = position.notional * fee_rate
        net = gross - position.entry_fee - exit_fee
        capital += net
        trades.append({
            "symbol": symbol, "direction": position.direction,
            "entry_time": position.entry_time, "exit_time": row["timestamp"],
            "entry_price": position.entry_price, "exit_price": exit_price,
            "notional": position.notional, "gross_pnl": gross,
            "fees": position.entry_fee + exit_fee, "net_pnl": net,
            "return_pct": net / position.notional * 100.0 if position.notional else 0.0,
            "exit_reason": "end_of_data",
        })

    trade_frame = pd.DataFrame(trades)
    wins = trade_frame.loc[trade_frame.get("net_pnl", pd.Series(dtype=float)) > 0] if not trade_frame.empty else trade_frame
    losses = trade_frame.loc[trade_frame.get("net_pnl", pd.Series(dtype=float)) < 0] if not trade_frame.empty else trade_frame
    gross_wins = float(wins["net_pnl"].sum()) if not wins.empty else 0.0
    gross_losses = abs(float(losses["net_pnl"].sum())) if not losses.empty else 0.0
    summary = {
        "symbol": symbol, "entry_eligible": bool(entry_eligible),
        "candles": int(len(ordered)), "trades": int(len(trade_frame)),
        "long_trades": int((trade_frame["direction"] == "LONG").sum()) if not trade_frame.empty else 0,
        "short_trades": int((trade_frame["direction"] == "SHORT").sum()) if not trade_frame.empty else 0,
        "starting_capital": starting_capital, "final_capital": capital,
        "return_pct": (capital / starting_capital - 1.0) * 100.0 if starting_capital else 0.0,
        "win_rate_pct": len(wins) / len(trade_frame) * 100.0 if len(trade_frame) else 0.0,
        "max_drawdown_pct": max_drawdown,
        "profit_factor": gross_wins / gross_losses if gross_losses else (float("inf") if gross_wins else 0.0),
    }
    return summary, trade_frame


def run_production_backtest(args: Any) -> pd.DataFrame:
    """Evaluate both current directional models; qualification is reported explicitly."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    starting_capital = float(getattr(args, "capital", 100000.0))
    use_all = bool(getattr(args, "all_assets", False))
    symbols = list_gmx_symbols(timeframe) if use_all else [str(getattr(args, "symbol", Config.GMX_SYMBOL)).upper()]
    long_model, long_scaler = ModelScalerCache().load()
    bearish_model, bearish_scaler = ModelScalerCache(BEARISH_MODEL_PATH, BEARISH_SCALER_PATH).load()
    calibration = load_bearish_calibration()
    bearish_threshold = float(calibration["threshold"])
    eligible_longs = _current_long_symbols()
    eligible_shorts = _current_short_symbols(calibration)
    ignore_qualification = bool(getattr(args, "ignore_qualification", False))

    summaries: list[dict[str, Any]] = []
    trades: list[pd.DataFrame] = []
    for symbol in symbols:
        symbol = str(symbol).upper()
        try:
            long_signals = predict_original_baseline_signals(long_model, long_scaler, symbol, timeframe)
            short_signals = predict_bearish_signals(
                bearish_model, bearish_scaler, symbol, timeframe,
                float(calibration.get("promoted_symbols", {}).get(symbol, bearish_threshold)),
            )
            combined = combine_directional_signals(long_signals, short_signals)
            eligible = ignore_qualification or symbol in eligible_longs or symbol in eligible_shorts
            # Direction-specific eligibility exactly mirrors the scheduler.
            combined["_entry_eligible"] = (
                ignore_qualification
                | (combined["model_direction"].eq("LONG") & (symbol in eligible_longs))
                | (combined["model_direction"].eq("SHORT") & (symbol in eligible_shorts))
            )
            summary, asset_trades = simulate_production_symbol(combined, symbol, starting_capital, eligible)
            summary["qualified_long"] = ignore_qualification or symbol in eligible_longs
            summary["qualified_short"] = ignore_qualification or symbol in eligible_shorts
            summaries.append(summary)
            if not asset_trades.empty:
                trades.append(asset_trades)
        except Exception as exc:
            logger.warning("Production backtest failed for %s: %s", symbol, exc)

    summary_df = pd.DataFrame(summaries)
    trades_df = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    trades_df.to_csv(TRADES_PATH, index=False)
    eligible = summary_df.loc[summary_df["entry_eligible"]] if not summary_df.empty else summary_df
    portfolio = {
        "engine": "V5_ORIGINAL_LONG_PLUS_BEARISH_SHORT",
        "timeframe": timeframe,
        "qualification_policy": (
            "IGNORED FOR DIAGNOSTIC RESEARCH; every selected asset may trade both directions"
            if ignore_qualification else
            "current daily GO list for LONG; current untouched-holdout promotion for SHORT"
        ),
        "qualified_long_symbols": sorted(eligible_longs),
        "qualified_short_symbols": sorted(eligible_shorts),
        "assets_tested": int(len(summary_df)),
        "eligible_assets": int(len(eligible)),
        "equal_weight_return_pct": float(eligible["return_pct"].mean()) if not eligible.empty else 0.0,
        "warning": (
            "Diagnostic only: qualification was bypassed and model results may include training-period observations."
            if ignore_qualification else
            "Current-snapshot qualification is not a historical daily walk-forward selector. Model results may include training-period observations."
        ),
    }
    PORTFOLIO_PATH.write_text(json.dumps(portfolio, indent=2), encoding="utf-8")
    print(f"V5 production backtest summary saved to {SUMMARY_PATH}")
    print(f"V5 production trades saved to {TRADES_PATH}")
    print(f"V5 production portfolio saved to {PORTFOLIO_PATH}")
    print(json.dumps(portfolio, indent=2))
    return summary_df
