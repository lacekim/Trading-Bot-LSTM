"""Backtest the directional model and protection rules used by the V5 scheduler.

This is intentionally separate from ``backtest_engine``.  That module is the
legacy, one-candle, upside-only research baseline and must not be presented as
a backtest of the production paper strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
from trading_bot_v4.backtesting.baseline_contract import active_execution_parity_audit
from trading_bot_v4.risk.market_cap_tiers import AssetRiskDecision, evaluate_asset_risk
from trading_bot_v4.utils.macd_confirmation import macd_entry_confirmation
from trading_bot_v4.execution.persistent_policy import policy_for


logger = build_logger("v5_production_backtest")
SUMMARY_PATH = Path("reports/v5_production_backtest_summary.csv")
TRADES_PATH = Path("reports/v5_production_backtest_trades.csv")
PORTFOLIO_PATH = Path("reports/v5_production_backtest_portfolio.json")
SIGNAL_CACHE_DIR = Path("data/cache/production_directional_signals")


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


def _cached_directional_signals(symbol: str, timeframe: str, long_model: Any, long_scaler: Any,
                                bearish_model: Any, bearish_scaler: Any,
                                bearish_threshold: float) -> pd.DataFrame:
    """Persist expensive full-history inference for repeated exit research."""
    source = Path("data/GMX_OHLCVT") / f"gmx_arbitrum_{symbol}_{timeframe}.csv"
    dependencies = [
        source, Config.MODEL_DIR / Config.MODEL_NAME, Config.MODEL_DIR / Config.SCALER_NAME,
        BEARISH_MODEL_PATH, BEARISH_SCALER_PATH, Path(__file__).parents[1] / "utils/macd_confirmation.py",
    ]
    signature = "|".join(
        f"{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}" for path in dependencies if path.exists()
    )
    key = hashlib.sha256(signature.encode()).hexdigest()[:16]
    path = SIGNAL_CACHE_DIR / f"{symbol}_{timeframe}_{key}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    long_signals = predict_original_baseline_signals(
        long_model, long_scaler, symbol, timeframe, closed_only=False
    )
    short_signals = predict_bearish_signals(
        bearish_model, bearish_scaler, symbol, timeframe, bearish_threshold
    )
    combined = combine_directional_signals(long_signals, short_signals)
    SIGNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(path)
    return combined


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
            return position.stop_price, "stop_loss"
        return position.target_price, "take_profit"
    direction = str(row["model_direction"]).upper()
    if direction in {"LONG", "SHORT"} and direction != position.direction:
        return float(row["price"]), "signal_reversal"
    max_candles = int(policy_for(str(row.get("symbol", ""))).max_hold_candles)
    if max_candles > 0 and bar - position.entry_bar >= max_candles:
        return float(row["price"]), "window_exit"
    return None


def simulate_production_symbol(signals: pd.DataFrame, symbol: str, starting_capital: float,
                               entry_eligible: bool = True, fee_bps: float | None = None,
                               slippage_bps: float | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """Simulate persistent single-symbol positions using production costs/rules."""
    capital = float(starting_capital)
    peak = capital
    max_drawdown = 0.0
    position: SimulatedPosition | None = None
    trades: list[dict[str, Any]] = []
    last_entry_bar = -10**9
    effective_slippage_bps = (
        Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS
        if slippage_bps is None else float(slippage_bps)
    )
    effective_fee_bps = Config.PAPER_FEE_BPS if fee_bps is None else float(fee_bps)
    slip = effective_slippage_bps / 10000.0
    fee_rate = effective_fee_bps / 10000.0

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
            policy = policy_for(symbol)
            atr = float(row.get("atr", float("nan")))
            legacy_atr_fallback = not np.isfinite(atr) or atr <= 0
            if legacy_atr_fallback:
                atr = (
                    float(row["price"]) * Config.PAPER_STOP_LOSS_PCT / 100.0
                    / max(policy.stop_atr, 1e-12)
                )
            stop_distance = atr * policy.stop_atr
            target_distance = atr * policy.target_atr
            tier_risk_pct = (float(row.get("_risk_pct", Config.RISK_PERCENTAGE))
                             if policy.use_tier_sizing else Config.RISK_PERCENTAGE)
            tier_max_position_pct = (float(row.get("_max_position_pct", Config.PAPER_MAX_POSITION_PCT))
                                     if policy.use_tier_sizing else Config.PAPER_MAX_POSITION_PCT)
            risk_amount = capital * min(Config.RISK_PERCENTAGE, tier_risk_pct) / 100.0
            risk_sized_notional = (risk_amount / stop_distance) * float(row["price"])
            notional = min(
                risk_sized_notional,
                capital * min(Config.PAPER_MAX_POSITION_PCT, tier_max_position_pct) / 100.0,
            )
            if notional >= Config.PAPER_MIN_ORDER_USD:
                reference = float(row["price"])
                fill = reference * (1 + slip if direction == "LONG" else 1 - slip)
                fee = notional * fee_rate
                quantity = notional / fill
                if legacy_atr_fallback:
                    stop_factor = Config.PAPER_STOP_LOSS_PCT / 100.0
                    target_factor = Config.PAPER_TAKE_PROFIT_PCT / 100.0
                    stop_price = fill * (1 - stop_factor if direction == "LONG" else 1 + stop_factor)
                    target_price = fill * (1 + target_factor if direction == "LONG" else 1 - target_factor)
                else:
                    stop_price = fill - stop_distance if direction == "LONG" else fill + stop_distance
                    target_price = fill + target_distance if direction == "LONG" else fill - target_distance
                position = SimulatedPosition(
                    direction=direction, entry_time=pd.Timestamp(row["timestamp"]), entry_price=fill,
                    quantity=quantity, notional=notional, entry_fee=fee,
                    stop_price=stop_price, target_price=target_price,
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
    model_sides = str(getattr(args, "model_sides", "both")).lower()
    symbols = list_gmx_symbols(timeframe) if use_all else [str(getattr(args, "symbol", Config.GMX_SYMBOL)).upper()]
    long_model, long_scaler = ModelScalerCache().load()
    bearish_model = bearish_scaler = None
    if model_sides in {"short", "both"}:
        bearish_model, bearish_scaler = ModelScalerCache(BEARISH_MODEL_PATH, BEARISH_SCALER_PATH).load()
    calibration = load_bearish_calibration()
    bearish_threshold = float(calibration["threshold"])
    eligible_longs = _current_long_symbols()
    eligible_shorts = _current_short_symbols(calibration)
    ignore_qualification = bool(getattr(args, "ignore_qualification", False))
    zero_costs = bool(getattr(args, "zero_costs", False))
    fee_bps = 0.0 if zero_costs else float(Config.PAPER_FEE_BPS)
    slippage_bps = 0.0 if zero_costs else float(Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS)

    summaries: list[dict[str, Any]] = []
    trades: list[pd.DataFrame] = []
    for symbol in symbols:
        symbol = str(symbol).upper()
        try:
            if Config.MARKET_CAP_RISK_ENABLED:
                direction_risk = {
                    direction: evaluate_asset_risk(symbol, direction)
                    for direction in ("LONG", "SHORT")
                }
            else:
                direction_risk = {
                    direction: AssetRiskDecision(
                        True, "DISABLED", 0.0, Config.RISK_PERCENTAGE,
                        Config.PAPER_MAX_POSITION_PCT,
                    ) for direction in ("LONG", "SHORT")
                }
            if model_sides == "both":
                combined = _cached_directional_signals(
                    symbol, timeframe, long_model, long_scaler, bearish_model, bearish_scaler,
                    float(calibration.get("promoted_symbols", {}).get(symbol, bearish_threshold)),
                )
            else:
                long_signals = (
                    predict_original_baseline_signals(long_model, long_scaler, symbol, timeframe, closed_only=False)
                    if model_sides == "long" else pd.DataFrame()
                )
                short_signals = (
                    predict_bearish_signals(
                        bearish_model, bearish_scaler, symbol, timeframe,
                        float(calibration.get("promoted_symbols", {}).get(symbol, bearish_threshold)),
                    ) if model_sides == "short" else pd.DataFrame()
                )
                combined = combine_directional_signals(long_signals, short_signals)
            qualification_eligible = ignore_qualification or symbol in eligible_longs or symbol in eligible_shorts
            # Direction-specific eligibility exactly mirrors the scheduler.
            combined["_entry_eligible"] = (
                ignore_qualification
                | (combined["model_direction"].eq("LONG") & (symbol in eligible_longs))
                | (combined["model_direction"].eq("SHORT") & (symbol in eligible_shorts))
            )
            combined["_asset_risk_allowed"] = combined["model_direction"].map(
                {direction: decision.allowed for direction, decision in direction_risk.items()}
            ).fillna(False)
            combined["_risk_pct"] = combined["model_direction"].map(
                {direction: decision.risk_pct for direction, decision in direction_risk.items()}
            ).fillna(0.0)
            combined["_max_position_pct"] = combined["model_direction"].map(
                {direction: decision.max_position_pct for direction, decision in direction_risk.items()}
            ).fillna(0.0)
            combined["_entry_eligible"] &= combined["_asset_risk_allowed"]
            if policy_for(symbol).require_macd:
                combined["_entry_eligible"] &= macd_entry_confirmation(combined)
            # Retain the final row so it can close a position opened on the
            # preceding candle, but never open a position without a following
            # candle available for its one-candle window exit.
            if not combined.empty:
                last_index = combined["timestamp"].idxmax()
                combined.loc[last_index, "_entry_eligible"] = False
            summary, asset_trades = simulate_production_symbol(
                combined, symbol, starting_capital,
                qualification_eligible and any(decision.allowed for decision in direction_risk.values()),
                fee_bps=fee_bps, slippage_bps=slippage_bps,
            )
            summary["qualified_long"] = ignore_qualification or symbol in eligible_longs
            summary["qualified_short"] = ignore_qualification or symbol in eligible_shorts
            summary["long_risk_allowed"] = direction_risk["LONG"].allowed
            summary["short_risk_allowed"] = direction_risk["SHORT"].allowed
            summary["market_cap_tier"] = direction_risk["LONG"].tier
            summary["market_cap_usd"] = direction_risk["LONG"].market_cap_usd
            summary["tier_risk_pct"] = direction_risk["LONG"].risk_pct
            summary["tier_max_position_pct"] = direction_risk["LONG"].max_position_pct
            summary["long_risk_reason"] = direction_risk["LONG"].reason
            summary["short_risk_reason"] = direction_risk["SHORT"].reason
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
        "model_sides": model_sides,
        "zero_costs": zero_costs,
        "qualification_policy": (
            "IGNORED FOR DIAGNOSTIC RESEARCH; every selected asset may trade both directions"
            if ignore_qualification else
            "current daily GO list for LONG; current untouched-holdout promotion for SHORT"
        ),
        "qualified_long_symbols": sorted(eligible_longs),
        "qualified_short_symbols": sorted(eligible_shorts),
        "assets_tested": int(len(summary_df)),
        "eligible_assets": int(len(eligible)),
        "market_cap_risk_enabled": bool(Config.MARKET_CAP_RISK_ENABLED),
        "risk_allowed_long_assets": int(summary_df["long_risk_allowed"].sum()) if not summary_df.empty else 0,
        "risk_allowed_short_assets": int(summary_df["short_risk_allowed"].sum()) if not summary_df.empty else 0,
        "equal_weight_return_pct": float(eligible["return_pct"].mean()) if not eligible.empty else 0.0,
        "warning": (
            "Diagnostic only: qualification was bypassed and model results may include training-period observations. "
            "Market-cap and GMX-depth admission use the latest available snapshot across history; historical cap tiers, "
            "historical executable spreads, and shared-portfolio small-cap concurrency cannot be reconstructed by this independent per-asset test."
            if ignore_qualification else
            "Current-snapshot qualification is not a historical daily walk-forward selector. Model results may include training-period observations. "
            "Market-cap and GMX-depth admission use the latest available snapshot across history."
        ),
        "baseline_parity": active_execution_parity_audit(
            fee_bps, slippage_bps, symbols[0] if len(symbols) == 1 else ""
        ),
    }
    PORTFOLIO_PATH.write_text(json.dumps(portfolio, indent=2), encoding="utf-8")
    print(f"V5 production backtest summary saved to {SUMMARY_PATH}")
    print(f"V5 production trades saved to {TRADES_PATH}")
    print(f"V5 production portfolio saved to {PORTFOLIO_PATH}")
    print(json.dumps(portfolio, indent=2))
    return summary_df
