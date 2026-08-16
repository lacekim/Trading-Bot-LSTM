"""Experiment: multi-day momentum horizon instead of short-horizon direction.

Three prior experiments (LSTM/threshold-target, LSTM/triple-barrier,
GBM+relative-strength/triple-barrier) all converged on holdout AUC ~0.50-0.53
-- no detectable edge -- while predicting direction 1-8 hours out
(PAPER_MAX_HOLD_CANDLES=8 on a 1h timeframe). That's the hardest version of
this problem: short-horizon directional prediction from public technical
indicators is exactly what liquid, arbitraged markets remove fastest, so ~0.50
AUC there is close to the expected null result, not evidence of a bad model.

Time-series momentum -- an asset that has been trending keeps trending, over
days to weeks -- is the most robustly documented cross-asset anomaly in the
empirical finance literature (Moskowitz/Ooi/Pedersen 2012 and much follow-up
work; shown across equities, futures, currencies, and crypto specifically).
This experiment tests that hypothesis directly: same GBM + BTC relative-
strength feature set and triple-barrier labeling machinery already built and
validated, but with the holding horizon stretched from ~8 hours to ~4 days
(96 hourly candles), and stop/target distances scaled by sqrt(hold period)
against the same hourly ATR (standard volatility time-scaling, not arbitrary
numbers) so a multi-day hold isn't stopped out by ordinary hourly noise.

This does NOT go through policy_for()/simulate_production_symbol's default
8-candle policy -- it intentionally tests a different strategy shape, so it
uses its own simulator (_simulate_symbol_swing_policy below), which mirrors
simulate_production_symbol's cost/sizing/exit logic exactly but takes
stop/target/max_hold as explicit parameters instead of the fixed default
policy. Writes to models/experiments/momentum_horizon/, never touches
production models/{long,short}/ artifacts.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from trading_bot import list_gmx_symbols
from trading_bot_v4.backtesting.production_backtest import SimulatedPosition
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from trading_bot_v4.ml.per_asset_trainer import MIN_CALIBRATION_TRADES, MIN_WINDOW_TRADES
from trading_bot_v4.risk.market_cap_tiers import liquid_symbols
from trading_bot_v4.utils.logger import build_logger
from train_model import DataHandler

logger = build_logger("v4_momentum_horizon_experiment")

Direction = Literal["long", "short"]

TIMEFRAME = "1h"
EXPERIMENT_ROOT = Path("models/experiments/momentum_horizon")
RELATIVE_STRENGTH_COLUMNS = ["btc_relative_return", "btc_relative_return_ma20", "btc_return_24h"]
FEATURE_COLUMNS = [*Config.FEATURE_COLUMNS, *RELATIVE_STRENGTH_COLUMNS]

# Swing policy: ~4 days on 1h candles. Stop/target distances scale the
# hourly ATR by sqrt(hold period) -- standard volatility time-scaling -- then
# apply the same 1:2 risk:reward ratio already used everywhere else in this
# codebase (ATR_SL_MULTIPLIER/ATR_TP_MULTIPLIER), instead of picking new
# arbitrary multipliers for a horizon nobody has tuned for.
MAX_HOLD_CANDLES = 96
_HOLD_SCALE = float(np.sqrt(MAX_HOLD_CANDLES))
SWING_STOP_ATR = Config.ATR_SL_MULTIPLIER * _HOLD_SCALE
SWING_TARGET_ATR = Config.ATR_TP_MULTIPLIER * _HOLD_SCALE

MIN_TOTAL_ROWS = 1500
MIN_SEGMENT_ROWS = 150
TRAIN_END, CALIBRATION_END = 0.70, 0.85
SIMULATION_STARTING_CAPITAL = 100_000.0

_ENDPOINT_COLUMNS = ["timestamp", "future_return", "target", "Open", "High", "Low", "Close", "ATR"]


@lru_cache(maxsize=4)
def _btc_return_frame(timeframe: str) -> pd.DataFrame:
    df = load_gmx_ohlc("BTC", timeframe)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    close = df["Close"].astype(float)
    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "btc_return_1h": np.log(close / close.shift(1)),
        "btc_return_24h": close.pct_change(24),
    })


def _swing_triple_barrier_target(df: pd.DataFrame, direction: Direction) -> pd.Series:
    """Same triple-barrier mechanics as run_triple_barrier_target_experiment,
    but with sqrt(hold)-scaled stop/target distances and a much longer
    max_hold, to label a multi-day swing entry instead of an ~8h one."""
    close = df["Close"].to_numpy(np.float64)
    high = df["High"].to_numpy(np.float64)
    low = df["Low"].to_numpy(np.float64)
    atr = df["ATR"].to_numpy(np.float64)
    n = len(df)
    stop_first = Config.PAPER_STOP_TARGET_PRIORITY == "STOP_FIRST"

    stop_distance = atr * SWING_STOP_ATR
    target_distance = atr * SWING_TARGET_ATR
    if direction == "long":
        stop_price = close - stop_distance
        target_price = close + target_distance
    else:
        stop_price = close + stop_distance
        target_price = close - target_distance

    exit_price = np.full(n, np.nan)
    resolved = np.zeros(n, dtype=bool)
    idx = np.arange(n)

    for offset in range(1, MAX_HOLD_CANDLES + 1):
        forward_idx = idx + offset
        valid = forward_idx < n
        forward_high = np.full(n, np.nan)
        forward_low = np.full(n, np.nan)
        forward_high[valid] = high[forward_idx[valid]]
        forward_low[valid] = low[forward_idx[valid]]
        if direction == "long":
            hit_stop = forward_low <= stop_price
            hit_target = forward_high >= target_price
        else:
            hit_stop = forward_high >= stop_price
            hit_target = forward_low <= target_price
        use_stop = hit_stop & (~hit_target | stop_first)
        use_target = hit_target & ~use_stop
        newly = ~resolved & valid & (use_stop | use_target)
        exit_price[newly] = np.where(use_stop[newly], stop_price[newly], target_price[newly])
        resolved |= newly

    time_idx = idx + MAX_HOLD_CANDLES
    valid_time = time_idx < n
    time_exit = ~resolved & valid_time
    exit_price[time_exit] = close[time_idx[time_exit]]
    resolved |= time_exit

    slip = (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) / 10000.0
    fee_rate = Config.PAPER_FEE_BPS / 10000.0
    round_trip_cost = 2 * (slip + fee_rate)

    raw_return = (exit_price - close) / close
    if direction == "short":
        raw_return = -raw_return
    net_return = raw_return - round_trip_cost

    label = np.where(net_return > 0, 1.0, 0.0)
    label = np.where(resolved, label, np.nan)
    return pd.Series(label, index=df.index)


def _load_symbol_frame(symbol: str, timeframe: str, direction: Direction) -> pd.DataFrame | None:
    df = load_gmx_ohlc(symbol, timeframe)
    if df is None or df.empty:
        return None

    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df = df.copy()
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr
    df = handler.prepare_features(df, prediction_horizon=1)
    if df.empty:
        return None

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})

    btc = _btc_return_frame(timeframe)
    df = df.merge(btc, on="timestamp", how="left")
    df[["btc_return_1h", "btc_return_24h"]] = df[["btc_return_1h", "btc_return_24h"]].ffill()
    df["btc_relative_return"] = df["returns"] - df["btc_return_1h"]
    df["btc_relative_return_ma20"] = df["btc_relative_return"].rolling(20).mean()
    df = df.dropna(subset=RELATIVE_STRENGTH_COLUMNS).reset_index(drop=True)
    if df.empty:
        return None

    df["target"] = _swing_triple_barrier_target(df, direction)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    n = len(df)
    train_end = int(n * TRAIN_END)
    calibration_end = int(n * CALIBRATION_END)
    boundaries = (train_end, calibration_end - train_end, n - calibration_end)
    if any(size < MIN_SEGMENT_ROWS for size in boundaries):
        return None
    return df.iloc[:train_end], df.iloc[train_end:calibration_end], df.iloc[calibration_end:]


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.10, random_state=42,
    )


def _exit_reference_swing(position: SimulatedPosition, row: pd.Series, bar: int) -> tuple[float, str] | None:
    high, low = float(row["high"]), float(row["low"])
    if position.direction == "LONG":
        hit_stop = low <= position.stop_price
        hit_target = high >= position.target_price
    else:
        hit_stop = high >= position.stop_price
        hit_target = low <= position.target_price
    if hit_stop or hit_target:
        stop_first = Config.PAPER_STOP_TARGET_PRIORITY == "STOP_FIRST"
        use_stop = hit_stop and (not hit_target or stop_first)
        return (position.stop_price, "stop_loss") if use_stop else (position.target_price, "take_profit")
    direction = str(row["model_direction"]).upper()
    if direction in {"LONG", "SHORT"} and direction != position.direction:
        return float(row["price"]), "signal_reversal"
    if bar - position.entry_bar >= MAX_HOLD_CANDLES:
        return float(row["price"]), "window_exit"
    return None


def _simulate_symbol_swing_policy(signals: pd.DataFrame, symbol: str, starting_capital: float) -> dict[str, Any]:
    """Mirrors production_backtest.simulate_production_symbol's cost/sizing/exit
    logic exactly, but reads stop/target/max_hold from this experiment's swing
    policy instead of policy_for(symbol) -- the multi-day hold this experiment
    tests isn't a policy the live default/persistent config represents."""
    capital = float(starting_capital)
    peak = capital
    max_drawdown = 0.0
    position: SimulatedPosition | None = None
    trades: list[dict[str, Any]] = []
    last_entry_bar = -10**9
    slip = (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) / 10000.0
    fee_rate = Config.PAPER_FEE_BPS / 10000.0

    ordered = signals.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for bar, row in ordered.iterrows():
        if position is not None:
            exit_event = _exit_reference_swing(position, row, bar)
            if exit_event:
                reference, reason = exit_event
                exit_price = reference * (1 - slip if position.direction == "LONG" else 1 + slip)
                gross = (exit_price - position.entry_price) * position.quantity
                if position.direction == "SHORT":
                    gross *= -1
                exit_fee = position.notional * fee_rate
                net = gross - position.entry_fee - exit_fee
                capital += net
                trades.append({"net_pnl": net, "notional": position.notional, "exit_reason": reason})
                position = None

        direction = str(row["model_direction"]).upper()
        if (position is None and direction in {"LONG", "SHORT"}
                and bar - last_entry_bar >= Config.PAPER_MIN_BARS_BETWEEN_TRADES):
            atr = float(row.get("atr", float("nan")))
            if np.isfinite(atr) and atr > 0:
                stop_distance = atr * SWING_STOP_ATR
                target_distance = atr * SWING_TARGET_ATR
                risk_amount = capital * Config.RISK_PERCENTAGE / 100.0
                risk_sized_notional = (risk_amount / stop_distance) * float(row["price"])
                notional = min(risk_sized_notional, capital * Config.PAPER_MAX_POSITION_PCT / 100.0)
                if notional >= Config.PAPER_MIN_ORDER_USD:
                    reference = float(row["price"])
                    fill = reference * (1 + slip if direction == "LONG" else 1 - slip)
                    fee = notional * fee_rate
                    quantity = notional / fill
                    stop_price = fill - stop_distance if direction == "LONG" else fill + stop_distance
                    target_price = fill + target_distance if direction == "LONG" else fill - target_distance
                    position = SimulatedPosition(
                        direction=direction, entry_time=pd.Timestamp(row["timestamp"]), entry_price=fill,
                        quantity=quantity, notional=notional, entry_fee=fee,
                        stop_price=stop_price, target_price=target_price, entry_bar=bar,
                    )
                    last_entry_bar = bar

        marked = capital
        if position is not None:
            move = (float(row["price"]) - position.entry_price) * position.quantity
            marked += move if position.direction == "LONG" else -move
        peak = max(peak, marked)
        max_drawdown = min(max_drawdown, (marked / peak - 1.0) * 100.0)

    if position is not None and not ordered.empty:
        row = ordered.iloc[-1]
        exit_price = float(row["price"]) * (1 - slip if position.direction == "LONG" else 1 + slip)
        gross = (exit_price - position.entry_price) * position.quantity
        if position.direction == "SHORT":
            gross *= -1
        exit_fee = position.notional * fee_rate
        net = gross - position.entry_fee - exit_fee
        capital += net
        trades.append({"net_pnl": net, "notional": position.notional, "exit_reason": "end_of_data"})

    trade_frame = pd.DataFrame(trades)
    wins = trade_frame.loc[trade_frame.get("net_pnl", pd.Series(dtype=float)) > 0] if not trade_frame.empty else trade_frame
    losses = trade_frame.loc[trade_frame.get("net_pnl", pd.Series(dtype=float)) < 0] if not trade_frame.empty else trade_frame
    gross_wins = float(wins["net_pnl"].sum()) if not wins.empty else 0.0
    gross_losses = abs(float(losses["net_pnl"].sum())) if not losses.empty else 0.0
    return {
        "symbol": symbol, "trades": int(len(trade_frame)),
        "return_pct": (capital / starting_capital - 1.0) * 100.0 if starting_capital else 0.0,
        "win_rate_pct": len(wins) / len(trade_frame) * 100.0 if len(trade_frame) else 0.0,
        "max_drawdown_pct": max_drawdown,
        "profit_factor": gross_wins / gross_losses if gross_losses else (float("inf") if gross_wins else 0.0),
    }


def _build_signals_frame(endpoints: pd.DataFrame, probability: np.ndarray, threshold: float, direction: Direction) -> pd.DataFrame:
    frame = endpoints.copy()
    frame["probability"] = probability
    label = "LONG" if direction == "long" else "SHORT"
    frame["model_direction"] = np.where(frame["probability"] >= threshold, label, "HOLD")
    frame["price"] = frame["Close"].astype(float)
    frame["open"] = frame["Open"].astype(float)
    frame["high"] = frame["High"].astype(float)
    frame["low"] = frame["Low"].astype(float)
    frame["close"] = frame["Close"].astype(float)
    frame["atr"] = frame["ATR"].astype(float)
    return frame


def _simulate_threshold(endpoints: pd.DataFrame, probability: np.ndarray, threshold: float, direction: Direction, symbol: str) -> dict[str, Any]:
    signals = _build_signals_frame(endpoints, probability, threshold, direction)
    return _simulate_symbol_swing_policy(signals, symbol, SIMULATION_STARTING_CAPITAL)


def _select_threshold(calibration_endpoints: pd.DataFrame, calibration_probability: np.ndarray, direction: Direction, symbol: str) -> tuple[float, dict[str, Any]]:
    candidates = []
    for threshold in np.arange(0.50, 0.91, 0.02):
        summary = _simulate_threshold(calibration_endpoints, calibration_probability, float(threshold), direction, symbol)
        if summary["trades"] >= MIN_CALIBRATION_TRADES:
            candidates.append((float(threshold), summary))
    viable = [c for c in candidates if c[1]["profit_factor"] >= 1.20 and c[1]["return_pct"] > 0]
    pool = viable or candidates
    if not pool:
        return 0.90, {"trades": 0, "return_pct": 0.0, "profit_factor": 0.0, "win_rate_pct": 0.0}
    selected = max(pool, key=lambda c: (c[1]["profit_factor"], c[1]["return_pct"]))
    return selected[0], selected[1]


def _walk_forward_rows(endpoints: pd.DataFrame, probability: np.ndarray, threshold: float, direction: Direction, symbol: str) -> list[dict[str, Any]]:
    rows = []
    for window, indices in enumerate(np.array_split(np.arange(len(endpoints)), 3), start=1):
        frame = endpoints.iloc[indices].reset_index(drop=True)
        frame_probability = probability[indices]
        summary = _simulate_threshold(frame, frame_probability, threshold, direction, symbol)
        rows.append({"window": window, "trades": summary["trades"], "win_rate_pct": summary["win_rate_pct"],
                     "return_pct": summary["return_pct"], "profit_factor": summary["profit_factor"],
                     "max_drawdown_pct": summary["max_drawdown_pct"]})
    return rows


def _model_path(direction: Direction, symbol: str) -> Path:
    return EXPERIMENT_ROOT / direction / f"gbm_{symbol.upper()}.pkl"


def train_one(symbol: str, direction: Direction, timeframe: str = TIMEFRAME, force: bool = False) -> dict[str, Any]:
    symbol = symbol.upper()
    result_base: dict[str, Any] = {"symbol": symbol, "direction": direction}

    if not force and _model_path(direction, symbol).exists():
        return {**result_base, "trained": True, "reason": "already trained (cached)"}

    try:
        df = _load_symbol_frame(symbol, timeframe, direction)
    except FileNotFoundError:
        df = None

    if df is None or len(df) < MIN_TOTAL_ROWS:
        rows = 0 if df is None else len(df)
        return {**result_base, "trained": False, "promoted": False,
                "reason": f"insufficient data: {rows} rows (< {MIN_TOTAL_ROWS})"}

    split = _chronological_split(df)
    if split is None:
        return {**result_base, "trained": False, "promoted": False,
                "reason": "insufficient rows per split segment", "train_rows": len(df)}

    train, calibration, holdout = split
    model = _new_model()
    model.fit(train[FEATURE_COLUMNS].to_numpy(np.float32), train["target"].to_numpy(int))

    calibration_probability = model.predict_proba(calibration[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    calibration_frame = calibration[_ENDPOINT_COLUMNS].reset_index(drop=True)
    threshold, _ = _select_threshold(calibration_frame, calibration_probability, direction, symbol)

    holdout_probability = model.predict_proba(holdout[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    holdout_frame = holdout[_ENDPOINT_COLUMNS].reset_index(drop=True)
    holdout_target = holdout_frame["target"].to_numpy(int)
    holdout_auc = (
        float(roc_auc_score(holdout_target, holdout_probability))
        if len(np.unique(holdout_target)) > 1 else float("nan")
    )
    holdout_summary = _simulate_threshold(holdout_frame, holdout_probability, threshold, direction, symbol)
    windows = _walk_forward_rows(holdout_frame, holdout_probability, threshold, direction, symbol)

    promoted = bool(
        holdout_summary["trades"] >= MIN_CALIBRATION_TRADES
        and all(row["trades"] >= MIN_WINDOW_TRADES and row["profit_factor"] >= 1.30 and row["return_pct"] > 0
                for row in windows)
    )

    _model_path(direction, symbol).parent.mkdir(parents=True, exist_ok=True)
    with _model_path(direction, symbol).open("wb") as handle:
        pickle.dump(model, handle)

    result = {
        **result_base, "trained": True, "promoted": promoted,
        "threshold": float(threshold), "reason": "promoted" if promoted else "did not clear promotion gate",
        "train_rows": len(df), "holdout_trades": int(holdout_summary["trades"]),
        "holdout_return_pct": float(holdout_summary["return_pct"]),
        "holdout_profit_factor": float(holdout_summary["profit_factor"]),
        "holdout_win_rate_pct": float(holdout_summary["win_rate_pct"]),
        "holdout_auc": holdout_auc, "windows": windows,
    }
    logger.info(
        f"{direction}/{symbol}: promoted={promoted} threshold={threshold:.2f} "
        f"holdout_trades={result['holdout_trades']} holdout_return_pct={result['holdout_return_pct']:.2f} "
        f"holdout_profit_factor={result['holdout_profit_factor']:.2f} holdout_auc={holdout_auc:.3f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None, help="explicit symbol subset (default: all liquid symbols)")
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    directions: list[Direction] = ["long", "short"] if args.direction == "both" else [args.direction]
    all_symbols = [s.upper() for s in (args.symbols or list_gmx_symbols(TIMEFRAME))]

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for direction in directions:
        symbols = liquid_symbols(direction, all_symbols) if not args.symbols else all_symbols
        logger.info(f"{direction}: {len(symbols)} symbols to train ({'explicit' if args.symbols else 'liquidity-filtered'})")
        for i, symbol in enumerate(symbols, start=1):
            started = time.time()
            try:
                result = train_one(symbol, direction, force=args.force)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"{direction}/{symbol}: FAILED: {exc}\n{traceback.format_exc()}")
                result = {"symbol": symbol, "direction": direction, "trained": False,
                          "promoted": False, "reason": f"exception: {exc}"}
            result["elapsed_sec"] = round(time.time() - started, 1)
            results.append(result)
            logger.info(f"[{i}/{len(symbols)}] {direction}/{symbol} done in {result['elapsed_sec']}s")
            (EXPERIMENT_ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    promoted = [r for r in results if r.get("promoted")]
    logger.info(f"DONE: {len(promoted)}/{len(results)} symbol/direction combos promoted")
    for r in promoted:
        logger.info(f"  PROMOTED {r['direction']}/{r['symbol']}: threshold={r['threshold']:.2f} "
                    f"return_pct={r['holdout_return_pct']:.2f} profit_factor={r['holdout_profit_factor']:.2f}")


if __name__ == "__main__":
    main()
