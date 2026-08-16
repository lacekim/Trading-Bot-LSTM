"""Experiment: replace per_asset_trainer's fixed-horizon threshold target
(future_return > MOVEMENT_THRESHOLD) with a triple-barrier label.

Motivation: the horizon=1 liquid-symbol sweep found 0/95 promotable
symbol/direction combos under the binary threshold target. That target asks
"did price move >1% by the next candle", which ignores path -- a candle can
tag +1.5% then reverse and stop out before horizon, or crawl to +0.9% and
lose to costs, and the label never knows. Triple-barrier labeling (Lopez de
Prado) asks the question that actually matters: simulate a hypothetical
entry at *every* candle using the exact same exit rules real trading uses
(ATR stop/target, PAPER_MAX_HOLD_CANDLES, STOP_FIRST tie-break, round-trip
fees/slippage) and label it by whether that hypothetical trade would have
been net-profitable. This mirrors simulate_production_symbol's
_exit_reference logic directly, just run per-candle instead of per-position.

Trains/evaluates only the liquidity-cleared symbols (risk/market_cap_tiers.
liquid_symbols) at horizon=1, so results are directly comparable to the
existing 0/95 baseline. All split/calibration/walk-forward/simulation logic
is reused unchanged from ml/per_asset_trainer.py -- only the labeling
function differs. Writes to models/experiments/triple_barrier/, never
touches production models/{long,short}/ artifacts or their calibration.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from trading_bot import list_gmx_symbols
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from trading_bot_v4.ml.cnn_lstm_model import V4CNNLSTMClassifier
from trading_bot_v4.ml.per_asset_trainer import (
    BATCH_SIZE,
    MAX_EPOCHS,
    MIN_CALIBRATION_TRADES,
    MIN_TOTAL_ROWS,
    MIN_WINDOW_TRADES,
    PATIENCE,
    SIMULATION_STARTING_CAPITAL,
    _chronological_split,
    _endpoints,
    _select_threshold,
    _simulate_threshold,
    _walk_forward_rows,
)
from trading_bot_v4.risk.market_cap_tiers import liquid_symbols
from trading_bot_v4.utils.logger import build_logger
from train_model import DataHandler

logger = build_logger("v4_triple_barrier_experiment")

Direction = Literal["long", "short"]

TIMEFRAME = "1h"
EXPERIMENT_ROOT = Path("models/experiments/triple_barrier")


def _triple_barrier_target(df: pd.DataFrame, direction: Direction) -> pd.Series:
    """Labels each row by simulating a hypothetical entry there under the real
    exit rules -- ATR stop/target, PAPER_MAX_HOLD_CANDLES, STOP_FIRST
    tie-break, round-trip fees/slippage -- instead of a fixed-horizon return
    threshold. Rows too close to the end of the series to resolve within
    PAPER_MAX_HOLD_CANDLES are left as NaN for the caller to drop."""
    close = df["Close"].to_numpy(np.float64)
    high = df["High"].to_numpy(np.float64)
    low = df["Low"].to_numpy(np.float64)
    atr = df["ATR"].to_numpy(np.float64)
    n = len(df)
    max_hold = int(Config.PAPER_MAX_HOLD_CANDLES)
    stop_first = Config.PAPER_STOP_TARGET_PRIORITY == "STOP_FIRST"

    stop_distance = atr * float(Config.ATR_SL_MULTIPLIER)
    target_distance = atr * float(Config.ATR_TP_MULTIPLIER)
    if direction == "long":
        stop_price = close - stop_distance
        target_price = close + target_distance
    else:
        stop_price = close + stop_distance
        target_price = close - target_distance

    exit_price = np.full(n, np.nan)
    resolved = np.zeros(n, dtype=bool)
    idx = np.arange(n)

    for offset in range(1, max_hold + 1):
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

    time_idx = idx + max_hold
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
    df["target"] = _triple_barrier_target(df, direction)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df


def _model_path(direction: Direction, symbol: str) -> Path:
    return EXPERIMENT_ROOT / direction / f"lstm_{symbol.upper()}_model.h5"


def _scaler_path(direction: Direction, symbol: str) -> Path:
    return EXPERIMENT_ROOT / direction / f"scaler_{symbol.upper()}.pkl"


def train_one(symbol: str, direction: Direction, timeframe: str = TIMEFRAME, force: bool = False) -> dict[str, Any]:
    symbol = symbol.upper()
    result_base: dict[str, Any] = {"symbol": symbol, "direction": direction}

    if not force and _model_path(direction, symbol).exists():
        logger.info(f"{direction}/{symbol}: experiment model already trained, skipping")
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
                "reason": "insufficient rows per split segment after accounting for sequence length"}

    train, validation, calibration, holdout = split
    handler = DataHandler()
    train_scaled = handler.normalize_data(train[Config.FEATURE_COLUMNS].to_numpy(np.float32), fit=True)
    validation_scaled = handler.normalize_data(validation[Config.FEATURE_COLUMNS].to_numpy(np.float32), fit=False)
    calibration_scaled = handler.normalize_data(calibration[Config.FEATURE_COLUMNS].to_numpy(np.float32), fit=False)
    holdout_scaled = handler.normalize_data(holdout[Config.FEATURE_COLUMNS].to_numpy(np.float32), fit=False)

    X_train, y_train = handler.prepare_sequences(train_scaled, train["target"].to_numpy(np.float32), Config.SEQUENCE_LENGTH)
    X_val, y_val = handler.prepare_sequences(validation_scaled, validation["target"].to_numpy(np.float32), Config.SEQUENCE_LENGTH)
    X_cal, _ = handler.prepare_sequences(calibration_scaled, calibration["target"].to_numpy(np.float32), Config.SEQUENCE_LENGTH)
    X_hold, _ = handler.prepare_sequences(holdout_scaled, holdout["target"].to_numpy(np.float32), Config.SEQUENCE_LENGTH)

    if min(len(X_train), len(X_val), len(X_cal), len(X_hold)) < 10:
        return {**result_base, "trained": False, "promoted": False,
                "reason": "too few sequences in one or more splits", "train_rows": len(df)}

    n_samples = len(y_train)
    n_positive = float(np.sum(y_train))
    n_negative = n_samples - n_positive
    class_weight = {
        0: n_samples / (2 * n_negative) if n_negative > 0 else 1.0,
        1: n_samples / (2 * n_positive) if n_positive > 0 else 1.0,
    }

    classifier = V4CNNLSTMClassifier()
    classifier.build_model((Config.SEQUENCE_LENGTH, len(Config.FEATURE_COLUMNS)))
    classifier.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(monitor="val_auc", patience=PATIENCE, restore_best_weights=True, min_delta=0.001),
            ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=3, min_lr=1e-7),
        ],
        verbose=0,
    )

    calibration_probability = classifier.model.predict(X_cal, verbose=0).reshape(-1)
    calibration_endpoints = _endpoints(calibration)
    threshold, _ = _select_threshold(calibration_endpoints, calibration_probability, direction, symbol)

    holdout_probability = classifier.model.predict(X_hold, verbose=0).reshape(-1)
    holdout_endpoints = _endpoints(holdout)
    holdout_target = holdout_endpoints["target"].to_numpy(int)
    holdout_auc = (
        float(roc_auc_score(holdout_target, holdout_probability))
        if len(np.unique(holdout_target)) > 1 else float("nan")
    )
    holdout_summary = _simulate_threshold(holdout_endpoints, holdout_probability, threshold, direction, symbol)
    windows = _walk_forward_rows(holdout_endpoints, holdout_probability, threshold, direction, symbol)

    promoted = bool(
        holdout_summary["trades"] >= MIN_CALIBRATION_TRADES
        and all(
            row["trades"] >= MIN_WINDOW_TRADES and row["profit_factor"] >= 1.30 and row["return_pct"] > 0
            for row in windows
        )
    )

    _model_path(direction, symbol).parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(_model_path(direction, symbol))
    with _scaler_path(direction, symbol).open("wb") as handle_file:
        pickle.dump(handler.scaler, handle_file)

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
        f"holdout_profit_factor={result['holdout_profit_factor']:.2f}"
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
            except Exception as exc:  # noqa: BLE001 -- keep the sweep alive past one symbol's failure
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
