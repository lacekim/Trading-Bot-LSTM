"""The one real untested combination from the v4 architecture diagram: base 18
features + all 38 (lookahead-fixed) SMC features, together, in a single
per-asset CNN/LSTM model -- not the pooled all-assets SMC model
(smc_trainer.py) and not the base-only per-asset model (per_asset_trainer.py),
which are the two disconnected paths that actually exist today.

Reuses per_asset_trainer.py's already-validated split/calibration/walk-forward
machinery (70/10/10/10 chronological split, threshold selection against real
simulated economics, 3-window promotion gate) and V4CNNLSTMClassifier
unchanged -- only the feature set and model save location differ. Restricted
to liquid_symbols() from the start (the lesson from the base-18 sweep: every
promotion that ignored liquidity turned out to be untradeable). Pilots on a
handful of liquid symbols first, same discipline as every other experiment
this session, before any full sweep.
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.smc_swings import SMC_FEATURE_COLUMNS
from trading_bot_v4.execution.gmx_adapter import list_gmx_symbols
from trading_bot_v4.features.smc_feature_builder import _build_smc_feature_frame
from trading_bot_v4.ml.cnn_lstm_model import V4CNNLSTMClassifier
from trading_bot_v4.ml.per_asset_trainer import (
    MIN_CALIBRATION_TRADES,
    MIN_TOTAL_ROWS,
    MIN_WINDOW_TRADES,
    Direction,
    PerAssetTrainingResult,
    _chronological_split,
    _endpoints,
    _load_symbol_frame as _load_base_frame,
    _select_threshold,
    _simulate_threshold,
    _walk_forward_rows,
)
from trading_bot_v4.risk.market_cap_tiers import liquid_symbols
from trading_bot_v4.utils.logger import build_logger
from train_model import DataHandler

logger = build_logger("v4_smc_combined_cnn_lstm_experiment")

TIMEFRAME = "1h"
FEATURE_COLUMNS = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS]
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 256
MODEL_ROOT = Path("models/experiments/smc_combined_cnn_lstm")


def _load_combined_frame(symbol: str, direction: Direction, horizon: int = 1):
    df = _load_base_frame(symbol, TIMEFRAME, direction, horizon)
    if df is None:
        return None
    df = df.set_index("timestamp")
    smc = _build_smc_feature_frame(symbol, TIMEFRAME)
    df = df.join(smc, how="inner")
    if df.empty:
        return None
    # Several SMC columns are only defined on rows where that pattern just
    # occurred -- NaN elsewhere by design, matching smc_feature_builder.py's
    # own handling (and run_smc_per_asset_experiment.py's).
    df[SMC_FEATURE_COLUMNS] = df[SMC_FEATURE_COLUMNS].ffill().fillna(0.0)
    df = df.reset_index()
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def _model_path(direction: Direction, symbol: str) -> Path:
    return MODEL_ROOT / direction / f"lstm_{symbol.upper()}_model.h5"


def _scaler_path(direction: Direction, symbol: str) -> Path:
    return MODEL_ROOT / direction / f"scaler_{symbol.upper()}.pkl"


def train_one(symbol: str, direction: Direction, force: bool = False) -> PerAssetTrainingResult:
    symbol = symbol.upper()
    if not force and _model_path(direction, symbol).exists():
        logger.info(f"{direction}/{symbol}: already trained, skipping (--force to retrain)")

    df = _load_combined_frame(symbol, direction)
    if df is None or len(df) < MIN_TOTAL_ROWS:
        rows = 0 if df is None else len(df)
        result = PerAssetTrainingResult(
            symbol=symbol, direction=direction, trained=False, promoted=False, threshold=1.0,
            reason=f"insufficient data: {rows} rows (< {MIN_TOTAL_ROWS})",
            train_rows=rows, holdout_trades=0, holdout_return_pct=0.0, holdout_profit_factor=0.0,
            holdout_win_rate_pct=0.0, holdout_auc=float("nan"), windows=[],
        )
        logger.info(result)
        return result

    split = _chronological_split(df)
    if split is None:
        result = PerAssetTrainingResult(
            symbol=symbol, direction=direction, trained=False, promoted=False, threshold=1.0,
            reason="insufficient rows per split segment after accounting for sequence length",
            train_rows=len(df), holdout_trades=0, holdout_return_pct=0.0, holdout_profit_factor=0.0,
            holdout_win_rate_pct=0.0, holdout_auc=float("nan"), windows=[],
        )
        logger.info(result)
        return result

    train, validation, calibration, holdout = split
    handler = DataHandler()
    train_scaled = handler.normalize_data(train[FEATURE_COLUMNS].to_numpy(np.float32), fit=True)
    validation_scaled = handler.normalize_data(validation[FEATURE_COLUMNS].to_numpy(np.float32), fit=False)
    calibration_scaled = handler.normalize_data(calibration[FEATURE_COLUMNS].to_numpy(np.float32), fit=False)
    holdout_scaled = handler.normalize_data(holdout[FEATURE_COLUMNS].to_numpy(np.float32), fit=False)

    X_train, y_train = handler.prepare_sequences(train_scaled, train["target"].to_numpy(np.float32), Config.SEQUENCE_LENGTH)
    X_val, y_val = handler.prepare_sequences(validation_scaled, validation["target"].to_numpy(np.float32), Config.SEQUENCE_LENGTH)
    X_cal, _ = handler.prepare_sequences(calibration_scaled, calibration["target"].to_numpy(np.float32), Config.SEQUENCE_LENGTH)
    X_hold, _ = handler.prepare_sequences(holdout_scaled, holdout["target"].to_numpy(np.float32), Config.SEQUENCE_LENGTH)

    if min(len(X_train), len(X_val), len(X_cal), len(X_hold)) < 10:
        result = PerAssetTrainingResult(
            symbol=symbol, direction=direction, trained=False, promoted=False, threshold=1.0,
            reason="too few sequences in one or more splits",
            train_rows=len(df), holdout_trades=0, holdout_return_pct=0.0, holdout_profit_factor=0.0,
            holdout_win_rate_pct=0.0, holdout_auc=float("nan"), windows=[],
        )
        logger.info(result)
        return result

    n_samples = len(y_train)
    n_positive = float(np.sum(y_train))
    n_negative = n_samples - n_positive
    class_weight = {
        0: n_samples / (2 * n_negative) if n_negative > 0 else 1.0,
        1: n_samples / (2 * n_positive) if n_positive > 0 else 1.0,
    }

    classifier = V4CNNLSTMClassifier()
    classifier.build_model((Config.SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))
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
    if len(calibration_endpoints) != len(calibration_probability):
        raise RuntimeError(
            f"{symbol}/{direction}: calibration alignment mismatch "
            f"{len(calibration_endpoints)} != {len(calibration_probability)}"
        )
    threshold, _ = _select_threshold(calibration_endpoints, calibration_probability, direction, symbol)

    holdout_probability = classifier.model.predict(X_hold, verbose=0).reshape(-1)
    holdout_endpoints = _endpoints(holdout)
    if len(holdout_endpoints) != len(holdout_probability):
        raise RuntimeError(
            f"{symbol}/{direction}: holdout alignment mismatch "
            f"{len(holdout_endpoints)} != {len(holdout_probability)}"
        )
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

    result = PerAssetTrainingResult(
        symbol=symbol, direction=direction, trained=True, promoted=promoted,
        threshold=float(threshold), reason="promoted" if promoted else "did not clear promotion gate",
        train_rows=len(df), holdout_trades=int(holdout_summary["trades"]),
        holdout_return_pct=float(holdout_summary["return_pct"]),
        holdout_profit_factor=float(holdout_summary["profit_factor"]),
        holdout_win_rate_pct=float(holdout_summary["win_rate_pct"]),
        holdout_auc=holdout_auc, windows=windows,
    )
    logger.info(
        f"{direction}/{symbol}: promoted={promoted} threshold={threshold:.2f} "
        f"holdout_trades={result.holdout_trades} holdout_return_pct={result.holdout_return_pct:.2f} "
        f"holdout_profit_factor={result.holdout_profit_factor:.2f} holdout_auc={holdout_auc:.3f}"
    )
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Limit to these symbols; default is a liquid-symbol pilot set")
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pilot-count", type=int, default=8,
                        help="With no --symbols given, how many liquid symbols to pilot on")
    args = parser.parse_args()
    directions = ["long", "short"] if args.direction == "both" else [args.direction]

    all_symbols = list_gmx_symbols(TIMEFRAME)
    results = []
    for direction in directions:
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
        else:
            symbols = liquid_symbols(direction.upper(), all_symbols)[: args.pilot_count]
        logger.info(f"{direction}: piloting on {symbols}")
        for symbol in symbols:
            result = train_one(symbol, direction, force=args.force)
            results.append(asdict(result))

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = MODEL_ROOT / "pilot_results.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {summary_path}")
