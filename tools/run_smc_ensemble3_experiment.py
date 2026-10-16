"""3-architecture ensemble per asset, mirroring binance_margin_bot's approach:
instead of one CNN+LSTM hybrid deciding alone, train a CNN-only, an LSTM-only,
and the existing hybrid independently on the same data, then combine their
probabilities (soft vote: average) so no single architecture's idiosyncratic
error drives the trade decision -- the same "one model hallucinating gets
outvoted" logic the live binance bot already uses, applied here instead of
run_smc_combined_cnn_lstm_experiment.py's single-hybrid-model approach.

Same feature set (base 18 + SMC 38), same liquid-symbols-only discipline, same
real-execution threshold calibration and 3-window promotion gate as everything
else this session -- only the "how many models decide" part changes.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import list_gmx_symbols
from trading_bot_v4.ml.cnn_lstm_model import V4CNNLSTMClassifier
from trading_bot_v4.ml.ensemble_architectures import CNNOnlyClassifier, LSTMOnlyClassifier
from trading_bot_v4.ml.per_asset_trainer import (
    MIN_CALIBRATION_TRADES,
    MIN_TOTAL_ROWS,
    MIN_WINDOW_TRADES,
    Direction,
    PerAssetTrainingResult,
    _chronological_split,
    _endpoints,
    _select_threshold,
    _simulate_threshold,
    _walk_forward_rows,
)
from trading_bot_v4.risk.market_cap_tiers import liquid_symbols
from trading_bot_v4.utils.logger import build_logger
from train_model import DataHandler
from tools.run_smc_combined_cnn_lstm_experiment import FEATURE_COLUMNS, _load_combined_frame

logger = build_logger("v4_smc_ensemble3_experiment")

MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 256
MODEL_ROOT = Path("models/experiments/smc_ensemble3")

ARCHITECTURES = {
    "cnn": CNNOnlyClassifier,
    "lstm": LSTMOnlyClassifier,
    "hybrid": V4CNNLSTMClassifier,
}


def _model_path(direction: Direction, symbol: str, arch: str, root: Path = MODEL_ROOT) -> Path:
    return root / direction / f"{arch}_{symbol.upper()}_model.h5"


def _scaler_path(direction: Direction, symbol: str, root: Path = MODEL_ROOT) -> Path:
    return root / direction / f"scaler_{symbol.upper()}.pkl"


def _train_one_architecture(arch_name: str, X_train, y_train, X_val, y_val, class_weight, n_features: int):
    classifier_cls = ARCHITECTURES[arch_name]
    classifier = classifier_cls()
    classifier.build_model((Config.SEQUENCE_LENGTH, n_features))
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
    return classifier


def train_one(symbol: str, direction: Direction, force: bool = False,
              feature_columns: list[str] = FEATURE_COLUMNS, model_root: Path = MODEL_ROOT) -> PerAssetTrainingResult:
    symbol = symbol.upper()
    if not force and _model_path(direction, symbol, "hybrid", model_root).exists():
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
    train_scaled = handler.normalize_data(train[feature_columns].to_numpy(np.float32), fit=True)
    validation_scaled = handler.normalize_data(validation[feature_columns].to_numpy(np.float32), fit=False)
    calibration_scaled = handler.normalize_data(calibration[feature_columns].to_numpy(np.float32), fit=False)
    holdout_scaled = handler.normalize_data(holdout[feature_columns].to_numpy(np.float32), fit=False)

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

    calibration_probs = []
    holdout_probs = []
    _model_path(direction, symbol, "hybrid", model_root).parent.mkdir(parents=True, exist_ok=True)
    for arch_name in ARCHITECTURES:
        classifier = _train_one_architecture(arch_name, X_train, y_train, X_val, y_val, class_weight, len(feature_columns))
        calibration_probs.append(classifier.model.predict(X_cal, verbose=0).reshape(-1))
        holdout_probs.append(classifier.model.predict(X_hold, verbose=0).reshape(-1))
        classifier.save_model(_model_path(direction, symbol, arch_name, model_root))
        logger.info(f"{direction}/{symbol}: {arch_name} trained")

    # Soft vote: average the 3 architectures' probabilities into one ensemble
    # signal, instead of a hard 2-of-3 vote -- lets a strongly-confident
    # minority outvote two weakly-confident majority calls, and reuses the
    # exact threshold-calibration/promotion machinery already validated
    # tonight for a single model's probability array.
    calibration_probability = np.mean(calibration_probs, axis=0)
    holdout_probability = np.mean(holdout_probs, axis=0)

    calibration_endpoints = _endpoints(calibration)
    threshold, _ = _select_threshold(calibration_endpoints, calibration_probability, direction, symbol)

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

    with _scaler_path(direction, symbol, model_root).open("wb") as handle_file:
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
    parser.add_argument("--pilot-count", type=int, default=5,
                        help="With no --symbols given, how many liquid symbols to pilot on")
    parser.add_argument("--no-smc", action="store_true",
                        help="Ablation: train on the base 18 features only, dropping all 38 SMC "
                             "features, to isolate whether SMC is what's dragging results down")
    args = parser.parse_args()
    directions = ["long", "short"] if args.direction == "both" else [args.direction]

    if args.no_smc:
        feature_columns = list(Config.FEATURE_COLUMNS)
        model_root = Path("models/experiments/smc_ensemble3_no_smc")
    else:
        feature_columns = FEATURE_COLUMNS
        model_root = MODEL_ROOT

    all_symbols = list_gmx_symbols("1h")
    results = []
    for direction in directions:
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
        else:
            symbols = liquid_symbols(direction.upper(), all_symbols)[: args.pilot_count]
        logger.info(f"{direction}: piloting on {symbols} ({len(feature_columns)} features)")
        for symbol in symbols:
            result = train_one(symbol, direction, force=args.force, feature_columns=feature_columns, model_root=model_root)
            results.append(asdict(result))

    import json
    model_root.mkdir(parents=True, exist_ok=True)
    summary_path = model_root / "pilot_results.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {summary_path}")
