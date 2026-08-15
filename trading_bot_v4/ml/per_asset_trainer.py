"""Per-asset LONG/SHORT model training.

Each GMX symbol gets its own independently-trained model per direction, instead
of one model pooled or shared across every asset (the bug this replaces: the
"original" LONG model was trained only on ADA and applied unchanged to all ~120
GMX assets; the bearish/SHORT model pooled every symbol into one shared fit).

Mirrors bearish_trainer.py's already-validated split/calibration/walk-forward
methodology (70/10/10/10 chronological split, 3-window walk-forward stability
gate) but trains one model per symbol instead of pooling symbols together, and
sources data from GMX per-symbol OHLCV via load_gmx_ohlc/DataHandler (the same
pipeline train_model.py already uses) instead of the SMC feature pipeline.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from trading_bot_v4.ml.cnn_lstm_model import V4CNNLSTMClassifier
from trading_bot_v4.utils.model_cache import (
    per_asset_calibration_path,
    per_asset_model_path,
    per_asset_model_root,
    per_asset_scaler_path,
)
from trading_bot_v4.utils.logger import build_logger
from train_model import DataHandler

logger = build_logger("v4_per_asset_trainer")

Direction = Literal["long", "short"]

MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 256
TRAIN_END, VALIDATION_END, CALIBRATION_END = 0.70, 0.80, 0.90
MIN_TOTAL_ROWS = max(int(Config.MIN_TRAIN_CANDLES), 500)


@dataclass(frozen=True)
class PerAssetTrainingResult:
    symbol: str
    direction: Direction
    trained: bool
    promoted: bool
    threshold: float
    reason: str
    train_rows: int
    holdout_signals: int
    holdout_precision: float
    holdout_auc: float
    windows: list[dict[str, Any]]


def _round_trip_cost() -> float:
    one_way_bps = Config.PAPER_FEE_BPS + Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS
    return 2.0 * one_way_bps / 10000.0


def _target_for_direction(future_return: pd.Series, direction: Direction) -> pd.Series:
    threshold = float(Config.MOVEMENT_THRESHOLD)
    if direction == "long":
        return (future_return > threshold).astype(int)
    return (future_return < -threshold).astype(int)


def _load_symbol_frame(symbol: str, timeframe: str, direction: Direction, horizon: int = 1) -> pd.DataFrame | None:
    df = load_gmx_ohlc(symbol, timeframe)
    if df is None or df.empty:
        return None

    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df = df.copy()
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr
    df = handler.prepare_features(df, prediction_horizon=horizon)
    if df.empty:
        return None

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["target"] = _target_for_direction(df["future_return"], direction)
    return df


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    n = len(df)
    train_end = int(n * TRAIN_END)
    validation_end = int(n * VALIDATION_END)
    calibration_end = int(n * CALIBRATION_END)
    boundaries = (
        train_end,
        validation_end - train_end,
        calibration_end - validation_end,
        n - calibration_end,
    )
    if any(size <= Config.SEQUENCE_LENGTH for size in boundaries):
        return None
    return (
        df.iloc[:train_end],
        df.iloc[train_end:validation_end],
        df.iloc[validation_end:calibration_end],
        df.iloc[calibration_end:],
    )


def _endpoints(segment: pd.DataFrame) -> pd.DataFrame:
    """Rows that align 1:1 with the sequences prepare_sequences() builds --
    the first SEQUENCE_LENGTH rows of a segment have no sequence of their own."""
    return segment.iloc[Config.SEQUENCE_LENGTH:][["timestamp", "future_return", "target"]].reset_index(drop=True)


def _walk_forward_rows(endpoints: pd.DataFrame, probability: np.ndarray, threshold: float, direction: Direction) -> list[dict[str, Any]]:
    evaluated = endpoints.copy()
    evaluated["probability"] = probability
    cost = _round_trip_cost()
    sign = 1.0 if direction == "long" else -1.0
    rows = []
    for window, indices in enumerate(np.array_split(np.arange(len(evaluated)), 3), start=1):
        frame = evaluated.iloc[indices]
        y = frame["target"].to_numpy(int)
        p = frame["probability"].to_numpy(float)
        selected = frame.loc[frame["probability"] >= threshold]
        returns = sign * selected["future_return"].to_numpy(float) - cost
        gains = float(returns[returns > 0].sum())
        losses = float(-returns[returns < 0].sum())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, (p >= threshold).astype(int), average="binary", zero_division=0
        )
        rows.append({
            "window": window,
            "start": str(frame["timestamp"].min()),
            "end": str(frame["timestamp"].max()),
            "samples": len(frame),
            "signals": len(selected),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "return_pct": float(returns.sum() * 100.0),
            "profit_factor": gains / losses if losses else (float("inf") if gains else 0.0),
        })
    return rows


def _select_threshold(calibration_target: np.ndarray, calibration_probability: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = []
    for threshold in np.arange(0.50, 0.91, 0.01):
        predicted = calibration_probability >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            calibration_target, predicted.astype(int), average="binary", zero_division=0
        )
        if predicted.sum() >= 15:
            candidates.append((float(threshold), float(precision), float(recall), float(f1)))
    eligible = [row for row in candidates if row[1] >= 0.55 and row[2] >= 0.05]
    selected = max(eligible, key=lambda row: (row[3], row[1])) if eligible else max(
        candidates or [(0.90, 0.0, 0.0, 0.0)], key=lambda row: (row[1], row[3])
    )
    return selected[0], {"precision": selected[1], "recall": selected[2], "f1": selected[3]}


def _update_direction_reports(direction: Direction, result: PerAssetTrainingResult, horizon: int = 1) -> None:
    """Merge this symbol's result into the direction's combined calibration.json
    (promoted_symbols dict, consumed by production_backtest.py's qualification
    logic) and validation/walk_forward CSVs, without disturbing other symbols'
    already-recorded entries."""
    calibration_path = per_asset_calibration_path(direction, horizon)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)

    calibration: dict[str, Any] = {}
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    promoted_symbols: dict[str, float] = calibration.get("promoted_symbols", {})
    per_symbol: dict[str, Any] = calibration.get("per_symbol", {})

    if result.promoted:
        promoted_symbols[result.symbol] = result.threshold
    else:
        promoted_symbols.pop(result.symbol, None)

    per_symbol[result.symbol] = {
        "trained": result.trained,
        "promoted": result.promoted,
        "reason": result.reason,
        "threshold": result.threshold,
        "train_rows": result.train_rows,
        "holdout_signals": result.holdout_signals,
        "holdout_precision": result.holdout_precision,
        "holdout_auc": result.holdout_auc,
        "windows": result.windows,
    }

    calibration.update({
        "direction": direction,
        "horizon": horizon,
        "target": f"next_{horizon}_candle_return {'>' if direction == 'long' else '<'} "
                  f"{'+' if direction == 'long' else '-'}{Config.MOVEMENT_THRESHOLD}",
        "split_policy": "70% train / 10% validation / 10% threshold calibration / 10% untouched promotion holdout, per symbol",
        "promoted": bool(promoted_symbols),
        "promoted_symbols": promoted_symbols,
        "per_symbol": per_symbol,
    })
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    validation_path = per_asset_model_root(direction, horizon) / "validation.csv"
    walk_forward_path = per_asset_model_root(direction, horizon) / "walk_forward.csv"

    validation_rows = pd.read_csv(validation_path).to_dict("records") if validation_path.exists() else []
    validation_rows = [row for row in validation_rows if row.get("symbol") != result.symbol]
    validation_rows.append({
        "symbol": result.symbol, "threshold": result.threshold, "promoted": result.promoted,
        "reason": result.reason, "train_rows": result.train_rows,
        "holdout_signals": result.holdout_signals, "holdout_precision": result.holdout_precision,
        "holdout_auc": result.holdout_auc,
    })
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)

    walk_forward_rows = pd.read_csv(walk_forward_path).to_dict("records") if walk_forward_path.exists() else []
    walk_forward_rows = [row for row in walk_forward_rows if row.get("symbol") != result.symbol]
    for window_row in result.windows:
        walk_forward_rows.append({"symbol": result.symbol, "threshold": result.threshold, **window_row})
    pd.DataFrame(walk_forward_rows).to_csv(walk_forward_path, index=False)


def already_trained(direction: Direction, symbol: str, horizon: int = 1) -> bool:
    calibration_path = per_asset_calibration_path(direction, horizon)
    if not calibration_path.exists():
        return False
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    return symbol in calibration.get("per_symbol", {}) and per_asset_model_path(direction, symbol, horizon).exists()


def train_one_asset(symbol: str, direction: Direction, timeframe: str = "1h", force: bool = False,
                    horizon: int = 1) -> PerAssetTrainingResult:
    symbol = symbol.upper()
    if not force and already_trained(direction, symbol, horizon):
        logger.info(f"{direction}/{symbol}: already trained, skipping (use force=True to retrain)")
        calibration = json.loads(per_asset_calibration_path(direction, horizon).read_text(encoding="utf-8"))
        cached = calibration["per_symbol"][symbol]
        return PerAssetTrainingResult(symbol=symbol, direction=direction, **{
            k: cached[k] for k in ("trained", "promoted", "reason", "threshold", "train_rows",
                                    "holdout_signals", "holdout_precision", "holdout_auc", "windows")
        })

    try:
        df = _load_symbol_frame(symbol, timeframe, direction, horizon)
    except FileNotFoundError:
        df = None

    if df is None or len(df) < MIN_TOTAL_ROWS:
        rows = 0 if df is None else len(df)
        result = PerAssetTrainingResult(
            symbol=symbol, direction=direction, trained=False, promoted=False, threshold=1.0,
            reason=f"insufficient data: {rows} rows (< {MIN_TOTAL_ROWS})",
            train_rows=rows, holdout_signals=0, holdout_precision=0.0, holdout_auc=float("nan"), windows=[],
        )
        _update_direction_reports(direction, result, horizon)
        return result

    split = _chronological_split(df)
    if split is None:
        result = PerAssetTrainingResult(
            symbol=symbol, direction=direction, trained=False, promoted=False, threshold=1.0,
            reason="insufficient rows per split segment after accounting for sequence length",
            train_rows=len(df), holdout_signals=0, holdout_precision=0.0, holdout_auc=float("nan"), windows=[],
        )
        _update_direction_reports(direction, result, horizon)
        return result

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
        result = PerAssetTrainingResult(
            symbol=symbol, direction=direction, trained=False, promoted=False, threshold=1.0,
            reason="too few sequences in one or more splits",
            train_rows=len(df), holdout_signals=0, holdout_precision=0.0, holdout_auc=float("nan"), windows=[],
        )
        _update_direction_reports(direction, result, horizon)
        return result

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
    if len(calibration_endpoints) != len(calibration_probability):
        raise RuntimeError(
            f"{symbol}/{direction}: calibration alignment mismatch "
            f"{len(calibration_endpoints)} != {len(calibration_probability)}"
        )
    threshold, _ = _select_threshold(calibration_endpoints["target"].to_numpy(int), calibration_probability)

    holdout_probability = classifier.model.predict(X_hold, verbose=0).reshape(-1)
    holdout_endpoints = _endpoints(holdout)
    if len(holdout_endpoints) != len(holdout_probability):
        raise RuntimeError(
            f"{symbol}/{direction}: holdout alignment mismatch "
            f"{len(holdout_endpoints)} != {len(holdout_probability)}"
        )
    holdout_predicted = holdout_probability >= threshold
    holdout_precision, _, _, _ = precision_recall_fscore_support(
        holdout_endpoints["target"].to_numpy(int), holdout_predicted.astype(int),
        average="binary", zero_division=0,
    )
    holdout_target = holdout_endpoints["target"].to_numpy(int)
    holdout_auc = (
        float(roc_auc_score(holdout_target, holdout_probability))
        if len(np.unique(holdout_target)) > 1 else float("nan")
    )
    windows = _walk_forward_rows(holdout_endpoints, holdout_probability, threshold, direction)

    promoted = bool(
        int(holdout_predicted.sum()) >= 15
        and holdout_precision >= 0.55
        and all(row["signals"] >= 3 and row["profit_factor"] >= 1.30 and row["return_pct"] > 0 for row in windows)
    )

    per_asset_model_path(direction, symbol, horizon).parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(per_asset_model_path(direction, symbol, horizon))
    with per_asset_scaler_path(direction, symbol, horizon).open("wb") as handle_file:
        pickle.dump(handler.scaler, handle_file)

    result = PerAssetTrainingResult(
        symbol=symbol, direction=direction, trained=True, promoted=promoted,
        threshold=float(threshold), reason="promoted" if promoted else "did not clear promotion gate",
        train_rows=len(df), holdout_signals=int(holdout_predicted.sum()),
        holdout_precision=float(holdout_precision), holdout_auc=holdout_auc, windows=windows,
    )
    _update_direction_reports(direction, result, horizon)
    logger.info(
        f"{direction}/{symbol}: trained={result.trained} promoted={result.promoted} "
        f"threshold={result.threshold:.2f} holdout_auc={result.holdout_auc:.3f} "
        f"holdout_precision={result.holdout_precision:.3f}"
    )
    return result
