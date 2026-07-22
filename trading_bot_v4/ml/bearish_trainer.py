"""Independent downside CNN/LSTM training and calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.smc_swings import SMC_FEATURE_COLUMNS
from trading_bot_v4.ml.cnn_lstm_model import V4CNNLSTMClassifier
from trading_bot_v4.ml.smc_trainer import _make_sequence_dataset, _split_and_scale_by_symbol


BEARISH_MODEL_PATH = Path("models/lstm_smc_bearish_model.h5")
BEARISH_SCALER_PATH = Path("models/scaler_smc_bearish.pkl")
BEARISH_CALIBRATION_PATH = Path("models/smc_bearish_calibration.json")
BEARISH_VALIDATION_PATH = Path("reports/v5_bearish_validation.csv")
BEARISH_WALK_FORWARD_PATH = Path("reports/v5_bearish_walk_forward.csv")
BEARISH_BATCH_SIZE = 256
BEARISH_MAX_EPOCHS = 30
BEARISH_HORIZON = 1
BEARISH_TRAIN_SPLIT = 0.70
BEARISH_SELECTION_SPLIT = 0.85


@dataclass(frozen=True)
class BearishTrainingResult:
    rows_used: int
    train_sequences: int
    validation_sequences: int
    validation_auc: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    threshold: float
    model_path: Path
    scaler_path: Path
    calibration_path: Path


def build_bearish_target(future_return: pd.Series, threshold: float | None = None) -> pd.Series:
    """Positive class means a future decline beyond the configured magnitude."""
    movement = float(threshold if threshold is not None else Config.MOVEMENT_THRESHOLD)
    return (pd.to_numeric(future_return, errors="coerce") < -movement).astype(int)


def _forward_compounded_return(dataset: pd.DataFrame, horizon: int) -> pd.Series:
    grouped = dataset.groupby("symbol", sort=False)["returns"]
    result = pd.Series(1.0, index=dataset.index, dtype=float)
    for step in range(1, horizon + 1):
        result *= 1.0 + grouped.shift(-step)
    return result - 1.0


def _load_bearish_dataset(timeframe: str) -> pd.DataFrame:
    path = Path("models") / f"training_data_smc_all_assets_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"SMC training data not found: {path}")
    dataset = pd.read_csv(path)
    features = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS]
    required = ["timestamp", "symbol", "future_return", "returns", *features]
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"Missing bearish training columns: {missing}")
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], errors="coerce")
    dataset["symbol"] = dataset["symbol"].astype(str).str.upper()
    numeric = [*features, "future_return"]
    dataset[numeric] = dataset[numeric].apply(pd.to_numeric, errors="coerce")
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "symbol", *numeric])
    dataset = dataset.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    dataset["bearish_future_return"] = _forward_compounded_return(dataset, BEARISH_HORIZON)
    dataset = dataset.dropna(subset=["bearish_future_return"])
    dataset["target"] = build_bearish_target(dataset["bearish_future_return"])
    return dataset.reset_index(drop=True)


def _split_bearish_three_way(dataset: pd.DataFrame, features: list[str]):
    scaler = StandardScaler()
    groups = {}
    for symbol, group in dataset.groupby("symbol", sort=True):
        group = group.sort_values("timestamp")
        train_end, selection_end = int(len(group) * BEARISH_TRAIN_SPLIT), int(len(group) * BEARISH_SELECTION_SPLIT)
        if train_end <= Config.SEQUENCE_LENGTH or selection_end - train_end <= Config.SEQUENCE_LENGTH or len(group) - selection_end <= Config.SEQUENCE_LENGTH:
            continue
        train, selection, test = group.iloc[:train_end], group.iloc[train_end:selection_end], group.iloc[selection_end:]
        scaler.partial_fit(train[features].to_numpy(np.float32))
        groups[symbol] = (train, selection, test)
    arrays, stats = {}, {"train_rows": 0, "validation_rows": 0, "test_rows": 0,
                         "train_sequences": 0, "validation_sequences": 0, "test_sequences": 0,
                         "train_positive": 0, "train_negative": 0}
    for symbol, (train, selection, test) in groups.items():
        entry = {}
        for name, frame in (("train", train), ("validation", selection), ("test", test)):
            entry[f"{name}_features"] = scaler.transform(frame[features].to_numpy(np.float32)).astype(np.float32)
            entry[f"{name}_target"] = frame["target"].to_numpy(np.float32)
            stats[f"{name}_rows"] += len(frame)
            stats[f"{name}_sequences"] += len(frame) - Config.SEQUENCE_LENGTH
        train_y = entry["train_target"][Config.SEQUENCE_LENGTH:]
        stats["train_positive"] += int(train_y.sum())
        stats["train_negative"] += int(len(train_y) - train_y.sum())
        arrays[symbol] = entry
    if not arrays:
        raise ValueError("No assets have enough rows for bearish train/selection/test splits")
    return arrays, scaler, stats


def _validation_endpoints(dataset: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in dataset.groupby("symbol", sort=True):
        split = int(len(group) * BEARISH_SELECTION_SPLIT)
        validation = group.iloc[split:]
        if len(validation) > Config.SEQUENCE_LENGTH:
            frames.append(validation.iloc[Config.SEQUENCE_LENGTH:][
                ["timestamp", "symbol", "bearish_future_return", "target"]
            ])
    return pd.concat(frames, ignore_index=True)


def _walk_forward_rows(endpoints: pd.DataFrame, probability: np.ndarray,
                       threshold: float) -> list[dict[str, float | int | str | bool]]:
    evaluated = endpoints.copy()
    evaluated["probability"] = probability
    evaluated = evaluated.sort_values("timestamp").reset_index(drop=True)
    round_trip_cost = 2.0 * (
        Config.PAPER_FEE_BPS + Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS
    ) / 10000.0
    rows = []
    for window, indices in enumerate(np.array_split(np.arange(len(evaluated)), 3), start=1):
        frame = evaluated.iloc[indices]
        y = frame["target"].to_numpy(int)
        p = frame["probability"].to_numpy(float)
        selected = frame.loc[frame["probability"] >= threshold]
        returns = -selected["bearish_future_return"].to_numpy(float) - round_trip_cost
        gains = float(returns[returns > 0].sum())
        losses = float(-returns[returns < 0].sum())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, (p >= threshold).astype(int), average="binary", zero_division=0
        )
        rows.append({
            "window": window, "start": str(frame["timestamp"].min()), "end": str(frame["timestamp"].max()),
            "samples": len(frame), "signals": len(selected), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "return_pct": float(returns.sum() * 100.0),
            "profit_factor": gains / losses if losses else (float("inf") if gains else 0.0),
        })
    return rows


def _calibrate_symbols(endpoints: pd.DataFrame, probability: np.ndarray) -> tuple[
    dict[str, float], list[dict[str, Any]], list[dict[str, Any]]
]:
    evaluated = endpoints.copy()
    evaluated["probability"] = probability
    promoted: dict[str, float] = {}
    report: list[dict[str, Any]] = []
    window_report: list[dict[str, Any]] = []
    for symbol, frame in evaluated.groupby("symbol", sort=True):
        y = frame["target"].to_numpy(int)
        p = frame["probability"].to_numpy(float)
        if len(frame) < 200 or len(np.unique(y)) < 2:
            continue
        candidates = []
        for threshold in np.arange(0.50, 0.91, 0.01):
            predicted = p >= threshold
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, predicted.astype(int), average="binary", zero_division=0
            )
            if predicted.sum() >= 15:
                candidates.append((float(threshold), float(precision), float(recall), float(f1)))
        eligible = [row for row in candidates if row[1] >= 0.55 and row[2] >= 0.05]
        selected = max(eligible, key=lambda row: (row[3], row[1])) if eligible else max(
            candidates or [(0.90, 0.0, 0.0, 0.0)], key=lambda row: (row[1], row[3])
        )
        windows = _walk_forward_rows(frame, p, selected[0])
        window_report.extend({"symbol": symbol, "threshold": selected[0], **row} for row in windows)
        stable = bool(eligible) and all(
            row["signals"] >= 3 and row["profit_factor"] >= 1.30 and row["return_pct"] > 0
            for row in windows
        )
        if stable:
            promoted[str(symbol)] = selected[0]
        report.append({
            "symbol": symbol, "threshold": selected[0], "precision": selected[1],
            "recall": selected[2], "f1": selected[3], "auc": float(roc_auc_score(y, p)),
            "signals": int((p >= selected[0]).sum()), "walk_forward_stable": stable,
            "promoted": stable,
        })
    return promoted, report, window_report


def _calibrate(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    candidates = []
    for threshold in np.arange(0.50, 0.86, 0.01):
        predicted = (probability >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predicted, average="binary", zero_division=0
        )
        candidates.append((float(threshold), float(precision), float(recall), float(f1)))
    eligible = [row for row in candidates if row[1] >= 0.55 and row[2] >= 0.10]
    selected = max(eligible or candidates, key=lambda row: (row[3], row[1], row[2]))
    auc = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "threshold": selected[0], "precision": selected[1], "recall": selected[2],
        "f1": selected[3], "auc": auc,
    }


def train_bearish_model(timeframe: str) -> BearishTrainingResult:
    dataset = _load_bearish_dataset(timeframe)
    feature_columns = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS]
    arrays, scaler, stats = _split_bearish_three_way(dataset, feature_columns)
    feature_count = len(feature_columns)
    train_sequences = int(stats["train_sequences"])
    positives, negatives = int(stats["train_positive"]), int(stats["train_negative"])
    weights = {
        0: train_sequences / (2 * negatives) if negatives else 1.0,
        1: train_sequences / (2 * positives) if positives else 1.0,
    }
    train_data = _make_sequence_dataset(
        arrays, "train_features", "train_target", feature_count,
        sample_weights=weights, batch_size=BEARISH_BATCH_SIZE,
    )
    validation_data = _make_sequence_dataset(
        arrays, "validation_features", "validation_target", feature_count,
        batch_size=BEARISH_BATCH_SIZE,
    )
    test_data = _make_sequence_dataset(
        arrays, "test_features", "test_target", feature_count, batch_size=BEARISH_BATCH_SIZE,
    )
    classifier = V4CNNLSTMClassifier()
    classifier.build_model((Config.SEQUENCE_LENGTH, feature_count))
    classifier.model.fit(
        train_data,
        validation_data=validation_data,
        epochs=min(Config.EPOCHS, BEARISH_MAX_EPOCHS),
        callbacks=[
            EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True, min_delta=0.001),
            ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=3, min_lr=1e-7),
        ],
        verbose=2,
    )
    y_true = np.concatenate([batch_y.numpy().reshape(-1) for _, batch_y in test_data])
    probability = classifier.model.predict(test_data, verbose=0).reshape(-1)
    calibration = _calibrate(y_true.astype(int), probability)
    endpoints = _validation_endpoints(dataset)
    if len(endpoints) != len(probability):
        raise RuntimeError(f"Bearish validation alignment mismatch: {len(endpoints)} != {len(probability)}")
    walk_forward = _walk_forward_rows(endpoints, probability, float(calibration["threshold"]))
    stable = all(
        row["profit_factor"] >= 1.30 and row["return_pct"] > 0 and row["auc"] >= 0.55
        for row in walk_forward
    )
    promoted_symbols, symbol_report, symbol_windows = _calibrate_symbols(endpoints, probability)
    calibration.update({
        "target": f"next_{BEARISH_HORIZON}_candle_return < -{Config.MOVEMENT_THRESHOLD}",
        "prediction_horizon_candles": BEARISH_HORIZON,
        "timeframe": timeframe,
        "validation_sequences": int(len(y_true)),
        "walk_forward_stable": stable,
        "promoted_symbols": promoted_symbols,
        "promoted": bool(promoted_symbols),
    })
    BEARISH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(BEARISH_MODEL_PATH)
    with BEARISH_SCALER_PATH.open("wb") as handle:
        pickle.dump(scaler, handle)
    BEARISH_CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    BEARISH_VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(symbol_report).to_csv(BEARISH_VALIDATION_PATH, index=False)
    pd.DataFrame(symbol_windows).to_csv(BEARISH_WALK_FORWARD_PATH, index=False)
    return BearishTrainingResult(
        rows_used=int(stats["train_rows"] + stats["validation_rows"] + stats["test_rows"]),
        train_sequences=train_sequences,
        validation_sequences=int(stats["test_sequences"]),
        validation_auc=float(calibration["auc"]),
        validation_precision=float(calibration["precision"]),
        validation_recall=float(calibration["recall"]),
        validation_f1=float(calibration["f1"]),
        threshold=float(calibration["threshold"]),
        model_path=BEARISH_MODEL_PATH,
        scaler_path=BEARISH_SCALER_PATH,
        calibration_path=BEARISH_CALIBRATION_PATH,
    )
