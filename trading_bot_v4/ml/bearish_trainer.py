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
BEARISH_VALIDATION_SPLIT = 0.80
BEARISH_CALIBRATION_SPLIT = 0.90


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


def _split_bearish_four_way(dataset: pd.DataFrame, features: list[str]):
    scaler = StandardScaler()
    groups = {}
    for symbol, group in dataset.groupby("symbol", sort=True):
        group = group.sort_values("timestamp")
        train_end = int(len(group) * BEARISH_TRAIN_SPLIT)
        validation_end = int(len(group) * BEARISH_VALIDATION_SPLIT)
        calibration_end = int(len(group) * BEARISH_CALIBRATION_SPLIT)
        boundaries = (train_end, validation_end - train_end,
                      calibration_end - validation_end, len(group) - calibration_end)
        if any(size <= Config.SEQUENCE_LENGTH for size in boundaries):
            continue
        train = group.iloc[:train_end]
        validation = group.iloc[train_end:validation_end]
        calibration = group.iloc[validation_end:calibration_end]
        holdout = group.iloc[calibration_end:]
        scaler.partial_fit(train[features].to_numpy(np.float32))
        groups[symbol] = (train, validation, calibration, holdout)
    arrays, stats = {}, {"train_rows": 0, "validation_rows": 0, "calibration_rows": 0, "holdout_rows": 0,
                         "train_sequences": 0, "validation_sequences": 0,
                         "calibration_sequences": 0, "holdout_sequences": 0,
                         "train_positive": 0, "train_negative": 0}
    for symbol, (train, validation, calibration, holdout) in groups.items():
        entry = {}
        for name, frame in (("train", train), ("validation", validation),
                            ("calibration", calibration), ("holdout", holdout)):
            entry[f"{name}_features"] = scaler.transform(frame[features].to_numpy(np.float32)).astype(np.float32)
            entry[f"{name}_target"] = frame["target"].to_numpy(np.float32)
            stats[f"{name}_rows"] += len(frame)
            stats[f"{name}_sequences"] += len(frame) - Config.SEQUENCE_LENGTH
        train_y = entry["train_target"][Config.SEQUENCE_LENGTH:]
        stats["train_positive"] += int(train_y.sum())
        stats["train_negative"] += int(len(train_y) - train_y.sum())
        arrays[symbol] = entry
    if not arrays:
        raise ValueError("No assets have enough rows for bearish train/validation/calibration/holdout splits")
    return arrays, scaler, stats


def _partition_endpoints(dataset: pd.DataFrame, start_fraction: float,
                         end_fraction: float = 1.0) -> pd.DataFrame:
    frames = []
    for _, group in dataset.groupby("symbol", sort=True):
        start = int(len(group) * start_fraction)
        end = int(len(group) * end_fraction)
        partition = group.iloc[start:end]
        if len(partition) > Config.SEQUENCE_LENGTH:
            frames.append(partition.iloc[Config.SEQUENCE_LENGTH:][
                ["timestamp", "symbol", "bearish_future_return", "target"]
            ])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def _calibrate_symbols(calibration_endpoints: pd.DataFrame, calibration_probability: np.ndarray,
                       holdout_endpoints: pd.DataFrame, holdout_probability: np.ndarray) -> tuple[
    dict[str, float], list[dict[str, Any]], list[dict[str, Any]]
]:
    selected_data = calibration_endpoints.copy()
    selected_data["probability"] = calibration_probability
    final_data = holdout_endpoints.copy()
    final_data["probability"] = holdout_probability
    promoted: dict[str, float] = {}
    report: list[dict[str, Any]] = []
    window_report: list[dict[str, Any]] = []
    for symbol, selection in selected_data.groupby("symbol", sort=True):
        final = final_data.loc[final_data["symbol"].eq(symbol)].copy()
        selection_y = selection["target"].to_numpy(int)
        selection_p = selection["probability"].to_numpy(float)
        final_y = final["target"].to_numpy(int)
        final_p = final["probability"].to_numpy(float)
        if len(selection) < 100 or len(final) < 100 or len(np.unique(selection_y)) < 2 or len(np.unique(final_y)) < 2:
            continue
        candidates = []
        for threshold in np.arange(0.50, 0.91, 0.01):
            predicted = selection_p >= threshold
            precision, recall, f1, _ = precision_recall_fscore_support(
                selection_y, predicted.astype(int), average="binary", zero_division=0
            )
            if predicted.sum() >= 15:
                candidates.append((float(threshold), float(precision), float(recall), float(f1)))
        eligible = [row for row in candidates if row[1] >= 0.55 and row[2] >= 0.05]
        selected = max(eligible, key=lambda row: (row[3], row[1])) if eligible else max(
            candidates or [(0.90, 0.0, 0.0, 0.0)], key=lambda row: (row[1], row[3])
        )
        final_predicted = final_p >= selected[0]
        final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(
            final_y, final_predicted.astype(int), average="binary", zero_division=0
        )
        windows = _walk_forward_rows(final, final_p, selected[0])
        window_report.extend({"symbol": symbol, "threshold": selected[0], **row} for row in windows)
        stable = bool(eligible) and final_predicted.sum() >= 15 and final_precision >= 0.55 and all(
            row["signals"] >= 3 and row["profit_factor"] >= 1.30 and row["return_pct"] > 0
            for row in windows
        )
        if stable:
            promoted[str(symbol)] = selected[0]
        report.append({
            "symbol": symbol, "threshold": selected[0],
            "calibration_precision": selected[1], "calibration_recall": selected[2],
            "calibration_f1": selected[3], "precision": float(final_precision),
            "recall": float(final_recall), "f1": float(final_f1),
            "auc": float(roc_auc_score(final_y, final_p)),
            "signals": int(final_predicted.sum()), "walk_forward_stable": stable,
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
    arrays, scaler, stats = _split_bearish_four_way(dataset, feature_columns)
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
    calibration_data = _make_sequence_dataset(
        arrays, "calibration_features", "calibration_target", feature_count,
        batch_size=BEARISH_BATCH_SIZE,
    )
    holdout_data = _make_sequence_dataset(
        arrays, "holdout_features", "holdout_target", feature_count,
        batch_size=BEARISH_BATCH_SIZE,
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
    calibration_y = np.concatenate([batch_y.numpy().reshape(-1) for _, batch_y in calibration_data])
    calibration_probability = classifier.model.predict(calibration_data, verbose=0).reshape(-1)
    calibration = _calibrate(calibration_y.astype(int), calibration_probability)
    calibration_endpoints = _partition_endpoints(
        dataset, BEARISH_VALIDATION_SPLIT, BEARISH_CALIBRATION_SPLIT,
    )
    if len(calibration_endpoints) != len(calibration_probability):
        raise RuntimeError(
            f"Bearish calibration alignment mismatch: {len(calibration_endpoints)} != {len(calibration_probability)}"
        )

    holdout_y = np.concatenate([batch_y.numpy().reshape(-1) for _, batch_y in holdout_data])
    holdout_probability = classifier.model.predict(holdout_data, verbose=0).reshape(-1)
    holdout_endpoints = _partition_endpoints(dataset, BEARISH_CALIBRATION_SPLIT)
    if len(holdout_endpoints) != len(holdout_probability):
        raise RuntimeError(
            f"Bearish holdout alignment mismatch: {len(holdout_endpoints)} != {len(holdout_probability)}"
        )
    holdout_precision, holdout_recall, holdout_f1, _ = precision_recall_fscore_support(
        holdout_y.astype(int), (holdout_probability >= calibration["threshold"]).astype(int),
        average="binary", zero_division=0,
    )
    holdout_auc = float(roc_auc_score(holdout_y, holdout_probability)) if len(np.unique(holdout_y)) > 1 else float("nan")
    walk_forward = _walk_forward_rows(
        holdout_endpoints, holdout_probability, float(calibration["threshold"]),
    )
    stable = all(
        row["profit_factor"] >= 1.30 and row["return_pct"] > 0 and row["auc"] >= 0.55
        for row in walk_forward
    )
    promoted_symbols, symbol_report, symbol_windows = _calibrate_symbols(
        calibration_endpoints, calibration_probability, holdout_endpoints, holdout_probability,
    )
    calibration.update({
        "target": f"next_{BEARISH_HORIZON}_candle_return < -{Config.MOVEMENT_THRESHOLD}",
        "prediction_horizon_candles": BEARISH_HORIZON,
        "timeframe": timeframe,
        "split_policy": "70% train / 10% model validation / 10% threshold calibration / 10% untouched promotion holdout",
        "calibration_sequences": int(len(calibration_y)),
        "validation_sequences": int(len(holdout_y)),
        "holdout_auc": holdout_auc,
        "holdout_precision": float(holdout_precision),
        "holdout_recall": float(holdout_recall),
        "holdout_f1": float(holdout_f1),
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
        rows_used=int(stats["train_rows"] + stats["validation_rows"] + stats["calibration_rows"] + stats["holdout_rows"]),
        train_sequences=train_sequences,
        validation_sequences=int(stats["holdout_sequences"]),
        validation_auc=holdout_auc,
        validation_precision=float(holdout_precision),
        validation_recall=float(holdout_recall),
        validation_f1=float(holdout_f1),
        threshold=float(calibration["threshold"]),
        model_path=BEARISH_MODEL_PATH,
        scaler_path=BEARISH_SCALER_PATH,
        calibration_path=BEARISH_CALIBRATION_PATH,
    )
