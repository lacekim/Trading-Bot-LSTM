"""Optional SMC-enhanced model training for V4 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.smc_swings import SMC_FEATURE_COLUMNS
from trading_bot_v4.ml.cnn_lstm_model import V4CNNLSTMClassifier
from trading_bot_v4.utils.logger import build_logger


logger = build_logger("v4_smc_trainer")

SMC_MODEL_PATH = Path("models/lstm_smc_model.h5")
SMC_SCALER_PATH = Path("models/scaler_smc.pkl")


@dataclass(frozen=True)
class SmcModelTrainingResult:
    rows_used: int
    feature_count: int
    train_rows: int
    validation_rows: int
    train_sequences: int
    validation_sequences: int
    validation_accuracy: float
    validation_loss: float
    model_path: Path
    scaler_path: Path


def _load_smc_training_data(timeframe: str) -> pd.DataFrame:
    path = Path("models") / f"training_data_smc_all_assets_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"SMC training data not found: {path}. "
            f"Build it first with --build-smc-training-data --all-assets --timeframe {timeframe}."
        )

    dataset = pd.read_csv(path)
    required = ["timestamp", "symbol", *Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS, "target"]
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"Missing SMC training data columns: {missing}")

    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], errors="coerce")
    dataset["symbol"] = dataset["symbol"].astype(str).str.upper()
    numeric_columns = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS, "target"]
    dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors="coerce")
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=["timestamp", "symbol", *numeric_columns])
    dataset["target"] = dataset["target"].astype(int)
    return dataset.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _split_and_scale_by_symbol(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], StandardScaler, dict[str, int]]:
    scaler = StandardScaler()
    grouped_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    train_rows = 0
    validation_rows = 0

    for symbol, group in dataset.groupby("symbol", sort=True):
        group = group.sort_values("timestamp")
        if len(group) <= Config.SEQUENCE_LENGTH + 1:
            logger.warning("Skipping %s: insufficient rows after cleanup (%s)", symbol, len(group))
            continue

        split_idx = int(len(group) * Config.TRAIN_SPLIT)
        if split_idx <= Config.SEQUENCE_LENGTH or (len(group) - split_idx) <= Config.SEQUENCE_LENGTH:
            logger.warning("Skipping %s: insufficient rows for train/validation sequences", symbol)
            continue

        train_group = group.iloc[:split_idx]
        validation_group = group.iloc[split_idx:]
        scaler.partial_fit(train_group[feature_columns].to_numpy(dtype=np.float32))
        grouped_frames[symbol] = (train_group, validation_group)
        train_rows += len(train_group)
        validation_rows += len(validation_group)

    if not grouped_frames:
        raise ValueError("No assets have enough rows to train the SMC model")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    train_sequences = 0
    validation_sequences = 0
    train_positive = 0
    train_negative = 0

    for symbol, (train_group, validation_group) in grouped_frames.items():
        train_features = scaler.transform(train_group[feature_columns].to_numpy(dtype=np.float32)).astype(np.float32)
        validation_features = scaler.transform(validation_group[feature_columns].to_numpy(dtype=np.float32)).astype(np.float32)
        train_target = train_group["target"].to_numpy(dtype=np.float32)
        validation_target = validation_group["target"].to_numpy(dtype=np.float32)

        train_y = train_target[Config.SEQUENCE_LENGTH:]
        train_positive += int(train_y.sum())
        train_negative += int(len(train_y) - train_y.sum())
        train_sequences += max(0, len(train_features) - Config.SEQUENCE_LENGTH)
        validation_sequences += max(0, len(validation_features) - Config.SEQUENCE_LENGTH)
        arrays[symbol] = {
            "train_features": train_features,
            "train_target": train_target,
            "validation_features": validation_features,
            "validation_target": validation_target,
        }

    stats = {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "train_positive": train_positive,
        "train_negative": train_negative,
    }
    return arrays, scaler, stats


def _sequence_generator(
    arrays: dict[str, dict[str, np.ndarray]],
    feature_key: str,
    target_key: str,
    sample_weights: dict[int, float] | None = None,
):
    seq_len = Config.SEQUENCE_LENGTH
    for symbol in sorted(arrays):
        features = arrays[symbol][feature_key]
        target = arrays[symbol][target_key]
        for start in range(0, len(features) - seq_len):
            end = start + seq_len
            y_value = target[end]
            if sample_weights is None:
                yield features[start:end], y_value
            else:
                yield features[start:end], y_value, np.float32(sample_weights[int(y_value)])


def _make_sequence_dataset(
    arrays: dict[str, dict[str, np.ndarray]],
    feature_key: str,
    target_key: str,
    feature_count: int,
    sample_weights: dict[int, float] | None = None,
    batch_size: int | None = None,
) -> tf.data.Dataset:
    if sample_weights is None:
        output_signature = (
            tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, feature_count), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(Config.SEQUENCE_LENGTH, feature_count), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    dataset = tf.data.Dataset.from_generator(
        lambda: _sequence_generator(arrays, feature_key, target_key, sample_weights=sample_weights),
        output_signature=output_signature,
    )
    return dataset.batch(batch_size or Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def train_smc_model(timeframe: str) -> SmcModelTrainingResult:
    """Train and save a separate SMC-enhanced CNN/LSTM model."""
    dataset = _load_smc_training_data(timeframe)
    feature_columns = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS]
    arrays, scaler, stats = _split_and_scale_by_symbol(dataset, feature_columns)

    feature_count = len(feature_columns)
    model = V4CNNLSTMClassifier()
    model.build_model(input_shape=(Config.SEQUENCE_LENGTH, feature_count))

    train_positive = stats["train_positive"]
    train_negative = stats["train_negative"]
    train_sequences = stats["train_sequences"]
    sample_weights = {
        0: train_sequences / (2 * train_negative) if train_negative > 0 else 1.0,
        1: train_sequences / (2 * train_positive) if train_positive > 0 else 1.0,
    }
    train_dataset = _make_sequence_dataset(
        arrays,
        "train_features",
        "train_target",
        feature_count,
        sample_weights=sample_weights,
    )
    validation_dataset = _make_sequence_dataset(arrays, "validation_features", "validation_target", feature_count)

    early_stop = EarlyStopping(
        monitor="val_auc",
        patience=15,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    logger.info("Starting optional SMC model training")
    logger.info("Rows used: %s", stats["train_rows"] + stats["validation_rows"])
    logger.info("Feature count: %s", feature_count)
    logger.info("Train/validation rows: %s/%s", stats["train_rows"], stats["validation_rows"])
    logger.info("Train/validation sequences: %s/%s", train_sequences, stats["validation_sequences"])

    model.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=Config.EPOCHS,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    evaluation = model.model.evaluate(validation_dataset, verbose=0)
    metrics = dict(zip(model.model.metrics_names, evaluation))
    validation_loss = float(metrics.get("loss", evaluation[0]))
    validation_accuracy = float(metrics.get("accuracy", np.nan))

    SMC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(SMC_MODEL_PATH)
    with open(SMC_SCALER_PATH, "wb") as file_handle:
        pickle.dump(scaler, file_handle)

    return SmcModelTrainingResult(
        rows_used=int(stats["train_rows"] + stats["validation_rows"]),
        feature_count=feature_count,
        train_rows=int(stats["train_rows"]),
        validation_rows=int(stats["validation_rows"]),
        train_sequences=int(train_sequences),
        validation_sequences=int(stats["validation_sequences"]),
        validation_accuracy=validation_accuracy,
        validation_loss=validation_loss,
        model_path=SMC_MODEL_PATH,
        scaler_path=SMC_SCALER_PATH,
    )
