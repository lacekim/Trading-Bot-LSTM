"""Cost-aware three-class CNN/LSTM challenger for explicit LONG/SHORT/HOLD."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.ml.cost_aware_long_trainer import FEATURES, _load, round_trip_cost_rate
from trading_bot_v4.ml.multi_horizon_directional import add_horizon_return, add_protective_outcomes


MODEL_PATH = Path("models/directional_cnn_lstm.keras")
SCALER_PATH = Path("models/directional_cnn_lstm_scaler.pkl")
CALIBRATION_PATH = Path("models/directional_cnn_lstm_calibration.json")
REPORT_PATH = Path("reports/v5_directional_cnn_lstm_validation.json")
SEQUENCE_LENGTH = 36
HORIZON = 8
HOLD, LONG, SHORT = 0, 1, 2


def protected_direction_labels(frame: pd.DataFrame, minimum_net_edge: float = 0.0) -> pd.Series:
    cost = round_trip_cost_rate()
    long_net = pd.to_numeric(frame["long_protected_return"], errors="coerce") - cost
    short_net = pd.to_numeric(frame["short_protected_return"], errors="coerce") - cost
    best = pd.concat([long_net, short_net], axis=1).max(axis=1)
    return pd.Series(
        np.where(best <= minimum_net_edge, HOLD, np.where(long_net >= short_net, LONG, SHORT)),
        index=frame.index, dtype="int8",
    )


def build_sequences(frame: pd.DataFrame, scaler: RobustScaler, fit_scaler: bool = False):
    values = frame[FEATURES].to_numpy(np.float32)
    values = scaler.fit_transform(values) if fit_scaler else scaler.transform(values)
    sequences, labels, rows = [], [], []
    offset = 0
    for _symbol, group in frame.groupby("symbol", sort=False):
        length = len(group)
        positions = np.arange(offset + SEQUENCE_LENGTH - 1, offset + length)
        for position in positions:
            sequences.append(values[position - SEQUENCE_LENGTH + 1:position + 1])
            labels.append(int(frame.iloc[position]["direction_target"]))
            rows.append(position)
        offset += length
    return np.asarray(sequences, np.float32), np.asarray(labels, np.int8), frame.iloc[rows].reset_index(drop=True)


def _model() -> tf.keras.Model:
    inputs = tf.keras.Input((SEQUENCE_LENGTH, len(FEATURES)))
    x = tf.keras.layers.Conv1D(48, 3, padding="causal", activation="swish")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25)(x)
    x = tf.keras.layers.LSTM(32, dropout=0.25)(x)
    x = tf.keras.layers.Dense(32, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def _metrics(rows: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    long_p, short_p = probability[:, LONG], probability[:, SHORT]
    direction = np.where(
        (long_p >= threshold) & (long_p >= short_p), LONG,
        np.where(short_p >= threshold, SHORT, HOLD),
    )
    selected = direction != HOLD
    chosen = rows.loc[selected].copy()
    chosen_direction = direction[selected]
    trade_return = np.where(
        chosen_direction == LONG, chosen["long_protected_return"], chosen["short_protected_return"]
    ) - round_trip_cost_rate()
    chosen["trade_return"] = trade_return
    chosen["strength"] = np.maximum(long_p[selected], short_p[selected])
    capital, pnl = 1.0, []
    for _timestamp, candidates in chosen.groupby("timestamp", sort=True):
        period = 0.0
        for _, trade in candidates.nlargest(Config.PAPER_MAX_OPEN_POSITIONS, "strength").iterrows():
            allocation = min(0.95 / Config.PAPER_MAX_OPEN_POSITIONS, 0.25)
            value = capital * allocation * float(trade["trade_return"])
            period += value; pnl.append(value)
        capital += period
    pnl = pd.Series(pnl, dtype=float); wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "trades": int(len(pnl)), "return_pct": float((capital - 1.0) * 100.0),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else 0.0,
        "win_rate_pct": float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0,
        "long_trades": int((chosen_direction == LONG).sum()),
        "short_trades": int((chosen_direction == SHORT).sum()),
    }


def train_directional_sequence_model(timeframe: str = "1h") -> dict:
    data = add_protective_outcomes(add_horizon_return(_load(timeframe), HORIZON), HORIZON, timeframe)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.dropna(subset=["timestamp"]).copy()
    data["direction_target"] = protected_direction_labels(data, minimum_net_edge=0.001)
    times = pd.Series(pd.to_datetime(data["timestamp"], utc=True).unique()).sort_values().reset_index(drop=True)
    train_end, calibration_end = times.iloc[int(len(times) * 0.70)], times.iloc[int(len(times) * 0.85)]
    train = data[data["timestamp"] < train_end].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    calibration = data[(data["timestamp"] >= train_end) & (data["timestamp"] < calibration_end)].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    holdout = data[data["timestamp"] >= calibration_end].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    scaler = RobustScaler(quantile_range=(5, 95))
    x_train, y_train, _ = build_sequences(train, scaler, fit_scaler=True)
    x_cal, y_cal, cal_rows = build_sequences(calibration, scaler)
    x_test, y_test, test_rows = build_sequences(holdout, scaler)
    counts = np.bincount(y_train, minlength=3)
    class_weight = {index: float(len(y_train) / max(1, 3 * count)) for index, count in enumerate(counts)}
    model = _model()
    model.fit(
        x_train, y_train, validation_data=(x_cal, y_cal), epochs=50, batch_size=256,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)], verbose=2,
    )
    cp = model.predict(x_cal, batch_size=512, verbose=0)
    candidates = [(float(value), _metrics(cal_rows, cp, float(value))) for value in np.arange(0.40, 0.91, 0.02)]
    viable = [item for item in candidates if item[1]["trades"] >= 100 and item[1]["profit_factor"] >= 1.20 and item[1]["return_pct"] > 0]
    threshold, calibration_metrics = max(viable or candidates, key=lambda item: (item[1]["profit_factor"], item[1]["return_pct"]))
    tp = model.predict(x_test, batch_size=512, verbose=0)
    holdout_metrics = _metrics(test_rows, tp, threshold)
    promoted = bool(
        calibration_metrics["trades"] >= 100 and holdout_metrics["trades"] >= 100
        and calibration_metrics["profit_factor"] >= 1.20 and holdout_metrics["profit_factor"] >= 1.20
        and calibration_metrics["return_pct"] > 0 and holdout_metrics["return_pct"] > 0
    )
    report = {
        "promoted": promoted, "timeframe": timeframe, "horizon": HORIZON,
        "sequence_length": SEQUENCE_LENGTH, "threshold": threshold,
        "target": "best LONG/SHORT protected 8h payoff after costs, otherwise HOLD",
        "split": {"train_end": str(train_end), "calibration_end": str(calibration_end)},
        "class_counts": counts.tolist(), "calibration": calibration_metrics, "holdout": holdout_metrics,
        "note": "MACD is an input feature only; it is not a mandatory entry crossover.",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with SCALER_PATH.open("wb") as handle: pickle.dump(scaler, handle)
    CALIBRATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
