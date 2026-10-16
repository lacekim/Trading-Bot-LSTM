"""Two additional, genuinely different architectures alongside V4CNNLSTMClassifier
(the existing CNN+LSTM hybrid), so a per-asset ensemble can vote across distinct
model families instead of relying on one architecture's prediction alone --
mirrors binance_margin_bot's Model_1 (CNN)/Model_2 (LSTM)/Model_3 (hybrid) split.

Same build_model(input_shape)/save_model(path) interface as V4CNNLSTMClassifier
so callers that already do classifier.build_model(...); classifier.model.fit(...)
work unchanged regardless of which architecture is plugged in.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2

from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_ensemble_architectures")


class CNNOnlyClassifier:
    """Pure convolutional classifier -- no recurrent layers, only local
    pattern detection across the sequence window."""

    def __init__(self):
        self.model = None

    def build_model(self, input_shape):
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation="relu", padding="same",
                   kernel_regularizer=l2(0.01), input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),

            Conv1D(filters=64, kernel_size=3, activation="relu", padding="same",
                   kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),

            GlobalAveragePooling1D(),

            Dense(16, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss="binary_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        self.model = model
        logger.info(f"CNN-only classifier built, {model.count_params():,} params")
        return model

    def save_model(self, path):
        self.model.save(path)
        logger.info(f"CNN-only model saved: {path}")


class LSTMOnlyClassifier:
    """Pure recurrent classifier -- no convolutional layers, only sequence/
    temporal-context modeling."""

    def __init__(self):
        self.model = None

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001), input_shape=input_shape),
            Dropout(0.5),

            LSTM(16, return_sequences=False, kernel_regularizer=l2(0.001)),
            Dropout(0.5),

            Dense(16, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss="binary_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        self.model = model
        logger.info(f"LSTM-only classifier built, {model.count_params():,} params")
        return model

    def save_model(self, path):
        self.model.save(path)
        logger.info(f"LSTM-only model saved: {path}")
