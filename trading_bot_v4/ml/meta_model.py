"""Meta-model placeholder that currently routes to the original CNN/LSTM model path."""

from __future__ import annotations

from trading_bot_v4.ml.cnn_lstm_model import V4CNNLSTMClassifier


class MetaModel:
    def __init__(self):
        self.model = V4CNNLSTMClassifier()

    def predict(self, features):
        return self.model.predict(features)
