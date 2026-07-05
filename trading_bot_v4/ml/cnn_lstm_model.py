"""CNN/LSTM model wrapper that preserves the existing architecture from train_model.py."""

from __future__ import annotations

from train_model import CNNLSTMClassifier as LegacyCNNLSTMClassifier


class V4CNNLSTMClassifier(LegacyCNNLSTMClassifier):
    """Drop-in compatibility wrapper for the original CNN/LSTM classifier."""

    pass


CNNLSTMClassifier = V4CNNLSTMClassifier
