"""V4 signal engine wrapper that preserves the original probability-based entry logic."""

from __future__ import annotations

from trading_bot_v4.config_v4 import V4Config as Config


class SignalEngine:
    def __init__(self, threshold: float | None = None):
        self.threshold = threshold or Config.MIN_SIGNAL_THRESHOLD

    def generate_signals(self, probabilities):
        long_threshold = self.threshold
        short_threshold = 1 - self.threshold
        signals = []
        for probability in probabilities:
            if probability > long_threshold:
                signals.append("LONG")
            elif probability < short_threshold:
                signals.append("SHORT")
            else:
                signals.append("HOLD")
        return signals
