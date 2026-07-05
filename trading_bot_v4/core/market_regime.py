"""Optional market-regime helper that is disabled by default and does not alter the original signal path."""

from __future__ import annotations


class MarketRegimeDetector:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def detect(self, df):
        if not self.enabled:
            return "neutral"
        return "neutral"
