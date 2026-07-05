"""Optional liquidity-sweep module for future SMC expansion. Disabled by default."""

from __future__ import annotations


class LiquiditySweepDetector:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def detect(self, df):
        if not self.enabled:
            return []
        return []
