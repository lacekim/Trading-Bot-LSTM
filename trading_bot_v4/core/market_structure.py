"""Optional SMC-style market-structure analysis. Disabled by default and not connected to live trading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketStructureAnalysis:
    enabled: bool = False
    regime: str = "neutral"
    summary: dict | None = None


class MarketStructureAnalyzer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def analyze(self, df):
        if not self.enabled:
            return MarketStructureAnalysis(enabled=False, regime="neutral", summary={})
        return MarketStructureAnalysis(enabled=True, regime="neutral", summary={"note": "SMC analysis disabled in V4"})
