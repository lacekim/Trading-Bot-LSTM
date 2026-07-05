"""Position sizing helper that preserves the original risk-percentage logic."""

from __future__ import annotations

from trading_bot_v4.config_v4 import V4Config as Config


class PositionSizer:
    def __init__(self, risk_percentage: float | None = None):
        self.risk_percentage = risk_percentage or Config.RISK_PERCENTAGE

    def calculate_position_size(self, capital: float, entry_price: float, stop_distance: float):
        if stop_distance <= 0:
            return 0.0
        risk_amount = capital * (self.risk_percentage / 100.0)
        return risk_amount / stop_distance
