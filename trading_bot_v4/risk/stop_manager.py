"""Stop management helper that preserves the original ATR-based stop parameters."""

from __future__ import annotations

from trading_bot_v4.config_v4 import V4Config as Config


class StopManager:
    def __init__(self, atr_period: int | None = None):
        self.atr_period = atr_period or Config.ATR_PERIOD

    def calculate_levels(self, entry_price: float, atr: float):
        stop_loss = entry_price - atr * Config.ATR_SL_MULTIPLIER
        take_profit = entry_price + atr * Config.ATR_TP_MULTIPLIER
        return stop_loss, take_profit
