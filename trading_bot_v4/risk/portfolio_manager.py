"""Portfolio manager placeholder for the modular V4 implementation."""

from __future__ import annotations


class PortfolioManager:
    def __init__(self):
        self.equity = 0.0

    def update_equity(self, value: float):
        self.equity = value
        return self.equity
