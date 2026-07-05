"""Order manager placeholder that currently delegates to the original trader state model."""

from __future__ import annotations


class OrderManager:
    def __init__(self, trader=None):
        self.trader = trader

    def sync(self):
        return None
