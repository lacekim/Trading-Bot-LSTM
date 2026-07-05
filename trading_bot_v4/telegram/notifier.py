"""Telegram notifier wrapper around the original bot notifier implementation."""

from __future__ import annotations

from trading_bot import TelegramNotifier as LegacyTelegramNotifier


class V4TelegramNotifier(LegacyTelegramNotifier):
    """Compatibility wrapper for the original Telegram notifier."""

    pass


TelegramNotifier = V4TelegramNotifier
