"""Kraken execution wrapper that reuses the original KrakenTrader implementation."""

from __future__ import annotations

from trading_bot import KrakenTrader as LegacyKrakenTrader


class V4KrakenTrader(LegacyKrakenTrader):
    """Compatibility layer around the original trading bot trader implementation."""

    pass


KrakenTrader = V4KrakenTrader
