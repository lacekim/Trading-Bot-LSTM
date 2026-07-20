import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_bot_v4.execution.order_manager import OrderManager
from trading_bot_v4.execution.position_monitor import (
    GMXMinutePriceProvider, GMXTickerPriceProvider, MarketPrice, PositionMonitor,
)
from trading_bot_v4.shutdown_controller import ShutdownController


def _signal():
    return pd.DataFrame([{
        "timestamp": "2026-01-01T01:00:00Z", "symbol": "BTC", "timeframe": "1h",
        "model_direction": "LONG", "model_probability": 0.8, "price": 100.0,
    }])


class FakeNotifier:
    is_running = True
    def __init__(self): self.messages = []
    def broadcast(self, text): self.messages.append(text)


class StaticProvider:
    def __init__(self, price): self.price = price; self.calls = []
    def fetch(self, symbol):
        self.calls.append(symbol)
        return MarketPrice(symbol, self.price, pd.Timestamp.now(tz="UTC").isoformat(), 0.0)


class FailingProvider:
    def fetch(self, symbol): raise ConnectionError(f"feed unavailable for {symbol}")


class PositionMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "paper.sqlite3"
        manager = OrderManager(self.db)
        manager.process_signals(_signal())
        manager.close()

    def tearDown(self): self.temp.cleanup()

    def test_monitor_closes_stop_immediately_and_persists_observed_price(self):
        controller, notifier, provider = ShutdownController(), FakeNotifier(), StaticProvider(97.0)
        monitor = PositionMonitor(controller, notifier, provider, self.db, interval_seconds=1)
        manager = OrderManager(self.db)
        events = monitor.run_once(manager)
        self.assertEqual(manager.open_position_count(), 0)
        self.assertEqual(events[0]["reason"], "stop_loss")
        row = manager.connection.execute("SELECT observed_price,accounting_price FROM position_exit_events").fetchone()
        self.assertEqual(row["observed_price"], 97.0)
        self.assertAlmostEqual(row["accounting_price"], 98.0686, places=4)
        self.assertTrue(any("IMMEDIATE PAPER EXIT" in text for text in notifier.messages))
        manager.close()

    def test_monitor_checks_persisted_position_without_watchlist(self):
        provider = StaticProvider(100.0)
        monitor = PositionMonitor(ShutdownController(), FakeNotifier(), provider, self.db)
        manager = OrderManager(self.db)
        monitor.run_once(manager)
        self.assertEqual(provider.calls, ["BTC"])
        self.assertEqual(manager.open_position_count(), 1)
        manager.close()

    def test_repeated_feed_failures_alert_at_threshold(self):
        notifier = FakeNotifier()
        monitor = PositionMonitor(ShutdownController(), notifier, FailingProvider(), self.db, failure_threshold=3)
        manager = OrderManager(self.db)
        for _ in range(3): monitor.run_once(manager)
        self.assertEqual(monitor._consecutive_failures, 3)
        self.assertTrue(any("POSITION MONITOR DEGRADED" in text for text in notifier.messages))
        status = manager.connection.execute("SELECT value FROM runtime_state WHERE key='position_monitor_status'").fetchone()["value"]
        self.assertEqual(status, "degraded")
        manager.close()

    def test_monitor_thread_starts_immediately_and_stops_cleanly(self):
        monitor = PositionMonitor(ShutdownController(), FakeNotifier(), StaticProvider(100.0), self.db, interval_seconds=1)
        monitor.start()
        for _ in range(20):
            if monitor._last_heartbeat_monotonic:
                break
            threading.Event().wait(0.05)
        self.assertTrue(monitor.is_healthy())
        monitor.stop()
        self.assertFalse(monitor.is_running)


class PriceProviderTests(unittest.TestCase):
    def test_newest_one_minute_close_is_selected(self):
        class Response:
            def raise_for_status(self): return None
            def json(self): return {"candles": [[100, 1, 2, 1, 1.5], [200, 1, 3, 1, 2.5]]}
        with patch("trading_bot_v4.execution.position_monitor.requests.get", return_value=Response()), \
             patch("trading_bot_v4.execution.position_monitor.Config.POSITION_MONITOR_MAX_PRICE_AGE_SECONDS", 10**12):
            quote = GMXMinutePriceProvider().fetch("BTC")
        self.assertEqual(quote.price, 2.5)

    def test_ticker_uses_min_for_long_and_max_for_short_with_token_scaling(self):
        class Response:
            def __init__(self, payload): self.payload = payload
            def raise_for_status(self): return None
            def json(self): return self.payload

        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        responses = {
            "tokens": Response({"tokens": [{"symbol": "PUMP", "decimals": 18}]}),
            "tickers": Response([{
                "tokenSymbol": "PUMP", "minPrice": "2000000000",
                "maxPrice": "2100000000", "updatedAt": now_ms,
            }]),
        }

        def fake_get(url, **_kwargs):
            return responses["tokens" if url.endswith("/tokens") else "tickers"]

        with patch("trading_bot_v4.execution.position_monitor.requests.get", side_effect=fake_get):
            provider = GMXTickerPriceProvider()
            long_quote = provider.fetch("PUMP", "LONG")
            short_quote = provider.fetch("PUMP", "SHORT")
        self.assertAlmostEqual(long_quote.price, 0.002)
        self.assertAlmostEqual(short_quote.price, 0.0021)
        self.assertEqual(long_quote.source, "GMX_TICKER_MIN")
        self.assertEqual(short_quote.source, "GMX_TICKER_MAX")

    def test_ticker_failure_falls_back_to_one_minute_provider(self):
        fallback = StaticProvider(0.002)
        provider = GMXTickerPriceProvider(fallback=fallback)
        with patch.object(provider, "_fetch_ticker", side_effect=ConnectionError("ticker down")):
            quote = provider.fetch("PUMP", "LONG")
        self.assertEqual(quote.source, "GMX_1M_FALLBACK")
        self.assertIn("ticker down", quote.fallback_reason)


if __name__ == "__main__": unittest.main()
