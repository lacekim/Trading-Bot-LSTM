import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_bot_v4.execution.order_manager import OrderManager
from trading_bot_v4.execution.position_monitor import MarketPricePair
from trading_bot_v4.execution.qualified_market_monitor import QualifiedMarketMonitor
from trading_bot_v4.shutdown_controller import ShutdownController


class PairProvider:
    def fetch_pair(self, symbol):
        return MarketPricePair(
            symbol, 99.0, 101.0, pd.Timestamp.now(tz="UTC").isoformat(), 0.1
        )


class QualifiedMarketMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "paper.sqlite3"
        self.market_cap_risk = patch(
            "trading_bot_v4.execution.order_manager.Config.MARKET_CAP_RISK_ENABLED", False
        )
        self.market_cap_risk.start()

    def tearDown(self):
        self.market_cap_risk.stop()
        self.temp.cleanup()

    def test_records_pair_spread_and_health_without_generating_entry(self):
        manager = OrderManager(self.db)
        monitor = QualifiedMarketMonitor(
            ShutdownController(), lambda: (["BTC"], ["CHZ"]),
            provider=PairProvider(), db_path=self.db,
        )
        with patch("trading_bot_v4.execution.qualified_market_monitor.Config.QUALIFIED_MARKET_MAX_SPREAD_BPS", 500):
            result = monitor.run_once(manager)
        rows = manager.connection.execute(
            "SELECT symbol,min_price,max_price,spread_bps,status FROM qualified_market_snapshots ORDER BY symbol"
        ).fetchall()
        self.assertEqual(result["checked"], 2)
        self.assertEqual(manager.open_position_count(), 0)
        self.assertEqual([row["symbol"] for row in rows], ["BTC", "CHZ"])
        self.assertTrue(all(row["status"] == "healthy" for row in rows))
        self.assertAlmostEqual(rows[0]["spread_bps"], 200.0, places=4)
        self.assertTrue(monitor._last_heartbeat_monotonic > 0)
        manager.close()

    def test_stopped_thread_is_reported_unhealthy(self):
        monitor = QualifiedMarketMonitor(
            ShutdownController(), lambda: ([], []), provider=PairProvider(), db_path=self.db,
        )
        self.assertFalse(monitor.is_healthy())
        self.assertEqual(monitor.health_issue(), "thread_stopped")

    def test_empty_universe_checks_real_canary_feed(self):
        manager = OrderManager(self.db)
        monitor = QualifiedMarketMonitor(
            ShutdownController(), lambda: ([], []), provider=PairProvider(), db_path=self.db,
        )
        with patch("trading_bot_v4.execution.qualified_market_monitor.Config.QUALIFIED_MARKET_CANARY_SYMBOL", "BTC"), \
             patch("trading_bot_v4.execution.qualified_market_monitor.Config.QUALIFIED_MARKET_MAX_SPREAD_BPS", 500):
            result = monitor.run_once(manager)
        self.assertTrue(result["canary_only"])
        self.assertEqual(result["symbols"], ["BTC"])
        self.assertEqual(result["checked"], 1)
        manager.close()

    def test_hourly_entry_uses_fresh_directional_snapshot_and_audits_prices(self):
        manager = OrderManager(self.db)
        observed_at = pd.Timestamp.now(tz="UTC").isoformat()
        manager.record_qualified_market_snapshot("BTC", 99.0, 101.0, observed_at, 200.0, 12.0, 0.1)
        timestamp = (pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)).isoformat()
        signal = pd.DataFrame([{
            "timestamp": timestamp, "symbol": "BTC", "timeframe": "1h",
            "model_direction": "LONG", "model_probability": .9, "price": 100.0,
            "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0,
            "candle_gap_seconds": 3600.0,
        }])
        result = manager.process_signals(signal)
        order = manager.connection.execute(
            "SELECT expected_price,model_price,observed_price,price_source,model_to_observed_bps FROM orders"
        ).fetchone()
        self.assertEqual(result.orders_opened, 1)
        self.assertEqual(order["model_price"], 100.0)
        self.assertEqual(order["observed_price"], 101.0)
        self.assertEqual(order["expected_price"], 101.0)
        self.assertEqual(order["price_source"], "GMX_TICKER_SNAPSHOT")
        self.assertAlmostEqual(order["model_to_observed_bps"], 100.0)
        manager.close()


if __name__ == "__main__":
    unittest.main()
