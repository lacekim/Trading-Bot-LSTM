import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_bot_v4.execution.order_manager import OrderManager
from trading_bot_v4.execution.shutdown import ShutdownCoordinator


def _signal(symbol="BTC"):
    return pd.DataFrame([{"timestamp": "2026-01-01T01:00:00Z", "symbol": symbol,
                          "timeframe": "1h", "model_direction": "LONG",
                          "model_probability": 0.8, "price": 100.0}])


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.market_cap_risk = patch(
            "trading_bot_v4.execution.order_manager.Config.MARKET_CAP_RISK_ENABLED", False
        )
        self.market_cap_risk.start()
        self.orders = OrderManager(Path(self.temp.name) / "paper.sqlite3")
        self.orders.process_signals(_signal())

    def tearDown(self):
        self.orders.close()
        self.market_cap_risk.stop()
        self.temp.cleanup()

    def test_graceful_preserves_position_and_blocks_entries(self):
        report = ShutdownCoordinator(self.orders).execute("GRACEFUL")
        self.assertFalse(report.account_flat)
        self.assertEqual(self.orders.sync().open_positions, 1)
        self.assertFalse(self.orders.new_entries_allowed())

    def test_close_positions_flattens_and_reconciles(self):
        report = ShutdownCoordinator(self.orders).execute("CLOSE_POSITIONS")
        self.assertTrue(report.account_flat)
        self.assertEqual(report.positions_closed, 1)

    def test_close_cancels_pending_entries_after_preserving_exit_until_flat(self):
        self.orders.connection.execute("""INSERT INTO orders(
            order_id,signal_id,symbol,side,status,expected_price,fill_price,notional,fee,created_at,order_kind,reduce_only
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("pending-entry", "s1", "ETH", "LONG", "PENDING", 1, 1, 10, 0, "now", "ENTRY", 0))
        self.orders.connection.execute("""INSERT INTO orders(
            order_id,signal_id,symbol,side,status,expected_price,fill_price,notional,fee,created_at,order_kind,reduce_only
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("protective-exit", "s2", "BTC", "SHORT", "PENDING", 90, 90, 10, 0, "now", "EXIT", 1))
        self.orders.connection.commit()
        report = ShutdownCoordinator(self.orders).execute("CLOSE_POSITIONS")
        statuses = dict(self.orders.connection.execute("SELECT order_id,status FROM orders WHERE order_id LIKE 'pending-%' OR order_id='protective-exit'").fetchall())
        self.assertEqual(report.pending_orders_cancelled, 1)
        self.assertEqual(statuses["pending-entry"], "CANCELLED")
        self.assertEqual(statuses["protective-exit"], "CANCELLED")

    def test_emergency_alerts_and_flattens(self):
        alerts = []
        report = ShutdownCoordinator(self.orders, alerts.append).execute("EMERGENCY")
        self.assertTrue(report.account_flat)
        self.assertTrue(alerts)

    def test_emergency_retries_failed_close_and_does_not_claim_flat(self):
        original = self.orders.close_all_positions
        attempts = []
        def fail_then_close(reason):
            attempts.append(reason)
            if len(attempts) < 3:
                raise RuntimeError("temporary close failure")
            return original(reason)
        self.orders.close_all_positions = fail_then_close
        report = ShutdownCoordinator(self.orders).execute("EMERGENCY")
        self.assertEqual(len(attempts), 3)
        self.assertTrue(report.account_flat)


if __name__ == "__main__":
    unittest.main()
