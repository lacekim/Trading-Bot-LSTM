import tempfile
import unittest
from pathlib import Path

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
        self.orders = OrderManager(Path(self.temp.name) / "paper.sqlite3")
        self.orders.process_signals(_signal())

    def tearDown(self):
        self.orders.close()
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

    def test_emergency_alerts_and_flattens(self):
        alerts = []
        report = ShutdownCoordinator(self.orders, alerts.append).execute("EMERGENCY")
        self.assertTrue(report.account_flat)
        self.assertTrue(alerts)


if __name__ == "__main__":
    unittest.main()
