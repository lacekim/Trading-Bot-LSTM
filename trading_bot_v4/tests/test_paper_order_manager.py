import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_bot_v4.execution.order_manager import OrderManager


def signal(timestamp="2026-01-01T01:00:00Z", direction="LONG", price=100.0):
    return pd.DataFrame([{"timestamp": timestamp, "symbol": "BTC", "timeframe": "1h",
                          "model_direction": direction, "model_probability": 0.8, "price": price}])


class PaperOrderManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "paper.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_persists_and_deduplicates_position(self):
        manager = OrderManager(self.db)
        first = manager.process_signals(signal())
        again = manager.process_signals(signal())
        manager.close()
        restored = OrderManager(self.db).sync()
        self.assertEqual(first.orders_opened, 1)
        self.assertEqual(again.orders_opened, 0)
        self.assertEqual(restored.open_positions, 1)

    def test_reversal_closes_then_opens_new_direction(self):
        manager = OrderManager(self.db)
        manager.process_signals(signal())
        result = manager.process_signals(signal("2026-01-01T02:00:00Z", "SHORT", 101.0))
        row = manager.connection.execute("SELECT direction FROM positions WHERE symbol='BTC'").fetchone()
        self.assertEqual(result.orders_closed, 1)
        self.assertEqual(result.orders_opened, 1)
        self.assertEqual(row["direction"], "SHORT")
        manager.close()

    def test_risk_limit_rejects_new_order(self):
        with patch("trading_bot_v4.execution.order_manager.Config.PAPER_MAX_OPEN_POSITIONS", 0):
            manager = OrderManager(self.db)
            result = manager.process_signals(signal())
            self.assertEqual(result.signals_rejected, 1)
            self.assertEqual(result.open_positions, 0)
            manager.close()


if __name__ == "__main__":
    unittest.main()
