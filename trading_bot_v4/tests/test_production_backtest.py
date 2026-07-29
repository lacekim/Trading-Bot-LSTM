import pandas as pd
import unittest
from unittest.mock import patch

from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol


def _signals(rows):
    return pd.DataFrame(rows, columns=[
        "timestamp", "symbol", "timeframe", "model_probability", "model_direction",
        "price", "open", "high", "low", "close",
    ])


class ProductionBacktestTests(unittest.TestCase):
    @patch("trading_bot_v4.backtesting.production_backtest.Config.PAPER_MIN_BARS_BETWEEN_TRADES", 0)
    def test_production_simulator_trades_long_and_short(self):
        frame = _signals([
            ("2025-01-01T00:00:00Z", "XRP", "1h", .8, "LONG", 100, 100, 101, 99, 100),
            ("2025-01-01T01:00:00Z", "XRP", "1h", .8, "HOLD", 104, 101, 105, 101, 104),
            ("2025-01-01T02:00:00Z", "XRP", "1h", .8, "SHORT", 100, 100, 101, 99, 100),
            ("2025-01-01T03:00:00Z", "XRP", "1h", .8, "HOLD", 96, 99, 99, 95, 96),
        ])
        summary, trades = simulate_production_symbol(frame, "XRP", 10_000)
        self.assertEqual(summary["long_trades"], 1)
        self.assertEqual(summary["short_trades"], 1)
        self.assertEqual(set(trades["exit_reason"]), {"take_profit"})

    def test_production_simulator_respects_entry_eligibility(self):
        frame = _signals([
            ("2025-01-01T00:00:00Z", "XRP", "1h", .8, "LONG", 100, 100, 101, 99, 100),
            ("2025-01-01T01:00:00Z", "XRP", "1h", .8, "HOLD", 104, 101, 105, 101, 104),
        ])
        summary, trades = simulate_production_symbol(frame, "XRP", 10_000, entry_eligible=False)
        self.assertEqual(summary["trades"], 0)
        self.assertTrue(trades.empty)

    def test_direction_specific_qualification_does_not_hide_reversal(self):
        frame = _signals([
            ("2025-01-01T00:00:00Z", "XRP", "1h", .8, "LONG", 100, 100, 101, 99, 100),
            ("2025-01-01T01:00:00Z", "XRP", "1h", .8, "SHORT", 101, 101, 102, 100, 101),
        ])
        frame["_entry_eligible"] = [True, False]
        summary, trades = simulate_production_symbol(frame, "XRP", 10_000)
        self.assertEqual(summary["trades"], 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "signal_reversal")

    @patch("trading_bot_v4.backtesting.production_backtest.Config.PAPER_STOP_TARGET_PRIORITY", "STOP_FIRST")
    def test_stop_has_priority_when_both_stop_and_target_touch(self):
        frame = _signals([
            ("2025-01-01T00:00:00Z", "XRP", "1h", .8, "LONG", 100, 100, 101, 99, 100),
            ("2025-01-01T01:00:00Z", "XRP", "1h", .8, "HOLD", 100, 100, 105, 97, 100),
        ])
        _, trades = simulate_production_symbol(frame, "XRP", 10_000)
        self.assertEqual(trades.iloc[0]["exit_reason"], "stop_loss")
