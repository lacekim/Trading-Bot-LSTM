import unittest

import pandas as pd

from trading_bot_v4.backtesting.trade_filter_research import EntryGate, apply_entry_gate


class TradeFilterResearchTests(unittest.TestCase):
    def test_gate_requires_confidence_trend_and_green_candle(self):
        rows = 205
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "model_probability": [0.80] * rows,
            "close": list(range(1, rows + 1)),
            "open": [value - 0.5 for value in range(1, rows + 1)],
            "atr": [2.0] * rows,
        })
        result = apply_entry_gate(frame, EntryGate(0.72, 200, True, 0.0))
        self.assertTrue((result.iloc[:199]["model_direction"] == "HOLD").all())
        self.assertEqual(result.iloc[-1]["model_direction"], "LONG")

    def test_gate_does_not_change_input_frame(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "model_probability": [0.9, 0.9], "close": [2.0, 1.0],
            "open": [1.0, 2.0], "atr": [0.1, 0.1],
            "model_direction": ["LONG", "LONG"],
        })
        apply_entry_gate(frame, EntryGate(0.72, 0, True, 0.0))
        self.assertEqual(frame["model_direction"].tolist(), ["LONG", "LONG"])


if __name__ == "__main__":
    unittest.main()
