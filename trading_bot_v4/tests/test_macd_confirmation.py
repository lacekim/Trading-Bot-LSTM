import unittest

import pandas as pd

from trading_bot_v4.utils.macd_confirmation import macd_entry_confirmation


class MacdConfirmationTests(unittest.TestCase):
    def test_requires_directional_macd_confirmation(self):
        signals = pd.DataFrame([
            {"model_direction": "LONG", "macd_line": 2, "macd_signal": 1, "macd_histogram": 1, "macd_histogram_previous": -1, "price_vs_ma200": .1},
            {"model_direction": "LONG", "macd_line": 0, "macd_signal": 1, "macd_histogram": -1, "macd_histogram_previous": 1, "price_vs_ma200": .1},
            {"model_direction": "SHORT", "macd_line": 0, "macd_signal": 1, "macd_histogram": -1, "macd_histogram_previous": 1, "price_vs_ma200": -.1},
            {"model_direction": "SHORT", "macd_line": 2, "macd_signal": 1, "macd_histogram": 1, "macd_histogram_previous": -1, "price_vs_ma200": -.1},
            {"model_direction": "HOLD", "macd_line": 2, "macd_signal": 1, "macd_histogram": 1, "macd_histogram_previous": -1, "price_vs_ma200": .1},
        ])
        self.assertEqual(macd_entry_confirmation(signals).tolist(), [True, False, True, False, False])


if __name__ == "__main__":
    unittest.main()
