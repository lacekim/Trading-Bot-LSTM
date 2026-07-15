import unittest
from unittest.mock import patch

import pandas as pd

from trading_bot_v4.research.market_scanner import scan_market_momentum


class MarketScannerTests(unittest.TestCase):
    @patch("trading_bot_v4.research.market_scanner.MARKET_MOMENTUM_PATH")
    @patch("trading_bot_v4.research.market_scanner.load_gmx_ohlc")
    @patch("trading_bot_v4.research.market_scanner.list_gmx_symbols")
    def test_market_leaders_are_ranked_by_recent_momentum(self, symbols, load, output_path):
        symbols.return_value = ["FAST", "SLOW"]
        index = pd.date_range("2026-01-01", periods=30, freq="h")
        frames = {
            "FAST": pd.DataFrame({"Close": [100.0] * 29 + [110.0]}, index=index),
            "SLOW": pd.DataFrame({"Close": [100.0] * 29 + [101.0]}, index=index),
        }
        load.side_effect = lambda symbol, timeframe: frames[symbol]
        output_path.parent.mkdir.return_value = None

        report = scan_market_momentum("1h", limit=2)

        self.assertEqual(report["symbol"].tolist(), ["FAST", "SLOW"])
        self.assertGreater(report.iloc[0]["return_24h_pct"], report.iloc[1]["return_24h_pct"])


if __name__ == "__main__":
    unittest.main()
