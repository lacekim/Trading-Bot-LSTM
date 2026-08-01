import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading_bot_v4.features.kraken_cross_venue import load_lagged_kraken_features


class KrakenCrossVenueTests(unittest.TestCase):
    def test_peer_features_are_lagged_one_completed_candle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for hour in range(30):
                rows.append([1_767_225_600 + hour * 3600, 100 + hour, 101 + hour,
                             99 + hour, 100 + hour, 10 + hour, 20 + hour])
            pd.DataFrame(rows).to_csv(root / "XBTUSD_60.csv", header=False, index=False)
            features = load_lagged_kraken_features("BTC", root)
        # At hour 2 the exposed one-hour return belongs to hour 1, not hour 2.
        row = features.iloc[2]
        self.assertAlmostEqual(row["kraken_return_1h"], 101 / 100 - 1)


if __name__ == "__main__":
    unittest.main()
