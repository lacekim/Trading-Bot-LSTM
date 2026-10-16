"""Unit tests for the cleaning rules in export_venue_candles.py (no pyarrow needed)."""
import unittest

import pandas as pd

import export_venue_candles as X


class T(unittest.TestCase):
    def test_kraken_drops_bad_close_sorts_dedups_utc(self):
        df = pd.DataFrame({
            "date": ["2026-01-01 02:00:00", "2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-01 01:00:00", "2026-01-01 03:00:00"],
            "open": [1, 1, 1, 1, 1], "high": [2, 2, 2, 3, 2], "low": [1, 1, 1, 1, 1],
            "close": [1.5, 1.0, 0.0, 1.7, None], "volume": [1, 1, 1, 9, 1]})
        out = X.clean_kraken(df)
        # the 01:00 bar with close 0.0 is dropped, but its duplicate with close 1.7 is kept; the null-close 03:00 bar is dropped
        self.assertEqual(list(out["date"].dt.strftime("%H")), ["00", "01", "02"])
        self.assertEqual(out["close"].tolist(), [1.0, 1.7, 1.5])
        self.assertEqual(str(out["date"].dt.tz), "UTC")
        self.assertEqual(out.columns.tolist(), ["date", "open", "high", "low", "close", "volume"])

    def test_alpaca_last_duplicate_kept_and_missing_close_dropped(self):
        df = pd.DataFrame({"symbol": ["B", "A", "A", "A", "B"],
                           "date": ["2026-01-02", "2026-01-02", "2026-01-01", "2026-01-01", "2026-01-01"],
                           "close": [10.0, 5.0, 4.0, 4.5, None], "volume": [1, 2, 3, 4, 5]})
        out = X.clean_alpaca(df)
        self.assertEqual(out["symbol"].tolist(), ["A", "A", "B"])
        a1 = out[(out.symbol == "A") & (out.date == pd.Timestamp("2026-01-01", tz="UTC"))].iloc[0]
        self.assertEqual((a1["close"], a1["volume"]), (4.5, 4))
        self.assertEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main()
