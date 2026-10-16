import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import validate_kraken_hourly as V


class T(unittest.TestCase):
    def test_minutes_to_hours_by_hand(self):
        # hour 0: minutes 0,1,2 ; hour 1 (3600): minute at 3600 only
        m = pd.DataFrame({"ts": [0, 60, 120, 3600], "open": [1.0, 2, 3, 9], "high": [5.0, 4, 8, 9], "low": [0.5, 2, 1, 9], "close": [2.0, 3, 4, 9], "volume": [1.0, 1, 2, 5]})
        h = V.minutes_to_hours(m)
        self.assertEqual(list(h["ts"]), [0, 3600])
        r = h.iloc[0]
        self.assertEqual((r.open, r.high, r.low, r.close, r.volume), (1.0, 8.0, 0.5, 4.0, 4.0))
        self.assertEqual(h.iloc[1].volume, 5.0)

    def test_compare_by_hand(self):
        a = pd.DataFrame({"ts": [0, 3600, 7200], "open": 1.0, "high": 2.0, "low": 1.0, "close": [100.0, 100.0, 100.0], "volume": [10.0, 10.0, 10.0]})
        b = pd.DataFrame({"ts": [3600, 7200, 10800], "open": 1.0, "high": 2.0, "low": 1.0, "close": [100.0, 102.0, 5.0], "volume": [10.0, 20.0, 1.0]})
        r = V.compare(a, b)
        self.assertEqual((r["hours_both"], r["only_a"], r["only_b"]), (2, 1, 1))
        self.assertAlmostEqual(r["close_gt_0.1pct"], 0.5)
        self.assertAlmostEqual(r["close_gt_1pct"], 0.5)
        self.assertAlmostEqual(r["close_max"], 0.02)
        self.assertAlmostEqual(r["volume_ratio_b_over_a"], 30.0 / 20.0)

    def test_identical_series_have_zero_difference(self):
        rng = np.random.default_rng(0)
        c = 100 + rng.random(50)
        a = pd.DataFrame({"ts": np.arange(50) * 3600, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": rng.random(50)})
        r = V.compare(a, a.copy())
        self.assertEqual((r["only_a"], r["only_b"], r["close_max"], r["volume_ratio_b_over_a"]), (0, 0, 0.0, 1.0))

    def test_read_dump_empty_file_and_missing_file(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "BNBUSD_1.csv"
            p.write_bytes(b"")
            self.assertEqual(len(V.read_dump(p)), 0)
            self.assertEqual(len(V.read_dump(Path(t) / "nope.csv")), 0)
            q = Path(t) / "x_60.csv"
            q.write_text("3600,1,2,0.5,1.5,10,3\n0,1,1,1,1,1,1\n3600,1,2,0.5,1.6,11,3\n7200,0,0,0,0,0,0\n")
            d = V.read_dump(q)
            self.assertEqual(list(d["ts"]), [0, 3600])                    # sorted, duplicate ts keeps the last, zero-close row dropped
            self.assertEqual(d["close"].iloc[1], 1.6)

    def test_read_candles_epoch(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "c.csv"
            p.write_text("date,open,high,low,close,volume\n1970-01-01 01:00:00,1,2,1,1,5\n1970-01-01 00:00:00,1,2,1,1,4\n")
            c = V.read_candles(p)
            self.assertEqual(list(c["ts"]), [0, 3600])


if __name__ == "__main__":
    unittest.main()
