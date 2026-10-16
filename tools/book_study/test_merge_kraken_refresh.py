"""Unit tests for the join rules in merge_kraken_refresh.py (no pyarrow needed)."""
import unittest

import pandas as pd

import merge_kraken_refresh as M


def bars(start, n, close0=10.0, step=0.0, skip=()):
    idx = [t for i, t in enumerate(pd.date_range(start, periods=n, freq="1h", tz="UTC")) if i not in skip]
    c = [close0 + step * i for i in range(len(idx))]
    return pd.DataFrame({"date": idx, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})


class T(unittest.TestCase):
    def test_old_wins_new_appended_after_and_last_new_bar_dropped(self):
        old = bars("2026-01-01 00:00", 10, close0=10.0)                      # 00:00 .. 09:00
        new = bars("2026-01-01 05:00", 12, close0=11.0)                      # 05:00 .. 16:00, last bar 16:00 is partial
        out, info = M.merge_old_new(old, new)
        self.assertEqual(out["date"].iloc[-1], pd.Timestamp("2026-01-01 15:00", tz="UTC"))     # 16:00 dropped
        self.assertEqual(len(out), 16)
        self.assertTrue((out["close"].iloc[:10] == 10.0).all())                                # old values kept on the overlap 05:00 .. 09:00
        self.assertEqual(info["new_bars_added"], 6)                                            # 10:00 .. 15:00
        self.assertEqual(info["junction_gap_hours"], 0)
        self.assertEqual(info["overlap_bars"], 5)
        self.assertAlmostEqual(info["overlap_close_max_rel_diff"], 0.1, places=12)
        self.assertEqual(info["dropped_partial_bar"], "2026-01-01 16:00:00+00:00")
        self.assertEqual(out["date"].is_monotonic_increasing and out["date"].is_unique, True)

    def test_gap_at_the_junction_is_reported(self):
        old = bars("2026-01-01 00:00", 5)                                                      # ends 04:00
        new = bars("2026-01-01 08:00", 5)                                                      # starts 08:00 -> 05,06,07 missing
        out, info = M.merge_old_new(old, new)
        self.assertEqual(info["junction_gap_hours"], 3)
        self.assertEqual(info["overlap_bars"], 0)
        self.assertIsNone(info["overlap_close_max_rel_diff"])
        self.assertEqual(len(out), 5 + 4)

    def test_old_only_is_unchanged_and_new_only_loses_its_last_bar(self):
        old = bars("2026-01-01", 4)
        out, info = M.merge_old_new(old, None)
        pd.testing.assert_frame_equal(out, old)
        self.assertEqual(info["new_bars_added"], 0)
        new = bars("2026-01-01", 4)
        out, info = M.merge_old_new(None, new)
        self.assertEqual(len(out), 3)
        self.assertEqual(info["new_bars_added"], 3)

    def test_new_entirely_inside_old_adds_nothing(self):
        old = bars("2026-01-01", 20)
        new = bars("2026-01-01 03:00", 5)
        out, info = M.merge_old_new(old, new)
        self.assertEqual(len(out), 20)
        self.assertEqual(info["new_bars_added"], 0)
        self.assertIsNone(info["junction_gap_hours"])
        self.assertEqual(info["overlap_bars"], 4)


if __name__ == "__main__":
    unittest.main()
