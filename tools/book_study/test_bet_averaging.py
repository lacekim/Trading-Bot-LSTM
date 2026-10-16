"""Tests for bet_averaging.py (hand-worked cases and a brute-force comparison).   python -m unittest test_bet_averaging -v
A pipeline-level test (the run's own per-event numbers are reproduced and the averaged output is written) runs only when BA_TEST_RUN names a run folder."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bet_averaging as BA  # noqa: E402

CS = 0.0012          # cost per side as a fraction


def hourly(n, start=0):
    return start + 3600 * np.arange(n, dtype=np.int64)


class TestAverage(unittest.TestCase):
    def test_single_event(self):
        avg, C = BA.average_positions(5, [1], [3], [1.0])
        np.testing.assert_array_equal(avg, [0, 1, 1, 0, 0])
        np.testing.assert_array_equal(C, [0, 1, 1, 0, 0])

    def test_overlap_is_plain_mean(self):
        avg, C = BA.average_positions(6, [1, 2], [4, 4], [0.5, 1.0])
        np.testing.assert_allclose(avg, [0, 0.5, 0.75, 0.75, 0, 0], atol=1e-15)
        np.testing.assert_array_equal(C, [0, 1, 2, 2, 0, 0])

    def test_opposite_bets_cancel(self):
        avg, C = BA.average_positions(6, [1, 2], [4, 4], [1.0, -1.0])
        np.testing.assert_allclose(avg, [0, 1, 0, 0, 0, 0], atol=1e-15)
        self.assertEqual(C[2], 2)                                   # still two active events

    def test_zero_size_event_counts_as_a_zero(self):
        avg, _ = BA.average_positions(6, [1, 2], [4, 4], [0.0, 1.0])
        np.testing.assert_allclose(avg, [0, 0, 0.5, 0.5, 0, 0], atol=1e-15)

    def test_event_ending_when_another_starts(self):
        avg, C = BA.average_positions(6, [1, 3], [3, 5], [1.0, -0.5])
        np.testing.assert_allclose(avg, [0, 1, 1, -0.5, -0.5, 0], atol=1e-15)
        np.testing.assert_array_equal(C, [0, 1, 1, 1, 1, 0])

    def test_matches_brute_force(self):
        rng = np.random.default_rng(5)
        n = 300
        pos = rng.integers(0, n - 2, 80)
        t1 = np.minimum(pos + rng.integers(1, 60, 80), n - 1)
        sig = rng.choice([-1.0, 1.0], 80) * rng.choice([0.0, 0.1, 0.5, 1.0], 80)
        avg, C = BA.average_positions(n, pos, t1, sig)
        for k in range(n):
            act = [i for i in range(80) if pos[i] <= k < t1[i]]
            self.assertEqual(C[k], len(act))
            self.assertAlmostEqual(avg[k], (sum(sig[i] for i in act) / len(act)) if act else 0.0, places=12)

    def test_bad_events_raise(self):
        for pos, t1 in (([2], [2]), ([3], [2]), ([-1], [2]), ([1], [9])):
            with self.assertRaises(ValueError):
                BA.average_positions(5, pos, t1, [1.0])


class TestDaily(unittest.TestCase):
    def test_isolated_one_bar_bet_equals_per_event_rule(self):
        close = np.array([100.0, 110.0, 99.0, 99.0])
        out = BA.symbol_daily(close, hourly(4), [1], [2], [-0.4], CS)          # short, size .4, bar 1 -> 2: return 99/110 - 1
        ret = 99.0 / 110.0 - 1.0
        self.assertAlmostEqual(out["gross"].sum(), -0.4 * ret, places=14)
        self.assertAlmostEqual(out["cost"].sum(), 2 * CS * 0.4, places=14)
        per_event = 0.4 * ((-1.0) * ret - 2 * CS)
        self.assertAlmostEqual(out["gross"].sum() - out["cost"].sum(), per_event, places=14)

    def test_hand_worked_two_bar_bet(self):
        close = np.array([100.0, 110.0, 121.0, 121.0, 110.0])
        out = BA.symbol_daily(close, hourly(5), [1], [3], [1.0], CS)
        self.assertAlmostEqual(out["gross"].sum(), 0.1 + 0.0, places=12)                  # position 1 on k = 1, 2: bar 1->2 is +10%, bar 2->3 is 0%
        self.assertAlmostEqual(out["cost"].sum(), 2 * CS, places=14)

    def test_opposite_bets_pay_less_cost(self):
        close = np.linspace(100, 105, 8)
        both = BA.symbol_daily(close, hourly(8), [1, 2], [5, 5], [1.0, -1.0], CS)
        alone = BA.symbol_daily(close, hourly(8), [1], [5], [1.0], CS)
        # pair: positions 0,1,0,0,0,0 for k = 0..5 -> turnover 1 (k=1) + 1 (k=2) = 2 -> cost 2*CS; alone: 1 in, 1 out -> 2*CS (the second bet adds nothing)
        self.assertAlmostEqual(both["cost"].sum(), 2 * CS, places=14)
        self.assertAlmostEqual(alone["cost"].sum(), 2 * CS, places=14)
        per_event_cost = 2 * CS * (1.0 + 1.0)
        self.assertLess(both["cost"].sum(), per_event_cost)

    def test_day_booking_across_midnight(self):
        ts = np.array([22, 23, 24, 25, 26], dtype=np.int64) * 3600                          # days 0, 0, 1, 1, 1
        close = np.array([100.0, 100.0, 110.0, 110.0, 110.0])
        out = BA.symbol_daily(close, ts, [1], [3], [1.0], CS)
        np.testing.assert_array_equal(out["days"], [0, 1])
        self.assertAlmostEqual(out["gross"][0], 0.0, places=14)                             # bar 1 -> 2 (+10%) is booked on the day of bar 2 = day 1
        self.assertAlmostEqual(out["gross"][1], 0.1, places=12)
        self.assertAlmostEqual(out["cost"][0], CS, places=14)                               # entry at bar 1 (day 0)
        self.assertAlmostEqual(out["cost"][1], CS, places=14)                               # exit at bar 3 (day 1)
        np.testing.assert_array_equal(out["active"], [1, 1])

    def test_combine_and_daily_bp(self):
        a = BA.symbol_daily(np.array([100.0, 101.0, 101.0, 101.0]), hourly(4), [0], [1], [1.0], CS)
        b = BA.symbol_daily(np.array([100.0, 100.0, 100.0, 100.0]), hourly(4), [0], [1], [0.0], CS)       # active with a zero signal: counts in the denominator
        c = BA.combine_days([a, b])
        days, bp, n = BA.daily_bp(c)
        self.assertEqual(list(n), [2])
        self.assertAlmostEqual(bp[0], 1e4 * (0.01 - 2 * CS) / 2, places=9)          # a: +1% gross, entry and exit cost; b: zero position, zero cost

    def test_days_without_active_instrument_are_dropped(self):
        ts = np.array([0, 1, 2, 24, 25, 26], dtype=np.int64) * 3600                         # days 0 and 1
        close = np.array([100.0, 101.0, 102.0, 102.0, 102.0, 102.0])
        out = BA.symbol_daily(close, ts, [0], [2], [1.0], CS)
        days, bp, n = BA.daily_bp(BA.combine_days([out]))
        self.assertEqual(list(days), [0])

    def test_window_bp(self):
        ts = np.array([0, 1, 24, 25, 48, 49], dtype=np.int64) * 3600                        # days 0, 0, 1, 1, 2, 2
        close = np.array([100.0, 101.0, 101.0, 102.01, 102.01, 102.01])
        out = BA.symbol_daily(close, ts, [0, 2], [1, 3], [1.0, 1.0], CS)
        comb = BA.combine_days([out])
        b0, n0 = BA.window_bp(comb, 0, 86400)
        self.assertEqual(n0, 1)
        b_all, n_all = BA.window_bp(comb, 0, 3 * 86400)
        d, bp, n = BA.daily_bp(comb)
        self.assertEqual(n_all, int(n.sum()))
        self.assertAlmostEqual(b_all * n_all, float((bp * n).sum()), places=9)
        b_none, n_none = BA.window_bp(comb, 10 * 86400, 11 * 86400)
        self.assertEqual(n_none, 0)
        self.assertTrue(np.isnan(b_none))

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            BA.symbol_daily(np.array([100.0]), hourly(1), [], [], [], CS)


@unittest.skipUnless(os.environ.get("BA_TEST_RUN"), "set BA_TEST_RUN to a run folder to run the pipeline-level test")
class TestPipeline(unittest.TestCase):
    def test_runs_and_reproduces_per_event_numbers(self):
        import run_bet_averaging as RBA
        with tempfile.TemporaryDirectory() as tmp:
            out = RBA.run_one(Path(os.environ["BA_TEST_RUN"]), Path(tmp) / "o", None)
            for f in ("averaged_paths.csv", "averaged_daily.csv", "averaged_windows.csv", "averaged_summary.json", "averaged_summary.md", "averaged_extra.json"):
                self.assertTrue((out / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
