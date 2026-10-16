"""Unit tests for book_stats.py.  Run:  python -m unittest tools/book_study/test_book_stats.py -v"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import book_stats as B  # noqa: E402


def literal_snippet_13_2(forecast, half_life_bars, eps, max_hp, pt, sl):
    """Line-by-line port of Snippet 13.2's inner loop (Python 3), with the Gaussian draws taken from eps rows."""
    phi = 2 ** (-1.0 / half_life_bars)
    out = []
    for k in range(eps.shape[0]):
        p, hp = 0.0, 0
        while True:
            p = (1 - phi) * forecast + phi * p + eps[k, hp]
            c_p = p - 0.0
            hp += 1
            if c_p > pt or c_p < -sl or hp > max_hp:
                out.append(c_p)
                break
    return np.array(out)


class TestOU(unittest.TestCase):
    def test_vectorised_mesh_matches_literal_snippet(self):
        rng = np.random.default_rng(1)
        max_hp = 20
        eps = rng.standard_normal((400, max_hp + 1))
        forecast, hl = 3.0, 10
        phi = 2 ** (-1.0 / hl)
        paths = B.simulate_ou_paths((1 - phi) * forecast, phi, eps)
        pts, sls = (0.5, 1.0, 2.5, 6.0), (0.5, 3.0, 10.0)
        stats = B.mesh_exit_stats(paths, pts, sls)
        for i, pt in enumerate(pts):
            for j, sl in enumerate(sls):
                lit = literal_snippet_13_2(forecast, hl, eps, max_hp, pt, sl)
                self.assertAlmostEqual(stats["mean"][i, j], lit.mean(), places=10)
                self.assertAlmostEqual(stats["std"][i, j], lit.std(), places=10)
                value, step, kind = B.first_exit(paths, pt, sl)
                np.testing.assert_allclose(value, lit, atol=1e-12)

    def test_exit_kinds_and_steps(self):
        paths = np.array([[0.2, 1.3, 0.0], [-0.4, -1.5, 0.0], [0.1, 0.2, 0.3]])
        value, step, kind = B.first_exit(paths, 1.0, 1.0)
        self.assertEqual(list(kind), [1, -1, 0])
        self.assertEqual(list(step), [2, 2, 3])
        np.testing.assert_allclose(value, [1.3, -1.5, 0.3])

    def test_mesh_pt_share(self):
        paths = np.array([[1.5, 0], [-1.5, 0], [2.0, 0], [0.0, 0.1]])
        st = B.mesh_exit_stats(paths, (1.0,), (1.0,))
        self.assertAlmostEqual(st["pt_share_of_barrier_exits"][0, 0], 2 / 3)

    def test_cost_shifts_mean_only(self):
        rng = np.random.default_rng(2)
        paths = B.simulate_ou_paths(0.0, 0.95, rng.standard_normal((500, 31)))
        a = B.mesh_exit_stats(paths, (1.0, 2.0), (1.0, 2.0))
        b = B.mesh_exit_stats(paths, (1.0, 2.0), (1.0, 2.0), cost=np.full(500, 0.4))
        np.testing.assert_allclose(a["mean"] - 0.4, b["mean_net"], atol=1e-12)
        np.testing.assert_allclose(a["std"], b["std_net"], atol=1e-12)
        np.testing.assert_allclose(a["mean"], b["mean"], atol=1e-12)
        self.assertNotIn("mean_net", a)

    def test_estimate_recovers_parameters(self):
        rng = np.random.default_rng(3)
        paths = B.simulate_ou_paths(0.02, 0.9, rng.standard_normal((20000, 60)))
        est = B.estimate_ou(paths)
        self.assertAlmostEqual(est["phi"], 0.9, delta=0.01)
        self.assertAlmostEqual(est["a"], 0.02, delta=0.02)
        self.assertAlmostEqual(est["sigma"], 1.0, delta=0.02)

    def test_estimate_chunking_is_exact(self):
        rng = np.random.default_rng(10)
        paths = B.simulate_ou_paths(0.01, 0.8, rng.standard_normal((1000, 30)))
        w = rng.uniform(0.1, 1.0, 1000)
        a, b = B.estimate_ou(paths, w, chunk=1000), B.estimate_ou(paths, w, chunk=77)
        for k in ("phi", "a", "sigma"):
            self.assertAlmostEqual(a[k], b[k], places=12)

    def test_estimate_weights(self):
        rng = np.random.default_rng(4)
        a = B.simulate_ou_paths(0.0, 0.95, rng.standard_normal((3000, 40)))
        b = B.simulate_ou_paths(0.0, 0.50, rng.standard_normal((3000, 40)))
        both = np.vstack([a, b])
        w_a = np.concatenate([np.ones(3000), np.zeros(3000)])
        self.assertAlmostEqual(B.estimate_ou(both, w_a)["phi"], B.estimate_ou(a)["phi"], places=10)

    def test_random_walk_has_no_optimal_rule(self):
        rng = np.random.default_rng(5)
        paths = B.simulate_ou_paths(0.0, 1.0, rng.standard_normal((40000, 51)))
        st = B.mesh_exit_stats(paths)
        self.assertLess(float(np.nanmax(np.abs(st["sharpe"]))), 0.03)   # martingale: no rule has expected profit

    def test_mean_reverting_process_prefers_small_target_wide_stop(self):
        rng = np.random.default_rng(6)
        phi = 2 ** (-1 / 5)
        paths = B.simulate_ou_paths(0.0, phi, rng.standard_normal((40000, 101)))
        st = B.mesh_exit_stats(paths)
        pt, sl, sr = B.best_node(st["sharpe"])
        self.assertGreater(sr, 0.5)
        self.assertLessEqual(pt, 3.0)
        self.assertGreaterEqual(sl, 5.0)

    def test_half_life(self):
        self.assertAlmostEqual(B.half_life(2 ** (-1 / 7)), 7.0)
        self.assertTrue(math.isinf(B.half_life(1.0)))


class TestSampling(unittest.TestCase):
    def test_cusum_matches_literal_snippet(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0, 0.01, 2000)
        h = 0.03
        s_pos = s_neg = 0.0
        lit = []
        for i in range(1, len(r)):
            s_pos, s_neg = max(0.0, s_pos + r[i]), min(0.0, s_neg + r[i])
            if s_neg < -h:
                s_neg = 0.0
                lit.append(i)
            elif s_pos > h:
                s_pos = 0.0
                lit.append(i)
        np.testing.assert_array_equal(B.cusum_events(r, h), np.array(lit))
        self.assertGreater(len(lit), 20)

    def test_cusum_time_varying_threshold_and_nan(self):
        r = np.array([0.0, 0.02, 0.02, np.nan, -0.05, 0.0])
        h = np.array([np.nan, 0.03, 0.03, 0.03, 0.03, 0.03])
        ev = B.cusum_events(r, h)
        self.assertEqual(list(ev), [2, 4])

    def test_average_uniqueness_matches_book_example(self):
        # book's Matrix 1 / Snippet 4.6: events [0,2], [2,3], [4,5]
        u = B.average_uniqueness(np.array([0, 2, 4]), np.array([2, 3, 5]), 6)
        np.testing.assert_allclose(u, [(1 + 1 + 0.5) / 3, (0.5 + 1) / 2, 1.0])

    def test_average_uniqueness_matches_literal_loop(self):
        rng = np.random.default_rng(8)
        n_bars = 300
        t0 = np.sort(rng.integers(0, n_bars - 12, 60))
        t1 = t0 + rng.integers(1, 12, 60)
        conc = np.zeros(n_bars)
        for a, b in zip(t0, t1):
            conc[a:b + 1] += 1
        lit = np.array([(1.0 / conc[a:b + 1]).mean() for a, b in zip(t0, t1)])
        np.testing.assert_allclose(B.average_uniqueness(t0, t1, n_bars), lit)

    def test_average_uniqueness_clips_at_series_end(self):
        u = B.average_uniqueness(np.array([8]), np.array([20]), 10)   # runs past the data: only bars 8, 9 exist
        self.assertTrue(np.isfinite(u).all())


class TestCPCV(unittest.TestCase):
    def test_book_example_6_2(self):
        splits = B.cpcv_splits(6, 2)
        self.assertEqual(len(splits), 15)
        assign, n_paths = B.cpcv_paths(6, 2)
        self.assertEqual(n_paths, 5)       # Figure 12.1: phi[6,2] = 5

    def test_8_2(self):
        assign, n_paths = B.cpcv_paths(8, 2)
        self.assertEqual(n_paths, 7)
        self.assertEqual(len(B.cpcv_splits(8, 2)), 28)
        for path in range(n_paths):
            groups = sorted(g for (s, g), p in assign.items() if p == path)
            self.assertEqual(groups, list(range(8)))   # each path covers every group exactly once

    def test_equation_40(self):
        for n, k in [(6, 2), (10, 3), (8, 4)]:
            _, n_paths = B.cpcv_paths(n, k)
            self.assertEqual(n_paths, round(k / n * math.comb(n, n - k)))

    def test_groups_and_purge_embargo(self):
        edges = B.group_edges(0.0, 80.0, 8)           # groups of width 10
        t0 = np.array([5.0, 25.0, 28.0, 31.0, 35.0, 41.0, 60.0, 79.0])
        t1 = t0 + 6.0
        groups = B.assign_groups(t0, edges)
        self.assertEqual(list(groups), [0, 2, 2, 3, 3, 4, 6, 7])
        keep = B.train_mask(t0, t1, groups, (3,), edges, embargo=3.0)   # test span [30, 40]
        # 28 ends at 34 -> overlaps the test span (purged); 25 ends at 31 -> overlaps (purged);
        # 31 and 35 are in the test group; 41 is within the 3-unit embargo after 40 (purged); 60, 79, 5 stay
        self.assertEqual(list(keep), [True, False, False, False, False, False, True, True])
        keep2 = B.train_mask(t0, t1, groups, (3,), edges, embargo=0.0)
        self.assertTrue(keep2[5])


class TestStatistics(unittest.TestCase):
    def test_psr_hand_value(self):
        sr, t = 0.1, 250
        num = sr * math.sqrt(t - 1)
        den = math.sqrt(1 + (3 - 1) / 4 * sr * sr)
        expected = 0.5 * (1 + math.erf(num / den / math.sqrt(2)))
        self.assertAlmostEqual(B.psr(sr, t, 0.0, 3.0), expected, places=12)

    def test_psr_direction(self):
        base = B.psr(0.1, 250, 0.0, 3.0)
        self.assertGreater(B.psr(0.1, 500, 0.0, 3.0), base)          # longer record
        self.assertLess(B.psr(0.1, 250, -1.0, 3.0), base)            # negative skew hurts
        self.assertLess(B.psr(0.1, 250, 0.0, 9.0), base)             # fat tails hurt
        self.assertLess(B.psr(0.1, 250, 0.0, 3.0, sr_star=0.05), base)

    def test_expected_max_sharpe(self):
        self.assertAlmostEqual(B.expected_max_sharpe(120, 1.0), 2.592, delta=0.01)
        self.assertAlmostEqual(B.expected_max_sharpe(240, 1.0), 2.825, delta=0.02)
        self.assertLess(B.expected_max_sharpe(10, 0.25), B.expected_max_sharpe(100, 0.25))
        self.assertTrue(math.isnan(B.expected_max_sharpe(1, 1.0)))

    def test_dsr_below_psr(self):
        self.assertLess(B.dsr(0.1, 250, 0.0, 3.0, 20, 0.01), B.psr(0.1, 250, 0.0, 3.0))

    def test_sharpe_moments_normal(self):
        x = np.random.default_rng(9).normal(0.1, 1.0, 200000)
        m = B.sharpe_moments(x)
        self.assertAlmostEqual(m["sr"], 0.1, delta=0.01)
        self.assertAlmostEqual(m["skew"], 0.0, delta=0.03)
        self.assertAlmostEqual(m["kurt"], 3.0, delta=0.05)
        self.assertTrue(math.isnan(B.sharpe_moments([1.0, 2.0])["sr"]))

    def test_bin_hr_round_trip_and_break_even(self):
        for sl, pt, freq, tsr in [(-1.0, 2.0, 100, 1.5), (-0.003, 0.1, 52, 2.0)]:
            p = B.bin_hr(sl, pt, freq, tsr)
            self.assertAlmostEqual(B.bin_sr(sl, pt, freq, p), tsr, places=9)
        self.assertAlmostEqual(B.bin_hr(-1.0, 2.0, 100, 0.0), 1 / 3)
        self.assertAlmostEqual(B.break_even_precision(2.0, 1.0), 1 / 3)
        self.assertAlmostEqual(B.break_even_precision(2.0, 1.0, cost=0.3), 1.3 / 3)

    def test_bin_freq_round_trip(self):
        f = B.bin_freq(-1.0, 2.0, 0.55, 1.0)
        self.assertAlmostEqual(B.bin_sr(-1.0, 2.0, f, 0.55), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
