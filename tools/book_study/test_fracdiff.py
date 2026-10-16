import unittest
from math import comb

import numpy as np

import fracdiff_core as F


def literal_weights(d, thres):
    """Straight copy of the book's loop."""
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])


class TestFracDiff(unittest.TestCase):
    def test_weights_match_the_book_loop(self):
        for d in (0.0, 0.1, 0.35, 0.5, 0.9, 1.0, 1.3):
            for th in (1e-2, 1e-5):
                np.testing.assert_allclose(F.weights_ffd(d, th), literal_weights(d, th), rtol=0, atol=0)

    def test_first_weights_are_binomial(self):
        w = F.weights_ffd(0.4, 1e-12)[::-1]            # newest first
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], -0.4)
        self.assertAlmostEqual(w[2], 0.4 * (0.4 - 1) / 2)       # d(d-1)/2 = -0.12

    def test_d1_is_the_first_difference_and_d0_is_identity(self):
        x = np.cumsum(np.random.default_rng(1).standard_normal(500))
        f1 = F.frac_diff_ffd(x, 1.0, 1e-5)
        np.testing.assert_allclose(f1[1:], np.diff(x), atol=1e-12)
        f0 = F.frac_diff_ffd(x, 0.0, 1e-5)
        np.testing.assert_allclose(f0, x, atol=1e-12)

    def test_matches_a_literal_loop(self):
        x = np.cumsum(np.random.default_rng(2).standard_normal(400))
        d, th = 0.45, 1e-3
        w = literal_weights(d, th)
        width = len(w) - 1
        want = np.array([np.dot(w, x[i - width:i + 1]) for i in range(width, len(x))])
        got = F.frac_diff_ffd(x, d, th)
        np.testing.assert_allclose(got[width:], want, atol=1e-12)
        self.assertTrue(np.isnan(got[:width]).all())

    def test_adf_separates_a_random_walk_from_a_stationary_series(self):
        rng = np.random.default_rng(3)
        e = rng.standard_normal(5000)
        rw = np.cumsum(e)
        ar = np.zeros(5000)
        for t in range(1, 5000):
            ar[t] = 0.5 * ar[t - 1] + e[t]
        s_rw, n = F.adf_stat(rw)
        s_ar, _ = F.adf_stat(ar)
        self.assertGreater(s_rw, F.adf_crit5(n))        # cannot reject a unit root
        self.assertLess(s_ar, F.adf_crit5(n) - 10)      # strongly stationary
        self.assertAlmostEqual(F.adf_crit5(10 ** 6), -2.8615, places=3)

    def test_adf_matches_an_explicit_normal_equations_calculation(self):
        rng = np.random.default_rng(4)
        y = np.cumsum(rng.standard_normal(300)) * 0.3 + rng.standard_normal(300)
        stat, n = F.adf_stat(y, lags=1)
        dy = np.diff(y)
        rows = [(1.0, y[t], dy[t - 1]) for t in range(1, len(dy))]       # target dy[t], regressors a, y[t], dy[t-1]
        X = np.array(rows)
        z = dy[1:]
        XtX = X.T @ X
        b = np.linalg.solve(XtX, X.T @ z)
        r = z - X @ b
        s2 = r @ r / (len(z) - 3)
        t = b[1] / np.sqrt(s2 * np.linalg.inv(XtX)[1, 1])
        self.assertEqual(n, len(z))
        self.assertAlmostEqual(stat, t, places=8)

    def test_min_d_is_small_for_a_trend_stationary_like_series_and_one_for_a_pure_random_walk(self):
        rng = np.random.default_rng(5)
        rw = np.cumsum(rng.standard_normal(3000))
        r = F.min_d(rw)
        self.assertGreaterEqual(r["d"], 0.2)           # a pure random walk is not stationary at d=0 and needs real differencing
        self.assertLessEqual(r["d"], 1.0)
        t0 = r["table"][0]
        self.assertFalse(t0[1] < t0[2])                # d = 0 fails the test
        ar = np.zeros(3000)
        e = rng.standard_normal(3000)
        for t in range(1, 3000):
            ar[t] = 0.9 * ar[t - 1] + e[t]
        r2 = F.min_d(ar)
        self.assertLessEqual(r2["d"], 0.1)             # already stationary
        d_prev = [t for t in r["table"] if t[0] < r["d"]]
        self.assertTrue(all(not (np.isfinite(t[1]) and t[1] < t[2]) for t in d_prev))   # nothing smaller passed


if __name__ == "__main__":
    unittest.main()
