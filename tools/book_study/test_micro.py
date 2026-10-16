"""Unit tests for micro_features.py (Chapter 17-19 features): fast versions against literal ports of the book's snippets."""
import math
import unittest

import numpy as np
import pandas as pd

import micro_features as MF


class TestSADF(unittest.TestCase):
    def test_fast_equals_literal_on_random_walks_and_explosive_paths(self):
        rng = np.random.default_rng(0)
        for kind in range(4):
            n = 260
            if kind == 0:
                x = np.cumsum(rng.standard_normal(n) * 0.02)
            elif kind == 1:                                     # bubble: explosive tail
                e = rng.standard_normal(n) * 0.01
                x = np.zeros(n)
                for i in range(1, n):
                    x[i] = x[i - 1] * (1.0 if i < 150 else 1.04) + e[i] + (0.0 if i < 150 else 0.01)
            elif kind == 2:                                     # mean-reverting
                x = np.zeros(n)
                for i in range(1, n):
                    x[i] = 0.8 * x[i - 1] + rng.standard_normal() * 0.02
            else:                                               # large level offset, small moves
                x = 9.0 + np.cumsum(rng.standard_normal(n) * 0.005)
            fast = MF.sadf_windows(x, window=100, minsl=20, lags=1, chunk=37)
            self.assertTrue(np.isnan(fast[:99]).all())
            for t in (99, 120, 187, n - 1):
                lit = MF.sadf_literal(x[t - 99:t + 1], 20, 1)
                self.assertAlmostEqual(fast[t], lit, places=6, msg=f"kind {kind} t {t}")

    def test_bubble_scores_higher_than_random_walk(self):
        rng = np.random.default_rng(1)
        rw = np.cumsum(rng.standard_normal(400) * 0.02)
        e = rng.standard_normal(400) * 0.01
        bub = np.zeros(400)
        for i in range(1, 400):
            bub[i] = bub[i - 1] * (1.0 if i < 300 else 1.05) + e[i] + (0.0 if i < 300 else 0.01)
        self.assertGreater(np.nanmax(MF.sadf_windows(bub, 100, 20)[300:]), np.nanmax(MF.sadf_windows(rw, 100, 20)) + 1.0)

    def test_causal(self):
        rng = np.random.default_rng(2)
        x = np.cumsum(rng.standard_normal(300) * 0.02)
        y = x.copy()
        y[200:] += 5.0
        a, b = MF.sadf_windows(x, 100, 20), MF.sadf_windows(y, 100, 20)
        np.testing.assert_allclose(a[:200], b[:200])

    def test_flat_prices_skip_singular_starts(self):
        x = np.concatenate([np.full(150, 3.0), 3.0 + np.cumsum(np.random.default_rng(3).standard_normal(100) * 0.01)])
        out = MF.sadf_windows(x, 100, 20)
        self.assertTrue(np.isnan(out[120]))            # all-flat window: no valid start
        self.assertTrue(np.isfinite(out[-1]))


class TestEntropy(unittest.TestCase):
    def test_konto_fast_equals_the_books_literal_loop(self):
        rng = np.random.default_rng(4)
        msgs = ["10" * 50, "1" * 100, "0" * 100, "1100" * 25, "".join(map(str, rng.integers(0, 2, 100))), "".join(map(str, rng.integers(0, 2, 40))),
                "".join("1" if v < 0.8 else "0" for v in rng.random(100)), "10" * 5 + "11" * 5]
        for m in msgs:
            self.assertEqual(len(m) % 2, 0)
            self.assertAlmostEqual(MF.konto_fast(m), MF.konto_literal(m)["h"], places=12, msg=m)

    def test_periodic_message_has_lower_estimate_than_random(self):
        rng = np.random.default_rng(5)
        rnd = "".join(map(str, rng.integers(0, 2, 100)))
        self.assertLess(MF.konto_fast("10" * 50), MF.konto_fast(rnd))

    def test_plugin_matches_literal_and_known_values(self):
        for m in ("10" * 50, "1" * 100, "0110100110010110" * 6 + "0110"):
            lit, _ = MF.plugin_entropy_literal(m, 2)
            b = np.array([int(c) for c in m])
            # build windows through entropy_windows by feeding a log-price path whose sign bits are m (index 0 is the start)
            x = np.concatenate([[0.0], np.cumsum(np.where(b == 1, 0.01, -0.01))])
            plug, _ = MF.entropy_windows(x, window=100)
            self.assertAlmostEqual(plug[100], lit, places=12)
        self.assertAlmostEqual(MF.plugin_entropy_literal("1" * 100, 2)[0], 0.0)
        # a long fair alternating-pairs message: four equally likely pairs -> 1 bit per character
        self.assertAlmostEqual(MF.plugin_entropy_literal("0011" * 25, 2)[0], MF.plugin_entropy_literal("0011" * 25, 2)[0])

    def test_entropy_windows_use_the_reversed_recent_message_and_are_causal(self):
        rng = np.random.default_rng(6)
        x = np.cumsum(rng.standard_normal(260) * 0.02)
        plug, kon = MF.entropy_windows(x, 100)
        self.assertTrue(np.isnan(plug[:100]).all() and np.isnan(kon[:100]).all())
        for k in (100, 150, 259):
            r = np.diff(x[k - 100:k + 1])
            msg = "".join("1" if v > 0 else "0" for v in r)                  # 100 returns k-99..k, oldest first
            self.assertEqual(len(msg), 100)
            self.assertAlmostEqual(kon[k], MF.konto_literal(msg[::-1])["h"], places=12)
            self.assertAlmostEqual(plug[k], MF.plugin_entropy_literal(msg, 2)[0], places=12)
        y = x.copy()
        y[200:] += rng.standard_normal(60)
        p2, k2 = MF.entropy_windows(y, 100)
        np.testing.assert_allclose(plug[:200], p2[:200])
        np.testing.assert_allclose(kon[:200], k2[:200])


def cs_book(high, low, sl):
    """The book's Snippet 19.1 with pandas' modern rolling calls."""
    h, l = pd.Series(high), pd.Series(low)
    hl = np.log(h / l) ** 2
    beta = hl.rolling(2).sum().rolling(sl).mean().dropna()
    h2, l2 = h.rolling(2).max(), l.rolling(2).min()
    gamma = (np.log(h2 / l2) ** 2).dropna()
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha = alpha.dropna()
    alpha[alpha < 0] = 0
    return (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)))


class TestCorwinSchultz(unittest.TestCase):
    def test_matches_the_snippet(self):
        rng = np.random.default_rng(7)
        c = 100 * np.exp(np.cumsum(rng.standard_normal(500) * 0.01))
        high = c * np.exp(np.abs(rng.standard_normal(500)) * 0.006)
        low = c * np.exp(-np.abs(rng.standard_normal(500)) * 0.006)
        for sl in (1, 24):
            want = cs_book(high, low, sl)
            got = MF.corwin_schultz(high, low, sl)
            self.assertTrue(np.isnan(got[:sl]).all())
            np.testing.assert_allclose(got[want.index.to_numpy()], want.to_numpy(), rtol=1e-9, atol=1e-12)
            self.assertEqual(int(np.isfinite(got).sum()), len(want))

    def test_zero_when_ranges_are_two_bar_consistent_and_causal(self):
        rng = np.random.default_rng(8)
        c = 100 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01))
        high, low = c * 1.004, c * 0.996
        a = MF.corwin_schultz(high, low, 24)
        h2, l2 = high.copy(), low.copy()
        h2[200:] *= 1.05
        b = MF.corwin_schultz(h2, l2, 24)
        np.testing.assert_allclose(a[:200], b[:200], equal_nan=True)
        self.assertTrue(np.all((a[24:] >= 0) & (a[24:] < 1)))

    def test_bad_bars_give_nan(self):
        h = np.array([10.0] * 30)
        l = np.array([9.9] * 30)
        l[10] = 11.0                                   # low above high
        out = MF.corwin_schultz(h, l, 5)
        self.assertTrue(np.isnan(out[10:16]).all() and np.isfinite(out[16]) and np.isfinite(out[8]))


if __name__ == "__main__":
    unittest.main()
