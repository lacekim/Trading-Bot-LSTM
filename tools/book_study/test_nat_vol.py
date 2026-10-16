"""Tests for nat_vol.py: Appendix B volatility equals the book's Figure B-2 numbers and nat_core's formula; hole handling; daily target."""
import unittest
import numpy as np
import nat_core as C
import nat_vol as NV

PRICES = [97.70, 99.50, 92.75, 95.85, 96.20, 102.45, 93.30, 91.15, 95.20, 102.80, 103.85]


class T(unittest.TestCase):
    def test_figure_b2_estimated_sd_zero_mean(self):
        x = np.diff(np.log(PRICES))
        v = NV.appendix_b_vol(x, len(x))[-1] / np.sqrt(7 / 365)
        self.assertAlmostEqual(v * 100, 39.93, delta=0.01)       # book: 39.93% (the 'estimated standard deviation, 0 mean' line)
        self.assertAlmostEqual(v, C.vol_appendix_b(PRICES, 7 / 365, mean="zero", sample=True), places=12)

    def test_rolling_window_and_nan_skip(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.01, 500)
        x[100:110] = np.nan
        out = NV.appendix_b_vol(x, 50)
        self.assertTrue(np.isnan(out[:49]).all())
        v = x[~np.isnan(x)]
        # at position 300 the last 50 valid returns are those with valid index 190..239 -> direct computation
        pos = 300
        valid_before = x[: pos + 1][~np.isnan(x[: pos + 1])][-50:]
        self.assertAlmostEqual(out[pos], np.sqrt(np.sum(valid_before ** 2) / 49), places=12)
        self.assertEqual(out[105], out[99])                          # a missing return carries the last value

    def test_daily_vol_shape(self):
        t = np.arange(0, 400 * 3600 * 6, 3600 * 6, dtype=np.int64)    # one bar per 6 hours
        close = 100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, len(t))))
        out = NV.daily_vol(t, close, 100)
        self.assertEqual(len(out), len(t))
        self.assertTrue(np.isfinite(out[-1]) and out[-1] > 0)


if __name__ == "__main__":
    unittest.main()
