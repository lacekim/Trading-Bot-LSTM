"""Tests for volume_features.py and meta_events_vol.py.  Run from tools/book_study:  python -m unittest test_volume_features -v
The reference implementation below is written with plain loops and math only, straight from the definitions in volume_features.py."""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import volume_features as VF  # noqa: E402

W_REG, W_FLOW, W_Z = VF.W_REG, VF.W_FLOW, VF.W_Z


def ref_tick(c):
    b, cur = [0.0] * len(c), 0.0
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        if d > 0:
            cur = 1.0
        elif d < 0:
            cur = -1.0
        b[i] = cur
    return b


def ref_slope(x, y):
    n = len(x)
    xm, ym = sum(x) / n, sum(y) / n
    sxx = sum((a - xm) ** 2 for a in x)
    if sxx <= 0:
        return float("nan")
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sxx


def ref_at(c, v, t):
    """kyle, amihud, vpin, dvol_z at bar t, from the definitions."""
    n = len(c)
    r = [float("nan")] + [math.log(c[i] / c[i - 1]) for i in range(1, n)]
    dv = [c[i] * v[i] for i in range(n)]
    b = ref_tick(c)
    nan = float("nan")
    kyle = amihud = vpin = z = nan
    if t >= W_REG:
        idx = range(t - W_REG + 1, t + 1)
        mean_dv = sum(dv[i] for i in idx) / W_REG
        kyle = ref_slope([b[i] * dv[i] for i in idx], [r[i] for i in idx]) * mean_dv
        amihud = ref_slope([dv[i] for i in idx], [abs(r[i]) for i in idx]) * mean_dv
    if t >= W_REG + W_FLOW - 1:
        num = den = 0.0
        bad = False
        for i in range(t - W_FLOW + 1, t + 1):
            rs = [r[j] for j in range(i - W_REG + 1, i + 1)]
            m = sum(rs) / W_REG
            sd = math.sqrt(sum((q - m) ** 2 for q in rs) / W_REG)
            if not sd > 0:
                bad = True
                break
            num += v[i] * abs(2.0 * 0.5 * (1.0 + math.erf(r[i] / (sd * math.sqrt(2.0)))) - 1.0)
            den += v[i]
        vpin = nan if (bad or not den > 0) else num / den
    if t >= W_FLOW - 1 + W_Z - 1:
        ms = []
        for s in range(t - W_Z + 1, t + 1):
            ms.append(math.log1p(sum(dv[i] for i in range(s - W_FLOW + 1, s + 1)) / W_FLOW))
        mm = sum(ms) / W_Z
        sd = math.sqrt(sum((q - mm) ** 2 for q in ms) / W_Z)
        z = (ms[-1] - mm) / sd if sd > 0 else nan
    return kyle, amihud, vpin, z


def synth(n=1200, seed=3, zero_runs=True):
    rng = np.random.default_rng(seed)
    lr = rng.normal(0, 0.01, n)
    c = 100 * np.exp(np.cumsum(lr))
    if zero_runs:
        c[200:204] = c[200]                        # flat bars -> zero changes (tick rule repeats the sign)
    v = rng.lognormal(3.0, 0.7, n) * (1 + 0.5 * np.abs(np.diff(np.log(c), prepend=np.log(c[0]))) * 100)
    return c, v


class TestFeatures(unittest.TestCase):
    def test_tick_rule(self):
        c = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 1.5, 1.5, 3.0])
        self.assertEqual(list(VF.tick_sign(c)), [0, 0, 1, 1, 1, -1, -1, 1])
        self.assertEqual(list(VF.tick_sign(c)), ref_tick(list(c)))

    def test_matches_reference_loops(self):
        c, v = synth()
        f = VF.volume_features(c, v)
        # probes cover: every feature's first defined bar, a run of flat bars, and bars deep in the series
        probes = [W_REG, W_REG + 1, W_REG + W_FLOW - 1, 205, 330, 500, W_FLOW + W_Z - 2, W_FLOW + W_Z - 1, 900, len(c) - 1]
        for t in probes:
            ref = ref_at(list(c), list(v), t)
            got = (f["kyle_lam"][t], f["amihud_lam"][t], f["vpin"][t], f["dvol_z"][t])
            for name, a, b in zip(VF.VOL_FEATURES, got, ref):
                if math.isnan(b):
                    self.assertTrue(math.isnan(a), f"{name} at {t}: expected NaN, got {a}")
                else:
                    self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(b)), msg=f"{name} at {t}")

    def test_first_defined_bars(self):
        c, v = synth()
        f = VF.volume_features(c, v)
        first = {k: int(np.flatnonzero(np.isfinite(f[k]))[0]) for k in VF.VOL_FEATURES}
        self.assertEqual(first["kyle_lam"], W_REG)
        self.assertEqual(first["amihud_lam"], W_REG)
        self.assertEqual(first["vpin"], W_REG + W_FLOW - 1)
        self.assertEqual(first["dvol_z"], W_FLOW - 1 + W_Z - 1)

    def test_causal(self):
        c, v = synth()
        full = VF.volume_features(c, v)
        cut = 777
        part = VF.volume_features(c[:cut], v[:cut])
        for k in VF.VOL_FEATURES:
            a, b = full[k][:cut], part[k]
            both = np.isfinite(a) & np.isfinite(b)
            self.assertTrue(np.array_equal(np.isfinite(a), np.isfinite(b)), k)
            self.assertTrue(np.allclose(a[both], b[both], rtol=1e-12, atol=0), k)

    def test_missing_volume_only_hits_its_windows(self):
        c, v = synth()
        base = VF.volume_features(c, v)
        v2 = v.copy()
        v2[600] = np.nan
        f = VF.volume_features(c, v2)
        for k, reach in (("kyle_lam", W_REG), ("amihud_lam", W_REG), ("vpin", W_FLOW), ("dvol_z", W_FLOW - 1 + W_Z - 1)):
            self.assertTrue(np.all(np.isnan(f[k][600:600 + reach])), k)
            self.assertTrue(np.array_equal(f[k][:600], base[k][:600], equal_nan=True), k)
            self.assertTrue(np.array_equal(f[k][600 + reach:], base[k][600 + reach:], equal_nan=True), k)

    def test_zero_volume_windows_are_nan_not_inf(self):
        c, v = synth()
        v[400:400 + W_FLOW] = 0.0
        f = VF.volume_features(c, v)
        self.assertTrue(math.isnan(f["vpin"][400 + W_FLOW - 1]))          # window of zero volume: zero denominator
        for k in VF.VOL_FEATURES:
            self.assertFalse(np.isinf(f[k]).any(), k)

    def test_vpin_in_unit_interval(self):
        c, v = synth()
        x = VF.volume_features(c, v)["vpin"]
        x = x[np.isfinite(x)]
        self.assertTrue(((x >= 0) & (x <= 1)).all())

    def test_exact_line_recovers_slope(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 400)
        y = 0.37 * x + 5.0
        s, m = VF._ols_slope(x, y, 100)
        self.assertTrue(np.allclose(s[99:], 0.37, rtol=0, atol=1e-12))
        self.assertTrue(np.allclose(m[99], x[:100].mean()))
        self.assertTrue(np.isnan(s[:99]).all())

    def test_constant_regressor_is_nan(self):
        s, _ = VF._ols_slope(np.ones(300), np.arange(300.0), 100)
        self.assertTrue(np.isnan(s).all())

    def test_too_short(self):
        f = VF.volume_features(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        for k in VF.VOL_FEATURES:
            self.assertTrue(np.isnan(f[k]).all())


@unittest.skipUnless(os.environ.get("VOLUME_TEST_CANDLES"), "set VOLUME_TEST_CANDLES to a folder of <SYMBOL>_1h.csv files to run the pipeline-level test")
class TestPipelineWrapper(unittest.TestCase):
    def test_base_columns_and_weights_equal_plain_pipeline_when_nothing_is_dropped(self):
        import meta_events as E
        import meta_events_vol as EV
        import venue_loader as VL

        d = Path(os.environ["VOLUME_TEST_CANDLES"])
        names = [n for n in VL.list_symbols(d) if n in ("ETH", "XRP", "XLM", "LINK", "ADA", "DOT", "ATOM", "TRX", "SOL", "AVAX", "XMR", "ALGO")]
        closes, _ = VL.load_closes(d, names, "1h", 4380)
        ohlc, _ = VL.load_ohlc(d, sorted(closes), "1h", 0)
        vols = {s: VF.load_volumes(d, s) for s in closes}
        plain, info0 = E.build_meta_events(closes, ohlc, log=lambda m: None)
        vol, info1 = EV.build_meta_events_vol(closes, ohlc, vols, log=lambda m: None)
        self.assertEqual(E.FEATURES, EV.BASE_FEATURES)                 # the global list was restored
        self.assertEqual(info1["events_dropped_missing_volume_feature"], 0)
        self.assertEqual(len(plain), len(vol))
        for col in plain.columns:
            a, b = plain[col].to_numpy(), vol[col].to_numpy()
            if a.dtype.kind in "fc":
                self.assertTrue(np.array_equal(a, b, equal_nan=True), col)
            else:
                self.assertTrue((a == b).all(), col)
        for k in VF.VOL_FEATURES:
            self.assertTrue(np.isfinite(vol[k].to_numpy()).all(), k)

    def test_dropping_recomputes_weights_on_remaining_events(self):
        import meta_events_vol as EV
        import venue_loader as VL

        d = Path(os.environ["VOLUME_TEST_CANDLES"])
        names = [n for n in VL.list_symbols(d) if n in ("ETH", "XRP", "XLM", "LINK", "ADA", "DOT", "ATOM", "TRX", "SOL", "AVAX", "XMR", "ALGO")]
        closes, _ = VL.load_closes(d, names, "1h", 4380)
        ohlc, _ = VL.load_ohlc(d, sorted(closes), "1h", 0)
        vols = {s: VF.load_volumes(d, s) for s in closes}
        vols["ETH"] = vols["ETH"].copy()
        vols["ETH"].iloc[60000:60400] = np.nan                           # a block of missing volume in one symbol
        df, info = EV.build_meta_events_vol(closes, ohlc, vols, log=lambda m: None)
        self.assertGreater(info["events_dropped_missing_volume_feature"], 0)
        self.assertAlmostEqual(float(df["w"].mean()), 1.0, places=9)
        self.assertTrue(((df["u"] > 0) & (df["u"] <= 1 + 1e-12)).all())
        self.assertEqual(len(df), info["events_kept"])


if __name__ == "__main__":
    unittest.main()
