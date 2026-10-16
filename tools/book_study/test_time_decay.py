"""Tests for time_decay.py and the weight wrapper of run_meta_decay.py.   python -m unittest test_time_decay -v

The pipeline-level test (control vs c = 1 vs c = 0.5 on real candles, quick settings, about 2 minutes) runs only when DECAY_TEST_CANDLES names a folder of <SYMBOL>_1h.csv files.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import time_decay as TD  # noqa: E402


class TestFactors(unittest.TestCase):
    def test_equal_uniqueness_linear(self):
        f = TD.decay_factors([1, 1, 1, 1], [10, 20, 30, 40], 0.5)
        np.testing.assert_allclose(f, [0.625, 0.75, 0.875, 1.0], atol=1e-15)

    def test_c_zero(self):
        np.testing.assert_allclose(TD.decay_factors([1, 1, 1, 1], [1, 2, 3, 4], 0.0), [0.25, 0.5, 0.75, 1.0], atol=1e-15)

    def test_unequal_uniqueness_uses_cumulative_sum(self):
        # sorted by time: u = .5, 1, .25, .25 -> cumulative .5, 1.5, 1.75, 2.0 ; c = .2 -> .2 + .8 * cum / 2
        f = TD.decay_factors([0.5, 1.0, 0.25, 0.25], [1, 2, 3, 4], 0.2)
        np.testing.assert_allclose(f, [0.4, 0.8, 0.9, 1.0], atol=1e-15)

    def test_c_one_is_exactly_one(self):
        f = TD.decay_factors(np.random.default_rng(1).uniform(0.1, 1, 50), np.arange(50.0), 1.0)
        self.assertTrue(np.array_equal(f, np.ones(50)))

    def test_output_is_aligned_with_input_order_not_time_order(self):
        u = np.array([1.0, 1.0, 1.0, 1.0])
        t = np.array([40, 10, 30, 20])                       # time order: index 1, 3, 2, 0
        f = TD.decay_factors(u, t, 0.5)
        np.testing.assert_allclose(f, [1.0, 0.625, 0.875, 0.75], atol=1e-15)

    def test_ties_are_ordered_by_position(self):
        f = TD.decay_factors([1, 1, 1], [5, 5, 5], 0.0)
        np.testing.assert_allclose(f, [1 / 3, 2 / 3, 1.0], atol=1e-15)

    def test_newest_is_one_and_monotone_in_time(self):
        rng = np.random.default_rng(2)
        u, t = rng.uniform(0.05, 1, 200), rng.permutation(200).astype(float)
        for c in (0.9, 0.5, 0.0, -0.3, -0.9):
            f = TD.decay_factors(u, t, c)
            o = np.argsort(t)
            self.assertAlmostEqual(f[o][-1], 1.0, places=14)
            self.assertTrue((np.diff(f[o]) >= -1e-15).all())
            self.assertTrue(((f >= 0) & (f <= 1 + 1e-15)).all())
            if c >= 0:
                self.assertTrue((f >= c - 1e-15).all())

    def test_negative_c_zeroes_the_oldest_share(self):
        # u = 1 each, n = 4, c = -.5: factor = 1 - (4 - cum) / 2 -> -.5, 0, .5, 1 -> cut at 0
        f = TD.decay_factors([1, 1, 1, 1], [1, 2, 3, 4], -0.5)
        np.testing.assert_allclose(f, [0.0, 0.0, 0.5, 1.0], atol=1e-15)

    def test_scale_of_uniqueness_does_not_matter(self):
        rng = np.random.default_rng(3)
        u, t = rng.uniform(0.1, 1, 30), np.arange(30.0)
        np.testing.assert_allclose(TD.decay_factors(u, t, 0.3), TD.decay_factors(7.5 * u, t, 0.3), atol=1e-13)

    def test_bad_inputs_raise(self):
        for args in (([1, 1], [1, 2], -1.0), ([1, 1], [1, 2], 1.5), ([1, 0], [1, 2], 0.5), ([1, 1], [1], 0.5), ([], [], 0.5),
                     ([1, float("nan")], [1, 2], 0.5), ([1, 1], [1, float("inf")], 0.5)):
            with self.assertRaises(ValueError, msg=str(args)):
                TD.decay_factors(*args)

    def test_tree_learner_uses_only_weight_ratios_exactly_under_power_of_two_scaling(self):
        # the learner's splits depend on the weights only through their ratios.  Scaling by a power of two is exact in floating point, so the trees are identical;
        # scaling by other factors (0.37, 1/3) was observed to change a few near-tied splits through rounding, so no equality is asserted for those.
        from sklearn.tree import DecisionTreeClassifier
        rng = np.random.default_rng(4)
        X = rng.normal(size=(400, 6))
        y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(int)
        w = rng.uniform(0.2, 2, 400)
        outs = []
        for k in (1.0, 0.5, 8.0, 0.125):
            t = DecisionTreeClassifier(criterion="entropy", max_features="sqrt", class_weight="balanced", min_weight_fraction_leaf=0.05, random_state=7)
            t.fit(X, y, sample_weight=k * w)
            outs.append(t.predict_proba(X))
        for o in outs[1:]:
            np.testing.assert_array_equal(outs[0], o)


class TestWrapper(unittest.TestCase):
    def test_weights_of_training_events_only_are_multiplied_and_the_call_is_recorded(self):
        import run_meta_decay as RD
        seen = {}

        def fake_fit(idx, data, forest_kw, seed):
            seen["idx"], seen["data"], seen["seed"], seen["kw"] = idx, data, seed, forest_kw
            return "forest"

        calls = []
        fit = RD.make_fit_wrapper(0.5, fake_fit, calls)
        n = 6
        X, y = np.zeros((n, 2)), np.zeros(n, dtype=int)
        w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        u = np.ones(n)
        t0 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        data = (X, y, w, np.zeros(n, int), np.arange(n), np.arange(n), t0, np.array([99.0, 98, 97, 96, 95, 94]), u)     # t1 deliberately in the opposite order of t0
        idx = np.array([5, 1, 3])                              # unsorted on purpose; time order of these: 1, 3, 5
        out = fit(idx, data, {"a": 1}, 99)
        self.assertEqual(out, "forest")
        self.assertEqual((seen["seed"], seen["kw"]), (99, {"a": 1}))
        d = TD.decay_factors(u[idx], t0[idx], 0.5)
        np.testing.assert_allclose(d, [1.0, 0.5 + 0.5 / 3, 0.5 + 0.5 * 2 / 3], atol=1e-15)   # event 5 newest = 1, event 1 oldest, event 3 middle
        np.testing.assert_allclose(seen["data"][2][idx], w[idx] * d)
        rest = np.setdiff1d(np.arange(n), idx)
        np.testing.assert_array_equal(seen["data"][2][rest], w[rest])                         # events outside the training set untouched
        np.testing.assert_array_equal(w, [1, 2, 3, 4, 5, 6])                                  # the caller's weights are not modified in place
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 0)
        np.testing.assert_array_equal(calls[0][1], idx)
        np.testing.assert_allclose(calls[0][2], d)
        fit(idx, data, {}, 100)
        self.assertEqual([c_[0] for c_ in calls], [0, 1])


@unittest.skipUnless(os.environ.get("DECAY_TEST_CANDLES"), "set DECAY_TEST_CANDLES to a folder of <SYMBOL>_1h.csv files to run the pipeline-level test")
class TestPipeline(unittest.TestCase):
    def test_c1_equals_control_and_c_half_differs(self):
        root = HERE.parents[1]
        names = "ETH XRP XLM LINK ADA DOT ATOM TRX SOL AVAX XMR ALGO".split()
        common = ["--venue", "kraken", "--candle-dir", os.environ["DECAY_TEST_CANDLES"], "--quick", "--skip-importance", "--symbols", *names, "--seed", "123", "--prior-trials", "104",
                  "--sell-off", "2026-09-27", "2026-10-08"]
        with tempfile.TemporaryDirectory() as tmp:
            outs = {}
            for tag, cmd in (("ctrl", ["run_meta_venue.py"]), ("c1", ["run_meta_decay.py", "--decay-c", "1"]), ("c05", ["run_meta_decay.py", "--decay-c", "0.5"])):
                out = Path(tmp) / tag
                r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:], *common, "--out", str(out)], cwd=root, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout[-500:] + r.stderr[-500:])
                outs[tag] = sorted(p for p in out.iterdir() if p.is_dir())[-1]
            for fn in ("cpcv_predictions.csv", "walk_forward_predictions.csv"):
                self.assertEqual((outs["ctrl"] / fn).read_bytes(), (outs["c1"] / fn).read_bytes(), fn)
                self.assertNotEqual((outs["ctrl"] / fn).read_bytes(), (outs["c05"] / fn).read_bytes(), fn)
            self.assertEqual((outs["ctrl"] / "events.csv").read_bytes(), (outs["c05"] / "events.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
