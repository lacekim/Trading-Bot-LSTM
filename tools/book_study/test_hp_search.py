#!/usr/bin/env python
"""Tests for run_hp_search.py / verify_hp_search.py.

Unit tests (synthetic data): the declared grid, the checker's own purged folds against meta_model.purged_kfold (including overlapping labels and ties in start time), the checker's AUC and
weighted log loss against sklearn / meta_model, the fold properties (disjoint, covering, no overlap with the training part), and the quick-mode switches.
Pipeline test (optional): set HP_TEST_RUN to the ABSOLUTE path of a 14-feature run folder; it runs a quick search with two workers (this also exercises the process-pool pickling of
fit_oof) on three configs whose best log loss is not the default, then runs the checker on the result.

    python -m unittest tools/book_study/test_hp_search.py -v
    HP_TEST_RUN=/abs/path/to/run python -m unittest tools/book_study/test_hp_search.py -v
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import meta_model as MM  # noqa: E402
import run_hp_search as RH  # noqa: E402
import verify_hp_search as VH  # noqa: E402


def synthetic(n=400, seed=3, ties=True):
    rng = np.random.default_rng(seed)
    t0 = np.sort(rng.integers(0, 5_000, n)).astype(float) if ties else np.sort(rng.random(n) * 5_000)
    t1 = t0 + rng.integers(1, 400, n)
    return t0, t1


class TestGrid(unittest.TestCase):
    def test_grid_is_the_declared_twelve(self):
        self.assertEqual(len(RH.GRID), 12)
        self.assertEqual([g["config"] for g in RH.GRID], list(range(12)))
        self.assertEqual(len({(g["min_weight_fraction_leaf"], str(g["max_features"])) for g in RH.GRID}), 12)
        self.assertEqual(sorted({g["min_weight_fraction_leaf"] for g in RH.GRID}), [0.005, 0.01, 0.025, 0.05, 0.1, 0.2])
        self.assertEqual({str(g["max_features"]) for g in RH.GRID}, {"sqrt", "0.5"})

    def test_default_is_one_grid_point_with_the_pipeline_setting(self):
        d = RH.GRID[RH.default_id()]
        self.assertEqual((d["min_weight_fraction_leaf"], d["max_features"]), (0.05, "sqrt"))
        self.assertEqual(RH.default_id(), 6)

    def test_checker_grid_equals_runner_grid(self):
        self.assertEqual(VH.GRID, [(g["min_weight_fraction_leaf"], g["max_features"]) for g in RH.GRID])
        self.assertEqual((VH.SEEDS, VH.SEARCH_SEED, VH.N_FOLDS, VH.EMBARGO), (RH.SEEDS, RH.SEARCH_SEED, RH.N_FOLDS, RH.EMBARGO))

    def test_quick_switches_only_size(self):
        cfg = RH.GRID[3]
        full, quick = RH.forest_kw(cfg, False), RH.forest_kw(cfg, True)
        self.assertEqual((full["n_estimators"], full["max_draws"], full["bootstrap"]), (100, 10000, "sequential"))
        self.assertEqual((quick["n_estimators"], quick["max_draws"]), (10, 3000))
        for k in ("min_weight_fraction_leaf", "max_features", "bootstrap"):
            self.assertEqual(full[k], quick[k])
        self.assertEqual(full["min_weight_fraction_leaf"], cfg["min_weight_fraction_leaf"])
        self.assertEqual(full["max_features"], cfg["max_features"])


class TestFolds(unittest.TestCase):
    def test_checker_folds_equal_pipeline_folds(self):
        for seed in (1, 2, 3):
            for ties in (True, False):
                t0, t1 = synthetic(seed=seed, ties=ties)
                idx = list(range(len(t0)))
                mine = VH.purged_folds(list(t0), list(t1), idx, 5, 0.01)
                ref = list(MM.purged_kfold(t0, t1, 5, 0.01))
                self.assertEqual(len(mine), len(ref))
                for (a, b), (c, d) in zip(mine, ref):
                    self.assertEqual(a, sorted(c.tolist()))
                    self.assertEqual(b, sorted(d.tolist()))

    def test_checker_folds_on_a_subset_with_global_indices(self):
        t0, t1 = synthetic(n=300, seed=5)
        sub = np.flatnonzero(t1 < 4000)
        ref = list(MM.purged_kfold(t0[sub], t1[sub], 5, 0.01))
        mine = VH.purged_folds(list(t0), list(t1), sub.tolist(), 5, 0.01)
        for (a, b), (c, d) in zip(mine, ref):
            self.assertEqual(a, sorted(sub[c].tolist()))
            self.assertEqual(b, sorted(sub[d].tolist()))

    def test_fold_properties(self):
        t0, t1 = synthetic(n=500, seed=9)
        folds = list(MM.purged_kfold(t0, t1, 5, 0.01))
        all_test = np.concatenate([te for _, te in folds])
        self.assertEqual(sorted(all_test.tolist()), list(range(500)))            # every event is tested exactly once
        for tr, te in folds:
            self.assertEqual(len(set(tr.tolist()) & set(te.tolist())), 0)
            lo, hi = t0[te].min(), t1[te].max()
            self.assertFalse(((t0[tr] <= hi) & (t1[tr] >= lo)).any())             # no training label touches the test span
            emb = 0.01 * (t0.max() + 1.0 - t0.min())
            self.assertFalse(((t0[tr] > hi) & (t0[tr] <= hi + emb)).any())        # nothing starts inside the embargo
            self.assertLess(len(tr), 500 - len(te) + 1)


class TestScores(unittest.TestCase):
    def test_auc_matches_sklearn_including_ties(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        for _ in range(20):
            y = rng.integers(0, 2, 200)
            p = np.round(rng.random(200), 1)                                      # many ties
            self.assertAlmostEqual(VH.auc(y.tolist(), p.tolist()), roc_auc_score(y, p), places=12)
        self.assertTrue(np.isnan(VH.auc([1, 1, 1], [0.1, 0.2, 0.3])))             # one class only

    def test_weighted_log_loss_matches_pipeline_function_and_clips(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        p = rng.random(300)
        p[:3] = [0.0, 1.0, 1.0]                                                   # exercises the clip
        w = rng.random(300) + 0.1
        self.assertAlmostEqual(VH.wll(y.tolist(), p.tolist(), w.tolist()), MM.weighted_log_loss(y.astype(float), p, w), places=12)

    def test_log_loss_prefers_the_constant_over_a_noisy_model_when_there_is_no_signal(self):
        # the property that makes the declared criterion risky on weak features: with no signal a constant prior beats noisy probabilities
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, 5000).astype(float)
        w = np.ones(5000)
        noisy = np.clip(0.5 + 0.1 * rng.standard_normal(5000), 0.01, 0.99)
        const = np.full(5000, y.mean())
        self.assertLess(MM.weighted_log_loss(y, const, w), MM.weighted_log_loss(y, noisy, w))
        self.assertLess(abs(VH.auc(y.tolist(), noisy.tolist()) - 0.5), 0.03)


@unittest.skipUnless(os.environ.get("HP_TEST_RUN"), "set HP_TEST_RUN to the absolute path of a 14-feature run folder")
class TestPipeline(unittest.TestCase):
    def test_quick_search_with_two_workers_then_checker(self):
        run = os.environ["HP_TEST_RUN"]
        self.assertTrue(os.path.isabs(run), "HP_TEST_RUN must be an absolute path")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "hp"
            r = subprocess.run([sys.executable, str(HERE / "run_hp_search.py"), "--run", run, "--out", str(out), "--quick", "--quick-configs", "0", "10", "11",
                                "--seeds", "20261017", "20261101", "--jobs", "2"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            for f in ("grid.csv", "oof_predictions.csv", "config_scores.csv", "folds.csv", "wf_predictions.csv", "wf_metrics.csv", "hp_extra.json", "hp_summary.md", "run.log"):
                self.assertTrue((out / f).exists(), f)
            c = subprocess.run([sys.executable, str(HERE / "verify_hp_search.py"), str(out), "--allow-quick"], capture_output=True, text=True)
            self.assertEqual(c.returncode, 0, c.stdout[-2500:])
            self.assertIn("10/10 checks passed", c.stdout)
            c2 = subprocess.run([sys.executable, str(HERE / "verify_hp_search.py"), str(out)], capture_output=True, text=True)    # without --allow-quick a quick run must fail
            self.assertNotEqual(c2.returncode, 0)

    def test_quick_configs_refused_without_quick(self):
        run = os.environ["HP_TEST_RUN"]
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([sys.executable, str(HERE / "run_hp_search.py"), "--run", run, "--out", str(Path(td) / "x"), "--quick-configs", "0"], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
