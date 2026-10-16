"""Tests for run_seed_pairs.py (arithmetic and extraction only; no pipeline is run).  python -m unittest test_seed_pairs -v"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_pairs as SP  # noqa: E402


def fake_summary(prior=104, with_windows=True):
    win = {"mean_bp_per_candidate": -10.0}
    s = {"params": {"prior_trials": prior}, "cpcv": {"auc_mean": 0.515},
         "walk_forward": {"auc": 0.528},
         "eval": {"summary": {"meta": {"mean_sr_annual": 1.13, "mean_bp_per_candidate": 3.1}, "meta_minus_placebo": {"mean_sr_daily_diff": 0.046}},
                  "windows": ({"meta|june_2026_ai_correction": {"mean_bp_per_candidate": 1.2}, "placebo|june_2026_ai_correction": {"mean_bp_per_candidate": 3.4},
                               "meta|sell_off_post_holdout": win, "placebo|sell_off_post_holdout": {"mean_bp_per_candidate": -5.5}} if with_windows else {}),
                  "dsr": {"by_trials": {str(prior + 1): {"dsr": 0.107}, str(prior): {"dsr": 0.108}}}},
         "walk_forward_eval": {"summary": {"meta": {"mean_sr_annual": 1.06}}}}
    return s


class T(unittest.TestCase):
    def test_extract(self):
        m = SP.extract_metrics(fake_summary())
        self.assertEqual(set(m), set(SP.METRICS))
        self.assertAlmostEqual(m["cpcv_meta_sr_annual"], 1.13)
        self.assertAlmostEqual(m["june_cpcv_meta_minus_placebo_bp"], 1.2 - 3.4)
        self.assertAlmostEqual(m["selloff_cpcv_meta_minus_placebo_bp"], -10.0 + 5.5)
        self.assertAlmostEqual(m["dsr_prior_plus_one"], 0.107)          # prior + 1, not prior
        self.assertAlmostEqual(m["wf_meta_sr_annual"], 1.06)

    def test_dsr_n_overrides_prior_plus_one(self):
        s = fake_summary(prior=105)                               # keys 105 (0.108) and 106 (0.107)
        self.assertAlmostEqual(SP.extract_metrics(s)["dsr_prior_plus_one"], 0.107)
        self.assertAlmostEqual(SP.extract_metrics(s, 105)["dsr_prior_plus_one"], 0.108)
        self.assertTrue(math.isnan(SP.extract_metrics(s, 120)["dsr_prior_plus_one"]))          # a trial count the run did not report is NaN, not a guess

    def test_commands_default_and_with_treatment_options(self):
        import argparse
        base = dict(venue="kraken", candle_dir="c", prior_trials=104, treat_prior_trials=None, sell_off=None, quick=False, treat_args="", treat_check_args="",
                    ctrl_checker="vc.py", treat_checker="vt.py")
        a = argparse.Namespace(**base)
        c = SP.run_cmd(a, "x.py", 7, "out", "treat")
        self.assertEqual(c[c.index("--prior-trials") + 1], "104")
        self.assertNotIn("--decay-c", c)
        self.assertEqual(SP.check_cmd(a, "treat", Path("r"))[-2:], ["--prior-trials", "104"])
        b = argparse.Namespace(**{**base, "treat_prior_trials": 106, "treat_args": "--decay-c 0.5", "treat_check_args": "--decay-c 0.5"})
        ct, cc = SP.run_cmd(b, "x.py", 7, "out", "treat"), SP.run_cmd(b, "x.py", 7, "out", "ctrl")
        self.assertEqual(ct[ct.index("--prior-trials") + 1], "106")
        self.assertEqual(ct[-2:], ["--decay-c", "0.5"])
        self.assertEqual(cc[cc.index("--prior-trials") + 1], "104")                  # the control keeps --prior-trials
        self.assertNotIn("--decay-c", cc)
        k = SP.check_cmd(b, "treat", Path("r"))
        self.assertEqual((k[k.index("--prior-trials") + 1], k[-2:]), ("106", ["--decay-c", "0.5"]))
        k0 = SP.check_cmd(b, "ctrl", Path("r"))
        self.assertEqual((k0[k0.index("--prior-trials") + 1], "--decay-c" in k0), ("104", False))

    def test_missing_windows_are_nan(self):
        m = SP.extract_metrics(fake_summary(with_windows=False))
        for k in ("june_cpcv_meta_bp", "june_cpcv_meta_minus_placebo_bp", "selloff_cpcv_meta_bp"):
            self.assertTrue(math.isnan(m[k]), k)

    def test_paired_stats(self):
        p = SP.paired_stats([1.0, 2.0, 3.0, -1.0, 5.0])
        self.assertEqual(p["n"], 5)
        self.assertAlmostEqual(p["mean"], 2.0)
        self.assertAlmostEqual(p["sd"], math.sqrt((1 + 0 + 1 + 9 + 9) / 4))
        self.assertAlmostEqual(p["se"], p["sd"] / math.sqrt(5))
        self.assertAlmostEqual(p["mean_over_se"], 2.0 / p["se"])
        self.assertEqual((p["min"], p["max"], p["n_positive"]), (-1.0, 5.0, 4))

    def test_paired_stats_nan_and_small_n(self):
        p = SP.paired_stats([float("nan"), 2.0])
        self.assertEqual(p["n"], 1)
        self.assertTrue(math.isnan(p["sd"]) and math.isnan(p["mean_over_se"]))
        self.assertEqual(SP.paired_stats([])["n"], 0)

    def test_zero_spread_gives_nan_ratio(self):
        self.assertTrue(math.isnan(SP.paired_stats([1.0, 1.0, 1.0])["mean_over_se"]))

    def test_arm_stats(self):
        a = SP.arm_stats([1.0, 3.0, float("nan")])
        self.assertEqual(a["n"], 2)
        self.assertAlmostEqual(a["mean"], 2.0)
        self.assertAlmostEqual(a["sd"], math.sqrt(2.0))


if __name__ == "__main__":
    unittest.main()
