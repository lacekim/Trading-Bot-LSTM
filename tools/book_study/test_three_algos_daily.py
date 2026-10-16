"""Tests for three_algo_daily.py, run_three_algos_daily.py and verify_three_algos_daily.py (stdlib unittest; synthetic data only).

Optional real-run test: set THREE_ALGO_DAILY_RUN to a finished runtime/book_three_algo_alpaca/<timestamp> folder.
"""
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import equity_events as EE
import run_three_algos_daily as RD
import three_algo as T0
import three_algo_daily as TD
import verify_three_algos_daily as VD

HERE = Path(__file__).resolve().parent


def series(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def events_for(sym, c, positions, side=None):
    """Event rows as the Alpaca meta run writes them (only the columns extend_events needs)."""
    t_sig = c.index.values[np.array(positions) - 1].astype("datetime64[s]").astype(np.int64)
    n = len(positions)
    return pd.DataFrame({"symbol": sym, "pos": positions, "t_sig": t_sig, "side": side if side is not None else np.ones(n), "ret_side": np.linspace(-0.02, 0.03, n), "bin": 1})


class TestMomSide(unittest.TestCase):
    def test_truth_table(self):
        r12 = np.array([1, 1, -1, -1, 1, np.nan, 0.0, 1, -1])
        r6 = np.array([1, -1, -1, 1, 1, 1, 1, np.nan, -1])
        self.assertEqual(TD.mom_side_from(r12, r6).tolist(), [1, 0, -1, 0, 1, 0, 0, 0, -1])

    def test_constants_and_features(self):
        self.assertEqual((TD.LB12, TD.LB6, TD.SKIP), (252, 126, 21))
        self.assertEqual(TD.MOM_LOOKBACKS, (252, 126))
        self.assertEqual(TD.MOM_FEATURES, EE.FEATURES)
        self.assertEqual(TD.DIR_FEATURES, [f for f in EE.FEATURES if f != "side"])
        self.assertEqual(len(TD.MOM_FEATURES), 13)
        self.assertEqual(TD.PRIOR_TRIALS, 120)
        self.assertTrue(TD.__doc__.lstrip().startswith("Three algorithms") and "DESIGN" in TD.__doc__)


class TestExtendEvents(unittest.TestCase):
    def test_hand_values(self):
        c = series(100.0 + np.arange(400))                                    # close[i] = 100 + i
        df = events_for("AAA", c, [300, 280])
        out = TD.extend_events(df, {"AAA": c})
        # event pos 300: signal bar 299; a = close[278] = 378; close[47] = 147; close[173] = 273
        self.assertAlmostEqual(out.loc[0, "ret12_1"], 378 / 147 - 1.0, places=12)
        self.assertAlmostEqual(out.loc[0, "ret6_1"], 378 / 273 - 1.0, places=12)
        self.assertAlmostEqual(out.loc[1, "ret12_1"], (100 + 258) / (100 + 27) - 1.0, places=12)
        self.assertAlmostEqual(out.loc[1, "ret6_1"], (100 + 258) / (100 + 153) - 1.0, places=12)
        self.assertEqual(out["mom_side"].tolist(), [1, 1])

    def test_falling_and_disagreeing_series(self):
        fall = series(1000.0 - 2.0 * np.arange(400))
        out = TD.extend_events(events_for("F", fall, [300], side=-np.ones(1)), {"F": fall})
        self.assertEqual(out["mom_side"].tolist(), [-1])
        # signal bar 299: close[47] = 100, close[173] = 200, close[278] = 150 -> 12-1 return +0.5, 6-1 return -0.25
        c = series(np.interp(np.arange(400), [0, 60, 150, 180, 278, 399], [100, 100, 200, 200, 150, 150]))
        out = TD.extend_events(events_for("D", c, [300]), {"D": c})
        r12, r6 = out.loc[0, "ret12_1"], out.loc[0, "ret6_1"]
        self.assertAlmostEqual(r12, 0.5, places=12)
        self.assertAlmostEqual(r6, -0.25, places=12)
        self.assertEqual(out["mom_side"].tolist(), [0])

    def test_boundaries(self):
        # exactly zero 12-1 return with a negative 6-1 return is no signal (not a short); zero long return is label 0
        v = np.interp(np.arange(400), [0, 60, 150, 180, 278, 399], [100, 100, 200, 200, 100, 100])       # close[47] = close[278] = 100 -> 12-1 return exactly 0; 6-1 return -0.5
        c = series(v)
        out = TD.extend_events(events_for("Z", c, [300]), {"Z": c})
        self.assertEqual((out.loc[0, "ret12_1"], out["mom_side"].tolist()), (0.0, [0]))
        df = events_for("Z", c, [300])
        df["ret_side"] = 0.0
        self.assertEqual(TD.extend_events(df, {"Z": c})["y_dir"].tolist(), [0])
        # the first usable signal bar is 252: pos 253 has returns, pos 252 (signal bar 251) has none
        c2 = series(100.0 + np.arange(400))
        out = TD.extend_events(events_for("B", c2, [253, 252]), {"B": c2})
        self.assertTrue(np.isfinite(out.loc[0, "ret12_1"]) and np.isfinite(out.loc[0, "ret6_1"]))
        self.assertTrue(np.isnan(out.loc[1, "ret12_1"]) and np.isnan(out.loc[1, "ret6_1"]))
        self.assertEqual(out["mom_side"].tolist(), [1, 0])
        self.assertAlmostEqual(out.loc[0, "ret12_1"], (100 + 231) / (100 + 0) - 1.0, places=12)       # signal bar 252: close[231] / close[0]

    def test_short_history_gives_nan_and_no_signal(self):
        c = series(100.0 + np.arange(400))
        out = TD.extend_events(events_for("S", c, [200]), {"S": c})                # signal bar 199 < 252
        self.assertTrue(np.isnan(out.loc[0, "ret12_1"]) and np.isnan(out.loc[0, "ret6_1"]))
        self.assertEqual(out["mom_side"].tolist(), [0])

    def test_ret_long_and_direction_label(self):
        c = series(100.0 + np.arange(400))
        df = events_for("AAA", c, [300, 301, 302], side=np.array([1.0, -1.0, 1.0]))
        df["ret_side"] = [0.01, 0.02, -0.03]
        out = TD.extend_events(df, {"AAA": c})
        self.assertEqual(out["ret_long"].tolist(), [0.01, -0.02, -0.03])
        self.assertEqual(out["y_dir"].tolist(), [1, 0, 0])

    def test_wrong_signal_bar_is_refused(self):
        c = series(100.0 + np.arange(400))
        df = events_for("AAA", c, [300])
        df["t_sig"] = df["t_sig"] + 86400
        with self.assertRaises(ValueError):
            TD.extend_events(df, {"AAA": c})

    def test_position_beyond_series_is_refused(self):
        c = series(100.0 + np.arange(300))
        df = events_for("AAA", c, [299])
        df["pos"] = 300
        with self.assertRaises(ValueError):
            TD.extend_events(df, {"AAA": c})


class TestDailyConventions(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        n = 600
        day0 = 1_600_000_000 // 86400
        self.df = pd.DataFrame({"t0": (day0 + rng.integers(0, 60, n)) * 86400, "ret_long": rng.normal(0.003, 0.03, n), "mkt_state": rng.integers(0, 2, n).astype(float),
                                "mom_side": rng.choice([-1, 0, 1], n)})
        rows = []
        for path in (0, 1):
            rows.append(pd.DataFrame({"event": np.arange(n), "split": 0, "path": path, "p_dir": rng.uniform(0.3, 0.7, n), "p_mom": rng.uniform(0.3, 0.7, n)}))
        self.preds = pd.concat(rows, ignore_index=True)
        self.win = {"w": (int(self.df["t0"].min()), int(self.df["t0"].min()) + 10 * 86400)}

    def test_only_annualisation_and_effective_sample_differ(self):
        h = 4
        a = TD.evaluate_three(self.df, self.preds, 0.002, 0.1, h, 7, self.win, 120)
        b = T0.evaluate_three(self.df, self.preds, 0.002, 0.1, h, 7, self.win, 120)
        pa, pb = pd.DataFrame(a["paths"]), pd.DataFrame(b["paths"])
        for col in ("events", "days", "mean_bp_per_candidate", "sr_daily", "skew", "kurt", "taken_share", "mean_size"):
            np.testing.assert_allclose(pa[col].to_numpy(float), pb[col].to_numpy(float), rtol=0, atol=1e-15, err_msg=col)
        np.testing.assert_allclose(pa["sr_annual"], pa["sr_daily"] * math.sqrt(252), rtol=1e-14)
        np.testing.assert_allclose(pb["sr_annual"], pb["sr_daily"] * math.sqrt(365), rtol=1e-14)
        np.testing.assert_allclose(pa["t_eff"], pa["days"] / 5.0, rtol=1e-14)               # holding period 5 days
        self.assertEqual(a["compare"].keys(), b["compare"].keys())
        pd.testing.assert_frame_equal(a["daily"], b["daily"])

    def test_dsr_trial_set(self):
        a = TD.evaluate_three(self.df, self.preds, 0.002, 0.1, 4, 7, self.win, TD.PRIOR_TRIALS)
        self.assertEqual(set(a["dsr"]["gate"]["by_trials"]), {"1", "10", "120", "121"})

    def test_checker_day_stats_uses_252(self):
        x = [0.1, -0.05, 0.2, 0.0, 0.07, -0.01]
        st = VD.day_stats_daily(x, 6, 5.0)
        self.assertAlmostEqual(st["sr_annual"], st["sr_daily"] * math.sqrt(252), places=12)
        self.assertAlmostEqual(st["t_eff"], 6 / 5.0, places=12)


class TestAlpacaRawReader(unittest.TestCase):
    def test_reader_rules(self):
        rows = ["symbol,date,close,volume",
                "AAA,2020-01-03,12.0,1", "AAA,2020-01-02,10.0,1", "AAA,2020-01-02,11.0,1",       # duplicate date: the later row wins; file order does not matter
                "AAA,2020-01-06,-3.0,1", "AAA,2020-01-07,0,1", "AAA,2020-01-08,,1",               # non-positive and missing closes are dropped
                "NA,2020-01-02,5.0,1", "NA,2020-01-03,6.0,1", "ZZZ,2020-01-02,9.0,1"]
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "a.csv.gz"
            with gzip.open(p, "wt", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
            out = VD.alpaca_raw(p, {"AAA", "NA"})
            self.assertEqual(set(out), {"AAA", "NA"})
            keys, closes = out["AAA"]
            self.assertEqual(closes, [11.0, 12.0])
            self.assertEqual(len(keys), 2)
            self.assertEqual(out["NA"][1], [5.0, 6.0])
            # the runner's loader gives the same closes
            ee = EE.load_alpaca(p)
            self.assertEqual(ee[ee["symbol"] == "AAA"]["close"].tolist(), [11.0, 12.0])
            self.assertEqual(sorted(ee["symbol"].unique()), ["AAA", "NA", "ZZZ"])


class TestRunnerPieces(unittest.TestCase):
    def test_load_source_closes_checks_bar_counts(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "a.csv.gz"
            dates = pd.bdate_range("2020-01-01", periods=10)
            rows = ["symbol,date,close,volume"] + [f"AAA,{d.date()},{10 + i},1" for i, d in enumerate(dates)]
            with gzip.open(p, "wt", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
            good = pd.DataFrame({"symbol": ["AAA"], "bars": [10], "last": [str(dates[-1])]})
            c = RD.load_source_closes(p, good)
            self.assertEqual(len(c["AAA"]), 10)
            cut = pd.DataFrame({"symbol": ["AAA"], "bars": [8], "last": [str(dates[7])]})          # the source run stopped earlier: later bars are cut
            self.assertEqual(len(RD.load_source_closes(p, cut)["AAA"]), 8)
            bad = pd.DataFrame({"symbol": ["AAA"], "bars": [9], "last": [str(dates[-1])]})
            with self.assertRaises(SystemExit):
                RD.load_source_closes(p, bad)

    def test_venue_section_counts(self):
        t0 = np.array([100, 101, 102, 200]) * 86400
        df = pd.DataFrame({"t0": t0, "mom_side": [1, 0, -1, 1], "side": [1, 1, -1, 1]})
        lines = RD.venue_section(Path("/s"), Path("/c.csv.gz"), "m" * 32, df, {"a": (100 * 86400, 103 * 86400), "b": (300 * 86400, 301 * 86400)}, 120)
        txt = "\n".join(lines)
        self.assertIn("| a | 3 | 1970-04-11 | 1970-04-13 |", txt)
        self.assertIn("| b | 0 | n/a | n/a |", txt)
        edge = RD.venue_section(Path("/s"), Path("/c.csv.gz"), "m" * 32, pd.DataFrame({"t0": np.array([100, 103]) * 86400, "mom_side": [1, 1], "side": [1, 1]}), {"a": (100 * 86400, 103 * 86400)}, 120)
        self.assertIn("| a | 1 | 1970-04-11 | 1970-04-11 |", "\n".join(edge))                 # [lo, hi): the event at hi is outside
        diff = RD.venue_section(Path("/s"), Path("/c.csv.gz"), "m" * 32, pd.DataFrame({"t0": np.array([100, 101]) * 86400, "mom_side": [1, 1], "side": [1, -1]}), {"a": (0, 10**9)}, 120)
        self.assertIn("the MOM side equals the source side: False", "\n".join(diff))
        self.assertIn("MOM events 3 of 4 (0.7500)", txt)
        self.assertIn("the MOM side equals the source side: True", txt)
        self.assertIn("prior trials 120; the gate is the 121st", txt)

    def test_non_alpaca_source_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "summary_numbers.json").write_text(json.dumps({"meta": {"venue": "kraken", "timeframe": "1h"}, "params": {}}), encoding="utf-8")
            old = sys.argv
            sys.argv = ["x", "--source", str(d), "--out", str(d / "out")]
            try:
                with self.assertRaises(SystemExit) as cm:
                    RD.main()
                self.assertIn("not 'alpaca'", str(cm.exception))
            finally:
                sys.argv = old


class TestCheckerSetup(unittest.TestCase):
    def test_markers_occur_once_and_patched_module_loads(self):
        src = (HERE / "verify_three_algos.py").read_text(encoding="utf-8")
        for a, b, _ in VD.REGIONS:
            self.assertEqual((src.count(a), src.count(b)), (1, 1), (a, b))
            self.assertLess(src.index(a), src.index(b))
        for old, new in VD.SUBSTITUTIONS:
            self.assertEqual(src.count(old), 1, old)
        mod, md5 = VD.load_patched()
        self.assertEqual(md5, hashlib.md5((HERE / "verify_three_algos.py").read_bytes()).hexdigest())
        self.assertEqual(mod.LOOKBACKS, (252, 126))
        self.assertEqual(mod.DECLARED_DIR, TD.DIR_FEATURES)
        self.assertEqual(mod.DECLARED_MOM, TD.MOM_FEATURES)

    def test_changed_original_is_refused(self):
        orig = list(VD.REGIONS)
        try:
            VD.REGIONS = orig + [("this marker is not in the file", "ALGOS = (", "x")]
            with self.assertRaises(SystemExit):
                VD.load_patched()
        finally:
            VD.REGIONS = orig

    def test_constants_agree(self):
        self.assertEqual((RD.PRIOR_TRIALS_DAILY, VD.PRIOR, TD.PRIOR_TRIALS), (120, 120, 120))
        self.assertEqual((VD.H_DAYS, VD.COST), (EE.H, 0.0020))


@unittest.skipUnless(os.environ.get("THREE_ALGO_DAILY_RUN"), "set THREE_ALGO_DAILY_RUN to a finished run folder")
class TestRealRun(unittest.TestCase):
    def test_checker_passes_on_real_run(self):
        self.assertEqual(VD.main(os.environ["THREE_ALGO_DAILY_RUN"], os.environ.get("THREE_ALGO_DAILY_SOURCE"), os.environ.get("THREE_ALGO_DAILY_CSV")), 0)


if __name__ == "__main__":
    unittest.main()
