"""Planted-error tests for verify_book_ports_bars.py: the checker must pass an untouched run folder (apart from the two checks that a --quick smoke run cannot satisfy) and must fail, on the named check,
when one recorded number or one line of SUMMARY.md is changed.  Needs a finished run folder: set BOOK_PORTS_RUN (default: the newest runtime/smoke_chk/*/ of the project); skipped when there is none."""
from __future__ import annotations

import copy
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_book_ports_bars as VB  # noqa: E402

RUN = os.environ.get("BOOK_PORTS_RUN") or (sorted(glob.glob(str(HERE.parents[1] / "runtime" / "smoke_chk" / "*")))[-1:] or [None])[0]
HAVE_RUN = bool(RUN) and Path(RUN, "summary_numbers.json").exists()
EXPECTED_QUICK_FAILS = {"not a --quick run (full sizes)", "every section present in the results"}


def run_checks(j, folder, only=None):
    _, _, src, ev = VB.load(folder)
    cfg = j["meta"]["config"]
    R = VB.Report()
    fns = {"integrity": lambda: VB.check_integrity(R, folder, j, src, ev), "ch3": lambda: VB.check_ch3(R, j, ev), "ch4": lambda: VB.check_ch4(R, j, ev, src, cfg),
           "ch7": lambda: VB.check_ch7(R, j, ev, cfg), "ch8": lambda: VB.check_ch8(R, j, ev, cfg), "ch15": lambda: VB.check_ch15(R, j, ev, src, folder),
           "ch17_19": lambda: VB.check_ch17_19(R, j), "ch20": lambda: VB.check_ch20(R, j), "ch21": lambda: VB.check_ch21(R, j, ev, cfg), "examples": lambda: VB.check_examples(R, j), "ch2": lambda: VB.check_ch2(R, j, cfg), "ch16": lambda: VB.check_ch16(R, j, cfg), "text": lambda: VB.check_text(R, Path(folder), j)}
    for k in (only or fns):
        fns[k]()
    return {name for ok, name, _ in R.rows if not ok}, R


def load_base():
    folder = Path(RUN)
    return folder, json.loads((folder / "summary_numbers.json").read_text(encoding="utf-8"))


PLANTS = [
    ("ch3", "ch3 meta_label_bin share of class 1", lambda j: j["results"]["ch3"]["meta_label_bin"]["classes"].__setitem__("1", j["results"]["ch3"]["meta_label_bin"]["classes"]["1"] + 1e-4)),
    ("ch4", "ch4 ETH: mean average uniqueness", lambda j: j["results"]["ch4"]["uniqueness_and_bootstrap"][0].__setitem__("mean_avg_uniqueness", j["results"]["ch4"]["uniqueness_and_bootstrap"][0]["mean_avg_uniqueness"] + 1e-3)),
    ("ch4", "ch4 MC: maxH", lambda j: j["results"]["ch4"]["monte_carlo"].__setitem__("maxH", j["results"]["ch4"]["monte_carlo"]["maxH"] + 1)),
    ("ch7", "ch7 fold 1: train left / right / total", lambda j: j["results"]["ch7"]["folds"][1].__setitem__("train_left", j["results"]["ch7"]["folds"][1]["train_left"] + 1)),
    ("ch7", "ch7 fold 2: training labels overlapping the test span", lambda j: j["results"]["ch7"]["folds"][2].__setitem__("overlapping_train_labels", 3)),
    ("ch8", "ch8 majority-class share", lambda j: j["results"]["ch8"]["sample"].__setitem__("majority_share", j["results"]["ch8"]["sample"]["majority_share"] + 0.01)),
    ("ch15", "ch15 primary: precision", lambda j: j["results"]["ch15"]["strategy_risk"][0].__setitem__("precision", j["results"]["ch15"]["strategy_risk"][0]["precision"] + 1e-3)),
    ("ch15", "ch15 meta: average loss", lambda j: j["results"]["ch15"]["strategy_risk"][1].__setitem__("avg_loss", j["results"]["ch15"]["strategy_risk"][1]["avg_loss"] * 1.01)),
    ("ch15", "ch15 primary: probability of failure for Sharpe 1 (as printed)", lambda j: j["results"]["ch15"]["strategy_risk"][0].__setitem__("prob_failure_sr1", 0.4)),
    ("ch15", "ch15 window primary/june_2026_ai_correction: bets", lambda j: [w.__setitem__("bets", w["bets"] + 1) for w in j["results"]["ch15"]["windows"] if w["variant"] == "primary" and w["window"] == "june_2026_ai_correction"]),
    ("ch17_19", "ch17/18 sadf: events re-computed by the book's code >= 30 and largest difference to the stored feature below 1e-8",
     lambda j: j["results"]["ch17"]["event_features_recomputed_by_book_code"]["sadf"].__setitem__("max_abs_diff", 1e-3)),
    ("ch20", "ch20 one-touch: single-thread = multiprocessing = vectorised", lambda j: j["results"]["ch20"]["one_touch"].__setitem__("single_equals_multi", False)),
    ("ch21", "ch21 dynamic solution: best Sharpe over all distinct trajectories", lambda j: j["results"]["ch21"].__setitem__("dynamic_sr", j["results"]["ch21"]["dynamic_sr"] + 0.01)),
    ("ch21", "ch21 static solution Sharpe (closed form, re-solved each block, with trading costs)", lambda j: j["results"]["ch21"].__setitem__("static_sr", j["results"]["ch21"]["static_sr"] - 0.01)),
    ("ch21", "ch21 block 0 asset 1: cost coefficient (book's uniform draw x volatility)", lambda j: j["results"]["ch21"]["cost_coefficients"][0].__setitem__(1, j["results"]["ch21"]["cost_coefficients"][0][1] * 1.1)),
    ("ch21", "ch21 distinct weight vectors (L1 sphere of radius k in Z^n)", lambda j: j["results"]["ch21"].__setitem__("weight_vectors_distinct", 80)),
    ("examples", "ch3 Snippet 2.4: printed CUSUM = the project's filter on real bars for every symbol", lambda j: j["results"]["ch3"]["cusum_snippet_2_4"][next(iter(j["results"]["ch3"]["cusum_snippet_2_4"]))].__setitem__("identical", False)),
    ("examples", "ch4 Snippet 4.6: sequential-bootstrap mean uniqueness near the exact value", lambda j: j["results"]["ch4"]["book_example_4_6"].__setitem__("sequential_bootstrap_uniqueness_mean", 0.9)),
    ("examples", "ch16 Snippet 16.4 reproduces Table 16.1 (printed to two decimals) for HRP, IVP and the CLA stand-in", lambda j: j["results"]["ch16"]["table_16_1"].__setitem__("HRP", 0.5)),
    ("ch2", "ch2 largest eigenvalue share", lambda j: j["results"]["ch2"]["eigen"].__setitem__("first_share", j["results"]["ch2"]["eigen"]["first_share"] + 0.01)),
    ("ch2", "ch2 symbols used", lambda j: j["results"]["ch2"].__setitem__("symbols_used", j["results"]["ch2"]["symbols_used"] - 1)),
    ("ch16", "ch16 inverse-variance portfolio: effective number of positions", lambda j: j["results"]["ch16"]["full_sample"]["effective_n"].__setitem__("ivp", j["results"]["ch16"]["full_sample"]["effective_n"]["ivp"] + 0.5)),
    ("ch16", "ch16 days and window", lambda j: j["results"]["ch16"]["data"].__setitem__("days", j["results"]["ch16"]["data"]["days"] + 1)),
    ("integrity", "no section failed", lambda j: j["meta"].__setitem__("errors", ["ch5"])),
    ("integrity", "script md5 book_port_c.py unchanged since the run", lambda j: j["meta"]["script_md5"].__setitem__("book_port_c.py", "0" * 32)),
]



@unittest.skipUnless(HAVE_RUN, "no finished run folder")
class TestChecker(unittest.TestCase):
    def test_untouched_run_passes_except_the_quick_run_checks(self):
        folder, j = load_base()
        fails, R = run_checks(j, folder)
        self.assertLessEqual(fails, EXPECTED_QUICK_FAILS)
        self.assertGreater(len(R.rows), 140)

    def test_every_planted_error_is_caught_by_its_check(self):
        folder, j0 = load_base()
        for section, expected, plant in PLANTS:
            with self.subTest(expected):
                j = copy.deepcopy(j0)
                plant(j)
                fails, _ = run_checks(j, folder, only=[section])
                self.assertIn(expected, fails, sorted(fails))

    def test_summary_text_tampering_is_caught(self):
        import tempfile
        folder, j = load_base()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "run"
            shutil.copytree(folder, d)
            t = (d / "SUMMARY.md").read_text(encoding="utf-8")
            sr = f"Sharpe {j['results']['ch21']['dynamic_sr']:.3f}"
            self.assertIn(sr, t)
            (d / "SUMMARY.md").write_text(t.replace(sr, "Sharpe 9.999"), encoding="utf-8")
            fails, _ = run_checks(j, d, only=["text"])
            self.assertIn("SUMMARY ch21 Sharpe values", fails)
            (d / "SUMMARY.md").write_text(t.replace("Caution on the 'probability of failure' column", "Remark"), encoding="utf-8")
            fails, _ = run_checks(j, d, only=["text"])
            self.assertIn("SUMMARY states the probability-of-failure caution", fails)


if __name__ == "__main__":
    unittest.main()
