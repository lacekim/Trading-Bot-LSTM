"""Tests for replay_stats.py and verify_replay.py (numpy/pandas only; no TensorFlow needed).

    python -m unittest tools/post_holdout_replay/test_replay_tools.py   (or run this file)
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_stats as S  # noqa: E402


def trades(pnls):
    return pd.DataFrame({"net_pnl": pnls, "return_pct": [p / 100 for p in pnls]})


class Stats(unittest.TestCase):
    def test_pf_basic(self):
        s = S.pooled_stats(trades([30.0, -10.0, -5.0, 15.0]))
        self.assertEqual(s["trades"], 4)
        self.assertAlmostEqual(s["profit_factor"], 45.0 / 15.0)
        self.assertAlmostEqual(s["win_rate_pct"], 50.0)
        self.assertAlmostEqual(s["net_pnl"], 30.0)

    def test_pf_no_losses_is_inf(self):
        self.assertTrue(np.isinf(S.pooled_stats(trades([1.0, 2.0]))["profit_factor"]))

    def test_pf_no_gains_is_zero(self):
        self.assertEqual(S.pooled_stats(trades([-1.0, -2.0]))["profit_factor"], 0.0)

    def test_empty(self):
        s = S.pooled_stats(trades([]))
        self.assertEqual(s["trades"], 0)
        self.assertTrue(np.isnan(s["profit_factor"]))

    def test_bounds_from_windows(self):
        e = {"windows": [{"start": "2025-09-05 01:00:00", "end": "2026-01-07 15:00:00"},
                         {"start": "2026-01-07 16:00:00", "end": "2026-04-25 17:00:00"},
                         {"start": "2026-04-25 18:00:00", "end": "2026-08-11 19:00:00"}]}
        self.assertEqual(S.parse_holdout_bounds(e), (pd.Timestamp("2025-09-05 01:00:00"), pd.Timestamp("2026-08-11 19:00:00")))
        self.assertIsNone(S.parse_holdout_bounds({"windows": []}))
        self.assertIsNone(S.parse_holdout_bounds({}))

    def test_drawdown_window(self):
        idx = pd.Series([1.0, 1.2, 1.1, 0.9, 1.0, 1.3], index=pd.date_range("2026-09-01", periods=6, freq="D"))
        peak, trough, dd = S.max_drawdown_window(idx)
        self.assertEqual(peak, pd.Timestamp("2026-09-02"))
        self.assertEqual(trough, pd.Timestamp("2026-09-04"))
        self.assertAlmostEqual(dd, (0.9 / 1.2 - 1) * 100)

    def test_equal_weight_index_only_uses_after_start(self):
        t = pd.date_range("2026-08-10", periods=100, freq="h")
        closes = {f"S{i}": pd.Series(100.0 * (1.001 ** np.arange(100)), index=t) for i in range(12)}
        start = t[49]
        idx = S.equal_weight_index(closes, start)
        self.assertEqual(idx.index[0], t[50])
        self.assertAlmostEqual(idx.iloc[-1], 1.001 ** 50, places=6)

    def test_equal_weight_index_drops_thin_timestamps(self):
        t = pd.date_range("2026-08-10", periods=10, freq="h")
        closes = {f"S{i}": pd.Series(np.arange(10) + 100.0, index=t) for i in range(3)}
        self.assertTrue(S.equal_weight_index(closes, t[0], min_symbols=10).empty)

    def test_spearman(self):
        self.assertAlmostEqual(S.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(S.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(np.isnan(S.spearman([1, 1, 1, 1], [1, 2, 3, 4])))
        self.assertTrue(np.isnan(S.spearman([1, 2], [1, 2])))

    def test_volatility_detector_shows_abs_but_not_signed_correlation(self):
        rng = np.random.default_rng(0)
        ret = rng.normal(0, 1, 5000) * rng.choice([0.2, 1.0], 5000)   # random sign, varying size
        prob = np.abs(ret) + rng.normal(0, 0.05, 5000)                 # a pure "size of move" predictor
        self.assertGreater(S.spearman(prob, np.abs(ret)), 0.8)
        self.assertLess(abs(S.spearman(prob, ret)), 0.05)
        self.assertLess(abs(S.top_decile_edge(prob, ret)), 0.15)       # no directional edge in the top decile

    def test_directional_model_shows_signed_correlation_and_edge(self):
        rng = np.random.default_rng(1)
        ret = rng.normal(0, 1, 5000)
        prob = ret + rng.normal(0, 0.5, 5000)
        self.assertGreater(S.spearman(prob, ret), 0.7)
        self.assertGreater(S.top_decile_edge(prob, ret), 1.0)

    def test_top_decile_edge_exact(self):
        prob = np.arange(10, dtype=float)                 # candle 9 has the highest probability
        ret = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])
        self.assertAlmostEqual(S.top_decile_edge(prob, ret), 1.0 - 0.1)
        self.assertTrue(np.isnan(S.top_decile_edge([], [])))

    def test_compare_selftest(self):
        stored = {"holdout_trades": 10, "holdout_profit_factor": 1.5, "holdout_return_pct": 2.0}
        ok = {"trades": 10, "profit_factor": 1.505, "return_pct": 2.01}
        self.assertTrue(S.compare_selftest(stored, ok)["passed"])
        self.assertFalse(S.compare_selftest(stored, {**ok, "trades": 11})["passed"])
        self.assertFalse(S.compare_selftest(stored, {**ok, "profit_factor": 1.6})["passed"])
        self.assertFalse(S.compare_selftest(stored, {**ok, "return_pct": 2.5})["passed"])
        inf_stored = {**stored, "holdout_profit_factor": float("inf")}
        self.assertTrue(S.compare_selftest(inf_stored, {**ok, "profit_factor": float("inf")})["passed"])
        self.assertFalse(S.compare_selftest(inf_stored, ok)["passed"])


def build_fixture(folder: Path, *, internal_error=False, no_rows=False, bad_rho=False, overlap=False, tamper_summary=False, integrity_ok=True, selftest_ok=True):
    """A tiny synthetic replay output folder, consistent unless a flaw is requested."""
    tr = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "BBB"], "direction": ["SHORT"] * 4,
        "entry_time": ["2026-08-12 01:00:00", "2026-08-13 01:00:00", "2026-08-12 05:00:00", "2026-08-14 05:00:00"],
        "exit_time": ["2026-08-12 09:00:00", "2026-08-13 09:00:00", "2026-08-12 09:00:00", "2026-08-14 09:00:00"],
        "net_pnl": [50.0, -20.0, -10.0, 30.0], "return_pct": [0.5, -0.2, -0.1, 0.3],
        "methodology": ["current", "current", "legacy", "legacy"], "promoted": [False] * 4,
    })
    if overlap:
        tr.loc[0, "entry_time"] = "2026-08-11 01:00:00"
    tr.to_csv(folder / "replay_trades.csv", index=False)
    pd.DataFrame({
        "direction": ["short", "short"], "symbol": ["AAA", "BBB"], "holdout_end": ["2026-08-11 20:00:00"] * 2,
        "trades": [2, 2], "rho_signed": [0.10, 0.20], "rho_abs": [0.30, 0.50 if not bad_rho else 1.5],
    }).to_csv(folder / "replay_per_symbol.csv", index=False)
    s = S.pooled_stats(tr)
    if tamper_summary:
        s["profit_factor"] = 99.0
    (folder / "summary_numbers.json").write_text(json.dumps(
        {"short": {"all": s, "_symbols": {"median_rho_signed": 0.15, "median_rho_abs": 0.40}}}), encoding="utf-8")
    snap = {"files": 3, "newest_mtime": 1.0, "calibration_md5": "x"}
    after = dict(snap) if integrity_ok else {**snap, "calibration_md5": "y"}
    (folder / "integrity.json").write_text(json.dumps(
        {"before": {"short": snap}, "after": {"short": after}, "unchanged": integrity_ok}), encoding="utf-8")
    pd.DataFrame({"direction": ["short"], "symbol": ["CCC"],
                  "reason": ["error: RuntimeError('replay window overlaps holdout')" if internal_error else "not trained"]}).to_csv(
        folder / "skipped.csv", index=False)
    if no_rows:
        pd.DataFrame(columns=["direction", "symbol", "holdout_end", "trades", "rho_signed", "rho_abs"]).to_csv(
            folder / "replay_per_symbol.csv", index=False)
    pd.DataFrame({"direction": ["short"], "symbol": ["AAA"], "status": ["PASS" if selftest_ok else "FAIL"]}).to_csv(
        folder / "selftest.csv", index=False)


def run_verify(folder: Path):
    return subprocess.run([sys.executable, str(HERE / "verify_replay.py"), str(folder)], capture_output=True, text=True)


class Verifier(unittest.TestCase):
    def check(self, **flaw):
        with tempfile.TemporaryDirectory() as tmp:
            build_fixture(Path(tmp), **flaw)
            return run_verify(Path(tmp))

    def test_clean_fixture_passes(self):
        r = self.check()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_internal_errors_are_caught(self):
        r = self.check(internal_error=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAIL  no symbol-direction ended in an internal error", r.stdout)

    def test_empty_results_are_caught(self):
        r = self.check(no_rows=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAIL  at least one symbol-direction was scored", r.stdout)

    def test_out_of_range_correlation_is_caught(self):
        r = self.check(bad_rho=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAIL  short: rho_abs", r.stdout)

    def test_overlap_with_holdout_is_caught(self):
        r = self.check(overlap=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAIL  all trades enter strictly after the holdout end", r.stdout)

    def test_tampered_summary_is_caught(self):
        r = self.check(tamper_summary=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("pooled trades / PF / net P&L", r.stdout)

    def test_changed_production_files_are_caught(self):
        r = self.check(integrity_ok=False)
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAIL  models/ calibration", r.stdout)

    def test_failed_selftest_is_caught(self):
        r = self.check(selftest_ok=False)
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAIL  self-test", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
