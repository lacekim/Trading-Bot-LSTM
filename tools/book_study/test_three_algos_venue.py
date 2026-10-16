"""Tests for run_three_algos_venue.py and verify_three_algos_venue.py (stdlib unittest; no real data needed).

Optional real-run test: set THREE_ALGO_VENUE_RUN to a finished runtime/book_three_algo_kraken/<timestamp> folder.
"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_three_algos_venue as RV
import venue_loader as VL
import verify_three_algos_venue as VT

HERE = Path(__file__).resolve().parent


def write_candles(d: Path, sym: str, index, close):
    pd.DataFrame({"date": [t.strftime("%Y-%m-%d %H:%M:%S") for t in index], "open": close, "high": np.asarray(close) * 1.001, "low": np.asarray(close) * 0.999, "close": close, "volume": 1.0}
                 ).to_csv(d / f"{sym}_1h.csv", index=False)


def gapped_index(n=1100, gap_after=500, gap_hours=100):
    base = pd.Timestamp("2024-01-01 00:00:00")
    return pd.DatetimeIndex([base + pd.Timedelta(hours=i + (gap_hours if i > gap_after else 0)) for i in range(n)])


class TestLoaderShim(unittest.TestCase):
    def test_shim_equals_venue_loader_and_ignores_second_folder(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            idx = gapped_index(50)
            write_candles(d, "AAA", idx, np.linspace(1, 2, 50))
            write_candles(d, "BBB", idx, np.linspace(3, 2, 50))
            got, sk = RV._VenueCandles.load_closes(d, "/nonexistent/other/folder", ["AAA", "BBB"], "1h", 0)
            ref, sk_ref = VL.load_closes(d, ["AAA", "BBB"], "1h", 0)
            self.assertEqual(sk, sk_ref)
            for s in ("AAA", "BBB"):
                pd.testing.assert_series_equal(got[s], ref[s])
            self.assertEqual(len(got["AAA"]), 50)
            self.assertEqual(got["AAA"].index[-1], idx[-1])


class TestSpanDiagnostic(unittest.TestCase):
    def setUp(self):
        self.idx = gapped_index()
        self.closes = {"AAA": pd.Series(np.ones(len(self.idx)), index=self.idx)}
        pos = [470, 490, 520, 1000, 1050]
        ts = self.idx.astype("datetime64[s]").astype(np.int64)
        self.df = pd.DataFrame({"symbol": "AAA", "pos": pos, "t0": [float(ts[p]) for p in pos]})
        self.cut = str(self.idx[760])                  # between pos 520 and pos 1000

    def test_hand_counts(self):
        out = RV.span_diagnostic(self.df, self.closes, self.cut)
        # the 100-hour gap lies between bar 500 and bar 501
        want = {"480": (4, 1, 2, 0), "720": (2, 2, 2, 2), "960": (2, 2, 2, 2)}
        for k, (n, off, npost, offpost) in want.items():
            r = out[k]
            self.assertEqual((r["events_with_lookback"], r["events_span_off"], r["events_post_cutoff"], r["events_post_cutoff_span_off"]), (n, off, npost, offpost), k)
        self.assertAlmostEqual(out["480"]["share_off"], 0.25)
        self.assertAlmostEqual(out["720"]["share_off_post_cutoff"], 1.0)

    def test_no_gap_gives_zero(self):
        idx = pd.date_range("2024-01-01", periods=1100, freq="h")
        closes = {"AAA": pd.Series(np.ones(1100), index=idx)}
        out = RV.span_diagnostic(self.df, closes, self.cut)
        self.assertTrue(all(v["events_span_off"] == 0 for v in out.values()))

    def test_brute_force_agreement(self):
        out = RV.span_diagnostic(self.df, self.closes, self.cut)
        ts = [t.timestamp() for t in self.idx]
        cut = pd.Timestamp(self.cut).timestamp()
        for L in (480, 720, 960):
            n = off = 0
            for p, t0 in zip(self.df["pos"], self.df["t0"]):
                if p >= L:
                    n += 1
                    off += abs((ts[p] - ts[p - L]) / 3600.0 / L - 1.0) > RV.SPAN_TOLERANCE
            self.assertEqual((out[str(L)]["events_with_lookback"], out[str(L)]["events_span_off"]), (n, off))


class TestCheckerSetup(unittest.TestCase):
    def test_constants_agree(self):
        self.assertEqual(RV.PRIOR_TRIALS_VENUE, 115)
        self.assertEqual(VT.PRIOR, RV.PRIOR_TRIALS_VENUE)

    def test_every_substitution_occurs_exactly_once(self):
        src = (HERE / "verify_three_algos.py").read_text(encoding="utf-8")
        for old, new in VT.SUBSTITUTIONS:
            self.assertEqual(src.count(old), 1, old)
            self.assertNotEqual(old, new)

    def test_patched_module_loads_with_new_trial_text(self):
        mod, md5 = VT.load_patched()
        self.assertEqual(md5, hashlib.md5((HERE / "verify_three_algos.py").read_bytes()).hexdigest())
        self.assertTrue(callable(mod.main))

    def test_changed_original_is_refused(self):
        orig = VT.SUBSTITUTIONS
        try:
            VT.SUBSTITUTIONS = orig + [("this text is not in the file", "x")]
            with self.assertRaises(SystemExit):
                VT.load_patched()
        finally:
            VT.SUBSTITUTIONS = orig

    def test_non_kraken_source_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "summary_numbers.json").write_text(json.dumps({"meta": {"venue": "gmx"}, "params": {}}), encoding="utf-8")
            import sys
            old = sys.argv
            sys.argv = ["x", "--source", str(d), "--out", str(d / "out")]
            try:
                with self.assertRaises(SystemExit) as cm:
                    RV.main()
                self.assertIn("not 'kraken'", str(cm.exception))
            finally:
                sys.argv = old


@unittest.skipUnless(os.environ.get("THREE_ALGO_VENUE_RUN"), "set THREE_ALGO_VENUE_RUN to a finished run folder")
class TestRealRun(unittest.TestCase):
    def test_checker_passes_on_real_run(self):
        self.assertEqual(VT.main(os.environ["THREE_ALGO_VENUE_RUN"], os.environ.get("THREE_ALGO_VENUE_SOURCE")), 0)


if __name__ == "__main__":
    unittest.main()
