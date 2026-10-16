#!/usr/bin/env python
"""Tests for hrp_alloc.py, run_hrp.py and verify_hrp.py.

Unit tests (synthetic): single linkage against scipy (heights and leaf order), HRP properties (weights sum to one; two assets = inverse variance; diagonal covariance = inverse variance; scale
invariance; a four-asset case worked by hand), long-only minimum variance (two assets by hand; a case where one name must be dropped, checked against a brute-force grid; the optimality
check rejects non-optimal weights), the daily portfolio rule (weights rescaled over the symbols with a bar that day) and the Sharpe / window / year metrics.
Synthetic end-to-end test: builds a small run folder (candles, events, summary, walk-forward predictions for two seeds) with an eligible group of symbols, one late-listed symbol and one thin
symbol, runs run_hrp.py and then verify_hrp.py on it, checks the eligibility decisions, and that a doctored weights file is rejected.
Optional pipeline test: HRP_TEST_RUN = absolute path of a real 14-feature run folder (runs run_hrp.py on it and the checker).

    python -m unittest tools/book_study/test_hrp.py -v
"""
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hrp_alloc as HA  # noqa: E402
import run_hrp as RH  # noqa: E402

DAY = 86400


def random_cov(n, seed):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((2, 3 * n))
    a = rng.standard_normal((n, 3 * n)) + rng.random((n, 1)) * f[0] + rng.random((n, 1)) * f[1]
    return np.cov(a)


class TestLinkage(unittest.TestCase):
    def test_heights_and_order_equal_scipy(self):
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        same_order = 0
        for t in range(60):
            n = 2 + t % 12
            D = HA.distance_of_distances(HA.corr_from_cov(random_cov(n, t)))
            m = HA.single_linkage(D)
            Z = linkage(squareform(D, checks=False), method="single")
            np.testing.assert_allclose([x[2] for x in m], Z[:, 2], rtol=1e-12, atol=1e-12)
            self.assertEqual([x[3] for x in m], [int(v) for v in Z[:, 3]])
            o, o2 = HA.leaf_order(m, n), [int(v) for v in leaves_list(Z)]
            same_order += o == o2
            self.assertTrue(o == o2 or o == o2[::-1], (t, o, o2))             # same dendrogram, possibly mirrored at the root
        self.assertGreaterEqual(same_order, 50)

    def test_child_rule_lower_id_first(self):
        D = HA.distance_of_distances(HA.corr_from_cov(random_cov(8, 4)))
        for a, b, h, s in HA.single_linkage(D):
            self.assertLess(a, b)

    def test_tie_break_is_deterministic(self):
        D = np.ones((4, 4)) - np.eye(4)                                          # all distances equal
        m = HA.single_linkage(D)
        self.assertEqual([(x[0], x[1]) for x in m], [(0, 1), (2, 3), (4, 5)])          # rule: smallest distance, then smaller lower id, then smaller higher id
        self.assertEqual(HA.leaf_order(m, 4), [0, 1, 2, 3])


class TestHRP(unittest.TestCase):
    def test_sum_one_positive(self):
        for t in range(20):
            w, order, m = HA.hrp_weights(random_cov(2 + t % 10, t))
            self.assertAlmostEqual(w.sum(), 1.0, places=12)
            self.assertTrue((w > 0).all())
            self.assertEqual(sorted(order), list(range(len(w))))

    def test_two_assets_equal_inverse_variance(self):
        cov = np.array([[1.0, 1.5], [1.5, 4.0]])
        w, _, _ = HA.hrp_weights(cov)
        np.testing.assert_allclose(w, [0.8, 0.2], atol=1e-12)
        np.testing.assert_allclose(w, HA.inverse_variance(cov), atol=1e-12)

    def test_diagonal_covariance_equals_inverse_variance(self):
        cov = np.diag([1.0, 2.0, 4.0, 5.0, 9.0, 0.5, 3.0])
        w, _, _ = HA.hrp_weights(cov)
        np.testing.assert_allclose(w, HA.inverse_variance(cov), atol=1e-12)

    def test_scale_invariance(self):
        cov = random_cov(7, 3)
        w1, o1, _ = HA.hrp_weights(cov)
        w2, o2, _ = HA.hrp_weights(cov * 1e6)
        np.testing.assert_allclose(w1, w2, atol=1e-12)
        self.assertEqual(o1, o2)

    def test_four_assets_by_hand(self):
        # A,B correlated 0.9 (sd 1 and 2), C,D correlated 0.9 (sd 1 and 1), nothing across the pairs
        cov = np.array([[1.0, 1.8, 0, 0], [1.8, 4.0, 0, 0], [0, 0, 1.0, 0.9], [0, 0, 0.9, 1.0]])
        # cov[A,B] = 0.9 * 1 * 2 = 1.8 ; variance of the pair (A, B) with inverse-variance weights (0.8, 0.2): 0.64 + 0.16 + 2 * 0.8 * 0.2 * 1.8 = 1.376
        # (C, D): weights (0.5, 0.5): 0.25 + 0.25 + 2 * 0.25 * 0.9 = 0.95
        a_ab = 1.0 - 1.376 / (1.376 + 0.95)
        w, order, m = HA.hrp_weights(cov)
        want = np.array([a_ab * 0.8, a_ab * 0.2, (1 - a_ab) * 0.5, (1 - a_ab) * 0.5])
        np.testing.assert_allclose(w, want, atol=1e-12)
        self.assertIn(set(order[:2]), ({0, 1}, {2, 3}))                            # the pairs stay together in the leaf order
        self.assertLess(w[:2].sum(), 0.5)                                          # the pair with the larger variance (1.376 against 0.95) gets the smaller share

    def test_three_assets_split_by_hand(self):
        # A alone (variance 2, uncorrelated), B and C correlated 0.9 (variances 1 and 4, covariance 1.8).  B and C merge first, so the leaf order is A, B, C and the first cut is
        # [A] against [B, C] (floor(3 / 2) = 1 item first).  V(A) = 2, V(B, C) = 1.376, so A gets 1 - 2 / (2 + 1.376); inside (B, C) the inverse-variance split is (0.8, 0.2).
        cov = np.array([[2.0, 0, 0], [0, 1.0, 1.8], [0, 1.8, 4.0]])
        w, order, m = HA.hrp_weights(cov)
        self.assertEqual(order, [0, 1, 2])
        a = 1.0 - 2.0 / (2.0 + 1.376)
        np.testing.assert_allclose(w, [a, (1 - a) * 0.8, (1 - a) * 0.2], atol=1e-12)

    def test_inverse_variance_and_equal_weight(self):
        cov = np.diag([1.0, 4.0])
        np.testing.assert_allclose(HA.inverse_variance(cov), [0.8, 0.2])
        np.testing.assert_allclose(HA.equal_weight(4), [0.25] * 4)

    def test_cluster_variance_hand(self):
        cov = np.array([[1.0, 1.8], [1.8, 4.0]])
        self.assertAlmostEqual(HA.cluster_variance(cov, [0, 1]), 1.376, places=12)
        self.assertAlmostEqual(HA.cluster_variance(cov, [1]), 4.0, places=12)


class TestMinVariance(unittest.TestCase):
    def test_two_uncorrelated(self):
        w = HA.min_variance_long_only(np.diag([1.0, 4.0]))
        np.testing.assert_allclose(w, [0.8, 0.2], atol=1e-6)

    def test_dropped_name_matches_grid(self):
        cov = np.array([[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 2.0]])
        cov[1, 1] = 1.2
        cov[0, 1] = cov[1, 0] = 1.0                                              # the unconstrained solution would short asset 1
        w = HA.min_variance_long_only(cov)
        best, bw = 1e9, None
        for a in np.linspace(0, 1, 201):
            for b in np.linspace(0, 1 - a, int(round((1 - a) * 200)) + 1):
                x = np.array([a, b, 1 - a - b])
                v = x @ cov @ x
                if v < best:
                    best, bw = v, x
        self.assertLessEqual(w @ cov @ w, best + 1e-9)
        self.assertAlmostEqual(w[1], 0.0, places=6)
        np.testing.assert_allclose(w, bw, atol=6e-3)

    def test_kkt_rejects_non_optimal_and_accepts_optimal(self):
        cov = random_cov(6, 11)
        w = HA.min_variance_long_only(cov)
        self.assertLess(HA.kkt_gap(cov, w), 1e-4)
        self.assertGreater(HA.kkt_gap(cov, HA.equal_weight(6)), 1e-2)
        self.assertGreater(HA.kkt_gap(cov, np.array([1.0, 0, 0, 0, 0, 0])), 1e-2)

    def test_not_worse_than_other_schemes(self):
        for t in range(5):
            cov = random_cov(8, 20 + t)
            w = HA.min_variance_long_only(cov)
            for other in (HA.equal_weight(8), HA.inverse_variance(cov), HA.hrp_weights(cov)[0]):
                self.assertLessEqual(w @ cov @ w, other @ cov @ other + 1e-9)


class TestPortfolio(unittest.TestCase):
    def test_rescale_over_symbols_with_a_bar(self):
        W = np.array([0.5, 0.5])
        days = [np.array([10, 11, 12]), np.array([11, 12, 13])]
        net = [np.array([0.01, 0.02, 0.03]), np.array([0.04, 0.05, 0.06])]
        d, bp = RH.portfolio_series(W, days, net, 10, 13)
        np.testing.assert_array_equal(d, [10, 11, 12, 13])
        # day 10 only symbol 0 (weight rescaled to 1): 0.01; day 11 both: 0.5*0.02 + 0.5*0.04 = 0.03; day 12: 0.5*0.03 + 0.5*0.05 = 0.04; day 13 only symbol 1: 0.06
        np.testing.assert_allclose(bp, [100.0, 300.0, 400.0, 600.0], atol=1e-9)

    def test_day_without_any_bar_is_left_out_and_zero_weight_gives_zero(self):
        d, bp = RH.portfolio_series(np.array([1.0, 0.0]), [np.array([10, 12]), np.array([10, 11, 12])], [np.array([0.01, 0.02]), np.array([0.5, 0.5, 0.5])], 10, 13)
        np.testing.assert_array_equal(d, [10, 11, 12])
        np.testing.assert_allclose(bp, [100.0, 0.0, 200.0], atol=1e-9)

    def test_metrics_windows_and_years(self):
        d0 = int(np.datetime64("2024-12-30").astype("datetime64[D]").astype(int))
        days = np.arange(d0, d0 + 6)                                             # 2024-12-30, 31, 2025-01-01..04
        bp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        w = {"w": (int((d0 + 2) * DAY), int((d0 + 4) * DAY))}
        m = RH.metrics(days, bp, w)
        self.assertEqual(m["n_days"], 6)
        self.assertAlmostEqual(m["mean_bp"], 3.5)
        self.assertAlmostEqual(m["sd_bp"], np.std(bp, ddof=1))
        self.assertAlmostEqual(m["win_w_bp"], 3.5)
        self.assertEqual(m["win_w_days"], 2)
        self.assertEqual((m["year_2024_days"], m["year_2025_days"]), (2, 4))
        self.assertAlmostEqual(m["year_2025_mean_bp"], 4.5)
        self.assertTrue(math.isnan(m["year_2024_sr_annual"]))                    # fewer than 4 days
        self.assertAlmostEqual(m["year_2025_sr_annual"], 4.5 / np.std([3, 4, 5, 6], ddof=1) * math.sqrt(365))
        self.assertTrue(math.isnan(RH.sharpe_annual([1.0, 1.0, 1.0, 1.0])))      # zero variance


def build_run(root: Path, seeds=(101, 202)):
    """Synthetic run folder: 8 symbols of random-walk hourly candles.  A..D cover the estimation window and also have one event that straddles the cutoff (starts before it, ends after);
    LATE starts in 2023-03; THIN has 15 events before the cutoff; EDGE100 and EDGE99 have exactly 100 and 99 one-bar events on distinct days of the window (so exactly 100 / 99 active
    days); every symbol trades after the cutoff."""
    rng = np.random.default_rng(7)
    cd = root / "candles"
    cd.mkdir()
    t_start = int(np.datetime64("2021-12-28T00:00").astype("datetime64[s]").astype(np.int64))
    t_end = int(np.datetime64("2024-03-01T00:00").astype("datetime64[s]").astype(np.int64))
    t_late = int(np.datetime64("2023-03-01T00:00").astype("datetime64[s]").astype(np.int64))
    common = rng.standard_normal(((t_end - t_start) // 3600) + 1) * 0.004
    bars, rows = {}, []
    for k, s in enumerate(["A", "B", "C", "D", "LATE", "THIN", "EDGE100", "EDGE99"]):
        t0_ = t_late if s == "LATE" else t_start
        ts = np.arange(t0_, t_end + 1, 3600, dtype=np.int64)
        if s == "B":                                                              # a gap of some days
            keep = ~((ts > t_start + 400 * DAY) & (ts < t_start + 404 * DAY))
            ts = ts[keep]
        r = 0.6 * common[(ts - t_start) // 3600] + rng.standard_normal(len(ts)) * 0.006 * (1 + k / 4)
        cl = 100.0 * np.exp(np.cumsum(r))
        bars[s] = (ts, cl)
        with open(cd / f"{s}_1h.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            for t, c in zip(ts, cl):
                w.writerow([np.datetime64(int(t), "s").astype(str).replace("T", " "), c, c, c, c, 1.0])
    cut = int(np.datetime64("2024-01-01T00:00").astype("datetime64[s]").astype(np.int64))
    ev = []
    for s, (ts, cl) in bars.items():
        n = len(ts)
        pre_n = 15 if s == "THIN" else 260
        post_n = 120
        if s.startswith("EDGE"):
            n_edge = 100 if s == "EDGE100" else 99
            day_starts = t_start + (np.arange(6, 6 + 7 * n_edge, 7)) * DAY                  # every 7th day from 2022-01-03: inside the window, none after 2023-12-29
            pre_pos = (day_starts - t_start) // 3600 + 5
            holds = {int(p_): 1 for p_ in pre_pos}
        else:
            pre_pos = np.sort(rng.choice(np.flatnonzero(ts < cut - 60 * 3600)[:-1], pre_n, replace=False))
            holds = {}
        post_pos = np.sort(rng.choice(np.flatnonzero((ts >= cut) & (np.arange(n) < n - 60)), post_n, replace=False))
        straddle = [int(np.flatnonzero(ts >= cut)[0]) - 20] if s in ("A", "B", "C", "D") else []
        for pos in list(pre_pos) + straddle + list(post_pos):
            h = holds.get(int(pos), 40 if pos in straddle else int(rng.integers(10, 50)))
            t1p = min(pos + h, n - 1)
            side = float(rng.choice([-1, 1]))
            ev.append({"symbol": s, "pos": int(pos), "t0": int(ts[pos]), "t1": int(ts[t1p]), "t1pos": int(t1p), "side": side, "ret_side": side * (cl[t1p] / cl[pos] - 1.0)})
    ev.sort(key=lambda e: (e["t0"], e["symbol"]))
    runs = []
    for si, seed in enumerate(seeds):
        r = root / f"run{seed}"
        r.mkdir()
        with open(r / "events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ev[0].keys()))
            w.writeheader()
            w.writerows(ev)
        j = {"params": {"cutoff": "2024-01-01", "step": 0.1, "cost_frac": 0.0024, "seed": seed},
             "meta": {"candle_dir": str(cd), "timeframe": "1h",
                      "window_epoch": {"win_a": [cut + 20 * DAY, cut + 50 * DAY], "old": [cut - 90 * DAY, cut - 30 * DAY]}}}
        (r / "summary_numbers.json").write_text(json.dumps(j), encoding="utf-8")
        g = np.random.default_rng(seed)
        with open(r / "walk_forward_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event", "p"])
            for i, e in enumerate(ev):
                if e["t0"] >= cut:
                    w.writerow([i, float(np.clip(0.5 + 0.12 * g.standard_normal(), 0.02, 0.98))])
        runs.append(r)
    return runs


class TestSyntheticEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        root = Path(cls.td.name)
        cls.runs = build_run(root)
        cls.out = root / "hrp"
        r = subprocess.run([sys.executable, str(HERE / "run_hrp.py"), "--run"] + [str(x) for x in cls.runs] + ["--out", str(cls.out)], capture_output=True, text=True)
        cls.rc, cls.log = r.returncode, r.stdout[-2000:] + r.stderr[-3000:]

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_runner_ran(self):
        self.assertEqual(self.rc, 0, self.log)

    def test_eligibility_decisions(self):
        with open(self.out / "eligibility.csv", newline="") as f:
            el = {r["symbol"]: r for r in csv.DictReader(f)}
        self.assertEqual({s for s, r in el.items() if r["eligible"] == "1"}, {"A", "B", "C", "D", "EDGE100"})
        self.assertEqual(el["EDGE100"]["active_days_in_window"], "100")
        self.assertEqual(el["EDGE99"]["active_days_in_window"], "99")
        self.assertIn("fewer than 100", el["EDGE99"]["reason"])
        self.assertEqual(el["A"]["events_pre_cutoff"], "260")                    # the straddling event (label ends after the cutoff) is not an estimation event
        self.assertEqual(el["A"]["events_post_cutoff"], "120")
        self.assertIn("do not cover", el["LATE"]["reason"])
        self.assertIn("fewer than 100", el["THIN"]["reason"])
        ex = json.loads((self.out / "hrp_extra.json").read_text())
        self.assertEqual(ex["events_post_cutoff_dropped"], 360)                  # LATE, THIN and EDGE99 trade 120 events each after the cutoff
        self.assertEqual(ex["eligible"], ["A", "B", "C", "D", "EDGE100"])

    def test_weights_valid(self):
        with open(self.out / "weights.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for sch in RH.SCHEMES:
            v = [float(r[sch]) for r in rows]
            self.assertAlmostEqual(sum(v), 1.0, places=9)
            self.assertGreaterEqual(min(v), 0.0)

    def test_checker_passes_and_rejects_a_doctored_weight(self):
        c = subprocess.run([sys.executable, str(HERE / "verify_hrp.py"), str(self.out)], capture_output=True, text=True)
        self.assertEqual(c.returncode, 0, c.stdout[-3000:])
        self.assertIn("12/12 checks passed", c.stdout)
        with tempfile.TemporaryDirectory() as td2:
            bad = Path(td2) / "bad"
            import shutil
            shutil.copytree(self.out, bad)
            with open(bad / "weights.csv", newline="") as f0:
                rows = list(csv.DictReader(f0))
            rows[0]["hrp"] = repr(float(rows[0]["hrp"]) * 1.2)
            with open(bad / "weights.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            c2 = subprocess.run([sys.executable, str(HERE / "verify_hrp.py"), str(bad)], capture_output=True, text=True)
            self.assertNotEqual(c2.returncode, 0)

    def test_duplicate_seed_refused(self):
        r = subprocess.run([sys.executable, str(HERE / "run_hrp.py"), "--run", str(self.runs[0]), str(self.runs[0]), "--out", str(Path(self.td.name) / "dup")], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("same seed", r.stdout + r.stderr)

    def test_post_cutoff_only_uses_estimation_window_weights(self):
        # the weights file does not change when the post-cutoff predictions change: rerun with a different second seed's predictions
        with tempfile.TemporaryDirectory() as td:
            runs = build_run(Path(td), seeds=(101, 303))
            out = Path(td) / "hrp"
            r = subprocess.run([sys.executable, str(HERE / "run_hrp.py"), "--run"] + [str(x) for x in runs] + ["--out", str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertEqual((out / "weights.csv").read_text(), (self.out / "weights.csv").read_text())
            self.assertNotEqual((out / "metrics.csv").read_text(), (self.out / "metrics.csv").read_text())


@unittest.skipUnless(os.environ.get("HRP_TEST_RUN"), "set HRP_TEST_RUN to the absolute path of a real 14-feature run folder")
class TestRealRun(unittest.TestCase):
    def test_real_run_and_checker(self):
        run = os.environ["HRP_TEST_RUN"]
        self.assertTrue(os.path.isabs(run))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "hrp"
            r = subprocess.run([sys.executable, str(HERE / "run_hrp.py"), "--run", run, "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            c = subprocess.run([sys.executable, str(HERE / "verify_hrp.py"), str(out)], capture_output=True, text=True)
            self.assertEqual(c.returncode, 0, c.stdout[-3000:])


if __name__ == "__main__":
    unittest.main()
