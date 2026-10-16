import os, tempfile, unittest
import numpy as np
import pandas as pd
from scipy.stats import binom

import book_mp as MP
import book_port_a as A


def _sq(x):
    return x * x


def mol_func(molecule, scale):
    return pd.Series([m * scale for m in molecule], index=list(molecule), dtype=float)


def frame_func(molecule, scale):
    return pd.DataFrame({"v": [m * scale for m in molecule]}, index=list(molecule))


class T(unittest.TestCase):
    # ---- Chapter 2
    def test_pca_weights_hit_the_risk_target_and_distribution(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 5)) @ rng.normal(size=(5, 5))
        cov = np.cov(x.T)
        w = A.pcaWeights(cov)                                       # default: all risk on the smallest-eigenvalue component
        self.assertAlmostEqual(float(np.sqrt((w.T @ cov @ w).item())), 1.0, places=9)
        ev, evec = np.linalg.eigh(cov)
        small = evec[:, 0]
        self.assertAlmostEqual(abs(float(np.dot(w[:, 0], small))) * np.sqrt(ev[0]), 1.0, places=9)
        rd = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        w2 = A.pcaWeights(cov, riskDist=rd, riskTarget=2.0)
        self.assertAlmostEqual(float(np.sqrt((w2.T @ cov @ w2).item())), 2.0, places=9)

    def test_roll_gaps_by_hand(self):
        idx = pd.date_range("2020-01-01", periods=6)
        df = pd.DataFrame({"Symbol": ["A", "A", "A", "B", "B", "B"], "Open": [10, 11, 12, 20, 21, 22.], "Close": [10.5, 11.5, 12.5, 20.5, 21.5, 22.5]}, index=idx)
        g = A.rollGaps(df, dictio={"Instrument": "Symbol", "Open": "Open", "Close": "Close"})
        # roll on day 4: open 20 - previous close 12.5 = 7.5 ; cumulated, then shifted so that the last value is 0 (roll backward)
        self.assertEqual(list(g.round(6)), [-7.5, -7.5, -7.5, 0, 0, 0])
        r = A.rolled_prices(df)
        self.assertAlmostEqual(r["Close"].iloc[2], 12.5 + 7.5, places=9)            # the old contract is shifted up to the new level
        self.assertAlmostEqual(r["Close"].iloc[-1], 22.5, places=9)
        self.assertTrue(np.isfinite(r["rPrices"].iloc[-1]))

    # ---- Chapter 3
    def test_drop_labels(self):
        ev = pd.DataFrame({"bin": [1] * 50 + [-1] * 48 + [0] * 2})
        out = A.dropLabels(ev, minPct=.05)
        self.assertEqual(set(out["bin"]), {1, -1})
        two = pd.DataFrame({"bin": [1] * 99 + [0] * 1})
        self.assertEqual(len(A.dropLabels(two, .05)), 100)                          # fewer than 3 labels: nothing is dropped
        ok = pd.DataFrame({"bin": [1] * 40 + [-1] * 40 + [0] * 20})
        self.assertEqual(len(A.dropLabels(ok, .05)), 100)

    # ---- Chapter 4
    def test_indicator_matrix_is_the_books_matrix_1(self):
        t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
        m = A.getIndMatrix(range(6), t1)
        want = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        np.testing.assert_array_equal(m.to_numpy(), want)
        u = A.getAvgUniqueness(m)
        self.assertAlmostEqual(u[0], (1 + 1 + 0.5) / 3)
        self.assertAlmostEqual(u[1], (0.5 + 1) / 2)
        self.assertAlmostEqual(u[2], 1.0)

    def test_avg_uniqueness_equals_the_projects_fast_version(self):
        rng = np.random.default_rng(1)
        starts = np.sort(rng.integers(0, 60, 25))
        ends = starts + rng.integers(1, 6, 25)
        t1 = pd.Series(ends, index=starts)
        m = A.getIndMatrix(range(int(ends.max()) + 1), t1)
        lit = A.getAvgUniqueness(m).to_numpy()
        # direct definition as the check: mean over the bars of the label of 1 / concurrency
        conc = m.sum(axis=1).to_numpy()
        want = np.array([np.mean(1.0 / conc[s:e + 1]) for s, e in zip(starts, ends)])
        np.testing.assert_allclose(lit, want, rtol=1e-12)

    def test_seq_bootstrap_beats_standard_on_average_and_has_the_right_size(self):
        np.random.seed(3)
        std, seq = [], []
        for _ in range(40):
            t1 = A.getRndT1(10, 100, 5)
            m = A.getIndMatrix(range(int(t1.max()) + 1), t1)
            phi = np.random.choice(m.columns, size=m.shape[1])
            std.append(A.getAvgUniqueness(m[phi]).mean())
            s = A.seqBootstrap(m)
            self.assertEqual(len(s), m.shape[1])
            seq.append(A.getAvgUniqueness(m[s]).mean())
        self.assertGreater(np.mean(seq), np.mean(std))

    def test_rnd_t1_and_example(self):
        np.random.seed(5)
        t1 = A.getRndT1(10, 100, 5)
        self.assertTrue(t1.index.is_monotonic_increasing)
        self.assertTrue(((t1.values - t1.index.values) >= 1).all() and ((t1.values - t1.index.values) < 5).all())
        phi, std, phi2, seq = A.example_seq_bootstrap()
        self.assertEqual(len(phi), 3)
        self.assertEqual(len(phi2), 3)
        self.assertTrue(0 < std <= 1 and 0 < seq <= 1)

    def test_main_mc_single_thread_and_processes_agree_in_shape(self):
        np.random.seed(7)
        d1 = A.mainMC(numObs=6, numBars=40, maxH=4, numIters=12, numThreads=1)
        self.assertEqual(list(d1.columns), ["stdU", "seqU"])
        self.assertEqual(int(d1.loc["count", "stdU"]), 12)
        d2 = A.mainMC(numObs=6, numBars=40, maxH=4, numIters=12, numThreads=2)
        self.assertEqual(int(d2.loc["count", "seqU"]), 12)

    # ---- Chapter 5
    def test_weights_and_expanding_fracdiff(self):
        w = A.getWeights(0.5, 4).ravel()[::-1]
        np.testing.assert_allclose(w, [1, -0.5, -0.125, -0.0625], rtol=1e-12)
        s = pd.DataFrame({"x": np.cumsum(np.random.default_rng(2).normal(size=40))}, index=pd.date_range("2020-01-01", periods=40))
        d1 = A.fracDiff(s, 1.0, thres=1.0)                                          # d=1, nothing skipped: first differences (first value is x[0])
        np.testing.assert_allclose(d1["x"].to_numpy()[1:], np.diff(s["x"].to_numpy()), rtol=1e-9, atol=1e-12)
        d0 = A.fracDiff(s, 0.0, thres=1.0)                                          # d=0: the series itself
        np.testing.assert_allclose(d0["x"].to_numpy(), s["x"].to_numpy(), rtol=1e-12)
        dd = A.fracDiff(s, 0.4, thres=0.05)
        self.assertLess(len(dd), len(s))

    # ---- Chapter 6
    def test_bagging_accuracy_is_the_binomial_tail(self):
        p, acc = A.bagging_accuracy(100, 1. / 3, 3.)
        self.assertAlmostEqual(acc, float(binom.sf(33, 100, 1. / 3)), places=12)
        self.assertAlmostEqual(A.bagging_accuracy(100, 0.55, 2.)[1], float(binom.sf(50, 100, 0.55)), places=12)

    def test_three_setups_build(self):
        c0, c1, c2 = A.three_rf_setups(0.4)
        self.assertEqual(c0.n_estimators, 1000)
        self.assertEqual(c1.max_samples, 0.4)
        self.assertEqual(c2.max_features, 1.0)

    # ---- Chapter 20
    def test_cartesian(self):
        d = {"a": ["1", "2"], "b": ["+", "*"], "c": ["!", "@"]}
        a, b = MP.cartesian_unvectorized(d), MP.cartesian_vectorized(d)
        self.assertEqual(len(a), 8)
        self.assertEqual(sorted(map(str, a)), sorted(map(str, b)))

    def test_barrier_touch(self):
        r = np.random.default_rng(4).normal(0, 0.05, size=(200, 60))
        first = MP.barrierTouch(r, 0.4)
        self.assertEqual(first, MP.barrier_touch_vectorised(r, 0.4))
        last = MP.barrier_touch_as_printed(r, 0.4)
        self.assertEqual(set(last), set(first))
        self.assertTrue(all(last[j] >= first[j] for j in first))
        self.assertTrue(any(last[j] > first[j] for j in first))                       # the printed `continue` really does differ
        self.assertEqual(MP.main1(r, numThreads=3, width=0.4), first)

    def test_partitions(self):
        p = MP.linParts(20, 6)
        self.assertEqual(p[0], 0)
        self.assertEqual(p[-1], 20)
        self.assertEqual(len(p), 7)
        n = MP.nestedParts(20, 6)
        self.assertEqual(n[0], 0)
        self.assertTrue((np.diff(n) > 0).all())
        self.assertAlmostEqual(n[-1], 20, delta=1)
        u = MP.nestedParts(20, 6, upperTriang=True)
        self.assertEqual(u[0], 0)
        self.assertTrue((np.diff(MP.nestedParts(20, 6, upperTriang=True)) > 0).all())

    def test_jobs_and_mpPandasObj(self):
        jobs = [{"func": _sq, "x": i} for i in range(5)]
        self.assertEqual(sorted(MP.processJobs_([dict(j) for j in jobs])), [0, 1, 4, 9, 16])
        self.assertEqual(sorted(MP.processJobs([dict(j) for j in jobs], numThreads=2)), [0, 1, 4, 9, 16])
        atoms = list(range(23))
        for nt, lin in ((1, True), (3, True), (3, False)):
            s = MP.mpPandasObj(mol_func, ("molecule", atoms), numThreads=nt, linMols=lin, scale=2.0)
            self.assertEqual(list(s.index), atoms)
            self.assertEqual(list(s.values), [2.0 * a for a in atoms])
        df = MP.mpPandasObj(frame_func, ("molecule", atoms), numThreads=2, scale=3.0)
        self.assertEqual(list(df["v"]), [3.0 * a for a in atoms])

    def test_redux_and_getpcs(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(8)
            idx = pd.date_range("2020-01-01", periods=10)
            names = []
            full = None
            eVec = pd.DataFrame(rng.normal(size=(3, 2)), index=["a", "b", "c"], columns=["pc1", "pc2"])
            for k in range(5):
                df = pd.DataFrame(rng.normal(size=(10, 3)), index=idx, columns=["a", "b", "c"])
                df.to_csv(os.path.join(d, f"f{k}.csv"))
                names.append(f"f{k}.csv")
                part = df.values @ eVec.loc[df.columns].values
                full = part if full is None else full + part
            pcs = MP.mpJobList(MP.getPCs, ("molecules", names), numThreads=2, path=d + os.sep, eVec=eVec, redux=pd.DataFrame.add)
            np.testing.assert_allclose(pcs.to_numpy(), full, rtol=1e-10, atol=1e-12)
            lst = MP.mpJobList(mol_func, ("molecule", list(range(7))), numThreads=3, scale=1.0)
            self.assertEqual(len(lst), 3)


if __name__ == "__main__":
    unittest.main()
