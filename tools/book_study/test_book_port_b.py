import os, tempfile, unittest
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import book_port_b as B


class T(unittest.TestCase):
    def test_get_train_times_three_overlap_cases(self):
        idx = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06", "2020-01-08", "2020-01-10", "2020-01-02"])
        ends = pd.to_datetime(["2020-01-02", "2020-01-05", "2020-01-07", "2020-01-12", "2020-01-11", "2020-01-20"])
        t1 = pd.Series(ends, index=idx).sort_index()
        test = pd.Series([pd.Timestamp("2020-01-09")], index=[pd.Timestamp("2020-01-06")])
        trn = B.getTrainTimes(t1, test)
        # test span 6..9: obs starting 6 and 8 start inside; obs (1-2 -> ends 20) envelops; obs 10-11 starts after and stays;  3-5 stays; 1-2 stays
        self.assertEqual(sorted(d.strftime("%m-%d") for d in trn.index), ["01-01", "01-03", "01-10"])

    def test_embargo_times(self):
        times = pd.date_range("2020-01-01", periods=10)
        m = B.getEmbargoTimes(times, 0.2)
        self.assertEqual(m.iloc[0], times[2])
        self.assertEqual(m.iloc[-1], times[-1])
        self.assertEqual(m.iloc[-2], times[-1])
        m0 = B.getEmbargoTimes(times, 0.0)
        self.assertTrue((m0.index == m0.values).all())
        tr, te = B.embargo_usage(m, pd.Series(times + pd.Timedelta(days=1), index=times), times[3], times[4])
        self.assertEqual(len(te), 2)

    def test_purged_kfold_has_no_leak_and_covers_tests(self):
        n = 200
        idx = pd.date_range("2020-01-01", periods=n)
        t1 = pd.Series(idx + pd.Timedelta(days=4), index=idx)
        X = pd.DataFrame(np.zeros((n, 2)), index=idx)
        cv = B.PurgedKFold(n_splits=5, t1=t1, pctEmbargo=0.02)
        tests = []
        for tr, te in cv.split(X):
            tests += list(te)
            t0, tmax = idx[te[0]], t1.iloc[te].max()
            s, e = t1.index[tr], t1.iloc[tr]
            self.assertFalse(((s <= tmax) & (e > t0)).any())                 # no training label overlaps the test span (the printed class lets a label end exactly AT the first test time: `t1<=t0`)
            self.assertGreater(len(tr), 100)
        self.assertEqual(sorted(tests), list(range(n)))
        with self.assertRaises(ValueError):
            B.PurgedKFold(3, t1=None)

    def test_mdi_mda_sfi_find_the_informative_features(self):
        np.random.seed(0)
        X, cont = B.getTestData(n_features=8, n_informative=3, n_redundant=2, n_samples=900)
        self.assertEqual(list(X.columns), ["I_0", "I_1", "I_2", "R_0", "R_1", "N_0", "N_1", "N_2"])
        # getTestData (shuffle=False, as printed) returns the rows grouped by cluster, so contiguous CV folds hold out whole clusters (see the next test).
        # Here the ROWS are permuted (the index, w and t1 stay in time order) so that each fold sees every cluster.
        perm = np.random.default_rng(0).permutation(len(X))
        X = X.copy(); X.iloc[:] = X.values[perm]
        cont = cont.copy(); cont["bin"] = cont["bin"].values[perm]
        for method in ("MDI", "MDA", "SFI"):
            imp, oob, oos = B.featImportance(X, cont, n_estimators=30, cv=4, numThreads=1 if method != "SFI" else 2, pctEmbargo=0.01, method=method, scoring="accuracy")
            m = imp["mean"].astype(float)
            inf = m[[c for c in m.index if c[0] in "IR"]].mean()
            noise = m[[c for c in m.index if c[0] == "N"]].mean()
            self.assertGreater(inf, noise, method)
            self.assertTrue(0.5 < oob <= 1 and 0.4 < oos <= 1, (method, oob, oos))

    def test_printed_test_data_is_grouped_by_cluster_so_contiguous_folds_score_below_chance(self):
        """Recorded behaviour of the printed getTestData (shuffle=False): every contiguous fold is one cluster, and the other clusters of the training part carry the opposite label there."""
        np.random.seed(0)
        X, cont = B.getTestData(n_features=8, n_informative=3, n_redundant=2, n_samples=900)
        blocks = [cont["bin"].values[i:i + 225].mean() for i in range(0, 900, 225)]
        self.assertTrue(blocks[0] < .05 and blocks[1] > .95 and blocks[2] < .05 and blocks[3] > .95, blocks)
        clf = B.BaggingClassifier(estimator=B.DecisionTreeClassifier(criterion="entropy", max_features=1, class_weight="balanced"), n_estimators=30, max_features=1., max_samples=900)
        sc = B.cvScore(clf, X, cont["bin"], cont["w"], scoring="accuracy", cv=4, t1=cont["t1"], pctEmbargo=.01)
        self.assertLess(sc.max(), .1, sc)

    def test_orthogonal_features_are_uncorrelated(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=(500, 3))
        X = pd.DataFrame(z @ rng.normal(size=(3, 6)) + 0.05 * rng.normal(size=(500, 6)), columns=list("abcdef"))
        P = B.orthoFeats(X, varThres=.95)
        self.assertLessEqual(P.shape[1], 4)
        c = np.corrcoef(P.T)
        np.testing.assert_allclose(c - np.diag(np.diag(c)), 0, atol=1e-8)
        eVal, eVec = B.get_eVec(pd.DataFrame(np.cov(X.T), index=X.columns, columns=X.columns), 0.5)
        self.assertEqual(eVec.shape[0], 6)
        self.assertTrue((np.diff(eVal.values) <= 0).all())

    def test_weighted_tau_example(self):
        from scipy.stats import weightedtau
        t = B.weighted_tau_example()
        self.assertAlmostEqual(t, weightedtau(np.array([.55, .33, .07, .05]), np.array([1, 2, 4, 3]) ** -1.)[0])
        self.assertGreater(t, 0.5)

    def test_log_uniform_passes_its_own_ks_test(self):
        ks, desc = B.test_logUniform(seed=3)
        self.assertGreater(ks.pvalue, 0.01)
        self.assertGreaterEqual(desc["min"], 1e-3)
        self.assertLessEqual(desc["max"], 1e3)
        v = B.logUniform(a=1e-2, b=1e2).rvs(size=20000, random_state=1)
        self.assertAlmostEqual(np.median(np.log(v)), 0.0, delta=0.1)

    def test_clf_hyper_fit_grid_and_randomized(self):
        np.random.seed(0)
        X, cont = B.getTestData(n_features=6, n_informative=3, n_redundant=1, n_samples=300)
        pipe = B.MyPipeline([("clf", DecisionTreeClassifier(criterion="entropy", class_weight="balanced"))])
        w = cont["w"].values
        g = B.clfHyperFit(X, cont["bin"], cont["t1"], pipe, {"clf__max_depth": [2, 4]}, cv=3, n_jobs=1, **{"clf__sample_weight": w})
        self.assertIn(g.named_steps["clf"].max_depth, (2, 4))
        r = B.clfHyperFit(X, cont["bin"], cont["t1"], pipe, {"clf__min_samples_leaf": loguniform(1e-3, 0.3)}, cv=3, rndSearchIter=3, n_jobs=1, bagging=[5, 0.7, 1.], **{"clf__sample_weight": w})
        self.assertTrue(hasattr(r, "predict_proba"))

    def test_testfunc_smoke(self):
        np.random.seed(0)
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd(); os.chdir(d); os.makedirs("testFunc")   # the book writes to ./testFunc/, which must exist
            try:
                self.assertIsNone(B.testFunc(n_features=8, n_informative=3, n_redundant=2, n_estimators=20, n_samples=500, cv=3))
            finally:
                os.chdir(cwd)
            out = pd.read_csv(os.path.join(d, "testFunc", "stats.csv"))
            self.assertEqual(list(out["method"]), ["MDA", "MDI", "SFI"])
            self.assertTrue(os.path.exists(os.path.join(d, "testFunc", "featImportance_MDI_accuracy_0.00_1.0.png")))


if __name__ == "__main__":
    unittest.main()
