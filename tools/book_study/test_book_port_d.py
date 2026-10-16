import itertools
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import book_port_d as D

BOOK_COLS = [2, 0, 4, 1, 1]          # from the Figure 16.5 caption: 6<-3, 7<-1, 8<-5, 9<-2, 10<-2 (labels 1-based)
TABLE_16_1 = {                       # book p.146
    'CLA': [14.44, 19.93, 19.73, 19.87, 18.68, 0.00, 5.86, 1.49, 0.00, 0.00],
    'HRP': [7.00, 7.59, 10.84, 19.03, 9.72, 10.19, 6.62, 9.10, 7.12, 12.79],
    'IVP': [10.36, 10.28, 10.36, 10.25, 10.31, 9.74, 9.80, 9.65, 9.64, 9.61]}


class Test16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.cols = D.generateData(10000, 5, 5, .25, cols=BOOK_COLS)
        cls.cov, cls.corr = cls.x.cov(), cls.x.corr()

    def test_clustering_order_matches_figure_16_5(self):
        import scipy.cluster.hierarchy as sch
        link = sch.linkage(D.correlDist(self.corr), 'single')
        self.assertEqual(self.corr.index[D.getQuasiDiag(link)].tolist(), [9, 2, 10, 1, 7, 3, 6, 4, 5, 8])

    def test_table_16_1_all_three_columns(self):
        cov, corr = self.cov, self.corr
        got = {'HRP': D.getHRP(cov, corr).values * 100, 'IVP': D.getIVP(cov) * 100, 'CLA': D.getCLA(cov) * 100}
        for k, ref in TABLE_16_1.items():
            np.testing.assert_allclose(np.round(got[k], 2), ref, atol=0.011, err_msg=k)

    def test_weights_are_proper(self):
        for w in (D.getHRP(self.cov, self.corr).values, D.getIVP(self.cov), D.getCLA(self.cov)):
            self.assertAlmostEqual(w.sum(), 1.)
            self.assertTrue((w >= -1e-12).all())

    def test_python3_default_draw_is_documented(self):
        pairs, hrp = D.main_16_4()
        self.assertEqual(pairs, [(4, 6), (1, 7), (3, 8), (3, 9), (2, 10)])
        self.assertAlmostEqual(hrp.sum(), 1.)

    def test_min_var_long_only_against_general_solver(self):
        rng = np.random.default_rng(0)
        for n in (3, 6, 10):
            for _ in range(5):
                a = rng.normal(size=(n, n + 2)); V = a @ a.T / (n + 2) + np.diag(rng.uniform(.01, .2, n))
                V[0, 1] = V[1, 0] = V[0, 1] + 2.0    # force some negative-weight directions
                V = (V + V.T) / 2 + np.eye(n) * 2.0
                w = D.minVarLongOnly(V)
                res = minimize(lambda z: z @ V @ z, np.ones(n) / n, jac=lambda z: 2 * V @ z, bounds=[(0, 1)] * n,
                               constraints=[{'type': 'eq', 'fun': lambda z: z.sum() - 1}], method='SLSQP', options={'ftol': 1e-14, 'maxiter': 500})
                self.assertLessEqual(w @ V @ w, res.fun + 1e-9)
                self.assertAlmostEqual(w.sum(), 1.)

    def test_recursive_bisection_halves_two_assets_by_inverse_variance(self):
        cov = pd.DataFrame(np.diag([1., 4.]), index=[0, 1], columns=[0, 1])
        w = D.getRecBipart(cov, [0, 1])
        self.assertAlmostEqual(w[0], 4 / 5.)
        self.assertAlmostEqual(w[1], 1 / 5.)

    def test_quasi_diag_is_a_permutation(self):
        import scipy.cluster.hierarchy as sch
        rng = np.random.default_rng(1)
        x = pd.DataFrame(rng.normal(size=(300, 9)))
        link = sch.linkage(D.correlDist(x.corr()), 'single')
        self.assertEqual(sorted(D.getQuasiDiag(link)), list(range(9)))

    def test_hrp_monte_carlo_runs_and_reports(self):
        np.random.seed(1); import random; random.seed(1)
        stats, rep = D.hrpMC(numIters=2, nObs=300, sLength=150, rebal=30)
        self.assertEqual(list(stats.columns), ['getIVP', 'getHRP', 'getCLA'])
        self.assertEqual(len(stats), 2)
        self.assertAlmostEqual(rep.loc['getHRP', 'var_vs_HRP'], 0.)


class Test21(unittest.TestCase):
    def test_pigeonhole_counts(self):
        parts = list(D.pigeonHole(3, 3))
        self.assertEqual(len(parts), 10)                        # C(5,3)
        self.assertTrue(all(sum(p) == 3 and len(p) == 3 for p in parts))
        self.assertIn([1, 2, 0], parts); self.assertIn([2, 1, 0], parts)   # different partitions (Figure 21.1)

    def test_all_weights(self):
        w = D.getAllWeights(2, 2)
        self.assertEqual(w.shape, (2, 12))
        np.testing.assert_allclose(np.abs(w).sum(axis=0), 1.)
        # as printed, a zero entry is multiplied by both signs, so the same vector appears more than once: 12 columns, 8 distinct (the optimum is unaffected, only the run time)
        self.assertEqual(len({tuple(c) for c in w.T.round(9)}), 8)

    def test_rank_of_random_matrix(self):
        x = D.rndMatWithRank(500, 5, 2, sigma=0., seed=0)
        self.assertEqual(np.linalg.matrix_rank(x), 2)
        np.testing.assert_allclose(D.rndMatWithRank(50, 4, 4, seed=7), D.rndMatWithRank(50, 4, 4, seed=7))

    def test_static_portfolio_properties(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(3, 1)); x = D.rndMatWithRank(1000, 3, 3, 0., seed=1); cov = np.cov(x, rowvar=False)
        w = D.statOptPortf(cov, a)
        self.assertAlmostEqual(abs(w).sum(), 1.)
        w0 = np.linalg.solve(cov, a); w0 /= abs(w0).sum()
        np.testing.assert_allclose(w, w0)                       # proportional to V^-1 a

    def test_t_costs_and_sr_by_hand(self):
        params = [{'mean': np.array([[1.], [0.]]), 'cov': np.eye(2), 'c': np.array([.1, .1])}, {'mean': np.array([[0.], [1.]]), 'cov': np.eye(2), 'c': np.array([.1, .1])}]
        w = np.array([[1., 0.], [0., 1.]])
        tc = D.evalTCosts(w, params)
        np.testing.assert_allclose(tc, [.1 * 1, .1 * (1 + 1)])  # sqrt(|dw|): 0 -> (1,0): .1; (1,0) -> (0,1): .1*(1+1)
        sr = D.evalSR(params, w, tc)
        self.assertAlmostEqual(sr, ((1 - .1) + (1 - .2)) / (1 + 1) ** .5)

    def test_dynamic_is_best_of_all_trajectories_by_independent_brute_force(self):
        np.random.seed(3)
        params = D.genParams(size=2, horizon=2, nSamples=500, seed=5)
        w_dyn, sr_dyn = D.dynamicSolution(params)
        # independent enumeration: every weight vector with |w|_1 = 1 on the 1/2 grid, every pair of them
        grid = [np.array(v) for v in itertools.product([-1, -.5, 0, .5, 1], repeat=2) if abs(sum(map(abs, v)) - 1) < 1e-12]
        best = -np.inf
        for w1, w2 in itertools.product(grid, repeat=2):
            tc = [(params[0]['c'] * np.abs(w1) ** .5).sum(), (params[1]['c'] * np.abs(w2 - w1) ** .5).sum()]
            m = sum(w @ p['mean'].ravel() - t for w, p, t in zip((w1, w2), params, tc))
            v = sum(w @ p['cov'] @ w for w, p in zip((w1, w2), params))
            best = max(best, m / v ** .5)
        self.assertAlmostEqual(sr_dyn, best, places=9)
        self.assertEqual(w_dyn.shape, (2, 2))

    def test_static_solution_runs(self):
        np.random.seed(3)
        params = D.genParams(size=3, horizon=2, nSamples=500, seed=5)
        w, sr = D.staticSolution(params)
        self.assertEqual(w.shape, (3, 2))
        self.assertTrue(np.isfinite(sr))


if __name__ == '__main__':
    unittest.main()
