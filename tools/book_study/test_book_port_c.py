import math
import random
import unittest

import numpy as np
import pandas as pd

import book_port_c as C


class Test10_4(unittest.TestCase):
    def test_book_worked_example(self):
        w, tPos, lP = C.main_10_4()
        self.assertAlmostEqual(w, 100 * (.95 ** -2 - 1))
        self.assertEqual(tPos, 97)                       # book p.90: order size 97
        self.assertAlmostEqual(lP, 112.3657, places=3)   # book p.90: 112.3657, between p_t=100 and f=115
        self.assertEqual(C.getTPos(w, 110, 100, 100), 95)  # book p.90: f=110 gives 95
        self.assertAlmostEqual(C.betSize(w, 10), .95)    # the calibration
        self.assertAlmostEqual(C.invPrice(115, w, C.betSize(w, 115 - 100)), 100)  # inverse function


class Test10_1_to_3(unittest.TestCase):
    def test_get_signal_hand_computed(self):
        from scipy.stats import norm
        ix = pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03'])
        ev = pd.DataFrame({'t1': pd.to_datetime(['2026-01-03', '2026-01-04', '2026-01-05']), 'side': [1, -1, 1]}, index=ix)
        prob = pd.Series([.9, .6, .55], index=ix); pred = pd.Series([1, 1, 1], index=ix)
        out = C.getSignal(ev, .1, prob, pred, 2, 1)
        m = lambda p: 2 * norm.cdf((p - .5) / (p * (1 - p)) ** .5) - 1      # noqa: E731
        s = [m(.9), -m(.6), m(.55)]
        # active at each change point: t=Jan1 {0}; Jan2 {0,1}; Jan3 {1,2} (bet 0 ended at Jan3); Jan4 {2}; Jan5 {}
        exp = [s[0], (s[0] + s[1]) / 2, (s[1] + s[2]) / 2, s[2], 0.]
        np.testing.assert_allclose(out.values, np.round(np.array(exp) / .1) * .1, atol=1e-12)
        self.assertEqual(list(out.index), list(pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05'])))

    def test_discrete_signal_caps_and_rounds(self):
        out = C.discreteSignal(pd.Series([-1.3, -.26, 0.04, .26, 1.4]), .1)
        np.testing.assert_allclose(out.values, [-1, -.3, 0, .3, 1])

    def test_empty_prob(self):
        self.assertEqual(len(C.getSignal(pd.DataFrame({'t1': []}), .1, pd.Series(dtype=float), pd.Series(dtype=float), 2, 1)), 0)


class Test13(unittest.TestCase):
    def test_deterministic_path_exit_step(self):
        # sigma=0: p_t = (1-phi)F + phi p_{t-1} from the seed; exits at the first t with p_t - seed > pt (profit take) or holding period > maxHP
        phi = 2 ** (-1 / 5.)
        F, pt = 10., 6.
        p, t = 0., 0
        while True:
            p = (1 - phi) * F + phi * p
            t += 1
            if p > pt or t > 100:
                break
        out = C.batch({'forecast': F, 'hl': 5, 'sigma': 0}, nIter=3, maxHP=100, rPT=[pt], rSLm=[3.])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0][2], p)             # mean of the (identical) exits
        self.assertAlmostEqual(out[0][3], 0.)

    def test_stop_loss_and_vertical(self):
        out = C.batch({'forecast': -10., 'hl': 5, 'sigma': 0}, nIter=2, maxHP=100, rPT=[6.], rSLm=[3.])
        self.assertLess(out[0][2], -3)                   # stopped out below -3
        out = C.batch({'forecast': 0., 'hl': 5, 'sigma': 0}, nIter=2, maxHP=7, rPT=[6.], rSLm=[3.])
        self.assertAlmostEqual(out[0][2], 0.)            # no move: exits on the vertical bar

    def test_mesh_shape_and_seeded_noise(self):
        random.seed(1)
        a = C.batch({'forecast': 5, 'hl': 10, 'sigma': 1}, nIter=30, maxHP=20, rPT=np.linspace(.5, 3, 3), rSLm=np.linspace(.5, 3, 3))
        random.seed(1)
        b = C.batch({'forecast': 5, 'hl': 10, 'sigma': 1}, nIter=30, maxHP=20, rPT=np.linspace(.5, 3, 3), rSLm=np.linspace(.5, 3, 3))
        self.assertEqual(len(a), 9)
        self.assertEqual(a, b)
        for (pt, sl, m, s, sr) in a:
            self.assertAlmostEqual(sr, m / s)


class Test14(unittest.TestCase):
    def setUp(self):
        self.ix = pd.date_range('2026-01-01', periods=8, freq='D')

    def test_bets_flatten_flip_last(self):
        tPos = pd.Series([0, 1, 1, 0, -1, -1, 1, 1.], index=self.ix)
        bets = C.getBets(tPos)
        # As printed: a series that STARTS flat has its first index in `bets` (the shifted first value is NaN, and NaN != 0), then flat at day 3 (after a non-zero), the flip -1 -> 1 at day 6,
        # and the last observation (day 7).
        self.assertEqual(list(bets), [self.ix[0], self.ix[3], self.ix[6], self.ix[7]])
        # a series that starts non-flat does not get that extra point
        self.assertEqual(list(C.getBets(pd.Series([1, 1, 0, 1, 1.], index=self.ix[:5]))), [self.ix[2], self.ix[4]])

    def test_holding_period_hand_computed(self):
        tPos = pd.Series([0, 1, 2, 1, 0.], index=self.ix[:5])
        # entries: 1 unit at t=1, 1 at t=2 -> avg entry 1.5; exits at t=3 (w=1) and t=4 (w=1): (1.5 + 2.5)/2 = 2.0
        self.assertAlmostEqual(C.getHoldingPeriod(tPos), 2.0)
        self.assertTrue(np.isnan(C.getHoldingPeriod(pd.Series([0, 1, 2, 2.], index=self.ix[:4]))))

    def test_holding_period_flip(self):
        tPos = pd.Series([0, 1, -1, 0.], index=self.ix[:4])
        # long 1 from t=1, flips at t=2 (dT=1, w=1), short entered t=2, flat at t=3 (dT=1, w=1) -> 1.0
        self.assertAlmostEqual(C.getHoldingPeriod(tPos), 1.0)

    def test_hhi(self):
        self.assertAlmostEqual(C.getHHI(pd.Series([1., 1., 1., 1.])), 0.)
        self.assertGreater(C.getHHI(pd.Series([100., .01, .01, .01])), .95)
        self.assertTrue(np.isnan(C.getHHI(pd.Series([1., 2.]))))
        x = pd.Series([1., 2., 3., 4.])
        w = x / x.sum()
        self.assertAlmostEqual(C.getHHI(x), ((w ** 2).sum() - .25) / .75)

    def test_hhis_on_dated_returns(self):
        ix = pd.date_range('2026-01-01', periods=60, freq='D')
        ret = pd.Series(np.r_[np.full(30, .01), np.full(30, -.01)], index=ix)
        pos, neg, t = C.getHHIs(ret)
        self.assertAlmostEqual(pos, 0.); self.assertAlmostEqual(neg, 0.)
        self.assertFalse(np.isnan(t))

    def test_dd_tuw_hand_computed(self):
        ix = pd.date_range('2026-01-01', periods=6, freq='D')
        s = pd.Series([1., 2., 1.5, 3., 2.4, 3.5], index=ix)
        dd, tuw = C.computeDD_TuW(s)
        self.assertEqual(list(dd.index), [ix[1], ix[3]])
        np.testing.assert_allclose(dd.values, [1 - 1.5 / 2, 1 - 2.4 / 3])
        self.assertEqual(len(tuw), 1)
        self.assertAlmostEqual(tuw.iloc[0], 2 / 365.2425)
        dd2, _ = C.computeDD_TuW(s, dollars=True)
        np.testing.assert_allclose(dd2.values, [.5, .6])


class Test15(unittest.TestCase):
    def test_binomial_draws(self):
        np.random.seed(0)
        m, s, sr = C.sharpe_of_binomial_draws(n=20000, p=.55)
        self.assertAlmostEqual(m, .1, delta=.03)
        self.assertAlmostEqual(s, math.sqrt(1 - .01), delta=.01)
        self.assertAlmostEqual(sr, m / s)

    def test_variance_factorisation(self):
        self.assertLess(C.factor_variance_check(), 1e-12)

    def test_implied_precision_and_frequency_roundtrip(self):
        for sl, pt, freq, tsr in [(-.1, .1, 260, 2.), (-.05, .1, 52, 1.5), (-.2, .05, 100, 1.)]:
            p = C.binHR(sl, pt, freq, tsr)
            self.assertAlmostEqual(C.binSR(sl, pt, freq, p), tsr, places=9)
            f = C.binFreq(sl, pt, p, tsr)
            self.assertAlmostEqual(f, freq, places=6)
        # symmetric barriers: p = (1 + sqrt(1 - n/(theta^2+n)))/2 (Equation 55)
        self.assertAlmostEqual(C.binHR(-.1, .1, 260, 2.), .5 * (1 + (1 - 260 / (4. + 260)) ** .5), places=12)

    def test_matches_earlier_module(self):
        import book_stats as BS
        for args in [(-.1, .1, 260, 2.), (-.03, .07, 90, 1.2)]:
            self.assertAlmostEqual(C.binHR(*args), BS.bin_hr(*args), places=12)
        self.assertAlmostEqual(C.binSR(-.1, .1, 260, .6), BS.bin_sr(-.1, .1, 260, .6), places=12)

    def test_probability_of_failure(self):
        np.random.seed(0)
        r = C.mixGaussians(.05, -.1, .05, .1, .75, 2600)
        self.assertEqual(len(r), 2600)
        self.assertAlmostEqual((r > 0).mean(), .75 * .8413 + .25 * .1587, delta=.03)  # P(N(.05,.05)>0)=.8413, P(N(-.1,.1)>0)=.1587
        pf = C.probFailure(r, 260, 2.)
        self.assertTrue(0 <= pf <= 1)
        np.random.seed(0)
        self.assertAlmostEqual(C.main_15_5(), pf)


class Test17(unittest.TestCase):
    def test_lags_and_design(self):
        s = pd.DataFrame({'c': np.cumsum(np.random.default_rng(0).normal(size=60))}, index=pd.date_range('2026-01-01', periods=60, freq='h'))
        y, x = C.getYX(s, constant='ct', lags=2)
        self.assertEqual(x.shape[1], 1 + 2 + 2)          # lagged level, 2 lagged differences (lags 1, 2), constant, trend
        self.assertEqual(y.shape[0], x.shape[0])
        d = s.diff().dropna()
        np.testing.assert_allclose(y[:, 0], d['c'].values[-len(y):])
        np.testing.assert_allclose(x[:, 0], s['c'].values[-len(y) - 1:-1])   # lagged level
        np.testing.assert_allclose(x[:, 1], d['c'].values[-len(y) - 1:-1])   # first lagged difference

    def test_betas_equal_ols(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 3)); y = x @ [1., -2., .5] + rng.normal(size=200)
        b, v = C.getBetas(y, x)
        ref = np.linalg.lstsq(x, y, rcond=None)[0]
        np.testing.assert_allclose(b, ref)
        res = y - x @ ref
        np.testing.assert_allclose(v, res @ res / (200 - 3) * np.linalg.inv(x.T @ x))

    def test_bsadf_larger_for_explosive_path(self):
        rng = np.random.default_rng(2)
        e = rng.normal(size=300) * .01
        rw = pd.DataFrame({'c': np.cumsum(e)}, index=pd.date_range('2026-01-01', periods=300, freq='h'))
        ex = np.zeros(300); ex[0] = 1.
        for i in range(1, 300):
            ex[i] = ex[i - 1] * 1.02 + e[i]
        exd = pd.DataFrame({'c': ex}, index=rw.index)
        a = C.get_bsadf(rw, minSL=30, constant='c', lags=1)['gsadf']
        b = C.get_bsadf(exd, minSL=30, constant='c', lags=1)['gsadf']
        self.assertGreater(b, a)
        self.assertEqual(C.get_bsadf(rw, 30, 'c', 1)['Time'], rw.index[-1])


class Test18(unittest.TestCase):
    def test_plug_in(self):
        h, pmf = C.plugIn('01' * 20, 1)
        self.assertAlmostEqual(h, 1.0, places=1)
        self.assertAlmostEqual(sum(pmf.values()), 1.)
        h2, _ = C.plugIn('0' * 40, 1)
        self.assertAlmostEqual(h2, 0.)
        h3, pmf3 = C.plugIn('01' * 20, 2)
        self.assertEqual(set(pmf3), {'01', '10'})

    def test_lz_library(self):
        self.assertEqual(C.lempelZiv_lib('aabab'), ['a', 'ab'])           # hand: i=1 'a' known, 'ab' new; i=3: 'a' known, 'ab' known, end
        self.assertEqual(C.lempelZiv_lib('abababab'), ['a', 'b', 'ab', 'aba'])  # hand: a | b | ab | aba, then 'b' is known to the end

    def test_match_length(self):
        self.assertEqual(C.matchLength('abcabc', 3, 3), (4, 'abc'))
        self.assertEqual(C.matchLength('abcxyz', 3, 3), (1, ''))

    def test_konto_orders_periodic_below_random(self):
        rng = np.random.default_rng(3)
        rnd = ''.join(map(str, rng.integers(0, 2, 200)))
        per = '10' * 100
        hr, hp = C.konto(rnd)['h'], C.konto(per)['h']
        self.assertGreater(hr, hp)
        self.assertEqual(C.konto(rnd, window=20)['num'], len(rnd) - 40 + 1)


if __name__ == '__main__':
    unittest.main()
