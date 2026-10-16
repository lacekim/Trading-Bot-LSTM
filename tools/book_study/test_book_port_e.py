import unittest

import numpy as np
import pandas as pd

import book_port_e as E


def hourly(n, seed=0, start='2026-01-01', drift=0.0, vol=0.004):
    rng = np.random.default_rng(seed)
    ix = pd.date_range(start, periods=n, freq='h')
    return pd.Series(100 * np.exp(np.cumsum(drift + vol * rng.normal(size=n))), index=ix)


class Test2_4(unittest.TestCase):
    def test_cusum_hand(self):
        ix = pd.date_range('2026-01-01', periods=8, freq='D')
        g = pd.Series([0, 1, 2.5, 2.4, 0.5, 0.4, -1.2, -1.3], index=ix)
        # diffs: 1,1.5,-.1,-1.9,-.1,-1.6,-.1 ; h=2: sPos 1, 2.5>2 -> event day2, 0; sNeg -.1 (day3) sPos 0; day4 sNeg -2.0 (not < -2) ; day5 sNeg -2.1 < -2 -> event; ...
        ev = E.getTEvents(g, 2.)
        self.assertEqual(list(ev), [ix[2], ix[5]])

    def test_matches_project_cusum(self):
        import book_stats as BS
        s = hourly(3000, 1)
        lg = np.log(s)
        h = 0.02
        ev = E.getTEvents(lg, h)
        r = np.concatenate([[np.nan], np.diff(lg.values)])
        ref = BS.cusum_events(r, np.full(len(r), h))
        self.assertEqual(list(ev), list(s.index[ref]))


class Test3(unittest.TestCase):
    def test_daily_vol_against_brute_force(self):
        c = hourly(400, 2)
        v = E.getDailyVol(c, span0=20)
        # brute force: for each bar, the last bar at least... the book takes the bar BEFORE the first one at or after (t - 1 day)
        rets = {}
        for i, t in enumerate(c.index):
            j = c.index.searchsorted(t - pd.Timedelta(days=1))
            if j > 0:
                rets[t] = c.iloc[i] / c.iloc[j - 1] - 1
        r = pd.Series(rets)
        np.testing.assert_allclose(v.values, r.ewm(span=20).std().values, equal_nan=True)

    def test_vertical_barrier(self):
        c = hourly(100, 3)
        t = pd.DatetimeIndex([c.index[0], c.index[10], c.index[90]])
        t1 = E.addVerticalBarrier(c, t, 1)
        self.assertEqual(list(t1.values), [c.index[24], c.index[34]])       # the third event has no bar one day later
        self.assertEqual(list(t1.index), [t[0], t[1]])

    def test_triple_barrier_hand(self):
        ix = pd.date_range('2026-01-01', periods=6, freq='D')
        c = pd.Series([100, 101, 103, 97.9, 97, 105.], index=ix)
        ev = pd.DataFrame({'t1': [ix[5], ix[5], ix[5]], 'trgt': [.02, .02, .02], 'side': [1., -1., 1.]}, index=[ix[0], ix[1], ix[3]])
        out = E.applyPtSlOnT1(c, ev, [1, 1], ev.index)
        # event 0, long from 100, pt = +2%: first touch day2 (+3%); sl = -2%: day3 (97.9: -2.1%)
        self.assertEqual(out.loc[ix[0], 'pt'], ix[2]); self.assertEqual(out.loc[ix[0], 'sl'], ix[3])
        # event 1 short from 101 (returns x -1): day2 103 is -1.98% (not below -2%), day3 97.9 is +3.07% -> pt; day5 105 is -3.96% -> sl
        self.assertEqual(out.loc[ix[1], 'pt'], ix[3]); self.assertEqual(out.loc[ix[1], 'sl'], ix[5])
        # event 2 long from 97.9: day4 97 is -0.92%; day5 105 is +7.25% -> pt
        self.assertEqual(out.loc[ix[3], 'pt'], ix[5]); self.assertTrue(pd.isna(out.loc[ix[3], 'sl']))

    def test_get_events_and_bins_meta_labeling(self):
        ix = pd.date_range('2026-01-01', periods=8, freq='D')
        c = pd.Series([100, 101, 103, 98, 97, 105, 104, 106.], index=ix)
        tE = pd.DatetimeIndex([ix[0], ix[3]])
        trgt = pd.Series(.02, index=ix)
        t1 = E.addVerticalBarrier(c, tE, 4)
        side = pd.Series([1., -1.], index=tE)
        ev = E.getEvents(c, tE, [1, 1], trgt, .0, 1, t1=t1, side=side)
        # event 0 long: pt at day2 -> t1 = day2 ; event at day3 short from 98: pt = -2% price move: day4 97 is -1.02%; sl: day5 105 -> t1 = day5 (vertical barrier day7 later)
        self.assertEqual(list(ev['t1']), [ix[2], ix[5]])
        bins = E.getBins(ev, c)
        np.testing.assert_allclose(bins['ret'].values, [103 / 100 - 1, -(105 / 98 - 1)])
        self.assertEqual(list(bins['bin']), [1., 0.])            # meta-label: 0 for a non-positive side-return
        # without side: bin is the sign of the price move
        ev2 = E.getEvents(c, tE, [1, 1], trgt, .0, 1, t1=t1)
        self.assertNotIn('side', ev2)
        self.assertEqual(list(E.getBins_3_5(ev2, c)['bin']), [1., 1.])
        self.assertEqual(list(E.getEvents_3_3(c, tE, 1, trgt, .0, 1, t1=t1)['t1']), list(ev2['t1']))

    def test_min_ret_filter(self):
        ix = pd.date_range('2026-01-01', periods=5, freq='D')
        c = pd.Series([100, 101, 102, 103, 104.], index=ix)
        trgt = pd.Series([.001, .05, .05, .05, .05], index=ix)
        ev = E.getEvents(c, pd.DatetimeIndex([ix[0], ix[1]]), [1, 1], trgt, .01, 1)
        self.assertEqual(list(ev.index), [ix[1]])


class Test4(unittest.TestCase):
    def setUp(self):
        self.ix = pd.date_range('2026-01-01', periods=5, freq='D')
        self.close = pd.Series([100, 101, 102, 101, 103.], index=self.ix)
        self.events = pd.DataFrame({'t1': [self.ix[2], self.ix[3], self.ix[4]]}, index=[self.ix[0], self.ix[1], self.ix[3]])

    def test_concurrency_uniqueness_weights_hand(self):
        nco, tW = E.getTW(self.events, self.close, 1)
        # bars 0..4: events A=[0,2], B=[1,3], C=[3,4] -> concurrency 1,2,2,2,1
        np.testing.assert_allclose(nco.values, [1, 2, 2, 2, 1])
        np.testing.assert_allclose(tW.values, [(1 + .5 + .5) / 3, .5, (.5 + 1) / 2])
        import book_stats as BS
        np.testing.assert_allclose(tW.values, BS.average_uniqueness(np.array([0, 1, 3]), np.array([2, 3, 4]), 5))
        w = E.getSampleW(self.events, nco, self.close, 1)
        r = np.log(self.close).diff()
        exp = np.abs([(r.iloc[1] / 1 + r.iloc[2] / 2), (r.iloc[2] / 2 + r.iloc[3] / 2), (r.iloc[4] / 1) + r.iloc[3] / 2])
        # event A covers bars 0..2 and its return at bar 0 is NaN (skipped by sum); concurrency at bars 1, 2 is 2 and 2:
        exp = np.abs([r.iloc[1] / 2 + r.iloc[2] / 2, r.iloc[1] / 2 + r.iloc[2] / 2 + r.iloc[3] / 2, r.iloc[3] / 2 + r.iloc[4] / 1])
        exp = exp * 3 / exp.sum()
        np.testing.assert_allclose(w.values, exp)
        self.assertAlmostEqual(w.sum(), 3.)

    def test_matches_port_a_matrix_version(self):
        import book_port_a as PA
        nco, tW = E.getTW(self.events, self.close, 1)
        t1pos = pd.Series([2, 3, 4], index=[0, 1, 3])
        u = PA.getAvgUniqueness(PA.getIndMatrix(range(5), t1pos))
        np.testing.assert_allclose(tW.values, u.values)

    def test_time_decay_hand(self):
        tW = pd.Series([.5, .5, .5, .5], index=pd.date_range('2026-01-01', periods=4))
        w, const, slope = E.getTimeDecay(tW, .5)
        np.testing.assert_allclose(w.values, [.625, .75, .875, 1.0]); self.assertAlmostEqual(const, .5); self.assertAlmostEqual(slope, .25)
        w, const, slope = E.getTimeDecay(tW, -.5)
        np.testing.assert_allclose(w.values, [0, 0, .5, 1.]); self.assertAlmostEqual(const, -1.); self.assertAlmostEqual(slope, 1.)
        w, _, _ = E.getTimeDecay(tW, 1.)
        np.testing.assert_allclose(w.values, 1.)
        # against the project's own function
        import time_decay as TD
        u = np.array([.5, .3, .9, .2, .7]); t0 = np.arange(5)
        for c in (1., .5, 0., -.5):
            mine, _, _ = E.getTimeDecay(pd.Series(u, index=pd.date_range('2026-01-01', periods=5)), c)
            np.testing.assert_allclose(mine.values, TD.decay_factors(u, t0, c), atol=1e-12)


class Test5(unittest.TestCase):
    def test_ffd_weights_are_binomial(self):
        d = .4
        w = E.getWeights_FFD(d, 1e-2).ravel()[::-1]
        from scipy.special import binom
        for k, wk in enumerate(w):
            self.assertAlmostEqual(wk, (-1) ** k * binom(d, k), places=12)
        self.assertLess(abs(-w[-1] / len(w) * (d - len(w) + 1)), 1e-2)       # the next one is below the threshold
        import fracdiff_core as F
        np.testing.assert_allclose(E.getWeights_FFD(d, 1e-2).ravel(), F.weights_ffd(d, 1e-2))

    def test_ffd_d1_is_first_difference(self):
        s = hourly(50, 5).to_frame('Close')
        out = E.fracDiff_FFD(s, 1., thres=1e-5)
        np.testing.assert_allclose(out['Close'].values, s['Close'].diff().dropna().values)

    def test_ffd_matches_project_frac_diff(self):
        import fracdiff_core as F
        s = np.log(hourly(300, 6))
        for d in (.2, .5, .9):
            mine = E.fracDiff_FFD(s.to_frame('Close'), d, thres=.01)['Close'].values
            ref = F.frac_diff_ffd(s.values, d, .01)
            ref = ref[np.isfinite(ref)]
            np.testing.assert_allclose(mine, ref, atol=1e-10)

    def test_adf_c1_properties(self):
        rng = np.random.default_rng(0)
        e = rng.normal(size=2000)
        stat_rw, crit, n = E.adfuller_c1(np.cumsum(e))
        ar = np.zeros(2000)
        for i in range(1, 2000):
            ar[i] = .5 * ar[i - 1] + e[i]
        stat_ar, _, _ = E.adfuller_c1(ar)
        self.assertAlmostEqual(crit, -2.8621, places=2)
        self.assertGreater(stat_rw, crit)              # random walk: do not reject
        self.assertLess(stat_ar, crit - 5)             # stationary: reject
        import fracdiff_core as F
        self.assertAlmostEqual(stat_ar, F.adf_stat(ar, 1)[0], places=8)
        self.assertAlmostEqual(crit, F.adf_crit5(n), places=10)

    def test_min_ffd_table_monotone(self):
        s = hourly(24 * 400, 7, vol=.01).to_frame('Close')
        t = E.minFFD_table(s, thres=.01, ds=np.array([0., .5, 1.]))
        self.assertTrue((np.diff(t['adfStat'].values) < 0).all())      # more differencing -> more stationary
        self.assertGreater(t.loc[0., 'corr'], t.loc[1., 'corr'])


class Test19(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 300
        c = 100 * np.exp(np.cumsum(.002 * rng.normal(size=n)))
        hi = c * np.exp(np.abs(.002 * rng.normal(size=n)) + .0005)
        lo = c * np.exp(-np.abs(.002 * rng.normal(size=n)) - .0005)
        self.df = pd.DataFrame({'High': hi, 'Low': lo}, index=pd.date_range('2026-01-01', periods=n, freq='h'))

    def test_matches_hand_formula(self):
        h, l = self.df['High'].values, self.df['Low'].values
        sp = E.corwinSchultz(self.df, sl=1)
        k = 5                                          # compare one position by hand
        b = np.log(h[k - 1] / l[k - 1]) ** 2 + np.log(h[k] / l[k]) ** 2
        g = np.log(max(h[k - 1], h[k]) / min(l[k - 1], l[k])) ** 2
        den = 3 - 2 * 2 ** .5
        a = max(0., (2 ** .5 - 1) * b ** .5 / den - (g / den) ** .5)
        self.assertAlmostEqual(sp['Spread'].iloc[k - 1], 2 * (np.exp(a) - 1) / (1 + np.exp(a)), places=12)

    def test_matches_project_corwin_schultz(self):
        import micro_features as MF
        for sl in (1, 24):
            mine = E.corwinSchultz(self.df, sl=sl)['Spread'].values
            ref = MF.corwin_schultz(self.df['High'].values, self.df['Low'].values, sl)
            ref = ref[np.isfinite(ref)]
            np.testing.assert_allclose(mine, ref, atol=1e-12)

    def test_sigma_non_negative(self):
        beta, gamma = E.getBeta(self.df, 1), E.getGamma(self.df)
        s = E.getSigma(beta, gamma.loc[beta.index] if len(gamma) else gamma)
        self.assertTrue((s >= 0).all())


if __name__ == '__main__':
    unittest.main()
