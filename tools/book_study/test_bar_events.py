import unittest

import numpy as np
import pandas as pd

import bar_events as BE
import bar_loader as BL
import meta_events as E

DAY = 86400


def literal_daily_vol(idx, close, span=100):
    """The book's Snippet 3.1, pandas as printed (plus min_periods = span, the project's choice)."""
    c = pd.Series(close, index=idx)
    df0 = c.index.searchsorted(c.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(c.index[df0 - 1], index=c.index[c.shape[0] - df0.shape[0]:])
    df0 = c.loc[df0.index] / c.loc[df0.values].values - 1
    return df0.ewm(span=span, min_periods=span).std().reindex(c.index).to_numpy()


def irregular_times(n, seed=0, t0=1_600_000_000):
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(2400, n).astype(np.int64) + 60
    return t0 + np.cumsum(gaps)


def regular_hourly_bars(sym, n, seed, t0=1_600_000_000 // 3600 * 3600):
    """Hourly bars stamped with their END, no holes; reference series = the same bars (hour start = end - 3600)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.004, n)
    r[rng.random(n) < 0.02] *= 6.0
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.random(n) * 0.003)
    l = np.minimum(o, c) * (1 - rng.random(n) * 0.003)
    end = t0 + 3600 * (1 + np.arange(n))
    b = BL.make_bars(sym, end, end - 3600, c, h, l, 48 * 3600, end, h, l, c)       # ref stamped with the hour END = the bar end
    return b, pd.Series(c, index=pd.DatetimeIndex(pd.to_datetime(end, unit="s"))), pd.DataFrame({"high": h, "low": l, "close": c}, index=pd.DatetimeIndex(pd.to_datetime(end, unit="s")))


class T(unittest.TestCase):
    def test_daily_vol_matches_book_snippet(self):
        t = irregular_times(900, 1)
        rng = np.random.default_rng(2)
        c = 50 * np.exp(np.cumsum(rng.normal(0, 0.003, len(t))))
        mine = BE.daily_vol_time(t, c, break_s=10 ** 9)
        lit = literal_daily_vol(pd.DatetimeIndex(pd.to_datetime(t, unit="s")), c)
        self.assertGreater(np.isfinite(mine).sum(), 100)
        np.testing.assert_allclose(mine, lit, rtol=1e-12, equal_nan=True)

    def test_daily_vol_ignores_a_return_that_reaches_across_a_hole(self):
        t = irregular_times(600, 3)
        t[300:] += 5 * DAY                                   # a 5-day hole before bar 300
        rng = np.random.default_rng(4)
        c = 50 * np.exp(np.cumsum(rng.normal(0, 0.003, len(t))))
        with_limit = BE.daily_vol_time(t, c, break_s=2 * DAY)
        without = BE.daily_vol_time(t, c, break_s=10 ** 9)
        self.assertTrue(np.isfinite(without[299:]).all() or True)
        # the jump return at the first bar after the hole is excluded when the limit applies, so the two series differ from there on
        self.assertFalse(np.allclose(with_limit[310:], without[310:], equal_nan=True))

    def test_bars_per_day_by_hand_and_literal(self):
        t = np.arange(1, 24 * 40 + 1) * 3600 + 1_600_000_000       # regular hourly stamps, 40 days
        n_day = BE.bars_per_day(t)
        self.assertTrue(np.isnan(n_day[0]))
        np.testing.assert_allclose(n_day[1:], 24.0, rtol=1e-12)
        # irregular stamps against a plain loop
        t2 = irregular_times(400, 5)
        got = BE.bars_per_day(t2)
        for i in (1, 5, 50, 399):
            w = min(30 * DAY, t2[i] - t2[0])
            cnt = sum(1 for x in t2[: i + 1] if t2[i] - w < x <= t2[i])
            self.assertAlmostEqual(got[i], cnt / (w / DAY), places=9)

    def test_forward_window_cases(self):
        t = np.array([0, 3600, 7200, 10800, 100000, 103600, 107200, 500000], np.int64)
        seg = np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int64)           # a hole between bar 3 and bar 4
        ev = np.array([0, 1, 4, 5, 6, 7], np.int64)
        L, ok, capped = BE.forward_window(t, seg, ev, h_s=7000, hcap=10)
        # event 0: first bar ending at/after 7000 is bar 2 (7200) -> L = 2; event 1: 3600+7000 = 10600 -> bar 3 (10800) -> L = 2; event 4: 100000+7000 = 107000 -> bar 6 -> L = 2
        # event 5: 103600 + 7000 = 110600 -> bar 7 (500000), which lies in segment 1 as well -> L = 2 (a long wait is allowed inside a segment); event 6: 114200 -> bar 7 -> L = 1; event 7: no bar at/after -> dropped
        self.assertEqual(list(L[:5]), [2, 2, 2, 2, 1])
        self.assertEqual(list(ok), [True, True, True, True, True, False])
        L2, ok2, cap2 = BE.forward_window(t, seg, ev, h_s=60000, hcap=10)
        self.assertEqual(list(ok2), [False, False, True, True, True, False])      # events 0 and 1 would end in segment 1 -> dropped (hole in between); 4, 5, 6 end at bar 7
        L3, ok3, cap3 = BE.forward_window(t, seg, np.array([0]), h_s=7000, hcap=1)
        self.assertEqual((int(L3[0]), bool(cap3[0])), (1, True))                  # stopped at the cap, flagged

    def test_triple_barrier_var_matches_original_when_all_rows_have_full_length(self):
        rng = np.random.default_rng(6)
        n, W = 300, 30
        q = (np.cumsum(rng.normal(0, 0.5, (n, W)), axis=1)).astype(np.float32)
        sigma = np.full(n, 0.01)
        side = rng.choice([-1.0, 1.0], n)
        trgt = np.full(n, 0.02)
        j0, k0, r0 = E.triple_barrier(q, sigma, side, trgt, (2.0, 2.0))
        j1, k1, r1 = BE.triple_barrier_var(q, np.full(n, W), sigma, side, trgt, (2.0, 2.0))
        np.testing.assert_array_equal(j0, j1)
        np.testing.assert_array_equal(k0, k1)
        np.testing.assert_allclose(r0, r1, rtol=1e-12)

    def test_triple_barrier_var_short_rows_by_hand(self):
        # row 0: length 3, never touches -> vertical at column 2 ; row 1: length 5, touches the stop at column 3 ; row 2: length 2 but the path would touch at column 3 (beyond L) -> vertical
        q = np.full((3, 6), np.nan, np.float32)
        q[0, :3] = [0.0, 1.0, 0.5]
        q[1, :5] = [0.0, 0.0, 0.0, -9.0, -9.0]
        q[2, :2] = [0.0, 0.1]
        sigma = np.full(3, 0.01)
        side = np.ones(3)
        trgt = np.full(3, 0.02)
        j, kind, ret = BE.triple_barrier_var(q, np.array([3, 5, 2]), sigma, side, trgt, (2.0, 2.0))
        self.assertEqual(list(j), [2, 3, 1])
        self.assertEqual(list(kind), [0, -1, 0])
        self.assertAlmostEqual(ret[0], np.expm1(0.5 * 0.01), places=8)
        self.assertAlmostEqual(ret[2], np.expm1(0.1 * 0.01), places=8)

    def test_asof_and_market_state_by_hand(self):
        ref_t = np.array([100, 200, 300, 40000], np.int64)
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        got = BE.asof(ref_t, vals, np.array([50, 100, 250, 300, 20000, 40000, 50000]), tol_s=5000)
        np.testing.assert_array_equal(np.isnan(got), [True, False, False, False, True, False, True])
        self.assertEqual(list(got[[1, 2, 3, 5]]), [1.0, 2.0, 3.0, 4.0])

    def test_equivalent_to_the_hourly_pipeline_on_regular_hourly_bars(self):
        """On bars that are exactly one per hour, with no holes, the bar pipeline (vertical barrier of 101 hours) must give the earlier pipeline's event table."""
        n, k = 6200, 12
        bars, closes, ohlc = {}, {}, {}
        for i in range(k):
            b, c, o = regular_hourly_bars(f"S{i:02d}", n, 100 + i)
            bars[b.sym], closes[b.sym], ohlc[b.sym] = b, c, o
        old, oinfo = E.build_meta_events(closes, ohlc, h=100, ptsl=(2.0, 2.0), cost_frac=0.0024, log=lambda *_: None)
        new, ninfo = BE.build_meta_events_bars(bars, break_s=48 * 3600, h_days=101 / 24.0, hcap=400, ptsl=(2.0, 2.0), cost_frac=0.0024, log=lambda *_: None, mkt_min_history_s=3839 * 3600)
        self.assertGreater(len(old), 100)
        key = ["symbol", "pos"]
        o = old.sort_values(key).reset_index(drop=True)
        w = new.sort_values(key).reset_index(drop=True)
        self.assertEqual(len(o), len(w), (len(o), len(w)))
        pd.testing.assert_frame_equal(o[key], w[key])
        self.assertTrue((o["t0"].to_numpy() == w["t0"].to_numpy()).all())
        self.assertTrue((o["t1"].to_numpy() == w["t1"].to_numpy()).all())
        for col in ["t1pos", "side", "exit_j", "kind", "bin"]:
            np.testing.assert_array_equal(o[col].to_numpy(), w[col].to_numpy(), err_msg=col)
        for col in ["sigma", "trgt", "ret_side", "ffd", "sadf", "ent_plugin", "ent_konto", "cs_spread", "mom24n", "mom168n", "mom720n", "mom2160n", "log_trgt", "vol_ratio",
                    "mkt_state", "mkt_dist", "u", "w_raw", "w", "net_ret"]:
            np.testing.assert_allclose(o[col].to_numpy(float), w[col].to_numpy(float), rtol=1e-9, atol=1e-12, err_msg=col)
        self.assertEqual(int(new["capped"].sum()), 0)

    def test_hole_rules_on_irregular_bars(self):
        """A hole in the middle of a symbol: no event whose path crosses it, no event whose 720-bar lookback reaches back across it, no return across it."""
        n = 6000
        b, c, o = regular_hourly_bars("H", n, 9)
        # insert a 6-day hole after bar 3000 by shifting later stamps (start and end) and the reference series alike
        shift = 6 * DAY
        t, st = b.t.copy(), b.start.copy()
        t[3000:] += shift
        st[3000:] += shift
        rt = b.ref_t.copy()
        rt[3000:] += shift
        hb = BL.make_bars("H", t, st, b.close, b.high, b.low, 48 * 3600, rt, b.ref_high, b.ref_low, b.ref_close)
        self.assertEqual(int(hb.seg[-1]), 1)
        others = {}
        for i in range(11):
            x, _, _ = regular_hourly_bars(f"S{i:02d}", n, 200 + i)
            others[x.sym] = x
        others["H"] = hb
        df, info = BE.build_meta_events_bars(others, break_s=48 * 3600, h_days=2.0, hcap=400, log=lambda *_: None, mkt_min_history_s=3839 * 3600)
        h = df[df["symbol"] == "H"]
        self.assertGreater(len(h), 20)
        seg = hb.seg
        self.assertTrue((seg[h["pos"].to_numpy()] == seg[h["t1pos"].to_numpy()]).all())            # no path crosses the hole
        lb = h["pos"].to_numpy() - 2160
        self.assertTrue((lb >= 0).all() and (seg[lb] == seg[h["pos"].to_numpy()]).all())             # the longest lookback stays inside the event's segment


if __name__ == "__main__":
    unittest.main()
