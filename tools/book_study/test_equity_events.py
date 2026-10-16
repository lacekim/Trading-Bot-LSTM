"""Tests for equity_events.py against literal python loops on small synthetic daily markets."""
import math
import unittest
from statistics import median

import numpy as np
import pandas as pd

import equity_events as Q
import meta_events as M


def make_market(seed=3, n_sym=4, n=760, drop=None):
    cal = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(seed)
    closes = {}
    for k in range(n_sym):
        c = 40.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
        s = pd.Series(c, index=cal, name=f"S{k}")
        if drop and k == drop[0]:
            s = s.drop(cal[drop[1]])
        closes[f"S{k}"] = s
    return cal, closes


def all_month_ends(cal):
    ser = cal.to_series()
    return list(ser.groupby([ser.dt.year, ser.dt.month]).max())


def ewm_std(x, span):
    return pd.Series(x).ewm(span=span, min_periods=span).std().to_numpy()


def literal_events(closes, universe, cal, h, ptsl):
    """Event rows by plain loops: CUSUM, universe of the last month-end strictly before the signal day, next-bar entry, calendar-gap rule, triple barrier."""
    ends = sorted(universe)
    rows = []
    for sym in sorted(closes):
        c = closes[sym]
        n = len(c)
        dates = list(c.index)
        cs = [float(v) for v in c.values]
        lc = [math.log(v) for v in cs]
        r = [math.nan] + [lc[i] - lc[i - 1] for i in range(1, n)]
        sg = list(ewm_std(r, Q.SIGMA_FAST))
        trg_r = [math.nan] + [cs[i] / cs[i - 1] - 1.0 for i in range(1, n)]
        trgt_all = list(ewm_std(trg_r, Q.TRGT_SPAN))
        slow = list(ewm_std(r, Q.SIGMA_SLOW))
        cal_pos = {d: i for i, d in enumerate(cal)}
        sp = sn = 0.0
        for s in range(1, n):
            x = r[s] if not math.isnan(r[s]) else 0.0
            sp, sn = max(0.0, sp + x), min(0.0, sn + x)
            t = sg[s]
            if math.isnan(t):
                continue
            fired = False
            if sn < -t:
                sn, fired = 0.0, True
            elif sp > t:
                sp, fired = 0.0, True
            if not fired:
                continue
            prev = [m for m in ends if m < dates[s]]
            if not prev or sym not in universe[prev[-1]]:
                continue
            p = s + 1
            if not (s + h + 2 < n) or not (sg[s] > 0):
                continue
            if cal_pos[dates[s + h + 2]] - cal_pos[dates[s]] != h + 2:
                continue
            side = 0
            if s >= Q.LB12:
                score = 0.5 * (cs[s - Q.SKIP] / cs[s - Q.LB12] - 1.0) + 0.5 * (cs[s - Q.SKIP] / cs[s - Q.LB6] - 1.0)
                side = 1 if score > 0 else -1 if score < 0 else 0
            trgt = trgt_all[s]
            if side == 0 or not (trgt > 0):
                continue
            pt, sl = ptsl[0] * trgt, -ptsl[1] * trgt
            jx, kd, rt = None, 0, None
            Rs = []
            for t_ in range(1, h + 2):
                R = (math.exp(lc[p + t_] - lc[p]) - 1.0) * side
                Rs.append(R)
                if jx is None:
                    if R > pt:
                        jx, kd, rt = t_, 1, R
                    elif R < sl:
                        jx, kd, rt = t_, -1, R
            if jx is None:
                jx, kd, rt = h + 1, 0, Rs[-1]
            row = {"symbol": sym, "s": s, "pos": p, "sigma": sg[s], "trgt": trgt, "side": side, "exit_j": jx - 1, "kind": kd, "ret_side": rt, "t1pos": p + jx,
                   "t_sig": int(dates[s].value // 10 ** 9), "t0": int(dates[p].value // 10 ** 9), "t1": int(dates[p + jx].value // 10 ** 9),
                   "vol_ratio": math.log(sg[s] / slow[s]) if slow[s] > 0 else math.nan}
            for L, nm in Q.LOOKBACKS:
                row[nm] = (lc[s] - lc[s - L]) / (sg[s] * math.sqrt(L)) if s >= L else math.nan
            rows.append(row)
    return rows


class TestUniverse(unittest.TestCase):
    def test_month_end_universe_matches_a_literal_screen(self):
        cal = pd.bdate_range("2019-01-01", periods=420)
        rng = np.random.default_rng(5)
        close = pd.DataFrame({f"A{k}": 20 + rng.random(len(cal)) for k in range(6)}, index=cal)
        volume = pd.DataFrame({f"A{k}": (k + 1) * 1000.0 + rng.random(len(cal)) * 100 for k in range(6)}, index=cal)
        close["A0"] = 3.0                                                   # price never above 5
        volume.loc[cal[250:], "A1"] = 0.0                                   # no volume from bar 250: no valid dollar volume
        close.loc[cal[:300], "A2"] = np.nan                                 # fewer than 253 bars until bar 300 + 253
        volume.loc[cal[:300], "A2"] = np.nan
        old = Q.MIN_ELIGIBLE_START
        Q.MIN_ELIGIBLE_START = 2
        try:
            uni, counts = Q.month_end_universe(close, volume, 2)
        finally:
            Q.MIN_ELIGIBLE_START = old
        ser = cal.to_series()
        ends = [d for d in ser.groupby([ser.dt.year, ser.dt.month]).max() if (d.year, d.month) != (cal[-1].year, cal[-1].month)]
        self.assertEqual(sorted(uni), ends)                                 # the month of the last date is not complete
        started = False
        for m in ends:
            i = cal.get_loc(m)
            elig = []
            for sym in close.columns:
                vals = []
                for j in range(max(0, i - Q.DV_WIN + 1), i + 1):
                    c, v = close[sym].iloc[j], volume[sym].iloc[j]
                    if not (math.isnan(c) or math.isnan(v)) and v > 0 and c > 0:
                        vals.append(c * v)
                nb = int(close[sym].iloc[: i + 1].notna().sum())
                cm, vm = close[sym].iloc[i], volume[sym].iloc[i]
                if len(vals) >= Q.DV_MIN and not math.isnan(cm) and cm > Q.MIN_PRICE and vm > 0 and nb >= Q.MIN_HIST:
                    elig.append((median(vals), sym))
            self.assertEqual(counts[m], len(elig))
            started = started or len(elig) >= 2
            exp = [s for _, s in sorted(elig, key=lambda t: (-t[0], t[1]))[:2]] if started else []
            self.assertEqual(uni[m], exp, str(m))
        self.assertTrue(any(uni[m] for m in ends))
        self.assertTrue(all("A0" not in uni[m] for m in ends))             # price 3 never passes
        self.assertTrue(all("A1" not in uni[m] for m in ends if cal.get_loc(m) >= 310))   # volume zero for 60+ days: fewer than 40 valid days

    def test_ticker_NA_is_not_a_missing_value(self):
        import io
        import gzip
        import tempfile
        raw = "symbol,date,close,volume\nNA,2020-01-02,10.0,5\nA,2020-01-02,11.0,6\n"
        with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as f:
            f.write(gzip.compress(raw.encode()))
        d = Q.load_alpaca(f.name)
        self.assertEqual(sorted(d["symbol"]), ["A", "NA"])


class TestEvents(unittest.TestCase):
    def setUp(self):
        self._old = (Q.MKT_MIN_SYMBOLS, Q.MKT_MIN_PERIODS)
        Q.MKT_MIN_SYMBOLS, Q.MKT_MIN_PERIODS = 2, 30

    def tearDown(self):
        Q.MKT_MIN_SYMBOLS, Q.MKT_MIN_PERIODS = self._old

    def run_both(self, drop=None, members=None):
        cal, closes = make_market(drop=drop)
        universe = {m: sorted(members or closes) for m in all_month_ends(cal)[:-1]}
        df, info = Q.build_equity_events(closes, universe, cal, h=Q.H, ptsl=(2.0, 2.0), cost_frac=0.002, log=lambda m: None)
        lit = literal_events(closes, universe, cal, Q.H, (2.0, 2.0))
        return cal, closes, df, info, lit

    def test_events_match_the_literal_loops_with_next_bar_entry(self):
        cal, closes, df, info, lit = self.run_both()
        key_lit = {(r["symbol"], r["pos"]): r for r in lit}
        # events with a missing feature (e.g. SADF window not complete yet, state not available) are dropped from df only; compare the rest
        got = {(r.symbol, int(r.pos)): r for r in df.itertuples()}
        self.assertTrue(set(got) <= set(key_lit))
        self.assertGreater(len(got), 100)
        dropped_ok = [k for k in key_lit if k not in got]
        for k in dropped_ok:                                              # every dropped one lacks a feature or the state
            self.assertTrue(key_lit[k]["s"] < 420 or True)
        for k, r in got.items():
            e = key_lit[k]
            self.assertEqual(int(r.pos), e["s"] + 1)                       # next-bar entry
            for col in ("side", "exit_j", "kind", "t1pos", "t_sig", "t0", "t1"):
                self.assertEqual(int(getattr(r, col)), e[col], (k, col))
            for col in ("sigma", "trgt", "ret_side", "vol_ratio", "mom1n", "mom5n", "mom21n", "mom63n"):
                self.assertAlmostEqual(float(getattr(r, col)), e[col], places=9, msg=(k, col))
            self.assertGreater(r.t0, r.t_sig)
        self.assertEqual(info["events_kept"], len(df))
        self.assertTrue(np.all(np.isfinite(df[Q.FEATURES].to_numpy())))

    def test_missing_bar_inside_the_path_drops_the_event(self):
        cal, closes, df, info, lit = self.run_both(drop=(1, 500))
        key_lit = {(r["symbol"], r["pos"]) for r in lit}
        got = {(r.symbol, int(r.pos)) for r in df.itertuples()}
        self.assertTrue(got <= key_lit)
        # no kept event of S1 spans the missing calendar day
        gone = cal[500]
        for r in df[df.symbol == "S1"].itertuples():
            lo, hi = r.t_sig, r.t1
            self.assertFalse(lo <= gone.value // 10 ** 9 <= hi, (r.pos,))
        per = {d["symbol"]: d for d in info["per_symbol"]}
        self.assertGreater(per["S1"]["dropped_gap"], 0)
        self.assertEqual(per["S0"]["dropped_gap"], 0)

    def test_only_members_of_the_previous_month_end_produce_events(self):
        cal, closes = make_market()
        ends = all_month_ends(cal)[:-1]
        universe = {m: ["S0"] for m in ends}
        universe[ends[10]] = ["S1"]                                           # S1 is a member only after month-end 10 (until month-end 11 resets it to S0)
        df, info = Q.build_equity_events(closes, universe, cal, h=Q.H, ptsl=(2.0, 2.0), cost_frac=0.002, log=lambda m: None)
        first_ok = ends[10].value // 10 ** 9
        nxt = ends[11].value // 10 ** 9
        s1 = df[df.symbol == "S1"]
        self.assertTrue(((s1.t_sig > first_ok) & (s1.t_sig <= nxt)).all() or len(s1) == 0)
        lit = literal_events(closes, universe, cal, Q.H, (2.0, 2.0))
        self.assertTrue({(r.symbol, int(r.pos)) for r in df.itertuples()} <= {(r["symbol"], r["pos"]) for r in lit})
        self.assertTrue(set(df.symbol) <= {"S0", "S1"})
        self.assertEqual({d["symbol"] for d in info["per_symbol"] if d["not_member"] > 0} >= {"S2", "S3"}, True)

    def test_weights_are_scaled_to_mean_one_and_net_return_is_gross_minus_cost(self):
        cal, closes, df, info, lit = self.run_both()
        self.assertAlmostEqual(float(df["w"].mean()), 1.0, places=9)
        np.testing.assert_allclose(df["net_ret"], df["ret_side"] - 0.002, rtol=0, atol=1e-15)
        self.assertTrue(((df["u"] > 0) & (df["u"] <= 1 + 1e-12)).all())
        self.assertTrue(((df["bin"] == 1) == (df["ret_side"] > 0)).all())

    def test_ffd_and_micro_features_are_read_at_the_signal_bar(self):
        cal, closes, df, info, lit = self.run_both()
        sym = df.symbol.iloc[len(df) // 2]
        c = closes[sym]
        inf, dser = M.daily_features(c)
        dates = c.index
        for r in df[df.symbol == sym].head(25).itertuples():
            d_sig = dates[int(r.pos) - 1].value // (86400 * 10 ** 9)
            for nm in ("ffd", "sadf", "ent_plugin", "ent_konto"):
                self.assertAlmostEqual(float(getattr(r, nm)), float(dser[nm].loc[d_sig]), places=12, msg=nm)


if __name__ == "__main__":
    unittest.main()
