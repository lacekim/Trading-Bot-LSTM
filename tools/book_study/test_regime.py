import math
import unittest

import numpy as np
import pandas as pd

import book_study_core as C
import regime_algo as R
import three_algo as T
from test_three_algo import synthetic_events


class TestIndexReturns(unittest.TestCase):
    def test_lookback_returns_match_a_literal_loop_with_gaps(self):
        rng = np.random.default_rng(1)
        hours = np.sort(rng.choice(np.arange(0, 6000), 4000, replace=False)).astype(np.int64)         # an index with missing hours
        t_idx = pd.to_datetime(hours * 3600 + 1_600_000_000, unit="s")
        idx = pd.Series(np.exp(np.cumsum(rng.standard_normal(len(hours)) * 0.01)), index=t_idx)
        ev_h = rng.choice(hours[1000:], 300, replace=False)
        t0 = ev_h * 3600 + 1_600_000_000
        got = R.index_lookback_returns(idx, t0, (240, 480, 960))
        keys = hours * 3600 + 1_600_000_000
        for i, t in enumerate(t0):
            for L in (240, 480, 960):
                now = max(j for j in range(len(keys)) if keys[j] <= t)
                past = [j for j in range(len(keys)) if keys[j] <= t - L * 3600]
                want = math.log(idx.iloc[now]) - math.log(idx.iloc[max(past)]) if past else math.nan
                if math.isnan(want):
                    self.assertTrue(math.isnan(got[L][i]))
                else:
                    self.assertAlmostEqual(got[L][i], want, places=12)

    def test_nan_when_lookback_precedes_the_index(self):
        idx = pd.Series([1.0, 1.1, 1.2], index=pd.to_datetime([1_600_000_000 + k * 3600 for k in range(3)], unit="s"))
        r = R.index_lookback_returns(idx, [1_600_000_000 + 2 * 3600], (1, 5))
        self.assertAlmostEqual(r[1][0], math.log(1.2 / 1.1), places=12)
        self.assertTrue(np.isnan(r[5][0]))

    def test_mkt_down_requires_every_lookback_and_nan_is_false(self):
        rets = {1: np.array([-1.0, -1.0, 1.0, np.nan, -1.0]), 2: np.array([-1.0, 1.0, -1.0, -1.0, 0.0])}
        self.assertEqual(R.mkt_down(rets, (1, 2)).tolist(), [True, False, False, False, False])


class TestRules(unittest.TestCase):
    def test_rules_by_hand(self):
        mom = np.array([1, 1, -1, -1, 0, 1, -1])
        state = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        a = np.array([False, True, True, True, True, False, True])
        down = {"base": a, "tight": ~a, "wide": np.zeros(7, dtype=bool)}
        r = R.build_rules(mom, state, down)
        self.assertEqual(r["mom_long_up"].tolist(), [1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(r["short_down"].tolist(), [0, -1, -1, -1, -1, 0, -1])
        self.assertEqual(r["short_down_mom"].tolist(), [0, 0, -1, -1, 0, 0, -1])
        self.assertEqual(r["long_up_notdown"].tolist(), [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(r["regime_rule"].tolist(), [1, 0, -1, -1, 0, 0, -1])
        self.assertEqual(r["regime_rule_tight"].tolist(), [0, 1, 0, 0, 0, 0, 0])                   # tight: down = ~a -> long needs not down: events 1 (a True -> tight False)
        self.assertEqual(r["regime_rule_wide"].tolist(), [1, 1, 0, 0, 0, 0, 0])                    # never down: only the long side
        self.assertEqual(r["long_all"].tolist(), [1] * 7)
        self.assertEqual(r["mom_all"].tolist(), mom.tolist())


class TestEvaluate(unittest.TestCase):
    @staticmethod
    def frame(n=2500, seed=2):
        rng = np.random.default_rng(seed)
        t0 = np.sort(rng.integers(1_500_000_000, 1_500_000_000 + 900 * 86400, n))
        df = pd.DataFrame({"t0": t0, "ret_long": rng.standard_normal(n) * 0.03 + 0.001, "mkt_state": (rng.random(n) < 0.5).astype(float),
                           "mom_side": rng.choice([-1, 0, 1], size=n)})
        down = {k: rng.random(n) < p for k, p in (("base", 0.3), ("tight", 0.4), ("wide", 0.2))}
        return df, down

    def test_matches_a_literal_loop(self):
        df, down = self.frame()
        win = {"w": (1_500_000_000 + 100 * 86400, 1_500_000_000 + 200 * 86400)}
        cut = 1_500_000_000 + 450 * 86400
        res, daily = R.evaluate_rules(df, down, 0.0024, 100, win, cut)
        sides = R.build_rules(df["mom_side"].to_numpy(), df["mkt_state"].to_numpy(), down)
        ret, t0 = df["ret_long"].to_numpy(), df["t0"].to_numpy()
        for v in R.VARIANTS:
            s = sides[v]
            pnl = [(s[i] * ret[i] - 0.0024) * 1e4 if s[i] != 0 else 0.0 for i in range(len(df))]
            self.assertAlmostEqual(res["stats"][v]["mean_bp_per_candidate"], float(np.mean(pnl)), places=6)
            self.assertAlmostEqual(res["stats"][v]["taken_share"], float(np.mean(s != 0)), places=9)
            idx = [i for i in range(len(df)) if win["w"][0] <= t0[i] < win["w"][1]]
            self.assertAlmostEqual(res["windows"][f"{v}|w"]["mean_bp_per_candidate"], float(np.mean([pnl[i] for i in idx])), places=6)
            self.assertEqual(res["windows"][f"{v}|w"]["trades"], int(sum(1 for i in idx if s[i] != 0)))
            pre = [i for i in range(len(df)) if t0[i] < cut]
            self.assertAlmostEqual(res["halves"][f"{v}|before"]["mean_bp_per_candidate"], float(np.mean([pnl[i] for i in pre])), places=6)
            for s_ in (0, 1):
                for dn in (0, 1):
                    ii = [i for i in range(len(df)) if df["mkt_state"][i] == s_ and down["base"][i] == bool(dn)]
                    self.assertAlmostEqual(res["cells"][f"{v}|state{s_}|down{dn}"]["mean_bp_per_candidate"], float(np.mean([pnl[i] for i in ii])), places=6)
            dd = daily[daily["variant"] == v]
            self.assertAlmostEqual(dd["sum_bp"].sum(), sum(pnl), places=5)
            self.assertEqual(int(dd["n"].sum()), len(df))
            ls = [pnl[i] for i in range(len(df)) if s[i] == 1]
            self.assertEqual(res["side_split"][f"{v}|long"]["trades"], len(ls))
        yrs = pd.DatetimeIndex(pd.to_datetime(t0, unit="s")).year
        self.assertEqual(sum(res["years"][f"long_all|{y}"]["events"] for y in set(yrs.tolist())), len(df))

    def test_dsr_tables_fall_with_more_trials(self):
        df, down = self.frame(seed=3)
        res, _ = R.evaluate_rules(df, down, 0.0024, 100, {}, 1_500_000_000 + 450 * 86400)
        for v in R.NEW_VARIANTS:
            t = res["dsr"][v]["by_trials"]
            self.assertEqual(sorted(t, key=int), ["1", "10", "88", "94"])
            ds = [t[k]["dsr"] for k in ("1", "10", "88", "94")]
            self.assertTrue(all(ds[i] >= ds[i + 1] - 1e-12 for i in range(3)), ds)

    def test_a_planted_regime_is_found(self):
        # shorts pay only while the index is down, longs pay only while it is up: the regime rule must beat both unconditional sides
        df, down = self.frame(n=6000, seed=5)
        rng = np.random.default_rng(6)
        dn = down["base"]
        df["ret_long"] = np.where(dn, -0.03, 0.03) + rng.standard_normal(len(df)) * 0.01
        df["mkt_state"] = 1.0
        res, _ = R.evaluate_rules(df, down, 0.0024, 100, {}, 1_500_000_000)
        s = res["stats"]
        self.assertGreater(s["regime_rule"]["mean_bp_per_candidate"], 80.0)    # about 1/3 of events trade, each earning about 276 bp
        self.assertGreater(s["regime_rule"]["sr_daily"], s["long_all"]["sr_daily"])
        self.assertLess(s["short_all"]["mean_bp_per_candidate"], 0.0)


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_on_synthetic_candles(self):
        closes, df0 = synthetic_events()
        df = T.extend_events(df0, closes)
        t0 = df["t0"].to_numpy(np.int64)
        state, idx, _ = C.market_state(closes, t0)
        self.assertTrue(np.array_equal(state.astype(float), df["mkt_state"].to_numpy()))
        rets = R.index_lookback_returns(idx, t0)
        down = {k: R.mkt_down(rets, v) for k, v in R.LOOKBACK_SETS.items()}
        self.assertTrue(all(0 < d.mean() < 1 for d in down.values()))
        lo = int(df["t0"].min())
        res, daily = R.evaluate_rules(df, down, 0.0024, 100, {"w": (lo, lo + 60 * 86400)}, lo + 400 * 86400)
        self.assertEqual(set(res["stats"]), set(R.VARIANTS))
        w = res["window_down_share"]["w"]
        self.assertTrue(0 <= w["base"] <= 1)
        self.assertEqual(set(daily["variant"]), set(R.VARIANTS))


if __name__ == "__main__":
    unittest.main()
