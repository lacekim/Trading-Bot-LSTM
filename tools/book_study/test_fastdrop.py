import math
import unittest

import numpy as np
import pandas as pd

import book_study_core as C
import fastdrop_algo as F
import regime_algo as R
import three_algo as T
from test_three_algo import synthetic_events


def make_index(seed=1, n_hours=6000, drop_gaps=True):
    rng = np.random.default_rng(seed)
    hours = np.arange(n_hours)
    if drop_gaps:
        hours = np.sort(rng.choice(hours, int(n_hours * 0.8), replace=False))
    t_idx = pd.to_datetime(hours * 3600 + 1_600_000_000, unit="s")
    r = rng.standard_normal(len(hours)) * 0.01
    return pd.Series(np.exp(np.cumsum(r)), index=t_idx), hours


class TestSigma(unittest.TestCase):
    def test_sigma_matches_a_literal_loop_with_gaps(self):
        idx, hours = make_index()
        keys = hours * 3600 + 1_600_000_000
        rng = np.random.default_rng(2)
        t0 = rng.choice(keys[300:], 200, replace=False) + rng.integers(0, 3500, 200)       # event times between index hours
        got = F.index_sigma_at(idx, t0, window=720)
        lv = np.log(idx.to_numpy())
        for i, t in enumerate(t0):
            now = max(j for j in range(len(keys)) if keys[j] <= t)
            if now - 720 + 1 < 1:                                                             # needs 720 returns, the first observation has none
                self.assertTrue(math.isnan(got[i]))
                continue
            r = [lv[j] - lv[j - 1] for j in range(now - 719, now + 1)]
            m = sum(r) / 720
            want = math.sqrt(sum((x - m) ** 2 for x in r) / 719)
            self.assertAlmostEqual(got[i], want, places=12)

    def test_nan_before_enough_history_and_before_the_index(self):
        idx, hours = make_index(drop_gaps=False, n_hours=1000)
        t0 = np.array([1_600_000_000 - 3600, 1_600_000_000 + 700 * 3600, 1_600_000_000 + 720 * 3600, 1_600_000_000 + 900 * 3600])
        got = F.index_sigma_at(idx, t0)
        self.assertTrue(np.isnan(got[0]) and np.isnan(got[1]))
        self.assertTrue(np.isfinite(got[2]) and np.isfinite(got[3]))                        # 720 returns exist from observation 720


class TestZ(unittest.TestCase):
    def test_z_and_flag(self):
        ret = np.array([-0.25, -0.2, np.nan, -0.2, 0.1])
        sig = np.array([0.01, np.nan, 0.01, 0.0, 0.01])
        z = F.fast_z(ret, sig, 100)
        self.assertAlmostEqual(z[0], -0.25 / (0.01 * 10), places=12)
        self.assertTrue(np.isnan(z[1]) and np.isnan(z[2]) and np.isnan(z[3]))
        self.assertEqual(F.fast_flag(z).tolist(), [True, False, False, False, False])
        self.assertEqual(F.fast_flag(np.array([-2.0, -2.0000001, -1.99])).tolist(), [False, True, False])     # strict < -2


class TestRules(unittest.TestCase):
    def test_rules_by_hand(self):
        mom = np.array([1, 1, -1, -1, 0, 1, -1])
        state = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        a = np.array([False, True, True, True, True, False, True])
        fast = {"base": a, "short": ~a, "long": np.zeros(7, dtype=bool)}
        r = F.build_rules(mom, state, fast)
        self.assertEqual(r["mom_long_up"].tolist(), [1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(r["long_up_nofast"].tolist(), [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(r["short_fast"].tolist(), [0, -1, -1, -1, -1, 0, -1])
        self.assertEqual(r["short_fast_mom"].tolist(), [0, 0, -1, -1, 0, 0, -1])
        self.assertEqual(r["fast_rule"].tolist(), [1, 0, -1, -1, 0, 0, -1])
        self.assertEqual(r["fast_rule_24"].tolist(), [0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(r["fast_rule_168"].tolist(), [1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(r["long_all"].tolist(), [1] * 7)
        self.assertEqual(r["mom_all"].tolist(), mom.tolist())


class TestEvaluate(unittest.TestCase):
    @staticmethod
    def frame(n=2500, seed=2):
        rng = np.random.default_rng(seed)
        t0 = np.sort(rng.integers(1_500_000_000, 1_500_000_000 + 900 * 86400, n))
        df = pd.DataFrame({"t0": t0, "ret_long": rng.standard_normal(n) * 0.03 + 0.001, "mkt_state": (rng.random(n) < 0.5).astype(float),
                           "mom_side": rng.choice([-1, 0, 1], size=n)})
        fast = {k: rng.random(n) < p for k, p in (("base", 0.3), ("short", 0.2), ("long", 0.4))}
        z = {k: np.where(v, -3.0, 0.5) - rng.random(n) * 0.1 for k, v in fast.items()}
        return df, z, fast

    def test_matches_a_literal_loop(self):
        df, z, fast = self.frame()
        win = {"sell_off_w": (1_500_000_000 + 100 * 86400, 1_500_000_000 + 200 * 86400)}
        cut = 1_500_000_000 + 450 * 86400
        res, daily = F.evaluate_fast(df, z, fast, 0.0024, 100, win, cut)
        sides = F.build_rules(df["mom_side"].to_numpy(), df["mkt_state"].to_numpy(), fast)
        ret, t0, st, mom = df["ret_long"].to_numpy(), df["t0"].to_numpy(), df["mkt_state"].to_numpy(), df["mom_side"].to_numpy()
        n = len(df)
        for v in F.VARIANTS:
            s = sides[v]
            pnl = [(s[i] * ret[i] - 0.0024) * 1e4 if s[i] != 0 else 0.0 for i in range(n)]
            self.assertAlmostEqual(res["stats"][v]["mean_bp_per_candidate"], float(np.mean(pnl)), places=6)
            idx = [i for i in range(n) if win["sell_off_w"][0] <= t0[i] < win["sell_off_w"][1]]
            self.assertAlmostEqual(res["windows"][f"{v}|sell_off_w"]["mean_bp_per_candidate"], float(np.mean([pnl[i] for i in idx])), places=6)
            for s_ in (0, 1):
                for fn in (0, 1):
                    ii = [i for i in range(n) if st[i] == s_ and fast["base"][i] == bool(fn)]
                    self.assertAlmostEqual(res["cells"][f"{v}|state{s_}|fast{fn}"]["mean_bp_per_candidate"], float(np.mean([pnl[i] for i in ii])), places=6)
                    self.assertEqual(res["cells"][f"{v}|state{s_}|fast{fn}"]["trades"], sum(1 for i in ii if s[i] != 0))
            dd = daily[daily["variant"] == v]
            self.assertAlmostEqual(dd["sum_bp"].sum(), sum(pnl), places=5)
        # removed versus kept table, literal
        pl = [(ret[i] - 0.0024) * 1e4 if (mom[i] == 1 and st[i] == 1.0) else 0.0 for i in range(n)]
        rem = [i for i in range(n) if mom[i] == 1 and st[i] == 1.0 and fast["base"][i]]
        kept = [i for i in range(n) if mom[i] == 1 and st[i] == 1.0 and not fast["base"][i]]
        f = res["filter"]["all"]
        self.assertEqual((f["removed"], f["kept"]), (len(rem), len(kept)))
        self.assertEqual(f["removed_days"], len({t0[i] // 86400 for i in rem}))
        self.assertAlmostEqual(f["removed_bp"], float(np.mean([pl[i] for i in rem])), places=6)
        self.assertAlmostEqual(f["kept_bp"], float(np.mean([pl[i] for i in kept])), places=6)
        self.assertEqual(sum(res["filter"][k]["removed"] + res["filter"][k]["kept"] for k in res["filter"] if k.startswith("y")), f["removed"] + f["kept"])
        # sell-off day table, literal
        days = sorted({t0[i] // 86400 for i in idx})
        self.assertEqual(sorted(int(k) for k in res["selloff_days"]), days)
        for dd_ in days:
            ii = [i for i in idx if t0[i] // 86400 == dd_]
            x = res["selloff_days"][str(dd_)]
            self.assertEqual(x["events"], len(ii))
            self.assertAlmostEqual(x["fast_share"], float(np.mean([fast["base"][i] for i in ii])), places=12)
            self.assertAlmostEqual(x["min_z72"], float(min(z["base"][i] for i in ii)), places=12)
            self.assertAlmostEqual(x["mom_long_up_bp"], float(np.mean([pl[i] for i in ii])), places=6)

    def test_dsr_tables_fall_with_more_trials(self):
        df, z, fast = self.frame(seed=3)
        res, _ = F.evaluate_fast(df, z, fast, 0.0024, 100, {}, 1_500_000_000 + 450 * 86400)
        for v in F.NEW_VARIANTS:
            t = res["dsr"][v]["by_trials"]
            self.assertEqual(sorted(t, key=int), ["1", "10", "94", "100"])
            ds = [t[k]["dsr"] for k in ("1", "10", "94", "100")]
            self.assertTrue(all(ds[i] >= ds[i + 1] - 1e-12 for i in range(3)), ds)

    def test_a_planted_fast_drop_is_found(self):
        df, z, fast = self.frame(n=6000, seed=5)
        rng = np.random.default_rng(6)
        fb = fast["base"]
        df["ret_long"] = np.where(fb, -0.03, 0.03) + rng.standard_normal(len(df)) * 0.01
        df["mkt_state"] = 1.0
        res, _ = F.evaluate_fast(df, z, fast, 0.0024, 100, {}, 1_500_000_000)
        s = res["stats"]
        self.assertGreater(s["fast_rule"]["mean_bp_per_candidate"], 80.0)
        self.assertGreater(s["long_up_nofast"]["mean_bp_per_candidate"], s["mom_long_up"]["mean_bp_per_candidate"])
        self.assertLess(res["filter"]["all"]["removed_bp"], 0.0)


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_on_synthetic_candles(self):
        closes, df0 = synthetic_events()
        df = T.extend_events(df0, closes)
        t0 = df["t0"].to_numpy(np.int64)
        state, idx, _ = C.market_state(closes, t0)
        rets = R.index_lookback_returns(idx, t0, F.ALL_HORIZONS)
        sigma = F.index_sigma_at(idx, t0)
        z = {k: F.fast_z(rets[L_], sigma, L_) for k, L_ in F.HORIZONS.items()}
        fast = {k: F.fast_flag(v) for k, v in z.items()}
        self.assertTrue(np.isfinite(sigma).mean() > 0.5)
        lo = int(df["t0"].min())
        res, daily = F.evaluate_fast(df, z, fast, 0.0024, 100, {"sell_off_w": (lo + 400 * 86400, lo + 460 * 86400)}, lo + 400 * 86400)
        self.assertEqual(set(res["stats"]), set(F.VARIANTS))
        self.assertEqual(set(daily["variant"]), set(F.VARIANTS))
        # a wider horizon with the same sigma: z scales by 1/sqrt(L); same inputs give the same flags twice
        self.assertTrue(np.array_equal(F.fast_flag(z["base"]), fast["base"]))


if __name__ == "__main__":
    unittest.main()
