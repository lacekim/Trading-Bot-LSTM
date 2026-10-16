import math
import unittest

import numpy as np
import pandas as pd

import meta_events as E
import meta_model as MM
import three_algo as T
from test_book_study_tools import make_closes


def synthetic_events(seed=5, n_sym=12, years=3):
    closes = make_closes(seed, 0.0, n_sym=n_sym, years=years)
    rng = np.random.default_rng(seed + 6)
    ohlc = {k_: pd.DataFrame({"high": c * np.exp(np.abs(rng.standard_normal(len(c))) * 0.004), "low": c * np.exp(-np.abs(rng.standard_normal(len(c))) * 0.004), "close": c.to_numpy()},
                             index=c.index) for k_, c in closes.items()}
    df, _ = E.build_meta_events(closes, ohlc, h=100, ptsl=(2.0, 2.0), cost_frac=0.0024, log=lambda m: None)
    return closes, df.sort_values(["t0", "symbol"], kind="stable").reset_index(drop=True)


class TestSignalsAndGate(unittest.TestCase):
    def test_mom_side_truth_table(self):
        r480 = np.array([1, 1, -1, -1, 1, np.nan, 0.0, 1])
        r720 = np.array([1, 1, -1, -1, -1, 1, 1, 1])
        r960 = np.array([1, -1, -1, -1, 1, 1, 1, np.nan])
        self.assertEqual(T.mom_side_from(r480, r720, r960).tolist(), [1, 0, -1, -1, 0, 0, 0, 0])

    def test_symmetric_barriers_make_short_exit_with_long(self):
        rng = np.random.default_rng(3)
        n, h = 500, 60
        q = np.cumsum(rng.standard_normal((n, h + 1)) * 0.6, axis=1).astype(np.float32)
        sigma = rng.uniform(0.005, 0.02, n)
        trgt = rng.uniform(0.02, 0.06, n)
        jl, kl, rl = E.triple_barrier(q, sigma, np.ones(n), trgt, (2.0, 2.0))
        js, ks, rs = E.triple_barrier(q, sigma, -np.ones(n), trgt, (2.0, 2.0))
        self.assertEqual(jl.tolist(), js.tolist())
        self.assertEqual(kl.tolist(), (-ks).tolist())
        np.testing.assert_allclose(rl, -rs, rtol=0, atol=1e-15)

    def test_gate_choice_cases(self):
        p_dir = np.array([0.7, 0.3, 0.5, 0.6, 0.45, 0.8])
        p_mom = np.array([0.9, 0.9, 0.9, 0.55, 0.40, 0.6])
        mom_side = np.array([-1, 1, 1, 1, -1, 0])
        side, algo, conf = T.gate_choice(p_dir, p_mom, mom_side)
        # 0: MOM (0.9 > 0.7) side -1; 1: MOM (0.9 > 0.7) side +1; 2: MOM 0.9 > 0.5; 3: LONG 0.6 beats MOM 0.55; 4: SHORT 0.55 beats MOM 0.40; 5: no MOM signal -> LONG 0.8
        self.assertEqual(side.tolist(), [-1, 1, 1, 1, -1, 1])
        self.assertEqual(algo.tolist(), [3, 3, 3, 1, 2, 1])
        np.testing.assert_allclose(conf, [0.9, 0.9, 0.9, 0.6, 0.55, 0.8])
        # ties go to MOM, then LONG
        side, algo, conf = T.gate_choice(np.array([0.7, 0.5]), np.array([0.7, 0.9]), np.array([-1, 0]))
        self.assertEqual(algo.tolist(), [3, 1])
        # inactive MOM never wins whatever p_mom is
        _, algo, _ = T.gate_choice(np.array([0.51]), np.array([0.99]), np.array([0]))
        self.assertEqual(algo.tolist(), [1])

    def test_variants_by_hand(self):
        p_dir = np.array([0.8, 0.2, 0.5, 0.55])
        p_mom = np.array([0.5, 0.5, 0.9, 0.7])
        mom_side = np.array([1, 0, -1, 1])
        state = np.array([1.0, 1.0, 0.0, 1.0])
        v = T.build_variants(p_dir, p_mom, mom_side, state, 0.1, np.random.default_rng(0))
        self.assertEqual(v["mom_all"][1].tolist(), [1.0, 0.0, 1.0, 1.0])
        self.assertEqual(v["mom_all"][0].tolist(), [1, 0, -1, 1])
        self.assertEqual(v["mom_long_up"][1].tolist(), [1.0, 0.0, 0.0, 1.0])               # long MOM events while the state is 1
        self.assertEqual(v["dir_model"][0].tolist(), [1, -1, -1, 1])
        want = MM.bet_size(np.array([0.8, 0.8, 0.5, 0.55]), 0.1)
        np.testing.assert_allclose(v["dir_model"][1], want)
        self.assertEqual(v["dir_model"][1][2], 0.0)                                        # confidence exactly 1/2: no bet
        np.testing.assert_allclose(v["mom_meta"][1], [0.0, 0.0, MM.bet_size(np.array([0.9]), 0.1)[0], MM.bet_size(np.array([0.7]), 0.1)[0]])
        self.assertEqual(v["mom_meta"][1][1], 0.0)                                         # no MOM signal: no mom_meta bet
        self.assertGreater(v["mom_meta"][1][2], 0.0)
        self.assertEqual(v["gate"][2].tolist()[2], 3)                                      # event 2: MOM (0.9)
        # nulls: same multiset of sizes / same sides as the gate
        self.assertEqual(sorted(v["gate_shuffled_size"][1].tolist()), sorted(v["gate"][1].tolist()))
        self.assertEqual(v["gate_shuffled_size"][0].tolist(), v["gate"][0].tolist())
        np.testing.assert_allclose(v["gate_random_side"][1], v["gate"][1])
        self.assertTrue(set(v["gate_random_side"][0].tolist()) <= {-1, 1})

    def test_best_days_share(self):
        x = np.array([1.0] * 30 + [10.0] * 2)
        self.assertAlmostEqual(T.best_days_share(x, 20), (2 * 10.0 + 18 * 1.0) / (30 * 1.0 + 2 * 10.0))
        self.assertTrue(math.isnan(T.best_days_share(np.array([-1.0, 2.0, -3.0]))))


class TestEvaluate(unittest.TestCase):
    @staticmethod
    def frame(n=900, seed=1):
        rng = np.random.default_rng(seed)
        t0 = np.sort(rng.integers(1_600_000_000, 1_600_000_000 + 200 * 86400, n))
        good = rng.random(n) < 0.5
        ret_long = np.where(good, 0.03, -0.03) + rng.standard_normal(n) * 0.01
        mom = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.3, 0.4])
        df = pd.DataFrame({"t0": t0, "ret_long": ret_long, "mkt_state": (rng.random(n) < 0.4).astype(float), "mom_side": mom})
        p_dir = np.clip(np.where(good, 0.7, 0.3) + rng.standard_normal(n) * 0.05, 0.01, 0.99)
        p_mom = np.clip(np.where(mom * ret_long > 0, 0.7, 0.3) + rng.standard_normal(n) * 0.05, 0.01, 0.99)
        return df, p_dir, p_mom, good

    def run_eval(self, df, p_dir, p_mom, n_paths=2, windows=None):
        preds = pd.DataFrame({"event": np.arange(len(df)), "split": 0, "path": np.arange(len(df)) % n_paths, "p_dir": p_dir, "p_mom": p_mom})
        return preds, T.evaluate_three(df, preds, 0.002, 0.1, 100, 5, windows or {"w": (1_600_000_000, 1_600_000_000 + 60 * 86400)}, 50)

    def test_matches_a_literal_loop(self):
        df, p_dir, p_mom, _ = self.frame()
        # one path only so that the events in the path are all events
        preds, ev = self.run_eval(df, p_dir, p_mom, n_paths=1)
        ret = df["ret_long"].to_numpy()
        mom = df["mom_side"].to_numpy()
        state = df["mkt_state"].to_numpy()
        n = len(df)
        self.assertAlmostEqual(ev["summary"]["long_all"]["mean_bp_per_candidate"], np.mean((ret - 0.002) * 1e4), places=6)
        self.assertAlmostEqual(ev["summary"]["short_all"]["mean_bp_per_candidate"], np.mean((-ret - 0.002) * 1e4), places=6)
        want_mom = np.mean([(mom[i] * ret[i] - 0.002) * 1e4 if mom[i] != 0 else 0.0 for i in range(n)])
        self.assertAlmostEqual(ev["summary"]["mom_all"]["mean_bp_per_candidate"], want_mom, places=6)
        want_up = np.mean([(ret[i] - 0.002) * 1e4 if (mom[i] == 1 and state[i] == 1.0) else 0.0 for i in range(n)])
        self.assertAlmostEqual(ev["summary"]["mom_long_up"]["mean_bp_per_candidate"], want_up, places=6)
        # gate by a literal loop
        tot = 0.0
        for i in range(n):
            cands = [(p_dir[i], 1, "LONG"), (1 - p_dir[i], -1, "SHORT")]
            if mom[i] != 0:
                cands = [(p_mom[i], int(mom[i]), "MOM")] + cands
            best = max(cands, key=lambda c: c[0])                                         # max() keeps the first of equal candidates
            sz = MM.bet_size(np.array([best[0]]), 0.1)[0]
            tot += sz * (best[1] * ret[i] - 0.002 * (sz > 0)) * 1e4
        self.assertAlmostEqual(ev["summary"]["gate"]["mean_bp_per_candidate"], tot / n, places=6)
        # conservation: the algorithms' contributions add up to the gate, by state and by window
        for s_ in (0, 1):
            parts = sum(ev["algo_state"][f"gate|{al}|state{s_}"]["bp_per_state_candidate"] for al in ("LONG", "SHORT", "MOM"))
            self.assertAlmostEqual(parts, ev["by_state"][f"gate|state{s_}"]["mean_bp_per_candidate"], places=6)
        parts = sum(ev["algo_window"][f"gate|{al}|w"]["bp_per_window_candidate"] for al in ("LONG", "SHORT", "MOM"))
        self.assertAlmostEqual(parts, ev["windows"]["gate|w"]["mean_bp_per_candidate"], places=6)
        d = ev["daily"]
        d = d[(d["variant"] == "long_all") & (d["path"] == 0)]
        self.assertAlmostEqual(d["sum_bp"].sum(), ((ret - 0.002) * 1e4).sum(), places=6)
        self.assertEqual(int(d["n"].sum()), n)

    def test_informative_probabilities_beat_the_nulls(self):
        df, p_dir, p_mom, _ = self.frame(n=6000, seed=3)
        _, ev = self.run_eval(df, p_dir, p_mom, n_paths=3)
        s = ev["summary"]
        self.assertGreater(s["gate"]["mean_sr_daily"], s["gate_random_side"]["mean_sr_daily"] + 0.2)         # the direction carries the result
        self.assertAlmostEqual(s["gate"]["mean_bp_per_candidate"], s["gate_shuffled_size"]["mean_bp_per_candidate"], delta=0.2 * abs(s["gate"]["mean_bp_per_candidate"]))
        self.assertGreater(s["gate"]["mean_sr_daily"], s["long_all"]["mean_sr_daily"] + 0.2)
        self.assertEqual(ev["compare"]["gate|gate_random_side"]["paths_sr_higher"], 3)

    def test_fixed_rules_do_not_depend_on_the_path(self):
        df, p_dir, p_mom, _ = self.frame(n=1200, seed=4)
        n = len(df)
        # two paths that each hold every event: the fixed rules must give identical paths, the gate (different p on each path) need not
        preds = pd.concat([pd.DataFrame({"event": np.arange(n), "split": 0, "path": 0, "p_dir": p_dir, "p_mom": p_mom}),
                           pd.DataFrame({"event": np.arange(n), "split": 1, "path": 1, "p_dir": 1 - p_dir, "p_mom": 1 - p_mom})], ignore_index=True)
        ev = T.evaluate_three(df, preds, 0.002, 0.1, 100, 5, {}, 50)
        pp = pd.DataFrame(ev["paths"])
        for v in ("long_all", "short_all", "mom_all", "mom_long_up"):
            x = pp[pp["variant"] == v]["sr_daily"].to_numpy()
            self.assertTrue(np.allclose(x, x[0]), v)
        g = pp[pp["variant"] == "gate"]["sr_daily"].to_numpy()
        self.assertFalse(np.allclose(g, g[0]))

    def test_dsr_trial_table(self):
        df, p_dir, p_mom, _ = self.frame(n=3000, seed=6)
        _, ev = self.run_eval(df, p_dir, p_mom, n_paths=3)
        for v in T.DSR_VARIANTS:
            self.assertEqual(sorted(ev["dsr"][v]["by_trials"], key=int), ["1", "10", "50", "51"])
            ds = [ev["dsr"][v]["by_trials"][k]["dsr"] for k in ("1", "10", "50", "51")]
            self.assertTrue(all(ds[i] >= ds[i + 1] - 1e-12 for i in range(3)), ds)         # more trials never raise the deflated Sharpe


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.closes, df = synthetic_events()
        cls.df0 = df

    def test_extend_events_and_forests(self):
        df = T.extend_events(self.df0, self.closes)
        # trailing returns against a direct computation
        for i in np.random.default_rng(1).choice(len(df), 60, replace=False):
            r = df.iloc[i]
            c = self.closes[r["symbol"]]
            c = c[c > 0]
            logc = np.log(c.to_numpy())
            p = int(r["pos"])
            for L, col in ((480, "ret480"), (720, "ret720"), (960, "ret960")):
                if p >= L:
                    self.assertAlmostEqual(r[col], logc[p] - logc[p - L], places=12)
                else:
                    self.assertTrue(np.isnan(r[col]))
        has = df["ret720"].notna()
        self.assertTrue((np.sign(df.loc[has, "ret720"]) == df.loc[has, "side"]).all())     # the source side is the sign of the 30-day return
        np.testing.assert_allclose(df["ret_long"], df["side"] * df["ret_side"])
        self.assertTrue((df["y_dir"] == (df["ret_long"] > 0)).all())
        act = df["mom_side"] != 0
        self.assertTrue((df.loc[act, "mom_side"] == df.loc[act, "side"]).all())
        self.assertGreater(int(act.sum()), 500)
        preds, met, meta = T.run_cpcv_gate(df, 4, 2, 0.01, dict(n_estimators=5, bootstrap="sequential", max_draws=1000), 1, log=lambda m: None)
        self.assertEqual(meta["n_paths"], 3)
        self.assertEqual(len(preds), len(df) * 3)
        self.assertTrue(((preds["p_dir"] >= 0) & (preds["p_dir"] <= 1) & (preds["p_mom"] >= 0) & (preds["p_mom"] <= 1)).all())
        self.assertTrue((met["n_train_mom"] <= met["n_train_dir"]).all())
        self.assertEqual(preds.groupby(["path", "event"]).size().max(), 1)                 # each event once per path
        ev = T.evaluate_three(df, preds, 0.0024, 0.1, 100, 3, {"w": (int(df["t0"].min()), int(df["t0"].min()) + 30 * 86400)}, 60)
        self.assertEqual(set(ev["summary"]), set(T.VARIANTS))
        wp, wrow = T.run_walk_forward_gate(df, pd.Timestamp(int(df["t0"].median()), unit="s"), dict(n_estimators=5, bootstrap="sequential", max_draws=1000), 2, log=lambda m: None)
        self.assertGreater(len(wp), 0)
        self.assertTrue((wp["event"].to_numpy() >= 0).all())

    def test_planted_direction_is_found_and_scored(self):
        df = self.df0.copy()
        rng = np.random.default_rng(9)
        sgn = np.sign(df["mom24n"].to_numpy())
        ret_long = 0.03 * sgn + rng.standard_normal(len(df)) * 0.01                         # direction fully set by a feature the forest can see
        df["ret_side"] = df["side"].to_numpy() * ret_long
        df["bin"] = (df["ret_side"] > 0).astype(np.int8)
        df = T.extend_events(df, self.closes)
        preds, met, meta = T.run_cpcv_gate(df, 4, 2, 0.01, dict(n_estimators=8, bootstrap="sequential", max_draws=1500), 2, log=lambda m: None)
        self.assertGreater(met["auc_dir"].mean(), 0.9)
        ev = T.evaluate_three(df, preds, 0.0024, 0.1, 100, 3, {}, 60)
        s = ev["summary"]
        self.assertGreater(s["dir_model"]["mean_sr_daily"], s["long_all"]["mean_sr_daily"] + 0.5)
        self.assertGreater(s["gate"]["mean_sr_daily"], s["gate_random_side"]["mean_sr_daily"] + 0.5)
        self.assertEqual(ev["compare"]["gate|gate_random_side"]["paths_sr_higher"], 3)
        self.assertGreater(s["gate"]["mean_bp_per_candidate"], 100.0)


if __name__ == "__main__":
    unittest.main()
