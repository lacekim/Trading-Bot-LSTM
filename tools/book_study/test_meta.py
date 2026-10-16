import math
import unittest

import numpy as np
import pandas as pd

import book_stats as B
import meta_eval as ME
import meta_events as E
import meta_model as MM
from test_book_study_tools import make_closes


def literal_barrier(path_logq, sigma, side, trgt, ptsl, h):
    """Book loop: step by step, strict crossings, vertical barrier at the last step."""
    for k in range(h + 1):
        r = math.expm1(path_logq[k] * sigma) * side
        if r > ptsl[0] * trgt:
            return k, 1, r
        if r < -ptsl[1] * trgt:
            return k, -1, r
    r = math.expm1(path_logq[h] * sigma) * side
    return h, 0, r


class TestLabels(unittest.TestCase):
    def test_triple_barrier_matches_a_literal_loop(self):
        rng = np.random.default_rng(1)
        n, h = 400, 60
        q = np.cumsum(rng.standard_normal((n, h + 1)) * 0.6, axis=1).astype(np.float32)
        sigma = rng.uniform(0.005, 0.02, n)
        side = rng.choice([-1.0, 1.0], n)
        trgt = rng.uniform(0.02, 0.06, n)
        j, kind, ret = E.triple_barrier(q, sigma, side, trgt, (2.0, 1.5))
        for i in range(n):
            jj, kk, rr = literal_barrier(q[i].astype(np.float64), sigma[i], side[i], trgt[i], (2.0, 1.5), h)
            self.assertEqual((j[i], kind[i]), (jj, kk), i)
            self.assertAlmostEqual(ret[i], rr, places=10)
        self.assertTrue({1, -1, 0} <= set(kind.tolist()))            # all three outcomes exercised

    def test_touching_exactly_is_not_a_hit(self):
        # a path whose side-signed simple return equals the barrier exactly at step 3 must not exit there
        trgt = np.array([0.05])
        sigma = np.array([0.01])
        target = math.log1p(2.0 * 0.05) / 0.01
        q = np.array([[0.0, 0.0, target, 0.0, 0.0]], dtype=np.float64)
        j, kind, _ = E.triple_barrier(q, sigma, np.array([1.0]), trgt, (2.0, 2.0))
        self.assertEqual((int(j[0]), int(kind[0])), (4, 0))

    def test_attribution_weights_match_a_literal_loop(self):
        rng = np.random.default_rng(2)
        n = 300
        logc = np.cumsum(rng.standard_normal(n) * 0.01)
        pos = np.sort(rng.choice(n - 40, 25, replace=False))
        t1 = pos + rng.integers(3, 35, size=len(pos))
        u, w = E.attribution_weights(pos, t1, logc)
        r = np.concatenate([[0.0], np.diff(logc)])
        conc = np.zeros(n)
        for a, b in zip(pos, t1):
            conc[a:b + 1] += 1
        for i, (a, b) in enumerate(zip(pos, t1)):
            self.assertAlmostEqual(u[i], np.mean(1.0 / conc[a:b + 1]), places=12)
            self.assertAlmostEqual(w[i], abs(np.sum(r[a:b + 1] / conc[a:b + 1])), places=12)

    def test_daily_vol_matches_an_explicit_ewm(self):
        import verify_book_study as V
        rng = np.random.default_rng(3)
        c = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(600) * 0.01)))
        got = E.daily_vol(c)
        r = [math.nan] * 24 + [c.iloc[i] / c.iloc[i - 24] - 1 for i in range(24, len(c))]
        want = V.ewm_sigma(r, 100)
        ok = np.isfinite(want)
        np.testing.assert_allclose(got[ok], np.array(want)[ok], rtol=1e-9)
        self.assertEqual(int(np.isfinite(got).sum()), int(ok.sum()))


class TestSequentialBootstrap(unittest.TestCase):
    def test_probabilities_follow_the_book_snippet(self):
        # Snippets 4.3-4.6 example: events spanning bars [0,2], [2,3], [4,5]
        starts, ends = np.array([0, 2, 4]), np.array([2, 3, 5])
        n_bars = 6

        def literal_probs(phi):
            avg = []
            for i in range(3):
                cols = phi + [i]
                ind = np.zeros((n_bars, len(cols)))
                for c_, e in enumerate(cols):
                    ind[starts[e]:ends[e] + 1, c_] = 1.0
                conc = ind.sum(axis=1)
                u = np.where(ind > 0, ind / np.where(conc > 0, conc, 1.0)[:, None], np.nan)
                avg.append(np.nanmean(u[:, -1]))
            p = np.array(avg)
            return p / p.sum()
        rng = np.random.default_rng(7)
        runs = np.array([MM.seq_bootstrap_one_symbol(starts, ends, 2, rng) for _ in range(30000)])
        for first in range(3):
            sel = runs[runs[:, 0] == first][:, 1]
            freq = np.bincount(sel, minlength=3) / len(sel)
            np.testing.assert_allclose(freq, literal_probs([first]), atol=0.012)
        np.testing.assert_allclose(np.bincount(runs[:, 0], minlength=3) / len(runs), literal_probs([]), atol=0.012)

    def test_raises_sample_uniqueness_over_ordinary_bootstrap(self):
        # 300 events of 20-59 bars over 3000 bars, 100 draws: far fewer draws than the data could absorb without overlap, so the sequential
        # procedure has room to avoid overlap (with ~1500 events and 600 draws every bar is covered ~8 times either way and there is no
        # room: measured 0.123 vs 0.123 in that setup, and the fast sampler matched the literal one draw for draw there).
        gains = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            n = 300
            starts = np.sort(rng.integers(0, 3000, n))
            ends = starts + rng.integers(20, 60, n)
            sym = np.zeros(n, dtype=np.int64)
            seq = MM.seq_bootstrap(sym, starts, ends, 100, rng)
            std = rng.integers(0, n, size=100)
            gains.append(MM.sample_avg_uniqueness(sym, starts, ends, seq) - MM.sample_avg_uniqueness(sym, starts, ends, std))
        self.assertGreater(min(gains), 0.04)
        self.assertGreater(float(np.mean(gains)), 0.08)

    def test_fast_sampler_matches_literal_incremental_procedure(self):
        # same uniform stream, literal recomputation of every candidate's average uniqueness after every draw
        rng = np.random.default_rng(3)
        n = 120
        starts = np.sort(rng.integers(0, 600, n))
        ends = starts + rng.integers(5, 30, n)
        fast = MM.seq_bootstrap_one_symbol(starts, ends, 40, np.random.default_rng(11))
        rr = np.random.default_rng(11)
        draws = []
        for _ in range(40):
            c = np.zeros(int(ends.max()) + 2)
            for j in draws:
                c[starts[j]:ends[j] + 1] += 1
            avg = np.array([np.mean(1.0 / (c[starts[i]:ends[i] + 1] + 1.0)) for i in range(n)])
            cum = np.cumsum(avg)
            draws.append(int(np.searchsorted(cum, rr.random() * cum[-1], side="right")))
        self.assertEqual(fast.tolist(), draws)

    def test_draw_counts_add_up_across_instruments(self):
        rng = np.random.default_rng(5)
        sym = np.repeat([0, 1, 2], [500, 300, 200])
        starts = np.concatenate([np.sort(rng.integers(0, 2000, k)) for k in (500, 300, 200)])
        ends = starts + 30
        idx = MM.seq_bootstrap(sym, starts, ends, 1000, rng)
        self.assertAlmostEqual(len(idx), 1000, delta=3)
        self.assertEqual(np.bincount(sym[idx]).tolist(), [500, 300, 200])


class TestForestAndImportance(unittest.TestCase):
    @staticmethod
    def synth(n=4000, seed=0, signal=True):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 4))
        logit = 1.6 * X[:, 0] if signal else np.zeros(n)
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
        sym = rng.integers(0, 4, n)
        starts = np.sort(rng.integers(0, 20000, n))
        ends = starts + 5
        t0 = starts.astype(float) * 3600
        t1 = ends.astype(float) * 3600
        return X, y, np.ones(n), sym, starts, ends, t0, t1

    def test_forest_learns_a_planted_signal_and_not_noise(self):
        from sklearn.metrics import roc_auc_score
        for signal, lo, hi in ((True, 0.70, 1.0), (False, 0.44, 0.56)):
            X, y, w, sym, st, en, t0, t1 = self.synth(signal=signal)
            tr, te = np.arange(0, 3000), np.arange(3000, 4000)
            f = MM.BaggedForest(n_estimators=30, bootstrap="sequential", max_draws=1500, seed=1).fit(X[tr], y[tr], w[tr], sym[tr], st[tr], en[tr], 0.8)
            auc = roc_auc_score(y[te], f.predict_proba1(X[te]))
            self.assertTrue(lo <= auc <= hi, (signal, auc))

    def test_purged_kfold_leaves_no_overlap_and_no_embargoed_events(self):
        rng = np.random.default_rng(6)
        t0 = np.sort(rng.integers(0, 10 ** 6, 800)).astype(float)
        t1 = t0 + rng.integers(1, 4 * 10 ** 4, 800)
        seen = []
        for tr, te in MM.purged_kfold(t0, t1, 5, 0.02):
            lo, hi = t0[te].min(), t1[te].max()
            self.assertFalse(((t0[tr] <= hi) & (t1[tr] >= lo)).any())
            emb = 0.02 * (t0.max() + 1 - t0.min())
            self.assertFalse(((t0[tr] > hi) & (t0[tr] <= hi + emb)).any())
            self.assertEqual(len(np.intersect1d(tr, te)), 0)
            seen += te.tolist()
        self.assertEqual(sorted(seen), list(range(800)))

    def test_mda_and_sfi_rank_the_planted_feature_first(self):
        X, y, w, sym, st, en, t0, t1 = self.synth(n=5000, seed=3)
        mk = lambda: MM.BaggedForest(n_estimators=20, bootstrap="sequential", max_draws=1500, seed=2)                    # noqa: E731
        names = ["signal", "n1", "n2", "n3"]
        mda, base = MM.mda_importance(X, y, w, sym, st, en, lambda tr: 0.8, t0, t1, names, 4, 0.01, mk, 0)
        self.assertEqual(mda["mean"].idxmax(), "signal")
        self.assertGreater(mda.loc["signal", "mean"], 5 * abs(mda.loc[["n1", "n2", "n3"], "mean"]).max())
        sfi = MM.sfi_importance(X, y, w, sym, st, en, lambda tr: 0.8, t0, t1, names, 4, 0.01, mk)
        self.assertEqual(sfi["mean"].idxmax(), "signal")

    def test_bet_size_formula(self):
        self.assertEqual(MM.bet_size([0.5, 0.45, 0.2]).tolist(), [0.0, 0.0, 0.0])
        p = 0.8
        z = (p - 0.5) / math.sqrt(p * (1 - p))
        want = 2 * 0.5 * (1 + math.erf(z / math.sqrt(2))) - 1
        self.assertAlmostEqual(float(MM.bet_size([p])[0]), want, places=9)
        s = MM.bet_size(np.array([0.55, 0.7, 0.99, 0.6, 0.65, 0.75]), 0.1)
        self.assertTrue(np.allclose(s / 0.1, np.round(s / 0.1)))
        self.assertLessEqual(float(s.max()), 1.0)
        self.assertTrue((np.diff(MM.bet_size(np.linspace(0.51, 0.99, 30))) >= 0).all())


class TestEvaluation(unittest.TestCase):
    @staticmethod
    def frame(n=6000, seed=0):
        rng = np.random.default_rng(seed)
        t0 = np.sort(rng.integers(1_600_000_000, 1_600_000_000 + 400 * 86400, n))
        good = rng.random(n) < 0.5
        ret = np.where(good, 0.03, -0.03) + rng.standard_normal(n) * 0.01
        state = (rng.random(n) < 0.4).astype(float)
        side = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        return pd.DataFrame({"t0": t0, "ret_side": ret, "mkt_state": state, "side": side}), good

    def test_informative_probabilities_beat_the_shuffled_sizes(self):
        df, good = self.frame()
        rng = np.random.default_rng(1)
        p = np.where(good, 0.75, 0.3) + rng.standard_normal(len(df)) * 0.02
        preds = pd.DataFrame({"event": np.arange(len(df)), "split": 0, "path": np.arange(len(df)) % 3, "p": p})
        ev = ME.evaluate(df, preds, 0.002, 0.1, 100, 5, {"w": (1_600_000_000, 1_610_000_000)}, 50)
        s = ev["summary"]
        self.assertGreater(s["meta"]["mean_sr_daily"], s["placebo"]["mean_sr_daily"] + 0.3)
        self.assertGreater(s["meta"]["mean_sr_daily"], s["primary"]["mean_sr_daily"])
        self.assertEqual(s["meta_minus_placebo"]["paths_sr_higher"], 3)

    def test_uninformative_probabilities_do_not(self):
        df, good = self.frame(seed=2)
        rng = np.random.default_rng(3)
        p = 0.5 + rng.random(len(df)) * 0.4
        preds = pd.DataFrame({"event": np.arange(len(df)), "split": 0, "path": np.arange(len(df)) % 3, "p": p})
        ev = ME.evaluate(df, preds, 0.002, 0.1, 100, 5, {"w": (1_600_000_000, 1_610_000_000)}, 50)
        self.assertLess(abs(ev["summary"]["meta_minus_placebo"]["mean_sr_daily_diff"]), 0.12)

    def test_by_state_table_matches_a_literal_loop(self):
        df, good = self.frame(n=1200, seed=7)
        rng = np.random.default_rng(8)
        p = np.where(good, 0.7, 0.35) + rng.standard_normal(len(df)) * 0.05
        preds = pd.DataFrame({"event": np.arange(len(df)), "split": 0, "path": np.arange(len(df)) % 2, "p": p})
        ev = ME.evaluate(df, preds, 0.002, 0.1, 100, 5, {"w": (1_600_000_000, 1_600_000_000 + 100 * 86400)}, 50)
        bs = ev["by_state"]
        for stv in (0, 1):
            idx = [i for i in range(len(df)) if df["mkt_state"][i] == stv]
            want_primary = np.mean([(df["ret_side"][i] - 0.002) * 1e4 for i in idx])
            self.assertAlmostEqual(bs[f"primary|state{stv}"]["mean_bp_per_candidate"], want_primary, places=6)
            sz = {i: MM.bet_size(np.array([p[i]]), 0.1)[0] for i in idx}
            want_meta = np.mean([sz[i] * (df["ret_side"][i] - 0.002 * (sz[i] > 0)) * 1e4 for i in idx])
            self.assertAlmostEqual(bs[f"meta|state{stv}"]["mean_bp_per_candidate"], want_meta, places=6)
            self.assertAlmostEqual(bs[f"primary|state{stv}"]["events_per_path"], len(idx) / 2, places=9)
        n_win = sum(1 for t in df["t0"] if 1_600_000_000 <= t < 1_600_000_000 + 100 * 86400)
        self.assertAlmostEqual(sum(v for k, v in ev["window_state_events"].items() if k.startswith("w|")), n_win / 2, places=9)

    def test_primary_pnl_identity(self):
        df, _ = self.frame(n=900, seed=4)
        preds = pd.DataFrame({"event": np.arange(len(df)), "split": 0, "path": 0, "p": np.full(len(df), 0.6)})
        ev = ME.evaluate(df, preds, 0.002, 0.1, 100, 5, {}, 50)
        d = ev["daily"]
        d = d[(d["variant"] == "primary") & (d["path"] == 0)]
        self.assertAlmostEqual(d["sum_bp"].sum(), ((df["ret_side"] - 0.002) * 1e4).sum(), places=6)
        self.assertEqual(int(d["n"].sum()), len(df))


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_on_synthetic_candles(self):
        closes = make_closes(5, 0.0, n_sym=12, years=3)
        rng = np.random.default_rng(11)
        ohlc = {k_: pd.DataFrame({"high": c * np.exp(np.abs(rng.standard_normal(len(c))) * 0.004), "low": c * np.exp(-np.abs(rng.standard_normal(len(c))) * 0.004), "close": c.to_numpy()},
                                 index=c.index) for k_, c in closes.items()}
        df, info = E.build_meta_events(closes, ohlc, h=100, ptsl=(2.0, 2.0), cost_frac=0.0024, log=lambda m: None)
        self.assertGreater(len(df), 3000)
        self.assertTrue(np.isfinite(df[E.FEATURES].to_numpy()).all())
        self.assertAlmostEqual(float(df["w"].mean()), 1.0, places=9)
        self.assertTrue(((df["u"] > 0) & (df["u"] <= 1)).all())
        self.assertTrue((df["t1"] > df["t0"]).all())
        preds, met, meta = ME.run_cpcv(df.sort_values("t0").reset_index(drop=True), 4, 2, 0.01, dict(n_estimators=5, bootstrap="sequential", max_draws=1000), 1, log=lambda m: None)
        self.assertEqual(meta["n_paths"], 3)
        self.assertEqual(len(preds), len(df) * 3)
        self.assertTrue(((preds["p"] >= 0) & (preds["p"] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
