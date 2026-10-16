"""Integration tests for the book-study pipeline on synthetic hourly candles (no project files needed)."""
import unittest

import numpy as np
import pandas as pd

import book_study_core as C

PARAMS = dict(n_groups=6, k=2, horizons=[8], n_paths=3000, embargo_pct=0.01, seed=7, cost_frac=0.0024,
              min_train=200, min_test=50, n_sample=5)
QUIET = lambda *_: None


def make_closes(seed, drift, n_sym=14, years=3):
    rng = np.random.default_rng(seed)
    n = 24 * 365 * years
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    common = rng.standard_normal(n) * 0.004
    return {f"S{i}": pd.Series(100 * np.exp(np.cumsum(common * 0.7 + rng.standard_normal(n) * 0.006 + drift)), index=idx)
            for i in range(n_sym)}


def run(closes):
    ev = C.build_events(closes, 8, log=QUIET)
    state, _, _ = C.market_state(closes, ev["t0"])
    return ev, state, C.run_study(ev, state, PARAMS, log=QUIET)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev0, cls.st0, cls.res0 = run(make_closes(5, 0.0))
        cls.ev1, cls.st1, cls.res1 = run(make_closes(11, 0.0006))
        cls.sum0 = C.summarize(cls.res0, {"w": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01"))})
        cls.sum1 = C.summarize(cls.res1, {"w": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01"))})

    def test_no_edge_control_has_no_gross_edge_and_loses_after_costs(self):
        for key, r in self.sum0["rows"].items():
            self.assertLess(abs(r["mean_gross_bp"]), 12.0, key)      # drift-free walk: gross near zero
            self.assertLess(r["mean_net_bp"], 0.0, key)              # costs make every row negative

    def test_planted_drift_is_found_in_long_not_short_not_placebo(self):
        long_ = self.sum1["rows"]["H8|all|long"]
        short = self.sum1["rows"]["H8|all|short"]
        plc = self.sum1["rows"]["H8|all|placebo"]
        self.assertGreater(long_["mean_gross_bp"], 30.0)
        self.assertLess(short["mean_gross_bp"], 5.0)
        self.assertLess(abs(plc["mean_gross_bp"]), 5.0)
        self.assertEqual(long_["paths_sr_net_negative"], 0)

    def test_deterministic_for_a_seed(self):
        _, _, again = run(make_closes(5, 0.0))
        a, b = self.res0["cells"].drop(columns=["half_life_bars"], errors="ignore"), again["cells"].drop(columns=["half_life_bars"], errors="ignore")
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_every_path_holds_every_event_once(self):
        d = self.res0["daily"]
        tot = d[(d.state == "all") & (d.direction == "long")].groupby("path")["n"].sum()
        self.assertEqual(tot.nunique(), 1)
        self.assertEqual(int(tot.iloc[0]), int((self.res0["events_index"]["group"] >= 0).sum()))

    def test_training_never_touches_test_events(self):
        from book_stats import train_mask
        ev = self.ev0
        t0 = ev["t0"].astype(float)
        t1 = t0 + 9 * 3600.0
        from book_stats import group_edges, assign_groups
        edges = group_edges(t0.min(), t0.max() + 1, 6)
        g = assign_groups(t0, edges)
        tr = train_mask(t0, t1, g, (1, 4), edges, 0.01 * (t0.max() - t0.min()))
        self.assertFalse(np.isin(g[tr], [1, 4]).any())
        # no training event's label window overlaps either test group's span
        for tg in (1, 4):
            self.assertFalse(((t0[tr] <= edges[tg + 1]) & (t1[tr] >= edges[tg])).any())


if __name__ == "__main__":
    unittest.main()


class TestSignals(unittest.TestCase):
    """Momentum rows: a planted persistent trend is found by momentum and not by the placebo; a driftless walk gives nothing."""

    @staticmethod
    def trending(seed, n_sym=14, years=3):
        rng = np.random.default_rng(seed)
        n = 24 * 365 * years
        idx = pd.date_range("2023-01-01", periods=n, freq="h")
        out = {}
        for i in range(n_sym):
            # drift that flips sign about every 20 days (AR(1) in the drift), plus noise
            d = np.zeros(n)
            for t in range(1, n):
                d[t] = 0.9995 * d[t - 1] + rng.standard_normal() * 0.00004
            r = d + rng.standard_normal(n) * 0.006
            out[f"S{i}"] = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        return out

    def run_sig(self, closes):
        ev = C.build_events(closes, 24, log=QUIET, signals=("mom168", "ma200d"))
        state, _, _ = C.market_state(closes, ev["t0"])
        p = dict(PARAMS, horizons=[24], signals=["mom168", "ma200d"], only_signals=True)
        res = C.run_study(ev, state, p, log=QUIET)
        return ev, C.summarize(res, {"w": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01"))}), res

    def test_signal_signs_use_only_past_bars(self):
        closes = self.trending(3, n_sym=3, years=1)
        sym = "S0"
        c = closes[sym]
        logc = np.log(c.to_numpy())
        pos = np.array([200, 500, 900])
        s = C.signal_at("mom168", c, logc, pos)
        for i, p_ in enumerate(pos):
            self.assertEqual(int(s[i]), int(np.sign(logc[p_] - logc[p_ - 168])))
        # changing the future must not change the signal
        c2 = c.copy()
        c2.iloc[1000:] *= 3.0
        s2 = C.signal_at("mom168", c2, np.log(c2.to_numpy()), pos)
        self.assertTrue((s == s2).all())
        self.assertEqual(int(C.signal_at("mom168", c, logc, np.array([100]))[0]), 0)    # fewer than L bars -> no signal

    def test_trend_found_by_momentum_not_by_placebo(self):
        ev, summ, res = self.run_sig(self.trending(21))
        mom = summ["rows"]["H24|all|mom168"]
        plc = summ["rows"]["H24|all|placebo"]
        self.assertGreater(mom["mean_gross_bp"], plc["mean_gross_bp"] + 15.0)
        # the placebo is not zero on trending paths: a cut-losers/ride-winners rule earns on the path's trendiness whatever the sign
        self.assertGreater(mom["mean_gross_bp"], 3 * plc["mean_gross_bp"])
        self.assertTrue(set(res["cells"]["direction"]) == {"mom168", "ma200d", "placebo"})

    def test_driftless_walk_gives_momentum_no_edge(self):
        ev, summ, res = self.run_sig(make_closes(9, 0.0))
        for k, r in summ["rows"].items():
            self.assertLess(abs(r["mean_gross_bp"]), 15.0, k)
            self.assertLess(r["mean_net_bp"], 0.0, k)
