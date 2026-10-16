"""Tests for momentum_decomp_core: a planted persistent trend gives a clearly positive winners-minus-losers spread; a driftless walk does not;
a pure long drift raises always-long but leaves the spread near zero; legs, pooled and spread obey their identities."""
import unittest

import numpy as np
import pandas as pd

import book_study_core as C
import momentum_decomp_core as M
from test_book_study_tools import QUIET, make_closes

LB = (168, 480)
HZ = (24,)
WINDOWS = {"w": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01"))}


def trending(seed, n_sym=14, years=3, rho=0.97, vol=0.0008):
    """Hourly returns with a slowly switching per-symbol drift (a persistent trend): momentum should sort future returns."""
    rng = np.random.default_rng(seed)
    n = 24 * 365 * years
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    out = {}
    for i in range(n_sym):
        d = np.zeros(n)
        for t in range(1, n):
            d[t] = rho * d[t - 1] + vol * rng.standard_normal() * (1 - rho) ** 0.5 * 4
        r = d + rng.standard_normal(n) * 0.006
        out[f"S{i}"] = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    return out


def decompose(closes, **kw):
    ev = C.build_events(closes, max(HZ), log=QUIET, signals=tuple(f"mom{L}" for L in LB))
    state, _, _ = C.market_state(closes, ev["t0"])
    return ev, state, M.decompose(ev, state, 0.0024, 300, 1, LB, HZ, WINDOWS)


class TestDecomp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev_n, cls.st_n, (cls.rows_n, cls.yr_n, cls.win_n) = decompose(make_closes(5, 0.0))
        cls.ev_d, cls.st_d, (cls.rows_d, cls.yr_d, cls.win_d) = decompose(make_closes(11, 0.0006))
        cls.ev_t, cls.st_t, (cls.rows_t, cls.yr_t, cls.win_t) = decompose(trending(21))

    @staticmethod
    def get(rows, L, state="all", h=24):
        return [r for r in rows if r["lookback_bars"] == L and r["state"] == state and r["horizon"] == h][0]

    def test_driftless_walk_spread_interval_includes_zero(self):
        for L in LB:
            r = self.get(self.rows_n, L)
            self.assertLess(r["spread_lo"], 0.0)
            self.assertGreater(r["spread_hi"], 0.0)

    def test_pure_drift_raises_always_long_but_not_the_spread(self):
        for L in LB:
            r, rn = self.get(self.rows_d, L), self.get(self.rows_n, L)
            self.assertGreater(r["always_long_bp"], rn["always_long_bp"] + 5.0)      # drift shows up in always-long
            self.assertLess(r["spread_lo"], 0.0)                                       # ... and cannot make the spread significant
            self.assertGreater(r["spread_hi"], 0.0)

    def test_planted_trend_gives_positive_spread(self):
        for L in LB:
            r = self.get(self.rows_t, L)
            self.assertGreater(r["spread_lo"], 0.0, r)
            self.assertGreater(r["pooled_gross_bp"], 0.0)

    def test_identities(self):
        for rows in (self.rows_n, self.rows_d, self.rows_t):
            for r in rows:
                w = r["share_long_signal"]
                self.assertAlmostEqual(r["pooled_gross_bp"], w * r["long_leg_bp"] + (1 - w) * r["short_leg_bp"], places=6)
                self.assertAlmostEqual(r["pooled_net_bp"], r["pooled_gross_bp"] - 24.0, places=9)
                self.assertLessEqual(r["pooled_lo"], r["pooled_gross_bp"])
                self.assertGreaterEqual(r["pooled_hi"], r["pooled_gross_bp"])

    def test_blocks_follow_the_hold_length(self):
        _, bd = M.block_ids(np.array([0, 86400]), 24)
        self.assertEqual(bd, 2)                       # 25 bars -> 2 days
        _, bd = M.block_ids(np.array([0]), 100)
        self.assertEqual(bd, 5)                       # 101 bars -> 5 days

    def test_window_rows_cover_every_lookback_horizon_window(self):
        self.assertEqual(len(self.win_t), len(LB) * len(HZ) * len(WINDOWS))
        for w in self.win_t:
            self.assertGreater(w["events"], 0)

    def test_year_rows_sum_to_row_events(self):
        for r in self.rows_t:
            n = sum(b["events"] for b in self.yr_t if b["lookback_bars"] == r["lookback_bars"] and b["horizon"] == r["horizon"] and b["state"] == r["state"])
            self.assertEqual(n, r["events"])

    def test_cluster_mean_and_spread_on_a_known_case(self):
        day = np.repeat(np.arange(200), 3)
        x = np.tile([1.0, 2.0, 3.0], 200)
        m, lo, hi = M.cluster_mean_ci(day, x, 400, 3)
        self.assertAlmostEqual(m, 2.0)
        self.assertEqual(lo, 2.0)
        self.assertEqual(hi, 2.0)                     # every block has the same mean -> no sampling spread
        up = np.tile([True, False, False], 200)
        sp, lo, hi = M.spread_ci(day, x, up, 400, 3)
        self.assertAlmostEqual(sp, 1.0 - 2.5)


if __name__ == "__main__":
    unittest.main()
