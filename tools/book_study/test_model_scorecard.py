"""Tests for the model-scorecard core on synthetic data (informative fake model vs noise fake model)."""
import unittest

import numpy as np
import pandas as pd

import book_study_core as C
import model_scorecard_core as M
import test_book_study_tools as T

WIN = {"w": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01"))}
RULES = {(d, h): (1.5, 4.0) for d in ("long", "short") for h in (8, 24)}


def make_table(ev, rng, informative):
    n = len(ev["t0"])
    rows = []
    cut = np.quantile(ev["t0"], 0.6)
    fwd8 = ev["q"][:, 8]
    for d, sgn in (("long", 1.0), ("short", -1.0)):
        if informative:
            # probability tied to the sign of the realised 8-bar move in this direction (a model that sees the future: edge by construction)
            score = 1.0 / (1.0 + np.exp(-(sgn * fwd8) * 1.5)) * 0.5 + rng.random(n) * 0.5
        else:
            score = rng.random(n)
        seg = np.where(ev["t0"] < cut, "holdout", "post_holdout")
        keep = ev["t0"] >= np.quantile(ev["t0"], 0.3)
        rows.append(pd.DataFrame({"eidx": np.flatnonzero(keep), "direction": d, "p": score[keep], "thr": 0.7,
                                  "segment": seg[keep], "promoted": False}))
    return pd.concat(rows, ignore_index=True)


class TestScorecard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.closes = T.make_closes(5, 0.0)
        cls.ev = C.build_events(cls.closes, 24, log=lambda *_: None)

    def run_it(self, informative):
        rng = np.random.default_rng(3)
        table = make_table(self.ev, rng, informative)
        return M.score(self.ev, table, RULES, 0.0024, WIN, seed=11, horizons=(8, 24), n_boot=400)

    def test_informative_model_has_lift_noise_model_does_not(self):
        rows, trials, events = self.run_it(True)
        for r in rows:
            if r["horizon"] == 8:
                self.assertGreater(r["lift_hold_lo"], 0.0, (r["direction"], r["segment"]))      # interval excludes zero, positive
        rows, trials, events = self.run_it(False)
        inside = [r["lift_hold_lo"] <= 0.0 <= r["lift_hold_hi"] for r in rows]
        self.assertGreaterEqual(sum(inside), len(inside) - 1)      # at most one of the 8 intervals misses zero by chance

    def test_event_table_is_complete_and_consistent(self):
        rows, trials, events = self.run_it(True)
        self.assertEqual(len(rows), 8)
        sel = events[events.selected]
        self.assertTrue((sel.p >= sel.thr).all())
        self.assertTrue((events[~events.selected].p < events[~events.selected].thr).all())
        self.assertTrue(np.allclose(events.net_bp, events.gross_bp - 24.0))
        # flagged mean in the rows equals the mean of the event table
        r = rows[0]
        m = events[(events.direction == r["direction"]) & (events.horizon == r["horizon"]) & (events.segment == r["segment"]) & events.selected]
        self.assertAlmostEqual(r["selected_gross_bp"], m.gross_bp.mean(), places=6)

    def test_bootstrap_is_deterministic(self):
        a = self.run_it(False)[0]
        b = self.run_it(False)[0]
        self.assertEqual([r["lift_gross_lo"] for r in a], [r["lift_gross_lo"] for r in b])

    def test_day_cluster_lift_hand_case(self):
        day = np.array([1, 1, 2, 2, 3, 3])
        x = np.array([10.0, 0.0, 6.0, 2.0, 2.0, 2.0])
        sel = np.array([True, False, True, False, True, False])
        lift, lo, hi, p = M.day_cluster_lift(day, x, sel, 200, 1)
        self.assertAlmostEqual(lift, (10 + 6 + 2) / 3 - (0 + 2 + 2) / 3)
        self.assertLessEqual(lo, lift)
        self.assertGreaterEqual(hi, lift)


if __name__ == "__main__":
    unittest.main()
