"""Tests for the exit-grid statistics helpers. Run: python -m unittest tools/post_holdout_replay/test_exit_grid_tools.py -v"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exit_grid_stats as G

SLIP = 7.0   # bp per side
FEE = 5.0


def make_trade(direction, ref_entry, ref_exit, notional=10_000.0, reason="take_profit"):
    """A trade row built the way the simulator books it: fills carry slippage, fees are charged on notional."""
    s = SLIP / 1e4
    long_ = direction == "LONG"
    entry = ref_entry * (1 + s) if long_ else ref_entry * (1 - s)
    exit_ = ref_exit * (1 - s) if long_ else ref_exit * (1 + s)
    qty = notional / entry
    gross = (exit_ - entry) * qty * (1 if long_ else -1)
    fees = notional * FEE / 1e4 * 2
    return {"direction": direction, "entry_price": entry, "exit_price": exit_, "notional": notional,
            "fees": fees, "net_pnl": gross - fees, "exit_reason": reason}


class TradeBp(unittest.TestCase):
    def test_flat_trade_costs_24bp_either_side(self):
        for d in ("LONG", "SHORT"):
            bp = G.trade_bp(pd.DataFrame([make_trade(d, 100.0, 100.0)]), SLIP)
            self.assertAlmostEqual(bp["pre_cost_bp"].iloc[0], 0.0, places=6)
            self.assertAlmostEqual(bp["pre_cost_bp"].iloc[0] - bp["net_bp"].iloc[0], 24.0, delta=0.05)

    def test_direction_signs(self):
        up = pd.DataFrame([make_trade("LONG", 100, 101), make_trade("SHORT", 100, 101)])
        bp = G.trade_bp(up, SLIP)
        self.assertAlmostEqual(bp["pre_cost_bp"].iloc[0], 100.0, delta=0.01)    # long gains from a rise
        self.assertAlmostEqual(bp["pre_cost_bp"].iloc[1], -100.0, delta=0.01)   # short loses from a rise

    def test_net_bp_matches_booked_pnl(self):
        t = pd.DataFrame([make_trade("LONG", 100, 103, 5_000.0), make_trade("SHORT", 50, 49, 20_000.0)])
        bp = G.trade_bp(t, SLIP)
        np.testing.assert_allclose(bp["net_bp"], t["net_pnl"] / t["notional"] * 1e4)

    def test_empty(self):
        self.assertEqual(len(G.trade_bp(pd.DataFrame(), SLIP)), 0)
        self.assertEqual(G.aggregate_trades(pd.DataFrame(), SLIP)["trades"], 0)


class Aggregation(unittest.TestCase):
    def setUp(self):
        self.t1 = pd.DataFrame([make_trade("LONG", 100, 102, reason="take_profit"), make_trade("LONG", 100, 99, reason="stop_loss")])
        self.t2 = pd.DataFrame([make_trade("LONG", 10, 10.1, reason="window_exit")])

    def test_additive(self):
        a, b = G.aggregate_trades(self.t1, SLIP), G.aggregate_trades(self.t2, SLIP)
        both = G.aggregate_trades(pd.concat([self.t1, self.t2], ignore_index=True), SLIP)
        for k in ("trades", "sum_net_bp", "sum_pre_cost_bp", "wins", "gross_win", "gross_loss", "net_pnl", "stop_exits", "target_exits", "time_exits"):
            self.assertAlmostEqual(a[k] + b[k], both[k], places=6, msg=k)

    def test_exit_reasons(self):
        a = G.aggregate_trades(self.t1, SLIP)
        self.assertEqual((a["stop_exits"], a["target_exits"], a["time_exits"]), (1, 1, 0))


def fake_per_symbol(scores: dict[str, dict[tuple, float]]):
    """scores[period][cell] = mean net bp; 2 symbols x 1000 trades each so trade minimums are met."""
    rows = []
    for period, cells in scores.items():
        for cell, val in cells.items():
            for sym in ("AAA", "BBB"):
                rows.append({"direction": "long", "period": period, "symbol": sym, "stop_atr": cell[0], "target_atr": cell[1], "max_hold": cell[2],
                             "trades": 1000, "sum_net_bp": val * 1000, "sum_pre_cost_bp": (val + 24) * 1000, "sum_fee_bp": 10_000.0,
                             "wins": 400, "gross_win": 1000.0 if val > 0 else 500.0, "gross_loss": 500.0 if val > 0 else 1000.0, "net_pnl": 1.0,
                             "stop_exits": 600, "target_exits": 300, "time_exits": 100})
    return pd.DataFrame(rows)


class Summary(unittest.TestCase):
    cells = [(s, t, h) for s in (0.5, 1.0, 2.0) for t in (1.0, 2.0) for h in (8, 24)]

    def test_pooled_means_and_pf(self):
        df = fake_per_symbol({"select": {c: -10.0 for c in self.cells}})
        s = G.summarize(df)
        self.assertEqual(len(s), len(self.cells))
        self.assertTrue(np.allclose(s["mean_net_bp"], -10.0))
        self.assertTrue(np.allclose(s["mean_cost_bp"], 24.0))
        self.assertTrue(np.allclose(s["profit_factor"], 0.5))
        self.assertTrue((s["symbols_with_trades"] == 2).all())

    def test_selection_and_rank_agreement(self):
        base = {c: float(i) for i, c in enumerate(self.cells)}
        same = {c: v * 2 + 1 for c, v in base.items()}                    # same ordering
        rev = {c: -v for c, v in base.items()}                            # reverse ordering
        s = G.summarize(fake_per_symbol({"select": base, "june_2026": same, "post_holdout": rev}))
        self.assertEqual(G.select_cell(s, "long", "select"), self.cells[-1])
        self.assertAlmostEqual(G.rank_agreement(s, "long", "select", "june_2026"), 1.0)
        self.assertAlmostEqual(G.rank_agreement(s, "long", "select", "post_holdout"), -1.0)
        self.assertEqual(G.percentile_rank(s, "long", "select", self.cells[-1]), (len(self.cells) - 1) / len(self.cells) * 100)

    def test_select_needs_enough_trades(self):
        s = G.summarize(fake_per_symbol({"select": {c: 5.0 for c in self.cells}}))
        self.assertIsNone(G.select_cell(s, "long", "select", min_trades=10_000))

    def test_grid_contains_default_and_size(self):
        self.assertIn(G.DEFAULT_CELL, G.make_cells())
        self.assertEqual(len(G.make_cells()), 75)

    def test_periods_do_not_touch(self):
        self.assertLessEqual(G.SELECT_WINDOW[1], G.JUNE_WINDOW[0])


if __name__ == "__main__":
    unittest.main()
