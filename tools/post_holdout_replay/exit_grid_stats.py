"""Pure helpers for the exit-rule grid test (numpy/pandas only, no project imports).

Metrics are per trade, in basis points of the trade's notional, so they do not depend on position size
(position size changes with the stop distance because the simulator sizes risk off the stop).
  pre_cost_bp  price move in the trade's direction with fees AND slippage/impact removed
  net_bp       what the simulator booked: net_pnl / notional
  cost_bp      pre_cost_bp - net_bp (fees + slippage + impact; about 24 bp per round trip with default costs)
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

import replay_stats as S

STOPS = (0.5, 0.75, 1.5, 3.0, 6.0)        # ATR multiples
TARGETS = (0.5, 0.75, 1.5, 3.0, 6.0)      # ATR multiples
HOLDS = (8, 24, 96)                       # candles (1h)
DEFAULT_CELL = (0.75, 1.5, 8)             # the simulator's current default (stop, target, max hold)

PERIODS = ("select", "june_2026", "post_holdout")
SELECT_WINDOW = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-01"))   # [start, end)
JUNE_WINDOW = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01"))     # [start, end)


def make_cells(stops=STOPS, targets=TARGETS, holds=HOLDS) -> list:
    return [(float(s), float(t), int(h)) for s, t, h in product(stops, targets, holds)]


def trade_bp(trades: pd.DataFrame, slip_bps_per_side: float) -> pd.DataFrame:
    """Per-trade pre-cost / fee / net basis points from the simulator's trade rows."""
    if len(trades) == 0:
        return pd.DataFrame(columns=["pre_cost_bp", "fee_bp", "net_bp"])
    s = slip_bps_per_side / 1e4
    entry = trades["entry_price"].astype(float).to_numpy()
    exit_ = trades["exit_price"].astype(float).to_numpy()
    is_long = (trades["direction"].astype(str).str.upper() == "LONG").to_numpy()
    # undo the slippage the simulator applied to each fill to get the reference-price move
    ref_entry = np.where(is_long, entry / (1 + s), entry / (1 - s))
    ref_exit = np.where(is_long, exit_ / (1 - s), exit_ / (1 + s))
    move = ref_exit / ref_entry - 1.0
    pre = np.where(is_long, move, -move) * 1e4
    notional = trades["notional"].astype(float).to_numpy()
    return pd.DataFrame({
        "pre_cost_bp": pre,
        "fee_bp": trades["fees"].astype(float).to_numpy() / notional * 1e4,
        "net_bp": trades["net_pnl"].astype(float).to_numpy() / notional * 1e4,
    })


def aggregate_trades(trades: pd.DataFrame, slip_bps_per_side: float) -> dict:
    """Additive statistics for one symbol / cell / period (summed later across symbols)."""
    n = int(len(trades))
    if n == 0:
        return {"trades": 0, "sum_net_bp": 0.0, "sum_pre_cost_bp": 0.0, "sum_fee_bp": 0.0, "wins": 0,
                "gross_win": 0.0, "gross_loss": 0.0, "net_pnl": 0.0, "stop_exits": 0, "target_exits": 0, "time_exits": 0}
    bp = trade_bp(trades, slip_bps_per_side)
    pnl = trades["net_pnl"].astype(float)
    reason = trades["exit_reason"].astype(str)
    return {
        "trades": n, "sum_net_bp": float(bp["net_bp"].sum()), "sum_pre_cost_bp": float(bp["pre_cost_bp"].sum()),
        "sum_fee_bp": float(bp["fee_bp"].sum()), "wins": int((pnl > 0).sum()),
        "gross_win": float(pnl[pnl > 0].sum()), "gross_loss": float(-pnl[pnl < 0].sum()), "net_pnl": float(pnl.sum()),
        "stop_exits": int((reason == "stop_loss").sum()), "target_exits": int((reason == "take_profit").sum()),
        "time_exits": int(reason.isin(["window_exit", "end_of_data", "signal_reversal"]).sum()),
    }


CELL_COLS = ["direction", "period", "stop_atr", "target_atr", "max_hold"]


def summarize(per_symbol: pd.DataFrame) -> pd.DataFrame:
    """One row per direction / period / cell, pooled over symbols."""
    if per_symbol.empty:
        return pd.DataFrame()
    g = per_symbol.groupby(CELL_COLS, sort=True)
    out = g[["trades", "sum_net_bp", "sum_pre_cost_bp", "sum_fee_bp", "wins", "gross_win", "gross_loss", "net_pnl",
             "stop_exits", "target_exits", "time_exits"]].sum()
    n = out["trades"].replace(0, np.nan)
    out["mean_net_bp"] = out["sum_net_bp"] / n
    out["mean_pre_cost_bp"] = out["sum_pre_cost_bp"] / n
    out["mean_cost_bp"] = out["mean_pre_cost_bp"] - out["mean_net_bp"]
    out["win_rate_pct"] = out["wins"] / n * 100.0
    out["profit_factor"] = np.where(out["gross_loss"] > 0, out["gross_win"] / out["gross_loss"].replace(0, np.nan),
                                    np.where(out["gross_win"] > 0, np.inf, np.nan))
    out["target_share_of_barrier_exits"] = out["target_exits"] / (out["target_exits"] + out["stop_exits"]).replace(0, np.nan)
    sym = per_symbol[per_symbol["trades"] > 0].copy()
    sym["sym_mean_net_bp"] = sym["sum_net_bp"] / sym["trades"]
    out["symbols_with_trades"] = sym.groupby(CELL_COLS).size()
    out["symbols_net_positive"] = sym[sym["sym_mean_net_bp"] > 0].groupby(CELL_COLS).size()
    out[["symbols_with_trades", "symbols_net_positive"]] = out[["symbols_with_trades", "symbols_net_positive"]].fillna(0).astype(int)
    return out.reset_index()


def cell_table(summary: pd.DataFrame, direction: str, period: str) -> pd.DataFrame:
    t = summary[(summary["direction"] == direction) & (summary["period"] == period)]
    return t.set_index(["stop_atr", "target_atr", "max_hold"])


def select_cell(summary: pd.DataFrame, direction: str, period: str = "select", min_trades: int = 1000):
    """Cell with the highest mean net bp per trade in `period` among cells with at least min_trades; None if none."""
    t = cell_table(summary, direction, period)
    t = t[t["trades"] >= min_trades]
    if t.empty:
        return None
    return tuple(t["mean_net_bp"].idxmax())


def rank_agreement(summary: pd.DataFrame, direction: str, period_a: str, period_b: str,
                   metric: str = "mean_net_bp", min_trades: int = 200) -> float:
    """Spearman correlation across grid cells of `metric` between two periods (does the grid carry information?)."""
    a, b = cell_table(summary, direction, period_a), cell_table(summary, direction, period_b)
    idx = [i for i in a.index if i in b.index and a.loc[i, "trades"] >= min_trades and b.loc[i, "trades"] >= min_trades]
    if len(idx) < 5:
        return float("nan")
    return S.spearman([a.loc[i, metric] for i in idx], [b.loc[i, metric] for i in idx])


def percentile_rank(summary: pd.DataFrame, direction: str, period: str, cell, metric: str = "mean_net_bp") -> float:
    """Share of cells (with trades) in `period` whose metric is strictly below this cell's (0..100)."""
    t = cell_table(summary, direction, period)
    t = t[t["trades"] > 0]
    if cell not in t.index or len(t) < 2:
        return float("nan")
    return float((t[metric] < t.loc[cell, metric]).mean() * 100.0)
