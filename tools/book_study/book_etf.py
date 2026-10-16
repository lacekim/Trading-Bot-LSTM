"""Section 2.4.1 of the book (the ETF trick), items 1-3, applied to the account of book_bot.py.  New file.

The book's basket accounting (Expression 1, Equations 16-18) is for a virtual ETF built from a roll-adjusted futures basket; we do not trade a basket as one instrument, so Expression 1 itself is not used.  The three rules that come with it are:
  1 Rebalance costs   c_t = sum_i (|h_{i,t-1}| p_{i,t} + |h_{i,t}| o_{i,t+1}) phi_i tau_i on every rebalance date t in B ("treat {c_t} as a (negative) dividend").  Here B is the quarterly HRP rebalance (book_final.weight_schedule); the holdings
                      are the instruments' dollar exposures per unit of capital, w_i x (averaged bet signal at the end of the previous day); phi = 1 (spot, USD); tau = the venue's cost per side.  Taken literally, the whole old allocation
                      is sold and the whole new one bought, so this is the book's charge, not just the turnover.
  2 Bid-ask spread    c~_t = sum_i |h_{i,t-1}| p_{i,t} phi tau: the cost of crossing the spread per unit traded.  The account already charges tau per side on every change of every instrument's averaged position; nothing to add.
  3 Volume            v_t = min_i v_{i,t} / |h_{i,t-1}| tradeable units.  For per-instrument trades: a trade of capital K changes instrument i's exposure by K |de_i| (de_i per unit of capital), and the bar can absorb it when K |de_i| <= p_i,t v_i,t.
                      capacity(trade) = p v / |de|, the largest account K for which that trade fits inside the volume of its own bar.
Everything here is computed from saved quantities of the run (events, predictions, weights); no parameter is added.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bet_averaging as BA

DAY = 86400
K_GRID = (1e3, 1e4, 1e5, 1e6, 1e7)


def bar_volume(bars_dir, sym: str, n_expected: int, t_expected: np.ndarray) -> np.ndarray:
    """Volume column of bars/<sym>.csv in the order bar_loader gives the bars (same filters), asserted against the loaded timestamps."""
    import bar_loader as BL
    b = pd.read_csv(bars_dir / "bars" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "close", "volume"])
    b = b.dropna(subset=["close"])
    b = b[b["close"] > 0]
    t = BL._epoch(b["date"])
    o = np.argsort(t, kind="stable")
    t = t[o]
    keep = np.concatenate([[True], np.diff(t) > 0]) if len(t) else np.zeros(0, bool)
    assert keep.sum() == n_expected and (t[keep] == t_expected).all(), f"{sym}: bar volume does not line up"
    return b["volume"].to_numpy(float)[o][keep]


def exposures(n_bars: int, pos, t1pos, signal) -> np.ndarray:
    """Averaged signed position held from each bar (Snippet 10.2), per unit of the instrument's slice."""
    avg, _ = BA.average_positions(n_bars, np.asarray(pos, np.int64), np.asarray(t1pos, np.int64), np.asarray(signal, float))
    return avg


def day_end_position(avg: np.ndarray, ts: np.ndarray, d0: int, d1: int) -> np.ndarray:
    """Position at the last bar of each UTC day d0..d1, carried forward over days without a bar; 0 before the first bar."""
    day = ts // DAY
    last = pd.Series(avg).groupby(day).last()
    return last.reindex(np.arange(min(d0, int(day.min())), d1 + 1)).ffill().fillna(0.0).loc[d0:d1].to_numpy()


def rebalance_costs(pos_day: dict, sched: dict, cost_side: float, d0: int, d1: int) -> np.ndarray:
    """Item 1: array over days d0..d1; on each rebalance day st the charge cost_side * sum_i (|w_old,i| + |w_new,i|) * |position_i at the end of day st-1|, zero elsewhere.
    pos_day[s] is the day-end position array for days d0..d1 (position on day d is the one carried into day d+1)."""
    out = np.zeros(d1 - d0 + 1)
    keys = sorted(sched)
    old: dict = {}
    for st in keys:
        new = sched[st] if len(sched[st]) else {}
        if d0 + 1 <= st <= d1:
            c = 0.0
            for s, p in pos_day.items():
                pv = abs(float(p[st - 1 - d0]))
                if pv:
                    c += (abs(float(old.get(s, 0.0))) + abs(float(new.get(s, 0.0)))) * pv
            out[st - d0] = cost_side * c
        old = new
    return out


def capacity(bars: dict, vols: dict, closes_ts: dict, ev, idx, side, size, t1pos, sched, d_lo: int, d_hi: int) -> dict:
    """Item 3 for one path (the walk-forward bets): every change of an instrument's exposure w(day) x averaged signal, at the bar where it happens, with the bar's own traded value.
    Returns the per-trade capacities (largest account K for which the trade fits in the bar's volume) summarised."""
    keys = sorted(sched)
    sym_ev, pos_ev = ev["symbol"].to_numpy(), ev["pos"].to_numpy(np.int64)
    caps = []
    for s, b in bars.items():
        sel = np.flatnonzero(sym_ev[idx] == s)
        if not len(sel):
            continue
        avg = exposures(b.n, pos_ev[idx][sel], t1pos[sel], side[sel] * size[sel])
        ts = b.t
        day = ts // DAY
        wv = np.zeros(b.n)
        for k, st in enumerate(keys):
            en = keys[k + 1] if k + 1 < len(keys) else d_hi + 1
            w = float(sched[st].get(s, 0.0)) if len(sched[st]) else 0.0
            wv[(day >= st) & (day < en)] = w
        e = wv * avg
        de = np.abs(np.diff(e, prepend=0.0))
        m = (de > 1e-12) & (day >= d_lo) & (day <= d_hi)
        if m.any():
            caps.append(b.close[m] * vols[s][m] / de[m])
    cap = np.concatenate(caps) if caps else np.zeros(0)
    out = {"n_trades": int(len(cap))}
    if len(cap):
        out.update({"capacity_min": float(cap.min()), "capacity_p01": float(np.quantile(cap, .01)), "capacity_p05": float(np.quantile(cap, .05)), "capacity_median": float(np.median(cap)),
                    "share_not_fitting": {str(int(k)): float((cap < k).mean()) for k in K_GRID}})
    return out
