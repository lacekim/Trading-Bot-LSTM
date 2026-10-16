"""Bar-by-bar averaging of overlapping bets (book Ch.10, the idea of Snippet 10.2), written from its logic and used as an EVALUATION of already-made predictions.

Rule used (declared; written from memory of the snippet's logic, not from its text):
  * every event is a bet with signal = side * size (size from the bet-size function of the meta probability; size 0 events are kept in the average as zeros);
  * per instrument and per bar k, the position is the plain mean of the signals of all events of that instrument that are active at k, an event being active on the
    bars k with pos <= k < t1pos (the position is held from the close of bar k to the close of bar k+1); 0 when no event is active;
  * the position is NOT rounded again to the bet-size step (the sizes are already rounded);
  * gross result of bar k = position[k] * (close[k+1] / close[k] - 1), booked on the UTC day of bar k+1;
  * cost = cost_side * |position[k] - position[k-1]| (position[-1] = 0), booked on the UTC day of bar k; so a bet that enters and exits alone pays cost_side * size twice,
    exactly the per-event convention, while overlapping or opposite bets net out and pay less;
  * portfolio result of a UTC day = (sum over instruments of gross - cost on that day) / (number of instruments with at least one candidate event active on that day), in bp.
    A symbol is active on a day when some active bar k has bar k or bar k+1 on that day.  The denominator depends only on the candidate events, so it is the same for
    the primary (all sizes 1), the meta (model sizes) and the placebo (the meta sizes shuffled among the same events) variants.
The numbers are not comparable in level with the per-event numbers of the pipeline (different denominators, netting); compare variants within this convention.
"""
from __future__ import annotations

import numpy as np

DAY = 86400


def average_positions(n_bars: int, pos, t1pos, signal):
    """Mean signal of the events active at each bar and the number of active events.  Events: pos, t1pos (bar indices, pos < t1pos <= n_bars - 1), signal (signed)."""
    pos, t1pos, signal = np.asarray(pos, dtype=np.int64), np.asarray(t1pos, dtype=np.int64), np.asarray(signal, dtype=np.float64)
    if len(pos) and (pos.min() < 0 or t1pos.max() > n_bars - 1 or (t1pos <= pos).any()):
        raise ValueError("event bar indices outside the series or t1pos <= pos")
    s = np.zeros(n_bars + 1)
    c = np.zeros(n_bars + 1, dtype=np.int64)
    np.add.at(s, pos, signal)
    np.add.at(s, t1pos, -signal)
    np.add.at(c, pos, 1)
    np.add.at(c, t1pos, -1)
    S, C = np.cumsum(s)[:n_bars], np.cumsum(c)[:n_bars]
    avg = np.where(C > 0, S / np.maximum(C, 1), 0.0)
    return avg, C


def symbol_daily(close, ts_epoch, pos, t1pos, signal, cost_side: float):
    """One instrument.  close: float array; ts_epoch: int seconds of each bar.  Returns dict with day-indexed arrays:
    days (unique UTC day numbers, sorted), gross (sum of gross results), cost (sum of costs), active (1 if the symbol is active that day), position (the bar positions)."""
    close = np.asarray(close, dtype=np.float64)
    ts = np.asarray(ts_epoch, dtype=np.int64)
    n = len(close)
    if n < 2 or len(ts) != n:
        raise ValueError("need at least two bars and one timestamp per bar")
    avg, C = average_positions(n, pos, t1pos, signal)
    ret = close[1:] / close[:-1] - 1.0                              # ret[k]: bar k -> bar k+1
    gross = avg[:-1] * ret
    day_k, day_k1 = ts // DAY, ts[1:] // DAY
    prev = np.concatenate(([0.0], avg[:-1]))
    cost = cost_side * np.abs(avg - prev)                            # booked on the day of bar k
    act = C[:-1] > 0
    active_days = np.unique(np.concatenate((day_k[:-1][act], day_k1[act])))
    days = np.unique(day_k)                                          # every bar-k day; the days of bars 1.. are among them
    out_g = np.zeros(len(days))
    out_c = np.zeros(len(days))
    np.add.at(out_g, np.searchsorted(days, day_k1), gross)
    np.add.at(out_c, np.searchsorted(days, day_k), cost)
    is_active = np.zeros(len(days), dtype=np.int64)
    is_active[np.searchsorted(days, active_days)] = 1
    return {"days": days, "gross": out_g, "cost": out_c, "active": is_active, "position": avg}


def combine_days(parts: list):
    """Sum symbol_daily outputs over instruments: returns days, net (gross - cost), gross, cost, n_active (instruments active that day)."""
    if not parts:
        raise ValueError("no instruments")
    lo, hi = min(int(p["days"][0]) for p in parts), max(int(p["days"][-1]) for p in parts)
    L = hi - lo + 1
    g, c, a = np.zeros(L), np.zeros(L), np.zeros(L, dtype=np.int64)
    for p in parts:
        ix = p["days"] - lo
        g[ix] += p["gross"]
        c[ix] += p["cost"]
        a[ix] += p["active"]
    return {"days": np.arange(lo, hi + 1), "gross": g, "cost": c, "net": g - c, "n_active": a}


def daily_bp(comb: dict):
    """Days with at least one active instrument: (day numbers, result in bp per active instrument, number of active instruments)."""
    m = comb["n_active"] > 0
    return comb["days"][m], 1e4 * comb["net"][m] / comb["n_active"][m], comb["n_active"][m]


def window_bp(comb: dict, lo_epoch: int, hi_epoch: int):
    """Mean bp per active instrument-day over the UTC days whose start lies in [lo_epoch, hi_epoch), and the number of instrument-days."""
    m = (comb["days"] * DAY >= lo_epoch) & (comb["days"] * DAY < hi_epoch)
    n = int(comb["n_active"][m].sum())
    return (1e4 * float(comb["net"][m].sum()) / n if n else float("nan")), n
