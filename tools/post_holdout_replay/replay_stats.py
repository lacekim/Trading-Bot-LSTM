"""Pure helpers for the post-holdout replay (numpy/pandas only, no project imports)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pooled_stats(trades: pd.DataFrame) -> dict:
    """Pooled trade statistics from a frame with a net_pnl column."""
    n = int(len(trades))
    if n == 0:
        return {"trades": 0, "wins": 0, "win_rate_pct": float("nan"),
                "profit_factor": float("nan"), "net_pnl": 0.0, "mean_return_pct": float("nan")}
    pnl = trades["net_pnl"].astype(float)
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = float("inf")
    else:
        pf = float("nan")
    wins = int((pnl > 0).sum())
    return {"trades": n, "wins": wins, "win_rate_pct": wins / n * 100.0,
            "profit_factor": pf, "net_pnl": float(pnl.sum()),
            "mean_return_pct": float(trades["return_pct"].astype(float).mean())}


def parse_holdout_bounds(entry: dict):
    """(start, end) of the stored promotion holdout, from the walk-forward windows.

    Window 1 start is the first holdout endpoint and window 3 end is the last,
    because the trainer splits the holdout endpoints into 3 consecutive windows.
    Returns None if the entry has no usable windows."""
    windows = entry.get("windows") or []
    if not windows:
        return None
    starts = [pd.Timestamp(w["start"]) for w in windows if w.get("start")]
    ends = [pd.Timestamp(w["end"]) for w in windows if w.get("end")]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def equal_weight_index(closes: dict, start_ts, min_symbols: int = 10) -> pd.Series:
    """Equal-weight price index (starts at 1.0) from hourly closes, strictly after start_ts.

    Per timestamp the index moves by the mean log return of the symbols that have a
    return at that timestamp; timestamps with fewer than min_symbols are dropped."""
    rets = {}
    for sym, close in closes.items():
        c = close[close.index > start_ts - pd.Timedelta(days=1)].astype(float)
        r = np.log(c).diff()
        rets[sym] = r[r.index > start_ts]
    frame = pd.DataFrame(rets)
    counts = frame.notna().sum(axis=1)
    mean_ret = frame.mean(axis=1)[counts >= min_symbols]
    return np.exp(mean_ret.cumsum())


def max_drawdown_window(index: pd.Series):
    """(peak_ts, trough_ts, drawdown_pct) of the deepest peak-to-trough fall."""
    if index.empty:
        return None
    run_max = index.cummax()
    dd = index / run_max - 1.0
    trough = dd.idxmin()
    peak = index.loc[:trough].idxmax()
    return peak, trough, float(dd.loc[trough] * 100.0)


def compare_selftest(stored: dict, got: dict, pf_tol: float = 0.01, ret_tol: float = 0.05) -> dict:
    """Compare a re-scored holdout with the numbers stored by the trainer."""
    trades_ok = int(stored["holdout_trades"]) == int(got["trades"])
    s_pf, g_pf = float(stored["holdout_profit_factor"]), float(got["profit_factor"])
    if np.isinf(s_pf) or np.isinf(g_pf):
        pf_ok = bool(np.isinf(s_pf) and np.isinf(g_pf))
    else:
        pf_ok = abs(s_pf - g_pf) <= pf_tol
    ret_ok = abs(float(stored["holdout_return_pct"]) - float(got["return_pct"])) <= ret_tol
    return {"trades_match": trades_ok, "pf_match": pf_ok, "return_match": ret_ok,
            "passed": bool(trades_ok and pf_ok and ret_ok)}


def spearman(x, y) -> float:
    """Spearman rank correlation (Pearson on ranks); nan if fewer than 3 points or no variation."""
    a, b = pd.Series(np.asarray(x, dtype=float)), pd.Series(np.asarray(y, dtype=float))
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def top_decile_edge(prob, signed_ret) -> float:
    """Mean signed return of the highest-probability 10% of candles minus the mean of all candles.

    signed_ret is the next-candle return in the trade's direction (return for long, minus
    return for short), as a fraction. Before costs. nan if there is no data."""
    p = np.asarray(prob, dtype=float)
    r = np.asarray(signed_ret, dtype=float)
    if len(p) == 0:
        return float("nan")
    k = max(1, len(p) // 10)
    top = np.argsort(-p, kind="stable")[:k]
    return float(r[top].mean() - r.mean())
