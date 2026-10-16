"""A fast-drop condition on the equal-weight index, as a filter for the long rule and a trigger for the short rule: fixed rules, nothing fitted.

DESIGN (fixed before any result of this module was seen; the same text is printed in SUMMARY.md):

  Why           the previous study showed that a slow market-down state (index down over each of the last 20, 30 and 40 days) never fires in the sell-off, so the
                MOM-long rule still lost about 100 bp per candidate there (every sell-off event was in the 'above the 200-day mean' state).  The sell-off was a
                fast reversal from above that mean.  This study asks whether a FAST drop of the index, defined once and measured on every event of the whole
                history, separates such events, and it reports every historical occurrence of the condition, not only the sell-off.
  Fast drop     z_L = (log idx now - log idx L hours earlier) / (sigma * sqrt(L)), where idx is the equal-weight index of the 200-day state (mean hourly log
                return of the symbols that have a return, hours with at least 10 symbols), 'now' is the last index hour at or before the event bar, 'L hours
                earlier' is the last index hour at or before the event bar minus L hours, and sigma is the sample standard deviation (ddof 1) of the 720 most
                recent hourly log returns of the index ending at 'now'.  FAST_L is true when z_L < -2.  The 2 is the width of the triple-barrier exits (2 daily
                standard deviations); sqrt(L) scales a one-hour sigma to L hours (random-walk scaling).  If fewer than 720 hourly index returns exist, or the
                lookback precedes the index, FAST_L is false.  Base L = 72 hours (3 days); neighbours L = 24 (1 day) and L = 168 (7 days) are declared and
                reported, never selected.  The horizons 24 / 72 / 168 were fixed before the index was looked at on any of them.
  Stock signal  MOM side as in the previous studies (20-, 30-, 40-day returns of the symbol all > 0: +1, all < 0: -1, else 0).
  Rules         every rule trades with size 1, 24 bp round trip on taken bets, at the CUSUM events of the source run, same exits (symmetric barriers):
                long_up_nofast    side +1 at events with MOM side +1 while the index is above its 200-day mean AND NOT FAST_72   (mom_long_up with the filter)
                short_fast        side -1 at every event while FAST_72
                short_fast_mom    side -1 at events with MOM side -1 while FAST_72
                fast_rule         long_up_nofast plus short_fast_mom (flat otherwise)
                fast_rule_24 / fast_rule_168   fast_rule with FAST_24 / FAST_168
                Baselines from the previous studies, recomputed here: long_all, short_all, mom_all, mom_long_up.
  Evaluation    one unit of capital per candidate event; a day's result = sum of size * (side * ret_long - cost * [size > 0]) over its events divided by ALL
                candidate events entered that day, in bp.  There is nothing to fit, so no cross-validation: every event is scored once.  Reported: the whole
                history, before / after 2024-01-01, each calendar year, June 2026, the sell-off and the deepest-fall windows, each cell of (200-day state x
                FAST_72), the share of events with FAST_L true in each window, and a table of what the long filter removes: the mom_long_up trades that
                FAST_72 blocks versus those it keeps, by calendar year, over the whole history and inside each window, with the number of distinct days
                the removed trades fall on (they are clustered).  For the sell-off window, a day-by-day table of the detector (events, FAST_72 share,
                lowest z_72, results of mom_long_up and long_up_nofast).  PSR on the daily series (effective sample = days / max(1, 101/24)); deflated Sharpe
                for the six new variants with N = 1, 10, 94 (the earlier looks) and 100 (those plus the six rules here).
  Not claimed   the horizons are 'fast' relative to the 20-40-day signal, but the author of this design knew that the sell-off window existed and that the
                earlier slow state missed it; the rules are therefore in-sample by construction, and the neighbours and the year / half splits show stability,
                not out-of-sample evidence.  The detector is not from the book (see the deviations).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import book_stats as B
import regime_algo as R

HORIZONS = {"base": 72, "short": 24, "long": 168}
ALL_HORIZONS = tuple(sorted(HORIZONS.values()))
SIGMA_WINDOW = 720
K_SIGMA = 2.0
BASELINES = ("long_all", "short_all", "mom_all", "mom_long_up")
NEW_VARIANTS = ("long_up_nofast", "short_fast", "short_fast_mom", "fast_rule", "fast_rule_24", "fast_rule_168")
VARIANTS = BASELINES + NEW_VARIANTS
PRIOR_TRIALS_BEFORE = 94
TRIALS_AFTER = PRIOR_TRIALS_BEFORE + len(NEW_VARIANTS)

DEVIATIONS = [
    "The condition is not from the book: the condensed edition has no market crash detector.  The nearest tool in the book is the SADF test of Chapter 17 (explosive behaviour), which is already used per symbol as a feature; it was not applied to the index here.  The volatility-scaled drop is this study's own, with no free constant beyond the horizons 24 / 72 / 168, the 720-hour volatility window and the multiple 2 (taken from the barrier width).",
    "The rules are motivated by the sell-off window and the earlier slow-state failure, so they are in-sample by construction; the neighbours (24 and 168 hours), the year / half splits and the removed-versus-kept table show stability across the history, not out-of-sample performance.",
    "Fixed rules have no fitted part, so there is no cross-validation, purge or embargo; overlapping holds of neighbouring events are not removed, so the daily-series Sharpe treats days as independent although the holds are 101 bars.",
    "Size is 1 on every rule (no bet sizing); no funding or borrow cost for shorts; the short side is assumed to be executable at the same 24 bp round trip as the long side; no capacity or liquidity limits.",
    "The equal-weight index is the one used for the 200-day state (hours with at least 10 symbols; symbols that exist today, so survivorship applies).  Its returns use the latest index hour at or before the stated time; sigma uses the 720 consecutive index observations ending there.",
    "A window is a few correlated days: its lines carry no intervals and cannot establish anything on their own.  The sell-off window is the one that motivated the rules.",
    "All source-run deviations apply (time bars not volume bars, hourly candles, survivorship in the symbol list).",
]


def index_sigma_at(idx: pd.Series, t0_sec, window: int = SIGMA_WINDOW) -> np.ndarray:
    """Sample std (ddof 1) of the `window` consecutive index log returns ending at the last index hour <= t0; NaN if fewer than `window` returns exist."""
    lv = np.log(idx.to_numpy(dtype=np.float64))
    r = np.full(len(lv), np.nan)
    r[1:] = np.diff(lv)
    sig = pd.Series(r).rolling(window, min_periods=window).std(ddof=1).to_numpy()
    keys = idx.index.to_numpy().astype("datetime64[s]").astype(np.int64)
    t_sec = np.asarray(t0_sec, dtype=np.int64)
    now = np.searchsorted(keys, t_sec, side="right") - 1
    out = np.full(len(t_sec), np.nan)
    ok = now >= 0
    out[ok] = sig[now[ok]]
    return out


def fast_z(ret_L: np.ndarray, sigma: np.ndarray, L: int) -> np.ndarray:
    """z_L = index return over L hours / (sigma * sqrt(L)); NaN where either input is NaN or sigma <= 0."""
    ret_L, sigma = np.asarray(ret_L, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    z = np.full(len(ret_L), np.nan)
    ok = np.isfinite(ret_L) & np.isfinite(sigma) & (sigma > 0)
    z[ok] = ret_L[ok] / (sigma[ok] * math.sqrt(L))
    return z


def fast_flag(z: np.ndarray) -> np.ndarray:
    return np.asarray(z) < -K_SIGMA            # NaN compares False


def build_rules(mom_side, state, fast: dict):
    """{variant: side array (+1 / -1 / 0)}; fast = {'base','short','long'} boolean arrays (FAST_72 / FAST_24 / FAST_168)."""
    m = np.asarray(mom_side)
    s = np.asarray(state) == 1.0
    n = len(m)
    fB, fS, fL = fast["base"], fast["short"], fast["long"]

    def rule(f):
        side = np.zeros(n, dtype=np.int64)
        side[(m == 1) & s & ~f] = 1
        side[(m == -1) & f] = -1
        return side

    return {
        "long_all": np.ones(n, dtype=np.int64), "short_all": -np.ones(n, dtype=np.int64), "mom_all": m.astype(np.int64),
        "mom_long_up": np.where((m == 1) & s, 1, 0).astype(np.int64),
        "long_up_nofast": np.where((m == 1) & s & ~fB, 1, 0).astype(np.int64),
        "short_fast": np.where(fB, -1, 0).astype(np.int64),
        "short_fast_mom": np.where(fB & (m == -1), -1, 0).astype(np.int64),
        "fast_rule": rule(fB), "fast_rule_24": rule(fS), "fast_rule_168": rule(fL),
    }


def evaluate_fast(df, z, fast, cost_frac, h, windows, cutoff_sec, prior_before=PRIOR_TRIALS_BEFORE):
    """df needs t0, ret_long, mkt_state, mom_side.  z: {'base','short','long'} arrays of z_L; fast: boolean arrays.  Returns the nested result dict and the daily frame."""
    ret = df["ret_long"].to_numpy(np.float64)
    t0 = df["t0"].to_numpy(np.int64)
    day = t0 // 86400
    year = pd.DatetimeIndex(pd.to_datetime(t0, unit="s")).year.to_numpy()
    state = df["mkt_state"].to_numpy(np.float64)
    mom = df["mom_side"].to_numpy()
    sides = build_rules(mom, state, fast)
    res = {"stats": {}, "windows": {}, "halves": {}, "years": {}, "cells": {}, "side_split": {}}
    daily_parts, pnl_all = [], {}
    for v in VARIANTS:
        side = sides[v]
        size = (side != 0).astype(np.float64)
        pnl = size * (side * ret - cost_frac * size) * 1e4
        pnl_all[v] = pnl
        st, d = R.series_stats(day, pnl, h)
        st["taken_share"] = float(size.mean())
        res["stats"][v] = st
        d.insert(0, "variant", v)
        daily_parts.append(d)
        for wn, (lo, hi) in windows.items():
            m = (t0 >= lo) & (t0 < hi)
            res["windows"][f"{v}|{wn}"] = {"events": int(m.sum()), "trades": int(size[m].sum()), "mean_bp_per_candidate": float(pnl[m].mean()) if m.any() else float("nan")}
        for hn, m in (("before", t0 < cutoff_sec), ("after", t0 >= cutoff_sec)):
            sth, _ = R.series_stats(day[m], pnl[m], h)
            sth["taken_share"] = float(size[m].mean()) if m.any() else float("nan")
            res["halves"][f"{v}|{hn}"] = sth
        for y in sorted(set(year.tolist())):
            m = year == y
            res["years"][f"{v}|{y}"] = {"events": int(m.sum()), "trades": int(size[m].sum()), "mean_bp_per_candidate": float(pnl[m].mean())}
        for s_ in (0, 1):
            for fn in (0, 1):
                m = (state == s_) & (fast["base"] == bool(fn))
                res["cells"][f"{v}|state{s_}|fast{fn}"] = {"events": int(m.sum()), "trades": int(size[m].sum()), "mean_bp_per_candidate": float(pnl[m].mean()) if m.any() else float("nan")}
        for sd, nm in ((1, "long"), (-1, "short")):
            m = side == sd
            res["side_split"][f"{v}|{nm}"] = {"trades": int(m.sum()), "bp_per_trade": float(pnl[m].mean()) if m.any() else float("nan")}
    res["window_fast_share"] = {}
    for wn, (lo, hi) in windows.items():
        m = (t0 >= lo) & (t0 < hi)
        res["window_fast_share"][wn] = {"events": int(m.sum()), **{k: (float(fast[k][m].mean()) if m.any() else float("nan")) for k in ("base", "short", "long")}}
    res["cell_events"] = {f"state{s_}|fast{fn}": int(((state == s_) & (fast["base"] == bool(fn))).sum()) for s_ in (0, 1) for fn in (0, 1)}
    res["fast_share"] = {k: float(fast[k].mean()) for k in ("base", "short", "long")}
    res["fast_days"] = {k: int(len(set(day[fast[k]].tolist()))) for k in ("base", "short", "long")}
    # what the long filter removes: mom_long_up trades blocked by FAST_72 versus kept
    base_long = (mom == 1) & (state == 1.0)
    pnl_l = pnl_all["mom_long_up"]
    filt = {}

    def row(m):
        rem, kept = m & base_long & fast["base"], m & base_long & ~fast["base"]
        return {"removed": int(rem.sum()), "removed_days": int(len(set(day[rem].tolist()))), "removed_bp": float(pnl_l[rem].mean()) if rem.any() else float("nan"),
                "kept": int(kept.sum()), "kept_bp": float(pnl_l[kept].mean()) if kept.any() else float("nan")}

    filt["all"] = row(np.ones(len(df), dtype=bool))
    for y in sorted(set(year.tolist())):
        filt[f"y{y}"] = row(year == y)
    for wn, (lo, hi) in windows.items():
        filt[f"w|{wn}"] = row((t0 >= lo) & (t0 < hi))
    res["filter"] = filt
    # sell-off day table
    sell = [k for k in windows if k.startswith("sell_off")]
    tab = {}
    if sell:
        lo, hi = windows[sell[0]]
        for dd in sorted(set(day[(t0 >= lo) & (t0 < hi)].tolist())):
            m = (day == dd) & (t0 >= lo) & (t0 < hi)
            zz = z["base"][m]
            tab[str(dd)] = {"events": int(m.sum()), "fast_share": float(fast["base"][m].mean()), "min_z72": float(np.nanmin(zz)) if np.isfinite(zz).any() else float("nan"),
                            "mom_long_up_bp": float(pnl_all["mom_long_up"][m].mean()), "long_up_nofast_bp": float(pnl_all["long_up_nofast"][m].mean())}
    res["selloff_days"] = tab
    dsr = {}
    for v in NEW_VARIANTS:
        st = res["stats"][v]
        sr, t_eff, sk, ku = st["sr_daily"], st["t_eff"], st["skew"], st["kurt"]
        var_asym = (1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2) / max(t_eff - 1.0, 1.0) if np.isfinite(sr) else float("nan")
        tab_ = {}
        for n_tr in (1, 10, int(prior_before), int(prior_before) + len(NEW_VARIANTS)):
            if n_tr <= 1:
                tab_[str(n_tr)] = {"sr_star": 0.0, "dsr": B.psr(sr, t_eff, sk, ku, 0.0) if np.isfinite(sr) else float("nan")}
            else:
                star = B.expected_max_sharpe(n_tr, var_asym) if np.isfinite(var_asym) and var_asym >= 0 else float("nan")
                tab_[str(n_tr)] = {"sr_star": star, "dsr": B.psr(sr, t_eff, sk, ku, star) if np.isfinite(star) else float("nan")}
        dsr[v] = {"sr_daily": sr, "var_sr_used": var_asym, "t_eff": t_eff, "skew": sk, "kurt": ku, "by_trials": tab_}
    res["dsr"] = dsr
    return res, pd.concat(daily_parts, ignore_index=True)
