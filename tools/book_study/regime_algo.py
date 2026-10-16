"""A market-regime rule for the short side, and the long side's matching filter: fixed rules, nothing fitted.

DESIGN (fixed before any result of this module was seen; the same text is printed in SUMMARY.md):

  Why           in the previous study the MOM-long rule that works while the equal-weight index is above its 200-day mean lost about 100 bp per
                candidate in the sell-off (every sell-off event was in that state), and short_all earned in June 2026, the deepest fall and the
                sell-off but loses 63 bp per candidate over the whole history.  The question is whether a market-level condition separates the two.
  Market state  MKT_DOWN = the equal-weight index (the index of the 200-day state: mean hourly log return of the symbols that have a return,
                hours with at least 10 symbols) has fallen over each of the last 480, 720 and 960 hours (20, 30, 40 days): log(idx at the
                event bar) - log(idx at the last index hour at or before the event bar minus L hours) < 0 for all three L.  The same three lookbacks as the stock
                signal; no new parameter.  If a lookback reaches before the index starts, MKT_DOWN is false.
                Two neighbours are declared and reported, never selected: 'tight' (240, 360, 480 hours = 10, 15, 20 days) and 'wide'
                (960, 1440, 1920 hours = 40, 60, 80 days).
  Stock signal  MOM side as in the previous study (20-, 30-, 40-day returns of the symbol all > 0: +1, all < 0: -1, else 0).
  Rules         every rule trades with size 1, 24 bp round trip on taken bets, at the CUSUM events of the source run, same exits (symmetric barriers):
                short_down        side -1 at every event while MKT_DOWN
                short_down_mom    side -1 at events with MOM side -1 while MKT_DOWN
                long_up_notdown   side +1 at events with MOM side +1 while the index is above its 200-day mean AND NOT MKT_DOWN
                regime_rule       long_up_notdown, plus short_down_mom (the two together; flat otherwise)
                regime_rule_tight / regime_rule_wide   regime_rule with the tight / wide MKT_DOWN
                Baselines from the previous study, recomputed here: long_all, short_all, mom_all, mom_long_up.
  Evaluation    one unit of capital per candidate event; a day's result = sum of size * (side * ret_long - cost * [size > 0]) over its events divided by
                ALL candidate events entered that day, in bp.  There is nothing to fit, so no cross-validation: every event is scored once.  Reported:
                the whole history, before / after 2024-01-01 separately, each calendar year, June 2026, the sell-off and the deepest-fall windows, each cell of
                (200-day state x MKT_DOWN), and the share of events with MKT_DOWN true in each window.  PSR on the daily series (effective sample = days /
                max(1, 101/24)); deflated Sharpe for the six new variants with N = 1, 10, 88 (the earlier looks) and 94 (those plus the six rules here).
  Not claimed   the rules were motivated by the sell-off window and the lookbacks and the 200-day state come from earlier diagnostics on this
                history: they are in-sample by construction.  The tight / wide neighbours and the year / half splits show stability, not out-of-sample evidence.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import book_stats as B

LOOKBACK_SETS = {"base": (480, 720, 960), "tight": (240, 360, 480), "wide": (960, 1440, 1920)}
ALL_LOOKBACKS = tuple(sorted({L for v in LOOKBACK_SETS.values() for L in v}))
BASELINES = ("long_all", "short_all", "mom_all", "mom_long_up")
NEW_VARIANTS = ("short_down", "short_down_mom", "long_up_notdown", "regime_rule", "regime_rule_tight", "regime_rule_wide")
VARIANTS = BASELINES + NEW_VARIANTS
PRIOR_TRIALS_BEFORE = 88
TRIALS_AFTER = PRIOR_TRIALS_BEFORE + len(NEW_VARIANTS)

DEVIATIONS = [
    "The rules are motivated by the sell-off window (a failure of the long rule there), so they are in-sample by construction. No free parameter was introduced, and the tight and wide neighbours are reported without selection, but that limits the evidence to stability, not out-of-sample performance.",
    "Fixed rules have no fitted part, so there is no cross-validation, purge or embargo; overlapping holds of neighbouring events are not removed, so the daily-series Sharpe treats days as independent although the holds are 101 bars.",
    "Size is 1 on every rule (no bet sizing); no funding or borrow cost for shorts; the short side is assumed to be executable at the same 24 bp round trip as the long side; no capacity or liquidity limits.",
    "The equal-weight index is the one used for the 200-day state (hours with at least 10 symbols; symbols that exist today, so survivorship applies). Its lookback returns use the latest index hour at or before the stated time.",
    "A window is a few correlated days: its lines carry no intervals and cannot establish anything on their own. The sell-off window is the one that motivated the rules.",
    "All source-run deviations apply (time bars not volume bars, hourly candles, survivorship in the symbol list).",
]


def index_lookback_returns(idx: pd.Series, t0_sec, lookbacks=ALL_LOOKBACKS):
    """{L: array} of log(idx at the last index hour <= t0) - log(idx at the last index hour <= t0 - L hours); NaN where the earlier time precedes the index."""
    t = pd.DatetimeIndex(pd.to_datetime(np.asarray(t0_sec, dtype=np.int64), unit="s"))
    lv = np.log(idx.to_numpy(dtype=np.float64))
    keys = idx.index.to_numpy().astype("datetime64[s]").astype(np.int64)
    t_sec = np.asarray(t0_sec, dtype=np.int64)
    now = np.searchsorted(keys, t_sec, side="right") - 1
    out = {}
    for L in lookbacks:
        past = np.searchsorted(keys, t_sec - L * 3600, side="right") - 1
        ok = (now >= 0) & (past >= 0)
        r = np.full(len(t_sec), np.nan)
        r[ok] = lv[now[ok]] - lv[past[ok]]
        out[L] = r
    return out


def mkt_down(rets: dict, lookbacks) -> np.ndarray:
    """True where the index return is < 0 for every lookback (NaN -> False)."""
    m = np.ones(len(next(iter(rets.values()))), dtype=bool)
    for L in lookbacks:
        m &= np.asarray(rets[L]) < 0
    return m


def build_rules(mom_side, state, down: dict):
    """{variant: side array (+1 / -1 / 0)}; a 0 side means no bet (size 0)."""
    m = np.asarray(mom_side)
    s = np.asarray(state) == 1.0
    n = len(m)
    dA, dT, dW = down["base"], down["tight"], down["wide"]

    def regime(d):
        side = np.zeros(n, dtype=np.int64)
        side[(m == 1) & s & ~d] = 1
        side[(m == -1) & d] = -1
        return side

    out = {
        "long_all": np.ones(n, dtype=np.int64), "short_all": -np.ones(n, dtype=np.int64), "mom_all": m.astype(np.int64),
        "mom_long_up": np.where((m == 1) & s, 1, 0).astype(np.int64),
        "short_down": np.where(dA, -1, 0).astype(np.int64),
        "short_down_mom": np.where(dA & (m == -1), -1, 0).astype(np.int64),
        "long_up_notdown": np.where((m == 1) & s & ~dA, 1, 0).astype(np.int64),
        "regime_rule": regime(dA), "regime_rule_tight": regime(dT), "regime_rule_wide": regime(dW),
    }
    return out


def _daily(day, num):
    return pd.DataFrame({"day": day, "num": num}).groupby("day").agg(sum_bp=("num", "sum"), n=("num", "size")).reset_index()


def best_days_share(x, k=20):
    x = np.asarray(x, dtype=np.float64)
    tot = x.sum()
    if not tot > 0 or len(x) == 0:
        return float("nan")
    return float(np.sort(x)[-k:].sum() / tot)


def series_stats(day, pnl, h):
    """Daily-series statistics of per-event pnl (bp): events, days, mean bp per candidate, Sharpe, skew, kurt, t_eff, PSR, best-20 share."""
    d = _daily(day, pnl)
    dm = (d["sum_bp"] / d["n"]).to_numpy()
    mom = B.sharpe_moments(dm)
    t_eff = mom["n"] / max(1.0, (h + 1) / 24.0)
    return {"events": int(len(pnl)), "days": int(len(d)), "mean_bp_per_candidate": float(np.mean(pnl)) if len(pnl) else float("nan"),
            "sr_daily": mom["sr"], "sr_annual": mom["sr"] * math.sqrt(365) if np.isfinite(mom["sr"]) else float("nan"), "skew": mom["skew"], "kurt": mom["kurt"], "t_eff": t_eff,
            "psr": B.psr(mom["sr"], t_eff, mom["skew"], mom["kurt"]) if np.isfinite(mom["sr"]) else float("nan"), "best20_share": best_days_share(dm)}, d


def evaluate_rules(df, down, cost_frac, h, windows, cutoff_sec, prior_before=PRIOR_TRIALS_BEFORE):
    """df needs t0, ret_long, mkt_state, mom_side.  down: {'base','tight','wide'} boolean arrays.  Returns the nested result dict and the daily frame."""
    ret = df["ret_long"].to_numpy(np.float64)
    t0 = df["t0"].to_numpy(np.int64)
    day = t0 // 86400
    year = pd.DatetimeIndex(pd.to_datetime(t0, unit="s")).year.to_numpy()
    state = df["mkt_state"].to_numpy(np.float64)
    sides = build_rules(df["mom_side"].to_numpy(), state, down)
    n = len(df)
    res = {"stats": {}, "taken_share": {}, "windows": {}, "halves": {}, "years": {}, "cells": {}, "side_split": {}}
    daily_parts = []
    pnl_all = {}
    for v in VARIANTS:
        side = sides[v]
        size = (side != 0).astype(np.float64)
        pnl = size * (side * ret - cost_frac * size) * 1e4
        pnl_all[v] = pnl
        st, d = series_stats(day, pnl, h)
        st["taken_share"] = float(size.mean())
        res["stats"][v] = st
        d.insert(0, "variant", v)
        daily_parts.append(d)
        for wn, (lo, hi) in windows.items():
            m = (t0 >= lo) & (t0 < hi)
            res["windows"][f"{v}|{wn}"] = {"events": int(m.sum()), "trades": int(size[m].sum()), "mean_bp_per_candidate": float(pnl[m].mean()) if m.any() else float("nan")}
        for hn, m in (("before", t0 < cutoff_sec), ("after", t0 >= cutoff_sec)):
            sth, _ = series_stats(day[m], pnl[m], h)
            sth["taken_share"] = float(size[m].mean()) if m.any() else float("nan")
            res["halves"][f"{v}|{hn}"] = sth
        for y in sorted(set(year.tolist())):
            m = year == y
            res["years"][f"{v}|{y}"] = {"events": int(m.sum()), "trades": int(size[m].sum()), "mean_bp_per_candidate": float(pnl[m].mean())}
        for s_ in (0, 1):
            for dn in (0, 1):
                m = (state == s_) & (down["base"] == bool(dn))
                res["cells"][f"{v}|state{s_}|down{dn}"] = {"events": int(m.sum()), "trades": int(size[m].sum()), "mean_bp_per_candidate": float(pnl[m].mean()) if m.any() else float("nan")}
        for sd, nm in ((1, "long"), (-1, "short")):
            m = side == sd
            res["side_split"][f"{v}|{nm}"] = {"trades": int(m.sum()), "bp_per_trade": float(pnl[m].mean()) if m.any() else float("nan")}
    res["window_down_share"] = {}
    for wn, (lo, hi) in windows.items():
        m = (t0 >= lo) & (t0 < hi)
        res["window_down_share"][wn] = {"events": int(m.sum()), **{k: (float(down[k][m].mean()) if m.any() else float("nan")) for k in ("base", "tight", "wide")}}
    res["cell_events"] = {f"state{s_}|down{dn}": int(((state == s_) & (down["base"] == bool(dn))).sum()) for s_ in (0, 1) for dn in (0, 1)}
    res["down_share"] = {k: float(down[k].mean()) for k in ("base", "tight", "wide")}
    dsr = {}
    for v in NEW_VARIANTS:
        st = res["stats"][v]
        sr, t_eff, sk, ku = st["sr_daily"], st["t_eff"], st["skew"], st["kurt"]
        var_asym = (1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2) / max(t_eff - 1.0, 1.0) if np.isfinite(sr) else float("nan")
        tab = {}
        for n_tr in (1, 10, int(prior_before), int(prior_before) + len(NEW_VARIANTS)):
            if n_tr <= 1:
                tab[str(n_tr)] = {"sr_star": 0.0, "dsr": B.psr(sr, t_eff, sk, ku, 0.0) if np.isfinite(sr) else float("nan")}
            else:
                star = B.expected_max_sharpe(n_tr, var_asym) if np.isfinite(var_asym) and var_asym >= 0 else float("nan")
                tab[str(n_tr)] = {"sr_star": star, "dsr": B.psr(sr, t_eff, sk, ku, star) if np.isfinite(star) else float("nan")}
        dsr[v] = {"sr_daily": sr, "var_sr_used": var_asym, "t_eff": t_eff, "skew": sk, "kurt": ku, "by_trials": tab}
    res["dsr"] = dsr
    daily = pd.concat(daily_parts, ignore_index=True)
    return res, daily
