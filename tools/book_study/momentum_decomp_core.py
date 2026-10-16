"""Rule-free decomposition of a momentum signal on CUSUM events (no project imports).

No stop/profit rule and no fitted parameter: every event is simply held for H+1 bars (the same 'hold-H' read as the model scorecard),
so the result cannot be an artefact of a rule chosen by simulation.  For one lookback L, one market state and one horizon H it reports:

  ret          raw forward log return of the event (bp): column H of the event's forward path times its one-bar sigma
  long leg     mean ret of events whose trailing return was positive (the signal says long)
  short leg    mean of -ret of events whose trailing return was negative (the signal says short)
  pooled       mean of sign * ret                         (the signal's gross result; random signs would give about 0 minus drift terms)
  always-long  mean ret over the same events              (what simply being long earned: the survivor drift)
  spread       mean ret | winners  -  mean ret | losers   (drift cancels: does the sign sort future returns at all?)

with 95% intervals from resampling blocks of whole UTC days (block length = ceil((H+1)/24) days, so that holds of neighbouring events in the same block overlap
less with the next block; events on one day share the market move).  Also by calendar year, how much of the total comes from the best 20 days, the share of symbols
that are positive, and one line per reporting window (June 2026, the sell-off, the deepest fall).  The lookbacks tried so far (7, 20, 30, 40, 90 days) are counted as trials and stated in the report.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import book_stats as B
import book_study_core as C

LOOKBACKS = (168, 480, 720, 960, 2160)
HORIZONS = (24, 100)


def cluster_mean_ci(day, x, n_boot, seed):
    """Mean of x with a day-cluster bootstrap 95% interval: (mean, lo, hi)."""
    day = np.asarray(day)
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    u, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=x, minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b0 in range(0, n_boot, 500):
        m = min(500, n_boot - b0)
        idx = rng.integers(0, len(u), size=(m, len(u)))
        out[b0:b0 + m] = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(x.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def spread_ci(day, x, up, n_boot, seed):
    """mean(x | up) - mean(x | not up) with a day-cluster bootstrap: (spread, lo, hi)."""
    up = np.asarray(up, dtype=bool)
    if up.sum() == 0 or (~up).sum() == 0:
        return float("nan"), float("nan"), float("nan")
    u, inv = np.unique(np.asarray(day), return_inverse=True)
    xs = np.asarray(x, dtype=np.float64)
    a_s = np.bincount(inv[up], weights=xs[up], minlength=len(u))
    a_n = np.bincount(inv[up], minlength=len(u)).astype(float)
    b_s = np.bincount(inv[~up], weights=xs[~up], minlength=len(u))
    b_n = np.bincount(inv[~up], minlength=len(u)).astype(float)
    spread = a_s.sum() / a_n.sum() - b_s.sum() / b_n.sum()
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b0 in range(0, n_boot, 500):
        m = min(500, n_boot - b0)
        idx = rng.integers(0, len(u), size=(m, len(u)))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[b0:b0 + m] = a_s[idx].sum(axis=1) / a_n[idx].sum(axis=1) - b_s[idx].sum(axis=1) / b_n[idx].sum(axis=1)
    out = out[np.isfinite(out)]
    return float(spread), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def block_ids(t0, h):
    """Cluster id = UTC-day number // block_days, block_days = ceil((h+1)/24)."""
    bd = int(math.ceil((h + 1) / 24.0))
    return (np.asarray(t0) // 86400).astype(np.int64) // bd, bd


def leg_stats(r, s):
    """(pooled, long leg, short leg, always-long, spread) means in bp for returns r and signs s (all non-zero)."""
    up = s > 0
    pooled = float((s * r).mean())
    ll = float(r[up].mean()) if up.any() else float("nan")
    sl = float(-r[~up].mean()) if (~up).any() else float("nan")
    al = float(r.mean())
    sp = float(r[up].mean() - r[~up].mean()) if up.any() and (~up).any() else float("nan")
    return pooled, ll, sl, al, sp


def decompose(ev: dict, state: np.ndarray, cost_frac: float, n_boot: int, seed: int, lookbacks=LOOKBACKS, horizons=HORIZONS, windows=None):
    """Returns (rows, by_year, win_rows) lists of dicts. ev must contain sig_mom<L> for every lookback.
    windows: {name: (start_ts, end_ts_exclusive)} (UTC); one row per (lookback, horizon, window) over events entered inside it, all states."""
    sigma = ev["sigma"].astype(np.float64)
    t0 = ev["t0"]
    day = (t0 // 86400).astype(np.int64)
    year = pd.to_datetime(t0, unit="s").year.to_numpy()
    sym = ev["sym"]
    rows, by_year, win_rows = [], [], []
    cost_bp = cost_frac * 1e4
    for L in lookbacks:
        sig = ev[f"sig_mom{L}"].astype(np.float64)
        for h in horizons:
            ret_all = ev["q"][:, h].astype(np.float64) * sigma * 1e4
            blk_all, bd = block_ids(t0, h)
            for sname in C.STATES:
                m = C.state_mask(state, sname) & (sig != 0)
                if m.sum() < 50:
                    continue
                r, s, d, b, y, sy = ret_all[m], sig[m], day[m], blk_all[m], year[m], sym[m]
                pooled = s * r
                up = s > 0
                row = {"lookback_bars": L, "lookback_days": L / 24.0, "horizon": h, "state": sname, "events": int(m.sum()),
                       "share_long_signal": float(up.mean()), "days": int(len(np.unique(d))), "block_days": bd,
                       "blocks": int(len(np.unique(b))), "symbols": int(len(np.unique(sy)))}
                mean, lo, hi = cluster_mean_ci(b, pooled, n_boot, seed + L + h)
                row.update({"pooled_gross_bp": mean, "pooled_lo": lo, "pooled_hi": hi, "pooled_net_bp": mean - cost_bp})
                row["long_leg_bp"] = float(r[up].mean()) if up.any() else float("nan")
                row["short_leg_bp"] = float(-r[~up].mean()) if (~up).any() else float("nan")
                al, al_lo, al_hi = cluster_mean_ci(b, r, n_boot, seed + 7 * L + h)
                row.update({"always_long_bp": al, "always_long_lo": al_lo, "always_long_hi": al_hi})
                sp, sp_lo, sp_hi = spread_ci(b, r, up, n_boot, seed + 13 * L + h)
                row.update({"spread_bp": sp, "spread_lo": sp_lo, "spread_hi": sp_hi})
                # concentration: the best 20 days' share of the total (events summed per UTC day)
                per_day = pd.Series(pooled).groupby(d).sum()
                tot = float(per_day.sum())
                best20 = float(per_day.sort_values(ascending=False).head(20).sum())
                row["total_gross_bp_events"] = tot
                row["best20_days_share_of_total"] = best20 / tot if tot > 0 else float("nan")
                pf = pd.DataFrame({"sym": sy, "p": pooled})
                ps = pf.groupby("sym")["p"].mean()
                row["symbols_positive_share"] = float((ps > 0).mean())
                tot_sym = pf.groupby("sym")["p"].sum().sort_values(ascending=False)
                row["top5_symbols_share_of_total"] = float(tot_sym.head(5).sum() / tot) if tot > 0 else float("nan")
                rows.append(row)
                for yv in np.unique(y):
                    k = y == yv
                    pg, ll, sl, al_y, sp_y = leg_stats(r[k], s[k])
                    by_year.append({"lookback_bars": L, "horizon": h, "state": sname, "year": int(yv), "events": int(k.sum()),
                                    "pooled_gross_bp": pg, "long_leg_bp": ll, "short_leg_bp": sl, "always_long_bp": al_y, "spread_bp": sp_y})
            for wname, (w0, w1) in (windows or {}).items():
                lo_s, hi_s = int(pd.Timestamp(w0).timestamp()), int(pd.Timestamp(w1).timestamp())
                m = (t0 >= lo_s) & (t0 < hi_s) & (sig != 0)
                wr = {"lookback_bars": L, "horizon": h, "window": wname, "events": int(m.sum())}
                if m.sum() > 0:
                    pg, ll, sl, al_w, sp_w = leg_stats(ret_all[m], sig[m])
                    wr.update({"pooled_gross_bp": pg, "pooled_net_bp": pg - cost_bp, "long_leg_bp": ll, "short_leg_bp": sl,
                               "always_long_bp": al_w, "spread_bp": sp_w, "days": int(len(np.unique(day[m]))), "symbols": int(len(np.unique(sym[m])))})
                win_rows.append(wr)
    return rows, by_year, win_rows


def render(rows, by_year, win_rows, meta, notes) -> str:
    def f(x, nd=1):
        return "n/a" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.{nd}f}"

    L = ["# Momentum signal, rule-free decomposition (read-only diagnostic)", ""]
    L.append("Each event is held for H+1 bars with no stop and no profit target, so no rule was chosen by simulation. 'pooled' is the signal's gross result; "
             "'always-long' is what the same events earned for a long holder (survivor drift); 'spread' is winners minus losers, in which drift cancels. "
             "Intervals are 95%, from resampling blocks of whole UTC days (block length in the 'block days' column). Net = pooled minus the round-trip cost. 'Best 20 days' share' and 'top-5 symbols' share' are shares of the pooled total; above 1.00 means the remaining days or symbols lost money, n/a means the total was not positive.")
    L.append("")
    for k, v in meta.items():
        L.append(f"- {k}: {v}")
    L.append("")
    for h in sorted({r["horizon"] for r in rows}):
        L.append(f"## Hold {h + 1} bars (H = {h})")
        L.append("")
        L.append("| lookback (days) | state | events | block days | long-signal share | pooled gross bp [95%] | net bp | long leg bp | short leg bp | always-long bp [95%] | winners-minus-losers bp [95%] | best 20 days' share | symbols positive | top-5 symbols' share |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["horizon"] != h:
                continue
            L.append("| {d:.0f} | {s} | {n} | {bd} | {ls} | {p} [{pl}, {ph}] | {net} | {lg} | {sh} | {al} [{all_}, {alh}] | {sp} [{spl}, {sph}] | {b20} | {sp_} | {t5} |".format(
                d=r["lookback_days"], s=r["state"], n=r["events"], bd=r["block_days"], ls=f(r["share_long_signal"], 2), p=f(r["pooled_gross_bp"]), pl=f(r["pooled_lo"]), ph=f(r["pooled_hi"]),
                net=f(r["pooled_net_bp"]), lg=f(r["long_leg_bp"]), sh=f(r["short_leg_bp"]), al=f(r["always_long_bp"]), all_=f(r["always_long_lo"]), alh=f(r["always_long_hi"]),
                sp=f(r["spread_bp"]), spl=f(r["spread_lo"]), sph=f(r["spread_hi"]), b20=f(r["best20_days_share_of_total"], 2),
                sp_=f(r["symbols_positive_share"], 2), t5=f(r["top5_symbols_share_of_total"], 2)))
        L.append("")
        L.append("By calendar year (pooled gross bp / winners-minus-losers bp), above-200-day state:")
        L.append("")
        years = sorted({b["year"] for b in by_year})
        L.append("| lookback (days) | " + " | ".join(str(y) for y in years) + " |")
        L.append("|---|" + "---|" * len(years))
        for lb in sorted({b["lookback_bars"] for b in by_year}):
            cells = []
            for y in years:
                b = [x for x in by_year if x["lookback_bars"] == lb and x["horizon"] == h and x["state"] == "above" and x["year"] == y]
                cells.append(f"{f(b[0]['pooled_gross_bp'])} / {f(b[0]['spread_bp'])} (n={b[0]['events']})" if b else "-")
            L.append(f"| {lb / 24:.0f} | " + " | ".join(cells) + " |")
        L.append("")
        wn = [w for w in dict.fromkeys(x["window"] for x in win_rows)]
        if wn:
            L.append("Reporting windows (events entered inside the window, all states; no intervals because a window is a handful of correlated days):")
            L.append("")
            L.append("| lookback (days) | window | events | days | pooled gross bp | net bp | long leg bp | short leg bp | always-long bp | winners-minus-losers bp |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for x in win_rows:
                if x["horizon"] != h:
                    continue
                if x["events"] == 0:
                    L.append(f"| {x['lookback_bars'] / 24:.0f} | {x['window']} | 0 | | | | | | | |")
                else:
                    L.append(f"| {x['lookback_bars'] / 24:.0f} | {x['window']} | {x['events']} | {x['days']} | {f(x['pooled_gross_bp'])} | {f(x['pooled_net_bp'])} | "
                             f"{f(x['long_leg_bp'])} | {f(x['short_leg_bp'])} | {f(x['always_long_bp'])} | {f(x['spread_bp'])} |")
            L.append("")
    L.append("## Notes and limits")
    L.append("")
    for i, n in enumerate(notes, 1):
        L.append(f"{i}. {n}")
    return "\n".join(L) + "\n"
