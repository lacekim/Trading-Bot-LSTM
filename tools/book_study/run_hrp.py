#!/usr/bin/env python
"""Ch.16 allocation across symbols (read-only diagnostic; no model is fitted, nothing in the run folders is touched).

What is allocated: the daily result stream of the strategy in each symbol (bet_averaging.symbol_daily: position = mean signal of the active events, zero when none, 12 bp per side on
position changes), as a fraction of the capital placed in that symbol.  It is the strategy stream, not the asset's own return.

Declared before any result:
  * estimation: the 730 UTC days before the walk-forward cutoff (EST_DAYS), primary stream (side, size 1) of the events whose labels end before the cutoff.  A symbol is eligible only if
    it has candles on or before the first day of the window and on or after the last, and at least 100 active days (a candidate event active) in the window (MIN_ACTIVE).
    Covariance: sample covariance (ddof 1) of the eligible symbols' daily results in bp, zeros on days without activity or without a bar.
  * schemes on the same eligible set: equal weight, inverse variance, HRP (hrp_alloc.py), long-only minimum variance.  Weights are fitted once and held fixed after the cutoff.
  * scoring: events starting on or after the cutoff of the eligible symbols.  Daily portfolio result (bp of capital) = sum over symbols of weight x symbol result, where a symbol whose
    candle data has ended (no bar that UTC day) is left out and the weights of the others are rescaled to sum to 1 that day (all schemes alike).  Days: UTC days from the cutoff day to the last
    day with a bar in any eligible symbol.  Streams: primary, and meta for every --run (walk-forward bet sizes of that run's seed).  The weights never use a meta stream or a post-cutoff day.
  * windows: the June 2026 AI correction and the sell-off (2026-09-27 to 2026-10-08) are reported separately (windows entirely before the cutoff are left out); also by calendar year.
No new model is fitted.  Choosing among four schemes on post-cutoff data is three new comparisons, counted in the notes.

    python tools/book_study/run_hrp.py --run "$(ls -d runtime/book_seed_pairs/volume_vs_14/ctrl_20261017/*/ | tail -1)" ... (one --run folder per seed, same events.csv)
    python tools/book_study/verify_hrp.py runtime/book_hrp/<timestamp>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import bet_averaging as BA  # noqa: E402
import hrp_alloc as HA  # noqa: E402
import meta_model as MM  # noqa: E402
import run_bet_averaging as RBA  # noqa: E402
import run_book_study as RB  # noqa: E402

DAY = 86400
EST_DAYS = 730
MIN_ACTIVE = 100
SCHEMES = ["equal_weight", "inverse_variance", "hrp", "min_variance"]


def symbol_series(close, ts, pos, t1pos, signal, cost_side):
    r = BA.symbol_daily(close, ts, pos, t1pos, signal, cost_side)
    return r["days"], r["gross"] - r["cost"], r["active"]


def sharpe_annual(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(365)) if sd > 0 else float("nan")


def portfolio_series(W, days_by_sym, net_by_sym, d_lo, d_hi):
    """W: weights (n,), per symbol (days, net) -> calendar days d_lo..d_hi, series in bp (sum w*net over symbols with a bar that day / sum of their weights), the days with any bar."""
    T = d_hi - d_lo + 1
    n = len(W)
    N, P = np.zeros((n, T)), np.zeros((n, T))
    for s in range(n):
        dd, nn = days_by_sym[s], net_by_sym[s]
        m = (dd >= d_lo) & (dd <= d_hi)
        N[s, dd[m] - d_lo] = nn[m]
        P[s, dd[m] - d_lo] = 1.0
    any_bar = P.sum(axis=0) > 0
    num, den = W @ N, W @ P
    ret = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return np.arange(d_lo, d_hi + 1)[any_bar], 1e4 * ret[any_bar]


def metrics(day_nums, bp, windows):
    out = {"n_days": int(len(bp)), "mean_bp": float(bp.mean()), "sd_bp": float(bp.std(ddof=1)), "sr_annual": sharpe_annual(bp)}
    for wn, (lo, hi) in windows.items():
        m = (day_nums * DAY >= lo) & (day_nums * DAY < hi)
        out[f"win_{wn}_bp"] = float(bp[m].mean()) if m.any() else float("nan")
        out[f"win_{wn}_days"] = int(m.sum())
    years = np.array([datetime.fromtimestamp(int(d) * DAY, tz=timezone.utc).year for d in day_nums])
    for y in sorted(set(years.tolist())):
        out[f"year_{y}_days"] = int((years == y).sum())
        out[f"year_{y}_mean_bp"] = float(bp[years == y].mean())
        out[f"year_{y}_sr_annual"] = sharpe_annual(bp[years == y])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="timestamp folders of 14-feature runs with the same events.csv (one per seed); the first supplies events, candles and the estimation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--candle-dir", default=None)
    a = ap.parse_args()
    runs = []
    for r in a.run:
        p = Path(r)
        runs.append((p if p.is_absolute() else ROOT / p).resolve())
    summ, ev, closes, cd = RBA.load_inputs(runs[0], a.candle_dir)
    RBA.check_alignment(ev, closes)
    seeds_seen = [int(json.loads((r / "summary_numbers.json").read_text(encoding="utf-8"))["params"]["seed"]) for r in runs]
    if len(set(seeds_seen)) != len(seeds_seen):
        raise SystemExit(f"two --run folders have the same seed {seeds_seen}; each run must be a different seed")
    ev_md5 = RB.md5_file(runs[0] / "events.csv")
    for r in runs[1:]:
        if RB.md5_file(r / "events.csv") != ev_md5:
            raise SystemExit(f"{r}: events.csv differs from the first run's")
    p = summ["params"]
    cost_side = float(p["cost_frac"]) / 2.0
    step = float(p["step"])
    cut = float(pd.Timestamp(p["cutoff"]).timestamp())
    cutday = int(cut // DAY)
    w0 = cutday - EST_DAYS
    windows = {k: (int(v[0]), int(v[1])) for k, v in summ["meta"]["window_epoch"].items() if int(v[1]) > cut}      # windows entirely before the cutoff are not scored here
    t0, t1 = ev["t0"].to_numpy(np.int64), ev["t1"].to_numpy(np.int64)
    side = ev["side"].to_numpy(np.float64)
    pos_all, t1p_all = ev["pos"].to_numpy(np.int64), ev["t1pos"].to_numpy(np.int64)
    sym_arr = ev["symbol"].to_numpy()
    ts_c = {s: c.index.astype("datetime64[s]").astype(np.int64) for s, c in closes.items()}
    cl_c = {s: c.to_numpy(dtype=np.float64) for s, c in closes.items()}

    # ---- eligibility and estimation stream (primary, labels ended before the cutoff)
    elig_rows, est_cols, kept = [], {}, []
    for sym in sorted(ev["symbol"].unique()):
        ts = ts_c[sym]
        first_day, last_day = int(ts[0] // DAY), int(ts[-1] // DAY)
        e_pre = np.flatnonzero((sym_arr == sym) & (t1 < cut))
        n_pre, n_post = int(len(e_pre)), int(((sym_arr == sym) & (t0 >= cut)).sum())
        act_days, reason = 0, ""
        if n_pre:
            dd, nn, ac = symbol_series(cl_c[sym], ts, pos_all[e_pre], t1p_all[e_pre], side[e_pre], cost_side)
            inwin = (dd >= w0) & (dd < cutday)
            act_days = int(ac[inwin].sum())
            col = np.zeros(EST_DAYS)
            col[dd[inwin] - w0] = 1e4 * nn[inwin]
        if first_day > w0 or last_day < cutday - 1:
            reason = "candles do not cover the estimation window"
        elif act_days < MIN_ACTIVE:
            reason = f"fewer than {MIN_ACTIVE} active days in the window"
        elif col.std(ddof=1) == 0:
            reason = "zero variance"
        ok = reason == ""
        if ok:
            est_cols[sym] = col
            kept.append(sym)
        elig_rows.append({"symbol": sym, "first_bar_day": first_day, "last_bar_day": last_day, "events_pre_cutoff": n_pre, "events_post_cutoff": n_post,
                          "active_days_in_window": act_days, "eligible": int(ok), "reason": reason})
    if len(kept) < 3:
        raise SystemExit(f"only {len(kept)} eligible symbols")
    M = np.column_stack([est_cols[s] for s in kept])
    cov = np.cov(M, rowvar=False, ddof=1)
    n = len(kept)
    w_hrp, order, merges = HA.hrp_weights(cov)
    W = {"equal_weight": HA.equal_weight(n), "inverse_variance": HA.inverse_variance(cov), "hrp": w_hrp, "min_variance": HA.min_variance_long_only(cov)}
    for k, v in W.items():
        if abs(v.sum() - 1.0) > 1e-9 or (v < -1e-12).any():
            raise SystemExit(f"{k}: weights not a long-only unit-sum vector")

    # ---- evaluation streams
    idx_of = {s: i for i, s in enumerate(kept)}
    post_all = np.flatnonzero(t0 >= cut)
    post_el = np.array([e for e in post_all if sym_arr[e] in idx_of], dtype=np.int64)
    streams = [("primary", "primary", None, np.ones(len(post_el)), post_el)]
    for r in runs:
        rs = json.loads((r / "summary_numbers.json").read_text(encoding="utf-8"))
        wfp = pd.read_csv(r / "walk_forward_predictions.csv")
        if set(wfp["event"].to_numpy(np.int64).tolist()) != set(post_all.tolist()):
            raise SystemExit(f"{r}: walk-forward predictions do not cover exactly the events starting on or after the cutoff")
        sz = pd.Series(MM.bet_size(wfp["p"].to_numpy(np.float64), step), index=wfp["event"].to_numpy(np.int64))
        keep = np.array([e for e in post_el], dtype=np.int64)
        streams.append((f"meta_{int(rs['params']['seed'])}", "meta", int(rs["params"]["seed"]), sz.loc[keep].to_numpy(np.float64), keep))
    d_hi = max(int(ts_c[s][-1] // DAY) for s in kept)
    metric_rows, daily_rows = [], []
    for sname, kind, seed, size, idx in streams:
        days_by, net_by = [], []
        for s in kept:
            m = np.flatnonzero(sym_arr[idx] == s)
            if len(m):
                dd, nn, _ = symbol_series(cl_c[s], ts_c[s], pos_all[idx[m]], t1p_all[idx[m]], side[idx[m]] * size[m], cost_side)
            else:
                dd, nn = np.unique(ts_c[s] // DAY), np.zeros(len(np.unique(ts_c[s] // DAY)))
            days_by.append(dd)
            net_by.append(nn)
        for sch in SCHEMES:
            dn, bp = portfolio_series(W[sch], days_by, net_by, cutday, d_hi)
            row = {"stream": sname, "kind": kind, "seed": seed if seed is not None else "", "scheme": sch}
            row.update(metrics(dn, bp, windows))
            metric_rows.append(row)
            daily_rows += [{"stream": sname, "scheme": sch, "day": int(d_), "bp": float(b_)} for d_, b_ in zip(dn, bp)]

    scheme_rows = []
    for sch in SCHEMES:
        w = W[sch]
        is_sd = float((M @ w).std(ddof=1))
        oos_sd = next(r["sd_bp"] for r in metric_rows if r["stream"] == "primary" and r["scheme"] == sch)
        scheme_rows.append({"scheme": sch, "effective_n": float(1.0 / (w ** 2).sum()), "max_weight": float(w.max()), "n_zero": int((w == 0).sum()), "in_sample_sd_bp": is_sd,
                            "oos_sd_bp_primary": oos_sd, "oos_over_is": oos_sd / is_sd})

    started = datetime.now(timezone.utc)
    out = Path(a.out) if a.out else ROOT / "runtime" / "book_hrp" / started.strftime("%Y%m%dT%H%M%SZ")
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(elig_rows).to_csv(out / "eligibility.csv", index=False)
    pd.DataFrame([{"symbol": s, **{sch: W[sch][i] for sch in SCHEMES}} for i, s in enumerate(kept)]).to_csv(out / "weights.csv", index=False, float_format="%.15g")
    pd.DataFrame(cov, index=kept, columns=kept).rename_axis("symbol").to_csv(out / "cov_bp2.csv", float_format="%.15g")
    pd.DataFrame([{"merge": i, "left": m[0], "right": m[1], "height": m[2], "size": m[3]} for i, m in enumerate(merges)]).to_csv(out / "linkage.csv", index=False, float_format="%.15g")
    pd.DataFrame({"position": range(n), "symbol": [kept[i] for i in order]}).to_csv(out / "hrp_order.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False, float_format="%.15g")
    pd.DataFrame(scheme_rows).to_csv(out / "scheme_stats.csv", index=False, float_format="%.15g")
    pd.DataFrame(daily_rows).to_csv(out / "portfolio_daily.csv", index=False, float_format="%.12g")
    el = pd.DataFrame(elig_rows)
    dropped = int(el.loc[el["eligible"] == 0, "events_post_cutoff"].sum())
    extra = {"source_runs": [str(r) for r in runs], "candle_dir": str(cd), "cutoff": p["cutoff"], "est_days": EST_DAYS, "min_active_days": MIN_ACTIVE, "window_first_day": w0, "cutoff_day": cutday,
             "last_eval_day": d_hi, "step": step, "cost_frac": float(p["cost_frac"]), "schemes": SCHEMES, "eligible": kept, "n_events": int(len(ev)), "events_post_cutoff": int(len(post_all)),
             "events_post_cutoff_dropped": dropped, "events_md5": ev_md5, "windows": {k: list(v) for k, v in windows.items()}, "prior_trials": 108, "new_comparisons": 3,
             "script_md5": {nme: RB.md5_file(HERE / nme) for nme in ("hrp_alloc.py", "run_hrp.py", "bet_averaging.py", "run_bet_averaging.py")}}
    (out / "hrp_extra.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    (out / "hrp_summary.md").write_text(render(extra, pd.DataFrame(elig_rows), kept, W, pd.DataFrame(scheme_rows), pd.DataFrame(metric_rows), order), encoding="utf-8")
    print((out / "hrp_summary.md").read_text(encoding="utf-8"))
    print(f"written to {out}")
    return 0


def f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def render(extra, el, kept, W, ss, mt, order) -> str:
    L = ["# Ch.16 allocation across symbols (read-only diagnostic)", "",
         f"Estimation window: the {extra['est_days']} UTC days before {extra['cutoff']} (primary stream, labels ended before the cutoff). Eligible: candles over the whole window and at least {extra['min_active_days']} active days. "
         f"Weights fitted once, held fixed after the cutoff. Streams are the strategy's daily result per symbol (Ch.10 averaging rule, 12 bp per side), not asset returns.", "",
         f"Eligible symbols ({len(kept)}): {', '.join(kept)}. Post-cutoff events of ineligible symbols are dropped: {extra['events_post_cutoff_dropped']} of {extra['events_post_cutoff']}.", "",
         "## Eligibility", "", "| symbol | first bar day | last bar day | events before cutoff | events after cutoff | active days in window | eligible | reason |", "|---|---|---|---|---|---|---|---|"]
    for r in el.itertuples():
        L.append(f"| {r.symbol} | {r.first_bar_day} | {r.last_bar_day} | {r.events_pre_cutoff} | {r.events_post_cutoff} | {r.active_days_in_window} | {'yes' if r.eligible else 'no'} | {r.reason if isinstance(r.reason, str) else ''} |")
    L += ["", f"HRP leaf order: {', '.join(kept[i] for i in order)}.", "", "## Weights", "", "| symbol | " + " | ".join(SCHEMES) + " |", "|---|" + "---|" * len(SCHEMES)]
    for i, s in enumerate(kept):
        L.append(f"| {s} | " + " | ".join(f(W[k][i], 4) for k in SCHEMES) + " |")
    L += ["", "## Weight structure and in-sample vs out-of-sample volatility (primary stream, bp of capital per day)", "",
          "| scheme | effective number of names | largest weight | names with zero weight | in-sample sd | out-of-sample sd | out-of-sample / in-sample |", "|---|---|---|---|---|---|---|"]
    for r in ss.itertuples():
        L.append(f"| {r.scheme} | {f(r.effective_n)} | {f(r.max_weight, 3)} | {r.n_zero} | {f(r.in_sample_sd_bp)} | {f(r.oos_sd_bp_primary)} | {f(r.oos_over_is, 2)} |")
    wn = list(extra["windows"])
    L += ["", "## Post-cutoff, primary stream", "", "| scheme | days | mean bp/day | sd bp/day | annualised Sharpe | " + " | ".join(f"{w} mean bp/day" for w in wn) + " |", "|---|---|---|---|---|" + "---|" * len(wn)]
    pr = mt[mt["stream"] == "primary"]
    for r in pr.itertuples(index=False):
        d = r._asdict()
        L.append(f"| {d['scheme']} | {d['n_days']} | {f(d['mean_bp'])} | {f(d['sd_bp'])} | {f(d['sr_annual'])} | " + " | ".join(f(d[f'win_{w}_bp']) for w in wn) + " |")
    years = sorted({int(c.split("_")[1]) for c in mt.columns if c.startswith("year_") and c.endswith("_mean_bp")})
    L += ["", "By calendar year (primary): mean bp/day (annualised Sharpe)", "", "| scheme | " + " | ".join(str(y) for y in years) + " |", "|---|" + "---|" * len(years)]
    for r in pr.itertuples(index=False):
        d = r._asdict()
        L.append(f"| {d['scheme']} | " + " | ".join(f"{f(d[f'year_{y}_mean_bp'])} ({f(d[f'year_{y}_sr_annual'])})" for y in years) + " |")
    me = mt[mt["kind"] == "meta"]
    seeds = sorted(me["seed"].unique())
    L += ["", f"## Post-cutoff, meta stream ({len(seeds)} seeds; mean over seeds, sd in brackets)", "", "| scheme | mean bp/day | sd bp/day | annualised Sharpe | " + " | ".join(f"{w} mean bp/day" for w in wn) + " |", "|---|---|---|---|" + "---|" * len(wn)]
    for sch in SCHEMES:
        g = me[me["scheme"] == sch]
        cells = [f"{f(g[c].mean())} ({f(g[c].std(ddof=1)) if len(g) > 1 else 'n/a'})" for c in ["mean_bp", "sd_bp", "sr_annual"] + [f"win_{w}_bp" for w in wn]]
        L.append(f"| {sch} | " + " | ".join(cells) + " |")
    L += ["", "Scheme minus equal weight, meta stream, per seed (mean difference; seeds with a higher value):", "", "| scheme | Sharpe difference | seeds higher | mean bp/day difference | seeds higher | " + " | ".join(f"{w} bp/day difference (seeds higher)" for w in wn) + " |",
          "|---|---|---|---|---|" + "---|" * len(wn)]
    base = me[me["scheme"] == "equal_weight"].set_index("seed")
    for sch in SCHEMES[1:]:
        g = me[me["scheme"] == sch].set_index("seed")
        cells = []
        for c in ["sr_annual", "mean_bp"] + [f"win_{w}_bp" for w in wn]:
            dlt = (g[c] - base[c]).dropna()
            cells.append(f"{f(dlt.mean(), 3)} ({int((dlt > 0).sum())} of {len(dlt)})" if c.startswith("win_") else f"{f(dlt.mean(), 3)} | {int((dlt > 0).sum())} of {len(dlt)}")
        L.append(f"| {sch} | " + " | ".join(cells) + " |")
    L += ["", f"Trial count: no model fitted; four schemes compared on post-cutoff data = {extra['new_comparisons']} new comparisons after look {extra['prior_trials']}. The post-cutoff period was seen in every earlier run.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
