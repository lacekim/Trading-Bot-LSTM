"""book_full.py with the Chapter 9 tuned forest (tuned_forest.json from book_ch8_ch9.py) and a fresh walk-forward fit, plus the Chapter 14 PSR / DSR of the one-year and whole-span account series.  New file; nothing audited is edited.
Otherwise identical to book_full.py (same stages, account, HRP, variants).

THE STRATEGY (every stage is a chapter of the book; the extract's own code where it has one, the audited ports otherwise):
  Ch.2   dollar bars (the volume clock) and the CUSUM filter (Snippet 2.4) select the bars at which a bet is considered
  Ch.3   labels with no side given (Snippets 3.5 / 3.6 / 3.7 case 1): bin = sign of the return at the first touch; the model learns side and size
  Ch.4   average uniqueness and return-attribution sample weights, sequential bootstrap
  Ch.5   fractional differentiation (FFD) feature;  Ch.17 SADF, Ch.18 entropy, Ch.19 Corwin-Schultz features;  Snippet 3.1 daily volatility feature
  Ch.6   bagged entropy forest;  Ch.7 purged k-fold / embargo;  Ch.12 combinatorial purged CV (5 backtest paths) and walk-forward (trained before 2024-01-01 only)
  Ch.10  side = predicted class, size = Snippets 10.1 / 10.3 of its probability, overlapping bets averaged (10.2)
  Ch.13  the optimal trading rule (profit-taking / stop-loss in sigma units, maximum holding) that the venue's Chapter 13 run chose from the pre-cutoff events only closes each bet
  Ch.16  Hierarchical Risk Parity (Snippet 16.4, book_port_d.getRecBipart / getQuasiDiag / correlDist as ported) allocates the account across instruments, re-estimated every quarter from the
         trailing 252 trading days of daily returns of the instruments that have them (no look-ahead: only returns before the rebalance day)
ACCOUNT: an instrument's slice of capital = its HRP weight (weights sum to 1; no leverage); the position inside the slice is the averaged signal in [-1, 1]; the account's daily return =
sum over instruments of weight x (gross - cost) of that instrument's day; costs on every position change (Kraken 12 bp per side, Alpaca 10 bp per side); compounded daily; instruments without
a weight (not enough history) sit in cash.  'equal' = the same with 1/N weights over the same eligible set, shown only to isolate the HRP step.
VARIANTS: book_bet (side x confidence size), pred_side_full_size, placebo_sizes (the book_bet sizes shuffled), random_side and always_long (references; walk-forward only).
Usage: python book_full_tuned.py --source <meta run> --bars <dollar bars> --ports <book_ports run> --forest-json <tuned_forest.json> --out <folder> [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import bet_averaging as BA  # noqa: E402
import book_otr_exit as OX  # noqa: E402
import book_port_d as PD  # noqa: E402
import meta_eval as ME  # noqa: E402
import meta_events as E  # noqa: E402
import meta_model as MM  # noqa: E402
import ports_common as PCm  # noqa: E402
import run_bet_averaging as RBA  # noqa: E402
import run_book_study as RB  # noqa: E402

DAY = 86400
FEATURES_BOOK = ["ffd", "log_trgt", "sadf", "ent_plugin", "ent_konto", "cs_spread"]
REB_WINDOW = 252


def hrp_weights(rets: pd.DataFrame, eq: bool = False) -> pd.Series:
    """Snippet 16.4's steps on a returns frame (columns = instruments): correlation distance, single linkage, quasi-diagonalisation, recursive bisection.  eq: 1/N over the same columns."""
    cols = list(rets.columns)
    if eq or len(cols) < 3:
        return pd.Series(1.0 / len(cols), index=cols)
    cov, corr = rets.cov(), rets.corr()
    dist = PD.correlDist(corr)
    link = PD.sch.linkage(dist, "single")
    sortIx = PD.getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    w = PD.getRecBipart(cov, sortIx)
    return w.reindex(cols)


def weight_schedule(rets: pd.DataFrame, day_lo: int, day_hi: int, eq: bool):
    """{first day of each calendar quarter in [day_lo, day_hi] (and day_lo itself): weights Series}.  Weights at day d use returns strictly before the rebalance day."""
    idx_days = (rets.index.to_numpy().astype("datetime64[s]").astype(np.int64)) // DAY
    days = np.arange(day_lo, day_hi + 1)
    dt = pd.to_datetime(days * DAY, unit="s")
    q = (dt.year * 4 + (dt.month - 1) // 3)
    starts = [int(days[0])] + [int(days[i]) for i in range(1, len(days)) if q[i] != q[i - 1]]
    sched = {}
    for r in starts:
        win = rets[idx_days < r].tail(REB_WINDOW)
        if len(win) < REB_WINDOW * 0.8:
            sched[r] = pd.Series(dtype=float)
            continue
        ok = win.columns[(win.notna().sum() >= 0.8 * len(win)) & (win.fillna(0.0).std() > 0)]
        w = hrp_weights(win[ok].fillna(0.0), eq)
        sched[r] = w / w.sum()
    return sched


def instrument_nets(ev, closes, idx, side, size, t1pos, cost_side):
    """{symbol: (days array, net array)}: the instrument's daily result (gross - cost) of the averaged signal side*size with exits t1pos, per unit of capital in the instrument."""
    sym_arr = ev["symbol"].to_numpy()
    pos_all = ev["pos"].to_numpy(np.int64)
    out = {}
    by = {}
    for j, e in enumerate(idx):
        by.setdefault(sym_arr[e], []).append(j)
    for sym, js in by.items():
        js = np.asarray(js)
        e = idx[js]
        c = closes[sym]
        r = BA.symbol_daily(c.to_numpy(np.float64), c.index.astype("datetime64[s]").astype(np.int64), pos_all[e], t1pos[js], side[js] * size[js], cost_side)
        out[sym] = (r["days"], r["gross"] - r["cost"], r["active"])
    return out


def account(nets: dict, sched: dict, lo: int, hi: int):
    """Daily returns of the account for days lo..hi: sum_i w_i(d) * net_i(d)."""
    syms = sorted(nets)
    D = hi - lo + 1
    r = np.zeros(D)
    act = np.zeros(D)
    keys = sorted(sched)
    for s in syms:
        days, net, active = nets[s]
        m = (days >= lo) & (days <= hi)
        di = days[m]
        wv = np.zeros(len(di))
        for k, st in enumerate(keys):
            en = keys[k + 1] if k + 1 < len(keys) else hi + 1
            sel = (di >= st) & (di < en)
            wv[sel] = float(sched[st].get(s, 0.0)) if len(sched[st]) else 0.0
        r[di - lo] += wv * net[m]
        act[di - lo] += (active[m] > 0) * (wv > 0)
    return r, act


def curve_stats(r: np.ndarray):
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate(([1.0], eq)))[1:]
    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    return eq, {"gain_pct": 100.0 * (float(eq[-1]) - 1.0), "max_drawdown_pct": 100.0 * float((eq / peak - 1.0).min()),
                "daily_sharpe_x_sqrt365": (float(r.mean()) / sd * math.sqrt(365)) if sd and np.isfinite(sd) and sd > 0 else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--bars", required=True)
    ap.add_argument("--ports", required=True)
    ap.add_argument("--forest-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.0f}s] {m}", flush=True)  # noqa: E731
    src, ports = Path(a.source).resolve(), Path(a.ports).resolve()
    tj = json.loads(Path(a.forest_json).read_text(encoding="utf-8"))
    summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    venue = summ["meta"]["venue"]
    p = summ["params"]
    step, seed, cost_frac, cutoff = float(p["step"]), int(p["seed"]), float(p["cost_frac"]), p["cutoff"]
    forest = dict(tj["forest"])
    n_cand = int(tj["n_candidates"])
    if a.quick:
        forest["n_estimators"] = 20
    c13 = json.loads((ports / "summary_numbers.json").read_text(encoding="utf-8"))["results"]["ch13"]
    pt, sl, K = float(c13["mesh_choice"]["chosen_pt"]), float(c13["mesh_choice"]["chosen_sl"]), int(c13["step1"]["K"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    before = RB.integrity_snapshot()
    V = PCm.Venue(venue, src, Path(a.bars), 40)
    ev0 = V.events.copy()
    r_long = ev0["ret_side"].to_numpy(float) * ev0["side"].to_numpy(float)
    keep = r_long != 0.0
    ev = ev0[keep].reset_index(drop=True)
    ev["ret_side"], ev["side"] = r_long[keep], 1.0
    ev["bin"] = (ev["ret_side"] > 0).astype(np.int8)
    pd.DataFrame({"src_row": np.flatnonzero(keep)}).to_csv(out / "events_free_index.csv", index=False)
    bf = out
    bars = V.bars(sorted(ev["symbol"].unique()))
    closes = {s: pd.Series(b.close, index=b.idx) for s, b in bars.items()}
    RBA.check_alignment(ev, closes)
    E.FEATURES = list(FEATURES_BOOK)
    # ---- model: combinatorial purged CV (5 paths) and walk-forward, side-free labels, book features only ----
    log(f"{venue}: {len(ev):,} events; OTR pt {pt:g} sl {sl:g} K {K}")
    cpred, cmet, cmeta = ME.run_cpcv(ev, 6, 2, float(p["embargo_pct"]), forest, seed, log)
    wpred, _wm = ME.run_walk_forward(ev, cutoff, forest, seed + 500, log)
    wpred.to_csv(out / "walk_forward_predictions.csv", index=False, float_format="%.12g")
    wpred = pd.read_csv(out / "walk_forward_predictions.csv")
    cpred.to_csv(out / "cpcv_predictions.csv", index=False, float_format="%.12g")
    cpred = pd.read_csv(out / "cpcv_predictions.csv")
    from sklearn.metrics import roc_auc_score
    ybin = ev["bin"].to_numpy()
    auc_cp = [float(roc_auc_score(ybin[g["event"].to_numpy()], g["p"].to_numpy())) for _, g in cpred.groupby("path")]
    acc_cp = [float((((g["p"].to_numpy() > 0.5).astype(int)) == ybin[g["event"].to_numpy()]).mean()) for _, g in cpred.groupby("path")]
    # ---- weights ----
    dc = PCm.trading_days(PCm.daily_closes(bars), .8)
    rets = PCm.returns_within_gap(dc, V.max_gap_days, "simple")
    rets = rets[rets.index >= dc.index[0]]
    cost_side = cost_frac / 2.0
    cutoff_day = int(pd.Timestamp(cutoff).timestamp()) // DAY
    first_event_day = int(ev["t0"].min() // DAY)
    LONG_CACHE = {}
    # ---- one path -> account series ----
    def run_path(pidx, pp, with_refs):
        side_pred = np.where(pp > 0.5, 1.0, -1.0)
        size = MM.bet_size(np.maximum(pp, 1.0 - pp), step)
        plc = np.random.default_rng(seed + 7 + 99).permutation(size)
        t1_model, tr = OX.otr_exits(bars, ev, pidx, side_pred, pt, sl, K)
        ones = np.ones(len(pidx))
        t1_long = LONG_CACHE.setdefault(tuple(pidx[:3]) + (len(pidx),), OX.otr_exits(bars, ev, pidx, ones, pt, sl, K)[0])
        sets = {"book_bet": (side_pred, size, t1_model), "pred_side_full_size": (side_pred, ones, t1_model), "placebo_sizes": (side_pred, plc, t1_model), "always_long": (ones, ones, t1_long)}
        if with_refs:
            rs = np.random.default_rng(seed + 11).choice([-1.0, 1.0], size=len(pidx))
            sets["random_side"] = (rs, ones, OX.otr_exits(bars, ev, pidx, rs, pt, sl, K)[0])
        nets = {}
        for cl, cs in (("net", cost_side), ("before_costs", 0.0)):
            for v, (sd, sz, t1n) in sets.items():
                nets[(cl, v)] = instrument_nets(ev, closes, pidx, sd, sz, t1n, cs)
        return nets, {"truncated_share": float(tr.mean()), "mean_bars_held": float((t1_model - ev["pos"].to_numpy()[pidx]).mean())}

    results = {}
    log("walk-forward path ...")
    w_idx = wpred["event"].to_numpy(np.int64)
    results["wf"] = run_path(w_idx, wpred["p"].to_numpy(float), True)
    for pth, g in cpred.groupby("path"):
        log(f"CPCV path {pth} ...")
        results[f"cpcv{pth}"] = run_path(g["event"].to_numpy(np.int64), g["p"].to_numpy(float), False)
    # ---- accounts ----
    all_net_days = [d for nets, _ in results.values() for dct in nets.values() for (days, _n, _a) in dct.values() for d in (int(days.min()), int(days.max()))]
    last_day, first_day_all = max(all_net_days), min(all_net_days)
    rows, curves = [], []
    series = {}
    scheds = {}
    for scheme in ("hrp", "equal"):
        scheds[scheme] = weight_schedule(rets, min(first_day_all, first_event_day) + REB_WINDOW, last_day, scheme == "equal")
        log(f"weights {scheme}: {len(scheds[scheme])} rebalances")
    spans = {"last_365_days": last_day - 364, "whole_walk_forward_span": cutoff_day}
    for scope, (nets, info_) in results.items():
        for (cl, v), dct in nets.items():
            for scheme in ("hrp", "equal"):
                if scope != "wf" and scheme == "equal":
                    continue
                for sname, lo in spans.items():
                    if scope != "wf" and sname == "whole_walk_forward_span":
                        lo = first_day_all + REB_WINDOW + 5
                    r, act = account(dct, scheds[scheme], lo, last_day)
                    eq, st = curve_stats(r)
                    if v == "book_bet" and cl == "net" and scheme == "hrp":
                        series[(scope, sname)] = r
                    st.update({"scope": scope, "span": sname if scope == "wf" else sname + ("" if sname == "last_365_days" else "_cpcv_full_history"), "scheme": scheme, "costs": cl, "variant": v,
                               "start_day": pd.Timestamp(lo * DAY, unit="s").strftime("%Y-%m-%d"), "end_day": pd.Timestamp(last_day * DAY, unit="s").strftime("%Y-%m-%d"),
                               "mean_instruments_with_position": float(act[act > 0].mean()) if (act > 0).any() else 0.0})
                    rows.append(st)
                    if scope == "wf":
                        curves.append(pd.DataFrame({"day": np.arange(lo, last_day + 1), "ret": r, "equity": eq, "span": sname, "scheme": scheme, "costs": cl, "variant": v}))
    # ---- Chapter 14: PSR / DSR of the book_bet HRP net account (daily series), trials counted as declared ----
    import book_stats as BS
    dsr_out = {}
    for sname in spans:
        r_wf = series[("wf", sname)]
        mo = BS.sharpe_moments(r_wf)
        path_sr = [BS.sharpe_moments(series[(k, "last_365_days" if sname == "last_365_days" else "whole_walk_forward_span")])["sr"] for k in results if k != "wf"] if sname == "last_365_days" else []
        var_paths = float(np.var(path_sr, ddof=1)) if len(path_sr) > 1 else float("nan")
        var_asym = (1.0 - mo["skew"] * mo["sr"] + (mo["kurt"] - 1.0) / 4.0 * mo["sr"] ** 2) / max(mo["n"] - 1.0, 1.0) if np.isfinite(mo["sr"]) else float("nan")
        var_sr = float(np.nanmax([var_paths, var_asym])) if np.isfinite([var_paths, var_asym]).any() else float("nan")
        d = {"n_days": mo["n"], "sr_daily": mo["sr"], "skew": mo["skew"], "kurt": mo["kurt"], "var_across_cpcv_paths": var_paths, "var_asymptotic": var_asym, "var_sr_used": var_sr, "psr_vs_0": BS.psr(mo["sr"], mo["n"], mo["skew"], mo["kurt"], 0.0), "by_trials": {}}
        for nt in (n_cand, n_cand + 5):
            star = BS.expected_max_sharpe(nt, var_sr)
            d["by_trials"][str(nt)] = {"sr_star": star, "dsr": BS.psr(mo["sr"], mo["n"], mo["skew"], mo["kurt"], star) if np.isfinite(star) else float("nan")}
        dsr_out[sname] = d
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False, float_format="%.10g")
    pd.concat(curves, ignore_index=True).to_csv(out / "equity_curves_walk_forward.csv", index=False, float_format="%.12g")
    pd.DataFrame([{"rebalance_day": k, "date": pd.Timestamp(k * DAY, unit="s").strftime("%Y-%m-%d"), "n_instruments": len(v), "max_weight": float(v.max()) if len(v) else float("nan"), "min_weight": float(v.min()) if len(v) else float("nan")}
                  for k, v in scheds["hrp"].items()]).to_csv(out / "hrp_rebalances.csv", index=False)
    pd.concat([pd.Series(v, name=pd.Timestamp(k * DAY, unit="s").strftime("%Y-%m-%d")) for k, v in scheds["hrp"].items() if len(v)], axis=1).T.to_csv(out / "hrp_weights.csv", float_format="%.10g")
    cmet.to_csv(out / "split_metrics.csv", index=False)
    after = RB.integrity_snapshot()
    wy = ybin[w_idx]
    meta = {"venue": venue, "source_run": str(src), "bar_dir": str(a.bars), "ports_run": str(ports), "bookfeat_run": str(bf), "seed": seed, "step": step, "cost_frac_round_trip": cost_frac,
            "cost_bp_per_side": cost_side * 1e4, "cutoff": cutoff, "forest": forest, "features": FEATURES_BOOK, "events": int(len(ev)), "otr": {"pt": pt, "sl": sl, "K": K},
            "cpcv": {"n_paths": int(cmeta["n_paths"]), "auc_by_path": auc_cp, "side_accuracy_by_path": acc_cp, "auc_mean": float(np.mean(auc_cp))},
            "walk_forward": {"auc": float(roc_auc_score(wy, wpred["p"].to_numpy())), "side_accuracy": float((((wpred["p"].to_numpy() > 0.5).astype(int)) == wy).mean()), "up_share_test": float(wy.mean()), "n_test": int(len(w_idx))},
            "exit_stats_wf": results["wf"][1], "hrp_rebalances": len(scheds["hrp"]), "reb_window": REB_WINDOW, "last_day": pd.Timestamp(last_day * DAY, unit="s").strftime("%Y-%m-%d"),
            "first_day": pd.Timestamp(first_day_all * DAY, unit="s").strftime("%Y-%m-%d"), "integrity_unchanged": before == after, "tuned_forest_json": str(a.forest_json), "n_trials_candidates": n_cand, "dsr": dsr_out, "quick": bool(a.quick), "seconds": round(time.time() - t_start),
            "script_md5": {n: RB.md5_file(HERE / n) for n in ("book_full_tuned.py", "book_otr_exit.py", "book_port_d.py", "bet_averaging.py", "meta_model.py", "meta_eval.py", "meta_events.py")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    L = [f"# The book as one strategy with the Ch.9 tuned forest, {venue}", "",
         f"Walk-forward AUC {meta['walk_forward']['auc']:.4f}, side accuracy {meta['walk_forward']['side_accuracy']:.4f} (share of test events that went up {meta['walk_forward']['up_share_test']:.4f}); CPCV AUC by path "
         f"{', '.join(f'{x:.4f}' for x in auc_cp)}. OTR pt {pt:g} / sl {sl:g} sigma, at most {K} bars. HRP re-estimated every quarter from the trailing {REB_WINDOW} trading days. Costs {meta['cost_bp_per_side']:.0f} bp per side.", ""]
    x = S[(S["scope"] == "wf") & (S["scheme"] == "hrp") & (S["costs"] == "net")]
    for sname in ("last_365_days", "whole_walk_forward_span"):
        y = x[x["span"] == sname]
        L += [f"## walk-forward, {sname}: {y['start_day'].iloc[0]} to {y['end_day'].iloc[0]} (HRP allocation, net of costs)", "", "| variant | gain % | max drawdown % | Sharpe (daily x sqrt365) |", "|---|---:|---:|---:|"]
        for _, r_ in y.iterrows():
            L.append(f"| {r_['variant']} | {r_['gain_pct']:.2f} | {r_['max_drawdown_pct']:.2f} | {r_['daily_sharpe_x_sqrt365']:.2f} |")
        L.append("")
    z = S[(S["scope"] != "wf") & (S["variant"] == "book_bet") & (S["costs"] == "net") & (S["span"] == "last_365_days")]
    L += ["## CPCV paths, last 365 days, book_bet, HRP, net of costs", "", "| path | gain % | max drawdown % |", "|---|---:|---:|"]
    for _, r_ in z.iterrows():
        L.append(f"| {r_['scope']} | {r_['gain_pct']:.2f} | {r_['max_drawdown_pct']:.2f} |")
    L += ["", f"Forest used: {forest}. DSR (Ch.14, book_bet HRP net daily series; trials = {n_cand} Ch.9 candidates, or {n_cand + 5} counting the five earlier strategy variants in this study): " + "; ".join(f"{k}: PSR vs 0 {v_['psr_vs_0']:.3f}, DSR " + ", ".join(f"{t} trials {x['dsr']:.3f}" for t, x in v_["by_trials"].items()) for k, v_ in dsr_out.items())]
    (out / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
