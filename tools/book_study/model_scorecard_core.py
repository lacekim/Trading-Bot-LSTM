"""Scoring of trained-model signals on CUSUM events, to the same yardstick as the controls (no project imports, no TensorFlow).

Why this differs from the controls: the models were trained once on a fixed chronological split (train / validation / calibration /
holdout), so a combinatorial cross-validation (which needs a model fitted per split) is not available. The only bars that are out of
sample for a model's weights and threshold are its holdout and post-holdout bars. On those bars each CUSUM event is either one the
model flagged (probability at or above its calibrated threshold) or not, and the question is whether the flagged events do better
than the unflagged events of the same symbols and period.

Rule: the stop/profit rule is NOT chosen here. It is the rule the control study chose (modal rule of the plain 'all' row of the same
direction and horizon), which was fitted without any model output. The whole 20 x 20 mesh is also scored (median Sharpe), and a rule-free
read (hold H bars, and hold 1 bar = the models' own target horizon) is reported next to it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import book_stats as B

SEGMENTS = ("holdout", "post_holdout")


def _group_exit(q_rows, sign, sigma, rule, h, cost_frac):
    """Per-event exit under rule=(pt, sl) for events with forward paths q_rows (n, >=h+1) and a direction sign (+1/-1)."""
    s = (q_rows[:, : h + 1] * sign).astype(np.float32)
    value, step, kind = B.first_exit(s, rule[0], rule[1])
    gross = value.astype(np.float64) * sigma * 1e4
    hold = s[:, h].astype(np.float64) * sigma * 1e4
    hold1 = s[:, 0].astype(np.float64) * sigma * 1e4
    return {"value": value.astype(np.float64), "step": step, "kind": kind, "gross_bp": gross, "net_bp": gross - cost_frac * 1e4,
            "hold_bp": hold, "hold1_bp": hold1, "s": s}


def day_cluster_lift(day, x, sel, n_boot, seed):
    """Mean(x | selected) - mean(x | not selected) with a day-cluster bootstrap. Returns (lift, lo, hi, share_of_draws_le_0)."""
    day = np.asarray(day)
    sel = np.asarray(sel, dtype=bool)
    if sel.sum() == 0 or (~sel).sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    u, inv = np.unique(day, return_inverse=True)
    s_sum = np.bincount(inv[sel], weights=x[sel], minlength=len(u))
    s_n = np.bincount(inv[sel], minlength=len(u)).astype(float)
    c_sum = np.bincount(inv[~sel], weights=x[~sel], minlength=len(u))
    c_n = np.bincount(inv[~sel], minlength=len(u)).astype(float)
    lift = s_sum.sum() / s_n.sum() - c_sum.sum() / c_n.sum()
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b0 in range(0, n_boot, 500):
        m = min(500, n_boot - b0)
        idx = rng.integers(0, len(u), size=(m, len(u)))
        sn, cn = s_n[idx].sum(axis=1), c_n[idx].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[b0:b0 + m] = s_sum[idx].sum(axis=1) / sn - c_sum[idx].sum(axis=1) / cn
    out = out[np.isfinite(out)]
    if not len(out):
        return float(lift), float("nan"), float("nan"), float("nan")
    return float(lift), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())


def score(ev: dict, table: pd.DataFrame, rules: dict, cost_frac: float, windows: dict, seed: int, horizons=(8, 24), n_boot: int = 2000):
    """table columns: eidx (row in ev), direction ('long'/'short'), p, thr, segment ('holdout'/'post_holdout'), promoted (bool).
    rules[(direction, h)] = (pt, sl). Returns (rows list, event-level DataFrame)."""
    sigma_all = ev["sigma"].astype(np.float64)
    day_all = (ev["t0"] // 86400).astype(np.int64)
    rows, event_parts = [], []
    for d in ("long", "short"):
        sign = 1.0 if d == "long" else -1.0
        for seg in SEGMENTS:
            t = table[(table["direction"] == d) & (table["segment"] == seg)]
            if t.empty:
                continue
            e = t["eidx"].to_numpy()
            selected = (t["p"].to_numpy() >= t["thr"].to_numpy())
            for h in horizons:
                rule = rules[(d, h)]
                x = _group_exit(ev["q"][e], sign, sigma_all[e], rule, h, cost_frac)
                day = day_all[e]
                cost_sigma = cost_frac / sigma_all[e]
                row = {"direction": d, "horizon": h, "segment": seg, "rule": [float(rule[0]), float(rule[1])],
                       "n_population": int(len(e)), "n_selected": int(selected.sum()),
                       "n_symbols_selected": int(pd.Series(ev["sym"][e][selected]).nunique()),
                       "days_with_selected": int(len(np.unique(day[selected])))}
                for name, mask in (("selected", selected), ("complement", ~selected), ("population", np.ones(len(e), bool))):
                    if mask.sum():
                        row[f"{name}_gross_bp"] = float(x["gross_bp"][mask].mean())
                        row[f"{name}_net_bp"] = float(x["net_bp"][mask].mean())
                        row[f"{name}_hold_bp"] = float(x["hold_bp"][mask].mean())
                        row[f"{name}_hold1_bp"] = float(x["hold1_bp"][mask].mean())
                        mt = B.mesh_exit_stats(x["s"][mask], cost=cost_sigma[mask])
                        row[f"{name}_mesh_median_sharpe_net"] = float(np.nanmedian(mt["sharpe_net"]))
                        row[f"{name}_mesh_share_positive_net"] = float(np.nanmean(mt["sharpe_net"] > 0))
                for key, arr in (("gross", x["gross_bp"]), ("hold", x["hold_bp"]), ("hold1", x["hold1_bp"])):
                    lift, lo, hi, p_le0 = day_cluster_lift(day, arr, selected, n_boot, seed + h)
                    row[f"lift_{key}_bp"], row[f"lift_{key}_lo"], row[f"lift_{key}_hi"], row[f"lift_{key}_p_le0"] = lift, lo, hi, p_le0
                # daily net series of the selected events (one number per calendar day with a signal)
                if selected.sum():
                    dd = pd.DataFrame({"day": day[selected], "net": x["net_bp"][selected]}).groupby("day")["net"].mean()
                    mom = B.sharpe_moments(dd.to_numpy())
                    t_eff = mom["n"] / max(1.0, (h + 1) / 24.0)
                    row.update({"daily_sr_net": mom["sr"], "daily_skew": mom["skew"], "daily_kurt": mom["kurt"], "t_eff": t_eff,
                                "psr_net": B.psr(mom["sr"], t_eff, mom["skew"], mom["kurt"])})
                # windows
                wrows = {}
                for wname, (lo_t, hi_t) in windows.items():
                    lo_d = int(pd.Timestamp(lo_t).timestamp() // 86400)
                    hi_d = int(pd.Timestamp(hi_t).timestamp() // 86400)
                    inw = (day >= lo_d) & (day < hi_d)
                    wrows[wname] = {
                        "n_population": int(inw.sum()), "n_selected": int((inw & selected).sum()),
                        "selected_net_bp": float(x["net_bp"][inw & selected].mean()) if (inw & selected).any() else None,
                        "population_net_bp": float(x["net_bp"][inw].mean()) if inw.any() else None,
                        "complement_net_bp": float(x["net_bp"][inw & ~selected].mean()) if (inw & ~selected).any() else None}
                row["windows"] = wrows
                rows.append(row)
                event_parts.append(pd.DataFrame({
                    "direction": d, "horizon": h, "segment": seg, "eidx": e, "symbol": [ev["symbols"][i] for i in ev["sym"][e]],
                    "t0_epoch": ev["t0"][e], "pos": ev["pos"][e], "sigma": sigma_all[e], "p": t["p"].to_numpy(), "thr": t["thr"].to_numpy(),
                    "selected": selected, "pt": rule[0], "sl": rule[1], "exit_value": x["value"], "exit_step": x["step"],
                    "exit_kind": x["kind"], "gross_bp": x["gross_bp"], "net_bp": x["net_bp"], "hold_bp": x["hold_bp"], "hold1_bp": x["hold1_bp"]}))
    # deflation across the rows examined (the book's variance across trials)
    srs = np.array([r.get("daily_sr_net", np.nan) for r in rows], dtype=float)
    fin = srs[np.isfinite(srs)]
    n_trials = int(len(fin))
    var_sr = float(fin.var(ddof=1)) if n_trials > 1 else float("nan")
    for r in rows:
        r["dsr_net"] = B.dsr(r["daily_sr_net"], r["t_eff"], r["daily_skew"], r["daily_kurt"], n_trials, var_sr) if "daily_sr_net" in r else None
    trials = {"n": n_trials, "var_sr_net": var_sr, "expected_max_sr": B.expected_max_sharpe(n_trials, var_sr)}
    events = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    return rows, trials, events


def _f(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.{nd}f}"


def render(rows, trials, meta, notes) -> str:
    L = ["# Trained models scored to the book's yardstick (read-only diagnostic)", ""]
    L.append("Every number comes from `summary_numbers.json` / `scored_events.csv` in this folder. Flagged = the model's probability at the bar "
             "was at or above its calibrated threshold. Lift = flagged minus not-flagged events of the same symbols and period, before costs "
             "(cost is the same for both), with a 95% interval from resampling whole days.")
    L.append("")
    for k, v in meta.items():
        L.append(f"- {k}: {v}")
    L.append(f"- Rows examined (trials) for the deflated Sharpe: {trials['n']}; expected maximum Sharpe of that many unskilled rows = {_f(trials['expected_max_sr'], 3)} per day.")
    L.append("")
    for seg in SEGMENTS:
        L.append(f"## Segment: {seg}" + (" (after each symbol's promotion holdout: clean)" if seg == "post_holdout" else " (seen by the promotion gate, not used to fit weights or threshold)"))
        L.append("")
        L.append("| model | H | rule (pt, sl) | events | flagged (symbols) | flagged gross / net bp | not flagged gross bp | lift gross bp [95%] | lift hold-H bp [95%] | lift hold-1 bp [95%] | mesh median net SR: flagged vs not | PSR | DSR |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["segment"] != seg:
                continue
            L.append("| {d} | {h} | ({pt:.1f}, {sl:.1f}) | {n} | {ns} ({nsym}) | {sg} / {sn} | {cg} | {lg} [{lgl}, {lgh}] | {lh} [{lhl}, {lhh}] | {l1} [{l1l}, {l1h}] | {ms} vs {mc} | {psr} | {dsr} |".format(
                d=r["direction"], h=r["horizon"], pt=r["rule"][0], sl=r["rule"][1], n=r["n_population"], ns=r["n_selected"], nsym=r["n_symbols_selected"],
                sg=_f(r.get("selected_gross_bp")), sn=_f(r.get("selected_net_bp")), cg=_f(r.get("complement_gross_bp")),
                lg=_f(r["lift_gross_bp"]), lgl=_f(r["lift_gross_lo"]), lgh=_f(r["lift_gross_hi"]),
                lh=_f(r["lift_hold_bp"]), lhl=_f(r["lift_hold_lo"]), lhh=_f(r["lift_hold_hi"]),
                l1=_f(r["lift_hold1_bp"]), l1l=_f(r["lift_hold1_lo"]), l1h=_f(r["lift_hold1_hi"]),
                ms=_f(r.get("selected_mesh_median_sharpe_net"), 3), mc=_f(r.get("complement_mesh_median_sharpe_net"), 3),
                psr=_f(r.get("psr_net"), 2), dsr=_f(r.get("dsr_net"), 2)))
        L.append("")
        L.append("Window lines (events entered inside the window; flagged events, then every event of the model's symbols for comparison):")
        L.append("")
        wn = list(rows[0]["windows"]) if rows else []
        L.append("| model | H | " + " | ".join(f"{w}: flagged n, flagged net bp, all-events n, all-events net bp" for w in wn) + " |")
        L.append("|---|---|" + "---|" * len(wn))
        for r in rows:
            if r["segment"] != seg:
                continue
            cells = []
            for w in wn:
                x = r["windows"][w]
                cells.append(f"{x['n_selected']}, {_f(x['selected_net_bp'])}, {x['n_population']}, {_f(x['population_net_bp'])}")
            L.append(f"| {r['direction']} | {r['horizon']} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("## Notes and limits")
    L.append("")
    for i, n in enumerate(notes, 1):
        L.append(f"{i}. {n}")
    return "\n".join(L) + "\n"
