#!/usr/bin/env python
"""Ch.9 hyperparameter search for the meta-labeling forest, done on pre-cutoff events only and scored once on the post-cutoff events (read-only diagnostic).

Steps (all declared before any result):
  1. Take events.csv of an existing 14-feature run (--run).  Search set = events whose labels end before the run's walk-forward cutoff (2024-01-01) - exactly the walk-forward
     training set.  Nothing at or after the cutoff is used to choose.
  2. Grid (fixed in GRID below): min_weight_fraction_leaf in {0.005, 0.01, 0.025, 0.05, 0.1, 0.2} x max_features in {sqrt, 0.5} = 12 configs; 100 trees, sequential bootstrap and
     max_draws 10000 stay as in the pipeline.  The pipeline's own setting (0.05, sqrt) is one grid point.
  3. Each config is scored by purged 5-fold CV (1% embargo, meta_model.purged_kfold) on the search set: pooled out-of-fold sample-weighted log loss (the book's neg-log-loss
     scoring); AUC is reported but not used to choose.  Lowest log loss wins (first in grid order on a tie).
  4. The default and the chosen config are each fitted once on the whole search set and scored on the post-cutoff events (meta_eval.run_walk_forward / evaluate, as in the
     pipeline) for each seed.  The default at the run's own seed must reproduce the run's walk_forward_predictions.csv or the tool stops (not in --quick mode).
No new model is shipped and no live or paper setting is touched.  Trial count: one new config reaches the post-cutoff data (look 108); the post-cutoff period was seen in every earlier run.

    python tools/book_study/run_hp_search.py --run runtime/book_seed_pairs/volume_vs_14/ctrl_20261017/<timestamp> --jobs 2
    python tools/book_study/verify_hp_search.py "$(ls -d runtime/book_hp_search/*/ | tail -1)"
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import meta_eval as ME  # noqa: E402
import meta_events as E  # noqa: E402
import meta_model as MM  # noqa: E402
import run_book_study as RB  # noqa: E402

MWFL = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
MAXF = ["sqrt", 0.5]
GRID = [{"config": i, "min_weight_fraction_leaf": a, "max_features": b} for i, (a, b) in enumerate(itertools.product(MWFL, MAXF))]
DEFAULT = {"min_weight_fraction_leaf": 0.05, "max_features": "sqrt"}
SEARCH_SEED = 20261201
SEEDS = [20261017, 20261101, 20261102, 20261103, 20261104]
N_FOLDS, EMBARGO = 5, 0.01
PRIOR_TRIALS = 107
BASE_KW = {"n_estimators": 100, "bootstrap": "sequential", "max_draws": 10000}


def default_id() -> int:
    return next(g["config"] for g in GRID if g["min_weight_fraction_leaf"] == DEFAULT["min_weight_fraction_leaf"] and g["max_features"] == DEFAULT["max_features"])


def forest_kw(cfg: dict, quick: bool) -> dict:
    kw = dict(BASE_KW, min_weight_fraction_leaf=cfg["min_weight_fraction_leaf"], max_features=cfg["max_features"])
    if quick:
        kw.update(n_estimators=10, max_draws=3000)
    return kw


def fit_oof(args):
    """One (config, fold): fit on the purged training part, predict the held-out fold.  Returns (config, fold, global test indices, p)."""
    cfg, fold, tr, te, kw, run_dir = args
    df = pd.read_csv(Path(run_dir) / "events.csv")
    data = ME.prepare(df)
    X = data[0]
    f = ME.fit_forest(tr, data, kw, SEARCH_SEED + fold)
    return cfg, fold, te, f.predict_proba1(X[te])


def metrics_of(y, p, w):
    return {"log_loss_w": MM.weighted_log_loss(y, p, w), "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"), "share_p_gt_half": float((p > 0.5).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="timestamp folder of a 14-feature run (run_meta_venue.py) whose events.csv is used")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="smoke test only (10 trees, 3000 draws, 3 configs); not for reporting")
    ap.add_argument("--quick-configs", type=int, nargs="+", default=None, help="with --quick only: config ids to run instead of the first three (used to test the chosen-differs-from-default path)")
    a = ap.parse_args()
    run = Path(a.run)
    if not run.is_absolute():
        run = ROOT / run
    run = run.resolve()
    summ = json.loads((run / "summary_numbers.json").read_text(encoding="utf-8"))
    p = summ["params"]
    if p["features"] != E.FEATURES:
        raise SystemExit("the source run does not use the 14-feature list of meta_events.FEATURES")
    df = pd.read_csv(run / "events.csv")
    data = ME.prepare(df)
    X, y, w, sym, starts, ends, t0, t1, u = data
    cut = float(pd.Timestamp(p["cutoff"]).timestamp())
    pre = np.flatnonzero(t1 < cut)
    started = datetime.now(timezone.utc)
    out = Path(a.out) if a.out else ROOT / "runtime" / "book_hp_search" / started.strftime("%Y%m%dT%H%M%SZ")
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    log_f = open(out / "run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    if a.quick_configs and not a.quick:
        raise SystemExit("--quick-configs is only for --quick smoke tests")
    grid = ([GRID[i] for i in a.quick_configs] if a.quick_configs else GRID[:3]) if a.quick else GRID
    # keep the default in the quick grid so that the comparison runs
    if a.quick and default_id() not in [g["config"] for g in grid]:
        grid = grid + [GRID[default_id()]]
    folds = list(MM.purged_kfold(t0[pre], t1[pre], N_FOLDS, EMBARGO))
    fold_rows = [{"fold": k, "n_train": int(len(tr)), "n_test": int(len(te))} for k, (tr, te) in enumerate(folds)]
    log(f"search set: {len(pre):,} events with labels ending before {p['cutoff']}; {len(grid)} configs x {N_FOLDS} folds")
    jobs = [(g["config"], k, pre[tr], pre[te], forest_kw(g, a.quick), str(run)) for g in grid for k, (tr, te) in enumerate(folds)]
    t_start = time.time()
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            res = list(ex.map(fit_oof, jobs))
    else:
        res = []
        for j in jobs:
            res.append(fit_oof(j))
            log(f"  config {j[0]} fold {j[1]} done ({time.time() - t_start:.0f}s)")
    oof_rows, scores = [], []
    for g in grid:
        mine = [r for r in res if r[0] == g["config"]]
        idx = np.concatenate([r[2] for r in mine])
        pp = np.concatenate([r[3] for r in mine])
        for r in mine:
            for e_, q_ in zip(r[2], r[3]):
                oof_rows.append((g["config"], r[1], int(e_), float(q_)))
        m = metrics_of(y[idx], pp, w[idx])
        per_fold = [MM.weighted_log_loss(y[r[2]], r[3], w[r[2]]) for r in mine]
        scores.append({**g, "n_oof": int(len(idx)), **m, "fold_log_loss_mean": float(np.mean(per_fold)), "fold_log_loss_sd": float(np.std(per_fold, ddof=1))})
    sc = pd.DataFrame(scores)
    sc["rank_log_loss"] = sc["log_loss_w"].rank(method="first").astype(int)
    sc["rank_auc"] = sc["auc"].rank(ascending=False, method="first").astype(int)
    chosen = int(sc.loc[sc["log_loss_w"].idxmin(), "config"])
    dflt = default_id()
    log(f"chosen config {chosen} ({GRID[chosen]}); default is config {dflt}")
    pd.DataFrame(GRID).to_csv(out / "grid.csv", index=False)
    pd.DataFrame(oof_rows, columns=["config", "fold", "event", "p"]).to_csv(out / "oof_predictions.csv", index=False, float_format="%.12g")
    sc.to_csv(out / "config_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "folds.csv", index=False)

    # ---- post-cutoff scoring of default and chosen ----
    windows = {k: (int(v[0]), int(v[1])) for k, v in summ["meta"]["window_epoch"].items()}
    cost_frac, step, h = float(p["cost_frac"]), float(p["step"]), int(p["h"])
    arms = [("default", dflt)] + ([("chosen", chosen)] if chosen != dflt else [])
    wf_pred_rows, wf_rows = [], []
    gate = {"checked": False}
    for name, cid in arms:
        kw = forest_kw(GRID[cid], a.quick)
        for s in a.seeds:
            wpred, wmet = ME.run_walk_forward(df, p["cutoff"], kw, s + 500, log)
            wev = ME.evaluate(df, wpred, cost_frac, step, h, s + 7, windows, PRIOR_TRIALS)
            if name == "default" and s == int(p["seed"]) and not a.quick:
                src = pd.read_csv(run / "walk_forward_predictions.csv")
                ok = len(src) == len(wpred) and (src["event"].to_numpy() == wpred["event"].to_numpy()).all() and float(np.abs(src["p"].to_numpy() - wpred["p"].to_numpy()).max()) <= 1e-6
                gate = {"checked": True, "max_abs_p_diff": float(np.abs(src["p"].to_numpy() - wpred["p"].to_numpy()).max()) if len(src) == len(wpred) else None, "passed": bool(ok)}
                if not ok:
                    raise SystemExit(f"the default config at seed {s} does not reproduce the source run's walk_forward_predictions.csv ({gate}); stopping")
            for e_, q_ in zip(wpred["event"].to_numpy(), wpred["p"].to_numpy()):
                wf_pred_rows.append((name, cid, s, int(e_), float(q_)))
            ws = wev["summary"]
            win = wev["windows"]
            row = {"arm": name, "config": cid, "seed": s, "wf_auc": wmet["auc"], "wf_log_loss_w": wmet["log_loss_w"], "n_train": wmet["n_train"], "n_test": wmet["n_test"],
                   "meta_sr_annual": ws["meta"]["mean_sr_annual"], "meta_bp": ws["meta"]["mean_bp_per_candidate"], "primary_sr_annual": ws["primary"]["mean_sr_annual"],
                   "placebo_sr_annual": ws["placebo"]["mean_sr_annual"], "meta_minus_placebo_sr_daily": ws["meta_minus_placebo"]["mean_sr_daily_diff"],
                   "taken_share": ws["meta"]["taken_share"], "mean_size": ws["meta"]["mean_size"]}
            for wn, tag in (("june_2026_ai_correction", "june"), ("sell_off_post_holdout", "selloff")):
                m_, p_ = win.get(f"meta|{wn}", {}).get("mean_bp_per_candidate", float("nan")), win.get(f"placebo|{wn}", {}).get("mean_bp_per_candidate", float("nan"))
                row[f"{tag}_meta_bp"], row[f"{tag}_meta_minus_placebo_bp"] = m_, m_ - p_
            wf_rows.append(row)
    pd.DataFrame(wf_pred_rows, columns=["arm", "config", "seed", "event", "p"]).to_csv(out / "wf_predictions.csv", index=False, float_format="%.12g")
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(out / "wf_metrics.csv", index=False)
    extra = {"source_run": str(run), "cutoff": p["cutoff"], "n_search_events": int(len(pre)), "n_events": int(len(df)), "embargo_pct": EMBARGO, "n_folds": N_FOLDS, "search_seed": SEARCH_SEED,
             "seeds": a.seeds, "grid": GRID if not a.quick else grid, "default_config": dflt, "chosen_config": chosen, "criterion": "pooled out-of-fold sample-weighted log loss (lowest)",
             "forest": BASE_KW, "quick": bool(a.quick), "prior_trials": PRIOR_TRIALS, "step": step, "cost_frac": cost_frac, "h": h, "gate": gate,
             "windows": {k: list(v) for k, v in windows.items()}, "script_md5": {n: RB.md5_file(HERE / n) for n in ("run_hp_search.py",)}}
    (out / "hp_extra.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    (out / "hp_summary.md").write_text(render(sc, wf, extra, chosen, dflt), encoding="utf-8")
    print((out / "hp_summary.md").read_text(encoding="utf-8"))
    log(f"written to {out}")
    return 0


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def render(sc: pd.DataFrame, wf: pd.DataFrame, extra: dict, chosen: int, dflt: int) -> str:
    L = ["# Ch.9 hyperparameter search (read-only diagnostic)", "",
         f"Source run: {extra['source_run']}. Search set: {extra['n_search_events']:,} events with labels ending before {extra['cutoff']} (of {extra['n_events']:,}); purged {extra['n_folds']}-fold, "
         f"{extra['embargo_pct'] * 100:.0f}% embargo; criterion: {extra['criterion']}. Gate (default config reproduces the source run's walk-forward predictions): {extra['gate']}.", "",
         "| config | min_weight_fraction_leaf | max_features | OOF weighted log loss | rank | OOF AUC | rank | share p > 0.5 |", "|---|---|---|---|---|---|---|---|"]
    for r in sc.itertuples():
        tag = " (default)" if r.config == dflt else ""
        tag += " (chosen)" if r.config == chosen else ""
        L.append(f"| {r.config}{tag} | {r.min_weight_fraction_leaf} | {r.max_features} | {fmt(r.log_loss_w)} | {r.rank_log_loss} | {fmt(r.auc)} | {r.rank_auc} | {fmt(r.share_p_gt_half, 3)} |")
    L += ["", f"Chosen config {chosen}; default config {dflt}.", ""]
    if chosen == dflt:
        L += ["The chosen config is the default: there is nothing to compare after the cutoff.", ""]
    cols = ["wf_auc", "wf_log_loss_w", "meta_sr_annual", "meta_bp", "meta_minus_placebo_sr_daily", "taken_share", "june_meta_bp", "june_meta_minus_placebo_bp", "selloff_meta_bp", "selloff_meta_minus_placebo_bp"]
    L += ["## Post-cutoff scoring (walk-forward fit on all search events; mean over seeds, sd in brackets)", "", "| metric | default | chosen | mean difference | sd of differences | seeds with difference > 0 |", "|---|---|---|---|---|---|"]
    d, c = wf[wf["arm"] == "default"].sort_values("seed"), wf[wf["arm"] == "chosen"].sort_values("seed")
    for k in cols:
        dv = d[k].to_numpy(dtype=float)
        if len(c):
            cv = c[k].to_numpy(dtype=float)
            diff = cv - dv
            diff_f = diff[np.isfinite(diff)]
            L.append(f"| {k} | {fmt(np.nanmean(dv))} ({fmt(np.nanstd(dv, ddof=1))}) | {fmt(np.nanmean(cv))} ({fmt(np.nanstd(cv, ddof=1))}) | {fmt(diff_f.mean() if len(diff_f) else float('nan'))} | "
                     f"{fmt(diff_f.std(ddof=1) if len(diff_f) > 1 else float('nan'))} | {int((diff_f > 0).sum())} of {len(diff_f)} |")
        else:
            L.append(f"| {k} | {fmt(np.nanmean(dv))} ({fmt(np.nanstd(dv, ddof=1))}) | - | - | - | - |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
