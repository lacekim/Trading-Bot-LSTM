"""Chapter 8 (feature importance) and Chapter 9 (hyperparameter tuning with purged k-fold CV) applied to the side-free book model, one venue per run.  New file; nothing audited is edited.
Data: only events whose label ENDS before the walk-forward cutoff (2024-01-01), so nothing the later walk-forward tests on is seen here.
Ch.8: MDI (Snippet 8.2, book_port_b.featImpMDI as ported, trees with max_features=1 so features cannot mask each other), MDA (Snippet 8.3, neg log-loss, purged 5-fold with embargo) and SFI (Snippet 8.4), all with the book's forest of
      entropy trees (sequential bootstrap).  Reported, in the book's sense of 'research through importance, not through backtesting'; no feature is dropped (the extract has no decision rule for dropping).
Ch.9: randomised search (Snippet 9.3: scoring neg_log_loss for symmetric labels, purged k-fold with embargo) over the forest's parameters, min_weight_fraction_leaf ~ logUniform (Snippet 9.4's distribution), max_features in a
      list; the book's untuned setting is candidate 0.  Winner = best mean OOS neg log-loss.  The winner is written to tuned_forest.json for book_full_tuned.py.
Usage: python book_ch8_ch9.py --source <meta run> --bars <dollar bars> --out <folder> [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import book_port_b as PB  # noqa: E402
import meta_eval as ME  # noqa: E402
import meta_events as E  # noqa: E402
import meta_model as MM  # noqa: E402
import ports_common as PCm  # noqa: E402
import run_book_study as RB  # noqa: E402
from sklearn.model_selection import ParameterSampler  # noqa: E402

FEATURES_BOOK = ["ffd", "log_trgt", "sadf", "ent_plugin", "ent_konto", "cs_spread"]


class _Shim:
    def __init__(self, trees):
        self.estimators_ = trees


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--bars", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_iter", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.0f}s] {m}", flush=True)  # noqa: E731
    src = Path(a.source).resolve()
    summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    venue, p = summ["meta"]["venue"], summ["params"]
    seed, cutoff, emb = int(p["seed"]), p["cutoff"], float(p["embargo_pct"])
    base_forest = dict(p["forest"])
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
    E.FEATURES = list(FEATURES_BOOK)
    X, y, w, sym, starts, ends, t0, t1, u = ME.prepare(ev)
    cut = float(pd.Timestamp(cutoff).timestamp())
    tr = np.flatnonzero(t1 < cut)
    X, y, w, sym, starts, ends, t0, t1, u = X[tr], y[tr], w[tr], sym[tr], starts[tr], ends[tr], t0[tr], t1[tr], u[tr]
    avg_u_fn = lambda ix: float(u[ix].mean())  # noqa: E731
    n_tree_imp = 20 if a.quick else 50
    log(f"{venue}: {len(tr):,} events with labels ending before {cutoff}")
    # ---------------- Chapter 8 ----------------
    mk1 = lambda: MM.BaggedForest(seed=seed + 900, **{**base_forest, "n_estimators": n_tree_imp, "max_features": 1})  # noqa: E731
    f = mk1().fit(X, y, w, sym, starts, ends, float(u.mean()))
    mdi = PB.featImpMDI(_Shim(f.trees), FEATURES_BOOK)
    mdi.to_csv(out / "mdi.csv")
    log("MDI done")
    mda, mda_base = MM.mda_importance(X, y, w, sym, starts, ends, avg_u_fn, t0, t1, FEATURES_BOOK, 5, emb, mk1, seed)
    mda.to_csv(out / "mda.csv")
    log("MDA done")
    mk2 = lambda: MM.BaggedForest(seed=seed + 901, **{**base_forest, "n_estimators": max(10, n_tree_imp // 2), "max_features": 1})  # noqa: E731
    sfi = MM.sfi_importance(X, y, w, sym, starts, ends, avg_u_fn, t0, t1, FEATURES_BOOK, 5, emb, mk2)
    sfi.to_csv(out / "sfi.csv")
    log("SFI done")
    # ---------------- Chapter 9 ----------------
    n_tree_search = 15 if a.quick else 30
    space = {"min_weight_fraction_leaf": PB.logUniform(0.005, 0.2), "max_features": [1, 2, 3, 6]}
    cands = [{"min_weight_fraction_leaf": 0.05, "max_features": "sqrt", "_note": "untuned setting used so far"}]
    cands += [dict(c, _note="random draw") for c in ParameterSampler(space, n_iter=2 if a.quick else a.n_iter, random_state=seed)]
    folds = list(MM.purged_kfold(t0, t1, 3, emb))
    rows = []
    for i, c in enumerate(cands):
        kw = {**base_forest, "n_estimators": n_tree_search, "min_weight_fraction_leaf": float(c["min_weight_fraction_leaf"]), "max_features": c["max_features"]}
        sc = []
        for trn, te in folds:
            ff = MM.BaggedForest(seed=seed + 1000 + i, **kw).fit(X[trn], y[trn], w[trn], sym[trn], starts[trn], ends[trn], avg_u_fn(trn))
            sc.append(-MM.weighted_log_loss(y[te], ff.predict_proba1(X[te]), w[te]))
        rows.append({"cand": i, "min_weight_fraction_leaf": float(c["min_weight_fraction_leaf"]), "max_features": c["max_features"], "mean_neg_log_loss": float(np.mean(sc)), "fold_scores": json.dumps([float(s) for s in sc]), "note": c["_note"]})
        log(f"search {i}: mwfl {rows[-1]['min_weight_fraction_leaf']:.4f} max_features {c['max_features']} -> {rows[-1]['mean_neg_log_loss']:.6f}")
    R = pd.DataFrame(rows)
    R.to_csv(out / "search.csv", index=False)
    best = R.loc[R["mean_neg_log_loss"].idxmax()]
    tuned = {**base_forest, "min_weight_fraction_leaf": float(best["min_weight_fraction_leaf"]), "max_features": (best["max_features"] if best["max_features"] == "sqrt" else int(best["max_features"]))}
    (out / "tuned_forest.json").write_text(json.dumps({"venue": venue, "forest": tuned, "chosen_candidate": int(best["cand"]), "n_candidates": int(len(R)), "baseline_neg_log_loss": float(R.iloc[0]["mean_neg_log_loss"]),
                                                      "best_neg_log_loss": float(best["mean_neg_log_loss"])}, indent=2), encoding="utf-8")
    meta = {"venue": venue, "source_run": str(src), "bar_dir": str(a.bars), "seed": seed, "cutoff": cutoff, "events_used": int(len(tr)), "embargo_pct": emb, "n_trees_importance": n_tree_imp, "n_trees_search": n_tree_search,
            "mda_baseline_neg_log_loss": mda_base, "integrity_unchanged": before == RB.integrity_snapshot(), "quick": bool(a.quick), "seconds": round(time.time() - t_start),
            "script_md5": {n: RB.md5_file(HERE / n) for n in ("book_ch8_ch9.py", "meta_model.py", "book_port_b.py")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    L = [f"# Chapters 8 and 9 on the side-free book model, {venue}", "", f"Events with labels ending before {cutoff}: {len(tr):,}.", "",
         "## Feature importance (Ch.8)", "", "| feature | MDI | MDA | SFI neg log-loss |", "|---|---:|---:|---:|"]
    for ft in FEATURES_BOOK:
        L.append(f"| {ft} | {mdi.loc[ft, 'mean']:.4f} | {mda.loc[ft, 'mean']:.4f} | {sfi.loc[ft, 'mean']:.5f} |")
    L += ["", f"MDA baseline OOS neg log-loss {mda_base:.5f}.", "", "## Hyperparameter search (Ch.9, purged 3-fold)", "", "| cand | min_weight_fraction_leaf | max_features | mean OOS neg log-loss |", "|---:|---:|---|---:|"]
    for _, r_ in R.iterrows():
        L.append(f"| {int(r_['cand'])}{' (untuned)' if r_['cand'] == 0 else ''} | {r_['min_weight_fraction_leaf']:.4f} | {r_['max_features']} | {r_['mean_neg_log_loss']:.6f} |")
    L += ["", f"Chosen: candidate {int(best['cand'])}."]
    (out / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
