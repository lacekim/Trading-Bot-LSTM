"""Runner sections for Chapters 7-10 (Snippets 7.1-7.4, 8.2-8.10, 9.1-9.4, 10.1-10.4)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, weightedtau
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import bet_averaging as BA
import book_mp as MP
import book_port_b as PB
import book_port_c as PC
import meta_model as MM
import run_bet_averaging as RBA
from ports_common import num, sorted_events, systematic_sample


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 7
# ---------------------------------------------------------------------------------------------------------------------
def ch7(V, cfg, log):
    """PurgedKFold (7.3) on ALL of the venue's events, pooled over symbols and ordered by start time; checks by brute force that no training label overlaps a test label or the embargo."""
    se = sorted_events(V.events)
    t1 = pd.Series(se["t1_dt"].to_numpy(), index=se.index)
    X = pd.DataFrame({"i": np.arange(len(se))}, index=se.index)
    out = {"events": int(len(se)), "unique_index": bool(se.index.is_unique), "folds": []}
    ns, emb = cfg["cv"], .01
    cv = PB.PurgedKFold(n_splits=ns, t1=t1, pctEmbargo=emb)
    t0v = se.index.to_numpy()
    t1v = t1.to_numpy()
    bad_total = 0
    for k, (tr, te) in enumerate(cv.split(X)):
        te_lo, te_hi = te[0], te[-1]
        tmin, tmax1 = t0v[te_lo], t1v[te].max()
        # brute-force: a training label must not overlap [tmin, tmax1]
        overlap = ((t0v[tr] <= tmax1) & (t1v[tr] >= tmin)).sum()
        mbrg = int(len(se) * emb)
        n_left = int((tr < te_lo).sum())
        n_right = int((tr > te_hi).sum())
        # what a plain (non-purged) k-fold would have trained on
        plain = len(se) - len(te)
        removed = plain - len(tr)
        left_purge_candidates = int(((np.arange(len(se)) < te_lo) & (t1v >= tmin)).sum())
        out["folds"].append({"fold": k, "test": int(len(te)), "train": int(len(tr)), "removed_vs_plain_kfold": int(removed), "overlapping_train_labels": int(overlap),
                             "train_left": n_left, "train_right": n_right, "left_labels_purged": left_purge_candidates, "embargo_events": mbrg})
        bad_total += int(overlap)
    out["total_overlapping_train_labels"] = bad_total
    # 7.2 on one fold: embargo times
    mbrg = PB.getEmbargoTimes(se.index, emb)
    te_idx = np.array_split(np.arange(len(se)), ns)[1]
    dt0, dt1 = se.index[te_idx[0]], se.index[te_idx[-1]]
    trainTimes, testTimes = PB.embargo_usage(mbrg, t1, dt0, dt1)
    brute = ((trainTimes.index <= mbrg[dt1]) & (trainTimes >= dt0)).sum()
    out["snippet_7_2_fold1"] = {"train": int(len(trainTimes)), "test": int(len(testTimes)), "train_labels_overlapping_test_plus_embargo": int(brute)}
    md = [f"Snippets 7.1-7.3 on all {len(se):,} events of the venue, pooled over symbols and ordered by start time ({ns} contiguous folds, 1% embargo = {int(len(se) * emb):,} events). "
          "Events of different symbols that start in the same second are separated by 1 ns steps in the order (start, symbol) so that the index is unique; the end times are untouched. "
          f"Brute-force check: training labels that overlap the test labels' span: {bad_total}.", "",
          "| fold | test events | train events | removed vs a plain k-fold | train events before / after the test block | labels from before the test block purged because they run into it |", "|---|---|---|---|---|---|"]
    for f in out["folds"]:
        md.append(f"| {f['fold']} | {f['test']:,} | {f['train']:,} | {f['removed_vs_plain_kfold']:,} | {f['train_left']:,} / {f['train_right']:,} | {f['left_labels_purged']:,} |")
    md.append("")
    md.append(f"Snippet 7.2's embargo-time version on fold 1: train {out['snippet_7_2_fold1']['train']:,}, test {out['snippet_7_2_fold1']['test']:,}, training labels overlapping the test span plus embargo: {out['snippet_7_2_fold1']['train_labels_overlapping_test_plus_embargo']}.")
    return out, md


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 8
# ---------------------------------------------------------------------------------------------------------------------
def _sample_frame(V, cfg):
    se = sorted_events(V.events)
    samp = systematic_sample(se, cfg["model_sample"])
    X = samp[V.features].astype(float)
    cont = pd.DataFrame({"bin": samp["bin"].astype(int), "w": samp["w"].astype(float) / samp["w"].sum(), "t1": pd.to_datetime(samp["t1_dt"].to_numpy())}, index=samp.index)
    return samp, X, cont, len(se)


def ch8(V, cfg, log):
    out, md = {}, []
    samp, X, cont, n_all = _sample_frame(V, cfg)
    res = {}
    for method in ("MDI", "MDA", "SFI"):
        np.random.seed(cfg["seed"])
        imp, oob, oos = PB.featImportance(X, cont, n_estimators=cfg["n_estimators_fi"], cv=cfg["cv"], max_samples=1., numThreads=cfg["threads"], pctEmbargo=.01, scoring="accuracy",
                                          method=method, minWLeaf=float(V.summary["params"].get("min_weight_fraction_leaf", 0.)))
        res[method] = (imp, oob, oos)
        log(f"  ch8 {method}: oob {oob:.4f} oos {oos:.4f}")
    base = float(max(cont["bin"].mean(), 1 - cont["bin"].mean()))
    out["sample"] = {"events": int(len(X)), "of": int(n_all), "label_share_1": float(cont["bin"].mean()), "majority_share": base, "n_estimators": cfg["n_estimators_fi"], "cv": cfg["cv"]}
    tabs = {}
    for m, (imp, oob, oos) in res.items():
        out[m] = {"oob_accuracy": float(oob), "oos_accuracy": float(oos), "importance_mean": {k: float(v) for k, v in imp["mean"].items()}, "importance_std": {k: float(v) for k, v in imp["std"].items()}}
        tabs[m] = imp
    # compare with the project's CPCV-based MDA / SFI (same features) through the weighted Kendall tau of Snippet 8.6
    comp = {}
    for m, f in (("MDA", "mda.csv"), ("SFI", "sfi.csv")):
        proj = pd.read_csv(V.src / f, index_col=0).iloc[:, 0:1] if f == "mda.csv" else pd.read_csv(V.src / f).set_index("feature")[["mean"]]
        if f == "mda.csv":
            proj = pd.read_csv(V.src / f, index_col=0)[["mean"]]
        mine = tabs[m]["mean"].reindex(proj.index)
        tau = weightedtau(mine.to_numpy(float), proj["mean"].to_numpy(float))[0]
        comp[m] = {"weighted_tau_book_port_vs_project": num(tau), "project_top3": list(proj["mean"].sort_values(ascending=False).index[:3]), "port_top3": list(mine.sort_values(ascending=False).index[:3])}
    out["compare_with_project"] = comp
    # orthogonal features (8.5) and the weighted tau of 8.6 on the venue's features
    P = PB.orthoFeats(X, varThres=.95)
    Pdf = pd.DataFrame(P, index=X.index, columns=[f"PC_{i + 1}" for i in range(P.shape[1])])
    np.random.seed(cfg["seed"])
    impP, oobP, oosP = PB.featImportance(Pdf, cont, n_estimators=cfg["n_estimators_fi"], cv=cfg["cv"], max_samples=1., numThreads=cfg["threads"], pctEmbargo=.01, scoring="accuracy", method="MDI",
                                         minWLeaf=float(V.summary["params"].get("min_weight_fraction_leaf", 0.)))
    pcRank = np.arange(1, Pdf.shape[1] + 1)
    tauP = weightedtau(impP["mean"].to_numpy(float), pcRank ** -1.)[0]
    out["orthogonal"] = {"components_for_95pct": int(P.shape[1]), "of_features": int(X.shape[1]), "oob_accuracy": float(oobP), "oos_accuracy": float(oosP), "weighted_tau_importance_vs_inverse_pc_rank": num(tauP),
                         "mdi_by_pc": {k: float(v) for k, v in impP["mean"].items()}}
    # synthetic check (8.7-8.9) at a reduced size
    np.random.seed(cfg["seed"])
    tx, tc = PB.getTestData(n_features=cfg["synth_features"], n_informative=cfg["synth_informative"], n_redundant=cfg["synth_redundant"], n_samples=cfg["synth_samples"])
    syn = {}
    for m in ("MDI", "MDA", "SFI"):
        np.random.seed(cfg["seed"])
        imp, oob, oos = PB.featImportance(tx, tc, n_estimators=cfg["n_estimators_fi"], cv=cfg["cv"], max_samples=1., numThreads=cfg["threads"], pctEmbargo=.01, scoring="accuracy", method=m)
        names = list(imp.index)
        info = imp.loc[[c for c in names if c[0] in "IR"], "mean"].mean()
        noise = imp.loc[[c for c in names if c[0] == "N"], "mean"].mean() if any(c[0] == "N" for c in names) else float("nan")
        syn[m] = {"oob": float(oob), "oos": float(oos), "mean_importance_informative_and_redundant": float(info), "mean_importance_noise": num(noise)}
    out["synthetic"] = {"settings": {"features": cfg["synth_features"], "informative": cfg["synth_informative"], "redundant": cfg["synth_redundant"], "samples": cfg["synth_samples"]}, "methods": syn}
    md += [f"Snippets 8.2-8.4, 8.8: feature importance (MDI, MDA with purged {cfg['cv']}-fold CV, SFI) of the project's {len(V.features)} features on a systematic sample of {len(X):,} of the venue's {n_all:,} events, "
           f"{cfg['n_estimators_fi']} trees (the printed example uses 1,000), accuracy scoring, weights = attribution weights. Majority-class share {base:.4f}.", "",
           "| method | OOB accuracy | purged-CV accuracy | three most important | weighted tau vs the project's CPCV-based ranking |", "|---|---|---|---|---|"]
    for m in ("MDI", "MDA", "SFI"):
        top3 = list(tabs[m]["mean"].sort_values(ascending=False).index[:3])
        t = comp.get(m, {}).get("weighted_tau_book_port_vs_project")
        md.append(f"| {m} | {res[m][1]:.4f} | {res[m][2]:.4f} | {', '.join(top3)} | {('n/a (no project MDI)' if t is None and m == 'MDI' else (f'{t:.3f}' if t is not None else 'n/a'))} |")
    md += ["", f"Snippets 8.5-8.6: {out['orthogonal']['components_for_95pct']} orthogonal components carry 95% of the variance of the {X.shape[1]} standardised features. "
           f"MDI importance of the components against the inverse of the eigenvalue rank: weighted Kendall tau {tauP:.3f} (1 would mean the forest finds the large-variance components the most useful). Purged-CV accuracy on the components {oosP:.4f}.", "",
           f"Snippets 8.7-8.9 on the book's synthetic data ({cfg['synth_features']} features: {cfg['synth_informative']} informative, {cfg['synth_redundant']} redundant, the rest noise; {cfg['synth_samples']} samples; printed generator, rows grouped by cluster):", "",
           "| method | OOB accuracy | purged-CV accuracy | mean importance, informative + redundant | mean importance, noise |", "|---|---|---|---|---|"]
    for m, d in syn.items():
        md.append(f"| {m} | {d['oob']:.4f} | {d['oos']:.4f} | {d['mean_importance_informative_and_redundant']:.4f} | {('n/a' if d['mean_importance_noise'] is None else format(d['mean_importance_noise'], '.4f'))} |")
    md += ["", "(The generator the book prints returns its rows grouped by cluster, so contiguous cross-validation folds hold out whole clusters; that is why the purged-CV accuracy on this data can sit below the out-of-bag accuracy. This is the printed data, not changed.)"]
    md += ["", "The three methods report the same out-of-bag accuracy because the same fitted forest (same seed) is the starting point of each; only the importance and the cross-validated accuracy differ."]
    out["snippet_8_6_printed_example"] = float(PB.weighted_tau_example())
    md += ["", f"Snippet 8.6 as printed (importance [.55, .33, .07, .05] against the inverse of the PCA ranks [1, 2, 4, 3]): weighted Kendall's tau {out['snippet_8_6_printed_example']:.4f}. The extract prints the call without its result."]
    return out, md, tabs


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 9
# ---------------------------------------------------------------------------------------------------------------------
def ch9(V, cfg, log):
    out, md = {}, []
    se = sorted_events(V.events)
    samp = systematic_sample(se, cfg["hp_sample"])
    X = samp[V.features].astype(float)
    y = samp["bin"].astype(int)
    w = samp["w"].astype(float)
    w = w / w.sum() * len(w)
    t1 = pd.Series(samp["t1_dt"].to_numpy(), index=samp.index)
    n_tr = int(len(samp) * .8)
    # purge the training part of labels that run into the test part (the book's PurgedKFold applies this inside CV; for the hold-out it is done here)
    test_start = samp.index[n_tr]
    tr_mask = np.arange(len(samp)) < n_tr
    tr_mask &= (t1.to_numpy() <= test_start.to_datetime64())
    Xtr, ytr, wtr, t1tr = X[tr_mask], y[tr_mask], w[tr_mask], t1[tr_mask]
    Xte, yte = X.iloc[n_tr:], y.iloc[n_tr:]
    out["split"] = {"sample": int(len(samp)), "train_after_purge": int(tr_mask.sum()), "test": int(len(Xte)), "test_start": str(test_start)}
    pipe = lambda: PB.MyPipeline([("sc", StandardScaler()), ("svc", SVC(kernel="rbf", class_weight="balanced"))])      # noqa: E731
    grid = {"svc__C": [1e-2, 1e-1, 1, 10, 100], "svc__gamma": [1e-2, 1e-1, 1, 10, 100]}
    fit_params = {"svc__sample_weight": wtr.to_numpy()}
    gs = PB.clfHyperFit(Xtr, ytr, t1tr, pipe(), grid, cv=3, bagging=[0, None, 1.], rndSearchIter=0, n_jobs=cfg["threads"], pctEmbargo=.01, **fit_params)
    rs_dist = {"svc__C": PB.logUniform(a=1e-2, b=1e2), "svc__gamma": PB.logUniform(a=1e-2, b=1e2)}
    np.random.seed(cfg["seed"])
    rs = PB.clfHyperFit(Xtr, ytr, t1tr, pipe(), rs_dist, cv=3, bagging=[0, None, 1.], rndSearchIter=cfg["hp_random_iter"], n_jobs=cfg["threads"], pctEmbargo=.01, **fit_params)
    base = pipe().fit(Xtr, ytr, svc__sample_weight=wtr.to_numpy())

    def ev(model):
        pred = model.predict(Xte)
        dec = model.decision_function(Xte)
        return {"f1": float(f1_score(yte, pred, zero_division=0)), "accuracy": float(accuracy_score(yte, pred)), "auc": num(roc_auc_score(yte, dec)) if yte.nunique() == 2 else None}
    out["grid_best"] = {"C": float(gs.named_steps["svc"].C), "gamma": float(gs.named_steps["svc"].gamma), **ev(gs)}
    out["random_best"] = {"C": float(rs.named_steps["svc"].C), "gamma": float(rs.named_steps["svc"].gamma), **ev(rs), "draws": cfg["hp_random_iter"]}
    out["default_svc"] = {**ev(base)}
    out["all_positive_f1"] = float(f1_score(yte, np.ones(len(yte), int), zero_division=0))
    out["label_share_1_test"] = float(yte.mean())
    ks, desc = PB.test_logUniform(a=1e-3, b=1e3, size=10000, seed=cfg["seed"])
    out["logUniform_ks"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}
    md += [f"Snippets 9.1-9.4: RBF support-vector classifier (scaled features, class-balanced, attribution weights) tuned with purged 3-fold CV (1% embargo) on the first 80% of a systematic sample of {len(samp):,} events "
           f"({int(tr_mask.sum()):,} left after purging labels that run into the test part), scored by F1 (the printed rule for 0/1 labels); evaluated on the last 20% ({len(Xte):,} events, label share 1: {out['label_share_1_test']:.3f}). "
           f"Grid: C and gamma in 0.01, 0.1, 1, 10, 100 (25 points). Randomised: {cfg['hp_random_iter']} draws of C and gamma log-uniform on [0.01, 100] (Snippet 9.4). Always predicting 1 has F1 {out['all_positive_f1']:.4f}.", "",
           "| setting | C | gamma | hold-out F1 | accuracy | AUC |", "|---|---|---|---|---|---|"]
    md.append(f"| grid search | {out['grid_best']['C']:g} | {out['grid_best']['gamma']:g} | {out['grid_best']['f1']:.4f} | {out['grid_best']['accuracy']:.4f} | {out['grid_best']['auc']:.4f} |")
    md.append(f"| randomised search | {out['random_best']['C']:.4g} | {out['random_best']['gamma']:.4g} | {out['random_best']['f1']:.4f} | {out['random_best']['accuracy']:.4f} | {out['random_best']['auc']:.4f} |")
    md.append(f"| untuned (C = 1, gamma = 'scale') | 1 | scale | {out['default_svc']['f1']:.4f} | {out['default_svc']['accuracy']:.4f} | {out['default_svc']['auc']:.4f} |")
    md += ["", f"Snippet 9.4's own test: Kolmogorov-Smirnov statistic of log(draws) against the uniform: {ks.statistic:.4f} (p = {ks.pvalue:.3f}) for 10,000 draws on [0.001, 1000]."]
    return out, md


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 10
# ---------------------------------------------------------------------------------------------------------------------
def _book_signals_for_symbol(args):
    """Worker: the book's getSignal (10.1-10.3) for one symbol and one path.  Returns the averaged, discretised signal series and the un-discretised average."""
    sym, t0, t1, side, p, step = args
    ix = pd.DatetimeIndex(pd.to_datetime(t0, unit="s"))
    ev = pd.DataFrame({"t1": pd.to_datetime(t1, unit="s"), "side": side}, index=ix)
    prob = pd.Series(p, index=ix)
    pred = pd.Series((p > .5).astype(float), index=ix)
    sig1 = PC.getSignal(ev, step, prob, pred, 2, 1)
    sig0 = pred * (2 * norm.cdf((prob - .5) / (prob * (1. - prob)) ** .5) - 1)
    sig0 = sig0 * ev["side"]
    sig_raw = PC.avgActiveSignals(sig0.to_frame("signal").join(ev[["t1"]]), 1)
    return sym, sig1, sig_raw


def ch10(V, cfg, log):
    out, md = {}, []
    step = float(V.summary["params"]["step"])
    pred = V.cpcv_predictions()
    # --- 10.1 vs the project's bet-size function ------------------------------------------------------------------------------------------
    p = np.clip(np.linspace(0.01, 0.99, 981), 1e-9, 1 - 1e-9)
    book = np.where(p > .5, 2 * norm.cdf((p - .5) / (p * (1 - p)) ** .5) - 1, 0.)
    book = np.clip((book / step).round() * step, -1, 1)
    out["bet_size_vs_project_max_abs_diff"] = float(np.max(np.abs(book - MM.bet_size(p, step))))
    # --- 10.1 - 10.3 on the venue's CPCV path 0 (per symbol), cross-checked with the project's bar-by-bar averaging ------------------------------------
    e = V.events
    pp = pred[pred["path"] == 0].set_index("event")["p"]
    e0 = e.assign(p=pp.reindex(e.index).to_numpy())
    syms = list(e.groupby("symbol").size().sort_values(ascending=False, kind="stable").index)
    if cfg["ch10_symbols"]:
        syms = syms[:cfg["ch10_symbols"]]
    jobs = []
    for s in syms:
        g = e0[e0["symbol"] == s].sort_values("pos")
        jobs.append({"func": _book_signals_for_symbol, "args": (s, g["t0"].to_numpy(), g["t1"].to_numpy(), g["side"].to_numpy(float), g["p"].to_numpy(float), step)})
    res = MP.processJobs([{"func": _expand, "job": j} for j in jobs], task="getSignal", numThreads=cfg["threads"]) if cfg["threads"] > 1 else [_expand(job=j) for j in jobs]
    V.cache['signals'] = {sym: sig1 for sym, sig1, _ in res}
    V.cache['signal_symbols'] = syms
    diffs, nz, tot, round_effect = [], 0, 0, []
    bars = V.bars(syms) if len(syms) <= 60 else V.bars(syms[:60])
    for sym, sig1, sig_raw in res:
        g = e0[e0["symbol"] == sym].sort_values("pos")
        b = bars.get(sym)
        if b is None:
            continue
        # project: per-bar mean position from the same un-rounded per-event signals
        size_raw = np.where(g["p"].to_numpy() > .5, 2 * norm.cdf((g["p"].to_numpy() - .5) / (g["p"].to_numpy() * (1 - g["p"].to_numpy())) ** .5) - 1, 0.)
        avg, C = BA.average_positions(b.n, g["pos"].to_numpy(), g["t1pos"].to_numpy(), g["side"].to_numpy(float) * size_raw)
        tt = np.asarray(sig_raw.index.astype("datetime64[s]").astype(np.int64))
        bar_ix = np.searchsorted(b.t, tt)
        ok = (bar_ix < b.n) & (b.t[np.minimum(bar_ix, b.n - 1)] == tt)
        d = np.abs(sig_raw.to_numpy()[ok] - avg[bar_ix[ok]])
        diffs.append(float(d.max()) if len(d) else 0.)
        nz += int((sig1 != 0).sum()); tot += int(len(sig1))
        # average-then-round (book) vs round-then-average (project) at the same change points
        size_r = MM.bet_size(g["p"].to_numpy(), step)
        avg_r, _ = BA.average_positions(b.n, g["pos"].to_numpy(), g["t1pos"].to_numpy(), g["side"].to_numpy(float) * size_r)
        round_effect.append(float(np.mean(np.abs(sig1.to_numpy()[ok] - avg_r[bar_ix[ok]]))) if ok.any() else 0.)
    out["averaging_crosscheck"] = {"symbols": len(res), "max_abs_diff_book_vs_project_unrounded": float(max(diffs)) if diffs else None,
                                   "mean_abs_diff_round_after_vs_round_before_averaging": float(np.mean(round_effect)) if round_effect else None,
                                   "share_of_change_points_with_nonzero_signal": nz / tot if tot else None, "change_points": tot}
    # --- the project's averaging evaluation on the bars (what run_bet_averaging.py does on candles) ---------------------------------------------------
    ev = e
    allsyms = sorted(ev["symbol"].unique())
    bars_all = V.bars(allsyms)
    closes = {s: pd.Series(b.close, index=b.idx) for s, b in bars_all.items()}
    RBA.check_alignment(ev, closes)
    sp = V.summary["params"]
    cost_frac = float(sp["cost_frac"])
    windows = {k: (int(v[0]), int(v[1])) for k, v in V.summary["meta"]["window_epoch"].items()}
    plist = RBA.sizes_by_path(pred, step, int(sp["seed"]))
    prow, _, wrow = RBA.evaluate_scope(ev, closes, "cpcv", plist, cost_frac / 2.0, cost_frac, windows)
    paths = pd.DataFrame(prow)
    wins = pd.DataFrame(wrow)
    S = RBA.summarise(paths, wins)["cpcv"]
    out["averaged_evaluation"] = S
    md += [f"Snippets 10.1-10.3 (the book's getSignal / avgActiveSignals / discreteSignal) on CPCV path 0 of the {len(res)} most active symbols' events: the average signal of the bets open at each start or end time, then rounded to steps of {step}. "
           f"Against the project's own bar-by-bar averaging on the same un-rounded signals at the same {tot:,} change points: largest absolute difference {out['averaging_crosscheck']['max_abs_diff_book_vs_project_unrounded']:.2e}. "
           f"The book rounds after averaging, the project rounds each bet before averaging; mean absolute difference between the two at the change points: {out['averaging_crosscheck']['mean_abs_diff_round_after_vs_round_before_averaging']:.4f}. "
           f"Bet-size function: largest difference between the book's formula and meta_model.bet_size over p in [0.01, 0.99]: {out['bet_size_vs_project_max_abs_diff']:.1e}.", "",
           f"The project's averaging evaluation (bets averaged while open, {cost_frac / 2 * 1e4:.0f} bp per side on position changes) run on the dollar bars, CPCV paths (mean over {S['primary']['paths']} paths):", "",
           "| variant | mean bp per active instrument-day | gross | cost | annualised Sharpe | paths with Sharpe < 0 |", "|---|---|---|---|---|---|"]
    for v in ("primary", "meta", "placebo"):
        x = S[v]
        md.append(f"| {v} | {x['mean_bp_per_symbol_day']:.2f} | {x['gross_bp_per_symbol_day']:.2f} | {x['cost_bp_per_symbol_day']:.2f} | {x['sr_annual']:.2f} | {x['paths_sr_negative']}/{x['paths']} |")
    mp = S["meta_minus_placebo"]
    md += ["", f"Meta minus placebo: Sharpe higher on {mp['paths_sr_higher']}/{mp['paths']} paths (mean daily-Sharpe difference {mp['mean_sr_daily_diff']:.4f}).", "",
           "Reporting windows (mean bp per active instrument-day, averaged over paths; the sell-off window and June 2026 are UTC calendar windows):", "", "| window | primary | meta | placebo | instrument-days per path |", "|---|---|---|---|---|"]
    for wn in windows:
        r = [S["windows"].get(f"{v}|{wn}", {}) for v in ("primary", "meta", "placebo")]
        md.append(f"| {wn} | " + " | ".join(("n/a" if x.get("mean_bp_per_symbol_day") is None or not math.isfinite(x.get("mean_bp_per_symbol_day")) else f"{x['mean_bp_per_symbol_day']:.2f}") for x in r) + f" | {r[0].get('symbol_days_per_path', float('nan')):.1f} |")
    # --- 10.4 -------------------------------------------------------------------------------------------------------------------------------
    w0, t0_, l0 = PC.main_10_4()
    trgt = float(e["trgt"].median())
    ptsl = float(V.summary["params"]["ptsl"][0])
    x_cal = ptsl * trgt * 1e4                         # profit-target distance in bp of price
    price = 1e4                                       # price level 10,000 bp = 1, so that divergences are in bp
    f = price + x_cal
    wv = PC.getW(x_cal, .95)
    half = PC.getTPos(wv, f, price + x_cal / 2., 100)
    start = PC.getTPos(wv, f, price, 100)
    lp = PC.limitPrice(start, 0, f, wv, 100)
    out["dynamic_size"] = {"book_example": {"w": w0, "tPos": t0_, "limit_price": l0}, "venue_median_daily_vol": trgt, "profit_target_bp": x_cal, "w_for_95pct_at_target": wv,
                           "target_position_at_entry": start, "target_position_halfway_to_target": half, "limit_price_bp_above_entry": lp - price}
    md += ["", f"Snippet 10.4 (dynamic position size and limit price): the book's own example gives target position {t0_} and limit price {l0:.4f} (p. 90: 97 and 112.3657). "
           f"On this venue's numbers: median daily-volatility target {trgt * 1e4:.0f} bp, profit-taking barrier {ptsl:g} x that = {x_cal:.0f} bp. Calibrating the sigmoid to bet size 0.95 at a divergence equal to the profit target, "
           f"a maximum position of 100 units gives {start} units at entry, {half} units when the price has moved half-way to the target, and a limit price {lp - price:.0f} bp above the entry price for the entry order."]
    return out, md


def _expand(job):
    """Worker adapter for MP.processJobs: expandCall deletes 'func' from the dict it is given."""
    return _book_signals_for_symbol(job["args"])
