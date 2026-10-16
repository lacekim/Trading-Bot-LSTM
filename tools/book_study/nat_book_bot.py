"""NAT COPY of book_bot_v2.py: the events module is nat_book_events_v2 (Natenberg volatility); nothing else changed.  New file; the original is not edited.
The book's strategy as ONE bot: one command per venue, raw dollar bars in, one consolidated result out.  New file; nothing audited is edited; nothing from outside the book.

Stages, in the book's order, all in one process on one set of events:
  1 events      Ch.2 CUSUM on dollar bars; Ch.3 side-free triple-barrier labels (Snippets 3.1-3.7, bin = sign of the return); Ch.4 uniqueness and attribution weights (4.1, 4.2, 4.10) and, as the book orders it,
                Snippet 3.8 dropLabels afterwards; Snippet 4.11 time decay (the snippet's own default c = 1, i.e. none; any other value would need per-split decay and is refused); Ch.5 FFD; Ch.17 SADF and
                Chu-Stinchcombe-White CUSUM; Ch.18 plug-in and Kontoyiannis entropy; Ch.19 Corwin-Schultz (19.1), Becker-Parkinson (19.2), Roll, Kyle / Amihud / Hasbrouck lambdas (Eq. 82-83, 90-92)
  2 Ch.13       optimal trading rule: Equations 13.5-13.7 on the long paths of events whose labels end before the cutoff; the 20 x 20 mesh (0.5 .. 10 sigma) with 100,000 simulated paths (vectorised form of
                Snippet 13.2, checked against the verbatim loop on a few nodes); Step 5a picks the node; Definition 2 (overfit rule) is evaluated on the paths after the cutoff
  3 Ch.8        feature importance on pre-cutoff events: MDI (8.2), MDA (8.3), SFI (8.4); orthogonal features (8.5) and the weighted Kendall's tau of MDI against the inverse PCA rank (8.6)
  4 Ch.9        randomised purged-CV search of the forest's parameters (9.3, 9.4) on pre-cutoff events; the winner is the model used below
  5 Ch.6/7/12   the tuned bagged entropy forest (sequential bootstrap) scored by combinatorial purged CV (5 paths) and by walk-forward (trained on labels ending before the cutoff)
  6 Ch.10       side = predicted class, size = Snippets 10.1 / 10.3, overlapping bets averaged (10.2); each bet is closed by the Ch.13 rule; Snippet 10.4 target position and limit price for every order
  7 Ch.16, 2.4.1  HRP across instruments (Snippet 16.4), re-estimated each calendar quarter on the trailing 252 trading days; costs per side; the section 2.4.1 rebalance cost on every rebalance date and the volume capacity of every trade; daily compounding
  8 Ch.14       PSR / DSR and Snippets 14.1-14.4, annualised return (Eq. 49), classification scores (Table 14.1)
  9 Ch.15       implied precision (15.3) and implied betting frequency (15.4), strategy risk (15.5) from the bot's own bets
Usage (from the project root):
  python tools/book_study/book_bot.py --venue kraken --bars runtime/dollar_bars/kraken --cost-bp-per-side 12 --out runtime/book_bot
  python tools/book_study/book_bot.py --venue alpaca --bars runtime/dollar_bars/alpaca --cost-bp-per-side 10 --universe runtime/book_meta_alpaca/20261012T023148Z/universe.csv --out runtime/book_bot
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from scipy.stats import weightedtau
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler

import bar_loader as BL
import bet_averaging as BA
import book_etf as EX
import nat_book_events_v2 as BV
import book_final as BF
import book_otr_exit as OX
import book_port_a as PA
import book_port_b as PB
import book_port_c as PC
import book_port_e as PE
import book_stats as BS
import meta_eval as ME
import meta_events as E
import meta_model as MM
import ports_common as PCm
import run_book_study as RB

DAY = 86400
SEED = 20261022
HCAP = {"kraken": 600, "alpaca": 150}
MAX_GAP_DAYS = {"kraken": 1, "alpaca": 5}
FEATURES = BV.FEATURES_BOOK
BASE_FOREST = dict(n_estimators=100, bootstrap="sequential", max_draws=10000)
CUTOFF, EMBARGO, STEP, K_OTR = "2024-01-01", 0.01, 0.1, 100
N_GROUPS, K_GROUPS = 6, 2
MESH = np.linspace(0.5, 10, 20)                     # Snippet 13.2's default mesh
N_MESH_PATHS = 100000
DECAY_C = 1.0                                       # Snippet 4.11's default clfLastW
TARGET_SR = (1.0, 2.0)                              # Ch.15: the book's example target is 2; 1 is added for reference
Q_MAX, M_STAR = 100, 0.95                           # Snippet 10.4: maximum position and the size at the calibration divergence (the snippet's own example values)


class _Shim:
    def __init__(self, trees):
        self.estimators_ = trees


# ----------------------------------------------------------------------------------------------------------------------
def load_refvol(bars_dir: Path, sym: str, b) -> np.ndarray:
    """Volume column of ref_1h/<sym>.csv in the order BL.load_one gives the reference series (same filters)."""
    r = pd.read_csv(bars_dir / "ref_1h" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "high", "low", "close", "volume"])
    r = r.dropna(subset=["close"])
    r = r[r["close"] > 0]
    rt = BL._epoch(r["date"]) + 3600
    o2 = np.argsort(rt, kind="stable")
    assert len(rt) == len(b.ref_t) and (rt[o2] == b.ref_t).all() and np.allclose(r["close"].to_numpy(float)[o2], b.ref_close), f"{sym}: reference volume does not line up"
    return r["volume"].to_numpy(float)[o2]


def stage_ch13(bars, ev, cutoff_ts, seed, run, log):
    """Step 1 (Eq. 13.5-13.7) on the long paths (K bars, units of the event's sigma) of events that end before the cutoff; Steps 2-4 on the 20 x 20 mesh with N_MESH_PATHS simulated paths; Step 5a: best Sharpe node;
    Definition 2 on the paths of events that start after the cutoff."""
    rows, paths = [], []
    for sym, g in ev.groupby("symbol", sort=False):
        b = bars[sym]
        lc = np.log(b.close)
        pos = g["pos"].to_numpy(np.int64)
        ok = pos + K_OTR < b.n
        pos_ok = pos[ok]
        same = b.seg[np.minimum(pos_ok + K_OTR, b.n - 1)] == b.seg[pos_ok]
        sel = np.flatnonzero(ok)[same]
        ps = pos[sel]
        q = (lc[ps[:, None] + np.arange(1, K_OTR + 1)[None, :]] - lc[ps][:, None]) / g["sigma"].to_numpy(float)[sel][:, None]
        paths.append(q.astype(np.float32))
        rows.append(g.index.to_numpy()[sel])
    q, ix = np.concatenate(paths), np.concatenate(rows)
    evk = ev.loc[ix]
    tr = evk["t1"].to_numpy() < cutoff_ts
    te = evk["t0"].to_numpy() >= cutoff_ts
    est = BS.estimate_ou(q[tr], evk["u"].to_numpy()[tr])
    phi, a = est["phi"], est["a"]
    hl = BS.half_life(phi)
    forecast = a / (1 - phi) if abs(1 - phi) > 1e-12 else float("nan")
    rng = np.random.default_rng(seed + 13)
    sim = BS.simulate_ou_paths(a, phi, rng.standard_normal((N_MESH_PATHS, K_OTR)) * est["sigma"])
    mesh = BS.mesh_exit_stats(sim, pts=tuple(MESH), sls=tuple(MESH))
    del sim
    sh = np.asarray(mesh["sharpe"])
    i, j = divmod(int(np.nanargmax(sh)), sh.shape[1])
    pt, sl = float(MESH[i]), float(MESH[j])
    # the verbatim loop of Snippet 13.2 on four nodes (the chosen one and three others), nIter paths each: agrees with the mesh within sampling error
    n_chk = 3000
    random.seed(seed + 1)
    coeffs = {"forecast": forecast if np.isfinite(forecast) else 0., "hl": hl if np.isfinite(hl) else 1e9, "sigma": est["sigma"]}
    chk_pt, chk_sl = sorted({pt, 2.5}), sorted({sl, 2.5})
    bt = PC.batch(coeffs, nIter=n_chk, maxHP=K_OTR, rPT=np.array(chk_pt), rSLm=np.array(chk_sl), seed=0)
    chk = [{"pt": float(x[0]), "sl": float(x[1]), "sharpe_verbatim_loop": PCm.num(x[4]), "sharpe_mesh": PCm.num(sh[int(np.argmin(np.abs(MESH - x[0]))), int(np.argmin(np.abs(MESH - x[1])))])} for x in bt]
    # Definition 2: OOS Sharpe of the chosen rule against the median OOS Sharpe over the whole mesh, on the paths of events that start after the cutoff
    mt = BS.mesh_exit_stats(q[te].astype(np.float32), pts=tuple(MESH), sls=tuple(MESH))
    sho = np.asarray(mt["sharpe"])
    oos_star, oos_med = float(sho[i, j]), float(np.nanmedian(sho))
    out = {"K": K_OTR, "fit_events": int(tr.sum()), "test_events": int(te.sum()), "phi": float(phi), "a": float(a), "sigma_fit": float(est["sigma"]), "half_life_bars": PCm.num(hl), "forecast_level_sigma": PCm.num(forecast),
           "mesh_pt": MESH.tolist(), "mesh_sl": MESH.tolist(), "n_mesh_paths": N_MESH_PATHS, "chosen_pt": pt, "chosen_sl": sl, "simulated_sharpe_at_chosen": float(sh[i, j]), "simulated_sharpe_median_over_mesh": float(np.nanmedian(sh)),
           "verbatim_loop_check": chk, "verbatim_loop_paths": n_chk, "definition_2": {"oos_sharpe_chosen": oos_star, "oos_sharpe_mesh_median": oos_med, "oos_sharpe_mesh_best_in_hindsight": float(np.nanmax(sho)), "overfit": bool(oos_star < oos_med)}}
    np.savetxt(run / "ch13_mesh_sharpe_simulated.csv", sh, delimiter=",", fmt="%.8g")
    np.savetxt(run / "ch13_mesh_sharpe_oos.csv", sho, delimiter=",", fmt="%.8g")
    log(f"Ch.13: phi {phi:.4f}, half-life {hl:.1f}, forecast {forecast:.3f}; chosen pt {pt:g} / sl {sl:g}; Definition 2 overfit = {out['definition_2']['overfit']}")
    return out


def stage_ch8_ch9(X, y, w, sym, starts, ends, t0, t1, u, seed, n_iter, n_tree_imp, n_tree_search, out, log):
    avg_u_fn = lambda ix: float(u[ix].mean())  # noqa: E731
    mk1 = lambda: MM.BaggedForest(seed=seed + 900, **{**BASE_FOREST, "n_estimators": n_tree_imp, "max_features": 1})  # noqa: E731
    f = mk1().fit(X, y, w, sym, starts, ends, float(u.mean()))
    mdi = PB.featImpMDI(_Shim(f.trees), FEATURES)
    mdi.to_csv(out / "mdi.csv")
    log("Ch.8 MDI done")
    mda, mda_base = MM.mda_importance(X, y, w, sym, starts, ends, avg_u_fn, t0, t1, FEATURES, 5, EMBARGO, mk1, seed)
    mda.to_csv(out / "mda.csv")
    log("Ch.8 MDA done")
    mk2 = lambda: MM.BaggedForest(seed=seed + 901, **{**BASE_FOREST, "n_estimators": max(10, n_tree_imp // 2), "max_features": 1})  # noqa: E731
    sfi = MM.sfi_importance(X, y, w, sym, starts, ends, avg_u_fn, t0, t1, FEATURES, 5, EMBARGO, mk2)
    sfi.to_csv(out / "sfi.csv")
    log("Ch.8 SFI done")
    # Snippets 8.5 / 8.6: orthogonal features (95% of the variance), MDI on them, weighted Kendall's tau against the inverse PCA rank
    dfX = pd.DataFrame(X, columns=FEATURES)
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    eVal, eVec = PB.get_eVec(dot, .95)
    P = np.dot(dfZ, eVec)
    pc_names = list(eVal.index)
    fp = MM.BaggedForest(seed=seed + 902, **{**BASE_FOREST, "n_estimators": n_tree_imp, "max_features": 1}).fit(P, y, w, sym, starts, ends, float(u.mean()))
    mdi_pc = PB.featImpMDI(_Shim(fp.trees), pc_names)
    mdi_pc["eigenvalue"] = eVal.to_numpy()
    mdi_pc.to_csv(out / "mdi_orthogonal.csv")
    tau = float(weightedtau(mdi_pc["mean"].to_numpy(), (np.arange(1, len(pc_names) + 1)) ** -1.)[0])
    log(f"Ch.8 orthogonal features: {len(pc_names)} components for 95% of the variance; weighted tau(MDI, inverse PCA rank) = {tau:.3f}")
    space = {"min_weight_fraction_leaf": PB.logUniform(0.005, 0.2), "max_features": [1, 2, 3, 6, len(FEATURES)]}
    cands = [{"min_weight_fraction_leaf": 0.05, "max_features": "sqrt", "_note": "untuned setting"}]
    cands += [dict(c, _note="random draw") for c in ParameterSampler(space, n_iter=n_iter, random_state=seed)]
    folds = list(MM.purged_kfold(t0, t1, 3, EMBARGO))
    rows = []
    for i, c in enumerate(cands):
        kw = {**BASE_FOREST, "n_estimators": n_tree_search, "min_weight_fraction_leaf": float(c["min_weight_fraction_leaf"]), "max_features": c["max_features"]}
        sc = []
        for trn, te in folds:
            ff = MM.BaggedForest(seed=seed + 1000 + i, **kw).fit(X[trn], y[trn], w[trn], sym[trn], starts[trn], ends[trn], avg_u_fn(trn))
            sc.append(-MM.weighted_log_loss(y[te], ff.predict_proba1(X[te]), w[te]))
        rows.append({"cand": i, "min_weight_fraction_leaf": float(c["min_weight_fraction_leaf"]), "max_features": c["max_features"], "mean_neg_log_loss": float(np.mean(sc)),
                     "fold_scores": json.dumps([float(s) for s in sc]), "note": c["_note"]})
        log(f"Ch.9 candidate {i}: mwfl {rows[-1]['min_weight_fraction_leaf']:.4f} max_features {c['max_features']} -> {rows[-1]['mean_neg_log_loss']:.6f}")
    R = pd.DataFrame(rows)
    R.to_csv(out / "search.csv", index=False)
    best = R.loc[R["mean_neg_log_loss"].idxmax()]
    tuned = {**BASE_FOREST, "min_weight_fraction_leaf": float(best["min_weight_fraction_leaf"]), "max_features": (best["max_features"] if best["max_features"] == "sqrt" else int(best["max_features"]))}
    tj = {"forest": tuned, "chosen_candidate": int(best["cand"]), "n_candidates": int(len(R)), "baseline_neg_log_loss": float(R.iloc[0]["mean_neg_log_loss"]), "best_neg_log_loss": float(best["mean_neg_log_loss"]),
          "mda_baseline_neg_log_loss": float(mda_base), "ortho_components": len(pc_names), "weighted_tau_mdi_vs_inverse_pca_rank": tau}
    (out / "tuned_forest.json").write_text(json.dumps(tj, indent=2), encoding="utf-8")
    return tj


def run_path(bars, closes, ev, pidx, pp, pt, sl, cost_side):
    side_pred = np.where(pp > 0.5, 1.0, -1.0)
    size = MM.bet_size(np.maximum(pp, 1.0 - pp), STEP)
    t1, tr = OX.otr_exits(bars, ev, pidx, side_pred, pt, sl, K_OTR)
    nets = BF.instrument_nets(ev, closes, pidx, side_pred, size, t1, cost_side)
    return nets, {"truncated_share": float(tr.mean()), "mean_bars_held": float((t1 - ev["pos"].to_numpy()[pidx]).mean())}, (side_pred, size, t1)


def stage_ch14(ev, closes, w_idx, side, size, t1pos, sched, r_year, last_day, run):
    lo = last_day - 364
    keys = sorted(sched)
    pos_ev, sym_ev, sig = ev["pos"].to_numpy(np.int64), ev["symbol"].to_numpy(), side * size
    E_day = np.zeros(last_day - lo + 1)
    for s, c in closes.items():
        sel = np.flatnonzero(sym_ev[w_idx] == s)
        if not len(sel):
            continue
        ts = c.index.astype("datetime64[s]").astype(np.int64)
        avg, _C = BA.average_positions(len(c), pos_ev[w_idx][sel], t1pos[sel], sig[sel])
        day = ts // DAY
        last_of_day = pd.Series(avg).groupby(day).last()
        pos_d = last_of_day.reindex(np.arange(min(lo, int(day.min())), last_day + 1)).ffill().fillna(0.0)
        for k, st in enumerate(keys):
            en = keys[k + 1] if k + 1 < len(keys) else last_day + 1
            wgt = float(sched[st].get(s, 0.0)) if len(sched[st]) else 0.0
            if wgt == 0.0:
                continue
            a_, b_ = max(st, lo), min(en, last_day + 1)
            if a_ < b_:
                E_day[a_ - lo:b_ - lo] += wgt * pos_d.loc[a_:b_ - 1].to_numpy()
    idx = pd.to_datetime(np.arange(lo, last_day + 1) * DAY, unit="s")
    tPos, ret_d = pd.Series(E_day, index=idx), pd.Series(r_year, index=idx)
    equity = (1.0 + ret_d).cumprod()
    bets, hp = PC.getBets(tPos), PC.getHoldingPeriod(tPos)
    prev, bet_ret = tPos.index[0], []
    for b in bets:
        seg = ret_d.loc[(ret_d.index > prev) & (ret_d.index <= b)] if prev != tPos.index[0] else ret_d.loc[ret_d.index <= b]
        bet_ret.append((b, float((1.0 + seg).prod() - 1.0)))
        prev = b
    bet_ret = pd.Series([v for _, v in bet_ret], index=[b for b, _ in bet_ret])
    hhi_pos, hhi_neg, t_hhi = PC.getHHIs(bet_ret)
    dd, tuw = PC.computeDD_TuW(equity)
    nn = lambda x: None if not np.isfinite(x) else float(x)  # noqa: E731
    years = len(idx) / 365.25
    phi_T = float(equity.iloc[-1])
    res = {"start": str(idx[0].date()), "end": str(idx[-1].date()), "days": int(len(idx)), "gain_pct": 100.0 * (phi_T - 1.0), "annualised_return_pct_eq49": 100.0 * (phi_T ** (1.0 / years) - 1.0), "years_elapsed": years,
           "days_with_nonzero_position": int((tPos != 0).sum()), "max_abs_position": float(tPos.abs().max()), "mean_abs_position": float(tPos.abs().mean()), "bets_14_1": int(len(bets)),
           "avg_holding_period_days_14_2": nn(hp), "hhi_positive_returns_14_3": nn(hhi_pos), "hhi_negative_returns_14_3": nn(hhi_neg), "hhi_bets_per_month_14_3": nn(t_hhi),
           "drawdowns_14_4": {"count": int(len(dd)), "max_pct": 100.0 * float(dd.max()) if len(dd) else 0.0, "p95_pct": 100.0 * float(np.percentile(dd, 95)) if len(dd) else 0.0},
           "time_under_water_years_14_4": {"count": int(len(tuw)), "max": float(tuw.max()) if len(tuw) else None, "p95": float(np.percentile(tuw, 95)) if len(tuw) else None}}
    tPos.rename("target_position").to_csv(run / "ch14_target_position.csv", float_format="%.12g")
    ret_d.rename("net_return").to_csv(run / "ch14_daily_return.csv", float_format="%.12g")
    (run / "ch14_year.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def stage_concentration(rets, sched, run):
    """Equations 77-78: the entropy-based concentration of each HRP allocation.  Risk loadings f = V' w on the eigenvectors V of the covariance matrix of the same trailing window the HRP used (eigenvalues L),
    theta_i = f_i^2 L_i / sum_n f_n^2 L_n, H = 1 - exp(-sum_i theta_i log theta_i) / N.  The extract prints the two equations but not the definition of f; f = V' w is the usual one and is an assumption."""
    rows = []
    for st in sorted(sched):
        wts = sched[st]
        if not len(wts):
            continue
        cols = list(wts.index)
        win = rets[rets.index < pd.Timestamp(st * DAY, unit="s")].tail(BF.REB_WINDOW)[cols].fillna(0.0)
        lam, V = np.linalg.eigh(win.cov().to_numpy())
        f = V.T @ wts.to_numpy(float)
        th = f ** 2 * lam
        th = th / th.sum()
        th = th[th > 0]
        rows.append((pd.Timestamp(st * DAY, unit="s").strftime("%Y-%m-%d"), len(cols), float(1.0 - math.exp(-(th * np.log(th)).sum()) / len(cols))))
    df = pd.DataFrame(rows, columns=["rebalance_day", "n_assets", "H_eq78"])
    df.to_csv(run / "portfolio_concentration_eq77_78.csv", index=False, float_format="%.10g")
    return {"rebalances": int(len(df)), "H_first": float(df["H_eq78"].iloc[0]), "H_last": float(df["H_eq78"].iloc[-1]), "H_mean": float(df["H_eq78"].mean()), "H_min": float(df["H_eq78"].min()), "H_max": float(df["H_eq78"].max())}


def classification_scores(y, pred):
    """Table 14.1: accuracy, precision, recall, F1 for the 'up' class, NaN where a denominator is 0."""
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp, tn, fn = int((y & pred).sum()), int((~y & pred).sum()), int((~y & ~pred).sum()), int((y & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": (tp + tn) / max(1, tp + fp + tn + fn), "precision": prec, "recall": rec, "f1": f1}


def stage_ch15(ev, closes, w_idx, side, t1pos, cost_side, last_day, run):
    """Per-bet unit returns of the walk-forward bets that started in the last 365 days: side x (exit close / entry close - 1) - round-trip cost.  Snippet 15.3 (implied precision), 15.4 (implied frequency) and 15.5 (strategy risk)."""
    sym_ev, pos_ev, t0_ev = ev["symbol"].to_numpy(), ev["pos"].to_numpy(np.int64), ev["t0"].to_numpy(np.int64)
    lo_t = (last_day - 364) * DAY
    ret = np.full(len(w_idx), np.nan)
    for s in np.unique(sym_ev[w_idx]):
        m = np.flatnonzero(sym_ev[w_idx] == s)
        c = closes[s].to_numpy(float)
        ret[m] = side[m] * (c[t1pos[m]] / c[pos_ev[w_idx][m]] - 1.0) - 2.0 * cost_side
    sel = (t0_ev[w_idx] >= lo_t) & np.isfinite(ret)
    r = ret[sel]
    p_obs = float((r > 0).mean())
    out = {"n_bets": int(sel.sum()), "observed_precision": float((r > 0).mean()), "t_value_of_precision_eq54": float((2 * p_obs - 1) / (2 * math.sqrt(p_obs * (1 - p_obs))) * math.sqrt(sel.sum())), "mean_positive_return": float(r[r > 0].mean()), "mean_nonpositive_return": float(r[r <= 0].mean()), "targets": {}}
    for tsr in TARGET_SR:
        rp, rn, freq = out["mean_positive_return"], out["mean_nonpositive_return"], float(len(r))
        p_star = PC.binHR(rn, rp, freq, tsr)
        out["targets"][str(tsr)] = {"implied_precision_15_3": float(p_star), "implied_frequency_15_4_at_observed_precision": PCm.num(PC.binFreq(rn, rp, out["observed_precision"], tsr) or float("nan")),
                                    "sharpe_of_observed_precision_eq56": float(PC.binSR(rn, rp, freq, out["observed_precision"])), "prob_failure_15_5": float(PC.probFailure(r, freq, tsr))}
    (run / "ch15.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def stage_ch10_orders(ev, w_idx, side, size, last_day, run):
    """Snippet 10.4 for every order that starts in the last 365 days: the forecast price is the barrier of the predicted side (entry x (1 +/- 2 x trgt)); w is calibrated so that the size is M_STAR at a
    divergence equal to that barrier distance; target position = int(betSize x Q); limit price = the snippet's limitPrice for a long order, mirrored around the entry price for a short one."""
    pos_ev, t0_ev, trgt = ev["pos"].to_numpy(np.int64), ev["t0"].to_numpy(np.int64), ev["trgt"].to_numpy(float)
    lo_t = (last_day - 364) * DAY
    rows = []
    for k, e in enumerate(w_idx):
        if t0_ev[e] < lo_t:
            continue
        x = 2.0 * trgt[e]                                          # divergence in return units (entry price 1)
        w = PC.getW(x, M_STAR)
        f = 1.0 + side[k] * x
        tp = PC.getTPos(w, f, 1.0, Q_MAX) if side[k] > 0 else -PC.getTPos(w, 1.0 + x, 1.0 - 0.0, Q_MAX)
        long_lp = PC.limitPrice(abs(tp), 0, 1.0 + x, w, Q_MAX) if tp != 0 else float("nan")
        lp = long_lp if side[k] > 0 else 2.0 - long_lp
        rows.append((int(e), int(side[k]), float(size[k]), int(tp), float(lp - 1.0)))
    df = pd.DataFrame(rows, columns=["event", "side", "size_10_3", "target_position_10_4", "limit_price_offset_from_entry"])
    df.to_csv(run / "orders_10_4.csv", index=False, float_format="%.10g")
    return {"n_orders": int(len(df)), "median_target_position": float(df["target_position_10_4"].abs().median()), "median_limit_offset_bp": float(1e4 * df["limit_price_offset_from_entry"].median())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=("kraken", "alpaca"))
    ap.add_argument("--bars", required=True)
    ap.add_argument("--cost-bp-per-side", type=float, required=True)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-years", type=float, default=0.5)
    ap.add_argument("--quick", action="store_true", help="smoke test only: few trees, few candidates")
    a = ap.parse_args()
    assert DECAY_C == 1.0, "Snippet 4.11 with c != 1 needs decay factors computed per training split"
    t_start = time.time()
    started = datetime.now(timezone.utc)
    venue, seed = a.venue, a.seed
    run = Path(a.out) / f"{venue}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    run.mkdir(parents=True, exist_ok=False)
    lf = open(run / "run.log", "w", encoding="utf-8")

    def log(m):
        line = f"[{time.time() - t_start:7.0f}s] {m}"
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()

    before = RB.integrity_snapshot()
    bars_dir = Path(a.bars)
    man = BL.read_manifest(bars_dir)
    assert man["venue"] == venue
    break_s = int(man["break_hours"]) * 3600
    names = BL.list_symbols(bars_dir)
    if a.symbols:
        names = [n for n in names if n.upper() in {s.upper() for s in a.symbols}]
    if a.limit:
        names = names[: a.limit]
    bars, skipped = BL.load_all(bars_dir, names, a.min_years, log)
    refvol = {s: load_refvol(bars_dir, s, b) for s, b in bars.items()}
    barvol = {s: EX.bar_volume(bars_dir, s, b.n, b.t) for s, b in bars.items()}
    log(f"{venue}: {len(bars)} symbols, {sum(b.n for b in bars.values()):,} bars")
    # ---------------- 1 events ----------------
    eligible = BV.membership(a.universe)[0] if a.universe else None
    ev, einfo = BV.build_book_events(bars, break_s, 4.0, HCAP[venue], (2.0, 2.0), eligible, refvol, log)
    n_all = len(ev)
    shares = ev["bin_book"].value_counts(normalize=True).sort_index().to_dict()
    ev = PA.dropLabels(ev.rename(columns={"bin_book": "bin"}), minPct=.05, verbose=True).rename(columns={"bin": "bin_book"}).reset_index(drop=True)      # Snippet 3.8
    labels_dropped = sorted(set(shares) - set(ev["bin_book"].unique()))
    ev["bin"] = (ev["bin_book"] > 0).astype(np.int8)
    tW = pd.Series(ev["u"].to_numpy(float))                                                                                                              # Snippet 4.11 (c = 1)
    clfW, dconst, dslope = PE.getTimeDecay(tW, clfLastW=DECAY_C)
    ev["w"] = ev["w"].to_numpy(float) * clfW.to_numpy(float)
    log(f"Snippet 3.8: label shares {shares}, dropped {labels_dropped}; {n_all - len(ev)} events removed; Snippet 4.11: const {dconst}, slope {dslope}")
    ev.to_csv(run / "events.csv", index=False, float_format="%.12g")
    pd.DataFrame(einfo["per_symbol"]).to_csv(run / "symbols.csv", index=False)
    closes = {s: pd.Series(b.close, index=b.idx) for s, b in bars.items()}
    E.FEATURES = list(FEATURES)
    cutoff_ts = float(pd.Timestamp(CUTOFF).timestamp())
    cutoff_day = int(cutoff_ts) // DAY
    # ---------------- 2 Ch.13 ----------------
    c13 = stage_ch13(bars, ev, cutoff_ts, seed, run, log)
    (run / "ch13.json").write_text(json.dumps(c13, indent=2), encoding="utf-8")
    pt, sl = c13["chosen_pt"], c13["chosen_sl"]
    # ---------------- 3-4 Ch.8, Ch.9 on pre-cutoff events ----------------
    X, y, w, sym, starts, ends, t0, t1, u = ME.prepare(ev)
    tr = np.flatnonzero(t1 < cutoff_ts)
    n_imp, n_srch, n_it = (20, 15, 2) if a.quick else (50, 30, 10)
    log(f"Ch.8/9 on {len(tr):,} events with labels ending before {CUTOFF}")
    tj = stage_ch8_ch9(X[tr], y[tr], w[tr], sym[tr], starts[tr], ends[tr], t0[tr], t1[tr], u[tr], seed, n_it, n_imp, n_srch, run, log)
    forest = dict(tj["forest"])
    if a.quick:
        forest["n_estimators"] = 20
    # ---------------- 5 model: CPCV + walk-forward ----------------
    log("CPCV ...")
    cpred, cmet, cmeta = ME.run_cpcv(ev, N_GROUPS, K_GROUPS, EMBARGO, forest, seed, log)
    log("walk-forward ...")
    wpred, wmet = ME.run_walk_forward(ev, CUTOFF, forest, seed + 500, log)
    ybin = ev["bin"].to_numpy()
    w_idx, pw = wpred["event"].to_numpy(np.int64), wpred["p"].to_numpy(float)
    auc_cp = [float(roc_auc_score(ybin[g["event"].to_numpy()], g["p"].to_numpy())) for _, g in cpred.groupby("path")]
    wy = ybin[w_idx]
    wf_stats = {"auc": float(roc_auc_score(wy, pw)), "side_accuracy": float((((pw > 0.5).astype(int)) == wy).mean()), "up_share_test": float(wy.mean()), "n_test": int(len(w_idx)),
                "table_14_1": classification_scores(wy, pw > 0.5)}
    log(f"walk-forward AUC {wf_stats['auc']:.4f}, side accuracy {wf_stats['side_accuracy']:.4f}")
    wpred.to_csv(run / "walk_forward_predictions.csv", index=False, float_format="%.12g")
    cpred.to_csv(run / "cpcv_predictions.csv.gz", index=False, float_format="%.12g", compression="gzip")
    cmet.to_csv(run / "split_metrics.csv", index=False)
    # ---------------- 6-7 bets, exits, HRP account ----------------
    cost_side = a.cost_bp_per_side / 1e4
    dc = PCm.trading_days(PCm.daily_closes(bars), .8)
    rets = PCm.returns_within_gap(dc, MAX_GAP_DAYS[venue], "simple")
    rets = rets[rets.index >= dc.index[0]]
    first_event_day = int(ev["t0"].min() // DAY)
    results = {}
    log("walk-forward account path ...")
    results["wf"] = run_path(bars, closes, ev, w_idx, pw, pt, sl, cost_side)
    pidx_of = {"wf": w_idx}
    for pth, g in cpred.groupby("path"):
        log(f"CPCV path {pth} ...")
        results[f"cpcv{pth}"] = run_path(bars, closes, ev, g["event"].to_numpy(np.int64), g["p"].to_numpy(float), pt, sl, cost_side)
        pidx_of[f"cpcv{pth}"] = g["event"].to_numpy(np.int64)
    all_days = [d for (nets, _i, _b) in results.values() for (days, _n, _a) in nets.values() for d in (int(days.min()), int(days.max()))]
    last_day, first_day_all = max(all_days), min(all_days)
    sched = BF.weight_schedule(rets, min(first_day_all, first_event_day) + BF.REB_WINDOW, last_day, False)
    log(f"HRP: {len(sched)} rebalances")
    # section 2.4.1 item 1: the rebalance cost c_t on every rebalance date, taken out of the day's return
    reb_cost, reb_effect = {}, {}
    for scope, (nets, info_, (sd_, sz_, t1_)) in results.items():
        pos_day = {}
        for s, b_ in bars.items():
            sel = np.flatnonzero(ev["symbol"].to_numpy()[pidx_of[scope]] == s)
            if not len(sel):
                continue
            avg = EX.exposures(b_.n, ev["pos"].to_numpy(np.int64)[pidx_of[scope]][sel], t1_[sel], sd_[sel] * sz_[sel])
            pos_day[s] = EX.day_end_position(avg, b_.t, first_day_all, last_day)
        reb_cost[scope] = EX.rebalance_costs(pos_day, sched, cost_side, first_day_all, last_day)
    spans = {"last_365_days": last_day - 364, "whole_walk_forward_span": cutoff_day}
    rows, curves, series = [], [], {}
    for scope, (nets, info_, _b) in results.items():
        for sname, lo in spans.items():
            if scope != "wf" and sname == "whole_walk_forward_span":
                lo = first_day_all + BF.REB_WINDOW + 5
            r, act = BF.account(nets, sched, lo, last_day)
            r_before = r.copy()
            r = r - reb_cost[scope][lo - first_day_all:]
            if scope == "wf":
                reb_effect[sname] = {"gain_pct_before": BF.curve_stats(r_before)[1]["gain_pct"], "gain_pct_after": BF.curve_stats(r)[1]["gain_pct"],
                                     "total_charge_pct_of_capital": 100.0 * float(reb_cost[scope][lo - first_day_all:].sum()), "rebalances_charged": int((reb_cost[scope][lo - first_day_all:] > 0).sum())}
            eq, st = BF.curve_stats(r)
            series[(scope, sname)] = r
            st.update({"scope": scope, "span": sname if scope == "wf" else sname + ("" if sname == "last_365_days" else "_cpcv_full_history"),
                       "start_day": pd.Timestamp(lo * DAY, unit="s").strftime("%Y-%m-%d"), "end_day": pd.Timestamp(last_day * DAY, unit="s").strftime("%Y-%m-%d"),
                       "mean_instruments_with_position": float(act[act > 0].mean()) if (act > 0).any() else 0.0})
            rows.append(st)
            if scope == "wf":
                curves.append(pd.DataFrame({"day": np.arange(lo, last_day + 1), "ret": r, "equity": eq, "span": sname}))
    S = pd.DataFrame(rows)
    S.to_csv(run / "summary.csv", index=False, float_format="%.10g")
    pd.concat(curves, ignore_index=True).to_csv(run / "equity_curves_walk_forward.csv", index=False, float_format="%.12g")
    pd.concat([pd.Series(v, name=pd.Timestamp(k * DAY, unit="s").strftime("%Y-%m-%d")) for k, v in sched.items() if len(v)], axis=1).T.to_csv(run / "hrp_weights.csv", float_format="%.10g")
    # ---------------- 8 Ch.14 ----------------
    n_cand = int(tj["n_candidates"])
    dsr_out = {}
    for sname in spans:
        r_wf = series[("wf", sname)]
        mo = BS.sharpe_moments(r_wf)
        path_sr = [BS.sharpe_moments(series[(k, "last_365_days")])["sr"] for k in results if k != "wf"] if sname == "last_365_days" else []
        var_paths = float(np.var(path_sr, ddof=1)) if len(path_sr) > 1 else float("nan")
        var_asym = (1.0 - mo["skew"] * mo["sr"] + (mo["kurt"] - 1.0) / 4.0 * mo["sr"] ** 2) / max(mo["n"] - 1.0, 1.0) if np.isfinite(mo["sr"]) else float("nan")
        var_sr = float(np.nanmax([var_paths, var_asym])) if np.isfinite([var_paths, var_asym]).any() else float("nan")
        star = BS.expected_max_sharpe(n_cand, var_sr)
        dsr_out[sname] = {"n_days": mo["n"], "sr_daily": mo["sr"], "skew": mo["skew"], "kurt": mo["kurt"], "var_across_cpcv_paths": var_paths, "var_asymptotic": var_asym, "var_sr_used": var_sr,
                          "psr_vs_0": BS.psr(mo["sr"], mo["n"], mo["skew"], mo["kurt"], 0.0), "trials": n_cand, "sr_star": star,
                          "dsr": BS.psr(mo["sr"], mo["n"], mo["skew"], mo["kurt"], star) if np.isfinite(star) else float("nan")}
    nets_wf, _i, (side_w, size_w, t1_w) = results["wf"]
    ch14 = stage_ch14(ev, closes, w_idx, side_w, size_w, t1_w, sched, series[("wf", "last_365_days")], last_day, run)
    conc = stage_concentration(rets, sched, run)
    log(f"Eq.77-78 concentration: mean H {conc['H_mean']:.3f}")
    log("Ch.14 done")
    ch15 = stage_ch15(ev, closes, w_idx, side_w, t1_w, cost_side, last_day, run)
    log("Ch.15 done")
    ch10 = stage_ch10_orders(ev, w_idx, side_w, size_w, last_day, run)
    log("Snippet 10.4 orders done")
    cap = EX.capacity(bars, barvol, None, ev, w_idx, side_w, size_w, t1_w, sched, cutoff_day, last_day)
    (run / "capacity_volume.json").write_text(json.dumps(cap, indent=2), encoding="utf-8")
    log(f"section 2.4.1 volume rule: {cap['n_trades']:,} walk-forward trades, capacity min {cap.get('capacity_min', float('nan')):,.0f}, median {cap.get('capacity_median', float('nan')):,.0f}")
    after = RB.integrity_snapshot()
    meta = {"venue": venue, "bar_dir": str(bars_dir), "seed": seed, "cost_bp_per_side": a.cost_bp_per_side, "cutoff": CUTOFF, "embargo_pct": EMBARGO, "step": STEP, "base_forest": BASE_FOREST, "forest": forest,
            "features": FEATURES, "symbols": len(bars), "events": int(len(ev)), "events_before_dropLabels": int(n_all), "label_shares_before_dropLabels": {str(k): v for k, v in shares.items()},
            "labels_dropped_3_8": [int(x) for x in labels_dropped], "time_decay_4_11": {"c": DECAY_C, "const": float(dconst), "slope": float(dslope)}, "events_pre_cutoff": int(len(tr)), "events_cusum": einfo["events_cusum"],
            "universe": a.universe, "otr": {"pt": pt, "sl": sl, "K": K_OTR}, "cpcv": {"n_paths": int(cmeta["n_paths"]), "auc_by_path": auc_cp, "auc_mean": float(np.mean(auc_cp))}, "walk_forward": wf_stats,
            "exit_stats_wf": results["wf"][1], "hrp_rebalances": len(sched), "reb_window": BF.REB_WINDOW, "first_day": pd.Timestamp(first_day_all * DAY, unit="s").strftime("%Y-%m-%d"),
            "last_day": pd.Timestamp(last_day * DAY, unit="s").strftime("%Y-%m-%d"), "dsr": dsr_out, "ch14": ch14, "ch15": ch15, "ch10_orders": ch10, "rebalance_cost_effect": reb_effect, "concentration_eq77_78": conc, "volume_capacity": cap, "tuned": tj, "integrity_unchanged": before == after,
            "quick": bool(a.quick), "seconds": round(time.time() - t_start), "run_utc": started.isoformat(timespec="seconds"),
            "script_md5": {n: RB.md5_file(HERE / n) for n in ("nat_book_bot.py", "nat_book_events_v2.py", "nat_vol.py", "book_events_v2.py", "book_vol_printed.py", "book_micro.py", "book_final.py", "book_otr_exit.py", "book_port_a.py", "book_port_b.py", "book_port_c.py", "book_port_d.py", "book_port_e.py",
                                                              "bet_averaging.py", "book_etf.py", "meta_model.py", "meta_eval.py", "bar_events.py", "bar_loader.py", "micro_features.py")}}
    (run / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log(f"done in {time.time() - t_start:.0f}s -> {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
