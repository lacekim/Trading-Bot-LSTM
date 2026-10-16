"""Three algorithms (long, short, momentum) and a gate that picks one per event -- the book's meta-labeling machinery applied to a
selection problem.  Reads a finished, independently verified meta-pipeline run (its events.csv) and adds columns; no candle-based
feature or label is rebuilt here except three trailing returns per event.  No project imports besides the study modules.

DESIGN (fixed before any result of this module was seen; the same text is printed in SUMMARY.md):

  Events         the 123,768-style event table of the source run: CUSUM events, triple-barrier exit (symmetric +/-2 daily-vol barriers on the
                 simple return, vertical 101 bars).  Because the barriers are symmetric, a short held over the same bars exits at the same
                 bar as a long with the opposite sign: ret_long = side * ret_side (side = sign of the 30-day return in the source run).
  Algorithms     LONG  = side +1 at every event.
                 SHORT = side -1 at every event.
                 MOM   = side +1 if the trailing 20-, 30- and 40-day (480, 720, 960 bar) log returns are ALL > 0, side -1 if all are < 0,
                         otherwise no signal (also none when the symbol has fewer than 960 bars before the event).
  Gate models    F_dir: the book's bagged forest on label y_dir = 1 when ret_long > 0 (gross), 13 features (the source 14 minus 'side'),
                 trained on every training event.  A forest for SHORT on the mirrored label would be the same model, so p(SHORT) = 1 - p(LONG).
                 F_mom: the same forest on the MOM events only, label 'bin' of the source run (the MOM side's gross return > 0), 14 features
                 (the side feature is the MOM side).
  Gate           at each event the candidates are LONG with confidence p_dir, SHORT with 1 - p_dir and, when MOM has a signal, MOM with its
                 side and confidence p_mom.  The candidate with the largest confidence is picked (ties: MOM, then LONG, then SHORT); the bet
                 is taken only if that confidence exceeds 1/2, with size from Snippets 10.1 / 10.3 (step 0.1).
  Variants       long_all, short_all, mom_all (size 1 on all MOM events), mom_long_up (a fixed rule: MOM side +1 while the equal-weight index is
                 above its own 200-day mean, size 1, nothing otherwise), dir_model (side by p_dir, size by its confidence), mom_meta (MOM events
                 sized by p_mom), gate; two nulls for the gate: gate_shuffled_size (the gate's sides, its sizes shuffled among all events) and
                 gate_random_side (the gate's sizes with coin-flip sides).
  Evaluation     as in meta_eval: one unit of capital per candidate event, daily result = sum of size * (side * ret_long - cost * [size > 0])
                 over the day's events divided by ALL candidate events that day, in bp; CPCV N=6, k=2 (15 splits, 5 paths); a walk-forward fit
                 on labels ending before 2024-01-01; June 2026, the sell-off and the deepest-fall windows; by market state.
  Trials         5 new variants carry a choice made here (mom_all's unanimity rule, mom_long_up, dir_model, mom_meta, gate); the deflated
                 Sharpe uses prior_trials = 83 (the source run's count after it) + 4 = 87, and shows N = 1, 10, 87 and 88 (the gate itself).
  Not claimed    mom_all and mom_long_up use rules chosen after seeing this history (the lookbacks and the state came from the earlier
                 diagnostics), so they are in-sample by construction; only the deflated Sharpe tempers that.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import book_stats as B
import meta_eval as ME
import meta_events as E
import meta_model as MM

MOM_LOOKBACKS = (480, 720, 960)             # 20, 30, 40 days of hourly bars
DIR_FEATURES = [f for f in E.FEATURES if f != "side"]
MOM_FEATURES = list(E.FEATURES)
PRIOR_TRIALS = 87
ALGO_CODE = {"none": 0, "LONG": 1, "SHORT": 2, "MOM": 3}
ALGO_NAME = {v: k for k, v in ALGO_CODE.items()}
VARIANTS = ("long_all", "short_all", "mom_all", "mom_long_up", "dir_model", "mom_meta", "gate", "gate_shuffled_size", "gate_random_side")
DSR_VARIANTS = ("mom_long_up", "dir_model", "mom_meta", "gate")

DEVIATIONS = [
    "The book has no neural network. The 'which algorithm' decision is made by the book's own bagged forests (secondary models); the neural-network overseer of the intended architecture is not built here.",
    "LONG and SHORT are the two signs of one decision; a forest on the long label and one on the mirrored short label would be the same model, so one forest (F_dir) is fitted and p(SHORT) = 1 - p(LONG).",
    "The gate compares confidences of two different forests (F_dir on all events, F_mom on the MOM events). Both use balanced class weights, which centres their probabilities near 1/2 whatever the base rate, but the confidences are not guaranteed to be comparable; the run reports each forest and each algorithm separately so the gate can be judged against them.",
    "F_mom is trained on the MOM events only but with the source run's uniqueness and attribution weights (computed over all events, so they treat the non-MOM events as concurrent); sequential bootstrap and sample weights are otherwise as in the source run.",
    "The lookbacks (20/30/40 days) and the 200-day state come from earlier diagnostics on this same history, so MOM and mom_long_up are in-sample choices. Cross-validation cannot remove that; it only removes the fitting of the forests.",
    "Long and short exits are identical by construction (symmetric barriers on the simple return); no separate short-side barrier study. The round-trip cost of 24 bp is charged on every taken bet; no funding or borrow cost for shorts, and no capacity or liquidity limits.",
    "All source-run deviations apply (time bars not volume bars, hourly candles, FFD choices, SADF / entropy / Corwin-Schultz choices, survivorship in the symbol list).",
]


# ---------------------------------------------------------------------------------------------------------------------
# new columns
# ---------------------------------------------------------------------------------------------------------------------
def mom_side_from(r480, r720, r960):
    """+1 if all three trailing log returns are > 0, -1 if all are < 0, else 0 (NaN anywhere gives 0)."""
    r480, r720, r960 = (np.asarray(x, dtype=np.float64) for x in (r480, r720, r960))
    up = (r480 > 0) & (r720 > 0) & (r960 > 0)
    dn = (r480 < 0) & (r720 < 0) & (r960 < 0)
    return np.where(up, 1, np.where(dn, -1, 0)).astype(np.int8)


def extend_events(df: pd.DataFrame, closes: dict) -> pd.DataFrame:
    """Adds ret480, ret720, ret960 (log return over that many bars ending at the event bar; NaN when the symbol has fewer bars), mom_side,
    ret_long = side * ret_side and y_dir = 1[ret_long > 0].  `closes[symbol]` must be the series (closes > 0) the source run used, cut at
    its last recorded bar, so that df['pos'] indexes it."""
    out = df.copy()
    n = len(out)
    cols = {L: np.full(n, np.nan) for L in MOM_LOOKBACKS}
    for sym, grp in out.groupby("symbol", sort=False):
        c = closes[sym].astype(float)
        c = c[c > 0]
        logc = np.log(c.to_numpy())
        ix = grp.index.to_numpy()
        p = grp["pos"].to_numpy(np.int64)
        if p.max() >= len(logc):
            raise ValueError(f"{sym}: event position beyond the candle series")
        for L in MOM_LOOKBACKS:
            ok = p >= L
            v = np.full(len(p), np.nan)
            v[ok] = logc[p[ok]] - logc[p[ok] - L]
            cols[L][ix] = v
    out["ret480"], out["ret720"], out["ret960"] = cols[480], cols[720], cols[960]
    out["mom_side"] = mom_side_from(out["ret480"], out["ret720"], out["ret960"])
    out["ret_long"] = out["side"].to_numpy(np.float64) * out["ret_side"].to_numpy(np.float64)
    out["y_dir"] = (out["ret_long"] > 0).astype(np.int8)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# forests: CPCV and walk-forward
# ---------------------------------------------------------------------------------------------------------------------
def _arrays(df):
    return dict(w=df["w"].to_numpy(np.float64), sym=pd.factorize(df["symbol"])[0].astype(np.int64), starts=df["pos"].to_numpy(np.int64),
                ends=df["t1pos"].to_numpy(np.int64), t0=df["t0"].to_numpy(np.float64), t1=df["t1"].to_numpy(np.float64), u=df["u"].to_numpy(np.float64),
                Xd=df[DIR_FEATURES].to_numpy(np.float64), Xm=df[MOM_FEATURES].to_numpy(np.float64), yd=df["y_dir"].to_numpy(np.int64),
                ym=df["bin"].to_numpy(np.int64), active=(df["mom_side"].to_numpy() != 0))


def _fit(idx, A, X, y, forest_kw, seed):
    f = MM.BaggedForest(seed=seed, **forest_kw)
    f.fit(X[idx], y[idx], A["w"][idx], A["sym"][idx], A["starts"][idx], A["ends"][idx], float(A["u"][idx].mean()))
    return f


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _split_row(si, tag, tg, tr_d, tr_m, te_idx, A, fd, fm, p_d, p_m):
    act = A["active"][te_idx]
    row = {"split": si, "test_groups": tag, "n_train_dir": int(len(tr_d)), "n_train_mom": int(len(tr_m)), "n_test": int(len(te_idx)), "n_test_mom": int(act.sum()),
           "draws_dir": int(fd.draws_info["draws_per_tree"]), "draws_mom": int(fm.draws_info["draws_per_tree"]),
           "avg_u_dir": float(fd.draws_info["avg_uniqueness"]), "avg_u_mom": float(fm.draws_info["avg_uniqueness"]),
           "auc_dir": _auc(A["yd"][te_idx], p_d), "auc_mom": _auc(A["ym"][te_idx][act], p_m[act]),
           "base_dir": float(A["yd"][te_idx].mean()), "base_mom": float(A["ym"][te_idx][act].mean()) if act.any() else float("nan"),
           "share_dir_gt_half": float((p_d > 0.5).mean()), "share_mom_gt_half": float((p_m[act] > 0.5).mean()) if act.any() else float("nan")}
    return row


def run_cpcv_gate(df, n_groups, k, embargo_pct, forest_kw, seed, log=print):
    A = _arrays(df)
    t0, t1 = A["t0"], A["t1"]
    edges = B.group_edges(float(t0.min()), float(t0.max()) + 1.0, n_groups)
    groups = B.assign_groups(t0, edges)
    embargo = embargo_pct * (float(t0.max()) + 1.0 - float(t0.min()))
    splits = B.cpcv_splits(n_groups, k)
    assign, n_paths = B.cpcv_paths(n_groups, k)
    parts, rows = [], []
    t_start = time.time()
    for si, tg in enumerate(splits):
        train = B.train_mask(t0, t1, groups, tg, edges, embargo)
        test = np.isin(groups, list(tg))
        tr_d, te_idx = np.flatnonzero(train), np.flatnonzero(test)
        tr_m = np.flatnonzero(train & A["active"])
        fd = _fit(tr_d, A, A["Xd"], A["yd"], forest_kw, seed + si)
        fm = _fit(tr_m, A, A["Xm"], A["ym"], forest_kw, seed + 1000 + si)
        p_d, p_m = fd.predict_proba1(A["Xd"][te_idx]), fm.predict_proba1(A["Xm"][te_idx])
        path = np.array([assign[(si, int(g))] for g in groups[te_idx]], dtype=np.int64)
        parts.append(pd.DataFrame({"event": te_idx, "split": si, "path": path, "p_dir": p_d, "p_mom": p_m}))
        rows.append(_split_row(si, "-".join(map(str, tg)), tg, tr_d, tr_m, te_idx, A, fd, fm, p_d, p_m))
        log(f"  split {si + 1}/{len(splits)}: train {len(tr_d):,} (mom {len(tr_m):,}) test {len(te_idx):,}  AUC dir {rows[-1]['auc_dir']:.3f} mom {rows[-1]['auc_mom']:.3f}  ({time.time() - t_start:.0f}s)")
    meta = {"edges": [float(e) for e in edges], "embargo_seconds": float(embargo), "n_groups": n_groups, "k": k, "n_splits": len(splits), "n_paths": n_paths}
    return pd.concat(parts, ignore_index=True), pd.DataFrame(rows), meta


def run_walk_forward_gate(df, cutoff, forest_kw, seed, log=print):
    A = _arrays(df)
    cut = float(pd.Timestamp(cutoff).timestamp())
    tr_d = np.flatnonzero(A["t1"] < cut)
    te_idx = np.flatnonzero(A["t0"] >= cut)
    tr_m = tr_d[A["active"][tr_d]]
    fd = _fit(tr_d, A, A["Xd"], A["yd"], forest_kw, seed)
    fm = _fit(tr_m, A, A["Xm"], A["ym"], forest_kw, seed + 1000)
    p_d, p_m = fd.predict_proba1(A["Xd"][te_idx]), fm.predict_proba1(A["Xm"][te_idx])
    row = _split_row(-1, "walk-forward", None, tr_d, tr_m, te_idx, A, fd, fm, p_d, p_m)
    row["cutoff"] = str(pd.Timestamp(cutoff))
    log(f"  walk-forward: train {len(tr_d):,} (mom {len(tr_m):,}), test {len(te_idx):,}, AUC dir {row['auc_dir']:.3f} mom {row['auc_mom']:.3f}")
    return pd.DataFrame({"event": te_idx, "split": -1, "path": 0, "p_dir": p_d, "p_mom": p_m}), row


# ---------------------------------------------------------------------------------------------------------------------
# gate and variants
# ---------------------------------------------------------------------------------------------------------------------
def gate_choice(p_dir, p_mom, mom_side):
    """(side, algo code, confidence) per event.  Candidates in tie-break order: MOM (only with a signal), LONG, SHORT."""
    p_dir, p_mom, mom_side = np.asarray(p_dir, np.float64), np.asarray(p_mom, np.float64), np.asarray(mom_side)
    conf = np.vstack([np.where(mom_side != 0, p_mom, -np.inf), p_dir, 1.0 - p_dir])          # rows: MOM, LONG, SHORT
    pick = conf.argmax(axis=0)                                                                  # first maximum wins ties
    best = conf[pick, np.arange(len(p_dir))]
    side = np.where(pick == 0, mom_side, np.where(pick == 1, 1, -1)).astype(np.int64)
    algo = np.where(pick == 0, ALGO_CODE["MOM"], np.where(pick == 1, ALGO_CODE["LONG"], ALGO_CODE["SHORT"])).astype(np.int8)
    return side, algo, best


def build_variants(p_dir, p_mom, mom_side, state, step, rng):
    """{variant: (side, size, algo)} for the events of one path (arrays aligned).  algo is the code of the algorithm credited with each event
    (only meaningful where size > 0).  The two nulls draw from `rng` (shuffled sizes; coin-flip sides)."""
    n = len(p_dir)
    ones = np.ones(n)
    act = mom_side != 0
    d_side = np.where(p_dir > 0.5, 1, -1)
    d_size = MM.bet_size(np.maximum(p_dir, 1.0 - p_dir), step)
    g_side, g_algo, g_conf = gate_choice(p_dir, p_mom, mom_side)
    g_size = MM.bet_size(g_conf, step)
    out = {
        "long_all": (np.ones(n, np.int64), ones, np.full(n, ALGO_CODE["LONG"], np.int8)),
        "short_all": (-np.ones(n, np.int64), ones, np.full(n, ALGO_CODE["SHORT"], np.int8)),
        "mom_all": (mom_side.astype(np.int64), act.astype(float), np.full(n, ALGO_CODE["MOM"], np.int8)),
        "mom_long_up": (np.ones(n, np.int64), ((mom_side == 1) & (state == 1.0)).astype(float), np.full(n, ALGO_CODE["MOM"], np.int8)),
        "dir_model": (d_side.astype(np.int64), d_size, np.where(d_side > 0, ALGO_CODE["LONG"], ALGO_CODE["SHORT"]).astype(np.int8)),
        "mom_meta": (mom_side.astype(np.int64), np.where(act, MM.bet_size(p_mom, step), 0.0), np.full(n, ALGO_CODE["MOM"], np.int8)),
        "gate": (g_side, g_size, g_algo),
        "gate_shuffled_size": (g_side, rng.permutation(g_size), g_algo),
        "gate_random_side": (rng.choice(np.array([-1, 1]), size=n).astype(np.int64), g_size, g_algo),
    }
    return out


def _daily(day, num):
    return pd.DataFrame({"day": day, "num": num}).groupby("day").agg(sum_bp=("num", "sum"), n=("num", "size")).reset_index()


def best_days_share(daily_mean, k=20):
    """Share of the total of a daily series that comes from its k best days (NaN when the total is not positive)."""
    x = np.asarray(daily_mean, dtype=np.float64)
    tot = x.sum()
    if not tot > 0 or len(x) == 0:
        return float("nan")
    return float(np.sort(x)[-k:].sum() / tot)


def evaluate_three(df, preds, cost_frac, step, h, seed, windows, prior_trials):
    ret_long = df["ret_long"].to_numpy(np.float64)
    day_all = df["t0"].to_numpy(np.int64) // 86400
    t0 = df["t0"].to_numpy(np.int64)
    state_all = df["mkt_state"].to_numpy(np.float64)
    mom_all_side = df["mom_side"].to_numpy(np.int64)
    rng = np.random.default_rng(seed + 99)
    t_eff_div = max(1.0, (h + 1) / 24.0)
    path_rows, daily_parts = [], []
    win_acc = {(v, wn): [0.0, 0] for v in VARIANTS for wn in windows}
    state_acc = {(v, s): [0.0, 0, 0.0] for v in VARIANTS for s in (0, 1)}
    algo_state = {}                      # (variant in gate / dir_model, algo, state) -> [pnl sum, trades]
    algo_win = {}
    cand_state = {s: 0 for s in (0, 1)}
    cand_win = {wn: 0 for wn in windows}
    for path, g in preds.groupby("path"):
        idx = g["event"].to_numpy(np.int64)
        va = build_variants(g["p_dir"].to_numpy(np.float64), g["p_mom"].to_numpy(np.float64), mom_all_side[idx], state_all[idx], step, rng)
        for s_ in (0, 1):
            cand_state[s_] += int((state_all[idx] == s_).sum())
        for wn, (lo_t, hi_t) in windows.items():
            cand_win[wn] += int(((t0[idx] >= lo_t) & (t0[idx] < hi_t)).sum())
        for vname in VARIANTS:
            side, size, algo = va[vname]
            pnl = size * (side * ret_long[idx] - cost_frac * (size > 0)) * 1e4
            d = _daily(day_all[idx], pnl)
            d.insert(0, "variant", vname)
            d.insert(0, "path", int(path))
            daily_parts.append(d)
            for s_ in (0, 1):
                m = state_all[idx] == s_
                a = state_acc[(vname, s_)]
                a[0] += float(pnl[m].sum()); a[1] += int(m.sum()); a[2] += float(size[m].sum())
            for wn, (lo_t, hi_t) in windows.items():
                m = (t0[idx] >= lo_t) & (t0[idx] < hi_t)
                win_acc[(vname, wn)][0] += float(pnl[m].sum()); win_acc[(vname, wn)][1] += int(m.sum())
            if vname in ("gate", "dir_model"):
                taken = size > 0
                for code in (1, 2, 3):
                    ma = taken & (algo == code)
                    for s_ in (0, 1):
                        m = ma & (state_all[idx] == s_)
                        a = algo_state.setdefault((vname, ALGO_NAME[code], s_), [0.0, 0])
                        a[0] += float(pnl[m].sum()); a[1] += int(m.sum())
                    for wn, (lo_t, hi_t) in windows.items():
                        m = ma & (t0[idx] >= lo_t) & (t0[idx] < hi_t)
                        a = algo_win.setdefault((vname, ALGO_NAME[code], wn), [0.0, 0])
                        a[0] += float(pnl[m].sum()); a[1] += int(m.sum())
            dm = (d["sum_bp"] / d["n"]).to_numpy()
            mom = B.sharpe_moments(dm)
            t_eff = mom["n"] / t_eff_div
            path_rows.append({"path": int(path), "variant": vname, "events": int(len(idx)), "days": int(len(d)), "mean_bp_per_candidate": float(pnl.mean()),
                              "sr_daily": mom["sr"], "sr_annual": mom["sr"] * math.sqrt(365) if np.isfinite(mom["sr"]) else float("nan"), "skew": mom["skew"], "kurt": mom["kurt"],
                              "t_eff": t_eff, "psr": B.psr(mom["sr"], t_eff, mom["skew"], mom["kurt"]) if np.isfinite(mom["sr"]) else float("nan"),
                              "taken_share": float((size > 0).mean()), "mean_size": float(size.mean()), "best20_share": best_days_share(dm)})
    daily = pd.concat(daily_parts, ignore_index=True)
    pp = pd.DataFrame(path_rows)
    n_paths = int(pp["path"].nunique())
    summary = {}
    for v, gdf in pp.groupby("variant"):
        summary[v] = {"mean_bp_per_candidate": float(gdf["mean_bp_per_candidate"].mean()), "mean_sr_daily": float(gdf["sr_daily"].mean()),
                      "mean_sr_annual": float(gdf["sr_annual"].mean()), "paths_sr_negative": int((gdf["sr_daily"] < 0).sum()), "mean_psr": float(gdf["psr"].mean()),
                      "taken_share": float(gdf["taken_share"].mean()), "mean_size": float(gdf["mean_size"].mean()), "best20_share": float(np.nanmean(gdf["best20_share"])) if gdf["best20_share"].notna().any() else float("nan"),
                      "paths": int(len(gdf))}
    piv = pp.pivot(index="path", columns="variant", values="sr_daily")
    pivb = pp.pivot(index="path", columns="variant", values="mean_bp_per_candidate")
    compare = {}
    for a_, b_ in (("gate", "gate_shuffled_size"), ("gate", "gate_random_side"), ("gate", "mom_all"), ("gate", "long_all"), ("gate", "dir_model"), ("gate", "mom_meta"), ("gate", "mom_long_up"),
                   ("mom_meta", "mom_all"), ("dir_model", "long_all")):
        compare[f"{a_}|{b_}"] = {"paths_sr_higher": int((piv[a_] > piv[b_]).sum()), "paths": int(len(piv)), "mean_sr_daily_diff": float((piv[a_] - piv[b_]).mean()),
                                 "mean_bp_diff": float((pivb[a_] - pivb[b_]).mean())}
    win_out = {f"{v}|{wn}": {"events_per_path": cnt / n_paths, "mean_bp_per_candidate": (tot / cnt) if cnt else float("nan")} for (v, wn), (tot, cnt) in win_acc.items()}
    state_out = {f"{v}|state{s_}": {"events_per_path": cnt / n_paths, "mean_bp_per_candidate": (tot / cnt) if cnt else float("nan"), "mean_size": (sz / cnt) if cnt else float("nan")}
                 for (v, s_), (tot, cnt, sz) in state_acc.items()}
    algo_state_out = {f"{v}|{al}|state{s_}": {"trades_per_path": nt / n_paths, "bp_per_state_candidate": (tot / cand_state[s_]) if cand_state[s_] else float("nan"),
                                              "bp_per_trade": (tot / nt) if nt else float("nan")} for (v, al, s_), (tot, nt) in algo_state.items()}
    algo_win_out = {f"{v}|{al}|{wn}": {"trades_per_path": nt / n_paths, "bp_per_window_candidate": (tot / cand_win[wn]) if cand_win[wn] else float("nan"),
                                       "bp_per_trade": (tot / nt) if nt else float("nan")} for (v, al, wn), (tot, nt) in algo_win.items()}
    dsr = {}
    for v in DSR_VARIANTS:
        m = pp[pp["variant"] == v]
        sr_hat = float(m["sr_daily"].mean())
        var_paths = float(m["sr_daily"].var(ddof=1)) if len(m) > 1 else float("nan")
        t_eff, skew, kurt = float(m["t_eff"].mean()), float(m["skew"].mean()), float(m["kurt"].mean())
        var_asym = (1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2) / max(t_eff - 1.0, 1.0) if np.isfinite(sr_hat) else float("nan")
        var_sr = float(np.nanmax([var_paths, var_asym])) if np.isfinite([var_paths, var_asym]).any() else float("nan")
        tab = {}
        for n_tr in sorted({1, 10, int(prior_trials), int(prior_trials) + 1}):
            if n_tr <= 1:
                tab[str(n_tr)] = {"sr_star": 0.0, "dsr": B.psr(sr_hat, t_eff, skew, kurt, 0.0)}
            else:
                star = B.expected_max_sharpe(n_tr, var_sr) if np.isfinite(var_sr) else float("nan")
                tab[str(n_tr)] = {"sr_star": star, "dsr": B.psr(sr_hat, t_eff, skew, kurt, star) if np.isfinite(star) else float("nan")}
        dsr[v] = {"sr_daily_mean": sr_hat, "var_sr_used": var_sr, "var_across_paths": var_paths, "var_asymptotic": var_asym, "t_eff": t_eff, "skew": skew, "kurt": kurt, "by_trials": tab}
    return {"paths": path_rows, "daily": daily, "summary": summary, "compare": compare, "windows": win_out, "by_state": state_out, "algo_state": algo_state_out,
            "algo_window": algo_win_out, "candidates_state": {f"state{s_}": c / n_paths for s_, c in cand_state.items()},
            "candidates_window": {wn: c / n_paths for wn, c in cand_win.items()}, "dsr": dsr}
