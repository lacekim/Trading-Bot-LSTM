"""DAILY-bar copy of meta_eval.py for the equities run (equity_events.py): identical scoring code, with the annualisation (252 trading days instead of 365),
the effective-sample divisor (the holding period in trading days instead of hours / 24) and the deviation list changed.

Scoring of the meta-labeling model: combinatorial purged cross-validation (Ch.12) with the forest of meta_model.py, a single
walk-forward fit as a time-ordered check, bet sizes (Ch.10), daily results per backtest path, PSR / DSR (Ch.14), window lines.

Evaluation convention (same as the earlier book_study runs): one unit of capital per candidate event.  The primary strategy takes every
event (size 1); the meta strategy takes event i with size s_i from the model's probability; the placebo shuffles the meta sizes among the
same events.  A day's result = sum over its events of s_i * (side-signed return - round-trip cost if s_i > 0), divided by the number of ALL
candidate events entered that day (unflagged ones count as zero), in basis points.  Bets are not averaged bar by bar (Snippet 10.2 is not
reproduced); see DEVIATIONS.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import book_stats as B
import equity_events as E
import meta_model as MM

ANNUAL_DAYS = 252             # trading days per year for annualised Sharpe ratios (the hourly crypto runs use 365)

DEVIATIONS = [
    "Daily time bars (Ch.2): Alpaca daily bars are used as they are (adjusted close and volume only); volume or dollar bars were not built.",
    "Events are the book's symmetric CUSUM filter on daily bars, threshold one causal daily sigma (EWM std, span 14 days; the hourly runs used 336 hours).",
    "Primary side = sign of the project's base momentum score at the signal bar, 0.5 x (12-1 month return) + 0.5 x (6-1 month return) with 252 / 126 / 21-day lookbacks (tools/run_alpaca_momentum_test.py), in every market state. The project's strategy is long-only; here shorts are scored too, with no borrow cost, locate or short-sale restriction modelled. The state is a model feature, not a filter.",
    "Entry at the NEXT bar's close, as the project's momentum test does; the signal, the barrier width and every feature are read at the signal bar (the close before entry). A feature window can therefore end one day before the label starts; the 1% embargo is far longer.",
    "Barriers (Ch.3): simple-return path times side from the entry close; horizontal barriers at 2 x the book's daily-volatility estimate (Snippet 3.1 on 1-day returns, span 100), vertical barrier 5 bars after entry (H = 4 plus one). The 5 bars were fixed before any result so that the barrier-to-horizon ratio stays near the hourly runs (101 hours is about 4.2 days); no search over them.",
    "Meta-label = 1 if the side's return at the exit bar is > 0 (gross, as in Snippet 3.7); profit and loss are scored net of a 20 bp round-trip cost (10 bp per side, the project's momentum-test cost).",
    "Fractional differentiation (Ch.5): per symbol, the smallest d on a 0.05 grid whose FFD (threshold 0.01) of the daily log close passes ADF (lag 1, 5% level, response-surface critical value). d is chosen on each symbol's whole history (mild look-ahead on a stationarity statistic, not on returns).",
    "Sequential bootstrap (Ch.4) is run per instrument (events of different instruments never share bars) and each tree draws at most max_draws indices in total (max_samples = average uniqueness, capped). Trees are entropy decision trees with balanced class weights and min_weight_fraction_leaf = 0.05 inside a hand-written bagging loop (Snippet 6.2, clf1).",
    "Sample weights = absolute return attribution (Snippet 4.10), scaled to mean 1; the book's time-decay (Snippet 4.11) is not used.",
    "Cross-validation: combinatorial purged CV with N groups and k test groups (Ch.12); purging and embargo as in the earlier studies (embargo = 1% of the time span after each test group).",
    "Bet size (Ch.10): Snippet 10.1 for two classes, discretised in steps (Snippet 10.3). The bar-by-bar averaging of active bets (Snippet 10.2) is not reproduced: each event is its own unit of capital, as in the earlier daily-result convention.",
    "MDA is run with the same forest; SFI with ordinary bootstrap and fewer trees for speed. Importance is reported only; features were fixed beforehand and are not selected from it.",
    "Chapter 17-18 features: SADF (Snippets 17.1-17.4) as the supremum ADF statistic (constant, one lagged difference) over start points of a trailing 100-day window of daily log closes with at least 20 days; entropy (Ch.18) of the sign of the last 100 daily returns, plug-in with word length 2 (Snippet 18.1) and the Kontoyiannis estimate (Snippets 18.3, 18.4). Read at the signal bar's own close.",
    "Corwin-Schultz spread (Snippet 19.1) is NOT built: the Alpaca daily file keeps no high or low. Not built either: Kyle, Amihud and Hasbrouck lambdas, VPIN, PIN, volume-based structural-break features (volume exists in the file; not attempted in this run), HRP allocation (Ch.16), the Chapter 20-22 material.",
    "Universe (the project's own screen, tools/run_alpaca_momentum_test.py, with N reduced): at each COMPLETED month-end, adjusted close > 5, volume > 0 that day, at least 253 bars of history and at least 40 valid dollar-volume days in the last 60, ranked by trailing-60-day median dollar volume; the top N are members for signals on later days (N = 200 here, 1000 in the project, so that every event can be rebuilt independently). Members exist from the first month-end with at least 200 eligible names. Prices are dividend- and split-adjusted.",
    "SURVIVORSHIP: the stock list is Alpaca's current active, tradable assets, filtered by name; companies that were delisted, went bankrupt or were acquired are almost certainly missing. This flatters long momentum in particular; every figure here is an optimistic bound on that count.",
    "A path is kept only if every trading day of the market calendar between the signal bar and the last path bar has a bar for the symbol (the daily analogue of the hourly runs' gap-free window). A label needs 5 bars after entry, so events entered in the last 5 trading days of the file cannot be labelled.",
    "Annualised Sharpe = daily Sharpe x sqrt(252), where a 'day' is a day with at least one event; the effective sample for PSR / DSR is the number of such days divided by the holding period in trading days (5).",
]


def prepare(df: pd.DataFrame):
    X = df[E.FEATURES].to_numpy(dtype=np.float64)
    y = df["bin"].to_numpy(dtype=np.int64)
    w = df["w"].to_numpy(dtype=np.float64)
    sym = pd.factorize(df["symbol"])[0].astype(np.int64)
    return X, y, w, sym, df["pos"].to_numpy(np.int64), df["t1pos"].to_numpy(np.int64), df["t0"].to_numpy(np.float64), df["t1"].to_numpy(np.float64), df["u"].to_numpy()


def fit_forest(idx, data, forest_kw, seed):
    X, y, w, sym, starts, ends, t0, t1, u = data
    f = MM.BaggedForest(seed=seed, **forest_kw)
    f.fit(X[idx], y[idx], w[idx], sym[idx], starts[idx], ends[idx], float(u[idx].mean()))
    return f


def split_metrics(y, p, w):
    out = {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
           "log_loss_w": MM.weighted_log_loss(y, p, w), "base_rate": float(y.mean()), "share_p_gt_half": float((p > 0.5).mean()),
           "acc": float(((p > 0.5).astype(int) == y).mean())}
    return out


def run_cpcv(df, n_groups, k, embargo_pct, forest_kw, seed, log=print):
    data = prepare(df)
    X, y, w, sym, starts, ends, t0, t1, u = data
    edges = B.group_edges(float(t0.min()), float(t0.max()) + 1.0, n_groups)
    groups = B.assign_groups(t0, edges)
    embargo = embargo_pct * (float(t0.max()) + 1.0 - float(t0.min()))
    splits = B.cpcv_splits(n_groups, k)
    assign, n_paths = B.cpcv_paths(n_groups, k)
    pred_parts, metrics = [], []
    t_start = time.time()
    for si, tg in enumerate(splits):
        train = B.train_mask(t0, t1, groups, tg, edges, embargo)
        test = np.isin(groups, list(tg))
        tr_idx, te_idx = np.flatnonzero(train), np.flatnonzero(test)
        f = fit_forest(tr_idx, data, forest_kw, seed + si)
        p = f.predict_proba1(X[te_idx])
        path = np.array([assign[(si, int(g))] for g in groups[te_idx]], dtype=np.int64)
        pred_parts.append(pd.DataFrame({"event": te_idx, "split": si, "path": path, "p": p}))
        m = {"split": si, "test_groups": "-".join(map(str, tg)), "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)), **f.draws_info,
             **split_metrics(y[te_idx], p, w[te_idx])}
        metrics.append(m)
        log(f"  split {si + 1}/{len(splits)}: train {len(tr_idx):,} test {len(te_idx):,} AUC {m['auc']:.3f}  ({time.time() - t_start:.0f}s)")
    meta = {"edges": [float(e) for e in edges], "embargo_seconds": float(embargo), "n_groups": n_groups, "k": k, "n_splits": len(splits), "n_paths": n_paths}
    return pd.concat(pred_parts, ignore_index=True), pd.DataFrame(metrics), meta


def run_walk_forward(df, cutoff, forest_kw, seed, log=print):
    data = prepare(df)
    X, y, w, sym, starts, ends, t0, t1, u = data
    cut = float(pd.Timestamp(cutoff).timestamp())
    tr_idx = np.flatnonzero(t1 < cut)
    te_idx = np.flatnonzero(t0 >= cut)
    f = fit_forest(tr_idx, data, forest_kw, seed)
    p = f.predict_proba1(X[te_idx])
    m = {"cutoff": str(pd.Timestamp(cutoff)), "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)), **f.draws_info, **split_metrics(y[te_idx], p, w[te_idx])}
    log(f"  walk-forward: train {len(tr_idx):,} (labels end before {m['cutoff']}), test {len(te_idx):,}, AUC {m['auc']:.3f}")
    return pd.DataFrame({"event": te_idx, "split": -1, "path": 0, "p": p}), m


def _daily(day, num, cnt):
    d = pd.DataFrame({"day": day, "num": num}).groupby("day").agg(sum_bp=("num", "sum"), n=("num", "size")).reset_index()
    return d


def evaluate(df, preds, cost_frac, step, h, seed, windows, prior_trials, horizon_days=None):
    """Per-path results for primary / meta / placebo. Returns dict with 'paths' (list of dict), 'daily' (DataFrame), 'windows', 'dsr'."""
    ret = df["ret_side"].to_numpy(np.float64)
    day_all = (df["t0"].to_numpy(np.int64) // 86400)
    t0 = df["t0"].to_numpy(np.int64)
    rng = np.random.default_rng(seed + 99)
    out_paths, daily_parts = [], []
    win_acc = {(v, wn): [] for v in ("primary", "meta", "placebo") for wn in windows}
    state_all = df["mkt_state"].to_numpy(np.float64)
    side_all = df["side"].to_numpy(np.float64)
    state_acc = {}                      # (variant, state[, side]) -> [sum pnl, events, sum size] over paths; primary and meta only (the placebo cannot be recomputed independently)
    wstate_acc = {}                     # (window, state) -> events over paths (candidate events, identical for every variant)
    t_eff_div = max(1.0, float(h + 1))          # bars are days: the holding period in days
    for path, g in preds.groupby("path"):
        idx = g["event"].to_numpy(np.int64)
        p = g["p"].to_numpy(np.float64)
        size = MM.bet_size(p, step)
        variants = {"primary": np.ones(len(idx)), "meta": size, "placebo": rng.permutation(size)}
        for vname, s in variants.items():
            pnl = s * (ret[idx] - cost_frac * (s > 0)) * 1e4
            d = _daily(day_all[idx], pnl, None)
            d.insert(0, "variant", vname)
            d.insert(0, "path", int(path))
            daily_parts.append(d)
            per_event_bp = pnl
            if vname in ("primary", "meta"):
                for stv in (0.0, 1.0):
                    mm = state_all[idx] == stv
                    a = state_acc.setdefault((vname, int(stv)), [0.0, 0, 0.0])
                    a[0] += float(pnl[mm].sum()); a[1] += int(mm.sum()); a[2] += float(s[mm].sum())
                    for sdv in (1.0, -1.0):
                        m2 = mm & (side_all[idx] == sdv)
                        a2 = state_acc.setdefault((vname, int(stv), int(sdv)), [0.0, 0, 0.0])
                        a2[0] += float(pnl[m2].sum()); a2[1] += int(m2.sum()); a2[2] += float(s[m2].sum())
            for wn, (lo_t, hi_t) in windows.items():
                m = (t0[idx] >= lo_t) & (t0[idx] < hi_t)
                if vname == "primary":
                    for stv in (0.0, 1.0):
                        wstate_acc[(wn, int(stv))] = wstate_acc.get((wn, int(stv)), 0) + int((m & (state_all[idx] == stv)).sum())
                win_acc[(vname, wn)].append((float(per_event_bp[m].sum()), int(m.sum())))
            mom = B.sharpe_moments((d["sum_bp"] / d["n"]).to_numpy())
            t_eff = mom["n"] / t_eff_div
            out_paths.append({"path": int(path), "variant": vname, "events": int(len(idx)), "days": int(len(d)), "mean_bp_per_candidate": float(pnl.mean()),
                              "sr_daily": mom["sr"], "sr_annual": mom["sr"] * math.sqrt(ANNUAL_DAYS) if np.isfinite(mom["sr"]) else float("nan"),
                              "skew": mom["skew"], "kurt": mom["kurt"], "t_eff": t_eff,
                              "psr": B.psr(mom["sr"], t_eff, mom["skew"], mom["kurt"]) if np.isfinite(mom["sr"]) else float("nan"),
                              "taken_share": float((s > 0).mean()), "mean_size": float(s.mean())})
    daily = pd.concat(daily_parts, ignore_index=True)
    pp = pd.DataFrame(out_paths)
    summary = {}
    for v, g in pp.groupby("variant"):
        summary[v] = {"paths": int(len(g)), "mean_bp_per_candidate": float(g["mean_bp_per_candidate"].mean()), "mean_sr_daily": float(g["sr_daily"].mean()),
                      "mean_sr_annual": float(g["sr_annual"].mean()), "paths_sr_negative": int((g["sr_daily"] < 0).sum()), "mean_psr": float(g["psr"].mean()),
                      "taken_share": float(g["taken_share"].mean()), "mean_size": float(g["mean_size"].mean()), "events_per_path": float(g["events"].mean())}
    piv = pp.pivot(index="path", columns="variant", values="sr_daily")
    pivb = pp.pivot(index="path", columns="variant", values="mean_bp_per_candidate")
    summary["meta_minus_placebo"] = {"paths_sr_higher": int((piv["meta"] > piv["placebo"]).sum()), "paths": int(len(piv)),
                                     "mean_sr_daily_diff": float((piv["meta"] - piv["placebo"]).mean()),
                                     "mean_bp_diff": float((pivb["meta"] - pivb["placebo"]).mean())}
    summary["meta_minus_primary"] = {"paths_sr_higher": int((piv["meta"] > piv["primary"]).sum()), "paths": int(len(piv)),
                                     "mean_sr_daily_diff": float((piv["meta"] - piv["primary"]).mean())}
    win_out = {}
    for (v, wn), lst in win_acc.items():
        tot_n = sum(n for _, n in lst)
        win_out[f"{v}|{wn}"] = {"events_total_over_paths": tot_n, "events_per_path": tot_n / max(1, len(lst)),
                               "mean_bp_per_candidate": (sum(s for s, _ in lst) / tot_n) if tot_n else float("nan")}
    # DSR for the meta variant: expected max Sharpe across n trials with variance of path Sharpes (proxy), shown for several trial counts
    m = pp[pp["variant"] == "meta"]
    sr_hat = float(m["sr_daily"].mean())
    var_paths = float(m["sr_daily"].var(ddof=1)) if len(m) > 1 else float("nan")
    t_eff = float(m["t_eff"].mean())
    skew, kurt = float(m["skew"].mean()), float(m["kurt"].mean())
    # variance of Sharpe estimates across (unskilled) trials: no cross-trial sample exists for one model, so use the larger of the variance across
    # the backtest paths and the asymptotic sampling variance of one Sharpe estimate (Equation 52's denominator)
    var_asym = (1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2) / max(t_eff - 1.0, 1.0) if np.isfinite(sr_hat) else float("nan")
    var_sr = float(np.nanmax([var_paths, var_asym])) if np.isfinite([var_paths, var_asym]).any() else float("nan")
    dsr = {}
    for n_tr in sorted({1, 10, int(prior_trials), int(prior_trials) + 1}):
        if n_tr <= 1:
            dsr[str(n_tr)] = {"sr_star": 0.0, "dsr": B.psr(sr_hat, t_eff, skew, kurt, 0.0)}
        else:
            star = B.expected_max_sharpe(n_tr, var_sr) if np.isfinite(var_sr) else float("nan")
            dsr[str(n_tr)] = {"sr_star": star, "dsr": B.psr(sr_hat, t_eff, skew, kurt, star) if np.isfinite(star) else float("nan")}
    n_paths_ = int(pp["path"].nunique())
    by_state = {}
    for key, (tot, cnt, szs) in state_acc.items():
        nm = "|".join([key[0], f"state{key[1]}"] + ([f"side{key[2]:+d}"] if len(key) == 3 else []))
        by_state[nm] = {"events_per_path": cnt / n_paths_, "mean_bp_per_candidate": (tot / cnt) if cnt else float("nan"), "mean_size": (szs / cnt) if cnt else float("nan")}
    win_state = {f"{wn}|state{stv}": n / n_paths_ for (wn, stv), n in wstate_acc.items()}
    return {"paths": out_paths, "daily": daily, "by_state": by_state, "window_state_events": win_state, "summary": summary, "windows": win_out,
            "dsr": {"sr_daily_mean": sr_hat, "var_sr_used": var_sr, "var_across_paths": var_paths, "var_asymptotic": var_asym, "t_eff": t_eff, "skew": skew, "kurt": kurt, "by_trials": dsr}}
