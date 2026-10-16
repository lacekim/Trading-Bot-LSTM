"""Runner sections for Chapters 2-6: each function takes (V, cfg, log) and returns (numbers dict, markdown lines).  V is ports_common.Venue."""
from __future__ import annotations

import numpy as np
import pandas as pd

import bar_events as BE
import book_mp as MP
import book_port_a as PA
import book_port_e as PE
import book_stats as BS
import fracdiff_core as FC
import meta_events as ME
import time_decay as TD
from ports_common import daily_closes, num, returns_within_gap, sorted_events, systematic_sample, trading_days


def ch2(V, cfg, log):
    """Snippet 2.1 on the cross-section of daily bar returns; 2.2 / 2.3 are not applicable (no futures contracts)."""
    bars = V.bars()
    dc = daily_closes(bars)
    dcf = trading_days(dc)
    r = returns_within_gap(dcf, V.max_gap_days)
    win = r.iloc[-cfg["ch2_days"]:]
    keep = [c for c in win.columns if win[c].notna().mean() >= 0.95]
    win = win[keep].dropna()
    n_dropped_days = int((dc.index.difference(dcf.index) >= win.index[0]).sum())       # data-hole days inside the window's span
    cov = win.cov().values
    n = cov.shape[0]
    out = {"symbols_in_subset": len(bars), "days_dropped_as_data_holes": n_dropped_days, "symbols_used": n, "days": int(len(win)), "window": [str(win.index[0].date()), str(win.index[-1].date())]}
    w_last = PA.pcaWeights(cov)                                      # default: all risk on the last (smallest) principal component
    out["last_pc"] = {"variance": float((w_last.T @ cov @ w_last).item()), "target_variance": 1.0, "gross": float(np.abs(w_last).sum()), "max_abs_weight": float(np.abs(w_last).max())}
    rd = np.ones(n) / n
    w_u = PA.pcaWeights(cov, riskDist=rd)                            # risk spread evenly over the components
    out["even_pcs"] = {"variance": float((w_u.T @ cov @ w_u).item()), "target_variance": 1.0, "gross": float(np.abs(w_u).sum()), "max_abs_weight": float(np.abs(w_u).max())}
    eVal = np.sort(np.linalg.eigvalsh(cov))[::-1]
    out["eigen"] = {"first_share": float(eVal[0] / eVal.sum()), "top3_share": float(eVal[:3].sum() / eVal.sum()), "smallest_over_largest": float(eVal[-1] / eVal[0])}
    md = [f"Covariance of daily log returns of {n} of the {len(bars)} subset symbols (last {len(win)} common days, {out['window'][0]} to {out['window'][1]}; symbols with fewer than 95% of days in the window are left out; the {n_dropped_days} days on which fewer than 80% of the subset symbols have a bar - data holes - are left out first, and no return is formed across a gap of more than {V.max_gap_days} day{'s' if V.max_gap_days > 1 else ''}).",
          f"The risk-allocation weights of Snippet 2.1 hit the requested variance: all risk on the last component {out['last_pc']['variance']:.6f}, risk spread evenly {out['even_pcs']['variance']:.6f} (target 1). "
          f"Gross weight {out['last_pc']['gross']:.2f} / {out['even_pcs']['gross']:.2f}. Largest eigenvalue carries {out['eigen']['first_share']:.1%} of the total variance, the three largest {out['eigen']['top3_share']:.1%}.",
          "Snippets 2.2 and 2.3 (futures roll gaps): not applicable, neither venue has futures contracts in these data; the port is tested on a synthetic roll only."]
    return out, md


def ch3(V, cfg, log):
    """Snippet 3.8 on the bar events' labels (the meta-label and the three-way barrier outcome); plus the verbatim 2.4 / 3.1-3.7 chain against the project's events on three symbols."""
    e = V.events
    out = {}
    md = []
    for name, col in (("meta_label_bin", e["bin"]), ("barrier_outcome_kind", e["kind"])):
        df = pd.DataFrame({"bin": col.to_numpy()})
        shares = df["bin"].value_counts(normalize=True).sort_index()
        kept = PA.dropLabels(df, minPct=.05)
        out[name] = {"classes": {str(k): float(v) for k, v in shares.items()}, "rows_before": int(len(df)), "rows_after": int(len(kept)), "dropped_classes": [str(k) for k in set(shares.index) - set(kept["bin"].unique())]}
        md.append(f"{name}: class shares " + ", ".join(f"{k}: {v:.3f}" for k, v in shares.items()) + f"; rows {len(df):,} -> {len(kept):,}; classes dropped (< 5%): {out[name]['dropped_classes'] or 'none'}.")
    # verbatim chain on a few symbols
    chain = chain_check(V, cfg, log)
    out["verbatim_chain"] = chain["numbers"]
    out["cusum_snippet_2_4"] = chain["cusum"]
    md += chain["md"]
    return out, md


def _daily_vol_tie_right(close, span0=100):
    """Snippet 3.1 with ONE change: the earlier bar is the last bar AT OR BEFORE (t - 1 day) (searchsorted side='right'); the printed code takes the last bar strictly before.  A diagnostic for the
    difference to the project's bar_events.daily_vol_time, which makes the same choice."""
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1), side="right")
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    return df0.ewm(span=span0).std()


def chain_check(V, cfg, log):
    """Run the book's own code (2.4 CUSUM, 3.1 daily vol, 3.4 vertical barrier, 3.6 triple barrier with side, 3.7 bins, 4.1/4.2/4.10 uniqueness and weights) on the bar closes of
    `cfg['chain_symbols']` symbols and compare with the project's events of the same symbols.  The CUSUM threshold per bar and the side are taken from the project's own columns
    (sigma-based threshold and mom720 side cannot be recomputed from the snippets: the book gives h and side as inputs), so that the comparison isolates the snippets' own logic."""
    pool = [s for s in V.by_events if int((V.events["symbol"] == s).sum()) >= cfg.get("chain_min_events", 200)] or list(V.by_events)      # symbols with at least 200 events (a symbol with 6 events proves nothing)
    syms = pool[-cfg["chain_symbols"]:] if cfg.get("chain_smallest", True) else pool[:cfg["chain_symbols"]]
    bars = V.bars(syms)
    nums, md_rows, cus_rows, cus_nums = {}, [], [], {}
    for s in syms:
        b = bars[s]
        ev = V.events[V.events["symbol"] == s].sort_values("pos")
        close = pd.Series(b.close, index=b.idx)
        close = close[~close.index.duplicated(keep="last")]
        # the project's CUSUM threshold per bar is not saved; use the events' own t0 as tEvents for the barrier chain; Snippet 2.4 is compared separately below (constant threshold)
        tE = pd.DatetimeIndex(pd.to_datetime(ev["t0"].to_numpy(), unit="s"))
        trgt_proj = pd.Series(ev["trgt"].to_numpy(), index=tE)
        side = pd.Series(ev["side"].to_numpy(float), index=tE)
        # (0) Snippet 2.4 on real bars: the printed CUSUM filter with a constant threshold against the project's cusum_events (book_stats) with the same threshold
        gcl = np.log(close.iloc[:cfg.get("cusum_bars", 20000)])
        h_c = 3.0 * float(gcl.diff().std())
        tb = PE.getTEvents(gcl, h_c)
        pb = BS.cusum_events(gcl.diff().to_numpy(), h_c)
        cus_same = bool(len(tb) == len(pb) and bool((gcl.index[pb] == tb).all()))
        cus_rows.append(f"| {s} | {len(gcl):,} | {h_c:.4f} | {len(tb):,} | {len(pb):,} | {cus_same} |")
        cus_nums[s] = {"bars": int(len(gcl)), "h": num(h_c), "events_book": int(len(tb)), "events_project": int(len(pb)), "identical": cus_same}
        # (a) daily vol: the snippet vs the project's trgt
        tv = PE.getDailyVol(close, span0=100)
        both = pd.concat([tv.reindex(tE), trgt_proj], axis=1, keys=["book", "project"]).dropna()
        reld = (np.abs(both["book"] - both["project"]) / both["project"]) if len(both) else pd.Series(dtype=float)
        dv_diff = float(reld.max()) if len(reld) else float("nan")
        dv_med = float(reld.median()) if len(reld) else float("nan")
        both2 = pd.concat([_daily_vol_tie_right(close, 100).reindex(tE), trgt_proj], axis=1, keys=["book", "project"]).dropna()
        dv_diff_tie = float(np.max(np.abs(both2["book"] - both2["project"]) / both2["project"])) if len(both2) else float("nan")
        # (b) vertical barrier + triple barrier + bins with the project's side, trgt, ptsl=[2,2], 4 days
        t1v = PE.addVerticalBarrier(close, tE, 4)
        evb = PE.getEvents(close, tE, [2, 2], trgt_proj, 0.0, 1, t1=t1v, side=side)
        bins = PE.getBins(evb, close)
        proj = ev.set_index(tE)
        proj_t1 = pd.Series(pd.to_datetime(proj["t1"].to_numpy(), unit="s"), index=tE)
        common = evb.index.intersection(tE)
        same_t1 = float((evb.loc[common, "t1"].to_numpy() == proj_t1.loc[common].to_numpy()).mean()) if len(common) else float("nan")
        same_bin = float((bins.loc[common, "bin"].to_numpy() == proj.loc[common, "bin"].to_numpy()).mean()) if len(common) else float("nan")
        nums[s] = {"events": int(len(ev)), "daily_vol_max_rel_diff": num(dv_diff), "daily_vol_median_rel_diff": num(dv_med), "daily_vol_max_rel_diff_with_tie_rule_changed": num(dv_diff_tie), "daily_vol_compared": int(len(both)), "t1_same_share": num(same_t1), "bin_same_share": num(same_bin), "events_book": int(len(evb))}
        # (c) uniqueness / weights (4.1, 4.2, 4.10) on the snippet's events vs the project's u and w_raw
        nco, tW = PE.getTW(evb.dropna(subset=["t1"]), close, 1)
        wB = PE.getSampleW(evb.dropna(subset=["t1"]), nco, close, 1)
        nums[s]["tW_events"] = int(len(tW))
        md_rows.append(f"| {s} | {len(ev):,} | {len(evb):,} | {dv_med:.2e} / {dv_diff:.2e} ({len(both):,}) | {dv_diff_tie:.2e} | {same_t1:.3f} | {same_bin:.3f} |")
        log(f"  chain {s}: events {len(ev)}, daily-vol max rel diff {dv_diff:.2e}, t1 same {same_t1:.3f}, bin same {same_bin:.3f}")
    md = ["", "The book's own code (Snippets 3.1, 3.4, 3.6, 3.7, 4.1, 4.2, 4.10) run on the bar closes of the three symbols with the fewest events among those with at least 200, with the project's event times, targets and sides as inputs, against the project's labels:",
          "", "| symbol | project events | book-code events | daily volatility: median / max relative difference to the project (n compared) | same, with the one tie rule changed (see below) | same exit time | same meta-label |", "|---|---|---|---|---|---|---|"] + md_rows + ["", "The printed Snippet 3.1 takes the last bar strictly before (t - 1 day); the project takes the last bar at or before it (bar_events.daily_vol_time says so). On bars stamped to the minute the two differ whenever a bar ended exactly one day earlier, and the exponentially weighted average carries those differences forward. Changing only that rule in the printed code brings the largest difference down to the values in column 5."]
    md += ["", "Snippet 2.4 (the printed symmetric CUSUM filter, constant threshold h = 3 x the standard deviation of the one-bar log return) on the first bars of the same symbols, against the project's cusum_events with the same h (the project's own filter uses a threshold that varies by bar; this comparison isolates the snippet's logic):", "",
           "| symbol | bars | h | events, printed snippet | events, project's filter | same events |", "|---|---|---|---|---|---|"] + cus_rows
    return {"numbers": nums, "md": md, "cusum": cus_nums}


def ch4(V, cfg, log):
    out, md = {}, []
    # --- 4.3 / 4.4 / 4.5 / 4.6 on slices of real events -----------------------------------------------------------------------------------------
    syms = V.by_events[:cfg["ch4_symbols"]]
    rows = []
    rng = np.random.RandomState(cfg["seed"])
    for s in syms:
        ev = V.events[V.events["symbol"] == s].sort_values("pos").iloc[:cfg["ch4_events"]]
        lo, hi = int(ev["pos"].min()), int(ev["t1pos"].max())
        t1 = pd.Series(ev["t1pos"].to_numpy(), index=ev["pos"].to_numpy())
        indM = PA.getIndMatrix(range(lo, hi + 1), t1)
        uP = PA.getAvgUniqueness(indM)
        uB = BS.average_uniqueness(ev["pos"].to_numpy() - lo, ev["t1pos"].to_numpy() - lo, hi - lo + 1)
        # bars between events are in the matrix; uniqueness is over the event's own bars only
        d_proj = float(np.max(np.abs(uP.to_numpy() - uB)))
        # sequential bootstrap vs standard on a smaller slice
        sub = indM.iloc[:, :cfg["ch4_seq_events"]]
        sub = sub.loc[sub.sum(axis=1) > 0]
        std, seq = [], []
        for rep in range(cfg["ch4_reps"]):
            np.random.seed(cfg["seed"] + rep)
            phi = np.random.choice(sub.columns, size=sub.shape[1])
            std.append(PA.getAvgUniqueness(sub[phi]).mean())
            np.random.seed(cfg["seed"] + rep)
            phi2 = PA.seqBootstrap(sub)
            seq.append(PA.getAvgUniqueness(sub[phi2]).mean())
        rows.append({"symbol": s, "events": int(len(ev)), "bars_spanned": int(hi - lo + 1), "mean_avg_uniqueness": float(uP.mean()), "max_abs_diff_vs_project": d_proj,
                     "seq_events": int(sub.shape[1]), "reps": cfg["ch4_reps"], "std_bootstrap_uniqueness": float(np.mean(std)), "seq_bootstrap_uniqueness": float(np.mean(seq)),
                     "seq_gt_std_in_reps": int(np.sum(np.array(seq) > np.array(std)))})
        log(f"  ch4 {s}: avgU {uP.mean():.4f}, diff vs project {d_proj:.2e}; bootstrap std {np.mean(std):.4f} vs seq {np.mean(seq):.4f}")
    out["uniqueness_and_bootstrap"] = rows
    md += ["Snippets 4.3-4.5: indicator matrix, average uniqueness and sequential bootstrap on the first events of the most active symbols. 'Max difference' compares Snippet 4.4's values with the project's average_uniqueness (book_stats) on the same events.",
           "", "| symbol | events (matrix) | bars spanned | mean avg. uniqueness | max abs diff vs project | events in bootstrap slice | uniqueness, standard bootstrap | uniqueness, sequential bootstrap | reps where sequential is higher |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['symbol']} | {r['events']} | {r['bars_spanned']:,} | {r['mean_avg_uniqueness']:.4f} | {r['max_abs_diff_vs_project']:.1e} | {r['seq_events']} | {r['std_bootstrap_uniqueness']:.4f} | {r['seq_bootstrap_uniqueness']:.4f} | {r['seq_gt_std_in_reps']} of {r['reps']} |")
    # --- 4.7 - 4.9 Monte Carlo calibrated to the venue ---------------------------------------------------------------------------------------
    e = V.events
    bars_per_sym = V.symbols_df.set_index("symbol")["bars"]
    ev_per_sym = e.groupby("symbol").size()
    dens = float((ev_per_sym / bars_per_sym.reindex(ev_per_sym.index)).median())
    hold = float((e["t1pos"] - e["pos"]).median())
    numBars = cfg["ch4_mc_bars"]
    numObs = int(np.clip(round(dens * numBars), 5, cfg["ch4_mc_obs_cap"]))
    maxH = int(max(3, round(2 * hold)))
    mc = PA.mainMC(numObs=numObs, numBars=numBars, maxH=maxH, numIters=cfg["ch4_mc_iters"], numThreads=cfg["threads"])
    out["monte_carlo"] = {"numObs": numObs, "numBars": numBars, "maxH": maxH, "iters": cfg["ch4_mc_iters"], "events_per_bar_median": dens, "median_holding_bars": hold,
                          "stdU_mean": float(mc.loc["mean", "stdU"]), "seqU_mean": float(mc.loc["mean", "seqU"]), "stdU_std": float(mc.loc["std", "stdU"]), "seqU_std": float(mc.loc["std", "seqU"])}
    m = out["monte_carlo"]
    md += ["", f"Snippets 4.7-4.9 (Monte Carlo of standard vs sequential bootstrap uniqueness), calibrated to this venue: {numObs} events in {numBars} bars (venue median: {dens:.4f} events per bar, capped at {cfg['ch4_mc_obs_cap']} for run time), "
           f"holding periods drawn from 1..{maxH - 1} bars (venue median holding {hold:.0f} bars), {m['iters']} iterations (the book's example uses 1,000,000). "
           f"Mean uniqueness: standard bootstrap {m['stdU_mean']:.4f} (sd {m['stdU_std']:.4f}), sequential {m['seqU_mean']:.4f} (sd {m['seqU_std']:.4f})."]
    # --- 4.11 time decay applied to the venue's average uniqueness --------------------------------------------------------------------------
    se = sorted_events(e)
    tW = pd.Series(se["u"].to_numpy(), index=se.index)
    dec = {}
    for c in (1., .5, 0., -.5):
        w, const, slope = PE.getTimeDecay(tW, c)
        ref = TD.decay_factors(se["u"].to_numpy(), se["t0"].to_numpy(), c)
        dec[str(c)] = {"const": float(const), "slope": float(slope), "oldest_weight": float(w.iloc[0]), "share_of_events_with_zero_weight": float((w == 0).mean()),
                       "max_abs_diff_vs_project": float(np.max(np.abs(w.to_numpy() - ref)))}
    out["time_decay"] = dec
    md += ["", "Snippet 4.11, time decay by cumulative uniqueness over all the venue's events in time order (the newest event has weight 1):", "", "| c | oldest weight | events with weight 0 | max abs diff vs project's time_decay.py |", "|---|---|---|---|"]
    for c, d in dec.items():
        md.append(f"| {c} | {d['oldest_weight']:.4f} | {d['share_of_events_with_zero_weight']:.3f} | {d['max_abs_diff_vs_project']:.1e} |")
    # Snippet 4.6 as printed (three labels: t0 = 0, 2, 4; t1 = 2, 3, 5), repeated to average out the draw
    np.random.seed(cfg["seed"])
    ex = [PA.example_seq_bootstrap() for _ in range(cfg.get("ch4_example_reps", 1000))]
    out["book_example_4_6"] = {"reps": len(ex), "standard_bootstrap_uniqueness_mean": float(np.mean([e[1] for e in ex])), "sequential_bootstrap_uniqueness_mean": float(np.mean([e[3] for e in ex]))}
    b46 = out["book_example_4_6"]
    md += ["", f"Snippet 4.6 as printed (the book's three-label example, t0 = 0, 2, 4 and t1 = 2, 3, 5), {b46['reps']:,} draws: mean uniqueness of the standard bootstrap sample {b46['standard_bootstrap_uniqueness_mean']:.4f}, of the sequential bootstrap sample {b46['sequential_bootstrap_uniqueness_mean']:.4f}."]
    return out, md


def ch5(V, cfg, log):
    """Snippet 5.2 (expanding-window fracdiff) vs 5.3 (fixed-width) on daily log closes; the minimum d that passes the ADF test, by 5.4's loop, against the project's d."""
    bars = V.bars()
    dc = daily_closes(bars)
    rows = []
    ds = np.round(np.arange(0, 1.0001, 0.1), 2)
    for s in dc.columns[:cfg["ch5_symbols"]]:
        c = dc[s].dropna()
        c = c.iloc[-cfg["ch5_days"]:]
        lg = np.log(c).to_frame("Close")
        table = []
        for d in ds:
            f = PE.fracDiff_FFD(lg, d, thres=.01)
            stat, crit, nobs = PE.adfuller_c1(f["Close"].values)
            table.append((d, stat, crit))
        passing_ffd = next((d for d, st, cr in table if st < cr), 1.0)
        # expanding window (5.2) at the same d, same threshold
        table2 = []
        for d in ds:
            f = PA.fracDiff(lg, d, thres=.01)
            stat, crit, nobs = PE.adfuller_c1(f["Close"].values)
            table2.append((d, stat, crit))
        passing_exp = next((d for d, st, cr in table2 if st < cr), 1.0)
        info = FC.min_d(lg["Close"].values, thres=.01)
        rows.append({"symbol": s, "days": int(len(c)), "min_d_ffd_book_code": float(passing_ffd), "min_d_expanding_book_code": float(passing_exp), "min_d_project": float(info["d"])})
        log(f"  ch5 {s}: min d FFD {passing_ffd}, expanding {passing_exp}, project {info['d']}")
    out = {"rows": rows}
    agree = np.mean([abs(r["min_d_ffd_book_code"] - r["min_d_project"]) < 1e-9 for r in rows]) if rows else float("nan")
    out["share_ffd_equals_project"] = num(agree)
    within = np.mean([abs(r["min_d_ffd_book_code"] - r["min_d_project"]) <= 0.1 + 1e-9 for r in rows]) if rows else float("nan")
    out["share_ffd_within_0_1_of_project"] = num(within)
    out["mean_ffd_minus_project"] = num(np.mean([r["min_d_ffd_book_code"] - r["min_d_project"] for r in rows])) if rows else None
    md = [f"Snippets 5.2-5.4 on the last {cfg['ch5_days']} daily log closes (last bar of each UTC day) of the first {len(rows)} subset symbols, d on a 0.1 grid, weight threshold 0.01; the ADF test is the 1-lag, constant version written in book_port_e.py (statsmodels is not installed).",
          f"Smallest d that passes: the book code searches a 0.1 grid, the project's search a 0.05 grid, so the two values are equal for {agree:.0%} of the symbols and within 0.1 of each other for {within:.0%}; mean (book code minus project) {out['mean_ffd_minus_project']:+.3f}. "
          "The series are consecutive rows of each symbol's daily closes; a data hole shows up as one large step in the series, for the project's search as well.", "",
          "| symbol | days | min d, fixed-width (5.3) | min d, expanding window (5.2) | min d, project |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['symbol']} | {r['days']} | {r['min_d_ffd_book_code']:.1f} | {r['min_d_expanding_book_code']:.1f} | {r['min_d_project']:.1f} |")
    return out, md


def ch6(V, cfg, log):
    """6.1 with the venue's measured accuracy; 6.2's three forest set-ups compared with a purged k-fold on a systematic sample of the venue's events."""
    import book_port_b as PB
    out, md = {}, []
    s = V.summary
    p_acc = None
    # accuracy of the venue's meta model: share of events whose predicted class (p>0.5) equals the label, over the five CPCV paths
    pred = V.cpcv_predictions()
    lab = V.events["bin"].to_numpy()
    acc = float(((pred["p"].to_numpy() > .5).astype(int) == lab[pred["event"].to_numpy()]).mean())
    out["cpcv_accuracy"] = acc
    for k in (2.,):
        p, prob = PA.bagging_accuracy(N=100, p=acc, k=k)
        out["bagging_k2"] = {"N": 100, "p": p, "prob_majority_correct_if_independent": float(prob)}
    p3, pr3 = PA.bagging_accuracy(N=100, p=1. / 3, k=3.)
    out["book_example_k3"] = {"p": p3, "prob": float(pr3)}
    md += [f"Snippet 6.1: the venue's meta-model accuracy over its five CPCV paths is {acc:.4f}. If 100 independent classifiers each had that accuracy, a majority vote (more than N/2 right) would be right with probability {out['bagging_k2']['prob_majority_correct_if_independent']:.4f}. "
           f"(The book's own example, p = 1/3, three classes, k = 3: {pr3:.4f}.) Trees fitted on overlapping, similar samples are not independent, which is why this is an upper bound and why Snippet 6.2's sequential bootstrap exists."]
    # 6.2: three set-ups, purged CV
    se = sorted_events(V.events)
    samp = systematic_sample(se, cfg["model_sample"])
    X = samp[V.features].astype(float)
    y = samp["bin"].astype(int)
    w = samp["w"].astype(float)
    w = w / w.sum()
    t1 = pd.Series(samp["t1_dt"].to_numpy(), index=samp.index)
    avgU = float(V.events["u"].mean())
    clfs = PA.three_rf_setups(avgU)
    names = ["RF (bootstrap, balanced_subsample)", "bagged decision tree with max_samples = avgU", "bagged single-tree RF (no bootstrap) with max_samples = avgU"]
    rows = []
    n_est = cfg["n_estimators"]
    for name, clf in zip(names, clfs):
        # the printed constructors use 1000 trees; the run uses n_est (time).  Bagging max_samples must be an integer here (scikit-learn >= 1.7 reads a float against the weight sum).
        if hasattr(clf, "estimator") and hasattr(clf, "max_samples"):
            ntr = int(len(samp) * (cfg["cv"] - 1) / cfg["cv"])
            clf.set_params(n_estimators=n_est, max_samples=max(1, int(round(avgU * ntr))), n_jobs=cfg["threads"])
        else:
            clf.set_params(n_estimators=n_est, n_jobs=cfg["threads"])
        np.random.seed(cfg["seed"])
        sc_nll = PB.cvScore(clf, X, y, w, scoring="neg_log_loss", cv=cfg["cv"], t1=t1, pctEmbargo=.01)
        np.random.seed(cfg["seed"])
        sc_acc = PB.cvScore(clf, X, y, w, scoring="accuracy", cv=cfg["cv"], t1=t1, pctEmbargo=.01)
        rows.append({"setup": name, "neg_log_loss_mean": float(sc_nll.mean()), "neg_log_loss_sd": float(sc_nll.std()), "accuracy_mean": float(sc_acc.mean()), "accuracy_sd": float(sc_acc.std())})
        log(f"  ch6 {name}: nll {sc_nll.mean():.4f}, acc {sc_acc.mean():.4f}")
    out["three_setups"] = {"rows": rows, "sample_events": int(len(samp)), "n_estimators": n_est, "avgU": avgU, "cv": cfg["cv"], "embargo": .01}
    md += ["", f"Snippet 6.2: the three forests of the book, {n_est} trees each (the printed 1,000 reduced for run time), purged {cfg['cv']}-fold CV with 1% embargo (Snippets 7.3-7.4) on a systematic sample of {len(samp):,} of the venue's {len(se):,} events "
           f"(every {len(se) // len(samp)}th event in time order; sample weights = the project's attribution weights, normalised), max_samples = average uniqueness {avgU:.4f}. Chance accuracy is the label share; log loss of a coin flip is -0.6931.", "",
           "| set-up | mean weighted log loss (higher is better) | mean accuracy |", "|---|---|---|"]
    for r in rows:
        md.append(f"| {r['setup']} | {r['neg_log_loss_mean']:.4f} (sd {r['neg_log_loss_sd']:.4f}) | {r['accuracy_mean']:.4f} (sd {r['accuracy_sd']:.4f}) |")
    return out, md
