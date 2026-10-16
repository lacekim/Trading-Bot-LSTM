"""Runner sections for Chapters 13-21 (Snippets 13.1-13.2, 14.1-14.4, 15.1-15.5, 16.1-16.5, 17.1-17.4, 18.1-18.4, 19.1-19.2, 20.x, 21.1-21.7)."""
from __future__ import annotations

import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.stats import norm

import bar_events as BE
import book_mp as MP
import book_port_a as PA
import book_port_c as PC
import book_port_d as PD
import book_port_e as PE
import book_stats as BS
import hrp_alloc as HA
import meta_events as ME
import micro_features as MF
from ports_common import daily_closes, num, returns_within_gap, sorted_events, trading_days


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 13
# ---------------------------------------------------------------------------------------------------------------------
def _event_paths(V, bars, events: pd.DataFrame, K: int):
    """Paths in sigma units: side * (log close[pos + k] - log close[pos]) / sigma, k = 1..K; events whose K bars cross a data hole or the end of the series are left out."""
    rows, paths, keep_idx = [], [], []
    for sym, g in events.groupby("symbol", sort=False):
        b = bars[sym]
        lc = np.log(b.close)
        pos = g["pos"].to_numpy(np.int64)
        ok = pos + K < b.n
        pos_ok = pos[ok]
        same = b.seg[np.minimum(pos_ok + K, b.n - 1)] == b.seg[pos_ok]
        sel = np.flatnonzero(ok)[same]
        pos_sel = pos[sel]
        steps = np.arange(1, K + 1)
        idx = pos_sel[:, None] + steps[None, :]
        q = (lc[idx] - lc[pos_sel][:, None]) / g["sigma"].to_numpy(float)[sel][:, None] * g["side"].to_numpy(float)[sel][:, None]
        paths.append(q.astype(np.float32))
        keep_idx.append(g.index.to_numpy()[sel])
    return np.concatenate(paths), np.concatenate(keep_idx)


def ch13(V, cfg, log):
    out, md = {}, []
    # (1) the verbatim batch (13.2) against the project's vectorised mesh on one (forecast, half-life) pair
    n_cmp = cfg["ch13_compare_iter"]
    grid = np.linspace(0, 10, 21)
    random.seed(cfg["seed"])
    t0 = time.time()
    bt = PC.batch({'forecast': 5, 'hl': 10, 'sigma': 1}, nIter=n_cmp, maxHP=100, rPT=grid[1::2], rSLm=grid[1::2], seed=0)
    t_batch = time.time() - t0
    phi = 2 ** (-1. / 10)
    rng = np.random.default_rng(cfg["seed"])
    eps = rng.standard_normal((cfg["ch13_ref_paths"], 100))
    sim = BS.simulate_ou_paths((1 - phi) * 5, phi, eps)
    ref = BS.mesh_exit_stats(sim, pts=tuple(grid[1::2]), sls=tuple(grid[1::2]))
    sh_ref = np.asarray(ref["sharpe"])
    sh_bat = np.array([b[4] for b in bt]).reshape(sh_ref.shape)
    mask = np.isfinite(sh_ref) & np.isfinite(sh_bat)
    se = 1. / math.sqrt(n_cmp)
    out["batch_vs_project_mesh"] = {"nIter_batch": n_cmp, "paths_project": cfg["ch13_ref_paths"], "nodes": int(mask.sum()), "max_abs_diff": float(np.max(np.abs(sh_ref[mask] - sh_bat[mask]))),
                                    "mean_abs_diff": float(np.mean(np.abs(sh_ref[mask] - sh_bat[mask]))), "approx_standard_error_of_a_sharpe": se, "corr": float(np.corrcoef(sh_ref[mask], sh_bat[mask])[0, 1]),
                                    "batch_seconds": t_batch, "best_node_batch": [float(bt[int(np.nanargmax([b[4] for b in bt]))][0]), float(bt[int(np.nanargmax([b[4] for b in bt]))][1])],
                                    "best_node_project": [float(grid[1::2][i]) for i in divmod(int(np.nanargmax(sh_ref)), sh_ref.shape[1])]}
    c = out["batch_vs_project_mesh"]
    md += [f"Snippet 13.2 (the book's loop, {n_cmp} paths per node) against the project's vectorised mesh ({cfg['ch13_ref_paths']:,} paths) for forecast 5, half-life 10, {c['nodes']} nodes of the stop-loss / profit-taking grid: correlation of the Sharpe surfaces {c['corr']:.3f}, "
           f"mean absolute difference {c['mean_abs_diff']:.3f}, largest {c['max_abs_diff']:.3f} (sampling error of a Sharpe ratio from {n_cmp} paths is about {se:.3f}). Best node: book loop pt {c['best_node_batch'][0]:g} / sl {c['best_node_batch'][1]:g}; project pt {c['best_node_project'][0]:g} / sl {c['best_node_project'][1]:g}."]
    # (2) Step 1 on the venue's bar paths (before the cutoff), the mesh by the book's loop, the chosen node evaluated on the paths after the cutoff
    bars = V.bars(sorted(V.events["symbol"].unique()) if len(V.events["symbol"].unique()) <= 60 else V.subset)
    ev = V.events[V.events["symbol"].isin(bars)].copy()
    ev = ev[ev["side"] != 0]
    K = cfg["ch13_steps"]
    cutoff = pd.Timestamp(V.summary["params"]["cutoff"]).timestamp()
    q, ix = _event_paths(V, bars, ev, K)
    evk = ev.loc[ix]
    tr = (evk["t1"].to_numpy() < cutoff)
    te = (evk["t0"].to_numpy() >= cutoff)
    est = BS.estimate_ou(q[tr], evk["u"].to_numpy()[tr])
    phi_hat = est["phi"]
    hl = BS.half_life(phi_hat)
    forecast = est["a"] / (1 - phi_hat) if abs(1 - phi_hat) > 1e-12 else float("nan")
    out["step1"] = {"K": K, "train_events": int(tr.sum()), "test_events": int(te.sum()), "phi": float(phi_hat), "a": float(est["a"]), "sigma_fit": float(est["sigma"]), "half_life_bars": num(hl), "forecast_level_sigma": num(forecast)}
    # mesh by the book's loop at the estimated parameters (coarse grid: pt, sl in 1..9 sigma, step 2 -> 5x5)
    coarse = np.array([1., 3., 5., 7., 9.])
    hl_arg = hl if np.isfinite(hl) else 1e9
    random.seed(cfg["seed"] + 1)
    bt2 = PC.batch({'forecast': forecast if np.isfinite(forecast) else 0., 'hl': hl_arg, 'sigma': est["sigma"]}, nIter=cfg["ch13_est_iter"], maxHP=K, rPT=coarse, rSLm=coarse, seed=0)
    k = int(np.nanargmax([b[4] for b in bt2]))
    pt, sl, sim_sr = bt2[k][0], bt2[k][1], bt2[k][4]
    s_te = q[te]
    mt = BS.mesh_exit_stats(s_te.astype(np.float32), pts=tuple(coarse), sls=tuple(coarse))
    i_s, j_s = list(coarse).index(pt), list(coarse).index(sl)
    # the book's reference rule for comparison: the triple barrier of the events (2 x trgt) is not a node of this grid; report the realised Sharpe at every node's median
    out["mesh_choice"] = {"coarse_grid_sigma": coarse.tolist(), "nIter": cfg["ch13_est_iter"], "chosen_pt": float(pt), "chosen_sl": float(sl), "simulated_sharpe_at_chosen": float(sim_sr),
                          "realised_sharpe_at_chosen_test": num(mt["sharpe"][i_s, j_s]), "realised_sharpe_median_over_grid_test": num(np.nanmedian(mt["sharpe"])),
                          "realised_sharpe_best_node_in_hindsight_test": num(np.nanmax(mt["sharpe"]))}
    m = out["mesh_choice"]
    md += ["", f"Step 1 (Equations 13.5-13.7) on {int(tr.sum()):,} events whose bars end before the cutoff {V.summary['params']['cutoff']}, paths of {K} bars in units of the event's bar volatility (side-signed log return): "
           f"phi = {phi_hat:.4f}, half-life {('%.1f bars' % hl) if np.isfinite(hl) else 'none (phi outside (0,1))'}, drift a = {est['a']:.4f}, residual sigma = {est['sigma']:.3f}. "
           f"Steps 2-5 by the book's loop at those parameters ({cfg['ch13_est_iter']} paths per node, 5 x 5 grid of 1, 3, 5, 7, 9 sigma): chosen profit-taking {pt:g} sigma, stop-loss {sl:g} sigma, simulated Sharpe {sim_sr:.3f}. "
           f"The same node on the {int(te.sum()):,} events starting after the cutoff: realised Sharpe (per-event, sigma units) {('n/a' if m['realised_sharpe_at_chosen_test'] is None else format(m['realised_sharpe_at_chosen_test'], '.3f'))}; "
           f"median over the grid {m['realised_sharpe_median_over_grid_test']:.3f}; best node chosen in hindsight {m['realised_sharpe_best_node_in_hindsight_test']:.3f}."]
    # Snippet 13.1 as printed: the 5 x 5 grid of (forecast, half-life) with the 21 x 21 mesh of profit-taking / stop-loss multiples; the iterations per node are cut from the printed 100,000
    t_m = time.time()
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    grid131 = PC.main_13_1(nIter=cfg.get("ch13_main_iter", 100), maxHP=100)
    best = {}
    for (fc, hl), nodes in grid131.items():
        arr = np.array(nodes, dtype=float)
        ok = np.isfinite(arr[:, 4])
        j = int(np.nanargmax(np.where(ok, arr[:, 4], -np.inf)))
        best[f"{fc}|{hl}"] = {"pt": float(arr[j, 0]), "sl": float(arr[j, 1]), "sharpe": float(arr[j, 4]), "median_sharpe_over_mesh": float(np.nanmedian(arr[ok, 4]))}
    out["main_13_1"] = {"iterations_per_node": int(cfg.get("ch13_main_iter", 100)), "nodes_per_combination": 441, "best_node": best, "seconds": time.time() - t_m}
    md += ["", f"Snippet 13.1 as printed (main): forecast in (10, 5, 0, -5, -10) x half-life in (5, 10, 25, 50, 100), sigma 1, 21 x 21 mesh of profit-taking and stop-loss multiples from 0 to 10, maximum holding 100 steps; {out['main_13_1']['iterations_per_node']} paths per node (the printed 100,000 would take hours here). "
           "The best node of a combination is the maximum over 441 noisy estimates (for a Sharpe ratio near zero the standard error of one estimate from n paths is about 1/sqrt(n), i.e. 0.1 at 100 paths), so its Sharpe ratio is biased upward; the median over the mesh is shown beside it.", "",
           "| forecast | half-life | best node: profit-taking, stop-loss (multiples of sigma) | Sharpe at the best node | median Sharpe over the mesh |", "|---|---|---|---|---|"]
    for key, d in best.items():
        fc, hl = key.split("|")
        md.append(f"| {fc} | {hl} | {d['pt']:.1f}, {d['sl']:.1f} | {d['sharpe']:.3f} | {d['median_sharpe_over_mesh']:.3f} |")
    return out, md


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 14 and 15
# ---------------------------------------------------------------------------------------------------------------------
def _bet_stream(V, variant: str):
    """CPCV path 0 bets: per event (index in events), size and the side-signed net return.  variant: 'primary' (every event, size 1) or 'meta' (events with p > 0.5, book bet size)."""
    pred = V.cpcv_predictions()
    p0 = pred[pred["path"] == 0].set_index("event")["p"].sort_index()
    e = V.events.loc[p0.index].copy()
    p = p0.to_numpy()
    if variant == "meta":
        z = (p - .5) / (p * (1 - p)) ** .5
        size = np.where(p > .5, 2 * norm.cdf(z) - 1, 0.)
        size = np.clip(np.round(size / float(V.summary["params"]["step"])) * float(V.summary["params"]["step"]), 0, 1)
    else:
        size = np.ones(len(p))
    e["size"] = size
    e = e[e["size"] > 0]
    e["t1_dt"] = pd.to_datetime(e["t1"], unit="s")
    e["t0_dt"] = pd.to_datetime(e["t0"], unit="s")
    return e


def ch14(V, cfg, log):
    out, md = {}, []
    sig = V.cache.get("signals", {})
    # 14.1 / 14.2 per symbol from the book's averaged target-position series (Snippet 10.1-10.3 output)
    hp_rows, bets_total = [], 0
    for sym, s in sig.items():
        if len(s) < 3 or (s != 0).sum() < 2:
            continue
        tpos = s.astype(float)
        bets = PC.getBets(tpos)
        hp = PC.getHoldingPeriod(tpos)
        w = float(np.abs(tpos.diff().fillna(0)).sum())
        hp_rows.append({"symbol": sym, "bets": int(len(bets)), "holding_days": num(hp), "weight": w})
        bets_total += int(len(bets))
    hpdf = pd.DataFrame(hp_rows).dropna()
    wavg = float((hpdf["holding_days"] * hpdf["weight"]).sum() / hpdf["weight"].sum()) if len(hpdf) else float("nan")
    # reference: the events' own mean duration in days among the same symbols (meta bets, weighted by size)
    meta = _bet_stream(V, "meta")
    ref_hp = float(np.average((meta["t1"] - meta["t0"]) / 86400., weights=meta["size"])) if len(meta) else float("nan")
    out["holding_period"] = {"symbols": int(len(hpdf)), "bets": bets_total, "weighted_mean_days": num(wavg), "median_symbol_days": num(hpdf["holding_days"].median() if len(hpdf) else float("nan")),
                             "p10_symbol_days": num(hpdf["holding_days"].quantile(.1) if len(hpdf) else float("nan")), "p90_symbol_days": num(hpdf["holding_days"].quantile(.9) if len(hpdf) else float("nan")),
                             "events_size_weighted_mean_duration_days": num(ref_hp)}
    h = out["holding_period"]
    md += [f"Snippets 14.1-14.2 on the target-position series (the averaged, rounded signal of Snippets 10.1-10.3, CPCV path 0) of {h['symbols']} symbols: {h['bets']:,} bets between flat positions or flips; "
           f"average holding period by the average-entry-time pairing algorithm, weighted by size traded: {wavg:.2f} days (per symbol: median {h['median_symbol_days']:.2f}, 10th-90th percentile {h['p10_symbol_days']:.2f}-{h['p90_symbol_days']:.2f}). "
           f"Size-weighted mean duration of the same path's events (start to exit): {ref_hp:.2f} days. Overlapping bets are netted by the averaging, so the two need not be equal."]
    # 14.3 concentration and 14.4 drawdown / time under water on the bet returns
    rows = []
    for variant in ("primary", "meta"):
        e = _bet_stream(V, variant)
        ret = pd.Series((e["size"] * e["net_ret"]).to_numpy(), index=e["t1_dt"]).sort_index()
        ret = ret.groupby(level=0).sum() if ret.index.has_duplicates else ret
        rpos, rneg, th = PC.getHHIs(ret)
        pnl = ret.cumsum()
        dd, tuw = PC.computeDD_TuW(pnl, dollars=True)
        rows.append({"variant": variant, "bets": int(len(e)), "hhi_positive": num(rpos), "hhi_negative": num(rneg), "hhi_bets_per_month": num(th),
                     "drawdown_max": num(dd.max() if len(dd) else float("nan")), "drawdown_p95": num(dd.quantile(.95) if len(dd) else float("nan")),
                     "tuw_years_mean": num(tuw.mean() if len(tuw) else float("nan")), "tuw_years_max": num(tuw.max() if len(tuw) else float("nan")), "drawdown_episodes": int(len(dd)),
                     "final_pnl_units": num(pnl.iloc[-1]) if len(pnl) else None, "span_years": num((pnl.index[-1] - pnl.index[0]).days / 365.25) if len(pnl) else None})
    out["concentration_dd_tuw"] = rows
    md += ["", "Snippets 14.3-14.4 on the bets' net returns after costs (CPCV path 0; one unit of notional per bet times its size; returns summed by exit time; the profit-and-loss curve is the running sum, so drawdowns are in units of one bet's notional, not a share of capital):", "",
           "| variant | bets | HHI of positive returns | HHI of negative returns | HHI of bets per month | max drawdown | 95th pct drawdown | mean time under water (yrs) | max time under water (yrs) | final P&L (units) |", "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, n=3: "n/a" if x is None else format(x, f".{n}f")      # noqa: E731
    for r in rows:
        md.append(f"| {r['variant']} | {r['bets']:,} | {'n/a' if r['hhi_positive'] is None else format(r['hhi_positive'], '.2e')} | {'n/a' if r['hhi_negative'] is None else format(r['hhi_negative'], '.2e')} | {'n/a' if r['hhi_bets_per_month'] is None else format(r['hhi_bets_per_month'], '.2e')} | {f(r['drawdown_max'], 2)} | {f(r['drawdown_p95'], 2)} | {f(r['tuw_years_mean'], 2)} | {f(r['tuw_years_max'], 2)} | {f(r['final_pnl_units'], 1)} |")
    return out, md


def ch15(V, cfg, log):
    out, md = {}, []
    np.random.seed(cfg["seed"])
    m, s, sr = PC.sharpe_of_binomial_draws(n=cfg["ch15_binom_n"], p=.55)
    out["snippet_15_1"] = {"n": cfg["ch15_binom_n"], "p": .55, "mean": float(m), "std": float(s), "sharpe": float(sr), "formula": (2 * .55 - 1) / (2 * math.sqrt(.55 * .45))}
    out["snippet_15_2_max_abs_diff"] = PC.factor_variance_check()
    md += [f"Snippet 15.1 as printed ({cfg['ch15_binom_n']:,} bets of +1 / -1 with probability 0.55): mean {m:.4f}, std {s:.4f}, ratio {sr:.4f}; exact value (2p-1)/(2 sqrt(p(1-p))) = {out['snippet_15_1']['formula']:.4f}. "
           f"Snippet 15.2 (SymPy is not installed): the identity it factors, p u^2 + (1-p) d^2 - (p u + (1-p) d)^2 = p (1-p) (u-d)^2, holds to {out['snippet_15_2_max_abs_diff']:.1e} on a grid."]
    rows = []
    for variant in ("primary", "meta"):
        e = _bet_stream(V, variant)
        r = e["net_ret"].to_numpy(float) * 1.0
        years = (e["t1"].max() - e["t0"].min()) / (365.25 * 86400.)
        freq = len(e) / years
        pos, neg = r[r > 0].mean(), r[r <= 0].mean()
        phat = float((r > 0).mean())
        row = {"variant": variant, "bets": int(len(e)), "years": float(years), "bets_per_year": float(freq), "precision": phat, "avg_win": float(pos), "avg_loss": float(neg)}
        for tsr in (1., 2.):
            ph = PC.binHR(neg, pos, freq, tsr)
            row[f"implied_precision_sr{int(tsr)}"] = float(ph)
            row[f"prob_failure_sr{int(tsr)}"] = float(PC.probFailure(r, freq, tsr))
            row[f"implied_freq_sr{int(tsr)}"] = num(PC.binFreq(neg, pos, phat, tsr))
        row["implied_sr_at_observed_precision"] = float(PC.binSR(neg, pos, freq, phat))
        rows.append(row)
    out["strategy_risk"] = rows
    md += ["", "Snippets 15.3-15.5 on the bets' net returns after costs (CPCV path 0, every bet counted once, bets per year = number of bets / years covered; this treats the pooled bets as independent draws of one strategy, which overlapping bets across symbols are not):", "",
           "| variant | bets | bets per year | observed precision | avg win | avg loss | precision needed for Sharpe 1 / 2 | Sharpe implied by observed precision | 'probability of failure' (Snippet 15.5) for Sharpe 1 / 2 | bets per year needed at observed precision for Sharpe 1 / 2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fq = lambda x: "none" if x is None else f"{x:,.0f}"        # noqa: E731
        md.append(f"| {r['variant']} | {r['bets']:,} | {r['bets_per_year']:,.0f} | {r['precision']:.4f} | {r['avg_win']:.5f} | {r['avg_loss']:.5f} | {r['implied_precision_sr1']:.4f} / {r['implied_precision_sr2']:.4f} | {r['implied_sr_at_observed_precision']:.2f} | "
                  f"{r['prob_failure_sr1']:.4f} / {r['prob_failure_sr2']:.4f} | {fq(r['implied_freq_sr1'])} / {fq(r['implied_freq_sr2'])} |")
    # reporting windows (UTC): bets whose exit time lies in the window
    windows = {k: (int(v[0]), int(v[1])) for k, v in V.summary["meta"]["window_epoch"].items()}
    wrows = []
    for variant in ("primary", "meta"):
        e = _bet_stream(V, variant)
        for wn, (a_, b_) in windows.items():
            m_ = e[(e["t1"] >= a_) & (e["t1"] <= b_)]
            if len(m_) < 30:
                wrows.append({"variant": variant, "window": wn, "bets": int(len(m_)), "precision": None, "mean_net_bp": None})
            else:
                r_ = m_["net_ret"].to_numpy(float)
                wrows.append({"variant": variant, "window": wn, "bets": int(len(m_)), "precision": float((r_ > 0).mean()), "mean_net_bp": float(r_.mean() * 1e4)})
    out["windows"] = wrows
    md += ["", "The same bets by reporting window (UTC calendar windows from the source run; a bet belongs to the window that contains its exit time; fewer than 30 bets = n/a):", "", "| variant | window | bets | precision | mean net return per bet (bp) |", "|---|---|---|---|---|"]
    for r in wrows:
        md.append(f"| {r['variant']} | {r['window']} | {r['bets']:,} | {'n/a' if r['precision'] is None else format(r['precision'], '.4f')} | {'n/a' if r['mean_net_bp'] is None else format(r['mean_net_bp'], '.1f')} |")
    np.random.seed(cfg["seed"])
    out["snippet_15_5_main"] = float(PC.main_15_5())
    md += ["", "Caution on the 'probability of failure' column: Snippet 15.5 as printed evaluates norm.cdf(required precision, observed precision, observed precision x (1 - observed precision)), i.e. it uses a standard deviation of p(1-p) (about 0.25) rather than the standard error of the precision estimate (which shrinks with the number of bets). "
          "With a spread that wide the result stays close to 0.5 whenever the required and observed precisions are near each other, so it carries little information about these bet counts. The code is kept as printed."]
    md += ["", f"Snippet 15.5's own example (mixture of two Gaussians, 2,600 draws, target Sharpe 2, 260 bets a year): {out['snippet_15_5_main']:.4f}."]
    return out, md


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 16
# ---------------------------------------------------------------------------------------------------------------------
def ch16(V, cfg, log):
    out, md = {}, []
    bars = V.bars()
    dc = daily_closes(bars)
    end = dc.index.max()
    start = end - pd.Timedelta(days=cfg["ch16_days"])
    sub = dc[(dc.index >= start)]
    n_all_days = len(sub)
    sub = trading_days(sub)
    n_hole_days = n_all_days - len(sub)
    sub = sub.ffill(limit=3)
    keep = [c for c in sub.columns if sub[c].notna().mean() >= .98]
    sub = sub[keep].dropna()
    r = returns_within_gap(sub, V.max_gap_days, "pct").dropna()
    cov, corr = r.cov(), r.corr()
    n = r.shape[1]
    out["data"] = {"symbols": n, "days": int(len(r)), "window": [str(r.index[0].date()), str(r.index[-1].date())]}
    # full sample
    w_hrp = PD.getHRP(cov.values, corr.values)
    w_hrp.index = cov.index[w_hrp.index]
    w_ivp = pd.Series(PD.getIVP(cov.values), index=cov.index)
    w_cla = pd.Series(PD.getCLA(cov.values), index=cov.index)
    # project's HRP for the cross-check
    wp = HA.hrp_weights(cov.values)
    wp = wp[0] if isinstance(wp, tuple) else wp
    out["full_sample"] = {"hrp_max_abs_diff_vs_project": float(np.max(np.abs(np.asarray(wp, float) - w_hrp.reindex(cov.index).to_numpy()))),
                          "hrp_top5": {k: float(v) for k, v in w_hrp.sort_values(ascending=False).head(5).items()}, "ivp_top5": {k: float(v) for k, v in w_ivp.sort_values(ascending=False).head(5).items()},
                          "cla_nonzero": int((w_cla > 1e-9).sum()), "cla_top5": {k: float(v) for k, v in w_cla.sort_values(ascending=False).head(5).items()},
                          "max_weight": {"hrp": float(w_hrp.max()), "ivp": float(w_ivp.max()), "cla": float(w_cla.max())},
                          "effective_n": {"hrp": float(1 / (w_hrp ** 2).sum()), "ivp": float(1 / (w_ivp ** 2).sum()), "cla": float(1 / (w_cla ** 2).sum())}}
    f = out["full_sample"]
    md += [f"Snippets 16.1-16.4 on daily returns of {n} symbols with at least 98% of days in the last {cfg['ch16_days']} calendar days ({out['data']['window'][0]} to {out['data']['window'][1]}, {len(r)} days; prices forward-filled for at most 3 days; the {n_hole_days} days of that span on which fewer than 80% of the subset symbols have a bar - data holes - are left out first, and no return is formed across a gap of more than {V.max_gap_days} day{'s' if V.max_gap_days > 1 else ''}). "
           f"Largest HRP weight difference against the project's hrp_alloc.py: {f['hrp_max_abs_diff_vs_project']:.1e}. "
           f"Effective number of positions (1 / sum of squared weights): HRP {f['effective_n']['hrp']:.1f}, inverse-variance {f['effective_n']['ivp']:.1f}, minimum-variance long-only (the CLA end point) {f['effective_n']['cla']:.1f}; the minimum-variance portfolio holds {f['cla_nonzero']} of {n} names.",
           f"Largest weights: HRP {', '.join(f'{k} {v:.1%}' for k, v in list(f['hrp_top5'].items())[:3])}; minimum-variance {', '.join(f'{k} {v:.1%}' for k, v in list(f['cla_top5'].items())[:3])}."]
    # rolling out-of-sample experiment on the real daily returns (Snippet 16.5's loop)
    sL, reb = cfg["ch16_slength"], cfg["ch16_rebal"]
    methods = [PD.getIVP, PD.getHRP, PD.getCLA]
    rets = {m.__name__: [] for m in methods}
    x = r.to_numpy()
    for pointer in range(sL, len(x), reb):
        xin = x[pointer - sL:pointer]
        cov_, corr_ = np.cov(xin, rowvar=0), np.corrcoef(xin, rowvar=0)
        xout = x[pointer:pointer + reb]
        for m in methods:
            w_ = m(cov=cov_, corr=corr_)
            rets[m.__name__].append(pd.Series(np.dot(xout, np.asarray(w_, float))))
    stats = {}
    for k, v in rets.items():
        s = pd.concat(v).reset_index(drop=True)
        stats[k] = {"oos_days": int(len(s)), "daily_std": float(s.std()), "annualised_vol": float(s.std() * math.sqrt(365)), "mean_daily": float(s.mean()), "sharpe_annual": float(s.mean() / s.std() * math.sqrt(365)), "total_return": float((1 + s).prod() - 1)}
    out["rolling_oos"] = {"sLength": sL, "rebal": reb, "stats": stats}
    md += ["", f"Snippet 16.5's loop on the real daily returns (weights from the previous {sL} days, held {reb} days, then recomputed; {stats['getHRP']['oos_days']} out-of-sample days):", "",
           "| allocation | daily std | annualised vol | annualised Sharpe | total return over the span |", "|---|---|---|---|---|"]
    for k in ("getIVP", "getHRP", "getCLA"):
        s = stats[k]
        md.append(f"| {k[3:]} | {s['daily_std']:.5f} | {s['annualised_vol']:.3f} | {s['sharpe_annual']:.2f} | {s['total_return']:.2f} |")
    # the book's synthetic Monte Carlo
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    st, rep = PD.hrpMC(numIters=cfg["ch16_mc_iters"])
    out["monte_carlo"] = {"iters": cfg["ch16_mc_iters"], "report": {k: {c: float(rep.loc[k, c]) for c in rep.columns} for k in rep.index}}
    md += ["", f"Snippet 16.5's Monte Carlo as printed (520 observations, 10 series, two common and two specific shocks, 260-day estimation, 22-day rebalance; {cfg['ch16_mc_iters']} iterations here, the book uses 10,000; the CLA column is the minimum-variance end point, see book_port_d.py). Venue-independent:", "",
           "| allocation | std of out-of-sample return | variance | variance relative to HRP, minus 1 |", "|---|---|---|---|"]
    for k in rep.index:
        md.append(f"| {k[3:]} | {rep.loc[k, 'std']:.5f} | {rep.loc[k, 'var']:.3e} | {rep.loc[k, 'var_vs_HRP']:+.3f} |")
    # Snippet 16.4 as printed (main) against the book's Table 16.1 (p.146; the draw of `cols` is the one read from the Figure 16.5 caption, see book_port_d.py)
    T161 = {"CLA": [14.44, 19.93, 19.73, 19.87, 18.68, 0.00, 5.86, 1.49, 0.00, 0.00], "HRP": [7.00, 7.59, 10.84, 19.03, 9.72, 10.19, 6.62, 9.10, 7.12, 12.79],
            "IVP": [10.36, 10.28, 10.36, 10.25, 10.31, 9.74, 9.80, 9.65, 9.64, 9.61]}
    np.random.seed(12345)
    x161, cols161 = PD.generateData(10000, 5, 5, .25, cols=[2, 0, 4, 1, 1])
    cov161, corr161 = x161.cov(), x161.corr()
    got161 = {"HRP": PD.getHRP(cov161, corr161).sort_index().values * 100, "IVP": PD.getIVP(cov161) * 100, "CLA": PD.getCLA(cov161) * 100}
    out["table_16_1"] = {k: float(np.max(np.abs(np.asarray(got161[k], float) - np.asarray(T161[k])))) for k in T161}
    md += ["", "Snippet 16.4 as printed on the book's own synthetic data (10,000 observations, 5 + 5 variables), against Table 16.1 of the book (weights in per cent, two decimals): largest absolute difference "
           + ", ".join(f"{k} {v:.3f}" for k, v in out["table_16_1"].items()) + " percentage points (CLA = the stand-in described in book_port_d.py)."]
    return out, md


# ---------------------------------------------------------------------------------------------------------------------
# Chapters 17-19
# ---------------------------------------------------------------------------------------------------------------------
def _daily_log(b):
    c = pd.Series(b.close, index=b.idx)
    d = np.log(c.resample("1D").last().dropna())
    return d


def ch17_18_19(V, cfg, log):
    out, md = {}, {}
    bars = V.bars()
    rng = np.random.default_rng(cfg["seed"])
    e = V.events
    W, MINSL = V.summary["params"]["micro"]["sadf"][0], V.summary["params"]["micro"]["sadf"][1]
    EW, EWORD = V.summary["params"]["micro"]["entropy"]
    sadf_diff, plug_diff, kon_diff, n_s, n_p, n_k = 0., 0., 0., 0, 0, 0
    lz_sizes = []
    per_sym = []
    for sym, b in bars.items():
        d = _daily_log(b)
        x = d.to_numpy()
        day_no = d.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        g = e[e["symbol"] == sym]
        if g.empty:
            continue
        ev_day = pd.DatetimeIndex(pd.to_datetime(g["t0"].to_numpy(), unit="s")).to_numpy().astype("datetime64[D]").astype(np.int64)
        prev = ev_day - 1
        lookup = pd.Series(np.arange(len(day_no)), index=day_no)
        pi = lookup.reindex(prev).to_numpy()
        ok = np.isfinite(pi)
        take = np.flatnonzero(ok)
        if len(take) > cfg["ch17_events_per_symbol"]:
            take = rng.choice(take, cfg["ch17_events_per_symbol"], replace=False)
        for t in take:
            k = int(pi[t])
            row = g.iloc[t]
            # SADF: the book's code on the trailing window of W daily log closes ending at k
            if k >= W - 1 and np.isfinite(row["sadf"]):
                win = pd.DataFrame({"c": x[k - W + 1:k + 1]}, index=d.index[k - W + 1:k + 1])
                v = PC.get_bsadf(win, MINSL, "c", 1)["gsadf"]
                sadf_diff = max(sadf_diff, abs(float(v) - float(row["sadf"])))
                n_s += 1
            # entropy: messages of the last EW daily return signs
            if k >= EW and np.isfinite(row["ent_plugin"]):
                bits = (np.diff(x[k - EW:k + 1]) > 0).astype(int)
                msg = "".join(map(str, bits))
                v_plug = PC.plugIn(msg, EWORD)[0]
                plug_diff = max(plug_diff, abs(v_plug - float(row["ent_plugin"])))
                n_p += 1
                v_kon = PC.konto(msg[::-1])["h"]
                kon_diff = max(kon_diff, abs(v_kon - float(row["ent_konto"])))
                n_k += 1
                if len(lz_sizes) < cfg["ch18_lz_messages"]:
                    lz_sizes.append(len(PC.lempelZiv_lib(msg)))
        per_sym.append(sym)
    out["event_features_recomputed_by_book_code"] = {"sadf": {"events_checked": n_s, "max_abs_diff": sadf_diff}, "ent_plugin": {"events_checked": n_p, "max_abs_diff": plug_diff}, "ent_konto": {"events_checked": n_k, "max_abs_diff": kon_diff},
                                                      "symbols": len(per_sym)}
    out["lempel_ziv_library"] = {"messages": len(lz_sizes), "mean_words": float(np.mean(lz_sizes)) if lz_sizes else None, "min_words": int(min(lz_sizes)) if lz_sizes else None, "max_words": int(max(lz_sizes)) if lz_sizes else None}
    ff = out["event_features_recomputed_by_book_code"]
    lz = out["lempel_ziv_library"]
    md17 = [f"Snippets 17.1-17.4 (the book's get_bsadf, window {W} daily log closes, minimum sample {MINSL}, constant, one lagged difference) recomputed for {ff['sadf']['events_checked']:,} randomly chosen events of {ff['symbols']} symbols and compared with the 'sadf' value stored in the event table: largest absolute difference {ff['sadf']['max_abs_diff']:.2e}."]
    md18 = [f"Snippets 18.1, 18.3, 18.4 (plugIn with word length {EWORD}, konto on the reversed message) recomputed on the last {EW} daily return signs for {ff['ent_plugin']['events_checked']:,} events: largest absolute difference to the stored plug-in entropy {ff['ent_plugin']['max_abs_diff']:.2e}, to the stored Kontoyiannis estimate {ff['ent_konto']['max_abs_diff']:.2e}.",
            f"Snippet 18.2: Lempel-Ziv library of {lz['messages']} such messages has {lz['mean_words']:.1f} words on average (min {lz['min_words']}, max {lz['max_words']}); a message of {EW} random bits has about {EW / math.log2(EW):.0f}-{EW / 3:.0f} words, a periodic one far fewer."] if lz["messages"] else ["Snippet 18.2: no messages."]
    # 19.1 / 19.2 on the hourly reference series
    cs_diff, n_c = 0., 0
    sig_ratio = []
    for sym, b in bars.items():
        if len(b.ref_t) < 200:
            continue
        df = pd.DataFrame({"High": b.ref_high, "Low": b.ref_low}, index=pd.DatetimeIndex(pd.to_datetime(b.ref_t, unit="s")))
        df = df[(df["High"] > 0) & (df["Low"] > 0) & (df["High"] >= df["Low"])]
        sp = PE.corwinSchultz(df, sl=V.summary["params"]["micro"]["cs_sl"])["Spread"]
        cs_ref = pd.Series(MF.corwin_schultz(b.ref_high, b.ref_low, V.summary["params"]["micro"]["cs_sl"]), index=pd.DatetimeIndex(pd.to_datetime(b.ref_t, unit="s")))
        both = pd.concat([sp, cs_ref], axis=1, keys=["book", "project"]).dropna()
        if len(both):
            cs_diff = max(cs_diff, float(np.max(np.abs(both["book"] - both["project"]))))
            n_c += len(both)
        beta, gamma = PE.getBeta(df, 1), PE.getGamma(df)
        sg = PE.getSigma(beta, gamma.reindex(beta.index).dropna())
        lr = np.log(df["High"] * 0 + pd.Series(b.ref_close, index=pd.DatetimeIndex(pd.to_datetime(b.ref_t, unit="s"))).reindex(df.index)).diff()
        blk = pd.concat([sg, lr], axis=1, keys=["hl", "ret"]).dropna()
        if len(blk) > 24 * 60:
            g = (np.arange(len(blk)) // 24)
            agg = blk.groupby(g).agg(hl=("hl", "mean"), ret=("ret", "std")).dropna()
            if len(agg) > 50 and agg["hl"].std() > 0:
                sig_ratio.append(float(agg["hl"].corr(agg["ret"], method="spearman")))
    out["corwin_schultz"] = {"values_compared": n_c, "max_abs_diff_vs_project": cs_diff, "rank_correlation_hl_sigma_vs_close_to_close_std_median": num(np.median(sig_ratio)) if sig_ratio else None, "symbols_for_sigma_check": len(sig_ratio)}
    cs = out["corwin_schultz"]
    md19 = [f"Snippet 19.1 (corwinSchultz, window {V.summary['params']['micro']['cs_sl']} hours) on the hourly reference series of every subset symbol against the project's implementation: {n_c:,} values compared, largest absolute difference {cs_diff:.2e}. "
            f"Snippet 19.2 (high-low volatility): on non-overlapping 24-hour blocks, the median over {cs['symbols_for_sigma_check']} symbols of the rank correlation between the block's mean high-low volatility estimate and the standard deviation of its hourly log returns: {('n/a' if cs['rank_correlation_hl_sigma_vs_close_to_close_std_median'] is None else format(cs['rank_correlation_hl_sigma_vs_close_to_close_std_median'], '.2f'))}."]
    return out, md17, md18, md19


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 20
# ---------------------------------------------------------------------------------------------------------------------
def ch20(V, cfg, log):
    out, md = {}, []
    # 20.1 / 20.2
    d = {"a": [1, 2, 3], "b": ["x", "y"], "c": [True, False]}
    out["cartesian"] = {"unvectorized": len(MP.cartesian_unvectorized(d)), "vectorized": len(MP.cartesian_vectorized(d)), "equal": MP.cartesian_unvectorized(d) == MP.cartesian_vectorized(d)}
    # 20.3 / 20.4 one-touch double barrier on the venue's bar returns: random resampling of its bar log returns into paths
    allr = []
    bars = V.bars()
    for s, b in bars.items():
        lc = np.log(b.close)
        r = np.diff(lc)
        r = r[~np.concatenate([b.brk, [False]])[:len(r)]] if len(b.brk) else r
        allr.append(r)
    allr = np.concatenate(allr)
    rng = np.random.default_rng(cfg["seed"])
    r_mat = np.expm1(rng.choice(allr, size=(cfg["ch20_rows"], cfg["ch20_cols"]), replace=True))
    width = float(np.percentile(np.abs(np.cumsum(np.log1p(r_mat), axis=0)), 90))
    t0 = time.time(); a = MP.barrierTouch(r_mat, width); t_single = time.time() - t0
    t0 = time.time(); b_ = MP.main1(r_mat, numThreads=cfg["threads"], width=width); t_multi = time.time() - t0
    ref = MP.barrier_touch_vectorised(r_mat, width)
    out["one_touch"] = {"rows": cfg["ch20_rows"], "paths": cfg["ch20_cols"], "width_log": width, "touched": len(a), "single_equals_multi": a == b_, "single_equals_vectorised": a == ref,
                        "single_seconds": t_single, "multi_seconds": t_multi, "threads": cfg["threads"], "last_touch_as_printed_differs_on_paths": int(sum(1 for j in a if MP.barrier_touch_as_printed(r_mat[:, j:j + 1], width).get(0) != a[j]))}
    o = out["one_touch"]
    # 20.5 / 20.6
    out["partitions"] = {"linParts_10_atoms_3": MP.linParts(10, 3).tolist(), "nestedParts_10_atoms_3": MP.nestedParts(10, 3).tolist(), "nestedParts_upper_10_3": MP.nestedParts(10, 3, True).tolist()}
    # 20.14 getPCs with 20.12 / 20.13 on daily returns split by symbol into files
    dc = daily_closes(bars)
    r = np.log(dc).diff().iloc[1:].iloc[-1000:]
    r = r.loc[:, r.notna().mean() >= .9].fillna(0.)
    cols = list(r.columns)
    with tempfile.TemporaryDirectory() as td:
        groups = np.array_split(np.arange(len(cols)), min(6, len(cols)))
        names = []
        for gi, g in enumerate(groups):
            fn = f"part{gi}.csv"
            r.iloc[:, g].to_csv(Path(td) / fn)
            names.append(fn)
        cov = r.cov()
        eVal, eVec = np.linalg.eigh(cov.values)
        o_ = eVal.argsort()[::-1]
        eVec = pd.DataFrame(eVec[:, o_][:, :3], index=cols, columns=["PC1", "PC2", "PC3"])
        pcs = MP.mpJobList(MP.getPCs, ("molecules", names), numThreads=min(cfg["threads"], len(names)), mpBatches=1, linMols=True, redux=pd.DataFrame.add, path=td + "/", eVec=eVec)
    direct = pd.DataFrame(r.values @ eVec.values, index=r.index, columns=eVec.columns)
    out["getPCs"] = {"files": len(names), "max_abs_diff_vs_direct": float(np.max(np.abs(pcs.reindex(direct.index).values - direct.values))), "symbols": len(cols)}
    md += [f"Snippets 20.1-20.2: the Cartesian product of three lists (3 x 2 x 2) gives {out['cartesian']['unvectorized']} and {out['cartesian']['vectorized']} combinations, identical: {out['cartesian']['equal']}.",
           f"Snippets 20.3-20.4: first-touch of a +-{width:.3f} log-return barrier in {o['paths']:,} paths of {o['rows']:,} bars resampled from this venue's bar returns: {o['touched']:,} paths touch; the single-thread, {o['threads']}-process and vectorised versions give identical results "
           f"({o['single_equals_multi'] and o['single_equals_vectorised']}); {o['single_seconds']:.1f} s single-thread, {o['multi_seconds']:.1f} s with the pool. As printed, the loop records the LAST row beyond the barrier; that differs from the earliest touch on {o['last_touch_as_printed_differs_on_paths']:,} of the touched paths.",
           f"Snippets 20.5-20.6: linParts(10, 3) = {out['partitions']['linParts_10_atoms_3']}; nestedParts(10, 3) = {out['partitions']['nestedParts_10_atoms_3']}; upper-triangular = {out['partitions']['nestedParts_upper_10_3']}.",
           f"Snippets 20.7-20.14 are used by the other sections (mpPandasObj in 10.x, processJobs in 4.9 and 10.x). Snippet 20.14 (getPCs with on-the-fly reduction): principal components of {len(cols)} symbols' daily returns computed from {len(names)} files in parallel; largest difference to the direct matrix product {out['getPCs']['max_abs_diff_vs_direct']:.1e}."]
    return out, md


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 21
# ---------------------------------------------------------------------------------------------------------------------
def ch21(V, cfg, log):
    out, md = {}, []
    bars = V.bars()
    top = [s for s in V.by_events[:cfg["ch21_assets"]] if s in bars]
    dc = daily_closes({s: bars[s] for s in top})          # the calendar of these assets only
    r = np.log(dc[top]).diff().dropna()
    if len(r) < cfg["ch21_days"]:
        raise RuntimeError(f"only {len(r)} common days for {top}")
    H = cfg["ch21_horizon"]
    rr = r.iloc[-cfg["ch21_days"]:]
    cuts = np.linspace(0, len(rr), H + 1).astype(int)
    blocks = [rr.iloc[cuts[i]:cuts[i + 1]] for i in range(H)]
    np.random.seed(cfg["seed"])
    params = []
    for blk in blocks:
        mean_ = blk.mean().to_numpy().reshape(-1, 1)
        cov_ = np.cov(blk.to_numpy(), rowvar=False)
        c_ = np.random.uniform(size=cov_.shape[0]) * np.diag(cov_) ** .5
        params.append({"mean": mean_, "cov": cov_, "c": c_})
    t0 = time.time()
    w_stat, sr_stat = PD.staticSolution(params)
    w_dyn, sr_dyn = PD.dynamicSolution(params)
    secs = time.time() - t0
    out = {"assets": top, "horizon": H, "block_days": [int(len(b)) for b in blocks], "trajectories": int(PD.getAllWeights(len(top), len(top)).shape[1] ** H),
           "weight_columns_printed_generator": int(PD.getAllWeights(len(top), len(top)).shape[1]), "weight_vectors_distinct": int(len(np.unique(PD.getAllWeights(len(top), len(top)).T, axis=0))),
           "trajectories_distinct": int(len(np.unique(PD.getAllWeights(len(top), len(top)).T, axis=0)) ** H), "static_sr": float(sr_stat), "dynamic_sr": float(sr_dyn),
           "static_weights": w_stat.round(4).tolist(), "dynamic_weights": w_dyn.round(4).tolist(), "seconds": secs, "mean_daily": [p["mean"].ravel().tolist() for p in params],
           "cost_coefficients": [p["c"].tolist() for p in params]}
    md += [f"Snippets 21.1-21.7 on the {len(top)} most active symbols ({', '.join(top)}), {H} consecutive blocks of daily log returns ({', '.join(str(n) for n in out['block_days'])} days; means and covariances per block), weights on a grid of 1/{len(top)} with total absolute weight 1 "
           f"(the printed generator returns {out['weight_columns_printed_generator']} weight columns, of which {out['weight_vectors_distinct']} are distinct; {out['trajectories']:,} trajectories searched as printed, {out['trajectories_distinct']:,} distinct; exhaustive search took {secs:.0f} s). The trading-cost coefficients are the book's random draw (uniform times the block volatility, seed {cfg['seed']}). "
           f"Static solution (Snippet 21.6, re-solved each block, unconstrained): Sharpe {sr_stat:.3f}; dynamic solution (Snippet 21.7, best trajectory including trading costs): Sharpe {sr_dyn:.3f}. These are in-sample numbers for the blocks they are computed on."]
    return out, md
