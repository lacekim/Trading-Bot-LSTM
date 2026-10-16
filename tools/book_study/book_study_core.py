"""Core of the book-procedure study: events from candle history, market state, the Chapter 13 procedure inside a
combinatorial purged cross-validation, and the summary tables.  No project imports: it takes plain pandas objects.

Procedure (all choices are declared before any result is looked at; see DEVIATIONS for where the book is not followed exactly):
  1. Events (Ch.2): symmetric CUSUM on hourly log returns with a causal threshold of one daily sigma.
  2. Each event's forward log-price path over H+1 bars, in units of the causal one-bar sigma (Ch.3 'trgt' idea; Ch.13 sigma units).
  3. Groups and splits (Ch.12): N equal-duration time groups, k test groups -> C(N,k) splits, C(N-1,k-1) backtest paths.
  4. For each split and each (state, direction): purge + embargo the training events (Ch.7), estimate the O-U drift and
     phi from the training paths with uniqueness weights (Ch.4 + Ch.13 step 1), simulate 100,000 paths (step 3), search the
     20 x 20 stop/profit mesh (step 2 and 4), pick the highest-Sharpe node (step 5a), then apply that single rule to the
     held-out test events.  Nothing about the test events is used to choose the rule.
  5. Definition 2 (Ch.13): the chosen rule is 'overfit' if its test Sharpe is below the median test Sharpe of the whole mesh.
  6. Paths of daily results -> Sharpe per path, PSR and DSR (Ch.14), with the number of cells examined as the trial count.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

import book_stats as B

STATES = ("all", "above", "below")
DIRECTIONS = ("long", "short")
SPAN = 336                       # bars (14 days) for the causal one-bar sigma
JUNE_2026 = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01"))     # [start, end)

DEVIATIONS = [
    "The book takes E0[P_T] (the strategy's price forecast) as an input; a plain control has no forecast, so the mean level is not "
    "identified when phi is close to 1. I estimate the per-step drift a = (1 - phi) * E0[P_T] directly from the training events: "
    "same O-U process, different parametrisation.",
    "Prices are the log-price change since entry divided by the event's causal one-bar sigma (EWM std of hourly log returns, "
    "span 336), so barriers are in the book's sigma units and symbols are pooled. Pooling across symbols is mine (the book "
    "estimates per opportunity).",
    "Barriers are checked on bar closes, as in Snippet 13.2. The live simulator checks intrabar highs and lows with ATR barriers; "
    "results here are therefore not the live simulator's numbers.",
    "The simulation uses no trading costs (as in Chapter 13). Costs (round trip, in the project's configured basis points) are "
    "subtracted afterwards from each test event, and both gross and net results are reported.",
    "Events are sampled with a CUSUM filter whose threshold is one causal daily sigma (the book's example uses a fixed h).",
    "Embargo is 1% of the whole time span, applied in time rather than in observations.",
    "Events overlap in time; training uses Chapter 4 average uniqueness within each symbol. Different symbols are not independent "
    "(crypto is highly correlated), so the headline Sharpe is computed on one number per calendar day (the mean over that day's "
    "events), and PSR's sample size is days / ((H+1)/24) when the horizon exceeds a day.",
    "Maximum holding horizons H of 8, 24 and 100 bars are all run (the book's example uses 100; the live system's maximum hold is 8); "
    "all are counted as separate trials in the deflated Sharpe ratio.",
    "Events whose forward window runs past the end of the data, or crosses a gap in the hourly candles, are dropped.",
    "Survivorship: the symbols are the ones with candle files today; assets that were delisted or dropped are absent, so long-side "
    "results are likely flattered, and the equal-weight market index is built from the same survivors and changes composition as "
    "symbols start (it is a rebased mean of hourly log returns of whichever symbols have a bar that hour, at least 10).",
    "The deflated Sharpe uses the book's variance across the trial Sharpe ratios. The long and short rows are mirror images on the same "
    "events, so when one is good the other is bad and that variance is inflated; the deflated Sharpe is therefore conservative here. "
    "PSR and the expected-maximum Sharpe are printed next to it so the reader can judge.",
    "Entry is at the close of the event bar and exit at a later bar close, as in the book's triple-barrier labels; no order delay is modelled.",
]


# ----------------------------------------------------------------------------------------------------------
# plain momentum / trend signals (no fitted parameters; every lookback is a trial)
# ----------------------------------------------------------------------------------------------------------
SIGNALS = {"mom168": 168, "mom480": 480, "mom720": 720, "mom960": 960, "mom2160": 2160, "ma200d": None}
SIGNAL_NOTE = ("mom<L>: +1 (long) if the log return over the last L hourly bars is positive, -1 (short) if negative, 0 (no signal) if "
               "there are fewer than L earlier bars or the return is exactly zero. ma200d: +1 if the close is above its own symbol's "
               "200-day mean (time window, at least 3,840 bars), -1 below, 0 if not available.")


def signal_at(name: str, close: pd.Series, logc: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Sign (+1/-1/0) of a plain momentum / trend rule at bar positions `pos`, using only bars up to and including `pos`."""
    if name == "ma200d":
        ma = close.rolling("200D", min_periods=3840).mean().to_numpy()
        m = ma[pos]
        cl = close.to_numpy()[pos]
        return np.where(np.isfinite(m), np.sign(cl - m), 0).astype(np.int8)
    lookback = SIGNALS[name]
    ok = pos >= lookback
    ret = np.where(ok, logc[pos] - logc[np.maximum(pos - lookback, 0)], 0.0)
    return np.sign(ret).astype(np.int8)


# ----------------------------------------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------------------------------------
def to_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    return pd.DatetimeIndex(index).to_numpy(dtype="datetime64[s]").astype("int64")


def build_events(closes: dict, h_max: int, span: int = SPAN, log=print, signals: tuple = ()) -> dict:
    """Events and their forward paths for every symbol in `closes` ({symbol: hourly close Series with a DatetimeIndex}).

    Returns arrays: sym (id), t0 (epoch seconds), pos (bar position in its symbol's series), sigma (one-bar sigma at the event),
    q (n, h_max+1) float32 = (log C[pos+t] - log C[pos]) / sigma, symbols (list), per_symbol (list of dicts) and n_bars (per symbol).
    signals: names from SIGNALS; each adds ev['sig_<name>'] (int8: +1 long, -1 short, 0 = no signal) from data known at the event bar."""
    symbols = sorted(closes)
    cols = {"sym": [], "t0": [], "pos": [], "sigma": []}
    for nm in signals:
        cols["sig_" + nm] = []
    q_parts, per_symbol, n_bars = [], [], []
    steps = np.arange(1, h_max + 2)
    for sid, sym in enumerate(symbols):
        c = closes[sym].astype(float)
        c = c[c > 0]
        n = len(c)
        n_bars.append(n)
        tsec = to_seconds(c.index)
        logc = np.log(c.to_numpy())
        r = np.concatenate([[np.nan], np.diff(logc)])
        sigma = pd.Series(r).ewm(span=span, min_periods=span).std().to_numpy()
        h = sigma * math.sqrt(24.0)
        ev = B.cusum_events(r, h)
        total = len(ev)
        ok = (ev + h_max + 1 < n) & np.isfinite(sigma[ev]) & (sigma[ev] > 0)
        ev_end = ev[ok]
        gap_ok = (tsec[ev_end + h_max + 1] - tsec[ev_end]) == (h_max + 1) * 3600
        kept = ev_end[gap_ok]
        per_symbol.append({"symbol": sym, "bars": n, "first": str(c.index[0]), "last": str(c.index[-1]),
                           "events_total": total, "dropped_end_or_sigma": int(total - ok.sum()),
                           "dropped_gap": int((~gap_ok).sum()), "events_kept": int(len(kept))})
        if len(kept):
            q = (logc[kept[:, None] + steps[None, :]] - logc[kept][:, None]) / sigma[kept][:, None]
            q_parts.append(q.astype(np.float32))
            cols["sym"].append(np.full(len(kept), sid, dtype=np.int32))
            cols["t0"].append(tsec[kept])
            cols["pos"].append(kept.astype(np.int64))
            cols["sigma"].append(sigma[kept])
            for nm in signals:
                cols["sig_" + nm].append(signal_at(nm, c, logc, kept))
        log(f"  {sym}: bars={n} events={total} kept={len(kept)}")
    if not q_parts:
        raise RuntimeError("no events were produced")
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["q"] = np.concatenate(q_parts, axis=0)
    out.update({"symbols": symbols, "per_symbol": per_symbol, "n_bars": n_bars, "h_max": h_max})
    order = np.argsort(out["t0"], kind="stable")
    for k in ("sym", "t0", "pos", "sigma", "q") + tuple("sig_" + nm for nm in signals):
        out[k] = out[k][order]
    return out


def event_uniqueness(ev: dict, horizon: int) -> np.ndarray:
    """Chapter 4 average uniqueness of every event whose label spans bars pos .. pos+H+1 of its own symbol."""
    u = np.empty(len(ev["pos"]), dtype=np.float64)
    for sid, n in enumerate(ev["n_bars"]):
        m = ev["sym"] == sid
        if m.any():
            u[m] = B.average_uniqueness(ev["pos"][m], ev["pos"][m] + horizon + 1, n)
    return u


def market_state(closes: dict, t0_sec: np.ndarray, window: str = "200D", min_periods: int = 3840, min_symbols: int = 10):
    """State per event: 1 if the equal-weight index is above its own `window` mean at the event bar, 0 if not, -1 if the average
    is not yet available. The index moves by the mean hourly log return of the symbols that have a return at that hour
    (hours with fewer than min_symbols are dropped)."""
    rets = {}
    for sym, c in closes.items():
        c = c[c > 0].astype(float)
        rets[sym] = np.log(c).diff()
    frame = pd.DataFrame(rets)
    counts = frame.notna().sum(axis=1)
    mean_ret = frame.mean(axis=1)[counts >= min_symbols]
    index = np.exp(mean_ret.cumsum())
    ma = index.rolling(window, min_periods=min_periods).mean()
    above = (index > ma).astype(float).where(ma.notna())
    ev_idx = pd.DatetimeIndex(pd.to_datetime(t0_sec, unit="s"))
    st = above.reindex(ev_idx)
    state = np.where(st.isna().to_numpy(), -1, st.to_numpy(dtype=float)).astype(np.int8)
    return state, index, ma


# ----------------------------------------------------------------------------------------------------------
# the study
# ----------------------------------------------------------------------------------------------------------
def state_mask(state: np.ndarray, name: str) -> np.ndarray:
    if name == "all":
        return np.ones(len(state), dtype=bool)
    return state == (1 if name == "above" else 0)


def run_study(ev: dict, state: np.ndarray, params: dict, log=print) -> dict:
    """Runs every split / state / direction / horizon. Returns dict of DataFrames: cells, daily, samples, events_index, meta."""
    t_start = time.time()
    n_groups, k = int(params["n_groups"]), int(params["k"])
    horizons = [int(h) for h in params["horizons"]]
    h_max = max(horizons)
    if h_max > ev["h_max"]:
        raise ValueError("events were built with a shorter maximum horizon")
    cost_frac = float(params["cost_frac"])
    min_train, min_test = int(params["min_train"]), int(params["min_test"])
    n_paths_sim = int(params["n_paths"])
    rng = np.random.default_rng(int(params["seed"]))
    eps_all = rng.standard_normal((n_paths_sim, h_max + 1))
    placebo_sign = np.where(np.random.default_rng(int(params["seed"]) + 1).random(len(ev["t0"])) < 0.5, -1.0, 1.0)
    signal_names = [str(x) for x in params.get("signals", [])]
    only_signals = bool(params.get("only_signals", False))
    n_ev = len(ev["t0"])
    sign_all = {"long": np.ones(n_ev), "short": -np.ones(n_ev), "placebo": placebo_sign}
    for nm in signal_names:
        sign_all[nm] = ev["sig_" + nm].astype(np.float64)

    t0 = ev["t0"].astype(np.float64)
    t_min, t_max = float(t0.min()), float(t0.max()) + 1.0
    edges = B.group_edges(t_min, t_max, n_groups)
    groups = B.assign_groups(t0, edges)
    embargo = float(params["embargo_pct"]) * (t_max - t_min)
    splits = B.cpcv_splits(n_groups, k)
    assign, n_paths = B.cpcv_paths(n_groups, k)
    sigma = ev["sigma"].astype(np.float64)
    day = (ev["t0"] // 86400).astype(np.int64)

    cells, daily_parts, samples = [], [], []
    sample_splits = {0, len(splits) // 2, len(splits) - 1}
    sample_rng = np.random.default_rng(int(params["seed"]) + 2)
    rows_spec = ([] if only_signals else [(s, d) for s in STATES for d in DIRECTIONS]) + [(s, nm) for nm in signal_names for s in STATES] + [("all", "placebo")]
    row_names = ([] if only_signals else list(DIRECTIONS)) + signal_names + ["placebo"]
    log(f"events={len(t0)} groups={n_groups} k={k} splits={len(splits)} paths={n_paths} embargo_days={embargo / 86400:.1f}")

    for h in horizons:
        q_h = ev["q"][:, : h + 1]
        u_h = event_uniqueness(ev, h)
        t1 = t0 + (h + 1) * 3600.0
        eps = eps_all[:, : h + 1]
        for si, test_groups in enumerate(splits):
            in_test_group = np.isin(groups, list(test_groups))
            train_ok = B.train_mask(t0, t1, groups, test_groups, edges, embargo)
            for sname, dname in rows_spec:
                sm = state_mask(state, sname)
                has = sign_all[dname] != 0
                tr = train_ok & sm & has
                te = in_test_group & sm & has
                base = {"horizon": h, "split": si, "test_groups": "-".join(map(str, test_groups)),
                        "state": sname, "direction": dname, "n_train": int(tr.sum()), "n_test": int(te.sum())}
                if tr.sum() < min_train or te.sum() < min_test:
                    cells.append({**base, "skipped": True})
                    continue
                sign_tr = sign_all[dname][tr]
                sign_te = sign_all[dname][te]
                s_tr = (q_h[tr] * sign_tr[:, None]).astype(np.float32)
                est = B.estimate_ou(s_tr, u_h[tr])
                sim = B.simulate_ou_paths(est["a"], est["phi"], eps)
                mesh = B.mesh_exit_stats(sim)
                node = B.best_node(mesh["sharpe"])
                del sim, s_tr
                if node is None:
                    cells.append({**base, "skipped": True})
                    continue
                pt, sl, sim_sr = node
                s_te = (q_h[te] * sign_te[:, None]).astype(np.float32)
                cost_sigma = cost_frac / sigma[te]
                mt = B.mesh_exit_stats(s_te, cost=cost_sigma)
                value, step, kind = B.first_exit(s_te, pt, sl)
                value = value.astype(np.float64)
                gross_bp = value * sigma[te] * 1e4
                net_bp = gross_bp - cost_frac * 1e4
                i_star, j_star = B.MESH.index(pt), B.MESH.index(sl)
                med_g, med_n = float(np.nanmedian(mt["sharpe"])), float(np.nanmedian(mt["sharpe_net"]))
                sr_g, sr_n = float(mt["sharpe"][i_star, j_star]), float(mt["sharpe_net"][i_star, j_star])
                barrier = kind != 0
                p_obs = float((kind == 1).sum() / barrier.sum()) if barrier.any() else float("nan")
                cells.append({
                    **base, "skipped": False, "a": est["a"], "phi": est["phi"], "sigma_fit": est["sigma"],
                    "half_life_bars": B.half_life(est["phi"]), "pt": pt, "sl": sl, "sim_sharpe": sim_sr,
                    "test_sharpe_gross": sr_g, "test_sharpe_net": sr_n,
                    "mesh_median_sharpe_gross": med_g, "mesh_median_sharpe_net": med_n,
                    "overfit_gross": bool(sr_g < med_g), "overfit_net": bool(sr_n < med_n),
                    "test_mean_gross_bp": float(gross_bp.mean()), "test_mean_net_bp": float(net_bp.mean()),
                    "observed_precision": p_obs,
                    "break_even_precision": B.break_even_precision(pt, sl, float(cost_sigma.mean())),
                    "share_vertical_exit": float((kind == 0).mean()), "mean_steps": float(step.mean()),
                })
                ev_path = np.array([assign[(si, int(g))] for g in groups[te]], dtype=np.int64)
                df = pd.DataFrame({"path": ev_path, "day": day[te], "gross": gross_bp, "net": net_bp})
                df["net_sq"] = df["net"] ** 2
                agg = df.groupby(["path", "day"], sort=True).agg(
                    n=("net", "size"), sum_gross_bp=("gross", "sum"), sum_net_bp=("net", "sum"), sumsq_net_bp=("net_sq", "sum")).reset_index()
                agg.insert(0, "direction", dname)
                agg.insert(0, "state", sname)
                agg.insert(0, "horizon", h)
                daily_parts.append(agg)
                if si in sample_splits:
                    te_idx = np.flatnonzero(te)
                    pick = sample_rng.choice(len(te_idx), size=min(int(params["n_sample"]), len(te_idx)), replace=False)
                    for pi in pick:
                        gi = te_idx[pi]
                        samples.append({
                            "horizon": h, "split": si, "state": sname, "direction": dname, "symbol": ev["symbols"][int(ev["sym"][gi])],
                            "t0_epoch": int(ev["t0"][gi]), "pos": int(ev["pos"][gi]), "sigma": float(sigma[gi]),
                            "sign": float(sign_te[pi]), "pt": pt, "sl": sl,
                            "exit_value": float(value[pi]), "exit_step": int(step[pi]), "exit_kind": int(kind[pi]),
                            "gross_bp": float(gross_bp[pi]), "net_bp": float(net_bp[pi]),
                        })
            log(f"  H={h} split {si + 1}/{len(splits)} done ({time.time() - t_start:.0f}s)")
    return {
        "cells": pd.DataFrame(cells),
        "daily": pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame(),
        "samples": pd.DataFrame(samples),
        "events_index": pd.DataFrame({"symbol": [ev["symbols"][i] for i in ev["sym"]], "t0_epoch": ev["t0"], "pos": ev["pos"],
                                       "sigma": ev["sigma"], "group": groups, "state": state,
                                       **{"sig_" + nm: ev["sig_" + nm] for nm in signal_names}}),
        "meta": {"edges": [float(e) for e in edges], "embargo_seconds": embargo, "n_groups": n_groups, "k": k,
                 "n_splits": len(splits), "n_paths": n_paths, "horizons": horizons, "cost_frac": cost_frac,
                 "n_sim_paths": n_paths_sim, "seed": int(params["seed"]), "row_names": row_names, "signals": signal_names,
                 "only_signals": only_signals, "min_train": min_train, "min_test": min_test,
                 "events": int(len(t0)), "runtime_seconds": time.time() - t_start},
    }


# ----------------------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------------------
def _in_window(day_idx: np.ndarray, window) -> np.ndarray:
    lo = int(pd.Timestamp(window[0]).timestamp() // 86400)
    hi = int(pd.Timestamp(window[1]).timestamp() // 86400)
    return (day_idx >= lo) & (day_idx < hi)


def summarize(result: dict, windows: dict) -> dict:
    """All headline numbers, as a JSON-able dict. windows = {name: (start, end_exclusive)}."""
    cells, daily, meta = result["cells"], result["daily"], result["meta"]
    out = {"rows": {}, "trials": {}}
    done = cells[~cells["skipped"].astype(bool)] if len(cells) else cells
    sr_net_by_row = {}
    detail = {}
    for (h, sname, dname), grp in daily.groupby(["horizon", "state", "direction"]):
        per_path = []
        for path, g in grp.groupby("path"):
            g = g.sort_values("day")
            d_net = (g["sum_net_bp"] / g["n"]).to_numpy()
            d_gross = (g["sum_gross_bp"] / g["n"]).to_numpy()
            mom = B.sharpe_moments(d_net)
            momg = B.sharpe_moments(d_gross)
            t_eff = mom["n"] / max(1.0, (h + 1) / 24.0)
            per_path.append({
                "path": int(path), "events": int(g["n"].sum()), "days": int(len(g)), "t_eff": t_eff,
                "mean_net_bp": float(g["sum_net_bp"].sum() / g["n"].sum()), "mean_gross_bp": float(g["sum_gross_bp"].sum() / g["n"].sum()),
                "sr_net": mom["sr"], "sr_gross": momg["sr"], "skew": mom["skew"], "kurt": mom["kurt"],
                "psr_net": B.psr(mom["sr"], t_eff, mom["skew"], mom["kurt"]),
                "psr_gross": B.psr(momg["sr"], t_eff, momg["skew"], momg["kurt"]),
            })
        pp = pd.DataFrame(per_path)
        key = f"H{h}|{sname}|{dname}"
        row = {
            "horizon": int(h), "state": sname, "direction": dname, "paths": int(len(pp)),
            "events_per_path": float(pp["events"].mean()), "days_per_path": float(pp["days"].mean()),
            "mean_net_bp": float(pp["mean_net_bp"].mean()), "mean_gross_bp": float(pp["mean_gross_bp"].mean()),
            "sr_net_mean": float(pp["sr_net"].mean()), "sr_net_sd": float(pp["sr_net"].std(ddof=1)) if len(pp) > 1 else float("nan"),
            "sr_gross_mean": float(pp["sr_gross"].mean()),
            "paths_sr_net_negative": int((pp["sr_net"] < 0).sum()),
            "psr_net_mean": float(pp["psr_net"].mean()), "psr_gross_mean": float(pp["psr_gross"].mean()),
            "t_eff_mean": float(pp["t_eff"].mean()), "skew_mean": float(pp["skew"].mean()), "kurt_mean": float(pp["kurt"].mean()),
        }
        sub = done[(done["horizon"] == h) & (done["state"] == sname) & (done["direction"] == dname)]
        if len(sub):
            modal = sub.groupby(["pt", "sl"]).size().sort_values(ascending=False)
            row.update({
                "splits_run": int(len(sub)), "modal_rule": [float(modal.index[0][0]), float(modal.index[0][1])],
                "modal_rule_share": float(modal.iloc[0] / len(sub)), "sim_sharpe_mean": float(sub["sim_sharpe"].mean()),
                "phi_mean": float(sub["phi"].mean()), "a_mean": float(sub["a"].mean()),
                "test_sharpe_gross_mean": float(sub["test_sharpe_gross"].mean()), "test_sharpe_net_mean": float(sub["test_sharpe_net"].mean()),
                "overfit_gross_share": float(sub["overfit_gross"].mean()), "overfit_net_share": float(sub["overfit_net"].mean()),
                "observed_precision_mean": float(sub["observed_precision"].mean()),
                "break_even_precision_mean": float(sub["break_even_precision"].mean()),
                "share_vertical_exit_mean": float(sub["share_vertical_exit"].mean()),
            })
        wsum = {}
        for wname, wr in windows.items():
            gw = grp[_in_window(grp["day"].to_numpy(), wr)]
            if len(gw):
                per = gw.groupby("path").agg(n=("n", "sum"), net=("sum_net_bp", "sum"), gross=("sum_gross_bp", "sum"))
                wsum[wname] = {"events_per_path": float(per["n"].mean()), "mean_net_bp": float((per["net"] / per["n"]).mean()),
                               "mean_gross_bp": float((per["gross"] / per["n"]).mean()), "paths": int(len(per))}
            else:
                wsum[wname] = None
        row["windows"] = wsum
        out["rows"][key] = row
        if dname != "placebo":
            sr_net_by_row[key] = row["sr_net_mean"]
            detail[key] = (row["t_eff_mean"], row["skew_mean"], row["kurt_mean"])
    vals = np.array([v for v in sr_net_by_row.values() if np.isfinite(v)])
    n_trials = int(len(vals))
    var_sr = float(vals.var(ddof=1)) if n_trials > 1 else float("nan")
    out["trials"] = {"n": n_trials, "var_sr_net": var_sr, "expected_max_sr": B.expected_max_sharpe(n_trials, var_sr),
                     "note": "trials = every (horizon, state, direction) cell reported, placebo excluded"}
    for key, (t_eff, skew, kurt) in detail.items():
        out["rows"][key]["dsr_net"] = B.dsr(out["rows"][key]["sr_net_mean"], t_eff, skew, kurt, n_trials, var_sr)
    out["meta"] = meta
    out["windows"] = {k: [str(v[0]), str(v[1])] for k, v in windows.items()}
    return out


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.{nd}f}"


def render_markdown(summary: dict, symbols_info: pd.DataFrame, run_info: dict) -> str:
    L = []
    sigs = summary["meta"].get("signals", [])
    L.append("# Book-procedure study: " + ("plain momentum / trend rules" if summary["meta"].get("only_signals") else "always-long and always-short controls") + " by market state")
    L.append("")
    L.append("Every number below comes from `summary_numbers.json` / `cells.csv` / `oos_daily.csv` in this folder. Sharpe ratios are per day "
             "(not annualised). 'net' subtracts the configured round-trip cost from every event; 'gross' does not.")
    L.append("")
    meta = summary["meta"]
    L.append(f"- Events: {meta['events']:,}; groups N={meta['n_groups']}, test groups k={meta['k']}, splits={meta['n_splits']}, "
             f"backtest paths={meta['n_paths']}; simulated paths per rule search={meta['n_sim_paths']:,}; seed={meta['seed']}.")
    L.append(f"- Round-trip cost used: {meta['cost_frac'] * 1e4:.1f} bp. Trials counted for the deflated Sharpe: {summary['trials']['n']} "
             f"(expected maximum Sharpe of that many unskilled trials = {_f(summary['trials']['expected_max_sr'], 3)} per day).")
    L.append(f"- Windows: " + "; ".join(f"{k} = {v[0][:10]} to {v[1][:10]}" for k, v in summary["windows"].items()))
    L.append(f"- Symbols used: {len(symbols_info)}; events kept per symbol are in `symbols.csv`.")
    L.append("")
    rows = summary["rows"]
    for h in meta["horizons"]:
        L.append(f"## Horizon H = {h} bars")
        L.append("")
        L.append("| state | direction | events/path | gross bp | net bp | daily SR net (mean +/- sd over paths) | paths SR<0 | PSR net | DSR net | rule most chosen (pt, sl) | OOS SR net vs mesh median: overfit share | barrier-exit precision observed vs break-even | vertical-exit share |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        names = [n for n in meta.get("row_names", list(DIRECTIONS) + ["placebo"]) if n != "placebo"]
        for s in list(STATES):
            for d in names + (["placebo"] if s == "all" else []):
                r = rows.get(f"H{h}|{s}|{d}")
                if r is None:
                    L.append(f"| {s} | {d} | no data | | | | | | | | | |")
                    continue
                mr = r.get("modal_rule")
                L.append("| {s} | {d} | {ev:.0f} | {g} | {n} | {sr} +/- {sd} | {neg}/{pp} | {psr} | {dsr} | {mr} | {ov} | {po} vs {be} | {vx} |".format(
                    s=s, d=d, ev=r["events_per_path"], g=_f(r["mean_gross_bp"], 1), n=_f(r["mean_net_bp"], 1),
                    sr=_f(r["sr_net_mean"], 3), sd=_f(r["sr_net_sd"], 3), neg=r["paths_sr_net_negative"], pp=r["paths"],
                    psr=_f(r["psr_net_mean"], 2), dsr=_f(r.get("dsr_net"), 2) if d != "placebo" else "-",
                    mr=("(%.1f, %.1f) %d%%" % (mr[0], mr[1], round(r["modal_rule_share"] * 100))) if mr else "n/a",
                    ov=_f(r.get("overfit_net_share"), 2), po=_f(r.get("observed_precision_mean"), 3), be=_f(r.get("break_even_precision_mean"), 3), vx=_f(r.get("share_vertical_exit_mean"), 2)))
        L.append("")
        L.append("Window lines (same paths; mean over paths of the events entering inside the window):")
        L.append("")
        L.append("| state | direction | " + " | ".join(f"{w}: events/path, net bp" for w in summary["windows"]) + " |")
        L.append("|---|---|" + "---|" * len(summary["windows"]))
        for s in list(STATES):
            for d in names:
                r = rows.get(f"H{h}|{s}|{d}")
                if r is None:
                    continue
                cells = []
                for w in summary["windows"]:
                    x = r["windows"].get(w)
                    cells.append("none" if x is None else f"{x['events_per_path']:.0f}, {x['mean_net_bp']:.1f}")
                L.append(f"| {s} | {d} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("## Reading guide")
    L.append("")
    L.append("- A rule is only worth keeping if its net daily Sharpe is positive on most paths AND its deflated Sharpe is high AND the Definition-2 overfit share is well below one half. The placebo row (random sign per event) is the baseline for what the chosen stop/profit rule earns by itself, whatever the direction: on a pure random walk it is zero before costs; where paths keep going (or partly reverse) after an event it is not, because a cut-losers/ride-winners rule earns on the trendiness of the path whatever the sign. A direction's edge is its row minus the placebo row, and it has to survive the cost.")
    L.append("- 'observed vs break-even precision': share of barrier exits that were profit-takings, against the share needed for zero net profit with that rule's barriers and the cost (Chapter 15).")
    L.append("")
    L.append("## Where this does not follow the book exactly")
    L.append("")
    devs = list(DEVIATIONS)
    if sigs:
        devs.append("Signals tested: " + SIGNAL_NOTE + " The signal is read only on CUSUM event bars (a large recent move has just happened), "
                    "so this is momentum conditional on an event, not a rebalancing-date momentum portfolio. The sign is the plain sign of the "
                    "return, not the volatility-scaled sum of the published time-series-momentum rule. Every (signal, state, horizon) row is a trial.")
    for i, d in enumerate(devs, 1):
        L.append(f"{i}. {d}")
    L.append("")
    L.append(f"Runtime: {run_info.get('runtime_seconds', 0):.0f} s.")
    return "\n".join(L) + "\n"
