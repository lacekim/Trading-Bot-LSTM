"""Meta-labeling inputs from DOLLAR BARS (Chapter 2's volume clock) instead of hourly candles.  Same book chapters and the same output table as meta_events.build_meta_events
(Ch.2 CUSUM events, Ch.3 triple-barrier labels, Ch.4 uniqueness / attribution weights, Ch.5 FFD, Ch.17-19 features); only the places that assumed 'one bar = one clock hour' differ.
meta_events.py / book_study_core.py are NOT edited (their md5 records stay valid); the functions below replace them for bar series with irregular durations.

WHAT CHANGES, AND WHY (declared before any result was seen):
  bar time        a bar is stamped with its END (build_dollar_bars.py).  Everything known at the stamp is at or before it.
  CUSUM filter    unchanged (Snippet 2.4) on bar log returns.  Threshold = the causal one-bar sigma (EWM std, span 336 bars) x sqrt(bars per calendar day), where bars per calendar day is the average over
                  the trailing min(30 days, history) (hourly bars: exactly 24, as before).  A return across a data hole is set to missing (CUSUM counts missing as 0, the EWM skips it).
  daily vol       Snippet 3.1 as the book writes it: for each bar the return from the last bar BEFORE (t - 1 calendar day), EWM std span 100 (needs 100 values).  Missing when that earlier bar is
                  older than the venue's hole limit.
  vertical barrier  Snippet 3.4's way: a TIME limit, H_DAYS calendar days after the event (the first bar that ends at or after t0 + H_DAYS).  At most HCAP bars are followed; a path that would be longer
                  is stopped at HCAP bars (counted and reported).  An event is dropped when no bar exists at or after the vertical time, or when a data hole lies between the event and its exit.
  exit time       t1 = the end time of the exit bar (not t0 + k hours).
  momentum        lookbacks stay in BARS (24 / 168 / 720 / 2160 bars = a fixed amount of traded value, the volume clock's own unit); a lookback that reaches back across a data hole is missing, so the event is dropped.
  market state    equal-weight index of hourly closes from the same rows (build_dollar_bars' ref_1h series; >= 10 symbols with a return in the hour) against its own 200-day mean; the mean needs
                  160 days of history (the 3,840 hourly bars of the earlier runs, as a time).  Read at the last completed hour at or before the event; missing when that is more than 6 hours old.
  Corwin-Schultz  Snippet 19.1 is defined on equal time intervals, so it is computed on the hourly reference series (24-row rolling mean) and read at the last completed hour at or before the event.
  daily features  (FFD, SADF, entropy) unchanged: built from the last bar close of each UTC day, read at the last completed UTC day.  Days without any bar are simply absent from that series.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import book_stats as B
import meta_events as E
import micro_features as MF

FEATURES = E.FEATURES
SPAN, SLOW_SPAN = 336, 2016
SIDE_SIGNAL_LOOKBACK = 720
LOOKBACKS = ((24, "mom24n"), (168, "mom168n"), (720, "mom720n"), (2160, "mom2160n"))
NDAY_WINDOW_S = 30 * 86400
STATE_TOL_S = 6 * 3600
MKT_WINDOW = "200D"
MKT_MIN_HISTORY_S = 160 * 86400
MKT_MIN_SYMBOLS = 10
H_DAYS = 4
DAY_S = 86400


def bars_per_day(t: np.ndarray) -> np.ndarray:
    """Average number of bars per calendar day over the trailing min(30 days, t - first stamp), counting bars with a stamp in (t - W, t].  NaN at the first bar."""
    n = len(t)
    w = np.minimum(NDAY_WINDOW_S, (t - t[0]).astype(np.float64))
    lo = np.searchsorted(t, t - w, side="right")                    # bars with stamp <= t - W are outside (t - W, t]
    cnt = (np.arange(n) + 1 - lo).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = cnt / (w / DAY_S)
    out[w <= 0] = np.nan
    return out


def daily_vol_time(t: np.ndarray, close: np.ndarray, break_s: int, span: int = 100) -> np.ndarray:
    """Snippet 3.1: return from the last bar at or before (t - 1 day) to this bar, then an exponentially weighted std (span 100, needs 100 values).
    The book's printed code takes the bar strictly before (t - 1 day); the two differ only when a bar ends exactly one day earlier, as on hourly bars, where this takes that bar (24 bars back),
    which is what the earlier hourly runs did."""
    n = len(t)
    ss = np.searchsorted(t, t - DAY_S, side="right")                # number of bars with stamp <= t - 1 day
    prev = ss - 1
    valid = prev >= 0
    pv = np.maximum(prev, 0)
    r = np.full(n, np.nan)
    ok = valid & ((t - t[pv]) <= break_s)
    r[ok] = close[ok] / close[pv[ok]] - 1.0
    return pd.Series(r).ewm(span=span, min_periods=span).std().to_numpy()


def forward_window(t: np.ndarray, seg: np.ndarray, ev: np.ndarray, h_s: int, hcap: int):
    """For events at bar positions ev: (L, ok, capped).  L = number of forward bars followed (to the first bar that ends at or after t0 + h_s, at most hcap)."""
    n = len(t)
    iv = np.searchsorted(t, t[ev] + h_s, side="left")
    L_full = iv - ev
    capped = L_full > hcap
    L = np.minimum(L_full, hcap)
    ok = (iv < n) & (L >= 1)
    end = np.minimum(ev + L, n - 1)
    ok &= seg[end] == seg[ev]
    return L.astype(np.int64), ok, capped


def triple_barrier_var(q: np.ndarray, L: np.ndarray, sigma: np.ndarray, side: np.ndarray, trgt: np.ndarray, ptsl=(2.0, 2.0)):
    """meta_events.triple_barrier with a different vertical barrier per row.  q[:, j] is the log return (in one-bar sigmas) over j+1 bars, NaN beyond L[i] columns.
    Returns (j_exit, kind, ret_side): j_exit indexes q's columns (exit bar = j_exit + 1 bars after the event); kind +1 profit-take, -1 stop-loss, 0 vertical barrier at column L-1."""
    n, W = q.shape
    with np.errstate(invalid="ignore"):
        R = np.expm1(q.astype(np.float64) * sigma[:, None]) * side[:, None]
        pt = ptsl[0] * trgt
        sl = -ptsl[1] * trgt
        hit_pt = R > pt[:, None]
        hit_sl = R < sl[:, None]
    has_pt, has_sl = hit_pt.any(axis=1), hit_sl.any(axis=1)
    j_pt = np.where(has_pt, hit_pt.argmax(axis=1), W)
    j_sl = np.where(has_sl, hit_sl.argmax(axis=1), W)
    j = np.minimum(np.minimum(j_pt, j_sl), L - 1)
    kind = np.zeros(n, dtype=np.int8)
    kind[(j_pt < j_sl) & (j_pt < L)] = 1
    kind[(j_sl < j_pt) & (j_sl < L)] = -1
    ret = R[np.arange(n), j]
    return j.astype(np.int64), kind, ret


def attribution_weights_t(pos: np.ndarray, t1pos: np.ndarray, logc: np.ndarray):
    return E.attribution_weights(pos, t1pos, logc)


def market_state_ref(bars: dict, window: str = MKT_WINDOW, min_history_s: int = MKT_MIN_HISTORY_S, min_symbols: int = MKT_MIN_SYMBOLS):
    """(index_t, above, dist): at each hour-end stamp with >= min_symbols returns, whether the equal-weight index of the hourly reference closes is above its own `window` mean (1.0 / 0.0, NaN until the
    mean has min_history_s of history) and log(index / mean)."""
    series = {}
    for s, b in bars.items():
        if len(b.ref_t) < 2:
            continue
        lc = np.log(b.ref_close)
        r = np.concatenate([[np.nan], np.diff(lc)])
        series[s] = pd.Series(r, index=b.ref_t)
    frame = pd.DataFrame(series)
    counts = frame.notna().sum(axis=1)
    mean_ret = frame.mean(axis=1)[counts >= min_symbols]
    index = np.exp(mean_ret.cumsum())
    di = pd.DatetimeIndex(pd.to_datetime(index.index.to_numpy(), unit="s"))
    idx_dt = pd.Series(index.to_numpy(), index=di)
    ma = idx_dt.rolling(window, min_periods=1).mean()
    ready = (di - di[0]) >= pd.Timedelta(seconds=min_history_s)
    above = np.where(ready, (idx_dt.to_numpy() > ma.to_numpy()).astype(float), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(ready, np.log(idx_dt.to_numpy() / ma.to_numpy()), np.nan)
    return index.index.to_numpy(np.int64), above, dist, idx_dt, ma


def asof(ref_t: np.ndarray, values: np.ndarray, t0: np.ndarray, tol_s: int = STATE_TOL_S) -> np.ndarray:
    """values at the last stamp <= t0, NaN when there is none or it is more than tol_s older than t0."""
    k = np.searchsorted(ref_t, t0, side="right") - 1
    kk = np.maximum(k, 0)
    ok = (k >= 0) & ((t0 - ref_t[kk]) <= tol_s)
    return np.where(ok, values[kk], np.nan)


def build_meta_events_bars(bars: dict, break_s: int, h_days: float = H_DAYS, hcap: int = 400, ptsl=(2.0, 2.0), cost_frac: float = 0.0024, eligible=None, log=print, mkt_min_history_s: int = MKT_MIN_HISTORY_S):
    """bars: {symbol: bar_loader.Bars}.  eligible(symbol, t0_array) -> bool array (point-in-time membership), or None for 'all'.
    Returns (df, info) shaped like meta_events.build_meta_events (same columns, plus n_fwd and capped)."""
    symbols = sorted(bars)
    h_s = int(round(h_days * DAY_S))
    parts, info_sym, per_symbol = [], [], []
    state_t, above, dist, idx_dt, ma = market_state_ref(bars, min_history_s=mkt_min_history_s)
    for sym in symbols:
        b = bars[sym]
        n = b.n
        t = b.t
        logc = np.log(b.close)
        r = np.concatenate([[np.nan], np.diff(logc)])
        if n > 1:
            r[1:][b.brk] = np.nan                                         # no return across a data hole
        sigma = pd.Series(r).ewm(span=SPAN, min_periods=SPAN).std().to_numpy()
        n_day = bars_per_day(t)
        h = sigma * np.sqrt(n_day)
        ev = B.cusum_events(r, h)
        total = len(ev)
        if total and eligible is not None:
            member = np.asarray(eligible(sym, t[ev]), dtype=bool)
        else:
            member = np.ones(total, dtype=bool)
        ev_m = ev[member]
        sig_ok = np.isfinite(sigma[ev_m]) & (sigma[ev_m] > 0)
        L, win_ok, capped = forward_window(t, b.seg, ev_m, h_s, hcap) if len(ev_m) else (np.zeros(0, np.int64), np.zeros(0, bool), np.zeros(0, bool))
        good = sig_ok & win_ok
        kept = ev_m[good]
        Lk, capk = L[good], capped[good]
        per_symbol.append({"symbol": sym, "bars": n, "first": str(b.idx[0]), "last": str(b.idx[-1]), "events_total": int(total), "not_member": int(total - len(ev_m)),
                           "dropped_sigma": int((~sig_ok).sum()), "dropped_window": int((sig_ok & ~win_ok).sum()), "events_kept": int(len(kept)), "capped": int(capk.sum())})
        if not len(kept):
            continue
        sg = sigma[kept]
        p = kept
        # side: sign of the trailing 720-bar log return (missing when the lookback reaches back across a hole or before the first bar)
        def lookback_ret(Lb):
            ok = (p >= Lb)
            v = np.full(len(p), np.nan)
            src = np.maximum(p - Lb, 0)
            ok &= b.seg[src] == b.seg[p]
            v[ok] = logc[p[ok]] - logc[src[ok]]
            return v
        r720 = lookback_ret(SIDE_SIGNAL_LOOKBACK)
        side = np.sign(np.nan_to_num(r720, nan=0.0))
        trgt_all = daily_vol_time(t, b.close, break_s)
        trgt = trgt_all[p]
        s_slow = pd.Series(r).ewm(span=SLOW_SPAN, min_periods=SLOW_SPAN).std().to_numpy()[p]
        cols = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["vol_ratio"] = np.log(sg / s_slow)
            for Lb, nm in LOOKBACKS:
                cols[nm] = lookback_ret(Lb) / (sg * math.sqrt(Lb))
        c_series = b.close_series()
        inf, dser = E.daily_features(c_series)
        day_no = (pd.DatetimeIndex(pd.to_datetime(t[p], unit="s")).to_numpy().astype("datetime64[D]").astype(np.int64))
        prev = day_no - 1
        for nm, ser in dser.items():
            cols[nm] = ser.reindex(prev).to_numpy()
        cs_ref = MF.corwin_schultz(b.ref_high, b.ref_low, E.CS_SL) if len(b.ref_t) else np.zeros(0)
        cols["cs_spread"] = asof(b.ref_t, cs_ref, t[p]) if len(cs_ref) else np.full(len(p), np.nan)
        ok0 = (side != 0) & np.isfinite(trgt) & (trgt > 0)
        j = np.zeros(len(p), dtype=np.int64)
        kind = np.zeros(len(p), dtype=np.int8)
        ret = np.full(len(p), np.nan)
        if ok0.any():
            pk = p[ok0]
            W = int(Lk[ok0].max())
            steps = np.arange(1, W + 1)
            idxm = np.minimum(pk[:, None] + steps[None, :], n - 1)
            q = (logc[idxm] - logc[pk][:, None]) / sg[ok0][:, None]
            q[steps[None, :] > Lk[ok0][:, None]] = np.nan
            jj, kk, rr = triple_barrier_var(q.astype(np.float32), Lk[ok0], sg[ok0], side[ok0], trgt[ok0], ptsl)
            j[ok0], kind[ok0], ret[ok0] = jj, kk, rr
        t1pos = p + j + 1
        parts.append(pd.DataFrame({"symbol": sym, "pos": p, "t0": t[p], "t1": t[t1pos], "t1pos": t1pos, "sigma": sg, "trgt": trgt, "side": side, "exit_j": j, "kind": kind, "ret_side": ret,
                                   "ffd": cols["ffd"], "sadf": cols["sadf"], "ent_plugin": cols["ent_plugin"], "ent_konto": cols["ent_konto"], "cs_spread": cols["cs_spread"],
                                   "mom24n": cols["mom24n"], "mom168n": cols["mom168n"], "mom720n": cols["mom720n"], "mom2160n": cols["mom2160n"], "vol_ratio": cols["vol_ratio"],
                                   "n_fwd": Lk, "capped": capk.astype(np.int8), "_ok0": ok0}))
        info_sym.append({"symbol": sym, "d": inf["d"], "adf": inf["adf"], "crit": inf["crit"], "corr": inf["corr"], "days": int(len(dser["ffd"]))})
    if not parts:
        raise RuntimeError("no events were produced")
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    df["mkt_state"] = asof(state_t, above, df["t0"].to_numpy(np.int64))
    df["mkt_dist"] = asof(state_t, dist, df["t0"].to_numpy(np.int64))
    df["log_trgt"] = np.log(df["trgt"])
    df["bin"] = (df["ret_side"] > 0).astype(np.int8)
    ok0 = df.pop("_ok0").to_numpy()
    keep = ok0 & np.isfinite(df[FEATURES].to_numpy(dtype=np.float64)).all(axis=1) & np.isfinite(df["ret_side"].to_numpy())
    df = df[keep].reset_index(drop=True)
    u_arr, w_arr = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for sym, grp in df.groupby("symbol", sort=False):
        ix = grp.index.to_numpy()
        u_arr[ix], w_arr[ix] = attribution_weights_t(grp["pos"].to_numpy(np.int64), grp["t1pos"].to_numpy(np.int64), np.log(bars[sym].close))
    df["u"], df["w_raw"] = u_arr, w_arr
    df["w"] = df["w_raw"] * len(df) / df["w_raw"].sum()
    df["net_ret"] = df["ret_side"] - cost_frac
    info = {"symbols": info_sym, "per_symbol": per_symbol, "events_cusum": int(n0), "events_dropped_no_side_or_vol": int((~ok0).sum()),
            "events_dropped_missing_feature": int((ok0 & ~keep).sum()), "events_kept": int(len(df)), "cost_frac": cost_frac, "ptsl": list(ptsl), "h_days": h_days, "hcap": hcap,
            "capped_share": float(df["capped"].mean()) if len(df) else float("nan"), "median_forward_bars": float(df["n_fwd"].median()) if len(df) else float("nan"),
            "kind_share": {str(k): float((df["kind"] == k).mean()) for k in (1, -1, 0)}, "bin_share": float(df["bin"].mean())}
    log(f"bar events: after universe / window rules {n0:,}, kept {len(df):,}; barrier exits pt {info['kind_share']['1']:.2f} sl {info['kind_share']['-1']:.2f} vertical {info['kind_share']['0']:.2f}; "
        f"bin=1 share {info['bin_share']:.3f}; paths stopped at the bar cap {info['capped_share']:.3f}; median forward bars {info['median_forward_bars']:.0f}")
    return df, info
