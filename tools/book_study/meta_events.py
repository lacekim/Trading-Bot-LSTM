"""Meta-labeling inputs from candle history, following the book (Ch.2 CUSUM events, Ch.3 triple barrier + meta-labels, Ch.4 uniqueness and
return-attribution weights, Ch.5 fractional-differentiation feature).  No project imports: takes {symbol: hourly close Series}.

Primary model (the 'side'): sign of the trailing 720-bar (30-day) log return at a CUSUM event.  Meta-label: 1 if the side's return at the
first barrier touch (or at the vertical barrier) is positive, else 0 (Snippet 3.7).  Barriers are on simple returns, sized by the book's
daily-volatility estimate (Snippet 3.1), horizontal pt/sl multiples and a vertical barrier of H bars.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import book_stats as B
import book_study_core as C
import fracdiff_core as F
import micro_features as MF

SIDE_SIGNAL = "mom720"
BARS_PER_DAY = 24
FEATURES = ["side", "ffd", "mom24n", "mom168n", "mom720n", "mom2160n", "log_trgt", "vol_ratio", "mkt_state", "mkt_dist",
            "sadf", "ent_plugin", "ent_konto", "cs_spread"]
FFD_THRES = 0.01
SADF_WINDOW, SADF_MINSL, SADF_LAGS = 100, 20, 1       # trailing window of daily log closes, minimum sample, lagged differences
ENT_WINDOW, ENT_WORD = 100, 2                          # daily returns in the message; plug-in word length
CS_SL = 24                                             # hourly bars in the Corwin-Schultz rolling mean


def daily_vol(close: pd.Series, span: int = 100, bars_per_day: int = BARS_PER_DAY) -> np.ndarray:
    """Snippet 3.1 on bars: the return over the previous day at each bar, then an exponentially weighted std (span 100, needs 100 values)."""
    c = close.to_numpy(dtype=np.float64)
    r = np.full(len(c), np.nan)
    r[bars_per_day:] = c[bars_per_day:] / c[:-bars_per_day] - 1.0
    return pd.Series(r).ewm(span=span, min_periods=span).std().to_numpy()


def triple_barrier(q: np.ndarray, sigma: np.ndarray, side: np.ndarray, trgt: np.ndarray, ptsl=(2.0, 2.0)):
    """First touch on the side-signed simple-return path (Snippets 3.2-3.4, 3.6). q[:, j] is the log return (in one-bar sigmas) over j+1 bars.
    Strict inequalities as in the book (> pt, < -sl). Returns (j_exit, kind, ret_side): j_exit indexes q's columns; kind +1 profit-take,
    -1 stop-loss, 0 vertical barrier at the last column; ret_side is the side-signed simple return at the exit bar."""
    n, hp1 = q.shape
    R = np.expm1(q.astype(np.float64) * sigma[:, None]) * side[:, None]
    pt = ptsl[0] * trgt
    sl = -ptsl[1] * trgt
    hit_pt = R > pt[:, None]
    hit_sl = R < sl[:, None]
    has_pt, has_sl = hit_pt.any(axis=1), hit_sl.any(axis=1)
    j_pt = np.where(has_pt, hit_pt.argmax(axis=1), hp1)
    j_sl = np.where(has_sl, hit_sl.argmax(axis=1), hp1)
    j = np.minimum(np.minimum(j_pt, j_sl), hp1 - 1)
    kind = np.zeros(n, dtype=np.int8)
    kind[(j_pt < j_sl) & (j_pt < hp1)] = 1
    kind[(j_sl < j_pt) & (j_sl < hp1)] = -1
    ret = R[np.arange(n), j]
    return j.astype(np.int64), kind, ret


def attribution_weights(pos: np.ndarray, t1pos: np.ndarray, logc: np.ndarray):
    """Snippets 4.1, 4.2, 4.10 for one symbol: average uniqueness and |sum of log-return / concurrency| over bars pos..t1pos inclusive."""
    n_bars = len(logc)
    u = B.average_uniqueness(pos, t1pos, n_bars)
    diff = np.zeros(n_bars + 1)
    np.add.at(diff, pos, 1.0)
    np.add.at(diff, np.minimum(t1pos + 1, n_bars), -1.0)
    conc = np.cumsum(diff)[:n_bars]
    r = np.concatenate([[0.0], np.diff(logc)])
    contrib = np.where(conc > 0, r / np.where(conc > 0, conc, 1.0), 0.0)
    pref = np.concatenate([[0.0], np.cumsum(contrib)])
    w = np.abs(pref[np.minimum(t1pos + 1, n_bars)] - pref[pos])
    return u, w


def daily_features(close: pd.Series, thres: float = FFD_THRES):
    """Per symbol, on the daily log close (last close of each UTC day): the smallest passing d (Ch.5) with its FFD series, and the Chapter 17-18
    series (SADF, plug-in entropy, Kontoyiannis entropy). Returns (fracdiff info, {name: Series indexed by UTC day number})."""
    d = np.log(close.resample("1D").last().dropna())
    x = d.to_numpy()
    info = F.min_d(x, thres=thres)
    day_no = (d.index.to_numpy().astype("datetime64[D]").astype(np.int64))
    plug, kon = MF.entropy_windows(x, ENT_WINDOW, ENT_WORD)
    return info, {"ffd": pd.Series(F.frac_diff_ffd(x, info["d"], thres), index=day_no),
                  "sadf": pd.Series(MF.sadf_windows(x, SADF_WINDOW, SADF_MINSL, SADF_LAGS), index=day_no),
                  "ent_plugin": pd.Series(plug, index=day_no), "ent_konto": pd.Series(kon, index=day_no)}


def ffd_daily(close: pd.Series, thres: float = FFD_THRES):
    info, ser = daily_features(close, thres)
    return info, ser["ffd"]


def build_meta_events(closes: dict, ohlc: dict, h: int = 100, ptsl=(2.0, 2.0), cost_frac: float = 0.0024, signal: str = SIDE_SIGNAL, log=print):
    """closes: {symbol: close Series}; ohlc: {symbol: DataFrame(high, low, close)} on the same index (for the Corwin-Schultz spread).
    Returns (df, info): df has one row per event with the columns used downstream; info has per-symbol fracdiff details and counts."""
    ev = C.build_events(closes, h, log=lambda m: None, signals=(signal,))
    state, idx, ma = C.market_state(closes, ev["t0"])
    n0 = len(ev["t0"])
    side = ev["sig_" + signal].astype(np.float64)
    symbols = ev["symbols"]
    sym_id = ev["sym"]
    pos = ev["pos"]
    sigma = ev["sigma"].astype(np.float64)
    t0 = ev["t0"].astype(np.int64)

    # per-symbol quantities at the event bars
    cols = {k: np.full(n0, np.nan) for k in ("trgt", "vol_ratio", "mom24n", "mom168n", "mom720n", "mom2160n", "ffd", "sadf", "ent_plugin", "ent_konto", "cs_spread")}
    info_sym = []
    # triple barrier for all events first (needs trgt)
    for sid, sym in enumerate(symbols):
        m = np.flatnonzero(sym_id == sid)
        if not len(m):
            continue
        c = closes[sym].astype(float)
        c = c[c > 0]
        logc = np.log(c.to_numpy())
        p = pos[m]
        trgt_all = daily_vol(c)
        cols["trgt"][m] = trgt_all[p]
        r = np.concatenate([[np.nan], np.diff(logc)])
        s_fast = sigma[m]
        s_slow = pd.Series(r).ewm(span=2016, min_periods=2016).std().to_numpy()[p]
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["vol_ratio"][m] = np.log(s_fast / s_slow)
            for L, nm in ((24, "mom24n"), (168, "mom168n"), (720, "mom720n"), (2160, "mom2160n")):
                ok = p >= L
                v = np.full(len(m), np.nan)
                v[ok] = (logc[p[ok]] - logc[p[ok] - L]) / (s_fast[ok] * math.sqrt(L))
                cols[nm][m] = v
        inf, dser = daily_features(c)
        day_no = (pd.DatetimeIndex(pd.to_datetime(t0[m], unit="s")).to_numpy().astype("datetime64[D]").astype(np.int64))
        prev = day_no - 1                                  # the last completed UTC day
        for nm, ser in dser.items():
            cols[nm][m] = ser.reindex(prev).to_numpy()
        hl = ohlc[sym].reindex(c.index)
        cs = MF.corwin_schultz(hl["high"].to_numpy(np.float64), hl["low"].to_numpy(np.float64), CS_SL)
        cols["cs_spread"][m] = cs[p]
        info_sym.append({"symbol": sym, "d": inf["d"], "adf": inf["adf"], "crit": inf["crit"], "corr": inf["corr"], "days": int(len(dser["ffd"]))})
    trgt = cols["trgt"]
    ok0 = (side != 0) & np.isfinite(trgt) & (trgt > 0)
    j = np.zeros(n0, dtype=np.int64)
    kind = np.zeros(n0, dtype=np.int8)
    ret = np.full(n0, np.nan)
    if ok0.any():
        jj, kk, rr = triple_barrier(ev["q"][ok0][:, : h + 1], sigma[ok0], side[ok0], trgt[ok0], ptsl)
        j[ok0], kind[ok0], ret[ok0] = jj, kk, rr
    t1pos = pos + j + 1
    t1 = t0 + (j + 1) * 3600

    # market distance (log of index over its 200-day mean) at the event time
    ev_ts = pd.DatetimeIndex(pd.to_datetime(t0, unit="s"))
    dist = np.log(idx / ma).reindex(ev_ts).to_numpy()
    df = pd.DataFrame({
        "symbol": [symbols[i] for i in sym_id], "pos": pos, "t0": t0, "t1": t1, "t1pos": t1pos, "sigma": sigma, "trgt": trgt,
        "side": side, "exit_j": j, "kind": kind, "ret_side": ret, "bin": (ret > 0).astype(np.int8), "u": np.nan, "w_raw": np.nan,
        "ffd": cols["ffd"], "sadf": cols["sadf"], "ent_plugin": cols["ent_plugin"], "ent_konto": cols["ent_konto"], "cs_spread": cols["cs_spread"], "mom24n": cols["mom24n"], "mom168n": cols["mom168n"], "mom720n": cols["mom720n"], "mom2160n": cols["mom2160n"],
        "log_trgt": np.log(trgt), "vol_ratio": cols["vol_ratio"], "mkt_state": np.where(state < 0, np.nan, state.astype(float)), "mkt_dist": dist,
    })
    keep = ok0 & np.isfinite(df[FEATURES].to_numpy(dtype=np.float64)).all(axis=1) & np.isfinite(ret)
    df = df[keep].reset_index(drop=True)
    # uniqueness and return-attribution weights over the events that are actually used (events dropped above do not count as concurrent)
    u_arr = np.full(len(df), np.nan)
    w_arr = np.full(len(df), np.nan)
    for sym, grp in df.groupby("symbol", sort=False):
        c = closes[sym].astype(float)
        c = c[c > 0]
        ix = grp.index.to_numpy()
        u_arr[ix], w_arr[ix] = attribution_weights(grp["pos"].to_numpy(np.int64), grp["t1pos"].to_numpy(np.int64), np.log(c.to_numpy()))
    df["u"], df["w_raw"] = u_arr, w_arr
    df["w"] = df["w_raw"] * len(df) / df["w_raw"].sum()       # Snippet 4.10: scaled to mean 1
    df["net_ret"] = df["ret_side"] - cost_frac
    info = {"symbols": info_sym, "per_symbol": ev["per_symbol"], "events_cusum": int(n0), "events_dropped_no_side_or_vol": int((~ok0).sum()),
            "events_dropped_missing_feature": int((ok0 & ~keep).sum()), "events_kept": int(len(df)), "cost_frac": cost_frac, "ptsl": list(ptsl), "h": h,
            "kind_share": {str(k): float((df["kind"] == k).mean()) for k in (1, -1, 0)}, "bin_share": float(df["bin"].mean())}
    log(f"meta events: cusum {n0:,}, kept {len(df):,}; barrier exits pt {info['kind_share']['1']:.2f} sl {info['kind_share']['-1']:.2f} vertical {info['kind_share']['0']:.2f}; bin=1 share {info['bin_share']:.3f}")
    return df, info
