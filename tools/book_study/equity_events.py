"""Meta-labeling inputs for DAILY equity bars (Alpaca), following the same book chapters as meta_events.py (Ch.2 CUSUM events, Ch.3 triple barrier and
meta-labels, Ch.4 uniqueness / return-attribution weights, Ch.5 fractional differentiation, Ch.17-18 SADF and entropy), with every hourly assumption
restated for one bar = one trading day.  Settings are taken from the project's own Alpaca momentum test (tools/run_alpaca_momentum_test.py) wherever it
declares one, and from the hourly runs otherwise; none is tuned.

  universe  at each COMPLETED month-end: adjusted close > 5, volume > 0 that day, >= 253 bars of history, >= 40 valid dollar-volume days in the last 60,
            ranked by the trailing-60-day MEDIAN dollar volume; the top N are members (project: N = 1000; here N is an argument, default 200).  A signal on
            day d may only use the universe of the last month-end strictly before d.  Members exist from the first month-end with >= 200 eligible names.
  events    symmetric CUSUM filter on daily log returns, threshold = causal one-bar sigma (EWM std, span 14, = 336 hours of the hourly runs).
  side      sign of the project's base momentum score at the signal bar: 0.5 * (C[t-21] / C[t-252] - 1) + 0.5 * (C[t-21] / C[t-126] - 1).
  entry     at the NEXT bar's close (the project: "the trade executes at the NEXT trading day's close").  pos = that entry bar; all features are read at the signal bar pos-1.
  barriers  +/- 2 x the book's daily-volatility estimate (Snippet 3.1 on 1-bar returns, span 100); vertical barrier H+1 = 5 bars after entry.
  complete  a path is kept only if every trading day of the market calendar between the signal bar and the last path bar has a bar for the symbol.
  market    equal-weight index of all loaded symbols (days with >= 10 symbols), above / below its own 200-calendar-day mean (>= 130 observations), read at the signal bar.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import book_stats as B
import book_study_core as C
import meta_events as M
from meta_events import ENT_WINDOW, ENT_WORD, FFD_THRES, SADF_LAGS, SADF_MINSL, SADF_WINDOW      # noqa: F401  (declared in meta_events.py, shared with the hourly runs)

FEATURES = ["side", "ffd", "mom1n", "mom5n", "mom21n", "mom63n", "log_trgt", "vol_ratio", "mkt_state", "mkt_dist", "sadf", "ent_plugin", "ent_konto"]
SIDE_SIGNAL = "mom12_1_6_1"
SIGMA_FAST, SIGMA_SLOW, TRGT_SPAN = 14, 84, 100          # bars (days); 336 h, 2016 h and span 100 of the hourly runs, in days
LOOKBACKS = ((1, "mom1n"), (5, "mom5n"), (21, "mom21n"), (63, "mom63n"))
LB12, LB6, SKIP = 252, 126, 21
H = 4                                                    # vertical barrier at H + 1 = 5 bars after entry
DV_WIN, DV_MIN, MIN_PRICE, MIN_HIST = 60, 40, 5.0, LB12 + 1
N_UNIVERSE, MIN_ELIGIBLE_START = 200, 200
MKT_WINDOW, MKT_MIN_PERIODS, MKT_MIN_SYMBOLS = "200D", 130, 10
COST_BP_PER_SIDE = 10.0                                  # project: "Costs: 10 bp per side on traded notional"


def load_alpaca(path) -> pd.DataFrame:
    """symbol, date, close, volume rows written by export_venue_candles.py; the ticker 'NA' must not become a missing value."""
    d = pd.read_csv(path, keep_default_na=False, na_values=[""], parse_dates=["date"])
    d["symbol"] = d["symbol"].astype(str)
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["close"])
    d = d[d["close"] > 0].sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return d.reset_index(drop=True)


def pivots(d: pd.DataFrame):
    close = d.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = d.pivot(index="date", columns="symbol", values="volume").sort_index()
    return close, volume


def month_end_universe(close: pd.DataFrame, volume: pd.DataFrame, n_universe: int = N_UNIVERSE):
    """({month_end: [member symbols]}, {month_end: eligible count}) for every COMPLETED month (the month of the last date is not complete)."""
    dates = close.index
    dv = (close * volume).where((volume > 0) & (close > 0))
    med = dv.rolling(DV_WIN, min_periods=DV_MIN).median()
    nbars = close.notna().cumsum()
    ser = dates.to_series()
    ends = ser.groupby([ser.dt.year, ser.dt.month]).max()
    last = dates[-1]
    ends = [d for d in ends if (d.year, d.month) != (last.year, last.month)]
    uni, counts, started = {}, {}, False
    for m in ends:
        ok = (close.loc[m] > MIN_PRICE) & (volume.loc[m] > 0) & med.loc[m].notna() & (nbars.loc[m] >= MIN_HIST)
        cand = med.loc[m][ok]
        counts[m] = int(len(cand))
        started = started or counts[m] >= MIN_ELIGIBLE_START
        order = sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))
        uni[m] = [s for s, _ in order[:n_universe]] if started else []
    return uni, counts


def base_score(cs: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The project's base momentum score at bar positions s (NaN where fewer than 253 bars exist)."""
    out = np.full(len(s), np.nan)
    ok = s >= LB12
    a = cs[s[ok] - SKIP]
    out[ok] = 0.5 * (a / cs[s[ok] - LB12] - 1.0) + 0.5 * (a / cs[s[ok] - LB6] - 1.0)
    return out


def build_equity_events(closes: dict, universe: dict, cal: pd.DatetimeIndex, h: int = H, ptsl=(2.0, 2.0), cost_frac: float = 0.0020, log=print):
    """closes: {symbol: daily close Series}; universe: {month_end: [symbols]}; cal: the market calendar (every trading date).
    Returns (df, info) shaped like meta_events.build_meta_events, with t_sig (epoch of the signal bar) added."""
    ends = sorted(universe)
    ends_ns = np.array(ends, dtype="datetime64[ns]")
    sets = [set(universe[m]) for m in ends]
    symbols = sorted(closes)
    parts, info_sym, per_symbol = [], [], []
    steps = np.arange(1, h + 2)
    for sym in symbols:
        c = closes[sym].astype(float)
        c = c[c > 0]
        n = len(c)
        dates = c.index
        cidx = cal.get_indexer(dates)
        cs = c.to_numpy()
        logc = np.log(cs)
        r = np.concatenate([[np.nan], np.diff(logc)])
        sigma = pd.Series(r).ewm(span=SIGMA_FAST, min_periods=SIGMA_FAST).std().to_numpy()
        evp = B.cusum_events(r, sigma)                        # threshold = the daily sigma itself (one bar = one day)
        total = len(evp)
        if total:
            k = np.searchsorted(ends_ns, dates.values[evp], side="left") - 1
            member = np.array([kk >= 0 and sym in sets[kk] for kk in k], dtype=bool)
        else:
            member = np.zeros(0, dtype=bool)
        ev_m = evp[member]
        ok = (ev_m + h + 2 < n) & np.isfinite(sigma[ev_m]) & (sigma[ev_m] > 0)
        ev_end = ev_m[ok]
        gap_ok = (cidx[ev_end + h + 2] - cidx[ev_end]) == (h + 2)
        kept = ev_end[gap_ok]
        per_symbol.append({"symbol": sym, "bars": n, "first": str(dates[0]), "last": str(dates[-1]), "events_total": int(total), "not_member": int(total - len(ev_m)),
                           "dropped_end_or_sigma": int(len(ev_m) - ok.sum()), "dropped_gap": int((~gap_ok).sum()), "events_kept": int(len(kept))})
        if not len(kept):
            continue
        p = kept + 1                                          # entry bar
        sg = sigma[kept]
        side = np.sign(np.nan_to_num(base_score(cs, kept), nan=0.0))
        trgt_all = M.daily_vol(c, span=TRGT_SPAN, bars_per_day=1)
        slow = pd.Series(r).ewm(span=SIGMA_SLOW, min_periods=SIGMA_SLOW).std().to_numpy()
        cols = {"trgt": trgt_all[kept]}
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["vol_ratio"] = np.log(sg / slow[kept])
            for L, nm in LOOKBACKS:
                v = np.full(len(kept), np.nan)
                okL = kept >= L
                v[okL] = (logc[kept[okL]] - logc[kept[okL] - L]) / (sg[okL] * math.sqrt(L))
                cols[nm] = v
        inf, dser = M.daily_features(c)
        day_sig = dates.values[kept].astype("datetime64[D]").astype(np.int64)
        for nm, ser in dser.items():
            cols[nm] = ser.reindex(day_sig).to_numpy()
        ok0 = (side != 0) & np.isfinite(cols["trgt"]) & (cols["trgt"] > 0)
        j = np.zeros(len(kept), dtype=np.int64)
        kind = np.zeros(len(kept), dtype=np.int8)
        ret = np.full(len(kept), np.nan)
        if ok0.any():
            q = (logc[p[ok0][:, None] + steps[None, :]] - logc[p[ok0]][:, None]) / sg[ok0][:, None]
            jj, kk, rr = M.triple_barrier(q, sg[ok0], side[ok0], cols["trgt"][ok0], ptsl)
            j[ok0], kind[ok0], ret[ok0] = jj, kk, rr
        t1pos = p + j + 1
        epoch = lambda idx: (dates.values[idx].astype("datetime64[s]").astype(np.int64))      # noqa: E731
        part = pd.DataFrame({"symbol": sym, "pos": p, "t_sig": epoch(kept), "t0": epoch(p), "t1": epoch(t1pos), "t1pos": t1pos, "sigma": sg, "trgt": cols["trgt"],
                             "side": side, "exit_j": j, "kind": kind, "ret_side": ret, "ffd": cols["ffd"], "sadf": cols["sadf"], "ent_plugin": cols["ent_plugin"],
                             "ent_konto": cols["ent_konto"], "mom1n": cols["mom1n"], "mom5n": cols["mom5n"], "mom21n": cols["mom21n"], "mom63n": cols["mom63n"],
                             "vol_ratio": cols["vol_ratio"], "_ok0": ok0})
        parts.append(part)
        info_sym.append({"symbol": sym, "d": inf["d"], "adf": inf["adf"], "crit": inf["crit"], "corr": inf["corr"], "days": int(len(dser["ffd"]))})
    if not parts:
        raise RuntimeError("no events were produced")
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    state, idx, ma = C.market_state(closes, df["t_sig"].to_numpy(), window=MKT_WINDOW, min_periods=MKT_MIN_PERIODS, min_symbols=MKT_MIN_SYMBOLS)
    ev_ts = pd.DatetimeIndex(pd.to_datetime(df["t_sig"].to_numpy(), unit="s"))
    df["mkt_state"] = np.where(state < 0, np.nan, state.astype(float))
    df["mkt_dist"] = np.log(idx / ma).reindex(ev_ts).to_numpy()
    df["log_trgt"] = np.log(df["trgt"])
    df["bin"] = (df["ret_side"] > 0).astype(np.int8)
    ok0 = df.pop("_ok0").to_numpy()
    keep = ok0 & np.isfinite(df[FEATURES].to_numpy(dtype=np.float64)).all(axis=1) & np.isfinite(df["ret_side"].to_numpy())
    df = df[keep].reset_index(drop=True)
    df["u"] = np.nan
    df["w_raw"] = np.nan
    u_arr, w_arr = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for sym, grp in df.groupby("symbol", sort=False):
        c = closes[sym].astype(float)
        c = c[c > 0]
        ix = grp.index.to_numpy()
        u_arr[ix], w_arr[ix] = M.attribution_weights(grp["pos"].to_numpy(np.int64), grp["t1pos"].to_numpy(np.int64), np.log(c.to_numpy()))
    df["u"], df["w_raw"] = u_arr, w_arr
    df["w"] = df["w_raw"] * len(df) / df["w_raw"].sum()
    df["net_ret"] = df["ret_side"] - cost_frac
    info = {"symbols": info_sym, "per_symbol": per_symbol, "events_cusum": int(n0), "events_dropped_no_side_or_vol": int((~ok0).sum()),
            "events_dropped_missing_feature": int((ok0 & ~keep).sum()), "events_kept": int(len(df)), "cost_frac": cost_frac, "ptsl": list(ptsl), "h": h,
            "kind_share": {str(k): float((df["kind"] == k).mean()) for k in (1, -1, 0)}, "bin_share": float(df["bin"].mean())}
    log(f"equity events: kept after universe / end / gap {n0:,}, kept {len(df):,}; barrier exits pt {info['kind_share']['1']:.2f} sl {info['kind_share']['-1']:.2f} "
        f"vertical {info['kind_share']['0']:.2f}; bin=1 share {info['bin_share']:.3f}")
    return df, info
