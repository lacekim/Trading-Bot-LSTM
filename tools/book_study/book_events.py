"""Event table of the book's strategy built straight from the dollar bars, with NO side and NO feature that is not in the book.  New file; bar_events.py / meta_events.py are not edited.

What bar_events.build_meta_events_bars did that the book does not: it gave every event a side (sign of the trailing 720-bar return, my stand-in), dropped events whose side was 0 or whose 720-bar lookback
crossed a data hole or the start of the series, and dropped events that lacked the 2,160-bar momentum or the 200-day market-state features.  None of that is in the book.  Here:
  Ch.2   CUSUM filter (Snippet 2.4) on the dollar bars; threshold = causal one-bar sigma x sqrt(bars per calendar day)  (bar_events.bars_per_day; project's choice, the book leaves the threshold open)
  Ch.3   triple barrier (Snippets 3.2-3.4) with symmetric horizontal barriers 2 x daily vol (Snippet 3.1) and a 4-day vertical barrier; side = None (Snippets 3.5-3.7): the path is the long price path, bin = sign of the return
         at the first touch in (-1, 0, 1); because the barriers are symmetric the exit time does not depend on the side, so this is the side-free label.  Snippet 3.8 (dropLabels) is applied by the caller after the weights.
  Ch.4   average uniqueness and return-attribution weights (bar_events.attribution_weights_t) on all events, as the book orders it (weights, then drop rare labels).
  Ch.5, 17, 18, 19  features: FFD, SADF, entropy (plug-in, Kontoyiannis), Corwin-Schultz spread, Becker-Parkinson volatility, Roll, Kyle / Amihud / Hasbrouck lambdas, Chu-Stinchcombe-White CUSUM (book_micro.py); Snippet 3.1 daily volatility (log).
An event is kept when its volatility, its vertical window, its twelve features and its label are all available.  Nothing else removes events except the point-in-time universe rule for Alpaca (a project choice, disclosed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bar_events as BE
import book_stats as B
import meta_events as E
import book_micro as BM
import micro_features as MF

FEATURES_BOOK = ["ffd", "log_trgt", "sadf", "ent_plugin", "ent_konto", "cs_spread", "bp_sigma", "roll", "kyle", "amihud", "hasbrouck", "csw"]


def membership(universe_csv):
    """eligible(symbol, t0_epoch_array) -> bool: the symbol was a member at the last month-end strictly before the event's UTC date (the daily Alpaca runs' rule)."""
    u = pd.read_csv(universe_csv, keep_default_na=False, na_values=[""], dtype={"symbol": str})
    sets = {}
    for me, g in u.groupby("month_end"):
        sets[pd.Timestamp(me)] = set(g["symbol"])
    ends = sorted(sets)
    ends_ns = np.array(ends, dtype="datetime64[ns]")
    sets_l = [sets[m] for m in ends]

    def eligible(sym, t0):
        days = pd.DatetimeIndex(pd.to_datetime(np.asarray(t0, np.int64), unit="s")).floor("D").to_numpy().astype("datetime64[ns]")
        k = np.searchsorted(ends_ns, days, side="left") - 1
        return np.array([kk >= 0 and sym in sets_l[kk] for kk in k], dtype=bool)
    return eligible, ends[0] if ends else None


def build_book_events(bars: dict, break_s: int, h_days: float = 4.0, hcap: int = 400, ptsl=(2.0, 2.0), eligible=None, refvol=None, log=print):
    symbols = sorted(bars)
    h_s = int(round(h_days * BE.DAY_S))
    parts, info_sym, per_symbol = [], [], []
    for sym in symbols:
        b = bars[sym]
        n, t = b.n, b.t
        logc = np.log(b.close)
        r = np.concatenate([[np.nan], np.diff(logc)])
        if n > 1:
            r[1:][b.brk] = np.nan
        sigma = pd.Series(r).ewm(span=BE.SPAN, min_periods=BE.SPAN).std().to_numpy()
        h = sigma * np.sqrt(BE.bars_per_day(t))
        ev = B.cusum_events(r, h)
        total = len(ev)
        member = np.asarray(eligible(sym, t[ev]), dtype=bool) if (total and eligible is not None) else np.ones(total, dtype=bool)
        ev_m = ev[member]
        sig_ok = np.isfinite(sigma[ev_m]) & (sigma[ev_m] > 0)
        L, win_ok, capped = BE.forward_window(t, b.seg, ev_m, h_s, hcap) if len(ev_m) else (np.zeros(0, np.int64), np.zeros(0, bool), np.zeros(0, bool))
        good = sig_ok & win_ok
        p, Lk, capk = ev_m[good], L[good], capped[good]
        rec = {"symbol": sym, "bars": int(n), "events_total": int(total), "not_member": int(total - len(ev_m)), "dropped_sigma": int((~sig_ok).sum()), "dropped_window": int((sig_ok & ~win_ok).sum())}
        if not len(p):
            per_symbol.append({**rec, "dropped_vol_target": 0, "dropped_feature": 0, "zero_return_kept": 0, "events_kept": 0, "capped": 0})
            continue
        sg = sigma[p]
        trgt = BE.daily_vol_time(t, b.close, break_s)[p]
        inf, dser = E.daily_features(b.close_series())
        day_no = pd.DatetimeIndex(pd.to_datetime(t[p], unit="s")).to_numpy().astype("datetime64[D]").astype(np.int64)
        prev = day_no - 1
        cols = {nm: dser[nm].reindex(prev).to_numpy() for nm in ("ffd", "sadf", "ent_plugin", "ent_konto")}
        cs_ref = MF.corwin_schultz(b.ref_high, b.ref_low, E.CS_SL) if len(b.ref_t) else np.zeros(0)
        cols["cs_spread"] = BE.asof(b.ref_t, cs_ref, t[p]) if len(cs_ref) else np.full(len(p), np.nan)
        mf = BM.ref_features(b.ref_high, b.ref_low, b.ref_close, refvol[sym]) if len(b.ref_t) else {}
        for nm in BM.NAMES:
            cols[nm] = BE.asof(b.ref_t, mf[nm], t[p]) if len(b.ref_t) else np.full(len(p), np.nan)
        ok0 = np.isfinite(trgt) & (trgt > 0)
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
            jj, kk, rr = BE.triple_barrier_var(q.astype(np.float32), Lk[ok0], sg[ok0], np.ones(int(ok0.sum())), trgt[ok0], ptsl)
            j[ok0], kind[ok0], ret[ok0] = jj, kk, rr
        t1pos = p + j + 1
        df = pd.DataFrame({"symbol": sym, "pos": p, "t0": t[p], "t1": t[t1pos], "t1pos": t1pos, "sigma": sg, "trgt": trgt, "side": 1.0, "exit_j": j, "kind": kind, "ret_side": ret,
                           "ffd": cols["ffd"], "sadf": cols["sadf"], "ent_plugin": cols["ent_plugin"], "ent_konto": cols["ent_konto"], "cs_spread": cols["cs_spread"],
                           **{nm: cols[nm] for nm in BM.NAMES}, "n_fwd": Lk, "capped": capk.astype(np.int8), "_ok0": ok0})
        df["log_trgt"] = np.log(df["trgt"].where(df["trgt"] > 0))
        feat_ok = np.isfinite(df[FEATURES_BOOK].to_numpy(dtype=np.float64)).all(axis=1)
        ret_ok = np.isfinite(df["ret_side"].to_numpy())
        keep = df["_ok0"].to_numpy() & feat_ok & ret_ok
        rec.update({"dropped_vol_target": int((~ok0).sum()), "dropped_feature": int((ok0 & ~feat_ok).sum()), "zero_return_kept": int((keep & (df["ret_side"].to_numpy() == 0.0)).sum()),
                    "events_kept": int(keep.sum()), "capped": int(capk[keep].sum())})
        per_symbol.append(rec)
        info_sym.append({"symbol": sym, "d": inf["d"], "adf": inf["adf"], "crit": inf["crit"], "corr": inf["corr"], "days": int(len(dser["ffd"]))})
        parts.append(df[keep].drop(columns="_ok0"))
    if not parts:
        raise RuntimeError("no events were produced")
    df = pd.concat(parts, ignore_index=True)
    df["bin_book"] = np.sign(df["ret_side"]).astype(np.int8)                       # Snippet 3.7 case 1: bin = sign(return) in (-1, 0, 1)
    df = df.sort_values(["t0", "symbol"], kind="stable").reset_index(drop=True)
    u_arr, w_arr = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for sym, grp in df.groupby("symbol", sort=False):
        ix = grp.index.to_numpy()
        u_arr[ix], w_arr[ix] = BE.attribution_weights_t(grp["pos"].to_numpy(np.int64), grp["t1pos"].to_numpy(np.int64), np.log(bars[sym].close))
    df["u"], df["w_raw"] = u_arr, w_arr
    df["w"] = df["w_raw"] * len(df) / df["w_raw"].sum()
    info = {"per_symbol": per_symbol, "symbols_fd": info_sym, "events_kept": int(len(df)), "events_cusum": int(sum(x["events_total"] for x in per_symbol)),
            "kind_share": {str(k): float((df["kind"] == k).mean()) for k in (1, -1, 0)}, "bin_share": float((df["bin_book"] > 0).mean()), "capped_share": float(df["capped"].mean()),
            "median_forward_bars": float(df["n_fwd"].median())}
    log(f"book events: CUSUM {info['events_cusum']:,}, kept {len(df):,}; exits pt {info['kind_share']['1']:.2f} sl {info['kind_share']['-1']:.2f} vertical {info['kind_share']['0']:.2f}; up-share {info['bin_share']:.3f}")
    return df, info
