"""Reads the dollar bars written by build_dollar_bars.py (<dir>/bars/<SYM>.csv, <dir>/ref_1h/<SYM>.csv, <dir>/manifest.json) for the clock-free pipeline.

A bar is stamped with its END (when its close happened).  Per symbol this gives a Bars object:
  sym, t (bar end, epoch s, int64), start (bar start, epoch s), close, high, low (float arrays), idx (naive-UTC DatetimeIndex of the ends),
  brk (bool, length n-1: True where the next bar starts more than the venue's break limit after this bar ended, i.e. a data hole sits between them),
  seg (int, segment number: increases by one after every hole),
  ref_t, ref_high, ref_low, ref_close (the hourly reference series from the same rows; stamped with the END of the hour = hour start + 3600 s).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Bars:
    sym: str
    t: np.ndarray
    start: np.ndarray
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    brk: np.ndarray
    seg: np.ndarray
    ref_t: np.ndarray
    ref_high: np.ndarray
    ref_low: np.ndarray
    ref_close: np.ndarray

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def idx(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(self.t, unit="s"))

    def close_series(self) -> pd.Series:
        return pd.Series(self.close, index=self.idx)


def _epoch(s: pd.Series) -> np.ndarray:
    return ((pd.to_datetime(s) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy(np.int64)


def make_bars(sym, t, start, close, high, low, break_s, ref_t, ref_high, ref_low, ref_close) -> Bars:
    t = np.asarray(t, np.int64)
    start = np.asarray(start, np.int64)
    brk = (start[1:] - t[:-1]) > break_s if len(t) > 1 else np.zeros(0, bool)
    seg = np.concatenate([[0], np.cumsum(brk)]).astype(np.int64) if len(t) else np.zeros(0, np.int64)
    return Bars(sym, t, start, np.asarray(close, float), np.asarray(high, float), np.asarray(low, float), brk, seg,
                np.asarray(ref_t, np.int64), np.asarray(ref_high, float), np.asarray(ref_low, float), np.asarray(ref_close, float))


def list_symbols(venue_dir):
    return sorted(f.stem for f in (Path(venue_dir) / "bars").glob("*.csv") if not f.name.startswith("._"))


def read_manifest(venue_dir) -> dict:
    return json.loads((Path(venue_dir) / "manifest.json").read_text())


def load_one(venue_dir, sym, break_s) -> Bars:
    b = pd.read_csv(Path(venue_dir) / "bars" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "start", "high", "low", "close"])
    b = b.dropna(subset=["close"])
    b = b[b["close"] > 0]
    t, st = _epoch(b["date"]), _epoch(b["start"])
    order = np.argsort(t, kind="stable")
    t, st = t[order], st[order]
    keep = np.concatenate([[True], np.diff(t) > 0]) if len(t) else np.zeros(0, bool)
    cols = {k: b[k].to_numpy(float)[order][keep] for k in ("close", "high", "low")}
    r = pd.read_csv(Path(venue_dir) / "ref_1h" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "high", "low", "close"])
    r = r.dropna(subset=["close"])
    r = r[r["close"] > 0]
    rt = _epoch(r["date"]) + 3600
    o2 = np.argsort(rt, kind="stable")
    return make_bars(sym, t[keep], st[keep], cols["close"], cols["high"], cols["low"], break_s, rt[o2], r["high"].to_numpy(float)[o2], r["low"].to_numpy(float)[o2], r["close"].to_numpy(float)[o2])


def load_all(venue_dir, symbols=None, min_years: float = 0.5, log=print):
    """({symbol: Bars}, [skipped dicts]).  A symbol is used when its bars span at least min_years (last end - first end)."""
    man = read_manifest(venue_dir)
    break_s = int(man["break_hours"]) * 3600
    out, skipped = {}, []
    for s in (symbols or list_symbols(venue_dir)):
        m = man["symbols"].get(s, {})
        if not m.get("bars"):
            skipped.append({"symbol": s, "reason": "no bars"})
            continue
        b = load_one(venue_dir, s, break_s)
        if b.n < 2 or (b.t[-1] - b.t[0]) < min_years * 365 * 86400:
            skipped.append({"symbol": s, "reason": f"bars span less than {min_years} years"})
            continue
        if b.n != int(m["bars"]):
            raise SystemExit(f"{s}: {b.n} bars read, the manifest says {m['bars']}; stopping")
        out[s] = b
    return out, skipped
