"""Shared helpers for run_book_ports_bars.py: loading a venue's dollar bars and its finished meta-label run, the pooled-event time index, deterministic samples, daily closes."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import bar_loader as BL

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Venue:
    """One venue: its bar folder, its finished run_meta_bars.py run (events, predictions, summary), and the bars of a set of symbols."""

    def __init__(self, name: str, src: Path, bars_dir: Path, subset_n: int):
        self.name, self.src, self.bars_dir = name, Path(src), Path(bars_dir)
        self.summary = json.loads((self.src / "summary_numbers.json").read_text(encoding="utf-8"))
        self.events = pd.read_csv(self.src / "events.csv")
        self.symbols_df = pd.read_csv(self.src / "symbols.csv")
        self.features = list(self.summary["params"]["features"])
        self.manifest = BL.read_manifest(self.bars_dir)
        self.break_s = int(self.manifest["break_hours"]) * 3600
        counts = self.events.groupby("symbol").size().sort_values(ascending=False, kind="stable")
        counts = counts.reset_index().sort_values([0, "symbol"], ascending=[False, True], kind="stable")
        self.by_events = list(counts["symbol"])
        self.subset = sorted(self.by_events if len(self.by_events) <= subset_n else self.by_events[:subset_n])
        self.max_gap_days = 1 if name == "kraken" else 5     # longest gap between two consecutive daily closes that still counts as one "daily" return (24/7 market vs weekends and holidays)
        self._bars: dict = {}
        self.cache: dict = {}

    def bars(self, symbols=None) -> dict:
        syms = self.subset if symbols is None else list(symbols)
        for s in syms:
            if s not in self._bars:
                self._bars[s] = BL.load_one(self.bars_dir, s, self.break_s)
        return {s: self._bars[s] for s in syms}

    def cpcv_predictions(self) -> pd.DataFrame:
        return pd.read_csv(self.src / "cpcv_predictions.csv")


def pooled_index(events: pd.DataFrame) -> pd.DatetimeIndex:
    """UTC start time of each event, made unique: events of different symbols that start in the same second are separated by 1 ns steps in the order (t0, symbol).  The t1 values are untouched."""
    order = np.lexsort((events["symbol"].to_numpy(), events["t0"].to_numpy()))
    t0 = events["t0"].to_numpy(np.int64)[order]
    # position within a run of equal t0
    grp = np.cumsum(np.concatenate([[1], (np.diff(t0) != 0).astype(np.int64)]))
    first = pd.Series(np.arange(len(t0))).groupby(grp).transform("min").to_numpy()
    k = np.arange(len(t0)) - first
    idx_sorted = pd.to_datetime(t0, unit="s") + pd.to_timedelta(k, unit="ns")
    arr = np.empty(len(t0), dtype="datetime64[ns]")
    arr[order] = idx_sorted.to_numpy()
    return pd.DatetimeIndex(arr)


def sorted_events(events: pd.DataFrame) -> pd.DataFrame:
    """Events sorted by (t0, symbol), index = the unique UTC start time (pooled_index), with the t1 as a datetime column."""
    e = events.copy()
    e.index = pooled_index(e)
    e = e.sort_index(kind="stable")
    e["t1_dt"] = pd.to_datetime(e["t1"], unit="s")
    return e


def systematic_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Every k-th row of a time-sorted frame (k = len // n), so that the sample spans the whole history."""
    if len(df) <= n:
        return df
    k = len(df) // n
    return df.iloc[::k].iloc[:n]


def daily_closes(bars: dict) -> pd.DataFrame:
    """Last bar close of each UTC day per symbol (rows: UTC day; columns: symbols)."""
    cols = {}
    for s, b in bars.items():
        cols[s] = pd.Series(b.close, index=b.idx).resample("1D").last().dropna()
    return pd.DataFrame(cols).sort_index()


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, msg: str):
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def num(x):
    """JSON-safe number."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def trading_days(dc: pd.DataFrame, min_share: float = .8) -> pd.DataFrame:
    """The UTC days on which at least `min_share` of the columns have a close: drops the days of a data hole on which only a few symbols traded."""
    return dc[dc.notna().mean(axis=1) >= min_share]


def returns_within_gap(dc: pd.DataFrame, max_gap_days: int, kind: str = "log") -> pd.DataFrame:
    """Daily returns between consecutive rows of `dc`, formed only when the two rows are at most `max_gap_days` calendar days apart (no return is formed across a data hole)."""
    r = np.log(dc).diff() if kind == "log" else dc.pct_change(fill_method=None)
    gap = pd.Series(dc.index, index=dc.index).diff().dt.days.to_numpy()
    ok = np.zeros(len(dc), dtype=bool)
    ok[1:] = gap[1:] <= max_gap_days
    return r[ok]

