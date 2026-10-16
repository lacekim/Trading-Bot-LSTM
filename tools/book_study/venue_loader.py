"""Reads hourly candle CSVs of ONE venue (e.g. Kraken) written by export_venue_candles.py: <dir>/<SYMBOL>_1h.csv with columns
date (UTC, 'YYYY-MM-DD HH:MM:SS'), open, high, low, close, volume.  Closes > 0 only; duplicate timestamps keep the last row.
Returns the same shapes as candle_loader: {symbol: close Series with a naive-UTC DatetimeIndex} and {symbol: DataFrame(high, low, close)}."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_symbols(candle_dir, timeframe: str = "1h"):
    suffix = f"_{timeframe}.csv"
    return sorted(f.name[: -len(suffix)] for f in Path(candle_dir).glob(f"*{suffix}") if not f.name.startswith("._"))


def _read(candle_dir, symbol, timeframe):
    d = pd.read_csv(Path(candle_dir) / f"{symbol}_{timeframe}.csv", usecols=["date", "high", "low", "close"], keep_default_na=False, na_values=[""])
    d["date"] = pd.to_datetime(d["date"])
    for c in ("high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["close"])
    d = d[d["close"] > 0]
    d = d.drop_duplicates("date", keep="last").sort_values("date")
    return d


def load_closes(candle_dir, symbols, timeframe: str = "1h", min_bars: int = 0):
    out, skipped = {}, []
    for s in symbols:
        d = _read(candle_dir, s, timeframe)
        if len(d) < max(1, min_bars):
            skipped.append({"symbol": s, "reason": f"fewer than {min_bars} bars ({len(d)})"})
            continue
        out[s] = pd.Series(d["close"].astype(float).to_numpy(), index=pd.DatetimeIndex(d["date"]))
    return out, skipped


def load_ohlc(candle_dir, symbols, timeframe: str = "1h", min_bars: int = 0):
    out, skipped = {}, []
    for s in symbols:
        d = _read(candle_dir, s, timeframe)
        if len(d) < max(1, min_bars):
            skipped.append({"symbol": s, "reason": f"fewer than {min_bars} bars ({len(d)})"})
            continue
        out[s] = pd.DataFrame({"high": d["high"].astype(float).to_numpy(), "low": d["low"].astype(float).to_numpy(), "close": d["close"].astype(float).to_numpy()},
                              index=pd.DatetimeIndex(d["date"]))
    return out, skipped
