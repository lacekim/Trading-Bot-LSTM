"""Reads hourly candle CSVs from directories (the same merge rule as the project loader and the independent checkers):
GMX candles, plus TradingView candles only where they are older than the first GMX candle; closes > 0 only.
Returns {symbol: close Series with a naive-UTC DatetimeIndex}."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_closes(gmx_dir, tv_dir, symbols, timeframe: str = "1h", min_bars: int = 0):
    out, skipped = {}, []
    for s in symbols:
        g, t = Path(gmx_dir) / f"gmx_arbitrum_{s}_{timeframe}.csv", Path(tv_dir) / f"tradingview_{s}_{timeframe}.csv"
        gm = pd.Series(dtype=float)
        tv = pd.Series(dtype=float)
        if g.exists():
            d = pd.read_csv(g, usecols=["open_time", "close"])
            gm = pd.Series(d["close"].astype(float).to_numpy(), index=pd.to_datetime(d["open_time"]))
        if t.exists():
            d = pd.read_csv(t, usecols=["Date", "Close"])
            tv = pd.Series(d["Close"].astype(float).to_numpy(), index=pd.to_datetime(d["Date"]))
        if len(gm):
            tv = tv[tv.index < gm.index.min()]
        c = pd.concat([tv, gm])
        c = c[~c.index.duplicated(keep="last")].sort_index()
        c = c[c > 0]
        if len(c) < max(1, min_bars):
            skipped.append({"symbol": s, "reason": f"fewer than {min_bars} bars ({len(c)})"})
            continue
        out[s] = c
    return out, skipped


def load_ohlc(gmx_dir, tv_dir, symbols, timeframe: str = "1h", min_bars: int = 0):
    """Like load_closes, with High and Low: {symbol: DataFrame(high, low, close)} on the same merged index (same merge rule, closes > 0 only)."""
    out, skipped = {}, []
    for s in symbols:
        g, t = Path(gmx_dir) / f"gmx_arbitrum_{s}_{timeframe}.csv", Path(tv_dir) / f"tradingview_{s}_{timeframe}.csv"
        gm = pd.DataFrame(columns=["high", "low", "close"], dtype=float)
        tv = pd.DataFrame(columns=["high", "low", "close"], dtype=float)
        if g.exists():
            d = pd.read_csv(g, usecols=["open_time", "high", "low", "close"])
            gm = pd.DataFrame({"high": d["high"].astype(float).to_numpy(), "low": d["low"].astype(float).to_numpy(), "close": d["close"].astype(float).to_numpy()},
                              index=pd.to_datetime(d["open_time"]))
        if t.exists():
            d = pd.read_csv(t, usecols=["Date", "High", "Low", "Close"])
            tv = pd.DataFrame({"high": d["High"].astype(float).to_numpy(), "low": d["Low"].astype(float).to_numpy(), "close": d["Close"].astype(float).to_numpy()},
                              index=pd.to_datetime(d["Date"]))
        if len(gm):
            tv = tv[tv.index < gm.index.min()]
        c = pd.concat([tv, gm])
        c = c[~c.index.duplicated(keep="last")].sort_index()
        c = c[c["close"] > 0]
        if len(c) < max(1, min_bars):
            skipped.append({"symbol": s, "reason": f"fewer than {min_bars} bars ({len(c)})"})
            continue
        out[s] = c
    return out, skipped
