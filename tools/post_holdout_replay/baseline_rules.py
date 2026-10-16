"""Pure helpers for the plain-rule baseline (numpy/pandas only, no project imports).

Three predeclared rules, no fitted parameters:
  ma200d  price vs its 200-DAY simple average (time-based window of 200 days of hourly closes;
          needs >= 80% of the 4,800 hourly bars present, otherwise no signal)   <- primary rule
  ma200h  price vs its 200-HOUR simple average (200 hourly bars; the same length as the project's
          price_vs_ma200 feature on 1h data)                                      <- variant
  always  a signal on every candle (control: shows what the stop/target/max-hold rules and
          costs do on their own, with no filter)                                  <- control
LONG signal: Close > average. SHORT signal: Close < average. Equal or missing average: no signal.
The average at candle k uses closes up to and including candle k only (no look-ahead), the same
timing as the models, whose input window ends on candle k and which trade at that candle's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = ("ma200d", "ma200h", "always")
PRIMARY_RULE = "ma200d"
MA_DAYS = 200
MA_DAYS_MIN_BARS = int(0.8 * MA_DAYS * 24)   # 3,840 of 4,800 hourly bars
MA_HOURS = 200

JUNE_2026 = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01"))   # [start, end)


def average_for_rule(close: pd.Series, rule: str) -> pd.Series:
    """Rolling average aligned to `close` (a Series with a unique, sorted DatetimeIndex)."""
    if rule == "ma200d":
        return close.rolling(f"{MA_DAYS}D", min_periods=MA_DAYS_MIN_BARS).mean()
    if rule == "ma200h":
        return close.rolling(MA_HOURS, min_periods=MA_HOURS).mean()
    if rule == "always":
        return pd.Series(np.nan, index=close.index)
    raise ValueError(f"unknown rule {rule!r}")


def signal_array(close, avg, direction: str, rule: str) -> np.ndarray:
    """Boolean array: True where the rule says to be in `direction` on that candle."""
    close = np.asarray(close, dtype=float)
    if rule == "always":
        return np.ones(len(close), dtype=bool)
    avg = np.asarray(avg, dtype=float)
    valid = np.isfinite(avg) & np.isfinite(close)
    if direction == "long":
        return valid & (close > avg)
    if direction == "short":
        return valid & (close < avg)
    raise ValueError(f"unknown direction {direction!r}")


def valid_average_bars(avg, rule: str) -> int:
    """Candles on which the rule could be evaluated (all of them for the control)."""
    if rule == "always":
        return int(len(avg))
    return int(np.isfinite(np.asarray(avg, dtype=float)).sum())


def window_slice(timestamps, start=None, end=None, min_len: int = 2):
    """(lo, hi) positional slice of sorted timestamps with start <= ts < end, or None if shorter than min_len."""
    ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    mask = pd.Series(True, index=ts.index)
    if start is not None:
        mask &= ts >= start
    if end is not None:
        mask &= ts < end
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) < min_len:
        return None
    return int(idx[0]), int(idx[-1]) + 1


def bars_in_market_pct(trades: pd.DataFrame, candles: int, hours_per_bar: float = 1.0) -> float:
    """Share of the candles during which a position was open (sum of trade durations / span)."""
    if candles <= 0 or trades is None or len(trades) == 0:
        return 0.0
    dur = (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])).dt.total_seconds() / 3600.0
    return float(min(100.0, dur.sum() / (candles * hours_per_bar) * 100.0))


def entries_in(trades: pd.DataFrame, start, end) -> pd.DataFrame:
    """Trades whose entry time is in [start, end]."""
    if len(trades) == 0:
        return trades
    et = pd.to_datetime(trades["entry_time"])
    return trades[(et >= start) & (et <= end)]


def entries_in_half_open(trades: pd.DataFrame, start, end) -> pd.DataFrame:
    """Trades whose entry time is in [start, end)."""
    if len(trades) == 0:
        return trades
    et = pd.to_datetime(trades["entry_time"])
    return trades[(et >= start) & (et < end)]
