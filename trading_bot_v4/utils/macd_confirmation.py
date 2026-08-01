"""Mandatory closed-candle MACD confirmation for directional entries."""

from __future__ import annotations

import pandas as pd


def macd_components(close: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(close, errors="coerce")
    line = values.ewm(span=12, adjust=False, min_periods=26).mean() - values.ewm(
        span=26, adjust=False, min_periods=26
    ).mean()
    signal = line.ewm(span=9, adjust=False, min_periods=9).mean()
    return pd.DataFrame({"macd_line": line, "macd_signal": signal, "macd_histogram": line - signal})


def macd_entry_confirmation(signals: pd.DataFrame) -> pd.Series:
    required = {
        "macd_line", "macd_signal", "macd_histogram", "macd_histogram_previous",
        "price_vs_ma200",
    }
    if not required.issubset(signals.columns):
        return pd.Series(False, index=signals.index, dtype=bool)
    direction = signals["model_direction"].astype(str).str.upper()
    line = pd.to_numeric(signals["macd_line"], errors="coerce")
    signal = pd.to_numeric(signals["macd_signal"], errors="coerce")
    histogram = pd.to_numeric(signals["macd_histogram"], errors="coerce")
    previous = pd.to_numeric(signals["macd_histogram_previous"], errors="coerce")
    trend = pd.to_numeric(signals["price_vs_ma200"], errors="coerce")
    bullish = line.gt(signal) & histogram.gt(0) & previous.le(0) & trend.gt(0)
    bearish = line.lt(signal) & histogram.lt(0) & previous.ge(0) & trend.lt(0)
    return (direction.eq("LONG") & bullish) | (direction.eq("SHORT") & bearish)
