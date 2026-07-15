"""Market-wide momentum discovery for the V4 paper scheduler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols, load_gmx_ohlc


MARKET_MOMENTUM_PATH = Path("logs/v4_market_momentum.csv")


def _return_pct(close: pd.Series, bars: int) -> float:
    clean = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= bars or float(clean.iloc[-bars - 1]) <= 0.0:
        return float("nan")
    return (float(clean.iloc[-1]) / float(clean.iloc[-bars - 1]) - 1.0) * 100.0


def scan_market_momentum(timeframe: str, limit: int = 10) -> pd.DataFrame:
    """Rank the complete cached market by recent, multi-horizon momentum."""
    rows: list[dict[str, object]] = []
    for symbol in list_gmx_symbols(timeframe):
        symbol = str(symbol).upper()
        try:
            raw = load_gmx_ohlc(symbol, timeframe)
            close = pd.to_numeric(raw["Close"], errors="coerce").dropna()
            if len(close) < 25:
                continue
            return_1h = _return_pct(close, 1)
            return_4h = _return_pct(close, 4)
            return_24h = _return_pct(close, 24)
            if not all(np.isfinite(value) for value in (return_1h, return_4h, return_24h)):
                continue
            # Favor sustained daily leadership while retaining breakout sensitivity.
            score = (0.15 * return_1h) + (0.30 * return_4h) + (0.55 * return_24h)
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "return_1h_pct": return_1h,
                    "return_4h_pct": return_4h,
                    "return_24h_pct": return_24h,
                    "momentum_score": score,
                    "latest_timestamp": close.index[-1],
                    "latest_price": float(close.iloc[-1]),
                }
            )
        except Exception:
            continue

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["latest_timestamp"] = pd.to_datetime(report["latest_timestamp"], errors="coerce")
    newest = report["latest_timestamp"].max()
    try:
        max_age = pd.Timedelta(timeframe) * 2
        report = report.loc[report["latest_timestamp"] >= newest - max_age].copy()
    except (TypeError, ValueError):
        pass
    if report.empty:
        return report
    report = report.sort_values(
        ["momentum_score", "return_24h_pct", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    report.insert(0, "momentum_rank", np.arange(1, len(report) + 1))
    MARKET_MOMENTUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(MARKET_MOMENTUM_PATH, index=False)
    return report.head(max(1, int(limit))).copy()


def momentum_symbols(timeframe: str, limit: int = 10) -> list[str]:
    report = scan_market_momentum(timeframe, limit=limit)
    return report["symbol"].astype(str).tolist() if not report.empty else []
