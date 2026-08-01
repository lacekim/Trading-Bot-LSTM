"""Import TradingView CSV exports and prepend them to GMX history.

TradingView is used only as supplementary history.  Whenever timestamps overlap,
the GMX candle wins so live/paper behaviour continues to use the execution venue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


OHLCV = ["Open", "High", "Low", "Close", "Volume"]
_COLUMN_ALIASES = {
    "time": "Date",
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


@dataclass(frozen=True)
class ImportResult:
    symbol: str
    rows: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    output_path: Path


def _normalize_symbol(value: str) -> str:
    value = Path(value).stem.upper()
    # Common exports: BINANCE_XRPUSDT, XRPUSD_1h, KRAKEN-XRPUSD.
    tokens = [token for token in re.split(r"[^A-Z0-9]+", value) if token]
    candidate = tokens[-1] if tokens else value
    candidate = re.sub(r"(?:USDT|USDC|USD|PERP)$", "", candidate)
    candidate = re.sub(r"(?:1H|60)$", "", candidate)
    if not candidate:
        raise ValueError(f"Cannot infer an asset symbol from {value!r}; pass --symbol")
    return candidate


def normalize_tradingview_csv(path: str | Path) -> pd.DataFrame:
    """Read a TradingView export and return UTC-naive, hourly OHLCV rows."""
    path = Path(path)
    raw = pd.read_csv(path)
    rename = {}
    for column in raw.columns:
        normalized = re.sub(r"[^a-z]", "", str(column).lower())
        if normalized in _COLUMN_ALIASES:
            rename[column] = _COLUMN_ALIASES[normalized]
    frame = raw.rename(columns=rename)
    missing = [column for column in ["Date", *OHLCV] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing TradingView columns: {', '.join(missing)}")

    date_values = frame["Date"]
    numeric_dates = pd.to_numeric(date_values, errors="coerce")
    if numeric_dates.notna().all():
        magnitude = float(numeric_dates.abs().median())
        unit = "ms" if magnitude >= 1e11 else "s"
        dates = pd.to_datetime(numeric_dates, unit=unit, utc=True, errors="coerce")
    else:
        dates = pd.to_datetime(date_values, utc=True, errors="coerce")
    frame["Date"] = dates.dt.tz_localize(None)
    for column in OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Date", *OHLCV]).set_index("Date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.empty:
        raise ValueError(f"{path} contains no valid OHLCV rows")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{path} contains non-positive prices")
    if (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError(f"{path} contains invalid high prices")
    if (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError(f"{path} contains invalid low prices")
    return frame[OHLCV]


def import_tradingview_csv(
    source: str | Path,
    output_dir: str | Path,
    symbol: str | None = None,
    timeframe: str = "1h",
) -> ImportResult:
    """Normalize one export into the bot's supplementary-history directory."""
    if timeframe != "1h":
        raise ValueError("TradingView history integration currently requires 1h exports")
    source = Path(source)
    resolved_symbol = (symbol or _normalize_symbol(source.name)).upper()
    frame = normalize_tradingview_csv(source)

    output = Path(output_dir) / f"tradingview_{resolved_symbol}_{timeframe}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = normalize_tradingview_csv(output)
        frame = pd.concat([existing, frame]).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
    frame.reset_index().to_csv(output, index=False)
    return ImportResult(resolved_symbol, len(frame), frame.index.min(), frame.index.max(), output)


def prepend_tradingview_history(
    gmx_frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    history_dir: str | Path,
) -> pd.DataFrame:
    """Add only pre-GMX TradingView candles; never replace a GMX candle."""
    path = Path(history_dir) / f"tradingview_{symbol.upper()}_{timeframe}.csv"
    if not path.exists() or gmx_frame.empty:
        return gmx_frame
    history = normalize_tradingview_csv(path)
    older = history.loc[history.index < gmx_frame.index.min()]
    if older.empty:
        return gmx_frame
    combined = pd.concat([older, gmx_frame[OHLCV]]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]
