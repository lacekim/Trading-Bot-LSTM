"""Causal Kraken features aligned to GMX hourly model rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import Config


ALIASES = {"BTC": "XBT", "DOGE": "XDG"}
MACOS_DATALESS_FLAG = 0x40000000
FEATURES = [
    "kraken_return_1h", "kraken_momentum_6h", "kraken_volatility_24h",
    "kraken_volume_ratio_24h", "kraken_trade_count_ratio_24h",
    "gmx_kraken_return_divergence",
]


def kraken_hourly_path(symbol: str, root: Path | None = None) -> Path | None:
    directory = Path(root or Config.KRAKEN_OHLC_DIR)
    base = ALIASES.get(str(symbol).upper(), str(symbol).upper())
    for quote in ("USD", "USDT", "USDC"):
        path = directory / f"{base}{quote}_60.csv"
        # iCloud placeholders report their logical size but block for roughly a
        # minute and then fail when read. Use only files resident on disk.
        flags = getattr(path.stat(), "st_flags", 0) if path.exists() else 0
        if path.exists() and not flags & MACOS_DATALESS_FLAG:
            return path
    return None


def load_lagged_kraken_features(symbol: str, root: Path | None = None) -> pd.DataFrame:
    path = kraken_hourly_path(symbol, root)
    if path is None:
        return pd.DataFrame(columns=["timestamp", *FEATURES])
    names = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
    frame = pd.read_csv(path, header=None, names=names)
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame["timestamp"], errors="coerce"), unit="s", utc=True, errors="coerce"
    )
    for column in names[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates("timestamp")
    returns = frame["close"].pct_change()
    features = pd.DataFrame({
        "timestamp": frame["timestamp"],
        "kraken_return_1h": returns,
        "kraken_momentum_6h": frame["close"].pct_change(6),
        "kraken_volatility_24h": returns.rolling(24, min_periods=12).std(),
        "kraken_volume_ratio_24h": frame["volume"] / frame["volume"].rolling(24, min_periods=12).median(),
        "kraken_trade_count_ratio_24h": frame["trades"] / frame["trades"].rolling(24, min_periods=12).median(),
    })
    # Kraken files use candle-open timestamps. Shift all derived information by
    # one full row so a GMX decision never consumes an unfinished peer candle.
    features[FEATURES[:-1]] = features[FEATURES[:-1]].shift(1)
    return features.replace([np.inf, -np.inf], np.nan)


def add_kraken_cross_venue_features(data: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    parts = []
    source = data.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    for symbol, group in source.groupby("symbol", sort=False):
        peer = load_lagged_kraken_features(str(symbol), root)
        if peer.empty:
            continue
        merged = group.merge(peer, on="timestamp", how="inner")
        merged["gmx_kraken_return_divergence"] = (
            pd.to_numeric(merged["returns"], errors="coerce")
            - pd.to_numeric(merged["kraken_return_1h"], errors="coerce")
        )
        parts.append(merged)
    if not parts:
        return pd.DataFrame(columns=[*source.columns, *FEATURES])
    result = pd.concat(parts, ignore_index=True)
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
