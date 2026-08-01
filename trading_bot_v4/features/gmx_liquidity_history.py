"""Historical hourly GMX JIT/GLV directional-liquidity features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests


LIQUIDITY_HISTORY_URLS = (
    "https://arbitrum.gmxapi.io/v1/jit/liquidity_history",
    "https://arbitrum.gmxapi.ai/v1/jit/liquidity_history",
)
LIQUIDITY_HISTORY_PATH = Path("data/GMX_MARKET_STATE/gmx_liquidity_history_1h.csv")
USD_SCALE = 1e30


def fetch_liquidity_history(from_timestamp: int, to_timestamp: int, timeout: float = 30.0) -> dict[str, Any]:
    errors = []
    params = {"period": "1h", "from": int(from_timestamp), "to": int(to_timestamp)}
    for url in LIQUIDITY_HISTORY_URLS:
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if "snapshots" not in payload:
                raise RuntimeError("response omitted snapshots")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("GMX liquidity history unavailable from both peers: " + " | ".join(errors))


def aggregate_liquidity_history(payload: dict[str, Any], market_symbols: dict[str, str]) -> pd.DataFrame:
    mapping = {str(key).lower(): value.upper() for key, value in market_symbols.items()}
    rows = []
    for item in payload.get("snapshots", []):
        symbol = mapping.get(str(item.get("marketAddress", "")).lower())
        if not symbol:
            continue
        try:
            rows.append({
                "timestamp": pd.to_datetime(int(item["timestamp"]), unit="s", utc=True),
                "symbol": symbol,
                "jit_liquidity_long_usd": float(item["longLiquidityUsd"]) / USD_SCALE,
                "jit_liquidity_short_usd": float(item["shortLiquidityUsd"]) / USD_SCALE,
            })
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).groupby(["timestamp", "symbol"], as_index=False).sum()
    total = frame["jit_liquidity_long_usd"] + frame["jit_liquidity_short_usd"]
    frame["jit_liquidity_total_usd"] = total
    frame["jit_liquidity_skew"] = (
        (frame["jit_liquidity_long_usd"] - frame["jit_liquidity_short_usd"])
        / total.clip(lower=1e-12)
    )
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def persist_liquidity_history(frame: pd.DataFrame, path: Path = LIQUIDITY_HISTORY_PATH) -> pd.DataFrame:
    if frame.empty:
        return frame
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce", format="mixed")
    combined = combined.dropna(subset=["timestamp", "symbol"])
    combined = combined.drop_duplicates(["timestamp", "symbol"], keep="last")
    combined = combined.sort_values(["timestamp", "symbol"])
    combined.to_csv(path, index=False)
    return combined


def backfill_liquidity_history(
    market_symbols: dict[str, str],
    from_timestamp: int,
    to_timestamp: int,
    chunk_days: int = 7,
    timeout: float = 30.0,
    path: Path = LIQUIDITY_HISTORY_PATH,
) -> pd.DataFrame:
    """Download bounded chunks because the API rejects very large ranges."""
    cursor = int(from_timestamp)
    end = int(to_timestamp)
    step = max(int(chunk_days), 1) * 86_400
    collected = []
    while cursor < end:
        chunk_end = min(cursor + step, end)
        payload = fetch_liquidity_history(cursor, chunk_end, timeout=timeout)
        frame = aggregate_liquidity_history(payload, market_symbols)
        if not frame.empty:
            collected.append(frame)
        cursor = chunk_end
    if not collected:
        return pd.DataFrame()
    return persist_liquidity_history(pd.concat(collected, ignore_index=True), path)
