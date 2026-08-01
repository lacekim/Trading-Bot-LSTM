"""Persistent causal GMX positioning/funding/liquidity snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


MARKETS_INFO_URL = "https://arbitrum-api.gmxinfra.io/markets/info"
RATES_URLS = (
    "https://arbitrum.gmxapi.io/v1/rates",
    "https://arbitrum.gmxapi.ai/v1/rates",
)
SNAPSHOT_PATH = Path("data/GMX_MARKET_STATE/gmx_market_state_1h.csv")
USD_SCALE = 1e30


def _number(value: Any, scale: float = USD_SCALE) -> float:
    try:
        return float(value) / scale
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _symbol(name: str) -> str:
    return str(name).split("/USD", 1)[0].strip().upper()


def market_symbol_map(payload: dict[str, Any]) -> dict[str, str]:
    """Map GMX market-token addresses to index symbols."""
    return {
        str(market["marketToken"]).lower(): _symbol(market.get("name", ""))
        for market in payload.get("markets", [])
        if market.get("isListed", False) and market.get("marketToken") and _symbol(market.get("name", ""))
    }


def normalize_markets_info(payload: dict[str, Any], observed_at: pd.Timestamp | None = None) -> pd.DataFrame:
    timestamp = pd.Timestamp.now(tz="UTC").floor("h") if observed_at is None else pd.Timestamp(observed_at)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    rows = []
    for market in payload.get("markets", []):
        if not market.get("isListed", False):
            continue
        symbol = _symbol(market.get("name", ""))
        if not symbol:
            continue
        rows.append({
            "timestamp": timestamp, "symbol": symbol,
            "open_interest_long_usd": _number(market.get("openInterestLong")),
            "open_interest_short_usd": _number(market.get("openInterestShort")),
            "available_liquidity_long_usd": _number(market.get("availableLiquidityLong")),
            "available_liquidity_short_usd": _number(market.get("availableLiquidityShort")),
            "funding_rate_long": _number(market.get("fundingRateLong")),
            "funding_rate_short": _number(market.get("fundingRateShort")),
            "borrowing_rate_long": _number(market.get("borrowingRateLong")),
            "borrowing_rate_short": _number(market.get("borrowingRateShort")),
            "net_rate_long": _number(market.get("netRateLong")),
            "net_rate_short": _number(market.get("netRateShort")),
        })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    results = []
    for symbol, group in raw.groupby("symbol", sort=True):
        long_oi = group["open_interest_long_usd"].sum()
        short_oi = group["open_interest_short_usd"].sum()
        row = {
            "timestamp": timestamp, "symbol": symbol,
            "open_interest_long_usd": long_oi,
            "open_interest_short_usd": short_oi,
            "open_interest_total_usd": long_oi + short_oi,
            "open_interest_skew": (long_oi - short_oi) / max(long_oi + short_oi, 1e-12),
            "available_liquidity_long_usd": group["available_liquidity_long_usd"].sum(),
            "available_liquidity_short_usd": group["available_liquidity_short_usd"].sum(),
            "market_pool_count": len(group),
        }
        for column, weight in (
            ("funding_rate_long", "open_interest_long_usd"),
            ("funding_rate_short", "open_interest_short_usd"),
            ("borrowing_rate_long", "open_interest_long_usd"),
            ("borrowing_rate_short", "open_interest_short_usd"),
            ("net_rate_long", "open_interest_long_usd"),
            ("net_rate_short", "open_interest_short_usd"),
        ):
            values, weights = group[column].to_numpy(float), group[weight].to_numpy(float)
            valid = np.isfinite(values) & np.isfinite(weights)
            row[column] = float(np.average(values[valid], weights=weights[valid])) if valid.any() and weights[valid].sum() > 0 else float("nan")
        results.append(row)
    return pd.DataFrame(results).sort_values("symbol").reset_index(drop=True)


def fetch_market_state(timeout: float = 10.0) -> pd.DataFrame:
    response = requests.get(MARKETS_INFO_URL, timeout=timeout)
    response.raise_for_status()
    frame = normalize_markets_info(response.json())
    if frame.empty:
        raise RuntimeError("GMX markets/info returned no listed markets")
    return frame


def persist_market_state(frame: pd.DataFrame, path: Path = SNAPSHOT_PATH) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True)
    # CSV timestamps use ``+00:00`` while a newly fetched frame may use ``Z``.
    # Pandas 2 parses a mixed column strictly unless format="mixed" is explicit.
    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    combined = combined.dropna(subset=["timestamp", "symbol"]).drop_duplicates(["timestamp", "symbol"], keep="last")
    combined = combined.sort_values(["timestamp", "symbol"])
    combined.to_csv(path, index=False)
    return combined


def collect_market_state(timeout: float = 10.0) -> pd.DataFrame:
    return persist_market_state(fetch_market_state(timeout=timeout))


def fetch_historical_rates(period: str = "1y", timeout: float = 20.0) -> Any:
    errors = []
    for url in RATES_URLS:
        try:
            response = requests.get(url, params={"period": period}, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("GMX historical rates unavailable from both peers: " + " | ".join(errors))
