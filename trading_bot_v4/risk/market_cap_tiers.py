"""Market-cap and GMX-liquidity admission/risk tiers for new positions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
COINPAPRIKA_TICKERS_URL = "https://api.coinpaprika.com/v1/tickers"
SNAPSHOT_PATH = Path("data/MARKET_CAP/market_cap_snapshots.csv")
GMX_STATE_PATH = Path("data/GMX_MARKET_STATE/gmx_market_state_1h.csv")
DEPRECATED_SYMBOLS = {"OM", "AI16Z"}
MAX_SNAPSHOT_AGE_HOURS = 48
MAX_SPREAD_BPS = {"PREMIUM": 50.0, "LARGE": 40.0, "MID": 30.0, "SMALL": 20.0}


@dataclass(frozen=True)
class AssetRiskDecision:
    allowed: bool
    tier: str
    market_cap_usd: float
    risk_pct: float
    max_position_pct: float
    reason: str = ""


TIERS = (
    # Liquidity/OI floors are calibrated to current GMX Arbitrum depth, not
    # centralized-exchange volumes. Position caps remain far below each floor.
    (10_000_000_000.0, "PREMIUM", 1.00, 20.0, 500_000.0, 200_000.0),
    (1_000_000_000.0, "LARGE", 0.75, 15.0, 100_000.0, 25_000.0),
    (100_000_000.0, "MID", 0.50, 10.0, 50_000.0, 5_000.0),
    (20_000_000.0, "SMALL", 0.25, 5.0, 25_000.0, 1_000.0),
)


def classify_market_cap(market_cap_usd: float) -> AssetRiskDecision:
    cap = float(market_cap_usd)
    for minimum, tier, risk, maximum, _liquidity, _oi in TIERS:
        if cap >= minimum:
            return AssetRiskDecision(True, tier, cap, risk, maximum)
    return AssetRiskDecision(False, "MICRO", cap, 0.0, 0.0, "micro-cap assets are watch-only")


def fetch_market_cap_snapshot(pages: int = 4, timeout: float = 15.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    observed_at = pd.Timestamp.now(tz="UTC").isoformat()
    try:
        for page in range(1, max(1, int(pages)) + 1):
            response = requests.get(COINGECKO_MARKETS_URL, params={
                "vs_currency": "usd", "order": "market_cap_desc", "per_page": 250,
                "page": page, "sparkline": "false",
            }, timeout=timeout)
            if response.status_code == 429 and rows:
                break
            response.raise_for_status()
            payload = response.json()
            if not payload:
                break
            rows.extend({
                "observed_at": observed_at, "source": "CoinGecko",
                "symbol": str(item.get("symbol", "")).upper(),
                "coin_id": str(item.get("id", "")), "name": str(item.get("name", "")),
                "market_cap_usd": item.get("market_cap"),
                "total_volume_usd": item.get("total_volume"),
            } for item in payload)
    except requests.RequestException:
        if not rows:
            response = requests.get(COINPAPRIKA_TICKERS_URL, params={"quotes": "USD"}, timeout=timeout)
            response.raise_for_status()
            rows.extend({
                "observed_at": observed_at, "source": "CoinPaprika",
                "symbol": str(item.get("symbol", "")).upper(),
                "coin_id": str(item.get("id", "")), "name": str(item.get("name", "")),
                "market_cap_usd": item.get("quotes", {}).get("USD", {}).get("market_cap"),
                "total_volume_usd": item.get("quotes", {}).get("USD", {}).get("volume_24h"),
            } for item in response.json())
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("CoinGecko returned no market-cap rows")
    frame[["market_cap_usd", "total_volume_usd"]] = frame[["market_cap_usd", "total_volume_usd"]].apply(
        pd.to_numeric, errors="coerce"
    )
    # Symbol collisions resolve conservatively to the most liquid/highest-cap
    # tracked coin. Deprecated contracts are rejected separately.
    return frame.sort_values(["symbol", "market_cap_usd"], ascending=[True, False]).drop_duplicates("symbol")


def persist_market_cap_snapshot(frame: pd.DataFrame, path: Path = SNAPSHOT_PATH) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True)
    combined["observed_at"] = pd.to_datetime(combined["observed_at"], utc=True, errors="coerce", format="mixed")
    combined = combined.dropna(subset=["observed_at", "symbol", "market_cap_usd"])
    combined = combined.drop_duplicates(["observed_at", "symbol"], keep="last").sort_values(["observed_at", "symbol"])
    combined.to_csv(path, index=False)
    return combined


def collect_market_caps(pages: int = 4, timeout: float = 15.0) -> pd.DataFrame:
    return persist_market_cap_snapshot(fetch_market_cap_snapshot(pages, timeout))


def evaluate_asset_risk(symbol: str, direction: str,
                        cap_path: Path = SNAPSHOT_PATH, gmx_path: Path = GMX_STATE_PATH) -> AssetRiskDecision:
    asset = str(symbol).upper()
    if asset in DEPRECATED_SYMBOLS:
        return AssetRiskDecision(False, "DEPRECATED", 0.0, 0.0, 0.0, "deprecated or migrated contract")
    if not cap_path.exists():
        return AssetRiskDecision(False, "UNKNOWN", 0.0, 0.0, 0.0, "market-cap snapshot unavailable")
    caps = pd.read_csv(cap_path)
    caps["observed_at"] = pd.to_datetime(caps["observed_at"], utc=True, errors="coerce", format="mixed")
    row = caps[caps["symbol"].astype(str).str.upper().eq(asset)].sort_values("observed_at").tail(1)
    if row.empty:
        return AssetRiskDecision(False, "UNKNOWN", 0.0, 0.0, 0.0, "market cap unavailable for symbol")
    age = (pd.Timestamp.now(tz="UTC") - row.iloc[0]["observed_at"]).total_seconds() / 3600.0
    if age > MAX_SNAPSHOT_AGE_HOURS:
        return AssetRiskDecision(False, "STALE", float(row.iloc[0]["market_cap_usd"]), 0.0, 0.0,
                                 "market-cap snapshot is stale")
    decision = classify_market_cap(float(row.iloc[0]["market_cap_usd"]))
    if not decision.allowed or not gmx_path.exists():
        return decision if not decision.allowed else AssetRiskDecision(
            False, decision.tier, decision.market_cap_usd, 0.0, 0.0, "GMX liquidity snapshot unavailable"
        )
    state = pd.read_csv(gmx_path)
    state["timestamp"] = pd.to_datetime(state["timestamp"], utc=True, errors="coerce", format="mixed")
    market = state[state["symbol"].astype(str).str.upper().eq(asset)].sort_values("timestamp").tail(1)
    if market.empty:
        return AssetRiskDecision(False, decision.tier, decision.market_cap_usd, 0.0, 0.0, "GMX liquidity unavailable")
    tier = next(values for values in TIERS if values[1] == decision.tier)
    directional = "available_liquidity_long_usd" if direction.upper() == "LONG" else "available_liquidity_short_usd"
    if float(market.iloc[0][directional]) < tier[4] or float(market.iloc[0]["open_interest_total_usd"]) < tier[5]:
        return AssetRiskDecision(False, decision.tier, decision.market_cap_usd, 0.0, 0.0,
                                 f"insufficient GMX liquidity/open interest for {decision.tier}")
    return decision


def liquid_symbols(direction: str, symbols: list[str]) -> list[str]:
    """Filters to symbols currently allowed to trade this direction under
    evaluate_asset_risk's market-cap tier + GMX liquidity/open-interest
    floors -- e.g. tonight's finding that every promoted per-asset symbol
    (WLD, MEW, ANIME, ORDI, SATS) turned out to be liquidity-blocked despite
    passing calibration. Restricting training to this set up front means any
    future promotion is automatically tradeable, instead of discovering the
    liquidity gap only after training on the full universe."""
    return [symbol for symbol in symbols if evaluate_asset_risk(symbol, direction).allowed]
