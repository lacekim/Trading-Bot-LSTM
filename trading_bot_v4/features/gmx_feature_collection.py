"""Bounded hourly collection of causal GMX research features."""

from __future__ import annotations

import time

import requests

from trading_bot_v4.features.gmx_liquidity_history import (
    aggregate_liquidity_history,
    fetch_liquidity_history,
    persist_liquidity_history,
)
from trading_bot_v4.features.gmx_market_state import (
    MARKETS_INFO_URL,
    market_symbol_map,
    normalize_markets_info,
    persist_market_state,
)


def collect_hourly_gmx_features(timeout: float = 10.0) -> dict[str, int]:
    """Collect one current state and a two-hour overlap-safe history slice."""
    response = requests.get(MARKETS_INFO_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    mapping = market_symbol_map(payload)
    state = normalize_markets_info(payload)
    if state.empty or not mapping:
        raise RuntimeError("GMX markets/info returned no usable listed markets")
    persisted_state = persist_market_state(state)
    end = int(time.time())
    liquidity_payload = fetch_liquidity_history(end - 2 * 3600, end, timeout=max(timeout, 15.0))
    liquidity = aggregate_liquidity_history(liquidity_payload, mapping)
    persisted_liquidity = persist_liquidity_history(liquidity)
    return {
        "state_symbols": int(state["symbol"].nunique()),
        "state_rows": int(len(persisted_state)),
        "liquidity_rows_collected": int(len(liquidity)),
        "liquidity_rows": int(len(persisted_liquidity)),
    }
