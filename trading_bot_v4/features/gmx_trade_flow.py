"""Causal hourly GMX order-flow features from public executed trades."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests


TRADES_SEARCH_URLS = (
    "https://arbitrum.gmxapi.io/v1/trades/search",
    "https://arbitrum.gmxapi.ai/v1/trades/search",
)
TRADE_FLOW_PATH = Path("data/GMX_MARKET_STATE/gmx_trade_flow_1h.csv")
USD_SCALE = 1e30
INCREASE_ORDER_TYPES = {2, 3}
DECREASE_ORDER_TYPES = {4, 5, 6}
LIQUIDATION_ORDER_TYPES = {7}


def fetch_trade_page(
    from_timestamp: int,
    to_timestamp: int,
    cursor: str | None = None,
    limit: int = 300,
    timeout: float = 20.0,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "forAllAccounts": True,
        "fromTimestamp": int(from_timestamp),
        "toTimestamp": int(to_timestamp),
        # The live API enforces 300 even though the published Swagger currently
        # describes 1000. Keep the collector aligned with the actual contract.
        "limit": min(max(int(limit), 1), 300),
        "showDebugValues": False,
        "orderEventCombinations": [
            {"orderType": order_type, "eventName": "OrderExecuted",
             "isDepositOrWithdraw": False, "isTwap": is_twap}
            for order_type in sorted(INCREASE_ORDER_TYPES | DECREASE_ORDER_TYPES | LIQUIDATION_ORDER_TYPES)
            for is_twap in (False, True)
        ],
    }
    if cursor:
        body["cursor"] = cursor
    errors = []
    for url in TRADES_SEARCH_URLS:
        try:
            response = requests.post(url, json=body, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if "trades" not in payload:
                raise RuntimeError("response omitted trades")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("GMX trade search unavailable from both peers: " + " | ".join(errors))


def iter_executed_trades(
    from_timestamp: int,
    to_timestamp: int,
    timeout: float = 20.0,
    max_pages: int | None = 500,
) -> Iterable[dict[str, Any]]:
    cursor = None
    seen_cursors = set()
    pages = 0
    while True:
        payload = fetch_trade_page(from_timestamp, to_timestamp, cursor=cursor, timeout=timeout)
        for trade in payload["trades"]:
            if trade.get("eventName") == "OrderExecuted":
                yield trade
        pages += 1
        next_cursor = payload.get("nextCursor")
        if not payload.get("hasMore") or not next_cursor:
            return
        if next_cursor in seen_cursors:
            raise RuntimeError(f"GMX trade pagination repeated cursor {next_cursor}")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        if max_pages is not None and pages >= max_pages:
            raise RuntimeError(f"GMX trade pagination exceeded {max_pages} pages")


def aggregate_trade_flow(
    trades: Iterable[dict[str, Any]], market_symbols: dict[str, str]
) -> pd.DataFrame:
    rows = []
    mapping = {str(key).lower(): value.upper() for key, value in market_symbols.items()}
    for trade in trades:
        if trade.get("eventName") != "OrderExecuted":
            continue
        symbol = mapping.get(str(trade.get("marketAddress", "")).lower())
        order_type = int(trade.get("orderType", -1))
        if not symbol or order_type not in INCREASE_ORDER_TYPES | DECREASE_ORDER_TYPES | LIQUIDATION_ORDER_TYPES:
            continue
        try:
            size_usd = float(trade.get("sizeDeltaUsd", 0)) / USD_SCALE
            timestamp = pd.to_datetime(int(trade["timestamp"]), unit="s", utc=True).floor("h")
        except (TypeError, ValueError, OverflowError, KeyError):
            continue
        direction = "long" if bool(trade.get("isLong")) else "short"
        action = "increase" if order_type in INCREASE_ORDER_TYPES else "decrease"
        if order_type in LIQUIDATION_ORDER_TYPES:
            action = "liquidation"
        rows.append({"timestamp": timestamp, "symbol": symbol, "direction": direction,
                     "action": action, "size_usd": size_usd})
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    sizes = raw.pivot_table(index=["timestamp", "symbol"], columns=["direction", "action"],
                            values="size_usd", aggfunc="sum", fill_value=0)
    counts = raw.pivot_table(index=["timestamp", "symbol"], columns=["direction", "action"],
                             values="size_usd", aggfunc="count", fill_value=0)
    desired = [(d, a) for d in ("long", "short") for a in ("increase", "decrease", "liquidation")]
    output = pd.DataFrame(index=sizes.index)
    for direction, action in desired:
        output[f"{direction}_{action}_usd"] = sizes.get((direction, action), 0.0)
        output[f"{direction}_{action}_count"] = counts.get((direction, action), 0).astype(int) if (direction, action) in counts else 0
    output = output.reset_index()
    output["increase_flow_skew"] = (
        (output["long_increase_usd"] - output["short_increase_usd"])
        / (output["long_increase_usd"] + output["short_increase_usd"]).clip(lower=1e-12)
    )
    output["net_position_flow_usd"] = (
        output["long_increase_usd"] - output["long_decrease_usd"]
        - output["short_increase_usd"] + output["short_decrease_usd"]
    )
    return output.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def persist_trade_flow(frame: pd.DataFrame, path: Path = TRADE_FLOW_PATH) -> pd.DataFrame:
    if frame.empty:
        return frame
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce", format="mixed")
    combined = combined.dropna(subset=["timestamp", "symbol"])
    numeric = [column for column in combined.columns if column not in {"timestamp", "symbol"}]
    combined[numeric] = combined[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
    # Re-downloaded time ranges replace rather than double-count identical hours.
    combined = combined.drop_duplicates(["timestamp", "symbol"], keep="last")
    combined = combined.sort_values(["timestamp", "symbol"])
    combined.to_csv(path, index=False)
    return combined


def backfill_trade_flow(
    market_symbols: dict[str, str],
    from_timestamp: int,
    to_timestamp: int,
    workers: int = 4,
    timeout: float = 20.0,
    path: Path = TRADE_FLOW_PATH,
) -> pd.DataFrame:
    """Backfill independent UTC-day slices with bounded concurrency."""
    start, end = int(from_timestamp), int(to_timestamp)
    manifest = path.with_suffix(path.suffix + ".completed_days")
    completed = set(manifest.read_text(encoding="utf-8").splitlines()) if manifest.exists() else set()
    ranges = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + 86_400, end)
        key = f"{cursor}:{chunk_end}"
        if key not in completed:
            ranges.append((cursor, chunk_end))
        cursor = chunk_end

    def download(bounds: tuple[int, int]) -> tuple[tuple[int, int], pd.DataFrame]:
        trades = iter_executed_trades(bounds[0], bounds[1], timeout=timeout)
        return bounds, aggregate_trade_flow(trades, market_symbols)

    latest = pd.read_csv(path) if path.exists() else pd.DataFrame()
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as pool:
        futures = {pool.submit(download, bounds): bounds for bounds in ranges}
        for future in as_completed(futures):
            bounds, frame = future.result()
            if not frame.empty:
                latest = persist_trade_flow(frame, path)
            completed.add(f"{bounds[0]}:{bounds[1]}")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("\n".join(sorted(completed)) + "\n", encoding="utf-8")
    return latest
