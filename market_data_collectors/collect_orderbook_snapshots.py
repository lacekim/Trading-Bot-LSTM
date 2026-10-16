"""Continuously polls Kraken's public order book (Depth) endpoint and logs
timestamped bid/ask snapshots to disk, building the historical order-book
dataset that doesn't exist anywhere for free -- confirmed earlier tonight
across six different sources (Kraken's own endpoints, Kraken's historical
trade-data download, an Apify connector, CryptoDataDownload, and Tardis.dev's
$700+/month paid tier being the only real historical source found).

Uses the public /Depth endpoint (no auth needed) rather than waiting on L3
API credentials to be set up -- known to have a rounding-up-to-3-decimals
bug on order sizes (can inflate reported liquidity up to 10x for small
orders, per the CryptoChassis/OCMDI writeup), which matters for precise
liquidity sizing but not for the primary goal here: measuring the realized
bid-ask spread over time, which reads off best bid/ask price levels, not
the (buggy) size field.

Runs as a long-lived loop (not relaunched per-poll by launchd) to keep
Python startup overhead out of the polling interval.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PAIR = "XBTUSD"
DEPTH_COUNT = 25  # levels per side
POLL_INTERVAL_SECONDS = 15
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = OUT_DIR / "kraken_orderbook_snapshots.csv"
LOG_PATH = OUT_DIR / "collector.log"

URL = f"https://api.kraken.com/0/public/Depth?pair={PAIR}&count={DEPTH_COUNT}"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def fetch_snapshot() -> dict | None:
    try:
        with urllib.request.urlopen(URL, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        log(f"fetch error: {exc}")
        return None
    if data.get("error"):
        log(f"api error: {data['error']}")
        return None
    result = data.get("result", {})
    if not result:
        return None
    pair_key = next(iter(result))
    return result[pair_key]


def ensure_header() -> None:
    if not OUT_PATH.exists():
        with OUT_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "snapshot_ts", "best_bid", "best_bid_size", "best_ask", "best_ask_size",
                "spread_abs", "spread_bps", "bid_ts", "ask_ts",
            ])


def append_row(book: dict) -> None:
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        return
    best_bid_price, best_bid_size, bid_ts = float(bids[0][0]), float(bids[0][1]), bids[0][2]
    best_ask_price, best_ask_size, ask_ts = float(asks[0][0]), float(asks[0][1]), asks[0][2]
    spread_abs = best_ask_price - best_bid_price
    mid = (best_ask_price + best_bid_price) / 2.0
    spread_bps = (spread_abs / mid) * 10000.0 if mid > 0 else float("nan")

    with OUT_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(), best_bid_price, best_bid_size,
            best_ask_price, best_ask_size, spread_abs, spread_bps, bid_ts, ask_ts,
        ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_header()
    log(f"starting order book collector: pair={PAIR} interval={POLL_INTERVAL_SECONDS}s -> {OUT_PATH}")
    poll_count = 0
    while True:
        book = fetch_snapshot()
        if book:
            append_row(book)
            poll_count += 1
            if poll_count % 240 == 0:  # ~once/hour at 15s interval
                log(f"{poll_count} snapshots collected so far")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
