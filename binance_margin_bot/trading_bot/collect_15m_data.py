"""Resumable collector for BTCUSDT-15m-data.csv.

Reads the last open_time already in the CSV and backfills forward from
there using Binance's public klines REST endpoint (no API key needed for
market data), in the exact same 13-column format the file already uses.
Safe to re-run any time -- always resumes from wherever the file currently
ends, never re-fetches or duplicates existing rows.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import requests

CSV_PATH = Path(__file__).parent / "BTCUSDT-15m-data.csv"
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000
KLINES_URL = "https://api.binance.us/api/v3/klines"
# api.binance.com returns HTTP 451 (geo-blocked) from US IPs; Binance.US
# serves the same public klines format for US-accessible requests.
BATCH_LIMIT = 1000  # Binance's max klines per request
COLUMNS = ["timestamp", "open_time", "open", "high", "low", "close", "volume",
           "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"]


def _last_open_time_ms() -> int:
    with CSV_PATH.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        last_row = None
        for row in reader:
            last_row = row
    if last_row is None:
        raise ValueError(f"{CSV_PATH} has no data rows to resume from")
    return int(last_row["open_time"])


def _fetch_klines(start_time_ms: int) -> list[list]:
    response = requests.get(KLINES_URL, params={
        "symbol": SYMBOL, "interval": INTERVAL,
        "startTime": start_time_ms, "limit": BATCH_LIMIT,
    }, timeout=15)
    response.raise_for_status()
    return response.json()


def _row_from_kline(kline: list) -> list:
    open_time_ms = int(kline[0])
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(open_time_ms / 1000))
    return [timestamp, open_time_ms, kline[1], kline[2], kline[3], kline[4], kline[5],
            int(kline[6]), kline[7], int(kline[8]), kline[9], kline[10], kline[11]]


def collect() -> int:
    last_open_time = _last_open_time_ms()
    next_start = last_open_time + INTERVAL_MS
    now_ms = int(time.time() * 1000)
    total_added = 0

    with CSV_PATH.open("a", newline="") as handle:
        writer = csv.writer(handle)
        while next_start < now_ms:
            klines = _fetch_klines(next_start)
            if not klines:
                break
            # Binance can return a partially-formed in-progress candle at the
            # tail -- only keep candles whose close_time has actually passed.
            closed = [k for k in klines if int(k[6]) < now_ms]
            for kline in closed:
                writer.writerow(_row_from_kline(kline))
            total_added += len(closed)
            if len(closed) < len(klines):
                break  # hit the in-progress candle, nothing more to fetch yet
            next_start = int(klines[-1][0]) + INTERVAL_MS
            handle.flush()
            time.sleep(0.3)  # stay well under Binance's public rate limit

    return total_added


if __name__ == "__main__":
    added = collect()
    print(f"Added {added} new 15m candles to {CSV_PATH}")
