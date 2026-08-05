#!/usr/bin/env python3
"""
Backfill pre-GMX historical OHLCV from Binance's free public data archive
(data.binance.vision) into the bot's supplementary TradingView-history slot.

load_gmx_ohlc() already calls prepend_tradingview_history() on every load, so
files written here are picked up automatically by training, backtesting, and
live signal generation -- no other code changes needed.

Usage:
  python tools/backfill_binance_history.py --symbols BTC,ETH
  python tools/backfill_binance_history.py --all --timeframe 1h
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot import gmx_ohlc_path, list_gmx_symbols  # noqa: E402
from trading_bot_v4.config_v4 import V4Config as Config  # noqa: E402
from trading_bot_v4.features.tradingview_history import import_tradingview_csv  # noqa: E402

ARCHIVE_URL = "https://data.binance.vision/data/spot/monthly/klines/{pair}/{interval}/{pair}-{interval}-{year:04d}-{month:02d}.zip"
BINANCE_LAUNCH = (2017, 8)
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _months_between(start: tuple[int, int], end: tuple[int, int]):
    year, month = start
    while (year, month) <= end:
        yield year, month
        month += 1
        if month > 12:
            month, year = 1, year + 1


def gmx_start_month(symbol: str, timeframe: str) -> tuple[int, int] | None:
    path = gmx_ohlc_path(symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=lambda c: c.lower() in ("open_time", "date"))
    if df.empty:
        return None
    col = df.columns[0]
    first = pd.to_datetime(df[col], utc=True, errors="coerce").min()
    if pd.isna(first):
        return None
    return (first.year, first.month)


def fetch_month(pair: str, interval: str, year: int, month: int, timeout: float = 30.0,
                 retries: int = 4) -> pd.DataFrame | None:
    url = ARCHIVE_URL.format(pair=pair, interval=interval, year=year, month=month)
    response = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            first_line = handle.readline()
    has_header = first_line.strip().startswith(b"open_time")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            frame = pd.read_csv(
                handle, header=0 if has_header else None, names=KLINE_COLUMNS,
            )
    frame = frame[["open_time", "open", "high", "low", "close", "volume"]]
    # Binance monthly archives switched open_time from millisecond to microsecond
    # epoch partway through 2025; detect the unit per-batch before it's merged
    # with other months, since a combined magnitude heuristic can't tell them apart.
    magnitude = float(pd.to_numeric(frame["open_time"], errors="coerce").abs().median())
    if magnitude >= 1e17:
        unit = "ns"
    elif magnitude >= 1e14:
        unit = "us"
    elif magnitude >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit=unit, utc=True).dt.tz_localize(None)
    return frame


def backfill_symbol(symbol: str, timeframe: str, history_dir: Path, quote: str = "USDT",
                     pause_seconds: float = 0.2, verbose: bool = True) -> dict:
    interval = timeframe
    pair = f"{symbol.upper()}{quote}"
    cutoff = gmx_start_month(symbol, timeframe)
    if cutoff is None:
        return {"symbol": symbol, "status": "no_local_gmx_file"}
    end_year, end_month = cutoff
    if end_month == 1:
        end = (end_year - 1, 12)
    else:
        end = (end_year, end_month - 1)
    if end < BINANCE_LAUNCH:
        return {"symbol": symbol, "status": "gmx_already_predates_binance"}

    frames = []
    found_any = False
    trailing_miss_months = []
    for year, month in _months_between(BINANCE_LAUNCH, end):
        try:
            frame = fetch_month(pair, interval, year, month)
        except requests.RequestException as exc:
            if verbose:
                print(f"  {symbol} {year}-{month:02d}: request error {exc}")
            time.sleep(pause_seconds)
            trailing_miss_months.append((year, month))
            continue
        if frame is None and found_any:
            # A 404 after we've already seen real data for this symbol is
            # suspicious -- Binance's archive host has been observed to
            # return a transient false 404 under load. Re-check once before
            # accepting it as a genuine gap/delisting.
            time.sleep(1.0)
            try:
                frame = fetch_month(pair, interval, year, month)
            except requests.RequestException:
                frame = None
        if frame is None:
            time.sleep(pause_seconds)
            trailing_miss_months.append((year, month))
            continue
        found_any = True
        frames.append(frame)
        trailing_miss_months = []
        time.sleep(pause_seconds)

    if not frames:
        return {"symbol": symbol, "status": "not_on_binance", "pair": pair}
    if len(trailing_miss_months) >= 6:
        print(
            f"  WARNING {symbol}: {len(trailing_miss_months)} consecutive misses "
            f"immediately before the walk's end ({trailing_miss_months[0]} .. {trailing_miss_months[-1]}) "
            f"-- verify this isn't a transient gap rather than a real delisting."
        )

    combined = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    # pandas' default CSV serialization of a datetime64 column can mix
    # fractional-second formatting across rows (e.g. some rows get a
    # trailing ".000" and others don't), which makes normalize_tradingview_csv's
    # single-format pd.to_datetime parse silently drop every row after the
    # first formatting change. Force one explicit, uniform string format.
    combined["open_time"] = combined["open_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    combined = combined.rename(columns={"open_time": "time"})

    tmp_path = history_dir / f"_binance_raw_{symbol.upper()}_{timeframe}.csv"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(tmp_path, index=False)
    try:
        result = import_tradingview_csv(tmp_path, history_dir, symbol=symbol.upper(), timeframe=timeframe)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "symbol": symbol, "status": "ok", "pair": pair, "rows": result.rows,
        "first": str(result.first_timestamp), "last": str(result.last_timestamp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols, e.g. BTC,ETH")
    parser.add_argument("--all", action="store_true", help="Backfill every symbol with local GMX data")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--quote", type=str, default="USDT")
    args = parser.parse_args()

    if args.all:
        symbols = list_gmx_symbols(args.timeframe)
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("Provide --symbols or --all")

    history_dir = Path(Config.TRADINGVIEW_HISTORY_DIR)
    print(f"Backfilling {len(symbols)} symbol(s) into {history_dir}")
    results = []
    for symbol in symbols:
        started = time.time()
        result = backfill_symbol(symbol, args.timeframe, history_dir, quote=args.quote)
        elapsed = time.time() - started
        results.append(result)
        print(f"[{elapsed:5.1f}s] {result}")

    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\nDone. {len(ok)}/{len(results)} symbols backfilled.")


if __name__ == "__main__":
    main()
