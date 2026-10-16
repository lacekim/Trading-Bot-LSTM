#!/usr/bin/env python3
"""Export 1h OHLC bars for the project's existing "robust" Alpaca-tradable
universe, as plain CSVs, so the long/short multibot GBDT pipeline (built
earlier this session for the 6 Kraken crypto symbols) can be run against
equities/bonds/commodity-ETF-proxies too.

RUN THIS ON A MACHINE/ENVIRONMENT THAT ACTUALLY HAS ALPACA NETWORK ACCESS --
the sandboxed remote shell used for the rest of this session confirmed it
has NO outbound network at all (tested against Alpaca's own endpoints and
unrelated control hosts; both failed identically), so this step can't be
done from there. Same pattern as the earlier Kraken feather->CSV export:
you run this where the live bot's own Alpaca credentials/network already
work, and the output lands in a file I can then read.

WHAT THIS DOES:
- Reuses the EXACT symbol universe and daily->Alpaca ticker mapping
  tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py already
  established (so this isn't a new/different universe -- same 72 "robust"
  equities, same GC=F->GLD / PA=F->PPLT commodity-proxy substitution, same
  bond ETFs IEF/SHY/TLT, excluding non-tradable index tickers ^TNX/^TYX,
  raw futures with no ETF proxy, forex, and the crypto already covered on
  Kraken/Binance).
- Pulls 1h bars (not 4h -- the multibot pipeline's features/labeling were
  built on 1h Kraken data, so 1h keeps the two runs comparable) as far back
  as Alpaca's feed will give (LOOKBACK_YEARS below, same 6-year convention
  the project's other Alpaca scripts already use).
- Writes one CSV per symbol to OUT_DIR, named "{symbol}_1h.csv", columns
  date,open,high,low,close,volume -- matching the Kraken CSV format exactly
  so the multibot pipeline's existing loader code barely has to change.

REAL CAVEAT, not hidden: "1h bars" means something different for equities
than for 24/7 crypto. A stock trades ~6.5 hours/day, 5 days/week, so the
MAX_HOLD=240-candle barrier in the multibot pipeline (240 CALENDAR hours,
~10 days, for crypto) would span roughly 240/6.5 =~ 37 TRADING days (~7.4
weeks) of real time for equities. That's a materially different real-world
holding period, carried over unchanged for comparability with the crypto
run -- flagged here, and will be flagged again in the final report, not
quietly adjusted.

USAGE:
    pip install alpaca-py python-dotenv pandas   # if not already installed
    python export_alpaca_1h_ohlc.py
Needs ALPACA_API_KEY / ALPACA_SECRET_KEY in the environment (same .env this
project's other Alpaca scripts use).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "tools") \
    else Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DAILY_RESULTS_PATH = PROJECT_ROOT / "runtime" / "cross_asset_golden_cross_results.json"
OUT_DIR = PROJECT_ROOT / "runtime" / "alpaca_1h_export"

LOOKBACK_YEARS = 6  # as far back as Alpaca's free/IEX feed will actually give us
REQUEST_DELAY_SECONDS = 0.3  # stay clear of free-tier rate limits across ~75 symbols
MIN_BARS_REQUIRED = 250  # a bit above SEQUENCE_WARMUP=210 in the multibot pipeline
FETCH_LIMIT = 50000  # alpaca-py paginates internally up to this many bars per symbol

# Same substitutions the project's existing Alpaca scripts already use --
# reused verbatim, not redefined, so this covers the same instruments the
# live equities signal would actually trade.
FUTURES_TO_ETF_PROXY = {"GC=F": "GLD", "PA=F": "PPLT"}
EQUITY_SYMBOL_MAP = {"BRK-B": "BRK.B"}
MIGRATED_CRYPTO = {"BTC-USD", "ETH-USD", "SOL-USD", "PAXG-USD"}  # already covered via Kraken/Binance this session
NON_TRADABLE_PREFIXES = ("^",)  # index tickers like ^TNX, ^TYX -- not a real position


def _alpaca_symbol(daily_symbol: str) -> str:
    return (
        FUTURES_TO_ETF_PROXY.get(daily_symbol)
        or EQUITY_SYMBOL_MAP.get(daily_symbol)
        or daily_symbol
    )


def load_universe() -> list[dict]:
    if not DAILY_RESULTS_PATH.exists():
        raise SystemExit(f"no daily backtest results at {DAILY_RESULTS_PATH}")
    data = json.loads(DAILY_RESULTS_PATH.read_text(encoding="utf-8"))
    out = []
    for r in data["results"]:
        sym = r["symbol"]
        if not r.get("robust"):
            continue
        if sym in MIGRATED_CRYPTO:
            continue
        if sym.startswith(NON_TRADABLE_PREFIXES):
            continue
        if r["asset_class"] == "Commodities" and sym not in FUTURES_TO_ETF_PROXY:
            continue  # raw future with no ETF proxy -- not Alpaca-tradable, skip
        if r["asset_class"] == "Forex":
            continue  # not part of this project's Alpaca-tradable universe
        out.append(r)
    return out


def fetch_1h_bars(client, symbol: str) -> pd.DataFrame | None:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    start = datetime.now(timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS)
    try:
        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=start,
            limit=FETCH_LIMIT,
        ))
        df = bars.df
    except Exception as exc:
        print(f"  {symbol}: fetch failed: {exc}")
        return None
    if df is None or df.empty:
        print(f"  {symbol}: no data returned")
        return None
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if len(df) < MIN_BARS_REQUIRED:
        print(f"  {symbol}: only {len(df)} 1h bars, need >={MIN_BARS_REQUIRED} -- skipping")
        return None
    return df


def main() -> None:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set (check .env)")

    from alpaca.data.historical import StockHistoricalDataClient
    client = StockHistoricalDataClient(api_key, secret_key)

    universe = load_universe()
    print(f"universe to export: {len(universe)} instruments "
          f"({sum(1 for r in universe if r['asset_class']=='Equities')} equities, "
          f"{sum(1 for r in universe if r['asset_class']=='Commodities')} commodity-ETF-proxies, "
          f"{sum(1 for r in universe if r['asset_class']=='Bonds/Rates')} bond ETFs)\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in universe:
        daily_symbol = r["symbol"]
        alpaca_symbol = _alpaca_symbol(daily_symbol)
        df = fetch_1h_bars(client, alpaca_symbol)
        time.sleep(REQUEST_DELAY_SECONDS)
        if df is None:
            summary.append({"symbol": daily_symbol, "alpaca_symbol": alpaca_symbol,
                             "asset_class": r["asset_class"], "n_bars": 0, "exported": False})
            continue
        out_path = OUT_DIR / f"{daily_symbol.replace('/', '-')}_1h.csv"
        out = df.reset_index().rename(columns={"timestamp": "date"})
        out["date"] = pd.to_datetime(out["date"], utc=True)
        out.to_csv(out_path, index=False)
        summary.append({"symbol": daily_symbol, "alpaca_symbol": alpaca_symbol,
                         "asset_class": r["asset_class"], "n_bars": len(df),
                         "first": str(df.index.min()), "last": str(df.index.max()),
                         "exported": True})
        print(f"  {daily_symbol} ({alpaca_symbol}): {len(df)} bars, "
              f"{df.index.min()} -> {df.index.max()} -> {out_path.name}")

    exported = [s for s in summary if s["exported"]]
    print(f"\nexported {len(exported)}/{len(universe)} symbols to {OUT_DIR}")
    (OUT_DIR / "_export_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {OUT_DIR / '_export_summary.json'}")


if __name__ == "__main__":
    main()
