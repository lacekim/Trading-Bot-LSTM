"""Downloads HOURLY bars from Alpaca for the stocks the daily Alpaca study used, so dollar bars (Chapter 2) can be built for equities.
Run on the Mac in the conda env `javier` (alpaca-py, python-dotenv, pandas).  It reads ALPACA_API_KEY / ALPACA_SECRET_KEY from the project's .env exactly as tools/run_alpaca_momentum_test.py does;
the keys are never printed or written anywhere.  It only READS market data (no orders, no account calls).

Universe: every symbol that appears in the daily run's universe.csv (the stocks that were ever in the top 200 by trailing dollar volume at a month-end).
Request: 1-hour bars, split- and dividend-adjusted (Adjustment.ALL), SIP feed, from --start to --end.  Alpaca's SDK pages through the full range.
What an hourly stock bar is: regular-session and extended-hours trades are both in the SIP feed, so the file contains pre-market, regular and after-hours hours.  A symbol has no row for an hour with no trade.
Nothing is filled in.  Resumable: a symbol whose CSV exists is skipped, so an interrupted run can be started again with the same command.

Output (--out): <SYMBOL>_1h.csv with columns date (UTC, 'YYYY-MM-DD HH:MM:SS' = the start of the hour), open, high, low, close, volume, trade_count, vwap;
coverage.txt (first / last bar and rows per symbol); manifest.json (md5 per file); failed.txt (symbols that failed all retries; the run exits with code 1 if any).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def read_universe(path) -> list[str]:
    u = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return sorted(u["symbol"].astype(str).unique())


def bars_to_frames(df: pd.DataFrame) -> dict:
    """df: the SDK's frame with a (symbol, timestamp) index or those two columns.  Returns {symbol: frame with the output columns}."""
    d = df.reset_index() if "symbol" not in df.columns else df.copy()
    d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_localize(None)
    out = {}
    for sym, g in d.groupby("symbol", sort=True):
        g = g.drop_duplicates("date", keep="last").sort_values("date")
        g = g[(g["close"] > 0)]
        o = pd.DataFrame({"date": g["date"].dt.strftime("%Y-%m-%d %H:%M:%S"), "open": g["open"].astype(float), "high": g["high"].astype(float), "low": g["low"].astype(float),
                          "close": g["close"].astype(float), "volume": g["volume"].astype(float),
                          "trade_count": g["trade_count"].astype(float) if "trade_count" in g.columns else float("nan"),
                          "vwap": g["vwap"].astype(float) if "vwap" in g.columns else float("nan")})
        out[str(sym)] = o
    return out


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def make_client():
    from dotenv import load_dotenv
    from alpaca.data.historical import StockHistoricalDataClient
    load_dotenv(ROOT / ".env")
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not k or not s:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set (check .env)")
    return StockHistoricalDataClient(k, s)


def fetch_batch(client, batch, start, end, feed, attempts=4):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    last = None
    for i in range(attempts):
        try:
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Hour, start=start, end=end, adjustment=Adjustment.ALL,
                                   feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
            return client.get_stock_bars(req).df
        except Exception as exc:                                   # noqa: BLE001
            last = exc
            print(f"  attempt {i + 1} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            time.sleep(3.0 * (i + 1))
    raise RuntimeError(f"batch failed after {attempts} attempts: {last}")


def run(universe, out, start, end, feed, batch_size, client, fetch=fetch_batch):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in universe if not (out / f"{s}_1h.csv").exists()]      # "._SYM_1h.csv" stubs are different names, so they never count as done
    print(f"{len(universe)} symbols, {len(universe) - len(todo)} already done, {len(todo)} to fetch", flush=True)
    failed = []
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        try:
            df = fetch(client, batch, start, end, feed)
        except Exception as exc:                                   # noqa: BLE001
            failed.extend(batch)
            print(f"  batch {i // batch_size}: giving up ({exc})", flush=True)
            continue
        frames = bars_to_frames(df) if df is not None and len(df) else {}
        for s in batch:
            f = frames.get(s)
            if f is None or not len(f):
                (out / f"{s}_1h.csv").write_text("date,open,high,low,close,volume,trade_count,vwap\n")     # an empty file records "asked, nothing returned"
            else:
                f.to_csv(out / f"{s}_1h.csv", index=False)
        print(f"  batch {i // batch_size}: {len(frames)}/{len(batch)} symbols with bars, {sum(len(f) for f in frames.values())} rows", flush=True)
        time.sleep(0.3)
    lines, manifest = [], {}
    for f in sorted(out.glob("*_1h.csv")):
        if f.name.startswith("._"):                                 # macOS AppleDouble stubs on external drives; they are not data and are not valid UTF-8
            continue
        d = pd.read_csv(f)
        manifest[f.name] = md5(f)
        lines.append(f"{f.name[:-7]:8} rows={len(d):>7} first={d['date'].iloc[0] if len(d) else '-'} last={d['date'].iloc[-1] if len(d) else '-'}")
    (out / "coverage.txt").write_text("\n".join(lines) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (out / "failed.txt").write_text("\n".join(failed) + ("\n" if failed else ""))
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, help="universe.csv of the daily Alpaca run (columns month_end, rank, symbol)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default=None, help="default: yesterday 00:00 UTC (the free SIP plan refuses queries that reach into the last 15 minutes, which the first attempt hit)")
    ap.add_argument("--feed", choices=("sip", "iex"), default="sip")
    ap.add_argument("--batch", type=int, default=20)
    a = ap.parse_args(argv)
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end
           else (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))
    failed = run(read_universe(a.universe), a.out, start, end, a.feed, a.batch, make_client())
    if failed:
        print(f"FAILED: {len(failed)} symbols (listed in failed.txt); run the same command again to retry them", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
