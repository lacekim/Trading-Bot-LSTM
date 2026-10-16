"""Exports the Kraken data that lies AFTER Kraken's official 1-minute files (those end 2025-03-31 23:59 UTC) to plain CSVs, so the dollar-bar builder can read it without pyarrow.
Run on the Mac in the conda env `javier` (it has pyarrow).  Reads only; nothing in the freqtrade folders is changed.

For each of the 27 coins it writes into --out (which must not exist):
  <SYM>_5m.csv                  freqtrade's 5-minute candles from 2025-04-01 00:00 UTC on  (columns ts, open, high, low, close, volume; ts = epoch seconds of the candle start)
  <SYM>_trades_ft_1m.csv        1-minute bars aggregated from freqtrade's <SYM>_USD-trades.feather        (ts, open, high, low, close, volume, count)
  <SYM>_trades_refresh_1m.csv   1-minute bars aggregated from runtime/kraken_refresh/<SYM>_USD-trades.feather (same columns)
  coverage.json / coverage.txt  first and last timestamp and row count of every file written, so it is visible which period each source covers
  manifest.json                 md5 of every file written
Trade rows are de-duplicated on their id, sorted by time, price > 0 and amount >= 0 only.  A minute with no trade has no row (nothing is filled in).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CUT = pd.Timestamp("2025-04-01", tz="UTC")
CUT_S = int(CUT.timestamp())
SYMBOLS = ["ADA", "ALGO", "ANKR", "APT", "ARB", "ATOM", "AVAX", "BAL", "BNB", "BTC", "DOGE", "DOT", "ETH", "FIL", "ICP", "ICX", "KSM", "LINK", "LRC", "NEAR", "OP", "PAXG", "SOL", "TRX", "XLM", "XMR", "XRP"]


def trades_to_minutes(tr: pd.DataFrame) -> pd.DataFrame:
    """tr has timestamp (int64 milliseconds), price, amount and optionally id.  Returns one row per minute that has a trade: ts (epoch s of the minute start), open, high, low, close, volume, count."""
    cols = [c for c in ("timestamp", "price", "amount", "id") if c in tr.columns]
    t = tr[cols].copy()
    if "id" in t.columns:
        t = t.drop_duplicates(subset=["id"], keep="first")
    t = t.dropna(subset=["timestamp", "price", "amount"])
    t = t[(t["price"] > 0) & (t["amount"] >= 0)]
    t = t.sort_values("timestamp", kind="stable")
    t["ts"] = (t["timestamp"].to_numpy(np.int64) // 60000) * 60
    g = t.groupby("ts", sort=True)
    out = g.agg(open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"), volume=("amount", "sum"), count=("price", "size")).reset_index()
    return out[["ts", "open", "high", "low", "close", "volume", "count"]]


def candles_to_rows(df: pd.DataFrame, cut_s: int = CUT_S) -> pd.DataFrame:
    """freqtrade candles (date tz-aware, open, high, low, close, volume) -> ts epoch seconds, rows with ts >= cut_s."""
    idx = pd.DatetimeIndex(df["date"])
    idx = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
    ts = ((idx - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy(np.int64)
    out = pd.DataFrame({"ts": ts, "open": df["open"].to_numpy(float), "high": df["high"].to_numpy(float), "low": df["low"].to_numpy(float),
                        "close": df["close"].to_numpy(float), "volume": df["volume"].to_numpy(float)})
    out = out[(out["ts"] >= cut_s) & (out["close"] > 0)].drop_duplicates("ts", keep="last").sort_values("ts")
    return out.reset_index(drop=True)


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def stamp(ts) -> str:
    return str(pd.Timestamp(int(ts), unit="s")) + "Z"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kraken-data", required=True, help="freqtrade-develop/user_data/data/kraken")
    ap.add_argument("--refresh", required=True, help="runtime/kraken_refresh")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    kd, rf = Path(a.kraken_data), Path(a.refresh)
    cover, manifest = {}, {}
    lines = [f"{'symbol':6} {'source':14} {'rows':>9}  first (UTC)            last (UTC)"]
    for sym in SYMBOLS:
        cover[sym] = {}
        jobs = [("5m", kd / f"{sym}_USD-5m.feather", "candles"), ("trades_ft_1m", kd / f"{sym}_USD-trades.feather", "trades"), ("trades_refresh_1m", rf / f"{sym}_USD-trades.feather", "trades")]
        for name, path, kind in jobs:
            if not path.exists():
                cover[sym][name] = None
                lines.append(f"{sym:6} {name:14} {'-':>9}  (no file)")
                continue
            df = pd.read_feather(path)
            if kind == "candles":
                rows = candles_to_rows(df)
            else:
                df = df[df["timestamp"] >= CUT_S * 1000]
                rows = trades_to_minutes(df) if len(df) else pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "count"])
            if not len(rows):
                cover[sym][name] = {"rows": 0}
                lines.append(f"{sym:6} {name:14} {0:>9}  (no rows on or after 2025-04-01)")
                continue
            dst = out / f"{sym}_{name}.csv"
            rows.to_csv(dst, index=False)
            cover[sym][name] = {"rows": int(len(rows)), "first_ts": int(rows["ts"].iloc[0]), "last_ts": int(rows["ts"].iloc[-1]), "days": int(rows["ts"].floordiv(86400).nunique())}
            manifest[dst.name] = md5(dst)
            lines.append(f"{sym:6} {name:14} {len(rows):>9}  {stamp(rows['ts'].iloc[0])}  {stamp(rows['ts'].iloc[-1])}")
            print(lines[-1], flush=True)
    (out / "coverage.json").write_text(json.dumps(cover, indent=2, sort_keys=True))
    (out / "coverage.txt").write_text("\n".join(lines) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
