"""Copy the Kraken (crypto) and Alpaca (equities) candle files this project already holds into plain CSV, read-only on the sources.

Why: the cloud workspace that runs the book study cannot read feather or parquet files. This script runs on the Mac, where pandas + pyarrow
are installed, reads the existing files and writes NEW files under an output folder. It never writes into, renames or deletes anything in
freqtrade-develop/ or runtime/alpaca_daily/ (the live and research caches).

Kraken (crypto):  freqtrade-develop/user_data/data/kraken/<SYM>_USD-1h.feather  ->  <out>/kraken_1h/<SYM>_1h.csv
                  columns date (UTC), open, high, low, close, volume
Alpaca (equities): runtime/alpaca_daily/batch_*.parquet (+ bench.parquet)       ->  <out>/alpaca_1d/alpaca_daily.csv.gz
                  columns symbol, date (UTC), close, volume  (dividend- and split-adjusted; the fetch script keeps no open/high/low)

Run from the project root, in the conda env that has pandas and pyarrow:
    python tools/book_study/export_venue_candles.py
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

KRAKEN_COLS = ["open", "high", "low", "close", "volume"]
ALPACA_COLS = ["symbol", "date", "close", "volume"]


def clean_kraken(df: pd.DataFrame) -> pd.DataFrame:
    """UTC-sorted unique hourly bars with the five OHLCV columns; bars with a non-positive or missing close are dropped (same rule as the GMX loader)."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    for c in KRAKEN_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["close"])
    d = d[d["close"] > 0]
    d = d.sort_values("date").drop_duplicates("date", keep="last")
    return d[["date"] + KRAKEN_COLS].reset_index(drop=True)


def clean_alpaca(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, UTC day), the way the project's own loader reads these batches: drop missing closes, keep the last duplicate."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d["symbol"] = d["symbol"].astype(str)
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["close"])
    d = d.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return d[ALPACA_COLS].reset_index(drop=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".", help="project root (default: current folder)")
    ap.add_argument("--out", default="runtime/venue_candles", help="output folder, created if missing")
    args = ap.parse_args()
    root = Path(args.project).resolve()
    out = (root / args.out).resolve()
    kdir = root / "freqtrade-develop" / "user_data" / "data" / "kraken"
    adir = root / "runtime" / "alpaca_daily"
    for src in (kdir, adir):
        if not src.is_dir():
            print(f"missing source folder: {src}")
            return 2
    if out == kdir or out == adir or kdir in out.parents or adir in out.parents:
        print("refusing to write inside a source folder")
        return 2
    (out / "kraken_1h").mkdir(parents=True, exist_ok=True)
    (out / "alpaca_1d").mkdir(parents=True, exist_ok=True)
    manifest = {"made_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kraken": {}, "alpaca": {}}

    print("Kraken hourly (crypto)")
    for f in sorted(kdir.glob("*_USD-1h.feather")):
        sym = f.name.split("_")[0]
        raw = pd.read_feather(f)
        d = clean_kraken(raw)
        if d.empty:
            print(f"  {sym}: no usable bars")
            continue
        dst = out / "kraken_1h" / f"{sym}_1h.csv"
        d.to_csv(dst, index=False, date_format="%Y-%m-%d %H:%M:%S")
        gaps = int((d["date"].diff().dropna() != pd.Timedelta("1h")).sum())
        manifest["kraken"][sym] = {"bars": len(d), "first": str(d["date"].iloc[0]), "last": str(d["date"].iloc[-1]),
                                   "gaps": gaps, "source_md5": md5(f), "csv_md5": md5(dst)}
        print(f"  {sym:6s} {len(d):7d} bars  {d['date'].iloc[0].date()} -> {d['date'].iloc[-1]}  gaps={gaps}")

    print("Alpaca daily (equities)")
    files = sorted(adir.glob("batch_*.parquet")) + sorted(adir.glob("bench.parquet"))
    parts = []
    for f in files:
        p = pd.read_parquet(f)
        if len(p):
            parts.append(p[[c for c in ALPACA_COLS if c in p.columns]])
    if not parts:
        print("  no Alpaca bars found")
        return 2
    d = clean_alpaca(pd.concat(parts, ignore_index=True))
    dst = out / "alpaca_1d" / "alpaca_daily.csv.gz"
    d.to_csv(dst, index=False, compression="gzip", date_format="%Y-%m-%d")
    feed = (adir / "feed.txt").read_text().strip() if (adir / "feed.txt").exists() else "unknown"
    manifest["alpaca"] = {"files": [f.name for f in files], "symbols": int(d["symbol"].nunique()), "rows": len(d), "first": str(d["date"].min()),
                          "last": str(d["date"].max()), "feed": feed, "source_md5": {f.name: md5(f) for f in files}, "csv_md5": md5(dst)}
    print(f"  {d['symbol'].nunique()} symbols, {len(d)} daily bars, {d['date'].min().date()} -> {d['date'].max().date()}, feed={feed}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
