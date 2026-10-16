"""Is the hourly Kraken data right?  Compares, for each of the 27 coins, three hourly series over the period they share:
  A  Kraken's own 60-minute file          (<DUMP>/<PAIR>_60.csv, epoch seconds = start of the hour)
  B  Kraken's own 1-minute file aggregated to hours by this script  (open = first minute's open, high = max, low = min, close = last minute's close, volume = sum)
  C  the hourly candles the book study has used so far  (runtime/venue_candles_v2/kraken_1h/<SYM>_1h.csv, date = start of the hour, UTC)
Pairs: A-vs-B tells whether Kraken's own files agree with each other; A-vs-C tells whether the study's hourly candles are the same as Kraken's official hours.
Only hours present in BOTH series of a pair are compared; hours present in one only are counted (nothing is filled in).
Reported per pair and coin: hours in both, hours only in the first / only in the second, share of hours whose close differs by more than 0.1%, 1%, the median and max relative close difference, the same for high, low,
and the ratio of total volume (second / first) over the shared hours.  Output is one CSV (--out) and a text line per pair; nothing in the source folders is changed.
Pair names in the dump: BTC -> XBT, DOGE -> XDG, everything else SYM + USD.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAIR = {"BTC": "XBTUSD", "DOGE": "XDGUSD"}
SYMBOLS = ["ADA", "ALGO", "ANKR", "APT", "ARB", "ATOM", "AVAX", "BAL", "BNB", "BTC", "DOGE", "DOT", "ETH", "FIL", "ICP", "ICX", "KSM", "LINK", "LRC", "NEAR", "OP", "PAXG", "SOL", "TRX", "XLM", "XMR", "XRP"]
COLS = ["ts", "open", "high", "low", "close", "volume", "trades"]


def read_dump(path) -> pd.DataFrame:
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return pd.DataFrame(columns=COLS[:6])
    d = pd.read_csv(path, header=None, names=COLS, usecols=range(6), dtype={"ts": "int64"})
    d = d[d["close"] > 0].drop_duplicates("ts", keep="last").sort_values("ts")
    return d.reset_index(drop=True)


def minutes_to_hours(m: pd.DataFrame) -> pd.DataFrame:
    """1-minute rows (ts epoch s) -> hourly rows keyed by the hour start (epoch s)."""
    if not len(m):
        return pd.DataFrame(columns=COLS[:6])
    h = (m["ts"] // 3600) * 3600
    g = m.groupby(h, sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).reset_index()
    out = out.rename(columns={out.columns[0]: "ts"})
    return out[COLS[:6]]


def read_candles(path) -> pd.DataFrame:
    c = pd.read_csv(path)
    ts = (pd.to_datetime(c["date"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    return pd.DataFrame({"ts": ts.astype("int64"), "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"], "volume": c["volume"]}).sort_values("ts").reset_index(drop=True)


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """a, b: hourly frames with ts, open, high, low, close, volume.  Relative differences are |b - a| / a on the hours both have."""
    m = a.merge(b, on="ts", suffixes=("_a", "_b"))
    res = {"hours_both": int(len(m)), "only_a": int(len(set(a["ts"]) - set(b["ts"]))), "only_b": int(len(set(b["ts"]) - set(a["ts"])))}
    if not len(m):
        return res
    for k in ("close", "high", "low"):
        rd = (m[f"{k}_b"] - m[f"{k}_a"]).abs() / m[f"{k}_a"]
        res[f"{k}_gt_0.1pct"] = float((rd > 0.001).mean())
        res[f"{k}_gt_1pct"] = float((rd > 0.01).mean())
        res[f"{k}_median"] = float(rd.median())
        res[f"{k}_max"] = float(rd.max())
    sa, sb = float(m["volume_a"].sum()), float(m["volume_b"].sum())
    res["volume_ratio_b_over_a"] = sb / sa if sa > 0 else float("nan")
    res["first_shared"] = str(pd.Timestamp(int(m["ts"].iloc[0]), unit="s"))
    res["last_shared"] = str(pd.Timestamp(int(m["ts"].iloc[-1]), unit="s"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="folder with <PAIR>_1.csv and <PAIR>_60.csv (Kraken's OHLCVT download)")
    ap.add_argument("--candles", required=True, help="runtime/venue_candles_v2/kraken_1h")
    ap.add_argument("--out", required=True, help="CSV to write; must not exist")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    rows = []
    for sym in SYMBOLS:
        pair = PAIR.get(sym, sym + "USD")
        A = read_dump(Path(a.dump) / f"{pair}_60.csv")
        M = read_dump(Path(a.dump) / f"{pair}_1.csv")
        B = minutes_to_hours(M)
        cp = Path(a.candles) / f"{sym}_1h.csv"
        C = read_candles(cp) if cp.exists() else pd.DataFrame(columns=COLS[:6])
        for name, x, y in (("official60_vs_minute_agg", A, B), ("official60_vs_study_candles", A, C), ("minute_agg_vs_study_candles", B, C)):
            r = compare(x, y)
            r.update({"symbol": sym, "pair": name, "rows_first": len(x), "rows_second": len(y)})
            rows.append(r)
            print(sym, name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k in ("hours_both", "only_a", "only_b", "close_gt_0.1pct", "close_gt_1pct", "volume_ratio_b_over_a")}, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
