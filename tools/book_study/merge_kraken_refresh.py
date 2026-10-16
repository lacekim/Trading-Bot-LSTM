"""Join the Kraken hourly candles this project already exported (runtime/venue_candles/kraken_1h, from freqtrade-develop/user_data/data/kraken) with the
2026 refresh that `freqtrade download-data --dl-trades` wrote into runtime/kraken_refresh, into a NEW folder runtime/venue_candles_v2/kraken_1h.

Rules (declared before any result was seen):
  * the old file wins for every hour up to its own last bar; refresh bars are appended only after that hour;
  * the LAST bar of every refresh file is dropped: it was built from the trades up to the moment the download ended, so the hour is incomplete;
  * a symbol with no refresh file is copied unchanged; a refresh file for a symbol with no old file is used on its own;
  * bars with a missing or non-positive close are dropped and duplicate hours keep the last row (same rules as export_venue_candles.clean_kraken).
For every symbol the manifest records how many old and new bars were used, the gap in hours at the junction (0 = continuous), and how far the close
of the refresh candles (built from trades) differs from the old candles on the hours both have.
Nothing under freqtrade-develop/, runtime/venue_candles/ or runtime/kraken_refresh/ is written to. Needs pandas + pyarrow (the conda env `javier`).

Run from the project root:
    python tools/book_study/merge_kraken_refresh.py
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import export_venue_candles as X

HOUR = pd.Timedelta("1h")


def merge_old_new(old, new):
    """old, new: cleaned DataFrames (date UTC, open, high, low, close, volume) or None. Returns (merged, info)."""
    info = {"old_bars": 0 if old is None else len(old), "new_bars_file": 0 if new is None else len(new), "new_bars_added": 0, "dropped_partial_bar": None,
            "junction_gap_hours": None, "overlap_bars": 0, "overlap_close_max_rel_diff": None}
    if new is not None and len(new):
        info["dropped_partial_bar"] = str(new["date"].iloc[-1])
        new = new.iloc[:-1]
    if old is None or not len(old):
        out = new if new is not None else old
        info["new_bars_added"] = 0 if new is None else len(new)
        return out.reset_index(drop=True), info
    if new is None or not len(new):
        return old.reset_index(drop=True), info
    cut = old["date"].iloc[-1]
    both = old.merge(new[["date", "close"]], on="date", suffixes=("_old", "_new"))
    info["overlap_bars"] = int(len(both))
    if len(both):
        info["overlap_close_max_rel_diff"] = float(((both["close_new"] - both["close_old"]).abs() / both["close_old"]).max())
    add = new[new["date"] > cut]
    info["new_bars_added"] = int(len(add))
    if len(add):
        info["junction_gap_hours"] = int((add["date"].iloc[0] - cut) / HOUR) - 1
    out = pd.concat([old, add], ignore_index=True)
    return out, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--old", default="runtime/venue_candles")
    ap.add_argument("--refresh", default="runtime/kraken_refresh")
    ap.add_argument("--out", default="runtime/venue_candles_v2")
    args = ap.parse_args()
    root = Path(args.project).resolve()
    old_root, refresh, out = root / args.old, root / args.refresh, root / args.out
    old_man = json.loads((old_root / "manifest.json").read_text(encoding="utf-8"))["kraken"]
    if (out / "kraken_1h").exists():
        print(f"refusing to overwrite {out / 'kraken_1h'}")
        return 2
    (out / "kraken_1h").mkdir(parents=True)
    syms_old = sorted(old_man)
    syms_new = sorted(f.name.split("_")[0] for f in refresh.glob("*_USD-1h.feather") if not f.name.startswith("._"))
    manifest = {"made_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kraken": {}, "rules": __doc__.split("Rules (declared before any result was seen):")[1].split("For every symbol")[0].strip()}
    print(f"old export: {len(syms_old)} symbols; refresh: {len(syms_new)} symbols")
    for sym in sorted(set(syms_old) | set(syms_new)):
        old = new = None
        src = {}
        if sym in old_man:
            p = old_root / "kraken_1h" / f"{sym}_1h.csv"
            got = X.md5(p)
            if got != old_man[sym]["csv_md5"]:
                print(f"{sym}: old CSV md5 differs from its manifest, stopping")
                return 2
            old = X.clean_kraken(pd.read_csv(p))
            src["old_csv_md5"] = got
        if sym in syms_new:
            p = refresh / f"{sym}_USD-1h.feather"
            new = X.clean_kraken(pd.read_feather(p))
            src["refresh_feather_md5"] = X.md5(p)
        d, info = merge_old_new(old, new)
        dst = out / "kraken_1h" / f"{sym}_1h.csv"
        d.to_csv(dst, index=False, date_format="%Y-%m-%d %H:%M:%S")
        gaps = int((d["date"].diff().dropna() != HOUR).sum())
        manifest["kraken"][sym] = {"bars": len(d), "first": str(d["date"].iloc[0]), "last": str(d["date"].iloc[-1]), "gaps": gaps, **info, **src, "csv_md5": X.md5(dst)}
        print(f"  {sym:6s} {len(d):7d} bars {d['date'].iloc[0].date()} -> {d['date'].iloc[-1]}  old {info['old_bars']:6d} + new {info['new_bars_added']:5d}  junction gap {info['junction_gap_hours']}  "
              f"overlap {info['overlap_bars']} max close diff {info['overlap_close_max_rel_diff']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
