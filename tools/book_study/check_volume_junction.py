#!/usr/bin/env python
"""Does 'volume' mean the same thing in the old Kraken candles and in the candles rebuilt from trades?  (Read-only; writes one CSV.)

For every refreshed symbol it compares, on the bars present in BOTH sources (the overlap), the volume of the old export (runtime/venue_candles/kraken_1h/<SYM>_1h.csv)
with the volume of the trade-built refresh file (runtime/kraken_refresh/<SYM>_USD-1h.feather).  It also reports, on the merged series
(runtime/venue_candles_v2/kraken_1h/<SYM>_1h.csv), the mean volume of the 168 bars before the junction against the 168 bars after it.

Run from the project root in the `javier` conda env (needs pyarrow for the feather files):
    python tools/book_study/check_volume_junction.py
Writes runtime/kraken_refresh/volume_junction_check.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "runtime/venue_candles/kraken_1h"
NEW = ROOT / "runtime/kraken_refresh"
MERGED = ROOT / "runtime/venue_candles_v2/kraken_1h"
W = 168


def read_csv(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=["date", "close", "volume"])
    d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates("date", keep="last").sort_values("date").set_index("date")


def compare_overlap(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """old, new: frames indexed by date with a 'volume' column (new also drops its last, partial bar before the call)."""
    both = old.index.intersection(new.index)
    if len(both) == 0:
        return {"overlap_bars": 0, "sum_ratio_new_over_old": np.nan, "median_bar_ratio": np.nan, "bars_equal_within_1pct": np.nan}
    vo, vn = old.loc[both, "volume"].astype(float), new.loc[both, "volume"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (vn / vo).replace([np.inf, -np.inf], np.nan)
    return {"overlap_bars": int(len(both)), "sum_ratio_new_over_old": float(vn.sum() / vo.sum()) if vo.sum() > 0 else np.nan,
            "median_bar_ratio": float(ratio.median()), "bars_equal_within_1pct": float((ratio.sub(1.0).abs() <= 0.01).mean())}


def junction_step(merged: pd.DataFrame, last_old_bar: pd.Timestamp) -> dict:
    """Mean volume of the W bars up to and including the last old bar, and of the W bars after it, in the merged series."""
    pos = merged.index.searchsorted(last_old_bar, side="right")
    before, after = merged["volume"].iloc[max(0, pos - W):pos], merged["volume"].iloc[pos:pos + W]
    return {"mean_vol_before_168": float(before.mean()) if len(before) else np.nan, "mean_vol_after_168": float(after.mean()) if len(after) else np.nan,
            "after_over_before": float(after.mean() / before.mean()) if len(before) and len(after) and before.mean() > 0 else np.nan}


def main() -> int:
    rows = []
    for f in sorted(NEW.glob("*_USD-1h.feather")):
        if f.name.startswith("._"):             # macOS AppleDouble stubs on the external drive, not data
            continue
        sym = f.name.split("_USD-1h")[0]
        old_p, mer_p = OLD / f"{sym}_1h.csv", MERGED / f"{sym}_1h.csv"
        if not old_p.exists() or not mer_p.exists():
            continue
        new = pd.read_feather(f)
        new["date"] = pd.to_datetime(new["date"]).dt.tz_localize(None)
        new = new.drop_duplicates("date", keep="last").sort_values("date").set_index("date").iloc[:-1]       # the merge drops the last (partial) bar
        old, mer = read_csv(old_p), read_csv(mer_p)
        rec = {"symbol": sym, **compare_overlap(old, new), **junction_step(mer, old.index[-1])}
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(NEW / "volume_junction_check.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
