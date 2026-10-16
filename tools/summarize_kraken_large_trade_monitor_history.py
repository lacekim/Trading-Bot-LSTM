"""Aggregates the per-run JSON snapshots that
tools/run_kraken_whale_monitor_loop.sh archives into
runtime/kraken_large_trade_monitor_history/ into one time-series view per
asset, so a day-plus of runs can be read as a single picture instead of
dozens of separate JSON files.

Purely descriptive, like every other diagnostic in this project: it does
not propose a rule, a threshold, or a trading decision -- it reports how
often each asset's verdict was ACCUMULATION / DISTRIBUTION / NEUTRAL
across the collected runs, whether that verdict was STABLE (mostly one
verdict) or FLIP-FLOPPING (switches often run to run), and the mean/
median buy-side percentage of large-trade notional over the window.

"Flip-flopping" here means, precisely: the fraction of consecutive
snapshot-to-snapshot transitions (run i -> run i+1) where the verdict
changed. This is about RUN-TO-RUN stability of the signal itself, not
about whether the underlying market is trending -- a signal that changes
almost every run is a weak candidate for anything beyond a rough overlay,
regardless of which direction it points.

Reads every *.json file in the history directory (one per loop run,
written by run_kraken_whale_monitor_loop.sh -- NOT modified by this
script), in filename order (the loop's own UTC timestamp naming sorts
chronologically as plain strings).

Run it after the loop has finished (or at any point mid-run to check on
progress):

    python tools/summarize_kraken_large_trade_monitor_history.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = REPO_ROOT / "runtime" / "kraken_large_trade_monitor_history"
OUT_PATH = REPO_ROOT / "runtime" / "kraken_large_trade_monitor_history_summary.json"


def load_history(history_dir: Path) -> list[dict]:
    """Loads every snapshot JSON in chronological (filename) order.
    Skips (and reports) any file that fails to parse rather than aborting
    the whole summary over one bad run."""
    snapshots = []
    for path in sorted(history_dir.glob("*.json")):
        try:
            snapshots.append({"path": path.name, "data": json.loads(path.read_text())})
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping unreadable snapshot {path.name}: {exc}", file=sys.stderr)
    return snapshots


def extract_asset_series(snapshots: list[dict], label: str) -> list[dict]:
    """For one asset label (e.g. "BTC"), pulls the per-run verdict and
    pct_of_large_notional_that_is_buy out of each snapshot's results
    block, in order, skipping any run where that asset errored or is
    absent (e.g. a FETCH FAILED run) rather than treating a gap as a
    data point."""
    series = []
    for snap in snapshots:
        entry = snap["data"].get("results", {}).get(label)
        if not entry or "verdict" not in entry:
            continue
        series.append({
            "run": snap["path"],
            "verdict": entry["verdict"],
            "pct_buy": entry.get("pct_of_large_notional_that_is_buy"),
        })
    return series


def flip_rate(verdicts: list[str]) -> float | None:
    """Fraction of consecutive run-to-run transitions where the verdict
    changed. None if there are fewer than 2 usable runs to compare."""
    if len(verdicts) < 2:
        return None
    flips = sum(1 for a, b in zip(verdicts, verdicts[1:]) if a != b)
    return flips / (len(verdicts) - 1)


def summarize_asset(series: list[dict]) -> dict:
    verdicts = [s["verdict"] for s in series]
    pct_buys = [s["pct_buy"] for s in series if s["pct_buy"] is not None and not np.isnan(s["pct_buy"])]
    n = len(series)
    counts = {v: verdicts.count(v) for v in ("ACCUMULATION", "DISTRIBUTION", "NEUTRAL")}
    dominant = max(counts, key=counts.get) if n else None
    dominant_share = (counts[dominant] / n) if n and dominant else None
    fr = flip_rate(verdicts)
    return {
        "n_runs": n,
        "verdict_counts": counts,
        "dominant_verdict": dominant,
        "dominant_verdict_share": dominant_share,
        "flip_rate": fr,
        "stability": (
            "insufficient_data" if fr is None else
            "stable" if fr <= 0.2 else
            "mixed" if fr <= 0.5 else
            "flip_flopping"
        ),
        "mean_pct_buy": float(np.mean(pct_buys)) if pct_buys else None,
        "median_pct_buy": float(np.median(pct_buys)) if pct_buys else None,
        "first_run": series[0]["run"] if series else None,
        "last_run": series[-1]["run"] if series else None,
    }


def main() -> None:
    if not HISTORY_DIR.exists():
        print(f"no history directory found at {HISTORY_DIR} -- run the loop script first "
              f"(tools/run_kraken_whale_monitor_loop.sh)")
        return

    snapshots = load_history(HISTORY_DIR)
    print(f"loaded {len(snapshots)} snapshot(s) from {HISTORY_DIR}")
    if not snapshots:
        print("nothing to summarize yet")
        return

    labels = sorted({label for snap in snapshots for label in snap["data"].get("results", {})})

    results = {}
    print("=" * 100)
    print(f"{'ASSET':<8}{'N':>4}  {'ACC':>4}{'DIST':>5}{'NEUT':>5}   {'DOMINANT':<14}{'SHARE':>7}  "
          f"{'FLIP%':>7}  {'STABILITY':<16}{'MEAN %BUY':>10}")
    print("=" * 100)
    for label in labels:
        series = extract_asset_series(snapshots, label)
        summary = summarize_asset(series)
        results[label] = summary
        counts = summary["verdict_counts"]
        flip_pct = f"{summary['flip_rate'] * 100:.0f}%" if summary["flip_rate"] is not None else "n/a"
        mean_buy = f"{summary['mean_pct_buy']:.1f}%" if summary["mean_pct_buy"] is not None else "n/a"
        share = f"{summary['dominant_verdict_share'] * 100:.0f}%" if summary["dominant_verdict_share"] else "n/a"
        print(f"{label:<8}{summary['n_runs']:>4}  {counts.get('ACCUMULATION', 0):>4}"
              f"{counts.get('DISTRIBUTION', 0):>5}{counts.get('NEUTRAL', 0):>5}   "
              f"{str(summary['dominant_verdict']):<14}{share:>7}  {flip_pct:>7}  "
              f"{summary['stability']:<16}{mean_buy:>10}")

    print("=" * 100)
    print("STABILITY: 'stable' = verdict flips on <=20% of run-to-run transitions, 'mixed' <=50%,")
    print("'flip_flopping' = >50%. A flip-flopping verdict is a weak candidate for anything beyond")
    print("a rough overlay, regardless of which side it currently leans.")
    print("This is descriptive only -- no rule or threshold is being proposed here.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "n_snapshots": len(snapshots),
        "history_dir": str(HISTORY_DIR),
        "results": results,
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
