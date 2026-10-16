#!/usr/bin/env python
"""Independent checker for a post-holdout replay output folder (stdlib only).

Recomputes everything it can from replay_trades.csv with its own code (it does not
import replay_stats) and compares with what the replay wrote.  Exit code 0 = all pass.

    python tools/post_holdout_replay/verify_replay.py runtime/post_holdout_replay/<timestamp>
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def pf(pnls: list[float]) -> float:
    gain = sum(p for p in pnls if p > 0)
    loss = -sum(p for p in pnls if p < 0)
    if loss > 0:
        return gain / loss
    return math.inf if gain > 0 else math.nan


def same(a: float, b: float, tol: float = 1e-6) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main(folder: str) -> int:
    d = Path(folder)
    checks: list[tuple[str, bool, str]] = []
    sym = read(d / "replay_per_symbol.csv")
    trades = read(d / "replay_trades.csv") if (d / "replay_trades.csv").stat().st_size > 1 else []
    summary = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    integrity = json.loads((d / "integrity.json").read_text(encoding="utf-8"))
    selftest = read(d / "selftest.csv") if (d / "selftest.csv").stat().st_size > 1 else []

    # 1. trade counts per symbol/direction match
    counts = defaultdict(int)
    for t in trades:
        counts[(t["direction"].lower(), t["symbol"])] += 1
    bad = [(r["direction"], r["symbol"], r["trades"], counts[(r["direction"], r["symbol"])])
           for r in sym if int(float(r["trades"])) != counts[(r["direction"], r["symbol"])]]
    checks.append(("per-symbol trade counts equal rows in replay_trades.csv", not bad, str(bad[:5])))

    # 2. no trade enters on or before that symbol's holdout end (no overlap with the holdout)
    hend = {(r["direction"], r["symbol"]): r["holdout_end"] for r in sym}
    orphans = [(t["direction"], t["symbol"]) for t in trades if (t["direction"].lower(), t["symbol"]) not in hend]
    overlap = [(t["direction"], t["symbol"], t["entry_time"]) for t in trades
               if (t["direction"].lower(), t["symbol"]) in hend and t["entry_time"] <= hend[(t["direction"].lower(), t["symbol"])]]
    checks.append(("every trade belongs to a scored symbol-direction", not orphans, str(sorted(set(orphans))[:5])))
    checks.append(("all trades enter strictly after the holdout end", not overlap, str(overlap[:5])))

    # 3. exits never precede entries
    order = [(t["symbol"], t["entry_time"], t["exit_time"]) for t in trades if t["exit_time"] < t["entry_time"]]
    checks.append(("no exit before its entry", not order, str(order[:5])))

    # 4. pooled PF / trades / net P&L per direction recomputed independently
    for direction, groups in summary.items():
        rows = [t for t in trades if t["direction"].lower() == direction]
        pnls = [num(t["net_pnl"]) for t in rows]
        s = groups["all"]
        ok = (len(pnls) == s["trades"] and same(pf(pnls), num(str(s["profit_factor"]))) and same(sum(pnls), float(s["net_pnl"]), 1e-6))
        checks.append((f"{direction}: pooled trades / PF / net P&L recomputed match summary", ok,
                       f"recomputed trades={len(pnls)} pf={pf(pnls)} net={sum(pnls):.2f} vs {s['trades']} {s['profit_factor']} {s['net_pnl']:.2f}"))
        cur = [num(t["net_pnl"]) for t in rows if t["methodology"] == "current"]
        leg = [num(t["net_pnl"]) for t in rows if t["methodology"] == "legacy"]
        checks.append((f"{direction}: current + legacy trades add up to all", len(cur) + len(leg) == len(pnls), f"{len(cur)}+{len(leg)} vs {len(pnls)}"))

    # 4b. direction-vs-size diagnostics: in range, and medians match the summary
    for direction, groups in summary.items():
        rows = [r for r in sym if r["direction"] == direction]
        for col, key in (("rho_signed", "median_rho_signed"), ("rho_abs", "median_rho_abs")):
            vals = sorted(v for v in (num(r.get(col, "nan")) for r in rows) if not math.isnan(v))
            in_range = all(-1.0 <= v <= 1.0 for v in vals)
            if vals:
                mid = len(vals) // 2
                med = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
            else:
                med = math.nan
            ok = in_range and same(med, num(str(groups["_symbols"].get(key, "nan"))))
            checks.append((f"{direction}: {col} within [-1, 1] and median matches summary", ok, f"recomputed median {med} vs {groups['_symbols'].get(key)}"))

    # 4c. the run actually scored something, and nothing failed internally
    skipped = read(d / "skipped.csv") if (d / "skipped.csv").exists() and (d / "skipped.csv").stat().st_size > 1 else []
    errs = [(r["direction"], r["symbol"], r["reason"][:60]) for r in skipped if r["reason"].startswith("error")]
    checks.append(("at least one symbol-direction was scored", len(sym) > 0, f"{len(sym)} rows"))
    checks.append(("no symbol-direction ended in an internal error", not errs, str(errs[:3])))

    # 5. production files untouched
    checks.append(("models/ calibration + file counts unchanged during the run", bool(integrity["unchanged"]), json.dumps(integrity["after"])[:200]))

    # 6. self-test reproduced stored holdout numbers
    fails = [r for r in selftest if r["status"] != "PASS"]
    checks.append((f"self-test: {len(selftest) - len(fails)}/{len(selftest)} stored holdouts reproduced", bool(selftest) and not fails,
                   str([(r['direction'], r['symbol']) for r in fails][:8]) if fails else ("no self-test rows" if not selftest else "")))

    width = max(len(c[0]) for c in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {'' if ok else detail}")
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
