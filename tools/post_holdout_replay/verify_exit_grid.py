#!/usr/bin/env python
"""Independent checker for an exit-grid output folder (stdlib only; imports neither exit_grid_stats nor replay_stats).

Recomputes the pooled cell statistics from the per-symbol rows, checks that the grid is complete and the three
periods are disjoint and correctly bounded, re-derives the default cell's basis-point statistics straight from the
saved baseline trade file, re-checks the self-test, and re-derives the selected cell and the rank agreements.
Exit code 0 = all pass.

    python tools/post_holdout_replay/verify_exit_grid.py runtime/post_holdout_exit_grid/<timestamp>
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


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def same(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return math.nan
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va and vb else math.nan


def cellkey(r) -> tuple:
    return (float(r["stop_atr"]), float(r["target_atr"]), int(float(r["max_hold"])))


def main(folder: str) -> int:
    d = Path(folder)
    checks: list[tuple[str, bool, str]] = []
    rows = read(d / "exit_grid_per_symbol.csv")
    cells_csv = read(d / "cell_summary.csv")
    blob = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    meta, numbers = blob["meta"], blob["numbers"]
    integrity = json.loads((d / "integrity.json").read_text(encoding="utf-8"))
    selftest = read(d / "selftest.csv") if nonempty(d / "selftest.csv") else []
    skipped = read(d / "skipped.csv") if nonempty(d / "skipped.csv") else []

    # 1. pooled cell statistics recomputed from the per-symbol rows
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        k = (r["direction"], r["period"]) + cellkey(r)
        for c in ("trades", "sum_net_bp", "sum_pre_cost_bp", "wins", "gross_win", "gross_loss", "net_pnl"):
            agg[k][c] += num(r[c])
    bad, seen = [], set()
    for c in cells_csv:
        k = (c["direction"], c["period"]) + cellkey(c)
        seen.add(k)
        a = agg.get(k)
        if a is None:
            bad.append(("missing in rows", k))
            continue
        n = a["trades"]
        exp_net = a["sum_net_bp"] / n if n else math.nan
        exp_pre = a["sum_pre_cost_bp"] / n if n else math.nan
        exp_pf = (a["gross_win"] / a["gross_loss"]) if a["gross_loss"] > 0 else (math.inf if a["gross_win"] > 0 else math.nan)
        if not (int(num(c["trades"])) == int(n) and same(num(c["mean_net_bp"]), exp_net) and same(num(c["mean_pre_cost_bp"]), exp_pre)
                and same(num(c["profit_factor"]), exp_pf) and same(num(c["net_pnl"]), a["net_pnl"], 1e-7)):
            bad.append((k, c["trades"], n))
    checks.append((f"cell_summary.csv matches a recomputation from the per-symbol rows ({len(cells_csv)} cells)", not bad and seen == set(agg), str(bad[:3])))

    # 2. grid completeness: every (direction, period, symbol) ran every cell, and windows match across cells
    all_cells = {cellkey(r) for r in rows}
    per_ds = defaultdict(set)
    windows = defaultdict(set)
    for r in rows:
        per_ds[(r["direction"], r["period"], r["symbol"])].add(cellkey(r))
        windows[(r["direction"], r["period"], r["symbol"])].add((r["window_start"], r["window_end"], r["candles"]))
    incomplete = [k for k, v in per_ds.items() if v != all_cells]
    checks.append((f"every symbol/period ran all {len(all_cells)} cells", not incomplete, str(incomplete[:3])))
    drift = [k for k, v in windows.items() if len(v) != 1]
    checks.append(("every cell used the identical candle window within a symbol/period", not drift, str(drift[:3])))
    checks.append(("the current default cell is in the grid", tuple(meta["default_cell"]) in {(a, b, int(c)) for a, b, c in all_cells}, str(meta["default_cell"])))

    distinct = {(r["direction"], r["period"], round(num(r["net_pnl"]), 6), int(num(r["trades"])), r["symbol"]) for r in rows}
    per_cell = defaultdict(set)
    for r in rows:
        per_cell[(r["direction"], r["period"], r["symbol"])].add((int(num(r["trades"])), round(num(r["net_pnl"]), 6), int(num(r["stop_exits"]))))
    flat = [k for k, v in per_cell.items() if len(all_cells) > 1 and len(v) == 1]
    checks.append(("the grid cells give different results (the exit override really took effect)", len(flat) < max(1, len(per_cell)) , f"{len(flat)} of {len(per_cell)} symbol/periods identical across all cells"))

    # 3. periods are disjoint and correctly bounded
    sel0, sel1 = meta["periods"]["select"]
    jun0, jun1 = meta["periods"]["june_2026"]
    by = defaultdict(list)
    for r in rows:
        by[r["period"]].append(r)
    ok_sel = all(sel0 <= r["window_start"] and r["window_end"] < sel1 for r in by.get("select", []))
    ok_jun = all(jun0 <= r["window_start"] and r["window_end"] < jun1 for r in by.get("june_2026", []))
    ok_post = all(r["window_start"] > r["holdout_end"] for r in by.get("post_holdout", []))
    checks.append(("select window lies inside 2026-01-01..2026-05-31", ok_sel, ""))
    checks.append(("june_2026 window lies inside 2026-06-01..2026-06-30", ok_jun, ""))
    checks.append(("post_holdout windows start strictly after each symbol's holdout end", ok_post, ""))
    last_sel = max((r["window_end"] for r in by.get("select", [])), default="")
    first_jun = min((r["window_start"] for r in by.get("june_2026", [])), default="9")
    checks.append(("select ends before june_2026 starts", last_sel < first_jun, f"{last_sel} vs {first_jun}"))

    # 4. self-test, re-derived from the saved baseline (rule 'always') without the script's own code
    bdir = meta.get("baseline_dir")
    if bdir and nonempty(Path(bdir) / "baseline_per_symbol.csv"):
        base = [r for r in read(Path(bdir) / "baseline_per_symbol.csv") if r["rule"] == "always"]
        dflt = tuple(meta["default_cell"])
        mism = []
        for seg in ("june_2026", "post_holdout"):
            g = {(r["direction"], r["symbol"]): int(num(r["trades"])) for r in rows if r["period"] == seg and cellkey(r) == (dflt[0], dflt[1], int(dflt[2]))}
            b = {(r["direction"], r["symbol"]): int(num(r["trades"])) for r in base if r["segment"] == seg}
            for k in g:
                if k in b and g[k] != b[k]:
                    mism.append((seg, k, g[k], b[k]))
        checks.append(("default cell reproduces the baseline's trade count for every symbol (june_2026 and post_holdout)", not mism, str(mism[:3])))
        # basis-point statistics re-derived from the baseline trades
        slip = float(meta["slippage_plus_impact_bps"]) / 1e4
        trades = read(Path(bdir) / "baseline_trades.csv") if nonempty(Path(bdir) / "baseline_trades.csv") else []
        stat = defaultdict(lambda: [0, 0.0, 0.0])
        for t in trades:
            if t["rule"] != "always" or t["segment"] not in ("june_2026", "post_holdout"):
                continue
            ent, ext, notional = num(t["entry_price"]), num(t["exit_price"]), num(t["notional"])
            long_ = t["direction"].upper() == "LONG"
            ref_e = ent / (1 + slip) if long_ else ent / (1 - slip)
            ref_x = ext / (1 - slip) if long_ else ext / (1 + slip)
            mv = ref_x / ref_e - 1.0
            s = stat[(t["segment"], t["direction"].lower())]
            s[0] += 1
            s[1] += (mv if long_ else -mv) * 1e4
            s[2] += num(t["net_pnl"]) / notional * 1e4
        bpbad = []
        for c in cells_csv:
            if (c["period"] in ("june_2026", "post_holdout")) and cellkey(c) == (dflt[0], dflt[1], int(dflt[2])):
                s = stat.get((c["period"], c["direction"]))
                if not s or s[0] != int(num(c["trades"])) or not same(s[1] / s[0], num(c["mean_pre_cost_bp"]), 1e-7) or not same(s[2] / s[0], num(c["mean_net_bp"]), 1e-7):
                    bpbad.append((c["period"], c["direction"], s and (s[0], s[1] / s[0], s[2] / s[0]), c["trades"], c["mean_pre_cost_bp"], c["mean_net_bp"]))
        checks.append(("default cell's pre-cost / net bp per trade re-derived from baseline_trades.csv match", bool(stat) and not bpbad, str(bpbad[:2]) if bpbad else "no baseline trades"))
    else:
        checks.append(("saved baseline available for the independent self-test", False, "baseline folder missing"))
    fails = [r for r in selftest if r["status"] != "PASS"]
    checks.append((f"script's own self-test: {len(selftest) - len(fails)}/{len(selftest)} comparisons PASS", bool(selftest) and not fails, str(fails[:2])))

    # 5. cost sanity: fee + slippage + impact must come to about 24 bp per trade in every well-populated cell
    odd = [(c["direction"], c["period"], cellkey(c), c["mean_cost_bp"]) for c in cells_csv if num(c["trades"]) >= 200 and not (23.0 <= num(c["mean_cost_bp"]) <= 25.0)]
    checks.append(("mean round-trip cost is about 24 bp in every cell with >= 200 trades", not odd, str(odd[:3])))

    # 6. selection and rank agreement re-derived
    table = defaultdict(dict)
    for c in cells_csv:
        table[(c["direction"], c["period"])][cellkey(c)] = c
    for direction, dnum in numbers["directions"].items():
        sel = table.get((direction, "select"), {})
        eligible = {k: num(v["mean_net_bp"]) for k, v in sel.items() if num(v["trades"]) >= meta["min_select_trades"]}
        exp = max(eligible, key=lambda k: eligible[k]) if eligible else None
        got = tuple(dnum["selected_cell"]) if dnum["selected_cell"] else None
        got = (got[0], got[1], int(got[2])) if got else None
        checks.append((f"{direction}: selected cell equals the best mean net bp in `select` ({exp})", exp == got, f"reported {got}"))
        for key, (pa, pb) in {"rank_agreement_select_vs_june": ("select", "june_2026"), "rank_agreement_select_vs_post": ("select", "post_holdout"),
                              "rank_agreement_june_vs_post": ("june_2026", "post_holdout")}.items():
            ta, tb = table.get((direction, pa), {}), table.get((direction, pb), {})
            idx = [k for k in ta if k in tb and num(ta[k]["trades"]) >= 200 and num(tb[k]["trades"]) >= 200]
            exp_r = spearman([num(ta[k]["mean_net_bp"]) for k in idx], [num(tb[k]["mean_net_bp"]) for k in idx]) if len(idx) >= 5 else math.nan
            checks.append((f"{direction}: {key} recomputed", same(exp_r, num(str(dnum[key])), 1e-9), f"{exp_r} vs {dnum[key]}"))

    # 7. run health
    errs = [(r["direction"], r["symbol"], r["reason"][:60]) for r in skipped if r["reason"].startswith("error")]
    checks.append(("at least one row was scored", len(rows) > 0, f"{len(rows)} rows"))
    checks.append(("no symbol ended in an internal error", not errs, str(errs[:3])))
    checks.append(("models/ calibration + file counts unchanged during the run", bool(integrity["unchanged"]), json.dumps(integrity["after"])[:200]))

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
