#!/usr/bin/env python
"""Checker for a run_bet_avg_pairs.py output folder (stdlib only).

  * pairs.json lists one control and one treatment run per seed, every listed output folder exists, belongs to the listed run, and the scripts next to this checker are the recorded ones;
  * every per-run output folder passes verify_bet_averaging.py (the brute-force recomputation); use --skip-children to skip this (about 1 minute per folder);
  * pairs.csv equals the metrics read from each folder's averaged_summary.json; pairs_diff.csv equals treatment minus control exactly; every cell of the summary table in
    pairs_summary.md (arm means and sds, mean and sd of the differences, mean / se, min, max, count of positive differences) is recomputed with plain arithmetic.
Exit code 0 = all checks pass.

    python tools/book_study/verify_bet_avg_pairs.py runtime/book_bet_avg/pairs_<TAG> [--skip-children]
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

import verify_bet_averaging as VB

HERE = Path(__file__).resolve().parent
NAN = math.nan
METRICS = ["avg_cpcv_meta_bp", "avg_cpcv_meta_sr_annual", "avg_cpcv_meta_minus_placebo_sr_daily", "avg_cpcv_meta_minus_primary_sr_daily", "avg_cpcv_meta_cost_bp",
           "avg_wf_meta_sr_annual", "avg_june_meta_bp", "avg_june_meta_minus_placebo_bp", "avg_selloff_meta_bp", "avg_selloff_meta_minus_placebo_bp"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def isnan(x):
    return isinstance(x, float) and math.isnan(x)


def same(a, b, tol):
    if isnan(a) or isnan(b):
        return isnan(a) and isnan(b)
    return abs(a - b) <= tol + 1e-12


def mean(v):
    return sum(v) / len(v) if v else NAN


def sd(v):
    if len(v) < 2:
        return NAN
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics_from(S):
    w = S["cpcv"]["windows"]
    g = lambda k: w[k]["mean_bp_per_symbol_day"] if k in w and w[k]["mean_bp_per_symbol_day"] is not None else NAN  # noqa: E731
    m = {"avg_cpcv_meta_bp": S["cpcv"]["meta"]["mean_bp_per_symbol_day"], "avg_cpcv_meta_sr_annual": S["cpcv"]["meta"]["sr_annual"],
         "avg_cpcv_meta_minus_placebo_sr_daily": S["cpcv"]["meta_minus_placebo"]["mean_sr_daily_diff"], "avg_cpcv_meta_minus_primary_sr_daily": S["cpcv"]["meta_minus_primary"]["mean_sr_daily_diff"],
         "avg_cpcv_meta_cost_bp": S["cpcv"]["meta"]["cost_bp_per_symbol_day"], "avg_wf_meta_sr_annual": S["wf"]["meta"]["sr_annual"]}
    for wn, tag in (("june_2026_ai_correction", "june"), ("sell_off_post_holdout", "selloff")):
        m[f"avg_{tag}_meta_bp"] = g(f"meta|{wn}")
        m[f"avg_{tag}_meta_minus_placebo_bp"] = g(f"meta|{wn}") - g(f"placebo|{wn}")
    return {k: (NAN if v is None else float(v)) for k, v in m.items()}


def main(folder: str, skip_children: bool = False) -> int:
    d = Path(folder)
    root = d.resolve().parents[2]
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    J = json.loads((d / "pairs.json").read_text(encoding="utf-8"))
    seeds, runs = J["seeds"], J["runs"]
    by_run = {(r["seed"], r["arm"]): r for r in runs}
    check("distinct seeds and exactly one control and one treatment run per seed", len(set(seeds)) == len(seeds) and len(runs) == 2 * len(seeds) and set(by_run) == {(s, a) for s in seeds for a in ("ctrl", "treat")})
    changed = [n for n, m in J["script_md5"].items() if not (HERE / n).exists() or hashlib.md5((HERE / n).read_bytes()).hexdigest() != m]
    check("the scripts next to this checker are the ones recorded in pairs.json", not changed, str(changed))
    bad_link = []
    for r in runs:
        o = root / r["out"]
        try:
            ex = json.loads((o / "averaged_extra.json").read_text(encoding="utf-8"))
            ok = Path(ex["source_run"]).resolve() == (root / r["run_dir"]).resolve()
        except (OSError, KeyError, ValueError):
            ok = False
        if not ok:
            bad_link.append((r["arm"], r["seed"]))
    check("every listed output folder exists and was made from the listed run", not bad_link, str(bad_link))
    if skip_children:
        check("per-run output folders were NOT re-checked (--skip-children)", True)
    else:
        bad_child = []
        for r in runs:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = VB.main(str(root / r["out"]))
            if rc != 0:
                bad_child.append((r["arm"], r["seed"]))
            print(f"  child {r['arm']} {r['seed']}: exit {rc}", flush=True)
        check("every per-run output folder passes verify_bet_averaging.py (brute-force recomputation of all its numbers)", not bad_child, str(bad_child))

    rows = {(int(x["seed"]), x["arm"]): x for x in read_csv(d / "pairs.csv")}
    exp = {(r["seed"], r["arm"]): metrics_from(json.loads((root / r["out"] / "averaged_summary.json").read_text(encoding="utf-8"))) for r in runs}
    bad_r = [(k, m) for k, e in exp.items() for m in METRICS if k not in rows or not same(num(rows[k][m]), e[m], 1e-9)]
    check("pairs.csv: every metric of every run equals the number in that run's averaged_summary.json", not bad_r and set(rows) == set(exp), str(bad_r[:3]))
    drows = {int(x["seed"]): x for x in read_csv(d / "pairs_diff.csv")}
    bad_d = [(s, m) for s in seeds for m in METRICS if s not in drows or not same(num(drows[s][m]), exp[(s, "treat")][m] - exp[(s, "ctrl")][m], 1e-12)]
    check("pairs_diff.csv equals treatment minus control for every seed and metric", not bad_d and len(drows) == len(seeds), str(bad_d[:3]))
    md = (d / "pairs_summary.md").read_text(encoding="utf-8")
    table = {}
    for ln in md.splitlines():
        m = re.match(r"^\| (\w+) \| (\S+) \((\S+)\) \| (\S+) \((\S+)\) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\S+) of (\d+) \|$", ln)
        if m and m.group(1) in METRICS:
            table[m.group(1)] = m.groups()[1:]
    bad_t = []
    for m in METRICS:
        if m not in table:
            bad_t.append((m, "missing"))
            continue
        cv, tv = [exp[(s, "ctrl")][m] for s in seeds], [exp[(s, "treat")][m] for s in seeds]
        dv = [t - c for t, c in zip(tv, cv) if not (isnan(t) or isnan(c))]
        cvf, tvf = [x for x in cv if not isnan(x)], [x for x in tv if not isnan(x)]
        sdd = sd(dv)
        se = sdd / math.sqrt(len(dv)) if len(dv) > 1 else NAN
        ex = [mean(cvf), sd(cvf), mean(tvf), sd(tvf), mean(dv), sdd, (mean(dv) / se) if (len(dv) > 1 and se > 0) else NAN, min(dv) if dv else NAN, max(dv) if dv else NAN]
        got = [num(x) for x in table[m][:9]]
        for k, (g, e, tl) in enumerate(zip(got, ex, [5e-4] * 6 + [5e-3, 5e-4, 5e-4])):
            if not same(g, e, tl):
                bad_t.append((m, k, g, e))
        if int(table[m][9]) != sum(1 for x in dv if x > 0) or int(table[m][10]) != len(dv):
            bad_t.append((m, "n_positive"))
    check("every cell of the summary table recomputed from the per-run summaries with plain arithmetic", not bad_t and set(table) == set(METRICS), str(bad_t[:3]))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
    n_ok = sum(1 for c in checks if c[1])
    print(f"\n{n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    skip = "--skip-children" in argv
    argv = [a for a in argv if a != "--skip-children"]
    try:
        raise SystemExit(main(argv[0], skip))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, FileNotFoundError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
