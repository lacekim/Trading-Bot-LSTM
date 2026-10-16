#!/usr/bin/env python
"""Independent checker for a run_seed_pairs.py output folder (stdlib only; does not import run_seed_pairs.py).

  * seed_pairs.json: distinct seeds, one control and one treatment run per seed, every run and every per-run checker exited 0 (and the checker logs say "N/N checks
    passed" with N = N), each run's recorded seed and prior trial count equal the declared ones (the control's, and the treatment's when it was declared separately), the
    treatment's feature list extends the control's (or equals it when the harness was told so), all runs have the same number of events, and the scripts next to this
    checker are the ones recorded; when the control runs were reused from an earlier folder they must lie in that folder; when the treatment arguments declare --decay-c, every
    treatment run's decay_extra.json must state that constant;
  * the headline metrics of every run are re-read from that run's SUMMARY.md (a different file and code path from the summary_numbers.json the harness used) and must equal
    the row of seed_pairs.csv within the rounding of the printed number;
  * seed_pairs_diff.csv must equal treatment - control computed from seed_pairs.csv, exactly;
  * every cell of the summary table in seed_pairs_summary.md (arm means and sds, mean and sd of the differences, mean / se, min, max, count of positive differences) is recomputed
    from seed_pairs_diff.csv / seed_pairs.csv with plain arithmetic.
Exit code 0 = all checks pass.

    python tools/book_study/verify_seed_pairs.py runtime/book_seed_pairs/<tag>
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAN = math.nan
METRICS = ["cpcv_auc", "cpcv_meta_sr_annual", "cpcv_meta_bp", "cpcv_meta_minus_placebo_sr_daily", "wf_auc", "wf_meta_sr_annual",
           "june_cpcv_meta_bp", "june_cpcv_meta_minus_placebo_bp", "selloff_cpcv_meta_bp", "selloff_cpcv_meta_minus_placebo_bp", "dsr_prior_plus_one"]


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def same(a, b, tol) -> bool:
    if isnan(a) or isnan(b):
        return isnan(a) and isnan(b)
    return abs(a - b) <= tol + 1e-12


def read_csv(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(v):
    return sum(v) / len(v) if v else NAN


def sd(v):
    if len(v) < 2:
        return NAN
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def md_metrics(md: str, prior_plus_one: int) -> tuple:
    """(values, tolerances) of the headline metrics read from one run's SUMMARY.md."""
    v, tol = {}, {}

    def one(pat, text, grp=1):
        m = re.search(pat, text, re.M)
        if not m:
            raise ValueError(f"pattern not found: {pat}")
        return m.group(grp)

    i_cp, i_wf = md.index("## Combinatorial purged CV"), md.index("## Walk-forward check")
    i_win = md.index("## Reporting windows")
    cp, wf, win = md[i_cp:i_wf], md[i_wf:i_win], md[i_win:]
    v["cpcv_auc"], tol["cpcv_auc"] = float(one(r"Out-of-sample AUC per split: mean (-?[0-9.]+)", cp)), 0.0005
    m = re.search(r"^\| meta \| (-?[0-9.]+) \| (-?[0-9.]+) \| (-?[0-9.]+) \|", cp, re.M)
    if not m:
        raise ValueError("CPCV meta row not found")
    v["cpcv_meta_bp"], tol["cpcv_meta_bp"] = float(m.group(1)), 0.05
    v["cpcv_meta_sr_annual"], tol["cpcv_meta_sr_annual"] = float(m.group(3)), 0.005
    v["cpcv_meta_minus_placebo_sr_daily"], tol["cpcv_meta_minus_placebo_sr_daily"] = float(one(r"mean daily-Sharpe difference (-?[0-9.]+)", cp)), 0.00005
    row = re.search(rf"^\| {prior_plus_one} \| (-?[0-9.]+) \| (-?[0-9.]+) \|", cp, re.M)
    if not row:
        raise ValueError("DSR row not found")
    v["dsr_prior_plus_one"], tol["dsr_prior_plus_one"] = float(row.group(2)), 0.0005
    v["wf_auc"], tol["wf_auc"] = float(one(r"Train events [0-9,]+, test events [0-9,]+, AUC (-?[0-9.]+)", wf)), 0.0005
    m = re.search(r"^\| meta \| (-?[0-9.]+) \| (-?[0-9.]+) \| (-?[0-9.]+) \|", wf, re.M)
    if not m:
        raise ValueError("walk-forward meta row not found")
    v["wf_meta_sr_annual"], tol["wf_meta_sr_annual"] = float(m.group(3)), 0.005
    for w, tag in (("june_2026_ai_correction", "june"), ("sell_off_post_holdout", "selloff")):
        m = re.search(rf"^\| {w} \| CPCV \| (\S+) \| (\S+) \| (\S+) \| ", win, re.M)
        if not m:
            raise ValueError(f"window row {w} not found")
        a, b = num(m.group(2)), num(m.group(3))
        v[f"{tag}_cpcv_meta_bp"], tol[f"{tag}_cpcv_meta_bp"] = a, 0.05
        v[f"{tag}_cpcv_meta_minus_placebo_bp"], tol[f"{tag}_cpcv_meta_minus_placebo_bp"] = (a - b if not (isnan(a) or isnan(b)) else NAN), 0.1
    return v, tol


def main(folder: str) -> int:
    d = Path(folder)
    root = d.resolve().parents[2]
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    def report():
        for name, ok, detail in checks:
            print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
        n_ok = sum(1 for c in checks if c[1])
        print(f"\n{n_ok}/{len(checks)} checks passed")
        return 0 if n_ok == len(checks) else 1

    J = json.loads((d / "seed_pairs.json").read_text(encoding="utf-8"))
    seeds, runs, st = J["seeds"], J["runs"], J["settings"]
    rows = read_csv(d / "seed_pairs.csv")
    diffs = read_csv(d / "seed_pairs_diff.csv")
    md_sum = (d / "seed_pairs_summary.md").read_text(encoding="utf-8")
    by_run = {(r["seed"], r["arm"]): r for r in runs}

    check("seeds are distinct and there is exactly one control and one treatment run per seed",
          len(set(seeds)) == len(seeds) and len(runs) == 2 * len(seeds) and set(by_run) == {(s, a) for s in seeds for a in ("ctrl", "treat")}, str(sorted(by_run))[:200])
    check("every run and every per-run checker exited 0", all(r["run_exit"] == 0 and r["check_exit"] == 0 and r["run_dir"] for r in runs), str([(r["arm"], r["seed"], r["run_exit"], r["check_exit"]) for r in runs if not (r["run_exit"] == 0 and r["check_exit"] == 0)]))
    bad_log = []
    for r in runs:
        p = d / "logs" / f"{r['arm']}_{r['seed']}.check.log"
        txt = p.read_text(encoding="utf-8") if p.exists() else ""
        m = re.search(r"^(\d+)/(\d+) checks passed", txt, re.M)
        if not m or m.group(1) != m.group(2) or "FAIL" in txt:
            bad_log.append((r["arm"], r["seed"]))
    check("each per-run checker log ends with 'N/N checks passed' and contains no FAIL line", not bad_log, str(bad_log))

    summ = {}
    for r in runs:
        summ[(r["seed"], r["arm"])] = json.loads((root / r["run_dir"] / "summary_numbers.json").read_text(encoding="utf-8"))
    prior_of = {"ctrl": int(st["prior_trials"]), "treat": int(st["treat_prior_trials"]) if st.get("treat_prior_trials") is not None else int(st["prior_trials"])}
    check("each run's recorded seed and prior trial count equal the declared ones",
          all(int(summ[(s, a)]["params"]["seed"]) == s and int(summ[(s, a)]["params"]["prior_trials"]) == prior_of[a] for s in seeds for a in ("ctrl", "treat")))
    same_feats = st.get("features_relation", "extends") == "same"
    cf = {s: summ[(s, "ctrl")]["params"]["features"] for s in seeds}
    tf = {s: summ[(s, "treat")]["params"]["features"] for s in seeds}
    if same_feats:
        check("the treatment feature list equals the control's list (the treatment changes something other than the features)", all(tf[s] == cf[s] for s in seeds))
    else:
        check("the treatment feature list extends the control's list (same features first, more at the end)", all(tf[s][:len(cf[s])] == cf[s] and len(tf[s]) > len(cf[s]) for s in seeds))
    if st.get("ctrl_from"):
        check("the control runs lie in the harness folder named by ctrl_from (reused, not re-run)", all(by_run[(s, "ctrl")]["run_dir"].startswith(f"runtime/book_seed_pairs/{st['ctrl_from']}/ctrl_{s}/") for s in seeds),
              str([by_run[(s, "ctrl")]["run_dir"] for s in seeds][:2]))
    targs = (st.get("treat_args") or "").split()
    if "--decay-c" in targs:
        c_decl = float(targs[targs.index("--decay-c") + 1])
        cargs = (st.get("treat_check_args") or "").split()
        got = []
        for s in seeds:
            q = root / by_run[(s, "treat")]["run_dir"] / "decay_extra.json"
            got.append(json.loads(q.read_text(encoding="utf-8")).get("decay_c") if q.exists() else None)
        check("the decay constant in the treatment arguments, in the checker arguments and in every treatment run's decay_extra.json is the same number",
              "--decay-c" in cargs and float(cargs[cargs.index("--decay-c") + 1]) == c_decl and all(g == c_decl for g in got), f"declared {c_decl}, runs {got}")
    ev_counts = {int(summ[k]["meta"]["events"]) for k in summ}
    check("all runs have the same number of events (same candles and same events in every seed and arm)", len(ev_counts) == 1, str(ev_counts))
    changed = [n for n, md5 in J["script_md5"].items() if not (HERE / n).exists() or hashlib.md5((HERE / n).read_bytes()).hexdigest() != md5]
    check("the harness, run and checker scripts next to this checker are the ones recorded in seed_pairs.json", not changed, str(changed))

    # metrics re-read from each SUMMARY.md
    by_csv = {(int(r["seed"]), r["arm"]): r for r in rows}
    check("seed_pairs.csv has one row per run with the metric columns", len(rows) == len(runs) and set(by_csv) == set(by_run) and all(m in rows[0] for m in METRICS), "")
    bad = []
    for (s, a), r in by_run.items():
        md = (root / r["run_dir"] / "SUMMARY.md").read_text(encoding="utf-8")
        vals, tol = md_metrics(md, int(st["dsr_n"]) if st.get("dsr_n") is not None else prior_of[a] + 1)
        for m in METRICS:
            if not same(vals[m], num(by_csv[(s, a)][m]), tol[m]):
                bad.append((s, a, m, vals[m], by_csv[(s, a)][m]))
    check("every headline metric of every run in seed_pairs.csv equals the number re-read from that run's SUMMARY.md (within the printed rounding)", not bad, str(bad[:3]))

    # differences
    bad_d = []
    drows = {int(r["seed"]): r for r in diffs}
    for s in seeds:
        for m in METRICS:
            a, b = num(by_csv[(s, "treat")][m]), num(by_csv[(s, "ctrl")][m])
            e = a - b
            if s not in drows or not same(num(drows[s][m]), e, 1e-12):
                bad_d.append((s, m))
    check("seed_pairs_diff.csv equals treatment minus control computed from seed_pairs.csv, for every seed and metric", not bad_d and len(drows) == len(seeds), str(bad_d[:3]))

    # summary table
    table = {}
    for ln in md_sum.splitlines():
        m = re.match(r"^\| (\w+) \| (\S+) \((\S+)\) \| (\S+) \((\S+)\) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\S+) of (\d+) \|$", ln)
        if m and m.group(1) in METRICS:
            table[m.group(1)] = m.groups()[1:]
    check("the summary table lists every metric", set(table) == set(METRICS), str(set(METRICS) - set(table)))
    bad_t = []
    for m in METRICS:
        if m not in table:
            continue
        cv = [num(by_csv[(s, "ctrl")][m]) for s in seeds]
        tv = [num(by_csv[(s, "treat")][m]) for s in seeds]
        dv = [t - c for t, c in zip(tv, cv) if not (isnan(t) or isnan(c))]
        cvf, tvf = [x for x in cv if not isnan(x)], [x for x in tv if not isnan(x)]
        sdd = sd(dv)
        se = sdd / math.sqrt(len(dv)) if len(dv) > 1 else NAN
        exp = [mean(cvf), sd(cvf), mean(tvf), sd(tvf), mean(dv), sdd, (mean(dv) / se) if (len(dv) > 1 and se > 0) else NAN, min(dv) if dv else NAN, max(dv) if dv else NAN]
        got = [num(x) for x in table[m][:9]]
        tols = [5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-3, 5e-5, 5e-5]
        for k, (g, e, tl) in enumerate(zip(got, exp, tols)):
            if not same(g, e, tl):
                bad_t.append((m, k, g, e))
        if int(table[m][9]) != sum(1 for x in dv if x > 0) or int(table[m][10]) != len(dv):
            bad_t.append((m, "n_positive", table[m][9], table[m][10]))
    check("every cell of the summary table (arm means / sds, mean and sd of the differences, mean / se, min, max, count of positive differences) recomputed from seed_pairs.csv", not bad_t, str(bad_t[:3]))
    check("the summary states that all run checkers passed", "All run checkers passed: yes" in md_sum)
    return report()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    try:
        raise SystemExit(main(sys.argv[1]))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, FileNotFoundError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
