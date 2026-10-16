#!/usr/bin/env python
"""Ch.10 bet averaging over a whole seed-pair harness folder (read-only; no model is fitted, no new look).

Takes runtime/book_seed_pairs/<TAG> (made by run_seed_pairs.py), runs run_bet_averaging.run_one() on every control and treatment run it lists, and reports treatment minus control
per seed for a few averaged-convention metrics, in the same layout as the harness table.  The sd across seeds is model randomness only (events, labels and features are
identical across seeds); it cannot make a difference significant.

    python tools/book_study/run_bet_avg_pairs.py --tag volume_vs_14
    python tools/book_study/verify_bet_avg_pairs.py runtime/book_bet_avg/pairs_volume_vs_14

Output: runtime/book_bet_avg/pairs_<TAG>/<arm>_<seed>/ (one run_bet_averaging output folder each), pairs.csv (one row per run), pairs_diff.csv (one row per seed),
pairs_summary.md, pairs.json (settings and the run folders used).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import run_bet_averaging as RBA  # noqa: E402
import run_book_study as RB  # noqa: E402
import run_seed_pairs as SP  # noqa: E402

METRICS = ["avg_cpcv_meta_bp", "avg_cpcv_meta_sr_annual", "avg_cpcv_meta_minus_placebo_sr_daily", "avg_cpcv_meta_minus_primary_sr_daily", "avg_cpcv_meta_cost_bp",
           "avg_wf_meta_sr_annual", "avg_june_meta_bp", "avg_june_meta_minus_placebo_bp", "avg_selloff_meta_bp", "avg_selloff_meta_minus_placebo_bp"]
NAN = float("nan")


def metrics_of(S: dict) -> dict:
    c, w = S["cpcv"], S["cpcv"]["windows"]
    m = {"avg_cpcv_meta_bp": c["meta"]["mean_bp_per_symbol_day"], "avg_cpcv_meta_sr_annual": c["meta"]["sr_annual"],
         "avg_cpcv_meta_minus_placebo_sr_daily": c["meta_minus_placebo"]["mean_sr_daily_diff"], "avg_cpcv_meta_minus_primary_sr_daily": c["meta_minus_primary"]["mean_sr_daily_diff"],
         "avg_cpcv_meta_cost_bp": c["meta"]["cost_bp_per_symbol_day"], "avg_wf_meta_sr_annual": S["wf"]["meta"]["sr_annual"]}
    for wn, tag in (("june_2026_ai_correction", "june"), ("sell_off_post_holdout", "selloff")):
        a, b = w.get(f"meta|{wn}", {}).get("mean_bp_per_symbol_day", NAN), w.get(f"placebo|{wn}", {}).get("mean_bp_per_symbol_day", NAN)
        m[f"avg_{tag}_meta_bp"] = a
        m[f"avg_{tag}_meta_minus_placebo_bp"] = a - b
    return {k: (NAN if v is None else float(v)) for k, v in m.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="a harness folder runtime/book_seed_pairs/<TAG>")
    a = ap.parse_args()
    src = ROOT / "runtime" / "book_seed_pairs" / a.tag
    J = json.loads((src / "seed_pairs.json").read_text(encoding="utf-8"))
    out = ROOT / "runtime" / "book_bet_avg" / f"pairs_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    seeds, rows, by, used = J["seeds"], [], {}, []
    for r in J["runs"]:
        run = ROOT / r["run_dir"]
        o = RBA.run_one(run, out / f"{r['arm']}_{r['seed']}", None)
        S = json.loads((o / "averaged_summary.json").read_text(encoding="utf-8"))
        m = metrics_of(S)
        rows.append({"seed": r["seed"], "arm": r["arm"], **m})
        by[(r["seed"], r["arm"])] = m
        used.append({"arm": r["arm"], "seed": r["seed"], "run_dir": r["run_dir"], "out": str(o.relative_to(ROOT))})
        print(f"done {r['arm']} {r['seed']}", flush=True)
    rows.sort(key=lambda x: (x["seed"], x["arm"]))
    cols = ["seed", "arm"] + METRICS
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([repr(r[c]) if isinstance(r[c], float) else r[c] for c in cols])
    diffs = [{"seed": s, **{k: by[(s, "treat")][k] - by[(s, "ctrl")][k] for k in METRICS}} for s in seeds if (s, "ctrl") in by and (s, "treat") in by]
    with open(out / "pairs_diff.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + METRICS)
        for d in diffs:
            w.writerow([d["seed"]] + [repr(d[k]) for k in METRICS])

    def f(x, nd=3):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"

    st = J["settings"]
    L = ["# Ch.10 bet averaging over a seed-pair folder (read-only diagnostic)", "",
         f"Source: runtime/book_seed_pairs/{a.tag} (control {st['ctrl_script']}" + (f", reused from {st['ctrl_from']}" if st.get("ctrl_from") else "") + f"; treatment {st['treat_script']} {st.get('treat_args') or ''}).".replace(" )", ")"),
         "Averaged convention of bet_averaging.py (position = mean signal of the active bets per instrument and bar; bp per active instrument-day; cost on position changes). No model was fitted. "
         "Seeds change only forest randomness; the sd across seeds cannot make a difference significant.", "",
         f"## Treatment minus control, per metric ({len(diffs)} seeds)", "",
         "| metric | control mean (sd) | treatment mean (sd) | mean difference | sd of differences | mean / se | min | max | seeds with difference > 0 |", "|---|---|---|---|---|---|---|---|---|"]
    for k in METRICS:
        c = SP.arm_stats([by[(s, "ctrl")][k] for s in seeds if (s, "ctrl") in by])
        t = SP.arm_stats([by[(s, "treat")][k] for s in seeds if (s, "treat") in by])
        p = SP.paired_stats([d[k] for d in diffs])
        L.append(f"| {k} | {f(c['mean'])} ({f(c['sd'])}) | {f(t['mean'])} ({f(t['sd'])}) | {f(p['mean'])} | {f(p['sd'])} | {f(p['mean_over_se'], 2)} | {f(p['min'])} | {f(p['max'])} | {p['n_positive']} of {p['n']} |")
    L += ["", "## Per seed", "", "| seed | arm | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for r in rows:
        L.append(f"| {r['seed']} | {r['arm']} | " + " | ".join(f(r[k]) for k in METRICS) + " |")
    L.append("")
    (out / "pairs_summary.md").write_text("\n".join(L), encoding="utf-8")
    (out / "pairs.json").write_text(json.dumps({"tag": a.tag, "seeds": seeds, "runs": used, "metrics": METRICS,
                                                "script_md5": {n: RB.md5_file(HERE / n) for n in ("run_bet_avg_pairs.py", "run_bet_averaging.py", "bet_averaging.py", "run_seed_pairs.py")}}, indent=2), encoding="utf-8")
    print((out / "pairs_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
