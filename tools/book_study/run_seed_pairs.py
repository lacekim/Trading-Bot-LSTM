#!/usr/bin/env python
"""Paired multi-seed comparison of two pipeline variants (READ-ONLY diagnostic; starts no process other than the study scripts and their checkers).

Why: one run of the forest pipeline depends on its seed (bootstrap draws, placebo shuffle).  A feature set that looks better in one seed pair may only be
seed noise.  This harness runs a CONTROL script and a TREATMENT script on the same candles for each of several fixed seeds, runs each run's own checker, and
reports the per-seed differences (treatment - control) with their spread.  Events, labels and features are identical across seeds, so the spread measures
the model's randomness only, NOT the sampling uncertainty of the data; it cannot make a result significant, it can only show a difference is not stable.

Defaults reproduce the volume-feature question on the Kraken v2 candles: control = run_meta_venue.py (14 features), treatment = run_meta_volume.py (18 features).
Seeds are fixed before any result: 20261017 (already run once, so this also checks reproducibility), 20261101, 20261102, 20261103, 20261104.

Optional (all off by default, so the default behaviour above is unchanged):
  --ctrl-from TAG            reuse the control runs of an earlier harness folder runtime/book_seed_pairs/TAG/ctrl_<seed> (never re-run; their checker is re-run) instead of running the control again
  --treat-prior-trials N     prior trial count of the treatment runs when it differs from --prior-trials (which stays the control's), e.g. because the treatment is a later look
  --treat-args "..."         extra arguments for the treatment script (for example "--decay-c 0.5"); --treat-check-args "..." the same for its checker
  --same-features            the treatment uses the same feature list as the control (default: it must extend it)
  --dsr-n N                  read the DSR metric at this trial count N for BOTH arms (default: each run's own prior trials + 1), so the two arms are deflated alike

Run in the `javier` conda env from the project root (about 6 minutes per run on the Mac; --jobs 2 runs two at a time):
    python tools/book_study/run_seed_pairs.py --tag volume_vs_14 --jobs 2
    python tools/book_study/verify_seed_pairs.py runtime/book_seed_pairs/volume_vs_14

Output (runtime/book_seed_pairs/<tag>/): <arm>_<seed>/<timestamp>/ (the full run), logs/, seed_pairs.csv (one row per run), seed_pairs_diff.csv (one row per seed),
seed_pairs_summary.md, seed_pairs.json (settings, run folders, checker exit codes, script md5).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

DEFAULT_SEEDS = [20261017, 20261101, 20261102, 20261103, 20261104]
METRICS = [
    "cpcv_auc", "cpcv_meta_sr_annual", "cpcv_meta_bp", "cpcv_meta_minus_placebo_sr_daily",
    "wf_auc", "wf_meta_sr_annual",
    "june_cpcv_meta_bp", "june_cpcv_meta_minus_placebo_bp", "selloff_cpcv_meta_bp", "selloff_cpcv_meta_minus_placebo_bp",
    "dsr_prior_plus_one",
]
NAN = float("nan")


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return NAN
        d = d[k]
    return float(d) if d is not None else NAN


def extract_metrics(s: dict, dsr_n: int | None = None) -> dict:
    """The headline numbers of one run from its summary_numbers.json (NaN where the run has none, e.g. a window with no events).
    The DSR metric is read at the run's own prior trials + 1, or at dsr_n when that is given."""
    ev, pm = s["eval"], int(s["params"]["prior_trials"])
    m = {
        "cpcv_auc": _g(s, "cpcv", "auc_mean"),
        "cpcv_meta_sr_annual": _g(ev, "summary", "meta", "mean_sr_annual"),
        "cpcv_meta_bp": _g(ev, "summary", "meta", "mean_bp_per_candidate"),
        "cpcv_meta_minus_placebo_sr_daily": _g(ev, "summary", "meta_minus_placebo", "mean_sr_daily_diff"),
        "wf_auc": _g(s, "walk_forward", "auc"),
        "wf_meta_sr_annual": _g(s, "walk_forward_eval", "summary", "meta", "mean_sr_annual"),
        "dsr_prior_plus_one": _g(ev, "dsr", "by_trials", str(pm + 1 if dsr_n is None else int(dsr_n)), "dsr"),
    }
    for w, tag in (("june_2026_ai_correction", "june"), ("sell_off_post_holdout", "selloff")):
        a = _g(ev, "windows", f"meta|{w}", "mean_bp_per_candidate")
        b = _g(ev, "windows", f"placebo|{w}", "mean_bp_per_candidate")
        m[f"{tag}_cpcv_meta_bp"] = a
        m[f"{tag}_cpcv_meta_minus_placebo_bp"] = a - b
    return m


def paired_stats(diffs: list) -> dict:
    """Mean, sample sd (ddof 1), standard error, mean / se, min, max and the number of positive differences of one metric over the seeds (NaN-free values only)."""
    d = [x for x in diffs if not math.isnan(x)]
    n = len(d)
    mean = statistics.fmean(d) if n else NAN
    sd = statistics.stdev(d) if n > 1 else NAN
    se = sd / math.sqrt(n) if n > 1 else NAN
    return {"n": n, "mean": mean, "sd": sd, "se": se, "mean_over_se": (mean / se) if (n > 1 and se > 0) else NAN,
            "min": min(d) if n else NAN, "max": max(d) if n else NAN, "n_positive": sum(1 for x in d if x > 0)}


def arm_stats(vals: list) -> dict:
    v = [x for x in vals if not math.isnan(x)]
    return {"n": len(v), "mean": statistics.fmean(v) if v else NAN, "sd": statistics.stdev(v) if len(v) > 1 else NAN}


def md5_file(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def find_run_dir(arm_dir: Path) -> Path | None:
    cand = sorted(p for p in arm_dir.iterdir() if p.is_dir() and (p / "summary_numbers.json").exists()) if arm_dir.exists() else []
    return cand[-1] if cand else None


def prior_for(a, arm: str) -> int:
    return int(a.treat_prior_trials) if (arm == "treat" and a.treat_prior_trials is not None) else int(a.prior_trials)


def run_cmd(a, script: str, seed: int, out_rel: str, arm: str = "ctrl") -> list:
    cmd = [sys.executable, str(HERE / script), "--venue", a.venue, "--candle-dir", a.candle_dir, "--out", out_rel, "--prior-trials", str(prior_for(a, arm)), "--seed", str(seed)]
    if a.sell_off:
        cmd += ["--sell-off", a.sell_off[0], a.sell_off[1]]
    if a.quick:
        cmd += ["--quick"]
    if arm == "treat" and a.treat_args:
        cmd += shlex.split(a.treat_args)
    return cmd


def check_cmd(a, arm: str, run_dir: Path) -> list:
    script = a.ctrl_checker if arm == "ctrl" else a.treat_checker
    cmd = [sys.executable, str(HERE / script), str(run_dir), "--prior-trials", str(prior_for(a, arm))]
    if arm == "treat" and a.treat_check_args:
        cmd += shlex.split(a.treat_check_args)
    return cmd


def execute(a, arm: str, seed: int, base: Path) -> dict:
    script = a.ctrl_script if arm == "ctrl" else a.treat_script
    reuse = arm == "ctrl" and bool(a.ctrl_from)
    arm_dir = (ROOT / "runtime" / "book_seed_pairs" / a.ctrl_from / f"ctrl_{seed}") if reuse else base / f"{arm}_{seed}"
    out_rel = str(arm_dir.relative_to(ROOT))
    (base / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_dir = find_run_dir(arm_dir)
    if run_dir is None and (a.aggregate_only or reuse):
        return {"arm": arm, "seed": seed, "run_dir": None, "run_exit": None, "check_exit": None}
    if run_dir is None:
        with open(base / "logs" / f"{arm}_{seed}.log", "w", encoding="utf-8") as lf:
            rc = subprocess.run(run_cmd(a, script, seed, out_rel, arm), cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode
        run_dir = find_run_dir(arm_dir)
        if rc != 0 or run_dir is None:
            return {"arm": arm, "seed": seed, "run_dir": None, "run_exit": rc, "check_exit": None}
    else:
        rc = 0
    with open(base / "logs" / f"{arm}_{seed}.check.log", "w", encoding="utf-8") as lf:
        crc = subprocess.run(check_cmd(a, arm, run_dir), cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {arm} seed {seed}: run exit {rc}, checker exit {crc} ({time.time() - t0:.0f}s)", flush=True)
    return {"arm": arm, "seed": seed, "run_dir": str(run_dir.relative_to(ROOT)), "run_exit": rc, "check_exit": crc}


def collect(base: Path, records: list, a_dsr_n: int | None = None) -> tuple:
    rows, by = [], {}
    for r in records:
        if r["run_dir"] is None:
            continue
        s = json.loads((ROOT / r["run_dir"] / "summary_numbers.json").read_text(encoding="utf-8"))
        m = extract_metrics(s, a_dsr_n)
        rows.append({"seed": r["seed"], "arm": r["arm"], "n_features": len(s["params"]["features"]), "events": int(s["meta"]["events"]), **m})
        by[(r["seed"], r["arm"])] = m
    return rows, by


def write_outputs(base: Path, a_settings: dict, records: list, seeds: list) -> int:
    rows, by = collect(base, records, a_settings.get("dsr_n"))
    rows.sort(key=lambda r: (r["seed"], r["arm"]))
    cols = ["seed", "arm", "n_features", "events"] + METRICS
    with open(base / "seed_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([repr(r[c]) if isinstance(r[c], float) else r[c] for c in cols])
    diffs = []
    for s in seeds:
        if (s, "ctrl") in by and (s, "treat") in by:
            diffs.append({"seed": s, **{k: by[(s, "treat")][k] - by[(s, "ctrl")][k] for k in METRICS}})
    with open(base / "seed_pairs_diff.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + METRICS)
        for d in diffs:
            w.writerow([d["seed"]] + [repr(d[k]) for k in METRICS])
    L = ["# Paired multi-seed comparison (read-only diagnostic)", "",
         f"Control: {a_settings['ctrl_script']}" + (f" (runs reused from {a_settings['ctrl_from']})" if a_settings.get("ctrl_from") else "")
         + f". Treatment: {a_settings['treat_script']}" + (f" {a_settings['treat_args']}" if a_settings.get("treat_args") else "")
         + f". Venue {a_settings['venue']}, candles {a_settings['candle_dir']}, prior trials {a_settings['prior_trials']}"
         + (f" (treatment {a_settings['treat_prior_trials']})" if a_settings.get("treat_prior_trials") is not None else "")
         + f", seeds {', '.join(map(str, seeds))}. DSR metric (dsr_prior_plus_one): "
         + (f"read at N = {a_settings['dsr_n']} for both arms." if a_settings.get("dsr_n") is not None else "read at each run's own prior trials + 1."),
         "Seeds change only the forest's bootstrap draws and the placebo shuffle; events, labels and features are the same in every run. The spread below is therefore model randomness, "
         "not the sampling uncertainty of the data, and cannot make a difference significant; a difference that changes sign across seeds is not stable.", "",
         "All run checkers passed: " + ("yes" if all(r["check_exit"] == 0 for r in records) and len(records) == 2 * len(seeds) else "NO - see seed_pairs.json"), "",
         f"## Treatment minus control, per metric ({len(diffs)} seeds)", "",
         "| metric | control mean (sd) | treatment mean (sd) | mean difference | sd of differences | mean / se | min | max | seeds with difference > 0 |", "|---|---|---|---|---|---|---|---|---|"]

    def f(x, nd=4):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"

    for k in METRICS:
        c = arm_stats([by[(s, "ctrl")][k] for s in seeds if (s, "ctrl") in by])
        t = arm_stats([by[(s, "treat")][k] for s in seeds if (s, "treat") in by])
        p = paired_stats([d[k] for d in diffs])
        L.append(f"| {k} | {f(c['mean'])} ({f(c['sd'])}) | {f(t['mean'])} ({f(t['sd'])}) | {f(p['mean'])} | {f(p['sd'])} | {f(p['mean_over_se'], 2)} | {f(p['min'])} | {f(p['max'])} | {p['n_positive']} of {p['n']} |")
    L += ["", "## Per seed", "", "| seed | arm | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for r in rows:
        L.append(f"| {r['seed']} | {r['arm']} | " + " | ".join(f(r[k]) for k in METRICS) + " |")
    L.append("")
    (base / "seed_pairs_summary.md").write_text("\n".join(L), encoding="utf-8")
    scripts = {n: md5_file(HERE / n) for n in sorted({a_settings["ctrl_script"], a_settings["treat_script"], a_settings["ctrl_checker"], a_settings["treat_checker"], "run_seed_pairs.py"})}
    (base / "seed_pairs.json").write_text(json.dumps({"settings": a_settings, "seeds": seeds, "runs": records, "metrics": METRICS, "script_md5": scripts}, indent=2), encoding="utf-8")
    return 0 if all(r["check_exit"] == 0 for r in records) and len(records) == 2 * len(seeds) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--venue", default="kraken")
    ap.add_argument("--candle-dir", default="runtime/venue_candles_v2/kraken_1h")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--prior-trials", type=int, default=104)
    ap.add_argument("--sell-off", nargs=2, default=["2026-09-27", "2026-10-08"], metavar=("START", "END"))
    ap.add_argument("--ctrl-script", default="run_meta_venue.py")
    ap.add_argument("--treat-script", default="run_meta_volume.py")
    ap.add_argument("--ctrl-checker", default="verify_meta_venue.py")
    ap.add_argument("--treat-checker", default="verify_meta_volume.py")
    ap.add_argument("--ctrl-from", default=None, metavar="TAG", help="reuse runtime/book_seed_pairs/TAG/ctrl_<seed> as the control runs")
    ap.add_argument("--treat-prior-trials", type=int, default=None, help="prior trial count of the treatment runs (default: --prior-trials)")
    ap.add_argument("--treat-args", default="", help='extra arguments for the treatment script, e.g. "--decay-c 0.5"')
    ap.add_argument("--treat-check-args", default="", help="extra arguments for the treatment checker")
    ap.add_argument("--same-features", action="store_true", help="the treatment uses the same feature list as the control (default: it must extend it)")
    ap.add_argument("--dsr-n", type=int, default=None, help="read the DSR metric at this trial count for both arms (default: own prior + 1)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="smoke test only (few trees, 4 groups); results are not for reporting")
    ap.add_argument("--aggregate-only", action="store_true", help="do not run anything; rebuild the tables from run folders already present (checker exit codes are re-run)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        raise SystemExit("seeds must be distinct")
    base = ROOT / "runtime" / "book_seed_pairs" / a.tag
    jobs = [(arm, s) for s in a.seeds for arm in ("ctrl", "treat")]
    if a.dry_run:
        for arm, s in jobs:
            script = a.ctrl_script if arm == "ctrl" else a.treat_script
            if arm == "ctrl" and a.ctrl_from:
                print(f"(control reused, not run) runtime/book_seed_pairs/{a.ctrl_from}/ctrl_{s}")
                continue
            print(" ".join(run_cmd(a, script, s, str((base / f"{arm}_{s}").relative_to(ROOT)), arm)))
        return 0
    base.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(a, k) for k in ("venue", "candle_dir", "prior_trials", "sell_off", "ctrl_script", "treat_script", "ctrl_checker", "treat_checker", "quick")}
    settings.update({"ctrl_from": a.ctrl_from, "treat_prior_trials": a.treat_prior_trials, "treat_args": a.treat_args, "treat_check_args": a.treat_check_args,
                     "features_relation": "same" if a.same_features else "extends", "dsr_n": a.dsr_n})
    settings["candle_dir"] = a.candle_dir
    print(f"{len(jobs)} runs, {a.jobs} at a time; output {base}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        records = list(ex.map(lambda j: execute(a, j[0], j[1], base), jobs))
    rc = write_outputs(base, settings, records, a.seeds)
    print((base / "seed_pairs_summary.md").read_text(encoding="utf-8"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
