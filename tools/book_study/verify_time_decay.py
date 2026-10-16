#!/usr/bin/env python
"""Independent checker for a run_meta_decay.py output folder (stdlib only; imports neither time_decay.py nor numpy).

It is verify_meta_venue.py (events, fracdiff, CPCV bookkeeping, predictions, daily results, paths, DSR, walk-forward, windows, SUMMARY cells; the 14-feature list) plus:
  * decay_extra.json must state the decay constant c given on the command line (the declaration is NOT read back from the run), this run's seed and prior trial count, and the md5 of
    time_decay.py and run_meta_decay.py next to this checker must equal the md5 recorded in the run;
  * the training set of every fit is rebuilt independently from events.csv (CPCV: events outside the test groups, not overlapping a test group's span in label time, not inside the
    embargo after it; walk-forward: labels ending before the cutoff) and must equal the event set of that call in decay_factors.csv, with exactly n_splits + 1 calls;
  * the decay factor of every event in every call is recomputed with plain loops from the definition in the docstring of time_decay.py (events ordered by start time, ties in file order;
    cumulative average uniqueness; linear in it; newest = 1) and compared with decay_factors.csv; the factors must be non-decreasing in start time, the newest factor 1, and for c >= 0 the
    smallest factor must be >= c.
Exit code 0 = all checks pass.

    python tools/book_study/verify_time_decay.py runtime/book_meta_kraken_decay/<timestamp> --decay-c 0.5 --prior-trials 105
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import verify_book_study as V
import verify_meta_venue as VM

HERE = Path(__file__).resolve().parent
DECLARED = {"c": None}


def factors_plain(u_list, t_list, c):
    """Decay factors by plain loops: order by (t, position), cumulative sum of u, linear in it (see time_decay.py docstring)."""
    n = len(u_list)
    order = sorted(range(n), key=lambda i: (t_list[i], i))
    cum, run = [0.0] * n, 0.0
    for r, i in enumerate(order):
        run += u_list[i]
        cum[r] = run
    total = cum[-1]
    out = [0.0] * n
    for r, i in enumerate(order):
        if c >= 0:
            out[i] = c + (1.0 - c) * cum[r] / total
        else:
            out[i] = max(0.0, 1.0 - (total - cum[r]) / ((1.0 + c) * total))
    return out


def extra_checks(d, meta, check):
    d = Path(d)
    c = DECLARED["c"]
    summ = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    params = summ["params"]
    p = d / "decay_extra.json"
    ok_file = p.exists()
    extra = json.loads(p.read_text(encoding="utf-8")) if ok_file else {}
    check("decay_extra.json states the decay constant declared on this checker's command line and this run's seed and prior trial count",
          ok_file and extra.get("decay_c") == c and extra.get("seed") == int(params["seed"]) and extra.get("prior_trials") == int(params["prior_trials"]) == VM.EXPECTED_PRIOR,
          str({k: extra.get(k) for k in ("decay_c", "seed", "prior_trials")}) + f" declared c {c}, expected prior {VM.EXPECTED_PRIOR}")
    changed = []
    for name, md5 in (extra.get("script_md5") or {}).items():
        q = HERE / name
        if not q.exists() or hashlib.md5(q.read_bytes()).hexdigest() != md5:
            changed.append(name)
    check("the decay scripts next to this checker (time_decay.py, run_meta_decay.py) are the ones that produced the run (md5 recorded in the run)",
          ok_file and sorted(extra.get("script_md5") or {}) == ["run_meta_decay.py", "time_decay.py"] and not changed, f"changed or missing: {changed}")

    ev = V.read(d / "events.csv")
    n_ev = len(ev)
    t0 = [int(float(e["t0"])) for e in ev]
    t1 = [int(float(e["t1"])) for e in ev]
    uu = [V.num(e["u"]) for e in ev]
    N, K, embargo_pct = int(params["n_groups"]), int(params["k"]), float(params["embargo_pct"])
    t_lo, t_hi = float(min(t0)), float(max(t0)) + 1.0
    edges = [t_lo + i * (t_hi - t_lo) / N for i in range(N + 1)]
    edges[-1] = t_hi
    grp = [min(max(bisect.bisect_right(edges, float(t)) - 1, 0), N - 1) for t in t0]
    embargo = embargo_pct * (t_hi - t_lo)
    combos = list(combinations(range(N), K))
    trains = []
    for tg in combos:
        tr = []
        for i in range(n_ev):
            if grp[i] in tg:
                continue
            ok = True
            for g in tg:
                lo, hi = edges[g], edges[g + 1]
                if (t0[i] <= hi and t1[i] >= lo) or (t0[i] > hi and t0[i] <= hi + embargo):
                    ok = False
                    break
            if ok:
                tr.append(i)
        trains.append(tr)
    cut = float(V.ts_key(params["cutoff"]))
    trains.append([i for i in range(n_ev) if t1[i] < cut])

    rows = V.read(d / "decay_factors.csv") if (d / "decay_factors.csv").exists() else []
    by_call = defaultdict(list)
    for r in rows:
        by_call[int(r["call"])].append((int(r["event"]), float(r["factor"])))
    check(f"decay_factors.csv has exactly {len(trains)} fit calls (the {len(combos)} CPCV splits in order, then the walk-forward fit) and the run recorded the same number",
          sorted(by_call) == list(range(len(trains))) and extra.get("n_fit_calls") == len(trains), f"calls {sorted(by_call)[:20]}, recorded {extra.get('n_fit_calls')}")
    bad_set, bad_val, bad_prop = [], [], []
    for k, tr in enumerate(trains):
        got = by_call.get(k, [])
        if [e_ for e_, _ in got] != tr and sorted(e_ for e_, _ in got) != tr:
            bad_set.append((k, len(got), len(tr)))
            continue
        gmap = dict(got)
        exp = factors_plain([uu[i] for i in tr], [t0[i] for i in tr], c)
        worst = max((abs(gmap[i] - e_) for i, e_ in zip(tr, exp)), default=0.0)
        if worst > 1e-9:
            bad_val.append((k, worst))
        seq = sorted(tr, key=lambda i: (t0[i], i))
        f_seq = [gmap[i] for i in seq]
        if any(b < a - 1e-12 for a, b in zip(f_seq, f_seq[1:])) or abs(f_seq[-1] - 1.0) > 1e-12 or (c >= 0 and min(f_seq) < c - 1e-12) or min(f_seq) < 0.0 or max(f_seq) > 1.0 + 1e-12:
            bad_prop.append(k)
    check("each fit's training events (CPCV purge + embargo / walk-forward cut, rebuilt from events.csv) equal the events listed for that call in decay_factors.csv", not bad_set, str(bad_set[:3]))
    check("every decay factor of every call equals the plain-loop recomputation from the definition (events by start time, cumulative average uniqueness, linear, newest = 1) within 1e-9", not bad_val and not bad_set, str(bad_val[:3]))
    check("in every call the factors do not decrease with event start time, the newest is 1, none is below 0 or above 1, and for c >= 0 none is below c", not bad_prop and not bad_set, str(bad_prop[:3]))


VM.EXTRA_CHECKS = extra_checks


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    for flag, cast in (("--decay-c", float), ("--prior-trials", int)):
        if flag not in argv:
            raise SystemExit(f"{flag} is required")
        i_ = argv.index(flag)
        val = cast(argv[i_ + 1])
        del argv[i_:i_ + 2]
        if flag == "--decay-c":
            DECLARED["c"] = val
        else:
            VM.EXPECTED_PRIOR = val
    if not (-1.0 < DECLARED["c"] <= 1.0):
        raise SystemExit("--decay-c must satisfy -1 < c <= 1")
    try:
        raise SystemExit(VM.main(argv[0]))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
