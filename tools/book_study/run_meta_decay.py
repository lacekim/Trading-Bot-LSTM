#!/usr/bin/env python
"""Meta-labeling pipeline on ONE venue's hourly candles with the Ch.4 time-decay factor applied to the training weights (see time_decay.py).

It runs run_meta_venue.main() unchanged (14 features, same events, same labels), after swapping in a wrapper around meta_eval.fit_forest that multiplies the
training set's sample weights by the decay factors of THAT training set (CPCV splits and the walk-forward fit).  MDA / SFI keep the ordinary weights (they are
reported only, not part of the scores).  No model, threshold, calibration file or execution code is touched; output goes to the --out folder only.

    python tools/book_study/run_meta_decay.py --decay-c 0.5 --venue kraken --candle-dir runtime/venue_candles_v2/kraken_1h --out runtime/book_meta_kraken_decay --sell-off 2026-09-27 2026-10-08
    python tools/book_study/verify_meta_decay.py "$(ls -d runtime/book_meta_kraken_decay/*/ | tail -1)" --decay-c 0.5

Extra files in the run folder: decay_factors.csv (call, event, factor: the factor used for every event of every training set; call 0..n_splits-1 = CPCV splits in order,
call n_splits = the walk-forward fit) and decay_extra.json (c, number of calls, seed, prior trials, md5 of the scripts).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import run_meta_venue as RV          # sets cwd / sys.path exactly like the plain venue run

import meta_eval as ME
import run_book_study as RB
import time_decay as TD

HERE = Path(__file__).resolve().parent
SEED = 20261017          # default only; the harness always passes --seed
PRIOR_TRIALS = 105       # default only; the harness passes --prior-trials. Earlier looks: 101 before the first Kraken run, that run (102), Alpaca equities (103), Kraken refresh (104), volume features (105).

_ORIG_DEVIATIONS = RV.venue_deviations
_ORIG_FIT = ME.fit_forest
WEIGHT_PREFIX = "Sample weights = absolute return attribution"


def make_deviations(c: float):
    def deviations(venue: str, prior_trials: int):
        out = _ORIG_DEVIATIONS(venue, prior_trials)
        hit = [i for i, t in enumerate(out) if t.startswith(WEIGHT_PREFIX)]
        dsr = [i for i, t in enumerate(out) if t.startswith("DSR trial count:")]
        if len(hit) != 1 or len(dsr) != 1:
            raise SystemExit("deviation entries to replace were not found exactly once")
        out[hit[0]] = (f"Sample weights = absolute return attribution (Snippet 4.10), scaled to mean 1 over all events, multiplied by a time-decay factor with c = {c} (Ch.4, the idea of Snippet 4.11, "
                       "written from its definition in time_decay.py; the oldest event of a training set gets about c, the newest 1, linear in cumulative average uniqueness). Declared before any result. "
                       "Differences from the book: the factors are computed per training set on the pooled events of all instruments ordered by event start time, so 'newest' means newest within "
                       "that training set; in combinatorial purged CV a training set can contain events later than the test group, so the decay there measures age inside the training set, "
                       "not distance to the test period; MDA / SFI keep the ordinary weights; the factors are not rescaled after multiplying (the learner uses only the ratios of the weights, but floating-point rounding can decide mathematically tied splits differently after any rescaling that is not a power of two).")
        out[dsr[0]] = (f"DSR trial count: there is one model here, but earlier diagnostics in this research (GMX / TradingView candles, then Kraken and Alpaca candles) looked at many momentum "
                       f"lookbacks, holds, states and feature sets; the DSR is shown for several trial counts, including the one that counts those earlier looks ({prior_trials}, this run "
                       f"being look {prior_trials + 1}). The 30-day momentum side was chosen on those earlier candles.")
        return out
    return deviations


def pop_decay_c(argv: list) -> float:
    if "--decay-c" not in argv:
        raise SystemExit("--decay-c is required (declared before any result)")
    i = argv.index("--decay-c")
    try:
        c = float(argv[i + 1])
    except (IndexError, ValueError):
        raise SystemExit("--decay-c needs a number")
    del argv[i:i + 2]
    if not (-1.0 < c <= 1.0):
        raise SystemExit("--decay-c must satisfy -1 < c <= 1")
    return c


def make_fit_wrapper(c: float, orig_fit, calls: list):
    """A drop-in for meta_eval.fit_forest that multiplies the training events' weights by their decay factors (computed on THIS training set) and records them in calls."""
    def fit_forest_decay(idx, data, forest_kw, seed):
        X, y, w, sym, starts, ends, t0, t1, u = data
        d = TD.decay_factors(u[idx], t0[idx], c)
        w2 = w.copy()
        w2[idx] = w[idx] * d
        calls.append((len(calls), np.asarray(idx, dtype=np.int64).copy(), d))
        return orig_fit(idx, (X, y, w2, sym, starts, ends, t0, t1, u), forest_kw, seed)
    return fit_forest_decay


def main() -> int:
    c = pop_decay_c(sys.argv)
    calls = []                                   # (call number, training indices, factors)
    fit_forest_decay = make_fit_wrapper(c, _ORIG_FIT, calls)

    ME.fit_forest = fit_forest_decay
    RV.SEED, RV.PRIOR_TRIALS = SEED, PRIOR_TRIALS
    RV.venue_deviations = make_deviations(c)
    try:
        rc = RV.main()
    finally:
        ME.fit_forest = _ORIG_FIT
    if "--out" not in sys.argv:
        raise SystemExit("--out is required")
    out_arg = sys.argv[sys.argv.index("--out") + 1]
    out_root = RV.ROOT / out_arg
    run_dir = sorted(p for p in out_root.iterdir() if p.is_dir())[-1]
    pd.DataFrame({"call": np.concatenate([np.full(len(i), k, dtype=np.int64) for k, i, _ in calls]),
                  "event": np.concatenate([i for _, i, _ in calls]),
                  "factor": np.concatenate([f for _, _, f in calls])}).to_csv(run_dir / "decay_factors.csv", index=False, float_format="%.17g")
    summ = json.loads((run_dir / "summary_numbers.json").read_text(encoding="utf-8"))
    extra = {"decay_c": c, "n_fit_calls": len(calls), "seed": int(summ["params"]["seed"]), "prior_trials": int(summ["params"]["prior_trials"]),
             "script_md5": {n: RB.md5_file(HERE / n) for n in ("time_decay.py", "run_meta_decay.py")}}
    (run_dir / "decay_extra.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
