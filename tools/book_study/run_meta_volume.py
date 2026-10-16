#!/usr/bin/env python
"""Meta-labeling pipeline on ONE venue's hourly candles with four volume-based microstructure features added (Ch.19 bar-level proxies; see volume_features.py).

It runs run_meta_venue.main() unchanged, after swapping in (1) the 18-feature list, (2) an event builder that calls meta_events.build_meta_events() unchanged and
adds the volume columns (meta_events_vol.py), (3) this run's seed and prior trial count, (4) deviation texts that describe the volume features.
No model, threshold, calibration file or execution code is touched; output goes to the --out folder only.

Run in the `javier` conda env, from the project root:
    python tools/book_study/run_meta_volume.py --venue kraken --candle-dir runtime/venue_candles_v2/kraken_1h --out runtime/book_meta_kraken_vol --sell-off 2026-09-27 2026-10-08
    python tools/book_study/verify_meta_volume.py "$(ls -d runtime/book_meta_kraken_vol/*/ | tail -1)"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_meta_venue as RV          # sets cwd / sys.path exactly like the plain venue run

import meta_events as E
import meta_events_vol as EV
import run_book_study as RB
import venue_loader as VL
import volume_features as VF

HERE = Path(__file__).resolve().parent
SEED = 20261017          # fixed before any result was seen
PRIOR_TRIALS = 104       # earlier looks: 101 before the first Kraken run, that run (102), the Alpaca equities run (103), the Kraken refresh run (104). This run is look 105.

_ORIG_DEVIATIONS = RV.venue_deviations        # captured before main() replaces it
NOT_BUILT_PREFIX = "Not built in this run: Kyle, Amihud and Hasbrouck lambdas"


def deviations(venue: str, prior_trials: int):
    out = _ORIG_DEVIATIONS(venue, prior_trials)
    hit = [i for i, t in enumerate(out) if t.startswith(NOT_BUILT_PREFIX)]
    dsr = [i for i, t in enumerate(out) if t.startswith("DSR trial count:")]
    if len(hit) != 1 or len(dsr) != 1:
        raise SystemExit("deviation entries to replace were not found exactly once")
    out[hit[0]] = ("Volume features (Ch.19), declared before any result: kyle_lam, amihud_lam, vpin, dvol_z (definitions in volume_features.py), added to the 14 other features. "
                   "The book computes Kyle's and Amihud's lambdas and VPIN from tick / trade data and volume buckets; here only hourly bars exist, so the tick rule uses bar closes, "
                   "buy volume comes from bulk volume classification on bar returns, buckets are time bars, and the lambdas are rolling 168-bar OLS slopes (with an intercept; whether "
                   "the book includes one was not confirmed) times the mean hourly dollar volume. Hasbrouck's lambda is not built (needs trade-level signed flow). "
                   "Not built: PIN, volume-based structural-break features, HRP allocation (Ch.16), the Chapter 20-22 material. A definition difference in volume between the older "
                   "candles and the trade-rebuilt refresh candles was not part of this run.")
    out[dsr[0]] = (f"DSR trial count: there is one model here, but earlier diagnostics in this research (GMX / TradingView candles, then Kraken and Alpaca candles) looked at many momentum "
                   f"lookbacks, holds, states and feature sets; the DSR is shown for several trial counts, including the one that counts those earlier looks ({prior_trials}, this run "
                   f"being look {prior_trials + 1}). The 30-day momentum side was chosen on those earlier candles.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--candle-dir")
    ap.add_argument("--out")
    known, _rest = ap.parse_known_args()
    if not known.candle_dir or not known.out:
        raise SystemExit("--candle-dir and --out are required")
    candle_dir = Path(known.candle_dir)
    if not candle_dir.is_absolute():
        candle_dir = RV.ROOT / candle_dir

    E.FEATURES = EV.FEATURES_18
    orig_build = E.build_meta_events

    def build_with_volume(closes, ohlc, h=100, ptsl=(2.0, 2.0), cost_frac=0.0024, signal=E.SIDE_SIGNAL, log=print):
        vols = {s: VF.load_volumes(candle_dir, s) for s in closes}
        return EV.build_meta_events_vol(closes, ohlc, vols, h=h, ptsl=ptsl, cost_frac=cost_frac, log=log)

    E.build_meta_events = build_with_volume
    RV.SEED, RV.PRIOR_TRIALS = SEED, PRIOR_TRIALS
    RV.venue_deviations = deviations
    try:
        rc = RV.main()
    finally:
        E.build_meta_events = orig_build
    out_root = RV.ROOT / known.out
    run_dir = sorted(p for p in out_root.iterdir() if p.is_dir())[-1]
    extra = {"volume_features": VF.VOL_FEATURES, "windows": {"W_REG": VF.W_REG, "W_FLOW": VF.W_FLOW, "W_Z": VF.W_Z}, "seed": SEED, "prior_trials": PRIOR_TRIALS,
             "script_md5": {n: RB.md5_file(HERE / n) for n in ("volume_features.py", "meta_events_vol.py", "run_meta_volume.py")}}
    (run_dir / "volume_extra.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
