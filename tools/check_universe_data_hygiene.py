"""Data-hygiene gate, run BEFORE exit-logic research, prompted by the
out-of-sample validation run reporting a full-history span starting
1962-01-02 -- implausible for this "robust non-crypto instruments" equities/
commodities/bonds universe, and almost certainly one symbol with either a
genuinely deep-history series (some commodity/bond continuous futures can
predate 1990) or corrupted/placeholder data.

This matters beyond the current OOS window specifically because breadth
(compute_breadth_series), sizing calibration (compute_calibration_before),
and portfolio capital availability are all computed from the FULL history,
shared across every candidate and every future exit variant -- a bad series
could quietly distort any of them even where the OOS-window numbers already
reported look sane.

Three questions, answered empirically rather than asserted:
  1. Which symbol(s) have implausibly early or otherwise anomalous data
     (flat/stale runs, extreme single-day moves, zero/negative prices,
     large coverage gaps)?
  2. Does removing the flagged symbol(s) change golden-cross BREADTH or the
     sizing CALIBRATION medians that the OOS validation and every sizing
     variant depend on -- and does any such change reach into the recent /
     OOS-relevant window, or is it confined to the implausible early period?
  3. What does the flagged symbol's OWN isolated trade history look like
     (holding periods, entry/exit prices) -- anything suggesting it held a
     position, and therefore competed for capital, based on corrupted data?

FROZEN THRESHOLDS (disclosed up front, not tuned after seeing results):
  SUSPICIOUS_START_DATE = 1990-01-01 -- any symbol with data before this is
    flagged for inspection (generous: even legitimate long-history
    commodity/bond series in this universe would be unusual before this).
  BREADTH_MATERIAL_DIFF = 0.02 (2 percentage points) -- max abs difference
    between breadth-with-flagged and breadth-without-flagged, in the
    RECENT window (2015-01-01 onward, which covers the OOS window and its
    pre-split calibration period), above which the contamination is judged
    material.
  CALIBRATION_MATERIAL_REL_DIFF = 0.05 (5% relative) -- relative change in
    the median_atr_pct / median_vol20 calibration inputs (computed exactly
    as compute_calibration_before does, before OOS_SPLIT_DATE) from
    removing the flagged symbol(s), above which it's judged material.

This script does not auto-exclude anything. It reports the empirical
answers to the three questions above and a mechanical SAFE-TO-PROCEED /
NEEDS-EXCLUSION verdict per the thresholds, and -- only if the verdict is
NEEDS-EXCLUSION -- writes the flagged symbol list to
runtime/universe_data_hygiene_flags.json so a subsequent script can read it
and exclude those symbols, rather than silently baking an exclusion
decision into this gate itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tools.run_sizing_allocation_backtest_equities import (
    ENTRY_COST, EXIT_COST, STARTING_CAPITAL, BASELINE_STOP_PCT,
    ENTRY_SETS, build_states, make_sizing_fn,
)
from tools.run_sizing_oos_validation_equities import (
    OOS_SPLIT_DATE, compute_calibration_before, compute_breadth_series, run_backtest_instrumented,
)
from tools.run_stoploss_tail_risk_test_equities import _load_equities_universe
from tools.run_pullback_entry_backtest_equities import fetch_ohlcv_batch

OUT_FLAGS_PATH = Path(__file__).resolve().parents[1] / "runtime" / "universe_data_hygiene_flags.json"

SUSPICIOUS_START_DATE = pd.Timestamp("1990-01-01")
RECENT_WINDOW_START = pd.Timestamp("2015-01-01")
BREADTH_MATERIAL_DIFF = 0.02
CALIBRATION_MATERIAL_REL_DIFF = 0.05
LONG_HOLD_TRADING_DAYS = 1500  # flag any single trade held longer than this


def per_symbol_stats(df: pd.DataFrame) -> dict:
    close = df["close"]
    first_date, last_date = df.index.min(), df.index.max()
    n_rows = len(df)
    expected_bdays = len(pd.bdate_range(first_date, last_date))
    coverage_ratio = n_rows / expected_bdays if expected_bdays else float("nan")

    daily_ret = close.pct_change().dropna()
    max_abs_ret_pct = float(daily_ret.abs().max() * 100) if len(daily_ret) else float("nan")

    is_flat = close.diff().fillna(1) == 0
    longest_flat_run = 0
    cur_run = 0
    for flat in is_flat:
        cur_run = cur_run + 1 if flat else 0
        longest_flat_run = max(longest_flat_run, cur_run)

    n_zero_or_neg = int((close <= 0).sum())
    n_nan = int(df.isna().sum().sum())
    n_dupe_index = int(df.index.duplicated().sum())

    return {
        "first_date": str(first_date.date()), "last_date": str(last_date.date()),
        "n_rows": n_rows, "expected_bdays": expected_bdays, "coverage_ratio": round(coverage_ratio, 3),
        "max_abs_daily_ret_pct": round(max_abs_ret_pct, 1), "longest_flat_run_days": longest_flat_run,
        "n_zero_or_negative_close": n_zero_or_neg, "n_nan_cells": n_nan, "n_duplicate_index_entries": n_dupe_index,
    }


def isolated_trade_history(states: dict, sym: str, atr_calib: float, vol_calib: float) -> list[dict]:
    single = {sym: states[sym]}
    sizing_fn = make_sizing_fn("EQUAL-DOLLAR", atr_calib, vol_calib)
    trades, _, _ = run_backtest_instrumented(single, ENTRY_SETS["BASELINE"], sizing_fn)
    if trades.empty:
        return []
    trades = trades.copy()
    trades["holding_days"] = (trades["exit_time"] - trades["entry_time"]).dt.days
    out = []
    for _, row in trades.iterrows():
        out.append({
            "entry_time": str(row["entry_time"].date()), "exit_time": str(row["exit_time"].date()),
            "holding_calendar_days": int(row["holding_days"]), "ret_pct": round(float(row["ret_pct"]), 2),
            "exit_reason": row["exit_reason"],
            "flag_long_hold": bool(row["holding_days"] > LONG_HOLD_TRADING_DAYS),
        })
    return out


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    stats = {sym: per_symbol_stats(df) for sym, df in price.items()}
    by_first_date = sorted(stats.items(), key=lambda kv: kv[1]["first_date"])

    print("=" * 90)
    print(f"10 DEEPEST-HISTORY SYMBOLS (sorted by first_date ascending)")
    print("=" * 90)
    for sym, s in by_first_date[:10]:
        print(f"  {sym:<8} first={s['first_date']}  last={s['last_date']}  n_rows={s['n_rows']:>6}  "
              f"coverage={s['coverage_ratio']:.2f}  max_abs_ret={s['max_abs_daily_ret_pct']:>6.1f}%  "
              f"longest_flat_run={s['longest_flat_run_days']:>4}d  n_zero/neg={s['n_zero_or_negative_close']}  "
              f"n_nan={s['n_nan_cells']}  n_dupe_idx={s['n_duplicate_index_entries']}")

    flagged = [sym for sym, s in stats.items() if pd.Timestamp(s["first_date"]) < SUSPICIOUS_START_DATE]
    print(f"\nflagged (first_date < {SUSPICIOUS_START_DATE.date()}): {flagged if flagged else 'NONE'}")

    if not flagged:
        print("\nNo symbol predates the suspicious-start threshold -- the 1962 date in the prior run must have "
              "come from a symbol whose first_date is >= 1990 but still earlier than the rest of the universe, "
              "or was a transient read. Re-run the OOS validation script's own span print to confirm; if it still "
              "shows 1962, widen SUSPICIOUS_START_DATE in this script and re-run.")
        OUT_FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_FLAGS_PATH.write_text(json.dumps({"flagged_symbols": [], "verdict": "SAFE-TO-PROCEED (nothing flagged)"}), encoding="utf-8")
        print(f"\nwrote {OUT_FLAGS_PATH}")
        return

    print("\n" + "=" * 90)
    print("PER-FLAGGED-SYMBOL DETAIL")
    print("=" * 90)
    for sym in flagged:
        print(f"  {sym}: {json.dumps(stats[sym])}")

    states = build_states(price)
    states_without = {sym: df for sym, df in states.items() if sym not in flagged}

    print("\n" + "=" * 90)
    print("QUESTION 1 -- BREADTH: does removing the flagged symbol(s) change golden-cross breadth,")
    print("and if so, does the change reach into the recent/OOS-relevant window?")
    print("=" * 90)
    breadth_with = compute_breadth_series(states)
    breadth_without = compute_breadth_series(states_without)
    # IMPORTANT: compare only on dates where BOTH series have a real (non-NaN,
    # pre-ffill) value -- a naive ffill().dropna() would silently discard
    # exactly the period where the flagged symbol is the SOLE contributor to
    # breadth_without (breadth_without is undefined there, not "equal"), which
    # is the period most likely to show contamination. That blind spot is
    # reported explicitly instead of hidden.
    raw_without_defined = breadth_without.notna()
    solo_contributor_days = int((~raw_without_defined).sum())
    solo_contributor_dates = breadth_without.index[~raw_without_defined]
    solo_overlaps_recent = bool((solo_contributor_dates >= RECENT_WINDOW_START).any()) if len(solo_contributor_dates) else False
    print(f"  days where breadth_without is UNDEFINED (flagged symbol is the sole/only contributor that day): "
          f"{solo_contributor_days}"
          + (f", spanning {solo_contributor_dates.min().date()} -> {solo_contributor_dates.max().date()}" if solo_contributor_days else ""))
    print(f"  does that solo-contributor period overlap the recent window ({RECENT_WINDOW_START.date()}+)? "
          f"{'YES -- this needs attention regardless of the diff numbers below' if solo_overlaps_recent else 'no'}")

    both = pd.DataFrame({"with": breadth_with, "without": breadth_without}).dropna()
    both["abs_diff"] = (both["with"] - both["without"]).abs()
    max_diff_all_time = float(both["abs_diff"].max()) if not both.empty else float("nan")
    recent = both[both.index >= RECENT_WINDOW_START]
    max_diff_recent = float(recent["abs_diff"].max()) if not recent.empty else float("nan")
    oos_only = both[both.index >= OOS_SPLIT_DATE]
    max_diff_oos = float(oos_only["abs_diff"].max()) if not oos_only.empty else float("nan")
    print(f"  [on days where both are defined] max |breadth_with - breadth_without| over full history: {max_diff_all_time:.4f}")
    print(f"  [on days where both are defined] max diff from {RECENT_WINDOW_START.date()} onward: {max_diff_recent:.4f}")
    print(f"  [on days where both are defined] max diff from {OOS_SPLIT_DATE.date()} (OOS window) onward: {max_diff_oos:.4f}")
    breadth_material = (max_diff_recent > BREADTH_MATERIAL_DIFF) or solo_overlaps_recent
    print(f"  MATERIAL (diff > {BREADTH_MATERIAL_DIFF} in recent window, OR solo-contributor period reaches recent window)? "
          f"{'YES' if breadth_material else 'no'}")

    print("\n" + "=" * 90)
    print("QUESTION 2 -- CALIBRATION: does removing the flagged symbol(s) change the sizing")
    print(f"calibration medians used by compute_calibration_before (split={OOS_SPLIT_DATE.date()})?")
    print("=" * 90)
    atr_with, vol_with = compute_calibration_before(states, OOS_SPLIT_DATE)
    atr_without, vol_without = compute_calibration_before(states_without, OOS_SPLIT_DATE)
    atr_rel_diff = abs(atr_with - atr_without) / atr_without if atr_without else float("nan")
    vol_rel_diff = abs(vol_with - vol_without) / vol_without if vol_without else float("nan")
    print(f"  median_atr_pct   with={atr_with*100:.4f}%  without={atr_without*100:.4f}%  rel_diff={atr_rel_diff*100:.2f}%")
    print(f"  median_vol20     with={vol_with*100:.4f}%  without={vol_without*100:.4f}%  rel_diff={vol_rel_diff*100:.2f}%")
    calibration_material = (atr_rel_diff > CALIBRATION_MATERIAL_REL_DIFF) or (vol_rel_diff > CALIBRATION_MATERIAL_REL_DIFF)
    print(f"  MATERIAL (> {CALIBRATION_MATERIAL_REL_DIFF*100:.0f}% relative)? {'YES' if calibration_material else 'no'}")

    print("\n" + "=" * 90)
    print("QUESTION 3 -- PORTFOLIO AVAILABILITY: flagged symbol's own isolated trade history")
    print("(BASELINE entries, EQUAL-DOLLAR sizing, run on that symbol alone)")
    print("=" * 90)
    any_long_hold = False
    for sym in flagged:
        history = isolated_trade_history(states, sym, atr_with, vol_with)
        if not history:
            print(f"  {sym}: no closed trades in isolation (never golden-crossed with a valid ma200, or still open)")
            continue
        print(f"  {sym}: {len(history)} isolated trade(s)")
        for h in history:
            flag = "  <-- LONG HOLD FLAG" if h["flag_long_hold"] else ""
            print(f"    entry={h['entry_time']}  exit={h['exit_time']}  hold={h['holding_calendar_days']:>5}d  "
                  f"ret={h['ret_pct']:>6.2f}%  reason={h['exit_reason']}{flag}")
            any_long_hold = any_long_hold or h["flag_long_hold"]

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    material = breadth_material or calibration_material or any_long_hold
    verdict = "NEEDS EXCLUSION" if material else "SAFE TO PROCEED"
    reasons = []
    if breadth_material:
        reasons.append(f"breadth differs by more than {BREADTH_MATERIAL_DIFF} in the recent window")
    if calibration_material:
        reasons.append(f"calibration medians differ by more than {CALIBRATION_MATERIAL_REL_DIFF*100:.0f}% relative")
    if any_long_hold:
        reasons.append(f"flagged symbol held an isolated position longer than {LONG_HOLD_TRADING_DAYS} calendar days")
    print(f"  {verdict}" + (f" -- {'; '.join(reasons)}" if reasons else " -- flagged symbol(s) exist but none of the "
          "three checks crossed a materiality threshold; the deep history appears cosmetic, not contaminating."))

    OUT_FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_FLAGS_PATH.write_text(json.dumps({
        "flagged_symbols": flagged,
        "stats": {s: stats[s] for s in flagged},
        "breadth_max_diff_recent": max_diff_recent,
        "breadth_max_diff_oos": max_diff_oos,
        "calibration_atr_rel_diff": atr_rel_diff,
        "calibration_vol_rel_diff": vol_rel_diff,
        "verdict": verdict,
        "reasons": reasons,
    }, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_FLAGS_PATH}")


if __name__ == "__main__":
    main()
