"""Quarter-end seasonality test, per the user's own claim: "at end of
quarter, most stocks sell off." This tests that claim empirically against
the SAME CURRENT universe (robust==True AND asset_class != "Crypto", 60
symbols) every prior script in this sequence has used -- no new data
source, no new universe -- following this project's own predeclare-then-
test discipline: the windows and the comparison method below were fixed
BEFORE this script was run against real data.

WHAT "QUARTER-END EFFECT" MEANS HERE, PRECISELY: for each symbol and each
historical completed calendar quarter (Mar 31 / Jun 30 / Sep 30 / Dec 31,
mapped to the last actual trading day on or before that date), the return
over the LAST N trading days ENDING AT that quarter-end day. Three window
sizes are tested (N = 3, 5, 10 trading days), predeclared together rather
than picked after looking at any one result, since a claim this informal
("most stocks sell off") doesn't specify a window and testing only one
would risk unintentionally cherry-picking whichever window happens to
show the effect.

THE KEY METHODOLOGICAL GUARD, DISCLOSED: an N-day return is often
negative just from ordinary volatility drag, independent of any calendar
effect -- so "quarter-end windows average a negative return" is not by
itself evidence of a quarter-end effect. This script instead compares
quarter-end-window returns against a BASELINE population of same-length
windows ending on ordinary (non-quarter-end) trading days, drawn from the
SAME symbols' SAME history, and reports the difference plus an effect
size and (if available) a significance test -- so the question actually
being answered is "are quarter-end windows different from typical windows
for these same instruments," not "are quarter-end windows negative in
isolation."

DISCLOSED LIMITATIONS (do not skip these when reading the output):
  - Overlapping windows across nearby quarters, and same-day comovement
    across the 60 symbols, both violate the independence assumption
    behind any p-value below. The Mann-Whitney p-value (when scipy is
    available) should be read as a rough signal-to-noise indicator, not
    a rigorous hypothesis test -- exactly the same caveat this project
    already applies to Mann-Whitney results elsewhere (see
    run_filter_quality_exposure_diagnostic_equities.py).
  - The baseline population excludes any window ending within
    EXCLUSION_BUFFER_DAYS trading days of a quarter-end (on either side),
    so it is not contaminated by the very effect being tested for -- but
    it is still ordinary market history, which has its own regimes/trends
    that this test does not otherwise control for.
  - This tests calendar seasonality only. It does not test, imply, or
    depend on WHY a quarter-end effect would exist (index rebalancing,
    institutional window dressing, tax-related selling, etc.) -- purely
    descriptive, like every diagnostic in this sequence.
  - Finding an effect here is not, by itself, a proposal to add a new
    entry/exit rule to the locked control -- that would be a separate,
    later decision, consistent with how every other diagnostic in this
    project has been kept apart from the control itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tools.run_stoploss_tail_risk_test_equities import _load_equities_universe
from tools.run_pullback_entry_backtest_equities import fetch_ohlcv_batch

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "quarter_end_seasonality_results.json"

# Predeclared, fixed before this script was run against real data.
QUARTER_END_WINDOWS = [3, 5, 10]          # trading-day window sizes tested
EXCLUSION_BUFFER_DAYS = 10                # baseline windows must end at least this many trading
                                          # days from any quarter-end (either side) -- equal to the
                                          # largest tested window, so no baseline window can overlap
                                          # even the widest quarter-end window
BASELINE_SUBSAMPLE_STRIDE = 5             # only every Nth eligible non-quarter-end day is used as a
                                          # baseline observation, to keep the comparison tractable
                                          # and reduce (not eliminate) overlapping-window autocorrelation
QUARTER_END_MONTHS_DAYS = [(3, 31), (6, 30), (9, 30), (12, 31)]


def _quarter_end_positions(index: pd.DatetimeIndex) -> list[int]:
    """Positional indices (into `index`) of the last trading day on or
    before each calendar quarter-end the data actually covers. Skips a
    quarter-end target that falls after the last available trading day
    (the still-in-progress current quarter) or whose matched trading day
    is more than 10 calendar days before the target (a data gap -- e.g.
    a symbol's history starting mid-quarter -- rather than a genuine
    completed quarter)."""
    if len(index) == 0:
        return []
    years = range(index[0].year, index[-1].year + 1)
    positions = []
    for year in years:
        for month, day in QUARTER_END_MONTHS_DAYS:
            target = pd.Timestamp(year=year, month=month, day=day)
            if target > index[-1]:
                continue
            pos = int(index.searchsorted(target, side="right")) - 1
            if pos < 0:
                continue
            matched_date = index[pos]
            if (target - matched_date).days > 10:
                continue
            positions.append(pos)
    return positions


def _window_return(closes: np.ndarray, end_idx: int, window: int) -> float | None:
    start_idx = end_idx - window
    if start_idx < 0 or closes[start_idx] == 0 or np.isnan(closes[start_idx]) or np.isnan(closes[end_idx]):
        return None
    return float(closes[end_idx] / closes[start_idx] - 1)


def collect_returns_for_symbol(close: pd.Series, window: int) -> tuple[list[float], list[float]]:
    """Returns (quarter_end_returns, baseline_returns) for one symbol at
    one window size, using the SAME close series and SAME quarter-end
    positions for both -- so the only thing that differs between the two
    lists is WHICH day the window ends on."""
    closes = close.to_numpy(dtype=float)
    index = close.index
    q_positions = _quarter_end_positions(index)
    min_history = max(QUARTER_END_WINDOWS)
    q_positions = [p for p in q_positions if p >= min_history]

    quarter_end_returns = []
    for pos in q_positions:
        r = _window_return(closes, pos, window)
        if r is not None:
            quarter_end_returns.append(r)

    baseline_returns = []
    if q_positions:
        for end_idx in range(min_history, len(closes), BASELINE_SUBSAMPLE_STRIDE):
            if any(abs(end_idx - q) < EXCLUSION_BUFFER_DAYS for q in q_positions):
                continue
            r = _window_return(closes, end_idx, window)
            if r is not None:
                baseline_returns.append(r)
    return quarter_end_returns, baseline_returns


def _effect_size_and_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Cohen's d (no scipy needed) always; Mann-Whitney U p-value if
    scipy is installed on the machine this actually runs on (checked,
    not assumed) -- same convention as
    run_filter_quality_exposure_diagnostic_equities.py's own
    _effect_size_and_test, deliberately mirrored here rather than
    reinvented differently."""
    out = {}
    if len(a) > 1 and len(b) > 1:
        pooled_std = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2))
        out["cohens_d"] = float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0
    if _HAVE_SCIPY and len(a) > 0 and len(b) > 0:
        try:
            _, p_value = _scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            out["mannwhitney_p"] = float(p_value)
        except Exception:
            pass
    return out


def _summary(vals: np.ndarray) -> dict:
    if len(vals) == 0:
        return {"n": 0}
    return {
        "n": int(len(vals)),
        "mean_pct": float(np.mean(vals) * 100),
        "median_pct": float(np.median(vals) * 100),
        "std_pct": float(np.std(vals, ddof=1) * 100) if len(vals) > 1 else 0.0,
        "pct_negative": float(np.mean(vals < 0) * 100),
    }


def main() -> None:
    symbols = _load_equities_universe()
    print(f"CURRENT universe: {len(symbols)} non-crypto robust instruments")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments (full available history)\n")

    results = {}
    for window in QUARTER_END_WINDOWS:
        pooled_qe, pooled_base = [], []
        per_symbol_qe_mean = {}
        for sym, df in price.items():
            qe, base = collect_returns_for_symbol(df["Close"], window)
            pooled_qe.extend(qe)
            pooled_base.extend(base)
            if qe:
                per_symbol_qe_mean[sym] = float(np.mean(qe))

        qe_arr, base_arr = np.array(pooled_qe), np.array(pooled_base)
        qe_summary, base_summary = _summary(qe_arr), _summary(base_arr)
        test = _effect_size_and_test(qe_arr, base_arr)
        n_symbols_negative_mean = sum(1 for v in per_symbol_qe_mean.values() if v < 0)

        print("=" * 100)
        print(f"WINDOW = last {window} trading days ending at quarter-end")
        print("=" * 100)
        print(f"  quarter-end windows   n={qe_summary.get('n', 0):>5}  mean={qe_summary.get('mean_pct', float('nan')):+.3f}%  "
              f"median={qe_summary.get('median_pct', float('nan')):+.3f}%  %negative={qe_summary.get('pct_negative', float('nan')):.1f}%")
        print(f"  baseline windows      n={base_summary.get('n', 0):>5}  mean={base_summary.get('mean_pct', float('nan')):+.3f}%  "
              f"median={base_summary.get('median_pct', float('nan')):+.3f}%  %negative={base_summary.get('pct_negative', float('nan')):.1f}%")
        diff = qe_summary.get("mean_pct", float("nan")) - base_summary.get("mean_pct", float("nan"))
        print(f"  difference (quarter-end mean - baseline mean): {diff:+.3f}pp"
              f"{'  cohens_d=' + format(test['cohens_d'], '+.3f') if 'cohens_d' in test else ''}"
              f"{'  mannwhitney_p=' + format(test['mannwhitney_p'], '.4f') if 'mannwhitney_p' in test else ''}")
        print(f"  symbols with negative MEAN quarter-end return: {n_symbols_negative_mean} of {len(per_symbol_qe_mean)}")

        results[str(window)] = {
            "quarter_end": qe_summary,
            "baseline": base_summary,
            "difference_mean_pct": diff,
            "test": test,
            "n_symbols_with_negative_mean_qe_return": n_symbols_negative_mean,
            "n_symbols_with_any_qe_observation": len(per_symbol_qe_mean),
            "per_symbol_mean_qe_return_pct": {k: v * 100 for k, v in per_symbol_qe_mean.items()},
        }

    if not _HAVE_SCIPY:
        print("\n(scipy not found on this machine -- Mann-Whitney p-values skipped, Cohen's d still shown.)")
    print("\n(|Cohen's d| < 0.2 is conventionally a 'small' effect, > 0.5 'medium', > 0.8 'large'.")
    print(" Read the p-value as a rough signal, not a rigorous test -- see this script's own")
    print(" docstring on the overlapping-window/comovement independence caveat.)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "windows_tested": QUARTER_END_WINDOWS,
        "exclusion_buffer_days": EXCLUSION_BUFFER_DAYS,
        "baseline_subsample_stride": BASELINE_SUBSAMPLE_STRIDE,
        "scipy_available": _HAVE_SCIPY,
        "results": results,
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
