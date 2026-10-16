"""Answers the two qualifications raised about the entry-filter result
(FILTER-MA200/VOLUME/STRUCTURE/FULL all lost badly on CAGR vs. baseline,
with win rate and stop-exit rate barely moving): does that actually prove
the filters were just gating exposure rather than removing worse trades,
and does a higher-PF-lower-CAGR variant like FILTER-FULL (PF 4.55 vs.
baseline's 3.99) still deserve a look for a different mandate?

Two diagnostics, both derived from ONE instrumented run of the exact same
baseline simulation already trusted (run_stoploss_tail_risk_test_equities'
own entry/exit/sizing rules, unchanged) rather than five separate re-runs,
because every filter variant is a strict SUBSET of baseline's own crosses
(same exit rule, same sizing -- a filter only decides whether a cross is
taken, never how it plays out once taken). So baseline's own realized
trades, each tagged with whether it would have passed each filter at its
own entry day, gives an exact partition into "kept" vs. "removed" per
filter without needing to re-simulate anything -- and the trades in each
bucket are the REAL, REALIZED outcomes, not a hypothetical.

DIAGNOSTIC 1 -- trade-outcome overlap (the user's own proposed check):
for each filter, split baseline's realized trades into kept/removed and
compare return, max adverse excursion (MAE -- worst intrabar drawdown
while the trade was open, i.e. how far it dipped before its actual exit),
max favorable excursion (MFE -- best intrabar gain while open, showing
how much of a winner's move was later given back), and holding period.
Heavy overlap in these distributions is exactly what "filters mostly gate
exposure, not quality" would look like; a real, separable difference in
the removed bucket (particularly negative MAE or short holding period)
would be evidence the filter IS catching something.

DIAGNOSTIC 2 -- exposure-normalized CAGR (the user's other ask): raw CAGR
comparisons are unfair if the filtered strategies simply have less capital
deployed on average. This re-runs each filter's own full backtest (not
just the baseline-derived bucket split, since AVERAGE EXPOSURE depends on
each variant's own capital path) tracking the time-weighted average share
of capital deployed at any moment, then reports CAGR divided by that
average exposure fraction alongside the raw number.

Caveat this script does NOT resolve (disclosed, not hidden): this is still
the same historical sample used for every earlier test in this thread, not
a genuinely untouched out-of-sample period, and it still ignores slippage/
commissions beyond the existing 12bps round-trip cost model and any
delisted-asset survivorship bias in the 60-symbol universe. Those require
new data sourcing, not just new code, and are flagged as open follow-ups,
not addressed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tools.run_cross_asset_golden_cross_backtest import ENTRY_COST, EXIT_COST
from tools.run_stoploss_tail_risk_test_equities import (
    STARTING_CAPITAL,
    PER_TRADE_ALLOCATION_PCT,
    _load_equities_universe,
)
from tools.run_pullback_entry_backtest_equities import fetch_ohlcv_batch, build_states
from tools.run_entry_filter_backtest_equities import FILTER_VARIANTS, BASELINE_STOP_PCT

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "filter_quality_exposure_diagnostic_results.json"


def run_instrumented_baseline(states: dict[str, pd.DataFrame], stop_pct: float = BASELINE_STOP_PCT) -> tuple[pd.DataFrame, float]:
    """Byte-for-byte the same rules as run_stoploss_tail_risk_test_equities.
    run_backtest() -- immediate entry on every cross_up, fixed-pct stop
    checked via that day's low, death-cross signal exit, 2% cash-constrained
    sizing -- with three purely OBSERVATIONAL additions that change no
    entry/exit/sizing decision: (1) running MAE/MFE tracked while a
    position is open, (2) each trade tagged with whether its OWN entry day
    would have passed each filter condition, (3) a time-weighted average
    exposure fraction accumulated every step. Sanity-check the printed
    n_trades/CAGR/PF against the already-trusted baseline (1785 trades,
    CAGR 8.47%, PF 3.99) before trusting anything derived from this."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions: dict[str, dict | None] = {sym: None for sym in states}
    trade_log = []
    exposure_sum = 0.0
    exposure_steps = 0

    for ts in all_index:
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional
        exposure_sum += deployed_notional / capital if capital > 0 else 0.0
        exposure_steps += 1

        for sym, state in states.items():
            if ts not in state.index:
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                continue
            pos = positions[sym]

            if pos is not None:
                pos["mae_low"] = min(pos["mae_low"], row["low"])
                pos["mfe_high"] = max(pos["mfe_high"], row["high"])
                stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                stop_hit = row["low"] <= stop_price
                if stop_hit:
                    exit_price = stop_price * (1 - EXIT_COST)
                    reason = "stop"
                elif row["cross_down"]:
                    exit_price = row["close"] * (1 - EXIT_COST)
                    reason = "signal"
                else:
                    continue
                ret = (exit_price / pos["entry_price"]) - 1
                pnl = pos["notional"] * ret
                capital += pnl
                holding_days = (ts - pos["entry_time"]).days
                trade_log.append({
                    "asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                    "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital,
                    "exit_reason": reason, "holding_days": holding_days,
                    "mae_pct": (pos["mae_low"] / pos["entry_price"] - 1) * 100,
                    "mfe_pct": (pos["mfe_high"] / pos["entry_price"] - 1) * 100,
                    "pass_ma200": pos["pass_ma200"], "pass_volume": pos["pass_volume"],
                    "pass_structure": pos["pass_structure"], "pass_ma50": pos["pass_ma50"],
                })
                positions[sym] = None
                free_capital += pos["notional"]
                continue

            if pos is None and row["cross_up"]:
                notional = capital * PER_TRADE_ALLOCATION_PCT / 100.0
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + ENTRY_COST)
                    positions[sym] = {
                        "entry_price": fill, "notional": notional, "entry_time": ts,
                        "mae_low": row["low"], "mfe_high": row["high"],
                        "pass_ma200": bool(row["ma200_rising"]),
                        "pass_volume": bool(row["volume_confirm"]),
                        "pass_structure": bool(row["structure_ok"]),
                        "pass_ma50": bool(row["ma50_rising"]),
                    }
                    free_capital -= notional

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    avg_exposure_pct = (exposure_sum / exposure_steps * 100) if exposure_steps else float("nan")
    return trades, avg_exposure_pct


def run_instrumented_filter_backtest(
    states: dict[str, pd.DataFrame], require_volume: bool, require_structure: bool, require_ma50: bool,
    stop_pct: float = BASELINE_STOP_PCT,
) -> tuple[pd.DataFrame, float]:
    """Same as run_entry_filter_backtest_equities.run_entry_filter_backtest,
    plus the same time-weighted average-exposure tracking as the
    instrumented baseline above, so raw CAGR and exposure-normalized CAGR
    can be reported side by side for each filter variant."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions: dict[str, dict | None] = {sym: None for sym in states}
    trade_log = []
    exposure_sum = 0.0
    exposure_steps = 0

    for ts in all_index:
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional
        exposure_sum += deployed_notional / capital if capital > 0 else 0.0
        exposure_steps += 1

        for sym, state in states.items():
            if ts not in state.index:
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                continue
            pos = positions[sym]

            if pos is not None:
                stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                stop_hit = row["low"] <= stop_price
                if stop_hit:
                    exit_price = stop_price * (1 - EXIT_COST)
                    reason = "stop"
                elif row["cross_down"]:
                    exit_price = row["close"] * (1 - EXIT_COST)
                    reason = "signal"
                else:
                    continue
                ret = (exit_price / pos["entry_price"]) - 1
                pnl = pos["notional"] * ret
                capital += pnl
                trade_log.append({
                    "asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                    "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital,
                    "exit_reason": reason,
                })
                positions[sym] = None
                free_capital += pos["notional"]
                continue

            if row["cross_up"]:
                filter_ok = (
                    row["ma200_rising"]
                    and (not require_volume or row["volume_confirm"])
                    and (not require_structure or row["structure_ok"])
                    and (not require_ma50 or row["ma50_rising"])
                )
                if not filter_ok:
                    continue
                notional = capital * PER_TRADE_ALLOCATION_PCT / 100.0
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + ENTRY_COST)
                    positions[sym] = {"entry_price": fill, "notional": notional, "entry_time": ts}
                    free_capital -= notional

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    avg_exposure_pct = (exposure_sum / exposure_steps * 100) if exposure_steps else float("nan")
    return trades, avg_exposure_pct


def summarize_core(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    wins = trades[trades.pnl > 0]
    gross_win = wins.pnl.sum()
    gross_loss = -trades[trades.pnl <= 0].pnl.sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    final_capital = trades.capital_after.iloc[-1]
    total_return = (final_capital / STARTING_CAPITAL - 1) * 100
    peak, max_dd = STARTING_CAPITAL, 0.0
    for c in trades.capital_after:
        peak = max(peak, c)
        max_dd = min(max_dd, (c / peak - 1) * 100)
    years = max((trades["exit_time"].max() - trades["entry_time"].min()).days / 365.25, 1e-9)
    cagr = (final_capital / STARTING_CAPITAL) ** (1 / years) - 1
    r = trades["ret_pct"].to_numpy() / 100.0
    trades_per_year = len(trades) / years
    mean_r, std_r = r.mean(), r.std(ddof=1)
    sharpe = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else float("nan")
    return {
        "n_trades": len(trades), "win_rate": len(wins) / len(trades) * 100, "pf": pf,
        "cagr_pct": cagr * 100, "total_return_pct": total_return, "max_dd_pct": max_dd,
        "worst_trade_pct": trades["ret_pct"].min(), "sharpe": sharpe,
    }


def _bucket_stats(sub: pd.DataFrame, col: str) -> dict:
    if sub.empty:
        return {"n": 0}
    vals = sub[col].to_numpy(dtype=float)
    return {"n": len(sub), "mean": float(np.mean(vals)), "median": float(np.median(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}


def _effect_size_and_test(kept: pd.Series, removed: pd.Series) -> dict:
    """Cohen's d (no scipy needed) always; Mann-Whitney U p-value if scipy
    is installed on the user's machine (checked, not assumed)."""
    k, rm = kept.to_numpy(dtype=float), removed.to_numpy(dtype=float)
    out = {}
    if len(k) > 1 and len(rm) > 1:
        pooled_std = np.sqrt(((len(k) - 1) * k.std(ddof=1) ** 2 + (len(rm) - 1) * rm.std(ddof=1) ** 2) / (len(k) + len(rm) - 2))
        out["cohens_d"] = float((rm.mean() - k.mean()) / pooled_std) if pooled_std > 0 else 0.0
    if _HAVE_SCIPY and len(k) > 0 and len(rm) > 0:
        try:
            u_stat, p_value = _scipy_stats.mannwhitneyu(k, rm, alternative="two-sided")
            out["mannwhitney_p"] = float(p_value)
        except Exception:
            pass
    return out


def diagnostic_1_trade_overlap(baseline_trades: pd.DataFrame) -> list[dict]:
    print("\n" + "=" * 78)
    print("DIAGNOSTIC 1 -- kept vs. removed trade-outcome overlap, per filter")
    print("(derived from baseline's own REALIZED trades -- same exit rule for")
    print("every trade regardless of filter, so this is the real outcome each")
    print("trade actually had, not a simulation of what a filter run would do)")
    print("=" * 78)
    rows = []
    for variant in FILTER_VARIANTS:
        pass_mask = baseline_trades["pass_ma200"].copy()
        if variant["require_volume"]:
            pass_mask &= baseline_trades["pass_volume"]
        if variant["require_structure"]:
            pass_mask &= baseline_trades["pass_structure"]
        if variant["require_ma50"]:
            pass_mask &= baseline_trades["pass_ma50"]
        kept, removed = baseline_trades[pass_mask], baseline_trades[~pass_mask]

        print(f"\n--- {variant['label']} --- kept={len(kept)}  removed={len(removed)}")
        row = {"variant": variant["label"], "n_kept": len(kept), "n_removed": len(removed)}
        for col, name in [("ret_pct", "return_pct"), ("mae_pct", "MAE_pct"), ("mfe_pct", "MFE_pct"), ("holding_days", "holding_days")]:
            k_stats, r_stats = _bucket_stats(kept, col), _bucket_stats(removed, col)
            test = _effect_size_and_test(kept[col], removed[col]) if len(kept) > 1 and len(removed) > 1 else {}
            print(f"  {name:<14} kept: mean={k_stats.get('mean', float('nan')):+8.2f} median={k_stats.get('median', float('nan')):+8.2f}  |  "
                  f"removed: mean={r_stats.get('mean', float('nan')):+8.2f} median={r_stats.get('median', float('nan')):+8.2f}"
                  f"{'  |  cohens_d=' + format(test['cohens_d'], '+.2f') if 'cohens_d' in test else ''}"
                  f"{'  p=' + format(test['mannwhitney_p'], '.3f') if 'mannwhitney_p' in test else ''}")
            row[f"kept_{name}"] = k_stats
            row[f"removed_{name}"] = r_stats
            row[f"{name}_test"] = test
        rows.append(row)
    print("\n(|Cohen's d| < 0.2 is conventionally a 'small' effect -- heavy overlap;")
    print(" > 0.5 is 'medium', > 0.8 'large'. p < 0.05 on Mann-Whitney = distributions")
    print(" differ at conventional significance, if scipy was available to compute it.)")
    if not _HAVE_SCIPY:
        print(" (scipy not found on this machine -- p-values skipped, Cohen's d still shown.)")
    return rows


def diagnostic_2_exposure_normalized(states: dict[str, pd.DataFrame], baseline_trades: pd.DataFrame, baseline_avg_exposure: float) -> list[dict]:
    print("\n" + "=" * 78)
    print("DIAGNOSTIC 2 -- exposure-normalized CAGR (CAGR / average capital")
    print("fraction actually deployed) -- re-runs each filter's own full")
    print("backtest to get ITS OWN average exposure, not baseline's.")
    print("=" * 78)
    baseline_summary = summarize_core(baseline_trades)
    baseline_norm = baseline_summary["cagr_pct"] / (baseline_avg_exposure / 100) if baseline_avg_exposure else float("nan")
    print(f"BASELINE          avg_exposure={baseline_avg_exposure:5.1f}%  CAGR={baseline_summary['cagr_pct']:5.2f}%  "
          f"CAGR/exposure={baseline_norm:6.2f}")

    rows = [{"variant": "BASELINE", "avg_exposure_pct": baseline_avg_exposure,
             "cagr_pct": baseline_summary["cagr_pct"], "cagr_per_exposure": baseline_norm}]
    for variant in FILTER_VARIANTS:
        trades, avg_exposure = run_instrumented_filter_backtest(
            states, require_volume=variant["require_volume"], require_structure=variant["require_structure"],
            require_ma50=variant["require_ma50"],
        )
        summ = summarize_core(trades)
        if not summ:
            print(f"{variant['label']:<18} no trades")
            continue
        norm = summ["cagr_pct"] / (avg_exposure / 100) if avg_exposure else float("nan")
        print(f"{variant['label']:<18} avg_exposure={avg_exposure:5.1f}%  CAGR={summ['cagr_pct']:5.2f}%  "
              f"CAGR/exposure={norm:6.2f}  (n={summ['n_trades']}, sanity-check vs. earlier "
              f"standalone run's n_trades/CAGR should match)")
        rows.append({"variant": variant["label"], "avg_exposure_pct": avg_exposure,
                      "cagr_pct": summ["cagr_pct"], "cagr_per_exposure": norm, "n_trades": summ["n_trades"]})
    print("\n(CAGR/exposure = CAGR earned per unit of average capital actually at risk --")
    print(" the fairer comparison the user asked for, since the filtered variants keep")
    print(" more capital in cash on average and raw CAGR penalizes that idle capital twice.)")
    return rows


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    states = build_states(price)

    print("=" * 78)
    print("Instrumented BASELINE run (must match the already-trusted baseline:")
    print("1785 trades, CAGR 8.47%, PF 3.99, worst trade -12.1% -- sanity-check")
    print("this before trusting anything derived below)")
    print("=" * 78)
    baseline_trades, baseline_avg_exposure = run_instrumented_baseline(states)
    baseline_summary = summarize_core(baseline_trades)
    print(f"n_trades={baseline_summary.get('n_trades')}  win_rate={baseline_summary.get('win_rate', float('nan')):.1f}%  "
          f"pf={baseline_summary.get('pf', float('nan')):.2f}  cagr={baseline_summary.get('cagr_pct', float('nan')):.2f}%  "
          f"worst_trade={baseline_summary.get('worst_trade_pct', float('nan')):.1f}%  "
          f"avg_exposure={baseline_avg_exposure:.1f}%")

    overlap_rows = diagnostic_1_trade_overlap(baseline_trades)
    exposure_rows = diagnostic_2_exposure_normalized(states, baseline_trades, baseline_avg_exposure)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "baseline_summary": baseline_summary,
        "baseline_avg_exposure_pct": baseline_avg_exposure,
        "diagnostic_1_trade_overlap": overlap_rows,
        "diagnostic_2_exposure_normalized": exposure_rows,
        "scipy_available": _HAVE_SCIPY,
    }, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
