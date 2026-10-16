"""Tests an "entry FILTER" variant of the golden-cross baseline: instead of
waiting for a pullback before entering (tools/run_pullback_entry_backtest_
equities.py -- which lost badly on CAGR/Sharpe in every stop configuration
tested, pure-ATR and hybrid-capped alike, because waiting for a qualifying
setup skipped 14-55% of all golden-cross regimes and left capital idle),
this keeps baseline's own behavior -- enter IMMEDIATELY on the cross, same
fixed -12% stop, same cash-constrained 2%-per-trade sizing -- and only adds
a same-day QUALITY FILTER on the cross itself: skip the trade entirely (not
delay it) when the cross looks weak by cheap, already-computed measures.

This directly tests the hypothesis raised after the hybrid-stop result: the
pullback rule's real problem wasn't entry timing or stop distance, it was
opportunity cost from waiting. If quality-filtering crosses (while still
taking every OTHER cross immediately, same as baseline) can lift baseline's
own numbers without that opportunity cost, that's a real edge; if it can't
beat baseline either, the golden-cross baseline's own simplicity is doing
something right that added conditions keep undermining.

Filters tested (all built on top of REQUIRE ma200_rising, the same "is this
actually a healthy uptrend, not a whipsaw" condition both PULLBACK-SIMPLE
and PULLBACK-FULL required in the earlier backtest -- kept as the floor
here too since it's the cheapest, most defensible one):

  FILTER-MA200:     ma200_rising only.
  FILTER-VOLUME:    + above-average volume on the cross day itself.
  FILTER-STRUCTURE: + higher-highs/higher-lows market structure.
  FILTER-FULL:      + volume AND structure AND a rising 50-day average --
                    the same four conditions PULLBACK-FULL's checklist
                    required, just checked on the cross day instead of
                    waited for during a pullback.

A cross that fails its filter is skipped for that regime -- it is NOT
retried later in the same regime (unlike the pullback rule's "arm and wait"
logic) since the whole point here is "trade immediately or not at all,"
not "wait for the filter to pass."

Reuses this project's own already-validated pieces directly: ENTRY_COST/
EXIT_COST/compute_trend_state from run_cross_asset_golden_cross_backtest;
STARTING_CAPITAL/PER_TRADE_ALLOCATION_PCT/the robust-universe loader/
run_backtest (baseline)/summarize from run_stoploss_tail_risk_test_
equities (so BASELINE here is byte-for-byte the same simulation the live
bot's -12% stop was validated against); and compute_pullback_indicators/
build_states/fetch_ohlcv_batch from run_pullback_entry_backtest_equities,
since ma200_rising/volume_confirm/structure_ok/ma50_rising are already
computed there exactly the same way this test needs them -- no reason to
recompute the same rolling windows a third time in a third file.

The entry-check/stop-check loop below deliberately mirrors run_stoploss_
tail_risk_test_equities.run_backtest()'s own shape almost line for line
(same on-the-fly stop_price = entry_price * (1 + stop_pct/100) computed at
check time, not cached at entry) so this is legibly "baseline plus one
added condition," not a structurally different simulation that could
quietly diverge from what's live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tools.run_cross_asset_golden_cross_backtest import ENTRY_COST, EXIT_COST
from tools.run_stoploss_tail_risk_test_equities import (
    STARTING_CAPITAL,
    PER_TRADE_ALLOCATION_PCT,
    _load_equities_universe,
    run_backtest as run_baseline_backtest,
    summarize,
)
from tools.run_pullback_entry_backtest_equities import (
    fetch_ohlcv_batch,
    build_states,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "entry_filter_backtest_results.json"
BASELINE_STOP_PCT = -12.0  # unchanged from live -- isolates the entry filter as the only variable

FILTER_VARIANTS = [
    {"label": "FILTER-MA200", "require_volume": False, "require_structure": False, "require_ma50": False},
    {"label": "FILTER-VOLUME", "require_volume": True, "require_structure": False, "require_ma50": False},
    {"label": "FILTER-STRUCTURE", "require_volume": False, "require_structure": True, "require_ma50": False},
    {"label": "FILTER-FULL", "require_volume": True, "require_structure": True, "require_ma50": True},
]


def run_entry_filter_backtest(
    states: dict[str, pd.DataFrame],
    require_volume: bool,
    require_structure: bool,
    require_ma50: bool,
    stop_pct: float = BASELINE_STOP_PCT,
) -> pd.DataFrame:
    """Baseline's own cash-constrained event loop (immediate entry on
    cross_up, fixed-pct stop checked on the fly, one trade per regime),
    with exactly one addition: a cross_up day is only tradeable when it
    passes the filter. A filtered-out cross is skipped for good -- this
    regime gets no trade at all, unlike the pullback script's "arm and
    wait for a later qualifying day" behavior."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions: dict[str, dict | None] = {sym: None for sym in states}
    trade_log = []
    skipped_crosses = {sym: 0 for sym in states}

    for ts in all_index:
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional

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
                    skipped_crosses[sym] += 1
                    continue

                notional = capital * PER_TRADE_ALLOCATION_PCT / 100.0
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + ENTRY_COST)
                    positions[sym] = {"entry_price": fill, "notional": notional, "entry_time": ts}
                    free_capital -= notional

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    trades.attrs["skipped_crosses"] = sum(skipped_crosses.values())
    return trades


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    price_hlc_only = {sym: df[["high", "low", "close"]] for sym, df in price.items()}

    print("=" * 78)
    print("BASELINE (identical to the live bot: immediate entry on every golden")
    print(f"cross, fixed {BASELINE_STOP_PCT:.0f}% stop) -- run_stoploss_tail_risk_test_")
    print("equities.py's own run_backtest(), called directly.")
    print("=" * 78)
    baseline_trades = run_baseline_backtest(price_hlc_only, BASELINE_STOP_PCT)
    baseline_summary = summarize(baseline_trades, "BASELINE (live-equivalent, -12% stop)")
    total_crosses = baseline_summary.get("n_trades", 0) if baseline_summary else 0

    print("\nprecomputing filter indicators once (reused from the pullback script)...")
    states = build_states(price)

    results: list[dict] = []
    for variant in FILTER_VARIANTS:
        trades = run_entry_filter_backtest(
            states,
            require_volume=variant["require_volume"],
            require_structure=variant["require_structure"],
            require_ma50=variant["require_ma50"],
            stop_pct=BASELINE_STOP_PCT,
        )
        skipped = trades.attrs.get("skipped_crosses", 0) if not trades.empty else None
        summ = summarize(trades, variant["label"])
        if skipped is not None:
            kept = summ.get("n_trades", 0)
            pct_kept = kept / total_crosses * 100 if total_crosses else float("nan")
            print(f"  ({skipped} of {total_crosses} crosses filtered out -- "
                  f"{kept} kept, {pct_kept:.0f}% of baseline's trade count)")
        results.append({"variant": variant["label"], "skipped_crosses": skipped, **summ})

    print("\n" + "=" * 78)
    print("ENTRY-FILTER SUMMARY (baseline shown once for reference)")
    print("=" * 78)
    header = (f"{'Variant':<18}{'Trades':>8}{'Stop%':>7}{'WinRate':>9}{'PF':>7}"
              f"{'CAGR':>8}{'TotalRet':>11}{'MaxDD':>8}{'WorstTrd':>9}")
    print(header)
    if baseline_summary:
        r = baseline_summary
        stop_pct_disp = (r["stop_exits"] / r["n_trades"] * 100) if r.get("n_trades") else float("nan")
        print(f"{'BASELINE':<18}{r['n_trades']:>8}{stop_pct_disp:>6.0f}%{r['win_rate']:>8.1f}%"
              f"{r['pf']:>7.2f}{r['cagr_pct']:>7.1f}%{r['total_return_pct']:>10.1f}%"
              f"{r['max_dd_pct']:>7.1f}%{r['worst_trade_pct']:>8.1f}%")
    for row in results:
        if row.get("n_trades") is None:
            print(f"{row['variant']:<18}  (no trades)")
            continue
        stop_pct_disp = (row["stop_exits"] / row["n_trades"] * 100) if row.get("n_trades") else float("nan")
        print(f"{row['variant']:<18}{row['n_trades']:>8}{stop_pct_disp:>6.0f}%{row['win_rate']:>8.1f}%"
              f"{row['pf']:>7.2f}{row['cagr_pct']:>7.1f}%{row['total_return_pct']:>10.1f}%"
              f"{row['max_dd_pct']:>7.1f}%{row['worst_trade_pct']:>8.1f}%")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "universe_size": len(price),
        "params": {
            "baseline_stop_pct": BASELINE_STOP_PCT,
            "per_trade_allocation_pct": PER_TRADE_ALLOCATION_PCT,
            "filter_variants": FILTER_VARIANTS,
        },
        "baseline_summary": baseline_summary,
        "total_crosses": total_crosses,
        "results": results,
    }, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
