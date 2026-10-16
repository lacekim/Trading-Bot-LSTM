"""Cost and liquidity sensitivity (item 3 of the post-exit-logic validation/
deployment-readiness sequence), per the user's own exact spec:

  "Keep the current 12 bps model as the baseline. Add explicit spread/
  slippage scenarios and liquidity caps. Report whether the conclusion
  changes, rather than selecting a preferred cost assumption after seeing
  results."

PREDECLARATION (fixed BEFORE this script was run against real data, exactly
as every prior script in this sequence has done): five scenarios, chosen in
advance and run identically, with no scenario added or dropped after seeing
results.

  1. BASELINE           -- the unchanged existing cost model: ENTRY_COST=
                            EXIT_COST=12bps each (24bps round trip total,
                            exactly as computed by every prior script's own
                            cost_bps_round_trip field), EQUAL-DOLLAR sizing,
                            uncapped. This byte-for-byte reproduces the
                            existing control.
  2. MODERATE_SPREAD     -- same 12bps/12bps commission, plus an explicit
                            +10bps each-way spread/slippage add-on (22bps/
                            22bps, 44bps round trip total), uncapped sizing.
  3. WIDE_SPREAD         -- same 12bps/12bps commission, plus a larger
                            +25bps each-way spread/slippage add-on (37bps/
                            37bps, 74bps round trip total), uncapped sizing.
  4. LIQUIDITY_CAP       -- unchanged 12bps/12bps commission, but EQUAL-
                            DOLLAR sizing is capped at 1% of the symbol's
                            own trailing 20-day average dollar volume
                            (ADV), computed through yesterday only (lag-1,
                            no look-ahead -- same convention this codebase
                            already uses for vol20_lag1/atr_pct_lag1 in
                            run_sizing_allocation_backtest_equities.py). If
                            the baseline 2%-of-capital notional would
                            exceed that cap, the position is sized DOWN to
                            the cap rather than skipped.
  5. COMBINED_STRESS     -- MODERATE_SPREAD's costs (not WIDE_SPREAD's --
                            deliberately a plausible combined case, not the
                            single worst case) AND the same 1%-of-ADV
                            liquidity cap, together.

DESIGN CHOICE, DISCLOSED: this is built on the SAME_CLOSE / exact-stop-
price control (run_backtest_exit_variant, unmodified, imported and called
exactly as every prior script has called it) -- NOT the item-2 gap-aware
stop model. Item 2 and item 3 are each a single, separate, additive change
to the control; combining them is a further study, not this one, and is
left out to keep "change exactly one thing at a time." Only the entry_cost/
exit_cost scalars and the sizing_fn passed into that same, unmodified
control loop differ across the five scenarios above -- the entry/exit/stop
timing logic itself is identical in all five and is never touched by this
script.

WHAT'S REPORTED: n_trades, CAGR, Sharpe, Sortino, max_dd for each of the
five scenarios; for the two capped scenarios, the number and % of entry
signals where the cap actually bound (reduced the position below its
uncapped 2%-of-capital size) and the average size reduction when it did;
and the CAGR/max_dd change of every non-baseline scenario relative to
BASELINE. No scenario is treated as "the" answer -- this reports whether
the retain-the-control conclusion is robust to each of these five cost/
liquidity assumptions individually, not a single blended verdict.

OUT OF SCOPE here (left for the remaining audit items): the gap-through-
stop fill model (item 2, already shipped separately); independent trade-
ledger reconciliation (item 4); universe/concentration diagnostics
(item 5)."""
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
from tools.run_stoploss_tail_risk_test_equities import _load_equities_universe
from tools.run_pullback_entry_backtest_equities import fetch_ohlcv_batch
from tools.run_time_based_exit_backtest_equities import (
    run_backtest_exit_variant as run_control,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "cost_liquidity_sensitivity_results.json"

ADV_WINDOW = 20
PARTICIPATION_RATE = 0.01  # 1% of trailing 20-day ADV, predeclared

SPREAD_BPS = {
    "BASELINE": 0.0,
    "MODERATE_SPREAD": 10.0,
    "WIDE_SPREAD": 25.0,
    "LIQUIDITY_CAP": 0.0,
    "COMBINED_STRESS": 10.0,
}
LIQUIDITY_CAPPED = {"LIQUIDITY_CAP", "COMBINED_STRESS"}


def _states_with_adv(states: dict, window: int = ADV_WINDOW) -> dict:
    """Adds a trailing-window average-dollar-volume column, lagged by one
    day (no look-ahead), to each symbol's own state DataFrame. Both 'close'
    and 'volume' already survive build_states() unchanged (unlike 'open',
    which build_states() drops -- see the item-1/item-2 scripts' own
    _states_with_open fix), so this needs no raw price dict at all: it is
    purely derived from columns already present on the state itself. Mirrors
    the existing vol20_lag1/atr_pct_lag1 convention in
    run_sizing_allocation_backtest_equities.py (compute the raw rolling
    statistic, then .shift(1))."""
    out = {}
    for sym, df in states.items():
        df2 = df.copy()
        dollar_vol = df2["close"] * df2["volume"]
        df2["adv_dollar_lag1"] = dollar_vol.rolling(window).mean().shift(1)
        out[sym] = df2
    return out


def make_liquidity_capped_sizing_fn(base_sizing_fn, participation_rate: float, stats: dict):
    """Wraps an existing sizing_fn(capital, row) -> notional with a cap at
    participation_rate * row['adv_dollar_lag1']. Falls back to the
    UNCAPPED base sizing when adv_dollar_lag1 is NaN or non-positive (
    insufficient trailing history -- a data-availability limitation, not a
    liquidity-cap decision, and tracked separately in `stats` so it is never
    silently folded into either the capped or the not-binding counts).
    `stats` is a plain dict the caller owns and inspects afterward; this
    function only ever increments its counters, never reads them."""
    def fn(capital, row):
        base_notional = base_sizing_fn(capital, row)
        adv = row.get("adv_dollar_lag1", float("nan"))
        if pd.isna(adv) or adv <= 0:
            stats["insufficient_history"] += 1
            return base_notional
        cap = adv * participation_rate
        if base_notional > cap:
            stats["capped"] += 1
            stats["capped_reduction_pct_sum"] += (1 - cap / base_notional) * 100
            return cap
        stats["not_binding"] += 1
        return base_notional
    return fn


def summarize(trades: pd.DataFrame, nav_series: pd.Series) -> dict:
    if nav_series.empty or len(nav_series) < 2:
        return {}
    nav_end = float(nav_series.iloc[-1])
    total_return = (nav_end / STARTING_CAPITAL - 1) * 100
    years = max((nav_series.index[-1] - nav_series.index[0]).days / 365.25, 1e-9)
    cagr = (nav_end / STARTING_CAPITAL) ** (1 / years) - 1
    daily_ret = nav_series.pct_change().dropna()
    ann_factor = np.sqrt(252)
    mean_d, std_d = (float(daily_ret.mean()), float(daily_ret.std(ddof=1))) if len(daily_ret) > 1 else (float("nan"), float("nan"))
    sharpe = (mean_d / std_d) * ann_factor if std_d and std_d > 0 else float("nan")
    downside = daily_ret[daily_ret < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = (mean_d / downside_std) * ann_factor if downside_std and downside_std > 0 else float("nan")
    running_max = nav_series.cummax()
    max_dd = float(((nav_series / running_max - 1) * 100).min())
    n_trades = 0 if trades.empty else len(trades)
    return {"cagr_pct": cagr * 100, "total_return_pct": total_return, "max_dd_pct": max_dd,
            "sharpe": sharpe, "sortino": sortino, "n_trades": n_trades}


def run_scenario(name: str, states_plain: dict, states_adv: dict, entry_filter_fn) -> dict:
    spread = SPREAD_BPS[name] / 10000
    entry_cost = ENTRY_COST + spread
    exit_cost = EXIT_COST + spread

    liq_stats = {"capped": 0, "not_binding": 0, "insufficient_history": 0, "capped_reduction_pct_sum": 0.0}
    base_sizing_fn = make_sizing_fn("EQUAL-DOLLAR", 0.0, 0.0)

    if name in LIQUIDITY_CAPPED:
        states_used = states_adv
        sizing_fn = make_liquidity_capped_sizing_fn(base_sizing_fn, PARTICIPATION_RATE, liq_stats)
    else:
        states_used = states_plain
        sizing_fn = base_sizing_fn

    trades, nav = run_control(states_used, entry_filter_fn, sizing_fn, stop_mode="FIXED",
                               entry_cost=entry_cost, exit_cost=exit_cost)
    summary = summarize(trades, nav)
    summary["cost_bps_round_trip"] = (entry_cost + exit_cost) * 10000

    if name in LIQUIDITY_CAPPED:
        total_calls = liq_stats["capped"] + liq_stats["not_binding"] + liq_stats["insufficient_history"]
        summary["n_entry_signals"] = total_calls
        summary["n_capped"] = liq_stats["capped"]
        summary["pct_capped"] = (liq_stats["capped"] / total_calls * 100) if total_calls else float("nan")
        summary["avg_reduction_pct_when_capped"] = (
            liq_stats["capped_reduction_pct_sum"] / liq_stats["capped"] if liq_stats["capped"] else float("nan")
        )
        summary["n_insufficient_history"] = liq_stats["insufficient_history"]

    return summary


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    states_plain = build_states(price)
    states_adv = _states_with_adv(states_plain)
    entry_filter_fn = ENTRY_SETS["BASELINE"]

    scenario_order = ["BASELINE", "MODERATE_SPREAD", "WIDE_SPREAD", "LIQUIDITY_CAP", "COMBINED_STRESS"]
    results = {name: run_scenario(name, states_plain, states_adv, entry_filter_fn) for name in scenario_order}

    print("=" * 108)
    print("Cost and liquidity sensitivity -- five predeclared scenarios, control's entry/exit/stop logic unchanged in all five")
    print("=" * 108)
    for name in scenario_order:
        s = results[name]
        extra = ""
        if name in LIQUIDITY_CAPPED:
            extra = (f"  capped={s['n_capped']:>4}/{s['n_entry_signals']:<4} ({s['pct_capped']:>5.1f}%)  "
                     f"avg_reduction={s['avg_reduction_pct_when_capped']:>5.1f}%")
        print(f"  {name:<17} cost_rt={s['cost_bps_round_trip']:>5.1f}bps  n_trades={s['n_trades']:>4}  "
              f"cagr={s['cagr_pct']:>6.2f}%  sharpe={s['sharpe']:>5.2f}  sortino={s['sortino']:>5.2f}  "
              f"max_dd={s['max_dd_pct']:>7.2f}%{extra}")

    baseline = results["BASELINE"]
    print("\nchange vs. BASELINE (each scenario independently, not combined into one verdict):")
    for name in scenario_order[1:]:
        s = results[name]
        cagr_delta = s["cagr_pct"] - baseline["cagr_pct"]
        dd_delta = s["max_dd_pct"] - baseline["max_dd_pct"]
        print(f"  {name:<17} CAGR {cagr_delta:+.3f}pp   max_dd {dd_delta:+.3f}pp (negative = worse/deeper)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "predeclared_scenarios": scenario_order,
        "adv_window_days": ADV_WINDOW,
        "participation_rate": PARTICIPATION_RATE,
        "spread_bps_each_way": SPREAD_BPS,
        "results": results,
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
