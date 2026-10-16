"""Gap-through-stop model (item 2 of the post-exit-logic validation/
deployment-readiness sequence). Item 1 (execution-timing audit) just
showed NEXT_OPEN and SAME_CLOSE are statistically indistinguishable on the
full 60-symbol universe (CAGR delta +0.00pp, identical Sharpe/Sortino,
574 vs. 578 stop exits) -- the control's performance does NOT depend on
the same-close/MOC execution assumption. Given that, this item is built
as a variant of the SAME_CLOSE control specifically (not NEXT_OPEN), to
keep changing exactly one thing at a time per this project's own
established practice; item 1's result means this choice is not expected
to matter, and can be revisited if a future OOS study on item 1 ever
shows otherwise.

THE GAP, AS DISCLOSED IN EVERY PRIOR SCRIPT: every stop exit in every
script so far fills at EXACTLY entry_price*(1+stop_pct/100), adjusted only
by the flat exit cost -- regardless of how far the actual bar undershoots
that price. A name that gaps down 20% overnight through a -12% stop is
still costed as if it filled at exactly -12%. This script fixes that,
per the user's own exact spec:

  - If the session's OPEN is already at or below the stop price, the stop
    order cannot get its own price -- it fills at the OPEN instead (a
    resting stop-market order becomes a market order the instant the
    market opens through it).
  - If the session's open is still above the stop price but the day's LOW
    subsequently touches or crosses it, the fill is UNCHANGED from every
    prior script: exactly the stop price (a limit-like, optimistic
    assumption for an intraday touch, which this item does not attempt to
    improve -- only the GAP case is being fixed here, per the user's own
    scope).

PRECEDENCE: unchanged from every prior script. The gap-check is simply the
new FIRST branch of the same stop check (a gap-open below the stop is
still fundamentally a stop event, just filled differently) -- if a gap-
open-stop and a death-cross signal coincide on the same bar, the stop
(gap or not) still wins, exactly as before. This is verified in the
dry-run suite by forcing the exact coincidence, mirroring item 1's own
precedence proof.

WHAT'S REPORTED, per the user's own spec: the number of gap exits (vs.
ordinary intraday-touch stop exits), the average slippage beyond the
nominal stop price (in percentage points of return, computed only over
the gap exits -- an ordinary intraday-touch stop exit has zero slippage
by construction, so including it would just dilute the average with
exact zeros rather than being wrong, but it is reported separately here
so the number is precisely "how bad were the gaps that happened," not
"how bad are stops on average"), and the resulting CAGR/max-drawdown
change vs. the OLD (optimistic, exact-stop-price) control.

OUT OF SCOPE here (left for the next two audit items): the entry-timing
question (already answered by item 1); the flat 12bps cost model itself
(item 3); independent trade-ledger reconciliation (item 4); universe/
concentration diagnostics (item 5)."""
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
    run_backtest_exit_variant as run_optimistic_stop_control,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "gap_through_stop_results.json"


def _states_with_open(states: dict, price: dict) -> dict:
    """Same disclosed fix as run_execution_timing_audit_equities.py's own
    _states_with_open: build_states() drops 'open' from its output even
    though fetch_ohlcv_batch provides it; copied back here from the same
    raw price dict build_states() was called on (identical index by
    construction, so this is an exact column assignment, not a join)."""
    out = {}
    for sym, df in states.items():
        df2 = df.copy()
        df2["open"] = price[sym]["open"]
        out[sym] = df2
    return out


def run_backtest_gap_aware_stop(states, entry_filter_fn, sizing_fn,
                                 stop_pct=BASELINE_STOP_PCT, entry_cost=ENTRY_COST, exit_cost=EXIT_COST):
    """Identical to the control (immediate entry at that day's close on
    cross_up, death-cross exit at that day's close, same-close timing
    throughout) EXCEPT for the stop fill: a stop event first checks
    whether the day's OPEN already gapped past the stop price (fills at
    the open, exit_reason='stop_gap', slippage_pp recorded) before falling
    back to the ORIGINAL exact-stop-price fill for an ordinary intraday
    touch (exit_reason='stop', slippage_pp=0.0 by construction)."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions = {sym: None for sym in states}
    trade_log = []
    nav_history = []

    for ts in all_index:
        nav_unrealized_sum = 0.0
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional

        for sym, state in states.items():
            if ts not in state.index:
                pos = positions[sym]
                if pos is not None:
                    nav_unrealized_sum += pos["notional"] * (pos["last_price"] / pos["entry_price"] - 1)
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                pos = positions[sym]
                if pos is not None:
                    nav_unrealized_sum += pos["notional"] * (pos["last_price"] / pos["entry_price"] - 1)
                continue
            pos = positions[sym]

            if pos is not None:
                stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                if row["open"] <= stop_price:
                    exit_price = row["open"] * (1 - exit_cost)
                    reason = "stop_gap"
                    nominal_stop_exit_price = stop_price * (1 - exit_cost)
                    slippage_pp = (exit_price / pos["entry_price"] - nominal_stop_exit_price / pos["entry_price"]) * 100
                elif row["low"] <= stop_price:
                    exit_price = stop_price * (1 - exit_cost)
                    reason = "stop"
                    slippage_pp = 0.0
                elif row["cross_down"]:
                    exit_price = row["close"] * (1 - exit_cost)
                    reason = "signal"
                    slippage_pp = float("nan")
                else:
                    pos["last_price"] = row["close"]
                    nav_unrealized_sum += pos["notional"] * (row["close"] / pos["entry_price"] - 1)
                    continue
                ret = (exit_price / pos["entry_price"]) - 1
                pnl = pos["notional"] * ret
                capital += pnl
                trade_log.append({"asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                                   "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital,
                                   "exit_reason": reason, "slippage_pp": slippage_pp})
                positions[sym] = None
                free_capital += pos["notional"]
                continue

            if row["cross_up"] and entry_filter_fn(row):
                notional = sizing_fn(capital, row)
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + entry_cost)
                    positions[sym] = {"entry_price": fill, "notional": notional, "entry_time": ts, "last_price": fill}
                    free_capital -= notional

        nav_t = capital + nav_unrealized_sum
        nav_history.append((ts, nav_t))

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    nav_series = pd.Series({t: v for t, v in nav_history}).sort_index() if nav_history else pd.Series(dtype=float)
    return trades, nav_series


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
    if trades.empty:
        n_trades, n_stop, n_gap, avg_slip = 0, 0, 0, float("nan")
    else:
        n_trades = len(trades)
        n_stop = int((trades["exit_reason"] == "stop").sum())
        gap_rows = trades[trades["exit_reason"] == "stop_gap"]
        n_gap = len(gap_rows)
        avg_slip = float(gap_rows["slippage_pp"].mean()) if n_gap else float("nan")
    return {"cagr_pct": cagr * 100, "total_return_pct": total_return, "max_dd_pct": max_dd,
            "sharpe": sharpe, "sortino": sortino, "n_trades": n_trades,
            "n_stop_exits_exact": n_stop, "n_stop_exits_gap": n_gap,
            "avg_slippage_beyond_stop_pp": avg_slip}


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    states = build_states(price)
    states_open = _states_with_open(states, price)
    entry_filter_fn = ENTRY_SETS["BASELINE"]
    sizing_fn = make_sizing_fn("EQUAL-DOLLAR", 0.0, 0.0)

    old_trades, old_nav = run_optimistic_stop_control(states, entry_filter_fn, sizing_fn, stop_mode="FIXED")
    new_trades, new_nav = run_backtest_gap_aware_stop(states_open, entry_filter_fn, sizing_fn)

    s_old = summarize(old_trades, old_nav)
    s_new = summarize(new_trades, new_nav)

    print("=" * 100)
    print("OLD (every prior script: exact stop-price fill regardless of gap size) vs. NEW (gap-aware: fills at "
          "the open when the session opens below the stop, unchanged exact-stop-price fill otherwise)")
    print("=" * 100)
    for label, s in [("OLD (optimistic, all scripts so far)", s_old), ("NEW (gap-aware)", s_new)]:
        print(f"  {label:<38} n_trades={s.get('n_trades', 0):>4}  stop_exact={s.get('n_stop_exits_exact', 0):>4}  "
              f"stop_gap={s.get('n_stop_exits_gap', 0):>4}  cagr={s.get('cagr_pct', float('nan')):>7.2f}%  "
              f"sharpe={s.get('sharpe', float('nan')):>5.2f}  sortino={s.get('sortino', float('nan')):>5.2f}  "
              f"max_dd={s.get('max_dd_pct', float('nan')):>7.2f}%")

    print(f"\naverage slippage beyond the nominal stop price, over gap exits only: "
          f"{s_new.get('avg_slippage_beyond_stop_pp', float('nan')):+.3f}pp of return per gap exit")
    cagr_delta = s_new.get("cagr_pct", float("nan")) - s_old.get("cagr_pct", float("nan"))
    dd_delta = s_new.get("max_dd_pct", float("nan")) - s_old.get("max_dd_pct", float("nan"))
    print(f"CAGR change (NEW minus OLD): {cagr_delta:+.3f}pp")
    print(f"max_dd change (NEW minus OLD, negative = worse/deeper): {dd_delta:+.3f}pp")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"old_optimistic": s_old, "new_gap_aware": s_new,
                                     "cagr_delta_pp": cagr_delta, "max_dd_delta_pp": dd_delta}, default=str, indent=2),
                         encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
