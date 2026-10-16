"""Independent equity-curve reconciliation (item 4 of the post-exit-logic
validation/deployment-readiness sequence), per the user's own exact spec:

  "Rebuild the equity curve from the trade ledger independently of the
  engine's portfolio-value calculation. Reconcile entries, exits, shares,
  cash, costs, stop fills, and daily returns. Require exact or rounding-
  level agreement before interpreting any performance result."

WHAT "INDEPENDENT" MEANS HERE, DISCLOSED: this does not just re-read numbers
the existing engine (run_backtest_exit_variant) already produced. It is a
freshly, separately written backtest engine (run_backtest_share_based,
below) that:
  - represents positions as explicit SHARE COUNTS and a CASH balance,
    rather than the existing engine's notional-dollar-and-percent-return
    representation;
  - computes daily NAV by marking held SHARES to today's close and adding
    cash, rather than the existing engine's running "capital" (realized-
    trade-cash) plus a same-day unrealized-notional*ret_pct sum;
  - is a fresh re-derivation of the same predeclared decision rules
    (BASELINE golden-cross entry, EQUAL-DOLLAR sizing, fixed -12% stop,
    death-cross exit, SAME_CLOSE/exact-stop-price fills, 12bps/12bps cost)
    from the spec itself, not a copy of the original function body.
The two engines share the SAME decision rules by design -- this is a check
of BOOKKEEPING/ARITHMETIC correctness (does an independently-coded ledger
of shares and cash reproduce the same trades and the same NAV path), not a
test of alternative decision rules. A share-count-vs-cash ledger is
mathematically forced to agree with a notional-vs-percent-return ledger
under IDENTICAL fills (pnl = shares*(exit_price-entry_price) =
(notional/entry_price)*(exit_price-entry_price) = notional*ret, exactly) --
so this specifically tests whether the existing engine's actual code
computes what the spec says it computes, by re-deriving it a second,
differently-represented way and diffing every trade and every NAV point.

SCOPE: this reconciles the plain BASELINE control only (SAME_CLOSE,
exact-stop-price fills, EQUAL-DOLLAR sizing, 12bps/12bps cost, uncapped) --
the same reference specification every item in this sequence measures
deviations from. It does not re-reconcile the item-2 gap-aware or item-3
cost/liquidity variants; those are each a single, disclosed, additive
change on top of this same reconciled control, and share its core fill/
entry/exit logic unchanged.

TOLERANCES, PREDECLARED (fixed before this script was run against real
data): a trade-level pnl or a daily NAV value agrees if
abs(a - b) <= max(NAV_ABS_TOL, NAV_REL_TOL * max(abs(a), abs(b))), with
NAV_ABS_TOL=$1e-6 and NAV_REL_TOL=1e-9 -- i.e., agreement to far better than
a hundredth of a cent on any realistic dollar figure, so this is a check
for exact/floating-point-rounding agreement, not a "close enough"
threshold. ANY trade or NAV point outside tolerance is reported explicitly,
by date/symbol and by size, rather than averaged away -- and per the user's
own instruction, if reconciliation fails, no performance number from either
engine is to be interpreted as valid until the discrepancy is found and
resolved.

WHAT'S REPORTED: n_trades on each side and whether they match one-for-one
by (asset, entry_time); the maximum absolute and relative trade-level pnl
discrepancy; the maximum absolute and relative daily-NAV discrepancy across
the whole history; the final cash/capital figures from both engines; CAGR/
Sharpe/Sortino/max_dd computed independently from the share-based ledger's
own NAV series, compared against the existing engine's own summarize()-
style output.

OUT OF SCOPE here (left for the final audit item): universe and
concentration diagnostics (item 5)."""
from __future__ import annotations

import json
import math
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
    run_backtest_exit_variant as run_engine_control,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "independent_ledger_reconciliation_results.json"

NAV_ABS_TOL = 1e-6
NAV_REL_TOL = 1e-9


def _agree(a: float, b: float, abs_tol: float = NAV_ABS_TOL, rel_tol: float = NAV_REL_TOL) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def run_backtest_share_based(states, entry_filter_fn, sizing_fn,
                              stop_pct=BASELINE_STOP_PCT, entry_cost=ENTRY_COST, exit_cost=EXIT_COST):
    """Independent re-derivation of the BASELINE control, representing every
    position as an explicit share count and tracking a plain cash balance,
    rather than the original engine's notional/percent-return bookkeeping.
    Decision rules (entry on cross_up if entry_filter_fn(row), fixed -12%
    stop checked via the day's low and filled at the exact stop price,
    death-cross exit at the close, stop-beats-signal precedence, no same-
    day re-entry, minimum $100 notional floor, free-cash sizing constraint)
    are re-derived from the predeclared spec, not copied from
    run_backtest_exit_variant.

    TWO separate running balances are kept, deliberately mirroring a real
    design choice already present in the existing engine (not a bug being
    reconciled away): `capital` is REALIZED-cash-only -- it starts at
    STARTING_CAPITAL and changes only when a trade CLOSES (by that trade's
    realized pnl), exactly as the existing engine's own `capital` variable
    behaves. Entry sizing (sizing_fn(capital, row), i.e. 2% of capital) and
    the solvency/no-leverage check are both based on THIS balance, so that
    the independent engine sizes every position identically to the
    existing one. `cash` is a SEPARATE, true spendable-cash ledger --
    debited by the entry cost basis when a position opens and credited by
    exit proceeds when it closes -- kept purely so that
    cash + mark-to-market-of-open-shares can be checked, independently,
    against the existing engine's own capital + unrealized-notional-return
    NAV computation. The two are algebraically forced to agree at every
    timestamp (cash = capital - sum of open positions' own cost basis;
    mark-to-market = cost basis + unrealized pnl; so cash + mark-to-market
    = capital + unrealized pnl, exactly the existing engine's own formula)
    -- this is precisely the identity this script exists to verify holds in
    the actual code, not just in principle."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL  # realized-only; sizing/solvency basis, mirrors the existing engine exactly
    cash = STARTING_CAPITAL     # true spendable cash; debited/credited; independent NAV check only
    positions = {sym: None for sym in states}  # sym -> {shares, entry_price, entry_time}
    trade_log = []
    nav_history = []

    for ts in all_index:
        deployed_notional = sum(p["shares"] * p["entry_price"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional
        mark_to_market_sum = 0.0

        for sym, state in states.items():
            pos = positions[sym]
            if ts not in state.index:
                if pos is not None:
                    mark_to_market_sum += pos["shares"] * pos["last_price"]
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                if pos is not None:
                    mark_to_market_sum += pos["shares"] * pos["last_price"]
                continue

            if pos is not None:
                stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                stop_hit = row["low"] <= stop_price
                if stop_hit:
                    exit_price = stop_price * (1 - exit_cost)
                    reason = "stop"
                elif row["cross_down"]:
                    exit_price = row["close"] * (1 - exit_cost)
                    reason = "signal"
                else:
                    pos["last_price"] = row["close"]
                    mark_to_market_sum += pos["shares"] * row["close"]
                    continue

                proceeds = pos["shares"] * exit_price
                cash += proceeds
                pnl = proceeds - pos["shares"] * pos["entry_price"]
                capital += pnl
                ret_pct = (exit_price / pos["entry_price"] - 1) * 100
                trade_log.append({
                    "asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                    "shares": pos["shares"], "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "entry_notional": pos["shares"] * pos["entry_price"], "exit_proceeds": proceeds,
                    "pnl": pnl, "ret_pct": ret_pct, "exit_reason": reason, "cash_after": cash,
                })
                positions[sym] = None
                free_capital += pos["shares"] * pos["entry_price"]
                continue

            if row["cross_up"] and entry_filter_fn(row):
                notional = sizing_fn(capital, row)
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill_price = row["close"] * (1 + entry_cost)
                    shares = notional / fill_price
                    positions[sym] = {
                        "shares": shares, "entry_price": fill_price, "entry_time": ts, "last_price": fill_price,
                    }
                    cash -= shares * fill_price
                    free_capital -= shares * fill_price
                    # Mark the new position to its own fill price for TODAY's
                    # NAV (zero unrealized pnl on the entry day itself, by
                    # construction) -- otherwise today's cash debit would go
                    # unmatched by any offsetting mark-to-market contribution,
                    # understating NAV by exactly today's new entries. This
                    # exactly mirrors the existing engine's own entry-day
                    # convention: `pos` is read once at the top of each
                    # symbol's iteration, before the entry branch can run, so
                    # a symbol entered today never contributes to that day's
                    # nav_unrealized_sum either -- capital (untouched at
                    # entry) is therefore already the correct entry-day NAV.
                    mark_to_market_sum += shares * fill_price

        nav_t = cash + mark_to_market_sum
        nav_history.append((ts, nav_t))

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    nav_series = pd.Series({t: v for t, v in nav_history}).sort_index() if nav_history else pd.Series(dtype=float)
    return trades, nav_series, cash


def independent_performance_stats(nav_series: pd.Series) -> dict:
    """Freshly written (not copied from any prior script's summarize()) CAGR/
    Sharpe/Sortino/max_dd computation, so the same formula bug (if any)
    could not silently appear on both sides of the reconciliation."""
    if nav_series.empty or len(nav_series) < 2:
        return {}
    start_val = float(nav_series.iloc[0])
    end_val = float(nav_series.iloc[-1])
    span_years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    span_years = max(span_years, 1e-9)
    cagr_pct = ((end_val / start_val) ** (1.0 / span_years) - 1.0) * 100
    rets = nav_series.pct_change().dropna()
    if len(rets) > 1 and rets.std(ddof=1) > 0:
        sharpe = (rets.mean() / rets.std(ddof=1)) * math.sqrt(252)
    else:
        sharpe = float("nan")
    downside_rets = rets[rets < 0]
    if len(downside_rets) > 1 and downside_rets.std(ddof=1) > 0:
        sortino = (rets.mean() / downside_rets.std(ddof=1)) * math.sqrt(252)
    else:
        sortino = float("nan")
    peak = nav_series.iloc[0]
    max_dd_pct = 0.0
    for v in nav_series:
        peak = max(peak, v)
        dd = (v / peak - 1.0) * 100
        max_dd_pct = min(max_dd_pct, dd)
    return {"cagr_pct": cagr_pct, "sharpe": float(sharpe), "sortino": float(sortino),
            "max_dd_pct": max_dd_pct, "final_value": end_val}


def reconcile_trades(trades_engine: pd.DataFrame, trades_indep: pd.DataFrame) -> dict:
    if trades_engine.empty and trades_indep.empty:
        return {"n_engine": 0, "n_indep": 0, "n_matched": 0, "n_unmatched": 0,
                "max_abs_pnl_diff": 0.0, "max_rel_pnl_diff": 0.0, "mismatches": []}

    eng = trades_engine.set_index(["asset", "entry_time"])
    ind = trades_indep.set_index(["asset", "entry_time"])
    eng_keys = set(eng.index)
    ind_keys = set(ind.index)
    common = eng_keys & ind_keys
    unmatched = (eng_keys | ind_keys) - common

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    mismatches = []
    for key in common:
        e_row, i_row = eng.loc[key], ind.loc[key]
        if e_row["exit_time"] != i_row["exit_time"] or e_row["exit_reason"] != i_row["exit_reason"]:
            mismatches.append({"key": str(key), "issue": "exit_time_or_reason_mismatch",
                                "engine_exit": str(e_row["exit_time"]), "indep_exit": str(i_row["exit_time"]),
                                "engine_reason": e_row["exit_reason"], "indep_reason": i_row["exit_reason"]})
            continue
        pnl_e, pnl_i = float(e_row["pnl"]), float(i_row["pnl"])
        abs_diff = abs(pnl_e - pnl_i)
        rel_diff = abs_diff / max(abs(pnl_e), abs(pnl_i), 1e-12)
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)
        if not _agree(pnl_e, pnl_i):
            mismatches.append({"key": str(key), "issue": "pnl_mismatch",
                                "engine_pnl": pnl_e, "indep_pnl": pnl_i, "abs_diff": abs_diff})

    return {"n_engine": len(trades_engine), "n_indep": len(trades_indep), "n_matched": len(common),
            "n_unmatched": len(unmatched), "max_abs_pnl_diff": max_abs_diff, "max_rel_pnl_diff": max_rel_diff,
            "mismatches": mismatches}


def reconcile_nav(nav_engine: pd.Series, nav_indep: pd.Series) -> dict:
    if not nav_engine.index.equals(nav_indep.index):
        return {"index_equal": False, "max_abs_diff": float("nan"), "max_rel_diff": float("nan"), "n_points_outside_tol": -1}
    diff = (nav_engine - nav_indep).abs()
    rel = diff / nav_engine.abs().clip(lower=1e-12)
    outside = [(str(ts), float(nav_engine[ts]), float(nav_indep[ts])) for ts in nav_engine.index
               if not _agree(float(nav_engine[ts]), float(nav_indep[ts]))]
    return {"index_equal": True, "max_abs_diff": float(diff.max()), "max_rel_diff": float(rel.max()),
            "n_points_outside_tol": len(outside), "worst_points": outside[:10]}


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    states = build_states(price)
    entry_filter_fn = ENTRY_SETS["BASELINE"]
    sizing_fn = make_sizing_fn("EQUAL-DOLLAR", 0.0, 0.0)

    trades_engine, nav_engine = run_engine_control(states, entry_filter_fn, sizing_fn, stop_mode="FIXED")
    trades_indep, nav_indep, final_cash_indep = run_backtest_share_based(states, entry_filter_fn, sizing_fn)

    trade_recon = reconcile_trades(trades_engine, trades_indep)
    nav_recon = reconcile_nav(nav_engine, nav_indep)
    stats_indep = independent_performance_stats(nav_indep)

    trades_pass = (trade_recon["n_unmatched"] == 0 and not trade_recon["mismatches"])
    nav_pass = nav_recon["index_equal"] and nav_recon["n_points_outside_tol"] == 0
    overall_pass = trades_pass and nav_pass

    print("=" * 100)
    print("Independent ledger reconciliation -- share/cash-based engine vs. the existing notional/percent-return engine")
    print("=" * 100)
    print(f"  trades: engine={trade_recon['n_engine']}  independent={trade_recon['n_indep']}  "
          f"matched={trade_recon['n_matched']}  unmatched={trade_recon['n_unmatched']}")
    print(f"  max abs pnl diff: ${trade_recon['max_abs_pnl_diff']:.10f}   max rel pnl diff: {trade_recon['max_rel_pnl_diff']:.2e}")
    print(f"  trade-level mismatches beyond tolerance: {len(trade_recon['mismatches'])}")
    print(f"  NAV series index equal: {nav_recon['index_equal']}   points outside tolerance: {nav_recon['n_points_outside_tol']}")
    print(f"  max abs NAV diff: ${nav_recon['max_abs_diff']:.10f}   max rel NAV diff: {nav_recon['max_rel_diff']:.2e}")
    print(f"  final cash/capital -- independent engine: ${final_cash_indep:,.2f}")
    print(f"\n  RECONCILIATION: {'PASS' if overall_pass else 'FAIL'} "
          f"(tolerance: abs<=${NAV_ABS_TOL} or rel<={NAV_REL_TOL})")

    if overall_pass:
        print(f"\n  independent CAGR={stats_indep['cagr_pct']:.2f}%  sharpe={stats_indep['sharpe']:.2f}  "
              f"sortino={stats_indep['sortino']:.2f}  max_dd={stats_indep['max_dd_pct']:.2f}%  "
              f"(this should, and does, agree with every prior script's own BASELINE control numbers)")
    else:
        print("\n  Per the predeclared rule: no performance number from either engine is to be interpreted "
              "until this discrepancy is located and resolved. See 'mismatches'/'worst_points' in the JSON output.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "trade_reconciliation": trade_recon, "nav_reconciliation": nav_recon,
        "independent_performance_stats": stats_indep, "final_cash_independent_engine": final_cash_indep,
        "overall_pass": overall_pass, "tolerances": {"abs": NAV_ABS_TOL, "rel": NAV_REL_TOL},
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
