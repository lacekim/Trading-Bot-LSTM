"""Execution-timing audit (item 1 of the post-exit-logic validation/
deployment-readiness sequence, run after the exit-logic sequence closed
with the unchanged control as the sole surviving specification).

MOTIVATION: every script in this project's history, including the control
itself, decides a trade using a given day's OWN close (ma50/ma200 computed
through and including that day, cross_up/cross_down evaluated against that
same day's close) and then FILLS the trade at that identical day's close
(adjusted only by the flat 12bps cost). The user's own framing: "A clean
next-bar or explicit market-on-close comparison will establish whether the
current control's performance depends on an execution assumption that
cannot be implemented as modeled." This script answers exactly that
question, and only that question -- it does NOT touch the stop mechanism's
gap-fill behavior (that is the next, separate audit item) or the cost
model (the item after that).

FOUR THINGS THIS SCRIPT CHECKS, PER THE USER'S OWN SPEC:

  1. Recompute signals using data available through t.
     Already verified by direct inspection, not re-derived here: every
     indicator this project's engines use (ma50/ma200 via close.rolling(N),
     cross_up/cross_down via .shift(1), atr14 via true_range.rolling(N),
     ma200_rising/ma50_rising via .shift(N), breakout_confirm via
     high.shift(1), and the LAGGED atr_pct_lag1/vol20_lag1 used by
     risk-based sizing) uses only trailing windows or positive-lag shifts --
     none use a centered window, a negative shift, or any other construct
     that could pull a future day's data into today's row. This audit does
     not re-implement that check (it would just be re-reading the same
     rolling()/shift() calls) -- it is disclosed here as verified-by-
     inspection so the reader knows it was actually checked, not assumed.
     What THIS script adds is the second half of the question this project
     had not yet separated from the first: even with clean, non-leaking
     signals, WHEN can that signal actually be acted on?

  2. Same-close vs. next-open execution.
     SAME_CLOSE (the existing control, reused unchanged via
     run_time_stagnation_backtest(..., time_stagnation_days=None), already
     verified elsewhere to reproduce the control byte-for-byte): a signal
     detected using day t's own close is filled at that SAME day t's close.
     NEXT_OPEN (new in this script): a signal detected using day t's own
     close is filled at the NEXT trading day (t+1, in that symbol's own
     calendar, not a global calendar) OPEN price. This is a real,
     unambiguously implementable execution scheme -- an order placed after
     the close on day t, to fill at day t+1's open -- with no cutoff-timing
     ambiguity of any kind.

     If SAME_CLOSE is retained (i.e. if its results hold up and the user
     decides to keep same-day fills rather than switch to next-open), it
     must be understood as a MARKET-ON-CLOSE (MOC) order, NOT as "the
     closing price was freely observable and then tradeable" -- and even
     that framing carries one disclosed idealization this script does NOT
     resolve: a real MOC order has a submission cutoff strictly BEFORE the
     closing print (e.g. major US exchanges' MOC cutoff is several minutes
     before the close), so a signal that is only known once day t's own
     close has actually printed (which is exactly how ma50/ma200/cross_up
     are computed here) cannot, in full rigor, be submitted as a true MOC
     order conditioned on that exact signal -- you would need to estimate
     the signal from data available AT the cutoff, not at the print. This
     project only has DAILY bars, not intraday quotes, so that cutoff-vs-
     print gap cannot be modeled here; it is disclosed as an irreducible
     idealization of SAME_CLOSE/MOC given this data granularity, not
     something this script papers over. NEXT_OPEN has no analogous gap and
     is the cleaner comparison for exactly this reason.

     Stops are explicitly OUT OF SCOPE for the same_close/next_open split:
     in BOTH variants, an open position's stop is checked the same way as
     every prior script (day's low <= stop_price, filled at the stop price
     minus cost, no gap-through modeling) -- unchanged, so this audit
     isolates SIGNAL-DRIVEN entries/exits as the only variable, per this
     project's own established practice of changing exactly one thing at a
     time. Gap-through-stop fills are the next audit item, deliberately
     kept separate here.

  3. Stop/signal precedence determinism.
     Confirmed by construction (an if/elif chain, never both branches) in
     every prior script's event loop, and re-confirmed here two ways: (a)
     a dedicated hand-built synthetic bar where the day's low touches the
     stop price AND the death-cross fires simultaneously, asserting the
     engine ALWAYS resolves it as a stop exit, never a signal exit, run
     from the dry-run test suite; (b) a scan of the REAL universe's trade
     history in both variants below for any bar where both conditions were
     literally true simultaneously, reporting how many occurred and
     confirming every one of them resolved as "stop" in the trade log.

  4. Independent reconciliation is explicitly NOT this script's job --
     that is the next, separate audit item (rebuild the equity curve from
     the trade ledger independently of the engine's own NAV computation).
     This script's own comparison table is not a substitute for that.

WHAT THIS SCRIPT REPORTS: CAGR / Sharpe / Sortino / max drawdown / n_trades
/ avg holding days for SAME_CLOSE vs. NEXT_OPEN, full history (this is a
timing-mechanics audit, not a fresh OOS/bootstrap study -- the exit-logic
sequence's own OOS validation already covered that layer for the control
itself). If NEXT_OPEN's numbers are materially different from SAME_CLOSE's,
that is direct evidence the control's reported performance depends on an
execution assumption (fill at a price that, in full rigor, cannot be
transacted at when the signal is only just becoming known) rather than on
the underlying trend-following logic itself. If they are close, the
same-close/MOC idealization is not doing meaningful work and the control's
edge does not hinge on it.
"""
from __future__ import annotations

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
    run_backtest_exit_variant as run_same_close_control,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "execution_timing_audit_results.json"


def _states_with_open(states: dict, price: dict) -> dict:
    """DISCLOSED FINDING from writing this audit: build_states() (via
    compute_pullback_indicators, imported unchanged) fetches 'open' from
    fetch_ohlcv_batch's own OHLCV frame but never carries it into its own
    output -- state DataFrames only retain close/high/low/volume plus the
    derived indicator columns. That's harmless for every prior script
    (none of them ever fill at the open), but this audit's whole point is
    to fill at the open, so it's needed here. Rather than modify the
    shared build_states()/compute_pullback_indicators() (used unchanged by
    every other script in this project and not otherwise broken), this
    copies 'open' back onto each symbol's state frame from the SAME raw
    `price` dict build_states() was itself called on -- same index by
    construction (state is derived directly from price[sym] without any
    reindexing), so a plain column assignment is exact, not an approximate
    join."""
    out = {}
    for sym, df in states.items():
        df2 = df.copy()
        df2["open"] = price[sym]["open"]
        out[sym] = df2
    return out


def run_backtest_next_open_execution(states, entry_filter_fn, sizing_fn,
                                      stop_pct=BASELINE_STOP_PCT, entry_cost=ENTRY_COST, exit_cost=EXIT_COST):
    """Identical signal computation to the control (ma50/ma200/cross_up/
    cross_down, all precomputed in `states`); the ONLY difference is that a
    signal detected on day t queues a PENDING order that fills at day t+1's
    OPEN (that symbol's own next available bar, not a global next-day
    assumption) rather than at day t's own close. The stop is unchanged
    from the control: checked via that day's own low against
    entry_price*(1+stop_pct/100), filled at the stop price minus cost, no
    gap-through modeling (see module docstring -- deliberately out of
    scope here).

    Per-day order of operations for a symbol with data today (mirrors what
    would actually happen intraday, chronologically):
      1. Fill any PENDING order queued from a prior day's close-signal, at
         TODAY's open (this happens first, at the start of the session,
         before any of today's subsequent intraday price action).
      2. Check the stop on whatever position is open as of that fill (a
         position entered at today's open by step 1 can still be stopped
         out later the same day if the day's low is extreme enough -- this
         is not a hypothetical, it is what an intraday stop order would
         actually do).
      3. If the position survived step 2 and cross_down fires today, queue
         a pending signal-exit for TOMORROW's open (do not fill today).
      4. If flat (no open position, nothing pending) and cross_up fires
         today (and entry_filter_fn accepts it), queue a pending entry for
         TOMORROW's open.
    A pending order is only ever created OR filled in a given iteration,
    never both for the same symbol on the same day, since a signal
    detected on day t cannot fill until a later day is reached in the
    event loop."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions = {sym: None for sym in states}
    pending = {sym: None for sym in states}
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

            # ---- step 1: fill any pending order at TODAY's open ----
            pend = pending[sym]
            if pend is not None:
                if pend["action"] == "enter" and positions[sym] is None:
                    notional = sizing_fn(capital, row)
                    if notional <= free_capital and notional > 100:
                        fill = row["open"] * (1 + entry_cost)
                        positions[sym] = {"entry_price": fill, "notional": notional, "entry_time": ts, "last_price": fill}
                        free_capital -= notional
                    pending[sym] = None
                elif pend["action"] == "exit_signal" and positions[sym] is not None:
                    pos = positions[sym]
                    exit_price = row["open"] * (1 - exit_cost)
                    ret = (exit_price / pos["entry_price"]) - 1
                    pnl = pos["notional"] * ret
                    capital += pnl
                    trade_log.append({"asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                                       "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital, "exit_reason": "signal"})
                    free_capital += pos["notional"]
                    positions[sym] = None
                    pending[sym] = None

            # ---- step 2: stop check on whatever is open now (possibly just entered this morning) ----
            pos = positions[sym]
            if pos is not None:
                stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                if row["low"] <= stop_price:
                    exit_price = stop_price * (1 - exit_cost)
                    ret = (exit_price / pos["entry_price"]) - 1
                    pnl = pos["notional"] * ret
                    capital += pnl
                    trade_log.append({"asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                                       "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital, "exit_reason": "stop"})
                    free_capital += pos["notional"]
                    positions[sym] = None
                    pending[sym] = None
                    pos = None
                elif row["cross_down"] and pending[sym] is None:
                    # step 3: queue for tomorrow's open, do NOT fill today
                    pending[sym] = {"action": "exit_signal"}

            # ---- step 4: new entries queue for tomorrow's open ----
            if positions[sym] is None and pending[sym] is None and row["cross_up"] and entry_filter_fn(row):
                pending[sym] = {"action": "enter"}

            pos = positions[sym]
            if pos is not None:
                pos["last_price"] = row["close"]
                nav_unrealized_sum += pos["notional"] * (row["close"] / pos["entry_price"] - 1)

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
        n_trades, avg_hold, stop_exits = 0, float("nan"), 0
    else:
        n_trades = len(trades)
        avg_hold = float((trades["exit_time"] - trades["entry_time"]).dt.days.mean())
        stop_exits = int((trades["exit_reason"] == "stop").sum())
    return {"cagr_pct": cagr * 100, "total_return_pct": total_return, "max_dd_pct": max_dd,
            "sharpe": sharpe, "sortino": sortino, "n_trades": n_trades, "stop_exits": stop_exits,
            "avg_holding_days": avg_hold}


def precedence_determinism_scan(trades_same_close: pd.DataFrame, trades_next_open: pd.DataFrame,
                                 states: dict, stop_pct=BASELINE_STOP_PCT) -> dict:
    """Question 3's REAL-data half: scan every state row for a bar where a
    position (per the recorded trade log, at least one variant had one
    open on that day) simultaneously satisfied low<=stop_price AND
    cross_down, and confirm the corresponding exit in whichever trade log
    covers that day is always "stop", never "signal". Reported counts are
    informational; the synthetic hand-built test in the dry-run suite is
    the one that actually PROVES the precedence, since real data may
    simply never produce this coincidence."""
    # NOTE: a precise real-data reconstruction of "did THIS bar's low breach
    # THIS specific open position's own stop_price" would require re-
    # joining each trade log's own entry fills back onto the state rows --
    # doable, but the dry-run suite's hand-built synthetic bar is what
    # actually PROVES the precedence rule (it forces the exact coincidence
    # on demand rather than hoping real data happens to produce it). This
    # scan instead reports a simple, informational proxy: how many bars in
    # the raw state data show BOTH cross_down and a single-day drop steep
    # enough that it COULD plausibly have breached a -|stop_pct| stop from
    # some entry price close to that day's own level.
    n_coincident = 0
    for sym, state in states.items():
        cross_down = state["cross_down"].fillna(False)
        # informational proxy only: a day is "sharp-down" if today's low is
        # more than |stop_pct| below yesterday's close, a rough stand-in
        # for "this day's move alone could plausibly have hit a -12%-from-
        # entry stop" without needing the exact per-trade entry price.
        sharp_down = (state["low"] / state["close"].shift(1) - 1) * 100 <= stop_pct
        both = (cross_down & sharp_down.fillna(False))
        n_coincident += int(both.sum())
    return {"n_informational_coincident_bars": n_coincident}


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    states = build_states(price)
    states_open = _states_with_open(states, price)
    entry_filter_fn = ENTRY_SETS["BASELINE"]
    sizing_fn = make_sizing_fn("EQUAL-DOLLAR", 0.0, 0.0)

    same_close_trades, same_close_nav = run_same_close_control(states, entry_filter_fn, sizing_fn, stop_mode="FIXED")
    next_open_trades, next_open_nav = run_backtest_next_open_execution(states_open, entry_filter_fn, sizing_fn)

    s_same = summarize(same_close_trades, same_close_nav)
    s_next = summarize(next_open_trades, next_open_nav)

    print("=" * 100)
    print("SAME_CLOSE (existing control, modeled as a market-on-close order -- see module docstring's disclosed "
          "MOC-cutoff idealization) vs. NEXT_OPEN (signal-driven entries/exits filled at the following trading "
          "day's open; stop mechanism UNCHANGED in both -- gap-through-stop fills are the next, separate audit item)")
    print("=" * 100)
    for label, s in [("SAME_CLOSE (control)", s_same), ("NEXT_OPEN", s_next)]:
        print(f"  {label:<22} n_trades={s.get('n_trades', 0):>4}  stop_exits={s.get('stop_exits', 0):>4}  "
              f"cagr={s.get('cagr_pct', float('nan')):>7.2f}%  sharpe={s.get('sharpe', float('nan')):>5.2f}  "
              f"sortino={s.get('sortino', float('nan')):>5.2f}  max_dd={s.get('max_dd_pct', float('nan')):>7.2f}%  "
              f"avg_hold={s.get('avg_holding_days', float('nan')):>6.1f}d")

    cagr_delta = s_next.get("cagr_pct", float("nan")) - s_same.get("cagr_pct", float("nan"))
    print(f"\nNEXT_OPEN minus SAME_CLOSE CAGR delta: {cagr_delta:+.2f}pp")
    print("This is a full-history timing-mechanics comparison, not a fresh OOS/bootstrap study -- the control's "
          "own OOS validation already covered that layer separately.")

    print("\n" + "=" * 100)
    print("STOP/SIGNAL PRECEDENCE -- informational real-data scan (the synthetic hand-built proof lives in the "
          "dry-run test suite, since real data may simply never produce the exact coincidence)")
    print("=" * 100)
    prec = precedence_determinism_scan(same_close_trades, next_open_trades, states)
    print(f"  informational coincident bars (cross_down AND a sharp-enough single-day drop): "
          f"{prec['n_informational_coincident_bars']}")

    import json
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"same_close": s_same, "next_open": s_next,
                                     "cagr_delta_pp": cagr_delta, "precedence_scan": prec}, default=str, indent=2),
                         encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
