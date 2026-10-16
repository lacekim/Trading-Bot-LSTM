"""Partial profit-taking backtest -- the fourth substantive experiment in the
exit-logic research phase, and the last of the predeclared sequence after
ATR trailing stops, Chandelier exits, and time-based stagnation exits all
failed (or, for time-based exits, are still pending their first real run) to
beat the control by cutting short the strategy's rare large winners.

FROZEN CONTROL (identical to the live bot, and to all three prior studies):
  - BASELINE golden-cross entries, EQUAL-DOLLAR sizing, death-cross exit,
    fixed -12% stop (BASELINE_STOP_PCT), same costs/execution timing/
    capital rules/60-symbol universe. All imported unchanged from
    run_sizing_allocation_backtest_equities.py.

TREATMENT: an ADDITIONAL, one-time partial-sale trigger layered on top of
the UNCHANGED fixed -12% stop and death-cross (the SAME additive philosophy
established for the time-based-exit study, for the same reason: this asks
"does locking in part of a gain early help or hurt," not "is the stop
itself wrong" -- conflating the two would re-litigate the already-answered
ATR/Chandelier question). Confirmed with the user (this experiment's exact
design was not fully specified in the brief, unlike the ATR/Chandelier/
time-based grids, so it was proposed and confirmed before being frozen):

  PROFIT_TAKE_THRESHOLDS_PCT = (25, 50, 100)   -- PREDECLARED grid, 3
                                                   independent variants (as
                                                   first run)
  PARTIAL_SALE_FRACTION      = 0.5              -- fixed across all variants

GRID EXTENSION (post-hoc, at the user's explicit request after reviewing the
25/50/100 in-sample results and the closing synthesis across all four
exit-logic families): PROFIT_TAKE_THRESHOLDS_PCT now also includes 150 and
200. Not re-litigating an already-decided question -- 150/200 are NEW,
previously untested points, added because the 25->50->100 progression was
still climbing steeply toward the control (CAGR 6.14%->6.98%->7.73%,
control-winner pnl retained 24.1%->39.5%->60.2%) rather than visibly
plateauing. The original 25/50/100 results and verdicts are unchanged and
reported exactly as before; 150/200 are appended, not substituted.

For each variant, a position sells HALF its current notional the first time
its unrealized gain reaches the variant's threshold; the other half keeps
riding under the exact same fixed -12% stop and death-cross exit as the
control, unmodified. Each position can trigger the partial sale AT MOST
ONCE (no tiered/repeated scale-outs in this design).

TRIGGER AND FILL CONVENTION (deliberately mirrors the fixed stop's own
convention, for consistency): a threshold PRICE level is fixed at entry --
threshold_price = entry_price * (1 + threshold_pct / 100) -- exactly as the
stop fixes stop_price = entry_price * (1 + stop_pct / 100) at entry. The
trigger checks today's HIGH against that level (today's low checks the
stop; today's high checks the profit target -- the natural mirror), and
fills AT the threshold level itself, not wherever the day's high actually
reached (fill = threshold_price * (1 - exit_cost)), exactly as a gap-through
on the stop side fills at the stop price, not the deeper actual low. This is
NOT lookahead: threshold_price is fixed at entry from information known at
entry, and the check uses only today's own high, the same way the existing
stop check uses only today's own low.

PRECEDENCE: a FULL close (stop, then death-cross) is always checked FIRST
and takes precedence over the partial-profit trigger -- if either fires,
the position closes completely today and the partial-profit trigger is
never evaluated. Only when neither a stop nor a death-cross fires today is
the partial-profit trigger checked. This mirrors the already-established
"full/stronger signal beats a weaker one" ordering (stop > death-cross >
time-stagnation) by placing the partial, non-terminal action at the very
bottom of the same precedence chain.

WHAT THE FIXED STOP DOES **NOT** DO after a partial sale: the remaining
position's stop level is STILL stop_price = entry_price * (1 + stop_pct /
100) -- entirely unchanged. It is never re-based off the reduced notional,
a new reference price, or the partial-sale price. This is an explicit design
check (see CHECK 4 below) because re-basing the stop after a partial sale is
an easy, unintentional way to quietly change the control's risk model at the
same time as testing profit-taking, confounding the two questions.

TRADE-LOG GRANULARITY (a reporting convention specific to this script,
disclosed up front): a position that triggers its partial sale produces TWO
rows in trade_log -- one for the sold slice (exit_reason="partial_profit",
pnl/ret_pct/holding_days computed on THAT SLICE alone) and one for the
remainder's eventual close (exit_reason="stop"/"signal", on the remaining
notional). This is deliberate, not an artifact: reporting at the slice level
correctly shows how outcomes decompose into "the locked-in portion" and "the
remainder's fate" for functions that operate row-by-row (closed_trade_only_
stats, concentration_stats, calendar_year_breakdown, results_excluding_top_
k). The ONE place slice-level rows would be wrong is winner_preservation_
diagnostics, which matches trades against the control 1:1 by (asset,
entry_time) -- the control never splits, so a variant with a split position
would otherwise show 2 rows against the control's 1, breaking the join. To
fix this, winner_preservation_diagnostics runs _aggregate_to_position_level
first (sums pnl across a position's slices, keeps the terminal slice's
holding_days/mfe fields, recomputes ret_pct against the ORIGINAL full
notional) on both sides before matching -- a no-op for any non-split
position, and the only place this aggregation is applied.

SIX IMPLEMENTATION CHECKS (see run_backtest_exit_variant's inline comments):
  1. The partial-profit trigger uses only today's own high against a level
     FIXED AT ENTRY -- no lookahead, exactly like the existing stop check.
  2. Deterministic precedence: a full close (stop, then death-cross) is
     always resolved before the partial-profit trigger is ever evaluated.
  3. A position's partial sale fires AT MOST ONCE (a per-position flag
     guards this even if price pulls back below the threshold and later
     re-exceeds it).
  4. The remaining position's stop level is NEVER re-based after a partial
     sale -- stays entry_price * (1 + stop_pct/100), unchanged.
  5. Capital/notional bookkeeping is exact: the sold notional's proceeds are
     credited to capital and freed for redeployment immediately; the
     remaining notional continues to be marked to market correctly for NAV.
  6. A position exiting FULLY today (stop or death-cross, whether or not it
     already had a partial sale) cannot be re-entered later the same day.

METHODOLOGY DISCLOSED UP FRONT (identical conventions to the prior three
studies, for direct comparability):
  - Full-history, in-sample-only first pass.
  - "Closed-trade" and "NAV" results reported separately.
  - mfe_high seeds at max(entry-day high, cost-inclusive fill); NEVER reset
    by a partial sale (it is a whole-of-position diagnostic, carried across
    both slices of a split position).
  - Concentration percentages are suppressed (near-zero total OR a
    sane-bound-exceeding share) -- the same corrected logic from the ATR/
    Chandelier clean reruns, carried forward from the start.
  - avg_mfe_capture_pct excludes trades with mfe_pct < MFE_CAPTURE_MIN_
    MFE_PCT from the aggregate, with the excluded count and a median
    reported alongside -- same fix, carried forward from the start.
  - Entries are NOT guaranteed byte-for-byte identical to the control
    (ENTRY-SET DIVERGENCE -- capital-timing noise, not a signal difference).
  - Top-K removal ranks each variant's OWN (slice-level) trades.
  - Winner-preservation diagnostics, including pct_control_winner_pnl_
    retained (introduced in the time-based-exit study), are reported for
    every variant against the control, using position-level aggregation as
    described above.
  - The SUMMARY table reports CAGR, Sharpe, Sortino, max drawdown, profit
    factor, average holding period, and pct_control_winner_pnl_retained for
    all 3 variants against the unchanged control.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tools.run_sizing_allocation_backtest_equities import (
    ENTRY_COST, EXIT_COST, STARTING_CAPITAL, PER_TRADE_ALLOCATION_PCT,
    BASELINE_STOP_PCT, ENTRY_SETS, build_states as _build_states_base, make_sizing_fn,
)
from tools.run_sizing_oos_validation_equities import compute_breadth_series
from tools.run_stoploss_tail_risk_test_equities import _load_equities_universe
from tools.run_pullback_entry_backtest_equities import fetch_ohlcv_batch

PROFIT_TAKE_THRESHOLDS_PCT = (25, 50, 100, 150, 200)  # PREDECLARED grid; 150/200 appended per the user's
                                                       # explicit post-hoc extension request (25/50/100
                                                       # results/verdicts unchanged) -- see module docstring
PARTIAL_SALE_FRACTION = 0.5                 # fixed fraction sold at trigger, confirmed with the user
TOP_K_REMOVE = (1, 2)
CONCENTRATION_FRACS = (0.01, 0.05, 0.10)
NEAR_ZERO_PNL_THRESHOLD_PCT = 1.0
CONCENTRATION_MAX_SANE_ABS_PCT = 300.0
MFE_CAPTURE_MIN_MFE_PCT = 1.0


def build_states(price):
    """No new indicator columns needed -- the profit-take threshold is a
    fixed multiple of entry_price, computed inline per position."""
    return _build_states_base(price)


def run_backtest_exit_variant(states, entry_filter_fn, sizing_fn, stop_mode,
                               profit_take_threshold_pct=None, partial_sale_fraction=PARTIAL_SALE_FRACTION,
                               stop_pct=BASELINE_STOP_PCT, entry_cost=ENTRY_COST, exit_cost=EXIT_COST):
    """stop_mode: "FIXED" is the only mode (the fixed -12% stop + death-cross
    are NEVER replaced -- see module docstring). profit_take_threshold_pct=
    None reproduces the prior studies' control exactly, byte-for-byte; a
    number adds the one-time partial-sale trigger at that gain level."""
    assert stop_mode == "FIXED"
    assert 0 < partial_sale_fraction < 1

    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions = {sym: None for sym in states}
    trade_log = []
    nav_history = []

    for ts in all_index:
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional
        nav_unrealized_sum = 0.0

        for sym, state in states.items():
            pos = positions[sym]
            if ts not in state.index:
                if pos is not None:
                    nav_unrealized_sum += pos["notional"] * (pos["last_price"] / pos["entry_price"] - 1)
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                if pos is not None:
                    nav_unrealized_sum += pos["notional"] * (pos["last_price"] / pos["entry_price"] - 1)
                continue

            if pos is not None:
                # MFE bookkeeping is a pure diagnostic, carried across both
                # slices of a split position -- never reset by a partial sale.
                pos["mfe_high"] = max(pos["mfe_high"], row["high"])

                # CHECK 4: stop_price is computed off entry_price every time
                # -- never re-based on the reduced notional or the partial-
                # sale price after a partial sale.
                stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                stop_hit = row["low"] <= stop_price

                # CHECK 2: full-close conditions (stop, then death-cross)
                # are resolved FIRST, identically to every prior study.
                exit_price = None
                reason = None
                if stop_hit:
                    exit_price = stop_price * (1 - exit_cost)
                    reason = "stop"
                elif row["cross_down"]:
                    exit_price = row["close"] * (1 - exit_cost)
                    reason = "signal"

                if reason is not None:
                    ret = (exit_price / pos["entry_price"]) - 1
                    pnl = pos["notional"] * ret
                    capital += pnl
                    holding_days = (ts - pos["entry_time"]).days
                    mfe_pct = (pos["mfe_high"] / pos["entry_price"] - 1) * 100
                    mfe_capture_pct = (ret * 100 / mfe_pct * 100) if mfe_pct > 1e-9 else float("nan")
                    trade_log.append({
                        "asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                        "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital, "exit_reason": reason,
                        "holding_days": holding_days, "mfe_pct": mfe_pct, "mfe_capture_pct": mfe_capture_pct,
                        "original_notional": pos["original_notional"],
                    })
                    positions[sym] = None
                    free_capital += pos["notional"]
                    # CHECK 6: `continue` skips the entry branch below for
                    # this symbol for the rest of today -- no same-day
                    # re-entry.
                    continue

                # CHECK 1 / CHECK 3: only reached when neither stop nor
                # death-cross fired above. threshold_price is fixed at
                # entry (see the entry branch below); the check here uses
                # only TODAY's own high -- no lookahead, exactly like the
                # stop's use of today's own low. pos["partial_taken"] makes
                # this a strictly one-time event per position.
                if profit_take_threshold_pct is not None and not pos["partial_taken"]:
                    threshold_price = pos["entry_price"] * (1 + profit_take_threshold_pct / 100)
                    if row["high"] >= threshold_price:
                        sale_notional = pos["notional"] * partial_sale_fraction
                        fill = threshold_price * (1 - exit_cost)
                        ret = (fill / pos["entry_price"]) - 1
                        pnl = sale_notional * ret
                        capital += pnl
                        holding_days = (ts - pos["entry_time"]).days
                        mfe_pct = (pos["mfe_high"] / pos["entry_price"] - 1) * 100
                        mfe_capture_pct = (ret * 100 / mfe_pct * 100) if mfe_pct > 1e-9 else float("nan")
                        trade_log.append({
                            "asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                            "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital,
                            "exit_reason": "partial_profit", "holding_days": holding_days,
                            "mfe_pct": mfe_pct, "mfe_capture_pct": mfe_capture_pct,
                            "original_notional": pos["original_notional"],
                        })
                        # CHECK 5: proceeds credited to capital above; sold
                        # notional freed for redeployment now.
                        pos["notional"] -= sale_notional
                        pos["partial_taken"] = True
                        free_capital += sale_notional

                pos["last_price"] = row["close"]
                nav_unrealized_sum += pos["notional"] * (row["close"] / pos["entry_price"] - 1)
                continue

            if row["cross_up"] and entry_filter_fn(row):
                notional = sizing_fn(capital, row)
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + entry_cost)
                    pos_dict = {
                        "entry_price": fill, "notional": notional, "original_notional": notional,
                        "entry_time": ts, "last_price": fill, "mfe_high": max(row["high"], fill),
                        "partial_taken": False,
                    }
                    positions[sym] = pos_dict
                    free_capital -= notional

        nav_t = capital + nav_unrealized_sum
        nav_history.append((ts, nav_t))

    trades = pd.DataFrame(trade_log).sort_values(["asset", "entry_time", "exit_time"]) if trade_log else pd.DataFrame()
    nav_series = pd.Series({t: v for t, v in nav_history}).sort_index() if nav_history else pd.Series(dtype=float)
    return trades, nav_series


def _trade_set(trades):
    """Position-level candidate set (unique (asset, entry_time) pairs) --
    unaffected by whether a position produced 1 or 2 trade_log rows."""
    if trades.empty:
        return set()
    return set(zip(trades["asset"], trades["entry_time"]))


def _aggregate_to_position_level(trades):
    """Collapses a possibly-split position (partial-sale slice + final-close
    slice) back into ONE row per (asset, entry_time), for use ONLY by
    winner_preservation_diagnostics (see module docstring's TRADE-LOG
    GRANULARITY note). A no-op (returns an equivalent single row per
    position) when no position in `trades` ever split. pnl is summed across
    a position's slices; ret_pct is recomputed against the ORIGINAL full
    notional (not either slice's own, smaller, notional) so it is directly
    comparable to the control's own ret_pct; holding_days/mfe_pct/
    mfe_capture_pct/exit_reason are taken from the terminal (final-closing)
    slice, since mfe_high is carried across a position's whole life and its
    max value is only fully reflected once the position is completely
    closed."""
    if trades.empty:
        return trades.copy()
    ordered = trades.sort_values(["asset", "entry_time", "exit_time"])
    grouped = ordered.groupby(["asset", "entry_time"], as_index=False).agg(
        pnl=("pnl", "sum"),
        original_notional=("original_notional", "first"),
        exit_time=("exit_time", "last"),
        holding_days=("holding_days", "last"),
        mfe_pct=("mfe_pct", "last"),
        mfe_capture_pct=("mfe_capture_pct", "last"),
        exit_reason=("exit_reason", "last"),
    )
    grouped["ret_pct"] = grouped["pnl"] / grouped["original_notional"] * 100
    return grouped


# --------------------------------------------------------------------------
# Reporting -- identical to the (corrected) prior scripts, plus the position-
# level aggregation step in winner_preservation_diagnostics.
# --------------------------------------------------------------------------

def summarize_nav(trades, nav_series):
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
    drawdown = (nav_series / running_max - 1) * 100
    max_dd = float(drawdown.min())
    worst_portfolio_day_pct = float(daily_ret.min() * 100) if len(daily_ret) else float("nan")
    return {"cagr_pct": cagr * 100, "total_return_pct": total_return, "max_dd_pct": max_dd,
            "worst_portfolio_day_pct": worst_portfolio_day_pct, "sharpe": sharpe, "sortino": sortino}


def closed_trade_only_stats(trades):
    if trades.empty:
        return {"n_trades": 0}
    total_pnl = float(trades["pnl"].sum())
    total_return_pct = total_pnl / STARTING_CAPITAL * 100
    span_years = max((trades["exit_time"].max() - trades["entry_time"].min()).days / 365.25, 1e-9)
    end_cap = STARTING_CAPITAL + total_pnl
    cagr_pct = (((end_cap / STARTING_CAPITAL) ** (1 / span_years)) - 1) * 100 if end_cap > 0 else float("-100")
    wins = trades[trades.pnl > 0]
    losses = trades[trades.pnl <= 0]
    gross_win = float(wins.pnl.sum())
    gross_loss = float(-losses.pnl.sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(wins) / len(trades) * 100
    avg_holding_days = float(trades["holding_days"].mean())
    max_holding_days = int(trades["holding_days"].max())
    stop_exit_pct = float((trades["exit_reason"] == "stop").mean() * 100)
    signal_exit_pct = float((trades["exit_reason"] == "signal").mean() * 100)
    partial_exit_pct = float((trades["exit_reason"] == "partial_profit").mean() * 100)
    mfe_cap_all = trades["mfe_capture_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    mfe_pct_all = trades.loc[mfe_cap_all.index, "mfe_pct"]
    mfe_included_mask = mfe_pct_all >= MFE_CAPTURE_MIN_MFE_PCT
    mfe_cap_included = mfe_cap_all[mfe_included_mask.to_numpy()]
    n_mfe_capture_included = int(len(mfe_cap_included))
    n_mfe_capture_excluded_small_mfe = int(len(mfe_cap_all) - n_mfe_capture_included)
    avg_mfe_capture_pct = float(mfe_cap_included.mean()) if n_mfe_capture_included else float("nan")
    median_mfe_capture_pct = float(mfe_cap_included.median()) if n_mfe_capture_included else float("nan")
    return {"n_trades": len(trades), "total_pnl": total_pnl, "total_return_pct": total_return_pct,
            "cagr_pct": cagr_pct, "win_rate": win_rate, "pf": pf, "avg_holding_days": avg_holding_days,
            "max_holding_days": max_holding_days, "stop_exit_pct": stop_exit_pct,
            "signal_exit_pct": signal_exit_pct, "partial_exit_pct": partial_exit_pct,
            "avg_mfe_capture_pct": avg_mfe_capture_pct, "median_mfe_capture_pct": median_mfe_capture_pct,
            "n_mfe_capture_included": n_mfe_capture_included,
            "n_mfe_capture_excluded_small_mfe": n_mfe_capture_excluded_small_mfe}


def concentration_stats(trades):
    """See NEAR_ZERO_PNL_THRESHOLD_PCT / CONCENTRATION_MAX_SANE_ABS_PCT above
    -- percentages are suppressed (None, with the raw dollar top-k pnl
    retained) whenever total realized pnl is near zero OR the resulting
    share exceeds the sane bound, exactly as corrected in the ATR/Chandelier
    scripts' clean reruns. Operates at the SLICE level (see module
    docstring's TRADE-LOG GRANULARITY note) -- a split position contributes
    its two slices as two separate entries here, which is informative for
    this study, not an error."""
    if trades.empty:
        return {}
    total_pnl = float(trades["pnl"].sum())
    near_zero = abs(total_pnl) < STARTING_CAPITAL * NEAR_ZERO_PNL_THRESHOLD_PCT / 100
    sorted_pnl = trades.sort_values("pnl", ascending=False)["pnl"].to_numpy()
    n = len(sorted_pnl)
    out = {}
    for frac in CONCENTRATION_FRACS:
        k = max(1, int(np.ceil(n * frac)))
        top_k_pnl = float(sorted_pnl[:k].sum())
        share_pct = (top_k_pnl / total_pnl * 100) if total_pnl else float("nan")
        out_of_sane_bounds = np.isfinite(share_pct) and abs(share_pct) > CONCENTRATION_MAX_SANE_ABS_PCT
        if near_zero or out_of_sane_bounds:
            if near_zero:
                reason = (f"total realized pnl (${total_pnl:,.0f}) is near zero "
                          f"(< {NEAR_ZERO_PNL_THRESHOLD_PCT}% of starting capital) -- "
                          f"percentage-of-total is unstable/uninterpretable here")
            else:
                reason = (f"share_pct ({share_pct:.1f}%) exceeds the "
                          f"+/-{CONCENTRATION_MAX_SANE_ABS_PCT:.0f}% sane bound -- total realized pnl "
                          f"(${total_pnl:,.0f}) is small relative to this bucket's own top-k pnl "
                          f"(${top_k_pnl:,.0f}), producing the same ratio-blowup as the near-zero case "
                          f"even though total_pnl alone is above the near-zero gate")
            out[f"top_{int(frac*100)}pct"] = {"n_trades": k, "pnl_share_pct": None, "top_k_pnl": top_k_pnl,
                                              "unavailable_reason": reason}
        else:
            out[f"top_{int(frac*100)}pct"] = {"n_trades": k, "pnl_share_pct": share_pct}
    return out


def calendar_year_breakdown(trades):
    if trades.empty:
        return {}
    out = {}
    for yr, sub in trades.groupby(trades["entry_time"].dt.year):
        wins = sub[sub.pnl > 0]
        out[str(yr)] = {"n_trades": len(sub), "win_rate": len(wins) / len(sub) * 100 if len(sub) else float("nan"),
                        "total_pnl": float(sub["pnl"].sum())}
    return out


def breadth_bucket_breakdown(trades, breadth):
    if trades.empty:
        return {}
    entry_breadth = trades["entry_time"].map(lambda t: breadth.asof(t))
    valid = entry_breadth.dropna()
    if valid.empty:
        return {}
    terciles = valid.quantile([1 / 3, 2 / 3]).values
    labels = pd.cut(entry_breadth, bins=[-np.inf, terciles[0], terciles[1], np.inf],
                     labels=["LOW-breadth", "MID-breadth", "HIGH-breadth"])
    out = {}
    for label in ["LOW-breadth", "MID-breadth", "HIGH-breadth"]:
        mask = labels == label
        sub = trades[mask.values]
        if sub.empty:
            out[label] = {"n_trades": 0}
            continue
        wins = sub[sub.pnl > 0]
        out[label] = {"n_trades": len(sub), "win_rate": len(wins) / len(sub) * 100, "total_pnl": float(sub["pnl"].sum())}
    return out


def results_excluding_top_k(trades, k):
    if trades.empty or len(trades) <= k:
        return {}
    sorted_desc = trades.sort_values("pnl", ascending=False)
    kept = sorted_desc.iloc[k:]
    total_pnl = float(kept["pnl"].sum())
    return {"n_trades_remaining": len(kept), "total_pnl_excl": total_pnl,
            "total_return_pct_excl": total_pnl / STARTING_CAPITAL * 100}


def winner_preservation_diagnostics(control_trades, variant_trades, control_label, variant_label):
    """Matches trades by (asset, entry_time), AFTER collapsing any split
    position back to one row per position via _aggregate_to_position_level
    (see module docstring's TRADE-LOG GRANULARITY note) -- required here,
    unlike the prior three scripts, because a variant trade set may contain
    2 rows for a position the control only ever recorded once. NOT
    guaranteed 1:1 even after that step -- see the ATR trailing-stop
    script's ENTRY-SET DIVERGENCE note (capital-timing effects). Restricted
    to the intersection; the unmatched count is reported, not hidden.

    pct_control_winner_pnl_retained: of the TOTAL realized pnl the control
    earned on trades that were winners IN THE CONTROL, what fraction does
    this variant still realize on those exact same (position-level)
    trades."""
    control_pos = _aggregate_to_position_level(control_trades)
    variant_pos = _aggregate_to_position_level(variant_trades)
    if control_pos.empty or variant_pos.empty:
        return {"n_matched": 0}
    c = control_pos.set_index(["asset", "entry_time"])
    v = variant_pos.set_index(["asset", "entry_time"])
    common = c.index.intersection(v.index)
    n_matched = len(common)
    n_control = len(c)
    note = None if n_matched == n_control else (
        f"{n_control - n_matched} of {n_control} control trades have no counterpart in {variant_label} "
        f"(capital-timing divergence, not a signal difference -- see module docstring); comparison below "
        f"uses only the {n_matched} commonly-taken trades")
    c_m = c.loc[common]
    v_m = v.loc[common]

    control_winner_mask = c_m["ret_pct"] > 0
    cut_short_mask = control_winner_mask & (v_m["ret_pct"].to_numpy() < c_m["ret_pct"].to_numpy())
    n_control_winners = int(control_winner_mask.sum())
    n_cut_short = int(cut_short_mask.sum())
    pct_winners_cut_short = (n_cut_short / n_control_winners * 100) if n_control_winners else float("nan")

    c_wins = c_m[c_m["pnl"] > 0]
    v_wins = v_m[v_m["pnl"] > 0]
    avg_winner_change = (float(v_wins["pnl"].mean()) if len(v_wins) else 0.0) - (float(c_wins["pnl"].mean()) if len(c_wins) else 0.0)
    largest_winner_change = float(v_m["pnl"].max()) - float(c_m["pnl"].max())

    control_winner_pnl_total = float(c_m.loc[control_winner_mask, "pnl"].sum())
    variant_pnl_on_control_winners = float(v_m.loc[control_winner_mask, "pnl"].sum())
    pct_control_winner_pnl_retained = (
        (variant_pnl_on_control_winners / control_winner_pnl_total * 100) if control_winner_pnl_total else float("nan")
    )

    c_mfe_all = c_m["mfe_capture_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    v_mfe_all = v_m["mfe_capture_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    c_mfe = c_mfe_all[(c_m.loc[c_mfe_all.index, "mfe_pct"] >= MFE_CAPTURE_MIN_MFE_PCT).to_numpy()]
    v_mfe = v_mfe_all[(v_m.loc[v_mfe_all.index, "mfe_pct"] >= MFE_CAPTURE_MIN_MFE_PCT).to_numpy()]
    mfe_capture_change = (float(v_mfe.mean()) if len(v_mfe) else float("nan")) - (float(c_mfe.mean()) if len(c_mfe) else float("nan"))

    return {"n_matched": n_matched, "n_control_winners": n_control_winners, "n_cut_short": n_cut_short,
            "pct_winners_cut_short": pct_winners_cut_short, "avg_winner_pnl_change": avg_winner_change,
            "largest_winner_pnl_change": largest_winner_change, "mfe_capture_pct_change": mfe_capture_change,
            "control_winner_pnl_total": control_winner_pnl_total,
            "variant_pnl_on_control_winners": variant_pnl_on_control_winners,
            "pct_control_winner_pnl_retained": pct_control_winner_pnl_retained,
            "note": note}


def print_variant_report(label, trades, nav_series, breadth, control_trades=None, control_label=None):
    print("\n" + "=" * 90)
    print(f"VARIANT: {label}")
    print("=" * 90)
    nav_summ = summarize_nav(trades, nav_series)
    closed_summ = closed_trade_only_stats(trades)
    print(f"  [NAV, full capital path]      cagr={nav_summ.get('cagr_pct', float('nan')):.2f}%  "
          f"total_return={nav_summ.get('total_return_pct', float('nan')):.2f}%  "
          f"sharpe={nav_summ.get('sharpe', float('nan')):.2f}  sortino={nav_summ.get('sortino', float('nan')):.2f}  "
          f"max_dd={nav_summ.get('max_dd_pct', float('nan')):.2f}%  "
          f"worst_day={nav_summ.get('worst_portfolio_day_pct', float('nan')):.2f}%")
    print(f"  [CLOSED TRADES ONLY, slice-level] n={closed_summ.get('n_trades', 0)}  "
          f"cagr={closed_summ.get('cagr_pct', float('nan')):.2f}%  "
          f"total_return={closed_summ.get('total_return_pct', float('nan')):.2f}%  "
          f"win_rate={closed_summ.get('win_rate', float('nan')):.1f}%  pf={closed_summ.get('pf', float('nan')):.2f}")
    print(f"  [HOLDING / EXIT MIX / MFE]    avg_hold={closed_summ.get('avg_holding_days', float('nan')):.0f}d  "
          f"max_hold={closed_summ.get('max_holding_days', 0)}d  "
          f"stop_exit_pct={closed_summ.get('stop_exit_pct', float('nan')):.1f}%  "
          f"signal_exit_pct={closed_summ.get('signal_exit_pct', float('nan')):.1f}%  "
          f"partial_exit_pct={closed_summ.get('partial_exit_pct', float('nan')):.1f}%")
    print(f"  [MFE]                         avg_mfe_capture={closed_summ.get('avg_mfe_capture_pct', float('nan')):.1f}%  "
          f"median_mfe_capture={closed_summ.get('median_mfe_capture_pct', float('nan')):.1f}%  "
          f"(n_included={closed_summ.get('n_mfe_capture_included', 0)}  "
          f"n_excluded_small_mfe={closed_summ.get('n_mfe_capture_excluded_small_mfe', 0)})")

    conc = concentration_stats(trades)
    for k, v in conc.items():
        if v.get("pnl_share_pct") is None:
            print(f"  [CONCENTRATION {k}]  n={v['n_trades']}  pnl_share=N/A (top-k pnl=${v['top_k_pnl']:,.0f}) -- {v['unavailable_reason']}")
        else:
            print(f"  [CONCENTRATION {k}]  n={v['n_trades']}  pnl_share={v['pnl_share_pct']:.1f}%")

    print("  [BY CALENDAR YEAR]")
    for yr, v in calendar_year_breakdown(trades).items():
        print(f"    {yr}: n={v.get('n_trades',0)}  win_rate={v.get('win_rate', float('nan')):.1f}%  "
              f"total_pnl={v.get('total_pnl', 0.0):,.0f}")

    print("  [BY BREADTH BUCKET AT ENTRY]")
    for label_b, v in breadth_bucket_breakdown(trades, breadth).items():
        print(f"    {label_b}: n={v.get('n_trades',0)}  win_rate={v.get('win_rate', float('nan')):.1f}%  "
              f"total_pnl={v.get('total_pnl', 0.0):,.0f}")

    print("  [RESULTS AFTER REMOVING THIS VARIANT'S OWN TOP-K TRADES]")
    for k in TOP_K_REMOVE:
        r = results_excluding_top_k(trades, k)
        if r:
            print(f"    excl_top{k}: n_remaining={r['n_trades_remaining']}  "
                  f"total_pnl_excl={r['total_pnl_excl']:,.0f}  total_return_excl={r['total_return_pct_excl']:.2f}%")

    wp = None
    if control_trades is not None:
        wp = winner_preservation_diagnostics(control_trades, trades, control_label, label)
        print(f"  [WINNER PRESERVATION vs {control_label}, position-level]")
        if wp.get("note"):
            print(f"    {wp['note']}")
        print(f"    matched_trades={wp.get('n_matched', 0)}  control_winners={wp.get('n_control_winners', 0)}  "
              f"cut_short={wp.get('n_cut_short', 0)} ({wp.get('pct_winners_cut_short', float('nan')):.1f}% of control winners)")
        print(f"    avg_winner_pnl_change={wp.get('avg_winner_pnl_change', float('nan')):,.0f}  "
              f"largest_winner_pnl_change={wp.get('largest_winner_pnl_change', float('nan')):,.0f}  "
              f"mfe_capture_pct_change={wp.get('mfe_capture_pct_change', float('nan')):.1f}pp")
        print(f"    control_winner_pnl_total={wp.get('control_winner_pnl_total', float('nan')):,.0f}  "
              f"variant_pnl_on_control_winners={wp.get('variant_pnl_on_control_winners', float('nan')):,.0f}  "
              f"pct_control_winner_pnl_retained={wp.get('pct_control_winner_pnl_retained', float('nan')):.1f}%")

    return nav_summ, closed_summ, wp


def main():
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live; hygiene gate cleared, no exclusions)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments")

    states = build_states(price)
    all_idx = sorted(set().union(*[s.index for s in states.values()]))
    print(f"full history span: {all_idx[0].date()} -> {all_idx[-1].date()}\n")

    entry_filter_fn = ENTRY_SETS["BASELINE"]
    sizing_fn = make_sizing_fn("EQUAL-DOLLAR", 0.0, 0.0)

    breadth = compute_breadth_series(states)

    control_trades, control_nav = run_backtest_exit_variant(states, entry_filter_fn, sizing_fn, stop_mode="FIXED")
    print_variant_report("CONTROL (fixed -12% stop + death cross, unchanged)", control_trades, control_nav, breadth)

    control_set = _trade_set(control_trades)
    control_nav_summ = summarize_nav(control_trades, control_nav)
    control_closed_summ = closed_trade_only_stats(control_trades)
    variant_results = {}
    for t in PROFIT_TAKE_THRESHOLDS_PCT:
        trades, nav_series = run_backtest_exit_variant(
            states, entry_filter_fn, sizing_fn, stop_mode="FIXED", profit_take_threshold_pct=t,
        )
        label = f"PARTIAL_PROFIT +{t}% (sell {int(PARTIAL_SALE_FRACTION*100)}%)"
        variant_set = _trade_set(trades)
        divergence = control_set.symmetric_difference(variant_set)
        if divergence:
            print(f"\n  NOTE: {label} trade set differs from control on {len(divergence)} of "
                  f"{len(control_set | variant_set)} total (symbol, entry_time) pairs -- expected capital-"
                  f"timing divergence (see module docstring), not a signal difference.")
        nav_summ, closed_summ, wp = print_variant_report(label, trades, nav_series, breadth,
                                                          control_trades, "CONTROL")
        variant_results[t] = (nav_summ, closed_summ, wp)

    print("\n" + "=" * 90)
    print("SUMMARY: does any partial-profit-taking variant beat the control?")
    print("=" * 90)
    print(f"  CONTROL: cagr={control_nav_summ.get('cagr_pct', float('nan')):.2f}%  "
          f"sharpe={control_nav_summ.get('sharpe', float('nan')):.2f}  "
          f"sortino={control_nav_summ.get('sortino', float('nan')):.2f}  "
          f"max_dd={control_nav_summ.get('max_dd_pct', float('nan')):.2f}%  "
          f"pf={control_closed_summ.get('pf', float('nan')):.2f}  "
          f"avg_hold={control_closed_summ.get('avg_holding_days', float('nan')):.0f}d")
    any_beats = False
    retained_pcts = []
    for t in PROFIT_TAKE_THRESHOLDS_PCT:
        nav_summ, closed_summ, wp = variant_results[t]
        beats_cagr = nav_summ.get("cagr_pct", float("-inf")) > control_nav_summ.get("cagr_pct", float("inf"))
        beats_sharpe = nav_summ.get("sharpe", float("-inf")) > control_nav_summ.get("sharpe", float("inf"))
        beats_sortino = nav_summ.get("sortino", float("-inf")) > control_nav_summ.get("sortino", float("inf"))
        beats_both_risk_adj = beats_sharpe and beats_sortino
        if beats_cagr and beats_both_risk_adj:
            verdict = "BEATS control on CAGR AND risk-adjusted (Sharpe+Sortino)"
        elif beats_cagr:
            verdict = "beats control on CAGR only (risk-adjusted did not improve)"
        elif beats_both_risk_adj:
            verdict = "beats control on risk-adjusted only (CAGR did not improve)"
        else:
            verdict = "does not beat control on CAGR or risk-adjusted metrics"
        any_beats = any_beats or (beats_cagr and beats_both_risk_adj)
        if wp:
            retained_pcts.append(wp.get("pct_control_winner_pnl_retained", float("nan")))
        print(f"  PARTIAL_PROFIT +{t}%: cagr={nav_summ.get('cagr_pct', float('nan')):.2f}%  "
              f"sharpe={nav_summ.get('sharpe', float('nan')):.2f}  "
              f"sortino={nav_summ.get('sortino', float('nan')):.2f}  "
              f"max_dd={nav_summ.get('max_dd_pct', float('nan')):.2f}%  "
              f"pf={closed_summ.get('pf', float('nan')):.2f}  "
              f"avg_hold={closed_summ.get('avg_holding_days', float('nan')):.0f}d  "
              f"control_winner_pnl_retained={wp.get('pct_control_winner_pnl_retained', float('nan')):.1f}% "
              f"-> {verdict}")

    if any_beats:
        closing_note = ("at least one partial-profit-taking variant clearly beats the control -- worth deeper "
                         "OOS validation before adopting.")
    else:
        avg_retained = float(np.nanmean(retained_pcts)) if retained_pcts else float("nan")
        closing_note = (
            f"no partial-profit-taking variant beats the control on both CAGR and risk-adjusted performance "
            f"(average control-winner pnl retained across all {len(PROFIT_TAKE_THRESHOLDS_PCT)} variants: "
            f"{avg_retained:.1f}%). Per the user's explicit grid-extension request, the next step is OOS + "
            f"block-bootstrap validation of the 100/150/200% region against the control (see "
            f"run_exit_logic_oos_bootstrap_validation_equities.py), not a further extension of this grid."
        )
    print(f"\n  Per the predeclared sequencing: {closing_note}")


if __name__ == "__main__":
    main()
