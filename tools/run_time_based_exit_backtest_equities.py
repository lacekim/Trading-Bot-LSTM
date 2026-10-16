"""Time-based (stagnation) exit backtest -- the third substantive experiment
in the exit-logic research phase, run because BOTH prior volatility-adaptive
trailing-stop families failed decisively against the control:

    CONTROL:      cagr=8.64%  sharpe=0.96  sortino=1.13  avg_hold=325d
    ATR (best):   cagr=0.74%  sharpe=0.43  sortino=0.46  cut_short=86.6%
    CHANDELIER (best): cagr=1.54% sharpe=0.58 sortino=0.65 cut_short=80.2%

Per the user's own explicit instruction after reviewing the Chandelier
results ("The results support moving on rather than reopening ATR or
Chandelier testing... The next clean experiment should therefore be the
predeclared sequence: 1. Time-based exits after 40, 60, and 100 days without
a new high. 2. Partial profit-taking variants."), this script is item 1.
The ATR and Chandelier exit families are CLOSED, not reopened.

FROZEN CONTROL (identical to the live bot, and to both prior studies):
  - BASELINE golden-cross entries, EQUAL-DOLLAR sizing, death-cross exit,
    fixed -12% stop (BASELINE_STOP_PCT), same costs/execution timing/
    capital rules/60-symbol universe. All imported unchanged from
    run_sizing_allocation_backtest_equities.py.

TREATMENT: an ADDITIONAL exit trigger layered on top of the UNCHANGED fixed
-12% stop and death-cross -- for a PREDECLARED grid, fixed before this
script was first run: TIME_STAGNATION_DAYS = (40, 60, 100) trading days.

GRID EXTENSION (post-hoc, at the user's explicit request after reviewing the
40/60/100 in-sample results and the closing synthesis across all four
exit-logic families): TIME_STAGNATION_DAYS now also includes 150 and 200.
This is NOT re-litigating an already-decided question -- 150d and 200d are
NEW, previously untested points, added because the 40->60->100 progression
was still climbing steeply toward the control (CAGR 2.95%->4.98%->7.73%,
control-winner pnl retained 3.5%->12.1%->61.1%) rather than visibly
plateauing, so it was unclear whether 100d was a genuine local optimum or
just the closest-to-control point in a grid that stopped before it
converged. The original 40/60/100 results and verdicts are unchanged and
reported exactly as before; 150d/200d are appended, not substituted.

DESIGN CHOICE -- ADDITIVE, NOT A REPLACEMENT (an explicit decision, since the
user's instruction did not specify one and this differs from how ATR/
Chandelier were tested): a stagnation exit answers a different question from
a volatility-adaptive stop. ATR/Chandelier asked "does a smarter STOP-LOSS
beat a dumb fixed one" -- both said no, because a tighter stop cuts short the
same rare winners the fixed stop lets run. A time-based exit asks a
different question entirely: "does giving up on a position that has gone
nowhere for N days, while still protecting it with the SAME fixed -12% floor
and the SAME death-cross exit, help or hurt." Replacing the fixed stop with
a time-based exit would conflate that question with the already-answered
"remove the fixed stop" question. So here the fixed -12% stop and
death-cross are left COMPLETELY untouched, and the time-based rule is a
THIRD, independent exit trigger added alongside them. This also means the
"CONTROL" in this script is byte-for-byte the same control as the ATR/
Chandelier studies (stop_mode="FIXED", time_stagnation_days=None reproduces
the prior control exactly), so the fixed -12% + death-cross result already
reported does not need to be re-derived to be comparable.

TIME-BASED (STAGNATION) EXIT DEFINITION:
  Track hwm_since_entry (running max of daily HIGHs since entry -- same
  "since entry" reference point as the Chandelier study) and
  days_since_new_high, a counter that resets to 0 on any day whose high
  exceeds hwm_since_entry and otherwise increments by 1. On day t, BEFORE
  checking today's exits, days_since_new_high reflects the state THROUGH
  yesterday only (today's own high has not yet been folded in). If that
  count is >= TIME_STAGNATION_DAYS, the position is exited TODAY at
  close*(1-exit_cost) (same fill convention as the death-cross exit -- a
  decided-at-end-of-day exit, not an intraday stop). Only AFTER today's exit
  check is resolved does today's high get folded into hwm_since_entry /
  days_since_new_high, so tomorrow's decision reflects today's price action
  but today's decision never does (no lookahead). At entry,
  hwm_since_entry seeds at max(entry-day high, fill) -- the same convention
  used for mfe_high (the entry-day high is the position's own first
  reference point) -- and days_since_new_high starts at 0.

PRECEDENCE (extends the already-established rule, does not change it): the
fixed stop is checked FIRST (unchanged catastrophic floor, highest
priority), then the death-cross (the primary signal-based exit, a stronger
statement than mere stagnation), and ONLY THEN the time-based stagnation
exit (weakest signal -- an admission of indecision, not a reversal). This
keeps the ATR/Chandelier studies' stop-before-signal precedence intact and
appends the new exit type at the bottom of the same deterministic order.

SIX IMPLEMENTATION CHECKS (see run_backtest_exit_variant's inline comments):
  1. days_since_new_high used by today's exit check reflects only state
     through yesterday; today's high is folded in only AFTER today's check.
  2. Intraday exit precedence is deterministic even when stop, death-cross,
     and time-stagnation all coincide on the same day (stop > signal > time).
  3. days_since_new_high resets to 0 on any new high, and only increments
     otherwise -- it never moves backward except via that reset.
  4. A position that keeps making new highs (or just doesn't stagnate for
     the full window) never fires the time-based exit.
  5. Open positions are marked to market identically across all variants --
     the fixed stop, death-cross, and NAV bookkeeping code paths are
     completely untouched by this experiment.
  6. A position exiting today via ANY reason (including the new
     time-stagnation one) cannot be re-entered later the same day.

METHODOLOGY DISCLOSED UP FRONT (identical conventions to the ATR/Chandelier
studies, for direct comparability):
  - Full-history, in-sample-only first pass, exactly as before.
  - "Closed-trade" and "NAV" results reported separately.
  - mfe_high seeds at max(entry-day high, cost-inclusive fill).
  - Concentration percentages are suppressed (None, with the raw dollar
    figure and a reason) whenever a variant's total realized pnl is near
    zero (NEAR_ZERO_PNL_THRESHOLD_PCT of STARTING_CAPITAL) OR whenever the
    resulting share exceeds CONCENTRATION_MAX_SANE_ABS_PCT even if the total
    alone clears the near-zero gate -- both fixes carried forward exactly as
    corrected in the ATR/Chandelier scripts' clean reruns.
  - avg_mfe_capture_pct is computed only over trades whose mfe_pct >=
    MFE_CAPTURE_MIN_MFE_PCT, with the excluded count reported and a median
    alongside the mean -- the same fix, carried forward from the start
    rather than being rediscovered.
  - Entries are NOT guaranteed byte-for-byte identical to the control
    (ENTRY-SET DIVERGENCE -- capital-timing noise, not a signal difference).
  - Top-K removal ranks each variant's OWN trades, not the control's.
  - Winner-preservation diagnostics (cut-short count, avg/largest winner
    change, MFE-capture change) are reported for every variant against the
    control, PLUS a NEW diagnostic requested for this phase:
    pct_control_winner_pnl_retained -- the sum of the variant's own realized
    pnl on exactly the trades that were WINNERS in the control, divided by
    the control's total pnl on those same winning trades. This is a single
    number answering "of the dollars the control's winners made, how many
    survive under this variant" -- a cleaner aggregate than the cut-short
    COUNT alone (a variant could cut short a small fraction of winners by
    trade count but still preserve most of the dollars, or vice versa).
  - The SUMMARY table below reports exactly the comparison set the user
    specified: CAGR, Sharpe, Sortino, max drawdown, profit factor, average
    holding period, and pct_control_winner_pnl_retained, for every variant
    against the unchanged control.
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

TIME_STAGNATION_DAYS = (40, 60, 100, 150, 200)  # PREDECLARED grid; 150/200 appended per the user's explicit
                                                 # post-hoc extension request (see module docstring's GRID
                                                 # EXTENSION note) -- 40/60/100 results/verdicts unchanged
TOP_K_REMOVE = (1, 2)
CONCENTRATION_FRACS = (0.01, 0.05, 0.10)
NEAR_ZERO_PNL_THRESHOLD_PCT = 1.0
# See run_atr_trailing_stop_backtest_equities.py's constants block for the
# full rationale (identical here -- same reporting-layer code): a total-pnl-
# near-zero gate alone doesn't stop the concentration ratio from blowing up
# when total_pnl is merely SMALL relative to its own top-k trades.
CONCENTRATION_MAX_SANE_ABS_PCT = 300.0
# Same rationale, applied to the per-trade MFE-capture ratio's denominator.
MFE_CAPTURE_MIN_MFE_PCT = 1.0


def build_states(price):
    """No new indicator columns needed beyond the base pipeline -- the
    stagnation counter is derived purely from daily highs the base pipeline
    already carries, computed inline in the engine below (state is
    per-position, not a precomputed column, since "days since a new high"
    depends on each position's own entry date)."""
    return _build_states_base(price)


def run_backtest_exit_variant(states, entry_filter_fn, sizing_fn, stop_mode,
                               time_stagnation_days=None,
                               stop_pct=BASELINE_STOP_PCT, entry_cost=ENTRY_COST, exit_cost=EXIT_COST):
    """stop_mode: "FIXED" is the only mode (the fixed -12% stop + death-cross
    are NEVER replaced in this study -- see module docstring's ADDITIVE, NOT
    A REPLACEMENT design note). time_stagnation_days=None reproduces the
    prior studies' control exactly, byte-for-byte; an integer N adds the
    stagnation exit as a third trigger, checked last."""
    assert stop_mode == "FIXED"

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
                # MFE bookkeeping is a pure diagnostic (never read by any
                # decision below), so it may safely use TODAY's own high,
                # including on the exit day itself.
                pos["mfe_high"] = max(pos["mfe_high"], row["high"])

                stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                # CHECK 1: days_since_new_high used here reflects state
                # THROUGH YESTERDAY only -- today's high is folded into
                # hwm_since_entry/days_since_new_high further below, AFTER
                # this check. So a time-exit decided today never uses
                # information from today's own bar.
                time_exit_hit = (
                    time_stagnation_days is not None
                    and pos["days_since_new_high"] >= time_stagnation_days
                )

                # CHECK 2: deterministic precedence -- stop (catastrophic
                # floor) first, then the death-cross (primary signal exit),
                # then the time-based stagnation exit (weakest signal),
                # identically in every variant.
                stop_hit = row["low"] <= stop_price
                if stop_hit:
                    exit_price = stop_price * (1 - exit_cost)
                    reason = "stop"
                elif row["cross_down"]:
                    exit_price = row["close"] * (1 - exit_cost)
                    reason = "signal"
                elif time_exit_hit:
                    exit_price = row["close"] * (1 - exit_cost)
                    reason = "time_stagnation"
                else:
                    pos["last_price"] = row["close"]
                    nav_unrealized_sum += pos["notional"] * (row["close"] / pos["entry_price"] - 1)
                    # CHECK 1 (continued) / CHECK 3: today's high is folded
                    # in only now, after the exit check above has already
                    # resolved -- it can only affect TOMORROW's decision.
                    # The counter resets to 0 on a new high and otherwise
                    # only increments (never moves backward except via that
                    # reset), so it can never "catch up" or skip stagnation.
                    if row["high"] > pos["hwm_since_entry"]:
                        pos["hwm_since_entry"] = row["high"]
                        pos["days_since_new_high"] = 0
                    else:
                        pos["days_since_new_high"] += 1
                    continue

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
                })
                positions[sym] = None
                free_capital += pos["notional"]
                # CHECK 6: `continue` skips the entry branch below for this
                # symbol for the rest of today -- no same-day re-entry.
                continue

            if row["cross_up"] and entry_filter_fn(row):
                notional = sizing_fn(capital, row)
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + entry_cost)
                    seed_high = max(row["high"], fill)
                    pos_dict = {
                        "entry_price": fill, "notional": notional, "entry_time": ts,
                        "last_price": fill, "mfe_high": seed_high,
                        # CHECK 4 (entry side): hwm_since_entry seeds at the
                        # same reference point as mfe_high -- the position's
                        # own entry-day high is its first "high", so
                        # days_since_new_high correctly starts at 0, not
                        # already stagnant.
                        "hwm_since_entry": seed_high, "days_since_new_high": 0,
                    }
                    positions[sym] = pos_dict
                    free_capital -= notional

        nav_t = capital + nav_unrealized_sum
        nav_history.append((ts, nav_t))

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    nav_series = pd.Series({t: v for t, v in nav_history}).sort_index() if nav_history else pd.Series(dtype=float)
    return trades, nav_series


def _trade_set(trades):
    if trades.empty:
        return set()
    return set(zip(trades["asset"], trades["entry_time"]))


# --------------------------------------------------------------------------
# Reporting -- identical to the (corrected) ATR/Chandelier scripts, plus the
# new pct_control_winner_pnl_retained diagnostic in winner_preservation_
# diagnostics and the expanded SUMMARY table below.
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
    avg_trade_pct = float(trades["ret_pct"].mean())
    median_trade_pct = float(trades["ret_pct"].median())
    avg_holding_days = float(trades["holding_days"].mean())
    max_holding_days = int(trades["holding_days"].max())
    stop_exit_pct = float((trades["exit_reason"] == "stop").mean() * 100)
    signal_exit_pct = float((trades["exit_reason"] == "signal").mean() * 100)
    time_exit_pct = float((trades["exit_reason"] == "time_stagnation").mean() * 100)
    mfe_cap_all = trades["mfe_capture_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    mfe_pct_all = trades.loc[mfe_cap_all.index, "mfe_pct"]
    mfe_included_mask = mfe_pct_all >= MFE_CAPTURE_MIN_MFE_PCT
    mfe_cap_included = mfe_cap_all[mfe_included_mask.to_numpy()]
    n_mfe_capture_included = int(len(mfe_cap_included))
    n_mfe_capture_excluded_small_mfe = int(len(mfe_cap_all) - n_mfe_capture_included)
    avg_mfe_capture_pct = float(mfe_cap_included.mean()) if n_mfe_capture_included else float("nan")
    median_mfe_capture_pct = float(mfe_cap_included.median()) if n_mfe_capture_included else float("nan")
    return {"n_trades": len(trades), "total_pnl": total_pnl, "total_return_pct": total_return_pct,
            "cagr_pct": cagr_pct, "win_rate": win_rate, "pf": pf, "avg_trade_pct": avg_trade_pct,
            "median_trade_pct": median_trade_pct, "avg_holding_days": avg_holding_days,
            "max_holding_days": max_holding_days, "stop_exit_pct": stop_exit_pct,
            "signal_exit_pct": signal_exit_pct, "time_exit_pct": time_exit_pct,
            "avg_mfe_capture_pct": avg_mfe_capture_pct, "median_mfe_capture_pct": median_mfe_capture_pct,
            "n_mfe_capture_included": n_mfe_capture_included,
            "n_mfe_capture_excluded_small_mfe": n_mfe_capture_excluded_small_mfe}


def concentration_stats(trades):
    """See NEAR_ZERO_PNL_THRESHOLD_PCT / CONCENTRATION_MAX_SANE_ABS_PCT above
    -- percentages are suppressed (None, with the raw dollar top-k pnl
    retained) whenever total realized pnl is near zero OR the resulting
    share exceeds the sane bound, exactly as corrected in the ATR/Chandelier
    scripts' clean reruns."""
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
    """Matches trades by (asset, entry_time). NOT guaranteed 1:1 -- see the
    ATR trailing-stop script's ENTRY-SET DIVERGENCE note. Restricted to the
    intersection; the unmatched count is reported, not hidden.

    pct_control_winner_pnl_retained (NEW for this phase, per the user's
    explicit request): of the TOTAL realized pnl the control earned on
    trades that were winners IN THE CONTROL, what fraction does this variant
    still realize on those exact same trades. A cleaner aggregate than the
    cut-short COUNT alone -- a variant could cut short a small fraction of
    winners by trade count but still lose most of the dollars (or the
    reverse), and this metric answers the dollar question directly."""
    if control_trades.empty or variant_trades.empty:
        return {"n_matched": 0}
    c = control_trades.set_index(["asset", "entry_time"])
    v = variant_trades.set_index(["asset", "entry_time"])
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
    print(f"  [CLOSED TRADES ONLY]          n={closed_summ.get('n_trades', 0)}  "
          f"cagr={closed_summ.get('cagr_pct', float('nan')):.2f}%  "
          f"total_return={closed_summ.get('total_return_pct', float('nan')):.2f}%  "
          f"win_rate={closed_summ.get('win_rate', float('nan')):.1f}%  pf={closed_summ.get('pf', float('nan')):.2f}  "
          f"avg_trade={closed_summ.get('avg_trade_pct', float('nan')):.2f}%  "
          f"median_trade={closed_summ.get('median_trade_pct', float('nan')):.2f}%")
    print(f"  [HOLDING / EXIT MIX / MFE]    avg_hold={closed_summ.get('avg_holding_days', float('nan')):.0f}d  "
          f"max_hold={closed_summ.get('max_holding_days', 0)}d  "
          f"stop_exit_pct={closed_summ.get('stop_exit_pct', float('nan')):.1f}%  "
          f"signal_exit_pct={closed_summ.get('signal_exit_pct', float('nan')):.1f}%  "
          f"time_exit_pct={closed_summ.get('time_exit_pct', float('nan')):.1f}%")
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
        print(f"  [WINNER PRESERVATION vs {control_label}]")
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
    for n in TIME_STAGNATION_DAYS:
        trades, nav_series = run_backtest_exit_variant(
            states, entry_filter_fn, sizing_fn, stop_mode="FIXED", time_stagnation_days=n,
        )
        label = f"TIME_STAGNATION {n}d"
        variant_set = _trade_set(trades)
        divergence = control_set.symmetric_difference(variant_set)
        if divergence:
            print(f"\n  NOTE: {label} trade set differs from control on {len(divergence)} of "
                  f"{len(control_set | variant_set)} total (symbol, entry_time) pairs -- expected capital-"
                  f"timing divergence (see module docstring), not a signal difference.")
        nav_summ, closed_summ, wp = print_variant_report(label, trades, nav_series, breadth,
                                                          control_trades, "CONTROL")
        variant_results[n] = (nav_summ, closed_summ, wp)

    print("\n" + "=" * 90)
    print("SUMMARY: does any time-based-exit variant beat the control?")
    print("=" * 90)
    print(f"  CONTROL: cagr={control_nav_summ.get('cagr_pct', float('nan')):.2f}%  "
          f"sharpe={control_nav_summ.get('sharpe', float('nan')):.2f}  "
          f"sortino={control_nav_summ.get('sortino', float('nan')):.2f}  "
          f"max_dd={control_nav_summ.get('max_dd_pct', float('nan')):.2f}%  "
          f"pf={control_closed_summ.get('pf', float('nan')):.2f}  "
          f"avg_hold={control_closed_summ.get('avg_holding_days', float('nan')):.0f}d")
    any_beats = False
    retained_pcts = []
    for n in TIME_STAGNATION_DAYS:
        nav_summ, closed_summ, wp = variant_results[n]
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
        print(f"  TIME_STAGNATION {n}d: cagr={nav_summ.get('cagr_pct', float('nan')):.2f}%  "
              f"sharpe={nav_summ.get('sharpe', float('nan')):.2f}  "
              f"sortino={nav_summ.get('sortino', float('nan')):.2f}  "
              f"max_dd={nav_summ.get('max_dd_pct', float('nan')):.2f}%  "
              f"pf={closed_summ.get('pf', float('nan')):.2f}  "
              f"avg_hold={closed_summ.get('avg_holding_days', float('nan')):.0f}d  "
              f"control_winner_pnl_retained={wp.get('pct_control_winner_pnl_retained', float('nan')):.1f}% "
              f"-> {verdict}")

    if any_beats:
        closing_note = ("at least one time-based-exit variant clearly beats the control -- worth deeper OOS "
                         "validation before adopting.")
    else:
        avg_retained = float(np.nanmean(retained_pcts)) if retained_pcts else float("nan")
        closing_note = (
            f"no time-based-exit variant beats the control on both CAGR and risk-adjusted performance "
            f"(average control-winner pnl retained across all {len(TIME_STAGNATION_DAYS)} variants: "
            f"{avg_retained:.1f}%). Per the user's explicit grid-extension request, the next step is OOS + "
            f"block-bootstrap validation of the 100d/150d/200d region against the control (see "
            f"run_exit_logic_oos_bootstrap_validation_equities.py), not a further extension of this grid."
        )
    print(f"\n  Per the predeclared sequencing: {closing_note}")


if __name__ == "__main__":
    main()
