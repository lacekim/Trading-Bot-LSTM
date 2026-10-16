"""ATR trailing-stop exit-logic backtest -- the first substantive experiment
in the exit-logic research phase, per the user's explicit sequencing:
"I would make ATR trailing stops the first substantive experiment. They are
the cleanest alternative to the fixed stop and can be evaluated without
introducing portfolio-level selection effects. If none of the ATR variants
beats the control out of sample, move to Chandelier exits."

This gate was cleared first: the 1962-history data-hygiene spot-check found
no corruption in any of the 38 pre-1990 "flagged" symbols (zero zero/negative
closes, zero NaN cells, zero duplicate index entries, plausible coverage
ratios and return magnitudes on every one) -- they are real, long-listed
blue-chip names. The one material finding (breadth differs by up to 24.5pp
with vs. without those symbols, from 2015 onward) is a structural fact about
breadth's sensitivity to universe composition, not a corruption signal, and
is carried forward as context for the separate, later breadth-based-exposure
research thread. Sizing calibration was NOT materially affected (<5% both
ways) and no exclusion was applied -- all 60 symbols are used below,
unchanged.

FROZEN CONTROL (identical to the live bot):
  - BASELINE golden-cross entries (immediate entry on ma50/ma200 cross-up,
    no additional filters) -- ENTRY_SETS["BASELINE"], imported unchanged.
  - EQUAL-DOLLAR sizing (fixed PER_TRADE_ALLOCATION_PCT of current capital)
    -- make_sizing_fn("EQUAL-DOLLAR", ...), imported unchanged. Calibration
    inputs are irrelevant to this method (it ignores them), so no median
    ATR%/vol20 calculation is needed for this experiment at all.
  - Death-cross exit -- unchanged in every variant below.
  - Fixed -12% stop (BASELINE_STOP_PCT) -- this is the ONE thing that
    changes between the control and the treatment variants.
  - Same 12bps round-trip cost, same capital/exposure rules, same execution
    timing (fill at that day's close +/- cost), same 60-symbol universe.

TREATMENT: replace ONLY the fixed -12% stop with an ATR trailing stop, for
multiplier in {2, 3, 4} -- a PREDECLARED grid, fixed before this script was
run, not selected after inspecting results. The death-cross exit is left
untouched in every variant, so any performance difference is attributable
to the stop mechanism alone, not to a confound with the signal exit.

ATR TRAILING STOP DEFINITION:
  stop_level(t) = max(stop_level(t-1),
                       high_water_mark(through day t-1) - multiplier * ATR(t-1))
  where high_water_mark is the running max of daily HIGHs since entry, and
  ATR(t-1) is the 14-day ATR as of the prior trading day (atr14_lag1 -- a new
  column added below, atr14.shift(1), mirroring the same lag convention the
  sizing scripts already use for atr_pct_lag1/vol20_lag1). So the stop level
  applied on day t is entirely determined by information available before
  day t's bar prints; day t's own high/low never feed day t's own stop
  level, only the NEXT day's. The stop only ratchets in the favorable
  direction (the max(...) makes it structurally impossible to loosen). At
  entry, the initial stop anchors off atr14_lag1 AT the entry bar (ATR as of
  the day BEFORE entry), so no stop is ever built from information
  unavailable at entry.

SIX IMPLEMENTATION CHECKS (see run_backtest_exit_variant's inline comments
for exactly where each is enforced; verified against synthetic data,
including deliberately adversarial cases, by the dry run in scratchpad
before this script was ever pointed at real data):
  1. Stops use only information available before execution.
  2. Intraday stop assumptions are consistent when daily OHLC can hit both a
     stop and an entry/exit condition (exits are resolved before entries,
     every day, in both modes).
  3. A trailing stop only moves in the favorable direction.
  4. The death-cross and stop exit have deterministic precedence when both
     trigger (stop is checked first, identically in every variant).
  5. Open positions are marked to market identically across all variants
     (the unrealized-PnL/NAV code path is untouched between modes; only the
     stop-level computation differs).
  6. A position exiting today cannot be re-entered using later information
     the same day (the `continue` after an exit skips that symbol's entry
     check for the rest of that day's iteration).

ENTRY-SET DIVERGENCE (a real effect, measured rather than assumed away):
in a shared-capital, multi-symbol portfolio, EQUAL-DOLLAR sizing is
proportional to CURRENT capital, so the moment the control and an ATR
variant close their first differing trade (different exit day or price),
their capital paths diverge -- which can change whether a LATER candidate's
notional still fits within free_capital in one variant but not the other.
This is the identical capital-constraint-divergence phenomenon already
disclosed for cost regimes in the sizing/allocation research thread, not a
new bug class, and no exit-only change could ever fully prevent it in a
shared capital pool. What IS guaranteed, and what the dry run verifies
directly: the SIGNAL itself (cross_up & entry_filter_fn(row), a pure
function of price/indicator data with zero capital dependency) produces an
IDENTICAL set of entry-eligible candidates regardless of stop mode, and
every actual entry in the control and all three ATR variants is a member of
that same candidate set -- divergence only ever comes from a capital-
availability skip, never from a different entry decision, and the fill
price for any commonly-taken entry is computed by the identical formula in
every variant. main() below reports the size of this divergence for the
real run rather than hiding it.

METHODOLOGY DISCLOSED UP FRONT:
  - This is a FULL-HISTORY, in-sample-only first pass, exactly as the user
    specified ("For the first pass, test one exit family at a time..."). No
    out-of-sample split is applied here -- if any ATR variant looks like a
    plausible improvement, it should go through the same OOS-validation
    discipline already used for the sizing/allocation research (frozen
    split date, frozen decision rule) BEFORE being treated as a candidate
    change to the live bot. That is explicitly a NEXT step, not done here.
  - "Closed-trade" results use ONLY realized pnl from closed trades, divided
    by STARTING_CAPITAL, ignoring any still-open position's unrealized
    mark-to-market gain/loss. "NAV" results use the full capital-path
    series, which includes unrealized gains on positions still open at the
    end of the data. The two are reported separately, per the user's
    request, precisely because they diverged materially in the OOS
    validation thread and that gap needs to stay visible here too.
  - Winner-preservation diagnostics match trades between the control and
    each ATR variant by (asset, entry_time) -- guaranteed to be a 1:1 match
    since entries are frozen identical. "Cut short" means: the control
    trade was a winner (ret_pct > 0) AND the ATR variant realized a smaller
    return on that same entry.
  - Top-K removal ("results after removing the top 1-2 trades") ranks each
    variant's OWN trades by pnl and removes that variant's own top K -- not
    the control's top K -- so each variant is judged on its own tail, not
    forced to drop the control's specific outlier trades.
  - P&L concentration (top 1%/5%/10%) is defined as: sum of pnl of the top
    ceil(n * pct) trades (ranked by pnl, winners and losers together) as a
    percentage of TOTAL realized pnl -- consistent with how this was framed
    earlier in this research thread ("46%-51% of total P&L from the top 1%
    of trades").
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

ATR_MULTIPLIERS = (2, 3, 4)              # PREDECLARED grid -- fixed before any result was seen
TOP_K_REMOVE = (1, 2)                    # "results after removing the top 1-2 trades"
CONCENTRATION_FRACS = (0.01, 0.05, 0.10)  # top-1% / 5% / 10% P&L concentration
BREADTH_RECENT_START = pd.Timestamp("2015-01-01")
# Below this (as a fraction of STARTING_CAPITAL), total realized pnl is
# treated as "near zero" and concentration percentages are suppressed rather
# than reported -- the ratio-to-total-pnl definition is unstable/uninterpretable
# there (see the first ATR run's ATR_TRAIL x2 output: -573.8%/-1615%/-2412%).
NEAR_ZERO_PNL_THRESHOLD_PCT = 1.0
# A NEAR_ZERO_PNL_THRESHOLD_PCT gate on the total isn't sufficient by itself --
# the first clean rerun of THIS script (ATR_TRAIL x2, total realized pnl =
# -$1,400 = 1.4% of starting capital, just above the 1.0% gate) still printed
# top_1/5/10pct shares of -588.1%/-1655.2%/-2472.2%: a total that is merely
# SMALL relative to its own top-k trades' pnl produces the identical blowup as
# a total near zero, because the failure mode is the ratio's magnitude, not
# specifically its distance from zero. So concentration is ALSO suppressed
# whenever the computed share, whatever the total_pnl gate says, would exceed
# this sane bound in absolute value.
CONCENTRATION_MAX_SANE_ABS_PCT = 300.0
# A per-trade MFE-capture ratio (realized_return_pct / mfe_pct) has the same
# near-zero-denominator blowup as the concentration ratio above, but the
# denominator here is each trade's OWN max-favorable-excursion percentage,
# not an aggregate total. On the real 60-symbol universe, the great majority
# of ATR/Chandelier-variant exits are tight stop-outs (96-99% stop_exit_pct)
# where price often ticks up only a few basis points before reversing --
# mfe_pct on the order of 0.01-0.5% is common there. Dividing a routine -1%
# to -4% loss by such a small mfe_pct produces a ratio in the thousands of
# percent for a single trade, and a NAIVE MEAN across ~1,800 trades is not
# robust to that (a handful of extreme-ratio trades dominate the average --
# this is why avg_mfe_capture_pct printed -238% to -283% in BOTH the original
# buggy run and this clean rerun: the MFE-SEEDING fix corrects mfe_pct's SIGN
# at entry, but does nothing about mfe_pct's MAGNITUDE being tiny on a tightly
# stopped-out trade, which is a separate, pre-existing instability in this
# derived statistic). Below this threshold, a trade's mfe_capture_pct is
# excluded from the AVERAGE (never from the per-trade record) and counted
# separately, analogous to the concentration suppression above.
MFE_CAPTURE_MIN_MFE_PCT = 1.0


# --------------------------------------------------------------------------
# Indicator pipeline: identical to the sizing scripts, plus one new column.
# --------------------------------------------------------------------------

def build_states(price):
    """Reuses the already-verified indicator pipeline unchanged, then adds
    atr14_lag1 = atr14.shift(1) -- the lagged ATR the trailing stop reads,
    so no stop calculation ever sees same-day ATR."""
    states = _build_states_base(price)
    for state in states.values():
        state["atr14_lag1"] = state["atr14"].shift(1)
    return states


# --------------------------------------------------------------------------
# Engine: one function, mode-switched, so control and treatment share every
# line of code except the stop-level computation itself.
# --------------------------------------------------------------------------

def run_backtest_exit_variant(states, entry_filter_fn, sizing_fn, stop_mode, atr_multiplier=None,
                               stop_pct=BASELINE_STOP_PCT, entry_cost=ENTRY_COST, exit_cost=EXIT_COST):
    """stop_mode: "FIXED" (control -- byte-for-byte the existing fixed -12%
    stop) or "ATR_TRAIL" (multiplier * ATR ratchet). Entry logic, sizing,
    cost model, and NAV bookkeeping are identical between modes -- only the
    stop-level computation inside the position-open branch differs."""
    assert stop_mode in ("FIXED", "ATR_TRAIL")
    if stop_mode == "ATR_TRAIL":
        assert atr_multiplier is not None and atr_multiplier > 0

    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions = {sym: None for sym in states}
    trade_log = []
    nav_history = []
    fallback_anchor_count = 0  # diagnostic: should be 0 on real data (see module docstring)

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

                if stop_mode == "FIXED":
                    stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                else:
                    atr_lag1_today = row["atr14_lag1"]
                    if pd.notna(atr_lag1_today) and atr_lag1_today > 0:
                        candidate_stop = pos["hwm"] - atr_multiplier * atr_lag1_today
                        # CHECK 3: ratchet -- can only move up.
                        pos["running_stop"] = max(pos["running_stop"], candidate_stop)
                    stop_price = pos["running_stop"]

                # CHECK 4: stop resolved before signal exit, identically in
                # every variant -- deterministic precedence.
                stop_hit = row["low"] <= stop_price
                if stop_hit:
                    exit_price = stop_price * (1 - exit_cost)
                    reason = "stop"
                elif row["cross_down"]:
                    exit_price = row["close"] * (1 - exit_cost)
                    reason = "signal"
                else:
                    pos["last_price"] = row["close"]
                    nav_unrealized_sum += pos["notional"] * (row["close"] / pos["entry_price"] - 1)
                    if stop_mode == "ATR_TRAIL":
                        # CHECK 1: hwm is updated with TODAY's high only
                        # AFTER today's stop check -- it can only affect
                        # TOMORROW's stop level, never today's.
                        pos["hwm"] = max(pos["hwm"], row["high"])
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
                    # mfe_high seeds at max(today's high, fill) -- NOT just
                    # today's high -- because fill already includes the
                    # entry cost markup (close*(1+entry_cost)), which on a
                    # low-range day can exceed that day's own high. Without
                    # this max(), mfe_pct (and mfe_capture_pct) would come
                    # out negative from the moment of entry on every such
                    # day, which is nonsensical for a "maximum favorable
                    # excursion" that is trivially >= 0 at its own starting
                    # point. Caught by re-reading the real run's output
                    # (avg_mfe_capture was implausibly and uniformly
                    # negative across every variant, control included).
                    pos_dict = {"entry_price": fill, "notional": notional, "entry_time": ts,
                                "last_price": fill, "mfe_high": max(row["high"], fill)}
                    if stop_mode == "ATR_TRAIL":
                        # CHECK 1 (entry side): anchor ATR is atr14_lag1 AT
                        # the entry bar -- ATR from strictly BEFORE the
                        # entry day. No stop is built from information
                        # unavailable at entry.
                        atr_at_entry = row["atr14_lag1"]
                        if pd.isna(atr_at_entry) or atr_at_entry <= 0:
                            # Safety net only -- should not fire on real data,
                            # since 14-day ATR warms up long before the
                            # 200-day ma200 that gates every entry. Falls
                            # back to the fixed stop's own distance so a
                            # missing ATR reading never gates the entry
                            # decision itself (the entry signal has no ATR
                            # dependency either way -- this fallback only
                            # affects the STOP anchor, never whether this
                            # candidate becomes a trade).
                            fallback_anchor_count += 1
                            atr_at_entry = fill * abs(stop_pct) / 100 / atr_multiplier
                        pos_dict["running_stop"] = fill - atr_multiplier * atr_at_entry
                        pos_dict["hwm"] = row["high"]
                    positions[sym] = pos_dict
                    free_capital -= notional

        nav_t = capital + nav_unrealized_sum
        nav_history.append((ts, nav_t))

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    nav_series = pd.Series({t: v for t, v in nav_history}).sort_index() if nav_history else pd.Series(dtype=float)
    return trades, nav_series, fallback_anchor_count


def _trade_set(trades):
    if trades.empty:
        return set()
    return set(zip(trades["asset"], trades["entry_time"]))


# --------------------------------------------------------------------------
# Reporting
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
    """Realized-pnl-only view: ignores any still-open position's unrealized
    mark-to-market gain/loss, reported separately from the NAV view above
    per the user's request."""
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
            "avg_mfe_capture_pct": avg_mfe_capture_pct, "median_mfe_capture_pct": median_mfe_capture_pct,
            "n_mfe_capture_included": n_mfe_capture_included,
            "n_mfe_capture_excluded_small_mfe": n_mfe_capture_excluded_small_mfe}


def concentration_stats(trades):
    """CAVEAT (surfaced, not discovered later): this is a ratio to TOTAL
    realized pnl, which can be small or negative on a thin trade sample --
    in that regime the percentage can exceed 100%, go negative, or behave
    non-monotonically across the 1%/5%/10% buckets. That's a property of
    the metric's definition near a zero/negative denominator, not a
    computation bug (the same caveat already applies to top_winner_
    sensitivity in the OOS validation script). Below NEAR_ZERO_PNL_THRESHOLD_PCT
    of STARTING_CAPITAL, the percentages are SUPPRESSED entirely (reported as
    "N/A", with the raw dollar total_pnl instead) rather than printed and
    potentially misread -- the first real run of this script produced
    -573.8%/-1615.0%/-2412.2% for a variant whose total realized pnl was a
    small negative number, which is technically the correct output of the
    formula but invites misreading if left in the research record unlabeled."""
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
    """Removes this variant's OWN top-k trades by pnl (not the control's),
    so each variant is judged on its own tail."""
    if trades.empty or len(trades) <= k:
        return {}
    sorted_desc = trades.sort_values("pnl", ascending=False)
    kept = sorted_desc.iloc[k:]
    total_pnl = float(kept["pnl"].sum())
    return {"n_trades_remaining": len(kept), "total_pnl_excl": total_pnl,
            "total_return_pct_excl": total_pnl / STARTING_CAPITAL * 100}


def winner_preservation_diagnostics(control_trades, variant_trades, control_label, variant_label):
    """Matches trades by (asset, entry_time). NOT guaranteed 1:1 -- see the
    module docstring's ENTRY-SET DIVERGENCE note: capital-timing effects can
    make a signal-eligible candidate get taken in one variant and skipped
    (for insufficient free_capital) in the other, purely because their
    capital paths diverged from an earlier differing exit. The comparison
    below is restricted to the trades both runs actually took (the
    intersection), and the unmatched count is reported, not hidden."""
    if control_trades.empty or variant_trades.empty:
        return {"n_matched": 0}
    c = control_trades.set_index(["asset", "entry_time"])
    v = variant_trades.set_index(["asset", "entry_time"])
    common = c.index.intersection(v.index)
    n_matched = len(common)
    n_control = len(c)
    if n_matched != n_control:
        note = (f"{n_control - n_matched} of {n_control} control trades have no counterpart in {variant_label} "
                f"(capital-timing divergence, not a signal difference -- see module docstring); comparison below "
                f"uses only the {n_matched} commonly-taken trades")
    else:
        note = None
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

    # Same MFE_CAPTURE_MIN_MFE_PCT filter as closed_trade_only_stats, applied
    # to each side's OWN matched-trade subset -- consistent with the
    # each-variant-judged-on-its-own-tail convention used for top-K removal.
    c_mfe_all = c_m["mfe_capture_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    v_mfe_all = v_m["mfe_capture_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    c_mfe = c_mfe_all[(c_m.loc[c_mfe_all.index, "mfe_pct"] >= MFE_CAPTURE_MIN_MFE_PCT).to_numpy()]
    v_mfe = v_mfe_all[(v_m.loc[v_mfe_all.index, "mfe_pct"] >= MFE_CAPTURE_MIN_MFE_PCT).to_numpy()]
    mfe_capture_change = (float(v_mfe.mean()) if len(v_mfe) else float("nan")) - (float(c_mfe.mean()) if len(c_mfe) else float("nan"))

    return {"n_matched": n_matched, "n_control_winners": n_control_winners, "n_cut_short": n_cut_short,
            "pct_winners_cut_short": pct_winners_cut_short, "avg_winner_pnl_change": avg_winner_change,
            "largest_winner_pnl_change": largest_winner_change, "mfe_capture_pct_change": mfe_capture_change,
            "note": note}


def print_variant_report(label, trades, nav_series, control_trades=None, control_label=None):
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
    print(f"  [HOLDING / STOP / MFE]        avg_hold={closed_summ.get('avg_holding_days', float('nan')):.0f}d  "
          f"max_hold={closed_summ.get('max_holding_days', 0)}d  "
          f"stop_exit_pct={closed_summ.get('stop_exit_pct', float('nan')):.1f}%  "
          f"avg_mfe_capture={closed_summ.get('avg_mfe_capture_pct', float('nan')):.1f}%  "
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
    breadth = print_variant_report._breadth
    for label_b, v in breadth_bucket_breakdown(trades, breadth).items():
        print(f"    {label_b}: n={v.get('n_trades',0)}  win_rate={v.get('win_rate', float('nan')):.1f}%  "
              f"total_pnl={v.get('total_pnl', 0.0):,.0f}")

    print("  [RESULTS AFTER REMOVING THIS VARIANT'S OWN TOP-K TRADES]")
    for k in TOP_K_REMOVE:
        r = results_excluding_top_k(trades, k)
        if r:
            print(f"    excl_top{k}: n_remaining={r['n_trades_remaining']}  "
                  f"total_pnl_excl={r['total_pnl_excl']:,.0f}  total_return_excl={r['total_return_pct_excl']:.2f}%")

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

    return nav_summ, closed_summ


def main():
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live; hygiene gate cleared, no exclusions)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments")

    states = build_states(price)
    all_idx = sorted(set().union(*[s.index for s in states.values()]))
    print(f"full history span: {all_idx[0].date()} -> {all_idx[-1].date()}\n")

    entry_filter_fn = ENTRY_SETS["BASELINE"]
    sizing_fn = make_sizing_fn("EQUAL-DOLLAR", 0.0, 0.0)  # calibration args unused by EQUAL-DOLLAR

    breadth = compute_breadth_series(states)
    print_variant_report._breadth = breadth  # stashed for print_variant_report's breadth-bucket section

    control_trades, control_nav, _ = run_backtest_exit_variant(states, entry_filter_fn, sizing_fn, stop_mode="FIXED")
    print_variant_report("CONTROL (fixed -12% stop + death cross)", control_trades, control_nav)

    control_set = _trade_set(control_trades)
    variant_results = {}
    for m in ATR_MULTIPLIERS:
        trades, nav_series, fallback_count = run_backtest_exit_variant(
            states, entry_filter_fn, sizing_fn, stop_mode="ATR_TRAIL", atr_multiplier=m,
        )
        label = f"ATR_TRAIL x{m}"
        variant_set = _trade_set(trades)
        divergence = control_set.symmetric_difference(variant_set)
        if divergence:
            print(f"\n  NOTE: {label} trade set differs from control on {len(divergence)} of "
                  f"{len(control_set | variant_set)} total (symbol, entry_time) pairs -- expected capital-timing "
                  f"divergence from the two variants' capital paths splitting apart after their first differing "
                  f"exit (see module docstring's ENTRY-SET DIVERGENCE note), not a signal difference. Winner-"
                  f"preservation diagnostics below use only the trades both runs actually took.")
        if fallback_count:
            print(f"\n  NOTE: {label} used the fixed-stop-distance fallback anchor on {fallback_count} entries "
                  f"(ATR was NaN/<=0 at entry) -- unexpected on real data, worth checking those specific entries.")
        nav_summ, closed_summ = print_variant_report(label, trades, nav_series, control_trades, "CONTROL")
        variant_results[m] = (nav_summ, closed_summ)

    print("\n" + "=" * 90)
    print("SUMMARY: does any ATR variant beat the control?")
    print("=" * 90)
    control_nav_summ = summarize_nav(control_trades, control_nav)
    control_closed_summ = closed_trade_only_stats(control_trades)
    print(f"  CONTROL: nav_cagr={control_nav_summ.get('cagr_pct', float('nan')):.2f}%  "
          f"sharpe={control_nav_summ.get('sharpe', float('nan')):.2f}  sortino={control_nav_summ.get('sortino', float('nan')):.2f}  "
          f"closed_cagr={control_closed_summ.get('cagr_pct', float('nan')):.2f}%")
    any_beats = False
    for m in ATR_MULTIPLIERS:
        nav_summ, closed_summ = variant_results[m]
        beats_cagr = nav_summ.get("cagr_pct", float("-inf")) > control_nav_summ.get("cagr_pct", float("inf"))
        beats_sharpe = nav_summ.get("sharpe", float("-inf")) > control_nav_summ.get("sharpe", float("inf"))
        beats_sortino = nav_summ.get("sortino", float("-inf")) > control_nav_summ.get("sortino", float("inf"))
        beats_both_risk_adj = beats_sharpe and beats_sortino
        verdict = "BEATS control on CAGR AND risk-adjusted (Sharpe+Sortino)" if (beats_cagr and beats_both_risk_adj) else \
                  "beats control on CAGR only (risk-adjusted did not improve)" if beats_cagr else \
                  "beats control on risk-adjusted only (CAGR did not improve)" if beats_both_risk_adj else \
                  "does not beat control on CAGR or risk-adjusted metrics"
        any_beats = any_beats or (beats_cagr and beats_both_risk_adj)
        print(f"  ATR_TRAIL x{m}: nav_cagr={nav_summ.get('cagr_pct', float('nan')):.2f}%  "
              f"sharpe={nav_summ.get('sharpe', float('nan')):.2f}  sortino={nav_summ.get('sortino', float('nan')):.2f}  "
              f"-> {verdict}")
    if any_beats:
        closing_note = ("at least one ATR variant clearly beats the control -- worth deeper OOS validation "
                         "before adopting.")
    else:
        closing_note = ("no ATR variant beats the control on both CAGR and risk-adjusted performance -- next "
                         "step is Chandelier exits, per the predeclared order.")
    print(f"\n  Per the predeclared sequencing: {closing_note}")


if __name__ == "__main__":
    main()
