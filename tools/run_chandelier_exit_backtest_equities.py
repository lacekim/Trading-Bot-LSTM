"""Chandelier-exit backtest -- the second substantive experiment in the
exit-logic research phase, run because ALL THREE ATR trailing-stop
multipliers (2, 3, 4) failed decisively against the control:

    CONTROL: cagr=8.64%  sharpe=0.96  sortino=1.13  avg_hold=325d  stop_exit=32.4%
    ATR x2:  cagr=-0.02% sharpe=-0.03 sortino=-0.03 avg_hold=12d   stop_exit=99.2%
    ATR x3:  cagr=0.43%  sharpe=0.36  sortino=0.38  avg_hold=27d   stop_exit=98.1%
    ATR x4:  cagr=0.74%  sharpe=0.43  sortino=0.46  avg_hold=45d   stop_exit=96.7%

Every ATR variant cut short 86.6%-94.8% of the control's winning trades and
lost over $1.2M of pnl from the single largest winner in every case. Per the
user's own predeclared sequencing ("If none of the ATR variants beats the
control out of sample, move to Chandelier exits"), that is exactly what this
script does -- the ATR exit family is now CLOSED, not reopened with
opportunistically wider multipliers.

FROZEN CONTROL (identical to the live bot, and identical to the ATR study):
  - BASELINE golden-cross entries, EQUAL-DOLLAR sizing, death-cross exit,
    fixed -12% stop (BASELINE_STOP_PCT), same costs/execution timing/
    capital rules/60-symbol universe. All imported unchanged from
    run_sizing_allocation_backtest_equities.py.

TREATMENT: replace the fixed -12% stop with a Chandelier exit, for every
combination of a PREDECLARED grid, fixed before this script was run:
  ATR_LOOKBACKS       = (20, 40, 60) trading days
  CHANDELIER_MULTIPLIERS = (3, 4, 5)
  -> 9 combinations, all reported, none selected after inspecting results.

CHANDELIER STOP DEFINITION (exactly as specified):
  chandelier_stop(t) = max(chandelier_stop(t-1),
                            high_water_mark(through day t-1) - k * ATR_N(t-1))
  where high_water_mark is the running max of daily HIGHs SINCE ENTRY (not a
  rolling N-day window -- the reference point is "since entry", per the
  brief), ATR_N(t-1) is the N-trading-day ATR as of the PRIOR day (lagged,
  same no-lookahead convention as the ATR trailing-stop study), and N is the
  grid's "Lookback" parameter (20/40/60) -- i.e. the ATR AVERAGING PERIOD,
  not the high-water-mark window, which stays "since entry" as specified.
  This is a deliberate departure from the textbook Chandelier Exit (which
  traditionally uses a ROLLING N-day highest-high, matching its ATR window)
  -- flagged explicitly here so it is never mistaken for that canonical
  indicator. If a rolling-highest-high variant is wanted later, it is a
  distinct, separately-declared experiment, not folded into this one.
  The stop only ratchets in the favorable direction (max(...) makes it
  structurally impossible to loosen), and the initial stop at entry anchors
  off ATR_N as of strictly BEFORE the entry day, so no stop is ever built
  from information unavailable at entry.

TRUE REPLACEMENT, NOT A FLOOR (an explicit design choice, per the brief's
"otherwise test the Chandelier stop as a true replacement and document the
distinction"): the fixed -12% stop is REMOVED, not kept as a catastrophic
floor underneath the Chandelier stop. Reasoning: (1) this keeps the
comparison apples-to-apples with the ATR trailing-stop study, which was also
a true replacement -- mixing a floor into only the second exit family tested
would confound "did Chandelier help" with "did adding a floor help"; (2) the
user's own sequencing note explicitly anticipates the fixed stop possibly
becoming "a particularly strong control" in its own right if BOTH new exit
families fail -- that comparison is only clean if neither family quietly
reintroduces the control's own mechanism. A floored variant (Chandelier
stop OR -12%, whichever is tighter) is a legitimate follow-up experiment if
wanted, but is NOT what is tested below, and is not needed to answer the
question this script asks: does a volatility-adaptive trailing stop, on its
own, ever beat simply not having one.

SIX IMPLEMENTATION CHECKS -- identical in kind and enforcement location to
the ATR trailing-stop study (see run_backtest_exit_variant's inline
comments), verified again from scratch against synthetic data (including
the same adversarial cases: ratchet monotonicity, a bar whose high AND low
both interact with the stop, gap-through, simultaneous stop/death-cross,
warm-up for the LONGEST lookback (60 days, still trivially satisfied by the
200-day ma200 entry gate), and a still-open position at data end) before
this script was ever pointed at real data:
  1. Stops use only information available before execution.
  2. Intraday stop/entry-exit consistency (exits resolved before entries).
  3. A trailing stop only moves in the favorable direction.
  4. Death-cross and stop-exit have deterministic precedence (stop first).
  5. Open positions are marked to market identically across all variants.
  6. No same-day exit-then-reentry.

METHODOLOGY DISCLOSED UP FRONT (identical conventions to the ATR study, for
direct comparability):
  - Full-history, in-sample-only first pass. No OOS split here -- any
    variant that looks promising goes through the same frozen-split OOS
    discipline used for the sizing thread BEFORE being treated as a
    candidate change. Not done in this script.
  - "Closed-trade" and "NAV" results reported separately (same reasoning as
    before -- unrealized-gain divergence on still-open positions).
  - mfe_high seeds at max(entry-day high, cost-inclusive fill) -- the exact
    fix applied to the ATR trailing-stop script after its first real run
    produced nonsensical negative avg_mfe_capture figures. Verified here
    from the start via the same adversarial low-range regression test.
  - Concentration percentages (top 1%/5%/10%) are SUPPRESSED (reported as
    N/A with the raw dollar figure) whenever a variant's total realized pnl
    is under NEAR_ZERO_PNL_THRESHOLD_PCT of STARTING_CAPITAL, in either
    direction -- the same fix applied to the ATR script's second revision.
  - Entries are NOT guaranteed byte-for-byte identical to the control (see
    ENTRY-SET DIVERGENCE note below) -- capital-timing noise from divergent
    exit histories, not a signal difference. Measured and reported, not
    hidden, exactly as in the ATR study.
  - Top-K removal ranks each variant's OWN trades by pnl and removes that
    variant's own top K, not the control's.
  - Winner-preservation diagnostics (cut-short count, avg/largest winner
    change, MFE-capture change) are reported for every one of the 9
    variants against the control -- this is the primary evidence the user
    asked this script to produce: if Chandelier exits ALSO cut down the
    largest winners and collapse holding periods the way the ATR stops did,
    that is strong evidence the control's exit simplicity is part of the
    strategy's edge, not an omission waiting to be fixed.
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

ATR_LOOKBACKS = (20, 40, 60)              # PREDECLARED grid -- the ATR averaging period
CHANDELIER_MULTIPLIERS = (3, 4, 5)        # PREDECLARED grid -- fixed before any result was seen
TOP_K_REMOVE = (1, 2)
CONCENTRATION_FRACS = (0.01, 0.05, 0.10)
NEAR_ZERO_PNL_THRESHOLD_PCT = 1.0          # see concentration_stats' docstring
# See run_atr_trailing_stop_backtest_equities.py's constants block for the
# full rationale (identical here -- same reporting-layer code, same real-data
# failure mode): a total-pnl-near-zero gate alone doesn't stop the
# concentration ratio from blowing up when total_pnl is merely SMALL relative
# to its own top-k trades, so a share exceeding this bound is ALSO suppressed.
CONCENTRATION_MAX_SANE_ABS_PCT = 300.0
# Same rationale, applied to the per-trade MFE-capture ratio's denominator
# (each trade's own mfe_pct) rather than the concentration ratio's total-pnl
# denominator: a trade whose mfe_pct is this small is excluded from the
# avg/median mfe-capture aggregate (never from the per-trade record).
MFE_CAPTURE_MIN_MFE_PCT = 1.0


# --------------------------------------------------------------------------
# Indicator pipeline: reuse the already-verified base pipeline, then add the
# three variable-lookback ATR columns this study needs (the base pipeline
# only computes a fixed 14-day ATR, used by the sizing/ATR-trailing-stop
# scripts -- this study needs 20/40/60-day ATR instead).
# --------------------------------------------------------------------------

def build_states(price):
    """Adds atr{20,40,60}_lag1 columns, each = (N-day rolling mean of true
    range).shift(1) -- lagged so no stop calculation ever sees same-day ATR.
    true_range is recomputed here (not re-exposed by the imported pipeline)
    using the IDENTICAL formula the base pipeline uses for its own 14-day
    ATR (max of high-low, |high-prev_close|, |low-prev_close|) -- verified
    in the dry run to reproduce the base pipeline's own atr14 exactly when
    rolled over the same 14-day window, confirming the recomputation is
    correct before trusting it at 20/40/60 days."""
    states = _build_states_base(price)
    for state in states.values():
        high, low, close = state["high"], state["low"], state["close"]
        true_range = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        for n in ATR_LOOKBACKS:
            state[f"atr{n}_lag1"] = true_range.rolling(n).mean().shift(1)
    return states


# --------------------------------------------------------------------------
# Engine: same mode-switched design as the ATR trailing-stop study. "FIXED"
# is the byte-for-byte control; "CHANDELIER" reads a caller-selected
# atr{N}_lag1 column and multiplier.
# --------------------------------------------------------------------------

def run_backtest_exit_variant(states, entry_filter_fn, sizing_fn, stop_mode,
                               chandelier_lookback=None, chandelier_multiplier=None,
                               stop_pct=BASELINE_STOP_PCT, entry_cost=ENTRY_COST, exit_cost=EXIT_COST):
    """stop_mode: "FIXED" (control) or "CHANDELIER" (highest-high-since-entry
    minus chandelier_multiplier * ATR_{chandelier_lookback}, lagged one day).
    Entry logic, sizing, cost model, and NAV bookkeeping are identical
    between modes -- only the stop-level computation differs."""
    assert stop_mode in ("FIXED", "CHANDELIER")
    if stop_mode == "CHANDELIER":
        assert chandelier_lookback in ATR_LOOKBACKS
        assert chandelier_multiplier is not None and chandelier_multiplier > 0
    atr_col = f"atr{chandelier_lookback}_lag1" if stop_mode == "CHANDELIER" else None

    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions = {sym: None for sym in states}
    trade_log = []
    nav_history = []
    fallback_anchor_count = 0  # diagnostic: should be 0 on real data

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
                    atr_lag1_today = row[atr_col]
                    if pd.notna(atr_lag1_today) and atr_lag1_today > 0:
                        candidate_stop = pos["hwm"] - chandelier_multiplier * atr_lag1_today
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
                    if stop_mode == "CHANDELIER":
                        # CHECK 1: hwm updated with TODAY's high only AFTER
                        # today's stop check -- affects only TOMORROW's stop.
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
                    # mfe_high seeds at max(today's high, fill) -- see the
                    # ATR trailing-stop script's regression-test writeup for
                    # why plain row["high"] alone is wrong on a low-range day.
                    pos_dict = {"entry_price": fill, "notional": notional, "entry_time": ts,
                                "last_price": fill, "mfe_high": max(row["high"], fill)}
                    if stop_mode == "CHANDELIER":
                        # CHECK 1 (entry side): anchor ATR is atr{N}_lag1 AT
                        # the entry bar -- from strictly BEFORE the entry
                        # day. No stop is built from information unavailable
                        # at entry.
                        atr_at_entry = row[atr_col]
                        if pd.isna(atr_at_entry) or atr_at_entry <= 0:
                            # Safety net only -- should not fire on real data
                            # (even a 60-day ATR warms up long before the
                            # 200-day ma200 entry gate). Falls back to the
                            # fixed stop's own distance so a missing ATR
                            # reading never gates the entry decision itself.
                            fallback_anchor_count += 1
                            atr_at_entry = fill * abs(stop_pct) / 100 / chandelier_multiplier
                        pos_dict["running_stop"] = fill - chandelier_multiplier * atr_at_entry
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
# Reporting -- identical to the (corrected) ATR trailing-stop script.
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
    """See NEAR_ZERO_PNL_THRESHOLD_PCT: percentages are suppressed (None,
    with the raw dollar top-k pnl retained) whenever total realized pnl is
    near zero, in either direction -- the ratio-to-total-pnl definition is
    unstable/uninterpretable there (first surfaced by the ATR trailing-stop
    script's x2 output: -573.8%/-1615.0%/-2412.2% on a near-zero total).
    ALSO suppressed whenever the resulting share exceeds
    CONCENTRATION_MAX_SANE_ABS_PCT even if total_pnl alone clears the
    near-zero gate -- a total merely SMALL relative to its own top-k trades
    produces the identical blowup (see the ATR script's clean-rerun ATR_TRAIL
    x2 output: -588.1%/-1655.2%/-2472.2% at total_pnl=-$1,400, 1.4% of
    capital, above the 1.0% near-zero gate)."""
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
    ATR trailing-stop script's ENTRY-SET DIVERGENCE note: capital-timing
    effects can make a signal-eligible candidate get taken in one variant
    and skipped (insufficient free_capital) in the other. Restricted to the
    intersection; the unmatched count is reported, not hidden."""
    if control_trades.empty or variant_trades.empty:
        return {"n_matched": 0}
    c = control_trades.set_index(["asset", "entry_time"])
    v = variant_trades.set_index(["asset", "entry_time"])
    common = c.index.intersection(v.index)
    n_matched = len(common)
    n_control = len(c)
    note = None if n_matched == n_control else (
        f"{n_control - n_matched} of {n_control} control trades have no counterpart in {variant_label}")
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
    c_mfe_all = c_m["mfe_capture_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    v_mfe_all = v_m["mfe_capture_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    c_mfe = c_mfe_all[(c_m.loc[c_mfe_all.index, "mfe_pct"] >= MFE_CAPTURE_MIN_MFE_PCT).to_numpy()]
    v_mfe = v_mfe_all[(v_m.loc[v_mfe_all.index, "mfe_pct"] >= MFE_CAPTURE_MIN_MFE_PCT).to_numpy()]
    mfe_capture_change = (float(v_mfe.mean()) if len(v_mfe) else float("nan")) - (float(c_mfe.mean()) if len(c_mfe) else float("nan"))
    return {"n_matched": n_matched, "n_control_winners": n_control_winners, "n_cut_short": n_cut_short,
            "pct_winners_cut_short": pct_winners_cut_short, "avg_winner_pnl_change": avg_winner_change,
            "largest_winner_pnl_change": largest_winner_change, "mfe_capture_pct_change": mfe_capture_change,
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

    return nav_summ, closed_summ, wp if control_trades is not None else None


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

    control_trades, control_nav, _ = run_backtest_exit_variant(states, entry_filter_fn, sizing_fn, stop_mode="FIXED")
    print_variant_report("CONTROL (fixed -12% stop + death cross)", control_trades, control_nav, breadth)

    control_set = _trade_set(control_trades)
    control_nav_summ = summarize_nav(control_trades, control_nav)
    variant_results = {}
    for n in ATR_LOOKBACKS:
        for k in CHANDELIER_MULTIPLIERS:
            trades, nav_series, fallback_count = run_backtest_exit_variant(
                states, entry_filter_fn, sizing_fn, stop_mode="CHANDELIER",
                chandelier_lookback=n, chandelier_multiplier=k,
            )
            label = f"CHANDELIER N={n} k={k}"
            variant_set = _trade_set(trades)
            divergence = control_set.symmetric_difference(variant_set)
            if divergence:
                print(f"\n  NOTE: {label} trade set differs from control on {len(divergence)} of "
                      f"{len(control_set | variant_set)} total (symbol, entry_time) pairs -- expected capital-"
                      f"timing divergence (see module docstring), not a signal difference.")
            if fallback_count:
                print(f"\n  NOTE: {label} used the fixed-stop-distance fallback anchor on {fallback_count} "
                      f"entries (ATR_{n} was NaN/<=0 at entry) -- unexpected on real data.")
            nav_summ, closed_summ, wp = print_variant_report(label, trades, nav_series, breadth,
                                                              control_trades, "CONTROL")
            variant_results[(n, k)] = (nav_summ, closed_summ, wp)

    print("\n" + "=" * 90)
    print("SUMMARY: does any Chandelier variant beat the control?")
    print("=" * 90)
    control_closed_summ = closed_trade_only_stats(control_trades)
    print(f"  CONTROL: nav_cagr={control_nav_summ.get('cagr_pct', float('nan')):.2f}%  "
          f"sharpe={control_nav_summ.get('sharpe', float('nan')):.2f}  sortino={control_nav_summ.get('sortino', float('nan')):.2f}  "
          f"closed_cagr={control_closed_summ.get('cagr_pct', float('nan')):.2f}%")
    any_beats = False
    cut_short_pcts = []
    for n in ATR_LOOKBACKS:
        for k in CHANDELIER_MULTIPLIERS:
            nav_summ, closed_summ, wp = variant_results[(n, k)]
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
                cut_short_pcts.append(wp.get("pct_winners_cut_short", float("nan")))
            print(f"  CHANDELIER N={n} k={k}: nav_cagr={nav_summ.get('cagr_pct', float('nan')):.2f}%  "
                  f"sharpe={nav_summ.get('sharpe', float('nan')):.2f}  sortino={nav_summ.get('sortino', float('nan')):.2f}  "
                  f"cut_short={wp.get('pct_winners_cut_short', float('nan')):.1f}% -> {verdict}")

    if any_beats:
        closing_note = ("at least one Chandelier variant clearly beats the control -- worth deeper OOS "
                         "validation before adopting.")
    else:
        avg_cut_short = float(np.nanmean(cut_short_pcts)) if cut_short_pcts else float("nan")
        closing_note = (
            f"no Chandelier variant beats the control on both CAGR and risk-adjusted performance (average "
            f"winner-cut-short rate across all 9 variants: {avg_cut_short:.1f}%). Per the predeclared "
            f"sequencing, the fixed -12% stop + death cross becomes a particularly strong control: TWO "
            f"independent volatility-adaptive trailing-stop families (ATR and Chandelier) both failed by "
            f"cutting short the same rare long winners the control depends on, which is evidence the "
            f"control's exit simplicity is part of the strategy's edge, not an omission waiting to be fixed. "
            f"Per the user's own instruction to keep breadth-based exposure reduction as a separate research "
            f"thread, the next predeclared items are time-based exits for stagnant positions and partial "
            f"profit-taking, in that order -- not a reopening of ATR or Chandelier with wider parameters."
        )
    print(f"\n  Per the predeclared sequencing: {closing_note}")


if __name__ == "__main__":
    main()
