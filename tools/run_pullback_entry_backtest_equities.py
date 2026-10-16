"""Backtests a "wait for the pullback" entry rule against this project's
existing golden-cross baseline (immediate entry on the 50/200-day cross),
for the same 60-symbol robust equities/commodities/bonds universe this
project already trades live (see tools/run_alpaca_paper_trader.py's
EQUITIES_HARD_STOP_PCT = -12.0%, which this script's BASELINE run
reproduces exactly).

Source of the pullback rule: a user-supplied summary of a standard
"buy the pullback, not the cross" approach (testfol.io / vtmarkets.com),
paraphrased here as two testable variants:

  PULLBACK-SIMPLE (the article's own "simple rules-based version"):
    Buy the first confirmed pullback while price and the 50-day average
    remain above the RISING 200-day average -- i.e. don't chase the
    initial breakout; wait for price to retrace toward a rising 50-day or
    200-day average, then enter on a bullish reversal confirmation.

  PULLBACK-FULL (the article's full checklist): PULLBACK-SIMPLE's rule
    PLUS the 50-day average must also be rising, price structure must
    show higher highs and higher lows, and the rebound must come on
    above-average volume.

Only ONE entry is taken per golden-cross regime (the "first" confirmed
pullback, per the article's own wording) -- if price never pulls back
before the trend dies (death cross) with no entry taken, that regime is
recorded as a skipped/untraded cross, not forced into a trade.

Exit rule (kept IDENTICAL in shape to the baseline this project already
validated, so only the entry timing + stop method actually differ):
death cross (ma50 crosses below ma200) OR a volatility-based stop
(entry_price - ATR_MULT * ATR14-at-entry breached intrabar), whichever
comes first.

=== ATR_MULT SWEEP (added after the first run) ===
The first run (ATR_MULT=1.5, the middle of the article's own "roughly
1-2 ATR below support" range) showed the entry-timing rule was NOT the
problem -- trades that survived to a real death-cross exit averaged
+51-52% over a median 439-503 day hold, clearly better than baseline's
average winner. The problem was the stop: 77-79% of all pullback trades
were getting stopped out at a median of just 11-12 days, for an average
-3% loss each, before the trend had any chance to develop. That's what
crushed CAGR (8.5% baseline -> 3.4%/2.3%) despite a fine profit factor.
ATR_MULT_SWEEP below re-runs both variants across a much wider range to
see whether that entry-timing edge survives once trades are given room
to breathe, and where (if anywhere) the stop-out rate stops dominating
the result.

Indicators are computed ONCE per symbol (build_states()) and reused
across every sweep point -- only the stop DISTANCE changes between runs,
not any of the underlying trend/pullback/breakout signal computation, so
the sweep is fast and each point differs from its neighbors by exactly
one thing.

Explicitly-disclosed simplifications (this is a backtest of a paraphrased
strategy, not a certified implementation of it):
  - "Both averages are rising, not flat" -> ma200 higher than
    MA200_SLOPE_LOOKBACK (20 trading days) ago; ma50 higher than
    MA50_SLOPE_LOOKBACK (10 days) ago. A slope proxy, not a statistical
    trend test.
  - "Market making higher highs and higher lows" -> this trading day's
    rolling STRUCTURE_LOOKBACK-day (20) high/low vs. the SAME window
    STRUCTURE_LOOKBACK days earlier. A coarse two-window comparison, not
    real swing-point detection.
  - "Pullback toward the 50 or 200 day average" -> close within
    PULLBACK_TOLERANCE_PCT (2%) of ma50 or ma200 (on either side).
  - "Close back above the prior day's high" is the only entry-trigger
    variant implemented (the article lists several; this is the most
    directly backtestable one).
  - "Stronger than normal volume" -> today's volume > its own trailing
    VOLUME_LOOKBACK-day (20) average. PULLBACK-FULL only.
  - The article's alternative stop ("below the recent swing low" instead
    of ATR) is still NOT implemented -- this sweep only varies the ATR
    multiplier, not the stop METHOD. Worth a separate follow-up.

Reuses this project's own already-validated pieces rather than
reimplementing them: compute_trend_state/ENTRY_COST/EXIT_COST from
tools.run_cross_asset_golden_cross_backtest, the robust-universe loader
and OHLC fetch pattern from tools.run_stoploss_tail_risk_test_equities,
and calls that module's own run_backtest(stop_pct=-12.0) directly for
the BASELINE row so it is byte-for-byte the same simulation the live
bot's stop setting was validated against -- not a re-derived copy that
could quietly drift from it.

NOTE on exit_reason labeling (bug in the first version of this script,
fixed here): the pullback backtest's own stop exits are now labeled
"stop" (not the first version's "atr_stop"), matching baseline's own
convention, PLUS this script now has its own summarize_pullback() that
reports the stop-exit rate directly instead of silently reading as zero
the way the first run's reused summarize() did (it only ever recognized
the literal string "stop").

=== HYBRID STOP (added after the ATR_MULT sweep) ===
The sweep above showed a structural problem, not a tuning problem: a pure
ATR-distance stop has no percentage ceiling, so widening it to cut the
77-79% premature-stop-out rate also let the worst single trade run from
-17.0% (ATR x1.5) to -43.9% (ATR x8) -- far past baseline's fixed -12%
cap -- and CAGR/Sharpe never beat baseline anywhere in the range as a
result.

HYBRID_FLOOR_PCT_SWEEP tests whether capping the ATR distance with a
fixed percentage floor recovers the entry-timing edge (signal-exit
trades averaged +30% to +52% across every variant tested, vs. baseline's
average winner) while restoring a bounded worst case. The rule: take
whichever stop is TIGHTER (closer to entry) -- atr_mult x ATR14, or the
fixed floor:

    atr_stop_price   = fill - atr_mult * atr14_at_entry
    floor_stop_price = fill * (1 + floor_pct / 100)
    stop_price       = max(atr_stop_price, floor_stop_price)   # higher = tighter

At -12% (HYBRID_FLOOR_PCT_SWEEP's first value), the floor is IDENTICAL to
baseline's own cap, so any variant that beats baseline here is winning on
entry timing alone, with the exact same worst-case guarantee baseline
has. -15% (one increment looser, reusing the same spacing used elsewhere
in this project's own stop-distance grids) tests whether a little extra
room still keeps tail risk acceptable while cutting premature stop-outs
further.

Each trade's stop_bound ("floor" or "atr") records which side of the max()
actually bound at entry, so the results show how often the floor is doing
real work (vs. the ATR distance already being tighter than it on its own)
rather than leaving that as a guess.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests

from tools.run_cross_asset_golden_cross_backtest import ENTRY_COST, EXIT_COST, compute_trend_state
from tools.run_stoploss_tail_risk_test_equities import (
    STARTING_CAPITAL,
    PER_TRADE_ALLOCATION_PCT,
    _load_equities_universe,
    run_backtest as run_baseline_backtest,
    summarize,
)

BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 2.0
OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "pullback_entry_backtest_results.json"

# --- pullback-rule parameters (all disclosed above) ---
PULLBACK_TOLERANCE_PCT = 2.0
MA200_SLOPE_LOOKBACK = 20
MA50_SLOPE_LOOKBACK = 10
STRUCTURE_LOOKBACK = 20
VOLUME_LOOKBACK = 20
ATR_LOOKBACK = 14

# The sweep: 1.5 is kept as the first run's reference point; the rest widen
# out based on that run's finding (stop was clearly the bottleneck, not the
# entry timing).
ATR_MULT_SWEEP = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

BASELINE_STOP_PCT = -12.0  # matches run_alpaca_paper_trader.py's EQUITIES_HARD_STOP_PCT

# Hybrid stop: cap the ATR distance with a fixed pct floor (whichever is
# tighter). -12% is baseline's own cap (most direct apples-to-apples
# comparison); -15% is one increment looser, reusing this project's own
# existing stop-distance grid spacing (see run_stoploss_tail_risk_test_equities's
# STOP_DISTANCES_PCT) rather than an arbitrary new number.
HYBRID_FLOOR_PCT_SWEEP = [-12.0, -15.0]


def fetch_ohlcv_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Same batching/session pattern as run_stoploss_tail_risk_test_equities's
    fetch_ohlc_batch, extended to also keep Volume and Open (needed for the
    ATR calc and the "prior day's high" breakout trigger)."""
    session = cffi_requests.Session(impersonate="chrome")
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        try:
            data = yf.download(
                batch, period="max", interval="1d", session=session,
                progress=False, group_by="ticker", threads=False,
            )
        except Exception as exc:
            print(f"batch failed: {exc}")
            time.sleep(BATCH_DELAY_SECONDS)
            continue
        for sym in batch:
            try:
                df = data[sym] if len(batch) > 1 else data
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                df.columns = ["open", "high", "low", "close", "volume"]
            except (KeyError, TypeError):
                continue
            if len(df) < 210:
                continue
            out[sym] = df
        time.sleep(BATCH_DELAY_SECONDS)
    return out


def compute_pullback_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds every column the pullback rule needs on top of compute_trend_state's
    close/ma50/ma200/above200/cross_up/cross_down. Does NOT depend on
    ATR_MULT -- the raw atr14 series is computed once here, and only
    multiplied by whichever ATR_MULT the sweep is currently testing at
    entry time in run_pullback_backtest(). This is what lets build_states()
    be called once and reused across the whole sweep."""
    state = compute_trend_state(df["close"])
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    ma200_rising = state["ma200"] > state["ma200"].shift(MA200_SLOPE_LOOKBACK)
    ma50_rising = state["ma50"] > state["ma50"].shift(MA50_SLOPE_LOOKBACK)

    roll_high = high.rolling(STRUCTURE_LOOKBACK).max()
    roll_low = low.rolling(STRUCTURE_LOOKBACK).min()
    higher_high = roll_high > roll_high.shift(STRUCTURE_LOOKBACK)
    higher_low = roll_low > roll_low.shift(STRUCTURE_LOOKBACK)
    structure_ok = higher_high & higher_low

    avg_volume = volume.rolling(VOLUME_LOOKBACK).mean()
    volume_confirm = volume > avg_volume

    near_ma50 = (close - state["ma50"]).abs() <= state["ma50"] * (PULLBACK_TOLERANCE_PCT / 100)
    near_ma200 = (close - state["ma200"]).abs() <= state["ma200"] * (PULLBACK_TOLERANCE_PCT / 100)
    in_pullback_zone = (near_ma50 | near_ma200) & (close >= state["ma200"] * (1 - PULLBACK_TOLERANCE_PCT / 100))

    breakout_confirm = close > high.shift(1)

    true_range = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = true_range.rolling(ATR_LOOKBACK).mean()

    out = state.copy()
    out["high"], out["low"], out["volume"] = high, low, volume
    out["ma200_rising"] = ma200_rising.fillna(False)
    out["ma50_rising"] = ma50_rising.fillna(False)
    out["structure_ok"] = structure_ok.fillna(False)
    out["volume_confirm"] = volume_confirm.fillna(False)
    out["in_pullback_zone"] = in_pullback_zone.fillna(False)
    out["breakout_confirm"] = breakout_confirm.fillna(False)
    out["atr14"] = atr14
    # Core trend condition common to BOTH variants (article step 1, minus
    # the higher-highs/higher-lows structure check, which PULLBACK-FULL
    # adds on top): price and the 50-day average above a RISING 200-day
    # average.
    out["trend_confirmed"] = out["above200"] & out["ma200_rising"]
    return out


def build_states(price: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {sym: compute_pullback_indicators(df) for sym, df in price.items()}


def run_pullback_backtest(
    states: dict[str, pd.DataFrame], require_full_checklist: bool, atr_mult: float,
    floor_pct: float | None = None,
) -> pd.DataFrame:
    """Same cash-constrained multi-asset event loop shape as
    run_stoploss_tail_risk_test_equities.run_backtest (2% sizing, reject an
    entry that would exceed capital minus what's already deployed), but with
    the pullback entry rule and an ATR-based stop (atr_mult x atr14-at-entry)
    instead of immediate-entry + fixed-percent stop. Takes PRECOMPUTED
    indicator states (see build_states()) so the sweep doesn't redo the
    rolling-window computation for every atr_mult tested.

    floor_pct: if given (e.g. -12.0), the stop is the HYBRID stop -- whichever
    is tighter (closer to entry) between the pure ATR distance and this fixed
    percentage floor. None (the default) keeps the original pure-ATR stop
    with no ceiling, unchanged from the first ATR_MULT sweep. Each position
    records which side actually bound (pos["stop_is_floor"]), and that's
    carried into the trade log as "stop_bound" so the results can show how
    often the floor is doing real work vs. the ATR distance already being
    tighter on its own."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions: dict[str, dict | None] = {sym: None for sym in states}
    # "armed" = a golden-cross regime is active for this symbol and no entry
    # has been taken in it yet -- only the FIRST qualifying pullback day
    # inside a regime is tradeable (see module docstring).
    armed: dict[str, bool] = {sym: False for sym in states}
    trade_log = []
    skipped_regimes = {sym: 0 for sym in states}

    for ts in all_index:
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional

        for sym, state in states.items():
            if ts not in state.index:
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]) or pd.isna(row["atr14"]):
                continue
            pos = positions[sym]

            if pos is not None:
                stop_hit = row["low"] <= pos["stop_price"]
                if stop_hit:
                    exit_price = pos["stop_price"] * (1 - EXIT_COST)
                    reason = "stop"  # matches baseline's own exit_reason convention
                    stop_bound = "floor" if pos.get("stop_is_floor") else "atr"
                elif row["cross_down"]:
                    exit_price = row["close"] * (1 - EXIT_COST)
                    reason = "signal"
                    stop_bound = None
                else:
                    continue
                ret = (exit_price / pos["entry_price"]) - 1
                pnl = pos["notional"] * ret
                capital += pnl
                trade_log.append({
                    "asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                    "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital,
                    "exit_reason": reason, "stop_bound": stop_bound,
                })
                positions[sym] = None
                free_capital += pos["notional"]
                continue

            # No open position. Track regime arming off the SAME cross_up/
            # cross_down events the baseline uses, so "one entry per regime"
            # lines up with baseline's "one trade per regime" for a fair
            # trade-count comparison.
            if row["cross_up"]:
                armed[sym] = True
            elif row["cross_down"] and armed[sym]:
                # Trend died before a qualifying pullback ever showed up --
                # this regime is recorded as untraded, not forced.
                armed[sym] = False
                skipped_regimes[sym] += 1

            if not armed[sym]:
                continue

            checklist = (
                row["trend_confirmed"] and row["in_pullback_zone"] and row["breakout_confirm"]
                and (not require_full_checklist or (row["ma50_rising"] and row["structure_ok"] and row["volume_confirm"]))
            )
            if not checklist:
                continue

            notional = capital * PER_TRADE_ALLOCATION_PCT / 100.0
            if notional <= 100 or notional > free_capital:
                continue
            fill = row["close"] * (1 + ENTRY_COST)
            atr_stop_price = fill - atr_mult * row["atr14"]
            if floor_pct is not None:
                floor_stop_price = fill * (1 + floor_pct / 100)
                stop_price = max(atr_stop_price, floor_stop_price)  # tighter (higher) wins
                stop_is_floor = floor_stop_price > atr_stop_price
            else:
                stop_price = atr_stop_price
                stop_is_floor = False
            positions[sym] = {
                "entry_price": fill, "notional": notional, "entry_time": ts,
                "stop_price": stop_price, "stop_is_floor": stop_is_floor,
            }
            free_capital -= notional
            armed[sym] = False  # this regime's one shot is used

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    trades.attrs["skipped_regimes"] = sum(skipped_regimes.values())
    return trades


def summarize_pullback(trades: pd.DataFrame, label: str, verbose: bool = True) -> dict:
    """Same core metrics as run_stoploss_tail_risk_test_equities.summarize()
    (n_trades, win_rate, pf, cagr_pct, total_return_pct, max_dd_pct,
    worst_trade_pct, sharpe), PLUS an honest stop-exit breakdown -- added
    after the first run's console output silently showed "stop exits: 0"
    for every pullback variant because the reused summarize() only ever
    recognized the exact string "stop", and the first version of this
    script labeled its stop exits "atr_stop". That's fixed at the source
    now (run_pullback_backtest labels them "stop"), but this function
    reports the breakdown explicitly anyway so a future relabeling
    mistake fails loudly instead of silently again."""
    if trades.empty:
        if verbose:
            print(f"{label}: no trades")
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

    worst_trade = trades["ret_pct"].min()

    by_reason = trades.copy()
    by_reason["days_held"] = (by_reason["exit_time"] - by_reason["entry_time"]).dt.days
    stop_mask = by_reason["exit_reason"] == "stop"
    n_stop = int(stop_mask.sum())
    n_signal = int((~stop_mask).sum())
    stop_exit_pct = n_stop / len(trades) * 100
    median_days_stop = float(by_reason.loc[stop_mask, "days_held"].median()) if n_stop else float("nan")
    median_days_signal = float(by_reason.loc[~stop_mask, "days_held"].median()) if n_signal else float("nan")
    avg_ret_stop = float(by_reason.loc[stop_mask, "ret_pct"].mean()) if n_stop else float("nan")
    avg_ret_signal = float(by_reason.loc[~stop_mask, "ret_pct"].mean()) if n_signal else float("nan")

    # Hybrid-stop diagnostic: which side of max(atr_stop, floor_stop) actually
    # bound at entry, for stop exits. Absent (None) for runs with floor_pct=None
    # (the original pure-ATR sweep), where "stop_bound" isn't in the trade log.
    n_floor_bound = n_atr_bound = None
    if n_stop and "stop_bound" in by_reason.columns:
        stop_rows = by_reason.loc[stop_mask]
        n_floor_bound = int((stop_rows["stop_bound"] == "floor").sum())
        n_atr_bound = int((stop_rows["stop_bound"] == "atr").sum())

    if verbose:
        print(f"\n===== {label} =====")
        print(f"trades: {len(trades)}  (stop exits: {n_stop}, {stop_exit_pct:.0f}%)  "
              f"win_rate: {len(wins)/len(trades)*100:.1f}%")
        print(f"profit_factor: {pf:.2f}  CAGR: {cagr*100:.2f}%  total_return: {total_return:,.1f}%")
        print(f"max_drawdown: {max_dd:.1f}%  worst_single_trade: {worst_trade:.1f}%  "
              f"Sharpe (per-trade, not annualized): {sharpe:.2f}")
        print(f"median days held -- stop exits: {median_days_stop:.0f}, signal exits: {median_days_signal:.0f}")
        print(f"avg return -- stop exits: {avg_ret_stop:+.1f}%, signal exits: {avg_ret_signal:+.1f}%")
        if n_floor_bound is not None:
            print(f"stop exits bound by: floor {n_floor_bound} ({n_floor_bound/n_stop*100:.0f}%), "
                  f"ATR {n_atr_bound} ({n_atr_bound/n_stop*100:.0f}%)")

    result = {
        "n_trades": len(trades), "n_stop": n_stop, "n_signal": n_signal,
        "stop_exit_pct": stop_exit_pct, "win_rate": len(wins) / len(trades) * 100,
        "pf": pf, "cagr_pct": cagr * 100, "total_return_pct": total_return,
        "max_dd_pct": max_dd, "worst_trade_pct": worst_trade, "sharpe": sharpe,
        "median_days_stop": median_days_stop, "median_days_signal": median_days_signal,
        "avg_ret_stop_pct": avg_ret_stop, "avg_ret_signal_pct": avg_ret_signal,
    }
    if n_floor_bound is not None:
        result["n_floor_bound_stops"] = n_floor_bound
        result["n_atr_bound_stops"] = n_atr_bound
    return result


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    price_hlc_only = {sym: df[["high", "low", "close"]] for sym, df in price.items()}

    print("=" * 78)
    print("Running BASELINE (identical to the live bot: immediate entry on golden")
    print(f"cross, fixed {BASELINE_STOP_PCT:.0f}% stop) -- this is run_stoploss_tail_")
    print("risk_test_equities.py's own run_backtest(), called directly, not")
    print("re-derived, so it can't quietly drift from what's actually live.")
    print("=" * 78)
    baseline_trades = run_baseline_backtest(price_hlc_only, BASELINE_STOP_PCT)
    baseline_summary = summarize(baseline_trades, "BASELINE (live-equivalent, -12% stop)")

    print("\nprecomputing pullback indicators once for the whole sweep...")
    states = build_states(price)

    sweep_results: list[dict] = []
    for atr_mult in ATR_MULT_SWEEP:
        print("\n" + "=" * 78)
        print(f"ATR_MULT = {atr_mult}")
        print("=" * 78)
        for require_full, variant_label in [(False, "PULLBACK-SIMPLE"), (True, "PULLBACK-FULL")]:
            trades = run_pullback_backtest(states, require_full_checklist=require_full, atr_mult=atr_mult)
            skipped = trades.attrs.get("skipped_regimes", 0) if not trades.empty else None
            label = f"{variant_label} (ATR x{atr_mult})"
            summ = summarize_pullback(trades, label)
            if skipped is not None:
                print(f"  ({skipped} regimes skipped -- no qualifying pullback found)")
            sweep_results.append({"variant": variant_label, "atr_mult": atr_mult, **summ})

    print("\n" + "=" * 78)
    print("SWEEP SUMMARY (baseline shown once for reference)")
    print("=" * 78)
    header = (f"{'Variant':<16}{'ATRx':>6}{'Trades':>8}{'Stop%':>7}{'WinRate':>9}{'PF':>7}"
              f"{'CAGR':>8}{'TotalRet':>11}{'MaxDD':>8}{'WorstTrd':>9}")
    print(header)
    if baseline_summary:
        r = baseline_summary
        print(f"{'BASELINE':<16}{'-12%':>6}{r['n_trades']:>8}{'n/a':>7}{r['win_rate']:>8.1f}%"
              f"{r['pf']:>7.2f}{r['cagr_pct']:>7.1f}%{r['total_return_pct']:>10.1f}%"
              f"{r['max_dd_pct']:>7.1f}%{r['worst_trade_pct']:>8.1f}%")
    for row in sweep_results:
        if row.get("n_trades") is None:
            print(f"{row['variant']:<16}{row['atr_mult']:>6}  (no trades)")
            continue
        print(f"{row['variant']:<16}{row['atr_mult']:>6}{row['n_trades']:>8}{row['stop_exit_pct']:>6.0f}%"
              f"{row['win_rate']:>8.1f}%{row['pf']:>7.2f}{row['cagr_pct']:>7.1f}%"
              f"{row['total_return_pct']:>10.1f}%{row['max_dd_pct']:>7.1f}%{row['worst_trade_pct']:>8.1f}%")

    print("\n" + "=" * 78)
    print("HYBRID STOP SWEEP -- stop = whichever is TIGHTER: atr_mult x ATR14,")
    print("or a fixed pct floor. Floor -12% = baseline's own cap exactly (any")
    print("edge here is entry-timing alone, same worst-case guarantee as live).")
    print("=" * 78)
    hybrid_results: list[dict] = []
    for floor_pct in HYBRID_FLOOR_PCT_SWEEP:
        for atr_mult in ATR_MULT_SWEEP:
            print("\n" + "-" * 78)
            print(f"floor = {floor_pct:.0f}%   ATR_MULT = {atr_mult}")
            print("-" * 78)
            for require_full, variant_label in [(False, "PULLBACK-SIMPLE"), (True, "PULLBACK-FULL")]:
                trades = run_pullback_backtest(
                    states, require_full_checklist=require_full, atr_mult=atr_mult, floor_pct=floor_pct,
                )
                skipped = trades.attrs.get("skipped_regimes", 0) if not trades.empty else None
                label = f"{variant_label} (ATR x{atr_mult}, floor {floor_pct:.0f}%)"
                summ = summarize_pullback(trades, label)
                if skipped is not None:
                    print(f"  ({skipped} regimes skipped -- no qualifying pullback found)")
                hybrid_results.append({
                    "variant": variant_label, "atr_mult": atr_mult, "floor_pct": floor_pct, **summ,
                })

    print("\n" + "=" * 78)
    print("HYBRID SWEEP SUMMARY (baseline shown once for reference)")
    print("=" * 78)
    hybrid_header = (f"{'Variant':<16}{'Floor':>7}{'ATRx':>6}{'Trades':>8}{'Stop%':>7}{'Floor%':>7}"
                     f"{'WinRate':>9}{'PF':>7}{'CAGR':>8}{'TotalRet':>11}{'MaxDD':>8}{'WorstTrd':>9}")
    print(hybrid_header)
    if baseline_summary:
        r = baseline_summary
        print(f"{'BASELINE':<16}{'-12%':>7}{'n/a':>6}{r['n_trades']:>8}{'n/a':>7}{'n/a':>7}{r['win_rate']:>8.1f}%"
              f"{r['pf']:>7.2f}{r['cagr_pct']:>7.1f}%{r['total_return_pct']:>10.1f}%"
              f"{r['max_dd_pct']:>7.1f}%{r['worst_trade_pct']:>8.1f}%")
    for row in hybrid_results:
        if row.get("n_trades") is None:
            print(f"{row['variant']:<16}{row['floor_pct']:>6.0f}%{row['atr_mult']:>6}  (no trades)")
            continue
        n_stop = row.get("n_stop", 0) or 0
        n_floor_bound = row.get("n_floor_bound_stops")
        floor_of_stops_pct = (n_floor_bound / n_stop * 100) if n_stop and n_floor_bound is not None else float("nan")
        print(f"{row['variant']:<16}{row['floor_pct']:>6.0f}%{row['atr_mult']:>6}{row['n_trades']:>8}"
              f"{row['stop_exit_pct']:>6.0f}%{floor_of_stops_pct:>6.0f}%"
              f"{row['win_rate']:>8.1f}%{row['pf']:>7.2f}{row['cagr_pct']:>7.1f}%"
              f"{row['total_return_pct']:>10.1f}%{row['max_dd_pct']:>7.1f}%{row['worst_trade_pct']:>8.1f}%")
    print("(Floor% = share of STOP exits bound by the fixed floor rather than the ATR distance)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "universe_size": len(price),
        "params": {
            "pullback_tolerance_pct": PULLBACK_TOLERANCE_PCT,
            "ma200_slope_lookback": MA200_SLOPE_LOOKBACK,
            "ma50_slope_lookback": MA50_SLOPE_LOOKBACK,
            "structure_lookback": STRUCTURE_LOOKBACK,
            "volume_lookback": VOLUME_LOOKBACK,
            "atr_lookback": ATR_LOOKBACK,
            "atr_mult_sweep": ATR_MULT_SWEEP,
            "hybrid_floor_pct_sweep": HYBRID_FLOOR_PCT_SWEEP,
            "baseline_stop_pct": BASELINE_STOP_PCT,
            "per_trade_allocation_pct": PER_TRADE_ALLOCATION_PCT,
        },
        "baseline_summary": baseline_summary,
        "sweep": sweep_results,
        "hybrid_sweep": hybrid_results,
    }, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
