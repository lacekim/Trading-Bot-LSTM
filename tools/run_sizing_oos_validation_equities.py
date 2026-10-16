"""Out-of-sample validation of the four sizing candidates that survived the
comparable-risk test, run BEFORE any exit-logic research and with every
parameter frozen in advance of seeing these results.

Per the user's explicit conclusion from the in-sample sizing study:
  - FILTER-FULL's apparent edge disappeared under vol-matching (it was an
    exposure/cash-allocation artifact) -- dropped from this test entirely.
  - POSITION-CAP's low native drawdown was under-deployment, not superior
    diversification (it got WORSE, not better, once matched to baseline's
    own exposure/vol) -- also dropped.
  - The remaining four are frozen as the entire candidate set:
      1. BASELINE / EQUAL-DOLLAR   (the live control)
      2. BASELINE / VOL-TARGET
      3. BASELINE / EQUAL-RISK-ATR
      4. BASELINE / INVERSE-VOL
Carrying the two failed candidates forward would only add multiple-testing
pressure with no plausible payoff, exactly as flagged.

FROZEN BEFORE VIEWING RESULTS (do not change after seeing the numbers):
  OOS_SPLIT_DATE = 2023-01-01. Everything at or after this date is the
  out-of-sample test window; this script has not been run, and this window
  has not been examined at the trade level, before this comment was written.
  Calibration inputs to EQUAL-RISK-ATR/INVERSE-VOL (the universe-median
  ATR%/vol20 they size against) are computed using ONLY data strictly
  before OOS_SPLIT_DATE, so no information from the test window leaks into
  the sizing formula itself -- the formula is fixed by the in-sample period
  and then simply applied, unchanged, going forward.
  MATERIALITY THRESHOLDS for the predetermined decision rule (also fixed
  now, not tuned after seeing results):
    SHARPE_SORTINO_MATERIAL_DELTA = 0.10 (absolute units)
    CAGR_SIMILAR_BAND_PP = 1.0 (percentage points)

Engine reuse: run_backtest, make_sizing_fn, ENTRY_SETS, build_states, and
every stop/entry/cost rule are IMPORTED UNCHANGED from the already-verified
run_sizing_allocation_backtest_equities.py -- no engine or decision logic is
re-implemented here. The one addition is run_backtest_instrumented, a copy
of that same engine that additionally records a per-day exposure/NAV trace
(needed to slice exposure and volatility to the OOS window specifically,
which the original script only reports as a whole-run average). This copy
is verified bit-identical to the imported run_backtest on synthetic data
before being trusted (see the dry run), exactly as this thread's earlier
"instrumented baseline" diagnostic was.

Methodology disclosed up front, not discovered by the reader later:
  - NAV-based metrics (CAGR, Sharpe, Sortino, max drawdown, worst day,
    realized vol, exposure) for the "OOS window" are computed from the
    OOS-dated SLICE of each candidate's full-history NAV series (the
    backtest is run over the full history so positions open before the
    split carry through correctly; only the reporting window is sliced).
    This is what a live investor who started paying attention to the
    strategy's returns on OOS_SPLIT_DATE would actually have experienced,
    including whatever pre-existing positions were still open.
  - Trade-level metrics (concentration, regime split, cost-divergence
    count, top-winner sensitivity) use only trades whose ENTRY_TIME falls
    inside the OOS window -- i.e. decisions actually made during the test
    period, not older positions that merely happen to close during it.
  - "Performance with/without the largest few winners": a simplified,
    disclosed approximation. It recomputes a REALIZED-PNL-ONLY capital path
    over the OOS window with the top-K OOS trades' pnl zeroed out, in
    exit-time order. It does NOT re-run the engine with those entries
    blocked, so it ignores the (generally small) second-order effect that
    a different capital trajectory would have changed subsequent position
    sizes. Good enough to show sensitivity to a few outsized trades; not a
    genuine counterfactual backtest. Also note: "top-K by pnl" means the K
    best-performing trades, not necessarily K winners -- on a thin OOS
    sample with few or no profitable trades, this can include small losers,
    and excluded_pnl_share_of_total_pct is not guaranteed to move
    monotonically with k when total OOS pnl is itself small or negative
    (dividing by a near-zero or negative denominator can even flip its
    sign). Read the raw dollar figures alongside the percentage in that
    case rather than the percentage alone.
  - "Results by market regime": regime is the day's golden-cross BREADTH --
    the fraction of the 60-symbol universe currently above its own 200dma
    -- computed once from indicators the engine already produces (no new
    external benchmark series). Trades are bucketed into terciles of this
    breadth measure, computed from the OOS window's own distribution.
  - Chronological subperiods are plain calendar-year buckets within the OOS
    window (simple sums of trade P&L per year, not a re-compounded NAV per
    year), to see whether conclusions hold up year by year or hinge on one
    of them.

Caveats NOT resolved here (disclosed, not hidden): still the same 60-symbol
universe and 12bps cost model as every earlier script in this thread; no
delisted-asset accounting; one holdout split, not a walk-forward study; the
OOS window's length is whatever data happens to exist after 2023-01-01, not
independently chosen for statistical power.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tools.run_sizing_allocation_backtest_equities import (
    ENTRY_COST, EXIT_COST, STARTING_CAPITAL, PER_TRADE_ALLOCATION_PCT,
    BASELINE_STOP_PCT, VOL_TARGET_ANNUAL_PCT, VOL_LOOKBACK_DAYS, VOL_THROTTLE_BOUNDS,
    ENTRY_SETS, build_states, make_sizing_fn, run_backtest, concentration_table,
)
from tools.run_stoploss_tail_risk_test_equities import _load_equities_universe
from tools.run_pullback_entry_backtest_equities import fetch_ohlcv_batch

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "sizing_oos_validation_results.json"

OOS_SPLIT_DATE = pd.Timestamp("2023-01-01")
CANDIDATES = ["EQUAL-DOLLAR", "VOL-TARGET", "EQUAL-RISK-ATR", "INVERSE-VOL"]
REFERENCE_CANDIDATE = "EQUAL-DOLLAR"
SHARPE_SORTINO_MATERIAL_DELTA = 0.10
CAGR_SIMILAR_BAND_PP = 1.0
TOP_K_SENSITIVITY = (1, 3, 5)


def compute_calibration_before(states: dict[str, pd.DataFrame], split_ts: pd.Timestamp) -> tuple[float, float]:
    """Same definition as compute_calibration in the sizing script (universe-
    median atr_pct_lag1/vol20_lag1 at cross_up days), but restricted to
    cross_up days strictly before split_ts, so nothing from the OOS window
    can leak into the sizing formula's calibration."""
    atr_pcts, vols = [], []
    for state in states.values():
        cross_rows = state[state["cross_up"] & (state.index < split_ts)]
        atr_pcts.append(cross_rows["atr_pct_lag1"].dropna())
        vols.append(cross_rows["vol20_lag1"].dropna())
    atr_series = pd.concat(atr_pcts) if atr_pcts else pd.Series(dtype=float)
    vol_series = pd.concat(vols) if vols else pd.Series(dtype=float)
    median_atr_pct = float(atr_series.median()) if len(atr_series) else 0.03
    median_vol20 = float(vol_series.median()) if len(vol_series) else 0.02
    return median_atr_pct, median_vol20


def run_backtest_instrumented(
    states, entry_filter_fn, sizing_fn, stop_pct=BASELINE_STOP_PCT, entry_cost=ENTRY_COST,
    exit_cost=EXIT_COST, size_multiplier=1.0, vol_target_annual=None,
):
    """Identical engine to run_backtest in run_sizing_allocation_backtest_equities.py
    (copied, not paraphrased -- verified bit-identical on synthetic data in
    the dry run before use), with one addition: a per-day (exposure_frac,
    open_count) trace, so exposure/concurrency can be sliced to the OOS
    window specifically rather than only reported as a whole-run average."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions: dict = {sym: None for sym in states}
    trade_log = []
    nav_history = []
    daily_nav_returns: list[float] = []
    daily_exposure_trace = []  # (ts, exposure_frac, open_count)

    for ts in all_index:
        if vol_target_annual is not None and len(daily_nav_returns) >= VOL_LOOKBACK_DAYS:
            window = daily_nav_returns[-VOL_LOOKBACK_DAYS:]
            realized_vol = float(np.std(window, ddof=1)) * np.sqrt(252) if len(window) > 1 else 0.0
            throttle = (vol_target_annual / realized_vol) if realized_vol > 1e-8 else 1.0
            throttle = min(max(throttle, VOL_THROTTLE_BOUNDS[0]), VOL_THROTTLE_BOUNDS[1])
        else:
            throttle = 1.0

        open_count = sum(1 for p in positions.values() if p is not None)
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
                    nav_unrealized_sum += pos["notional"] * (row["close"] / pos["entry_price"] - 1)
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
                open_count -= 1
                continue

            if row["cross_up"] and entry_filter_fn(row):
                notional = sizing_fn(capital, row) * size_multiplier * throttle
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + entry_cost)
                    positions[sym] = {
                        "entry_price": fill, "notional": notional, "entry_time": ts,
                        "last_price": fill,
                    }
                    free_capital -= notional
                    open_count += 1

        nav_t = capital + nav_unrealized_sum
        if nav_history:
            prev_nav = nav_history[-1][1]
            if prev_nav > 0:
                daily_nav_returns.append(nav_t / prev_nav - 1)
        nav_history.append((ts, nav_t))

        deployed_notional_now = sum(p["notional"] for p in positions.values() if p is not None)
        exposure_frac = deployed_notional_now / nav_t if nav_t > 0 else 0.0
        open_count_now = sum(1 for p in positions.values() if p is not None)
        daily_exposure_trace.append((ts, exposure_frac, open_count_now))

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    nav_series = pd.Series({t: v for t, v in nav_history}).sort_index() if nav_history else pd.Series(dtype=float)
    exposure_df = pd.DataFrame(daily_exposure_trace, columns=["ts", "exposure_frac", "open_count"]).set_index("ts")
    return trades, nav_series, exposure_df


def summarize_window(trades: pd.DataFrame, nav_series: pd.Series, window_start: pd.Timestamp | None = None) -> dict:
    """Parameterized copy of summarize_nav from the sizing script: identical
    math, but the starting-capital reference is the FIRST value of whatever
    nav_series slice is passed in (so it works for a full-history series
    referenced against STARTING_CAPITAL, or an OOS-only slice referenced
    against its own first value), and trades are expected to already be
    pre-filtered by the caller to the window of interest."""
    if nav_series.empty or len(nav_series) < 2:
        return {}
    start_capital = float(nav_series.iloc[0]) if window_start is not None else STARTING_CAPITAL
    nav_end = float(nav_series.iloc[-1])
    total_return = (nav_end / start_capital - 1) * 100
    years = max((nav_series.index[-1] - nav_series.index[0]).days / 365.25, 1e-9)
    cagr = (nav_end / start_capital) ** (1 / years) - 1

    daily_ret = nav_series.pct_change().dropna()
    ann_factor = np.sqrt(252)
    mean_d, std_d = (float(daily_ret.mean()), float(daily_ret.std(ddof=1))) if len(daily_ret) > 1 else (float("nan"), float("nan"))
    sharpe = (mean_d / std_d) * ann_factor if std_d and std_d > 0 else float("nan")
    downside = daily_ret[daily_ret < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = (mean_d / downside_std) * ann_factor if downside_std and downside_std > 0 else float("nan")
    realized_annual_vol_pct = (std_d * ann_factor * 100) if std_d and std_d > 0 else float("nan")

    running_max = nav_series.cummax()
    drawdown = (nav_series / running_max - 1) * 100
    max_dd = float(drawdown.min())
    worst_portfolio_day_pct = float(daily_ret.min() * 100) if len(daily_ret) else float("nan")

    if trades.empty:
        n_trades, win_rate, pf, worst_trade_pct = 0, float("nan"), float("nan"), float("nan")
    else:
        wins = trades[trades.pnl > 0]
        gross_win = wins.pnl.sum()
        gross_loss = -trades[trades.pnl <= 0].pnl.sum()
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        win_rate = len(wins) / len(trades) * 100
        worst_trade_pct = float(trades["ret_pct"].min())
        n_trades = len(trades)

    return {
        "n_trades": n_trades, "win_rate": win_rate, "pf": pf,
        "cagr_pct": cagr * 100, "total_return_pct": total_return, "max_dd_pct": max_dd,
        "worst_trade_pct": worst_trade_pct, "worst_portfolio_day_pct": worst_portfolio_day_pct,
        "sharpe": sharpe, "sortino": sortino, "realized_annual_vol_pct": realized_annual_vol_pct,
        "start_capital_ref": start_capital,
    }


def compute_breadth_series(states: dict[str, pd.DataFrame]) -> pd.Series:
    """Golden-cross breadth: fraction of the universe with above200==True on
    each day, using only the 'above200' indicator every candidate's engine
    already computes -- no new external data source."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    frac = []
    for ts in all_index:
        vals = []
        for state in states.values():
            if ts in state.index and not pd.isna(state.loc[ts, "ma200"]):
                vals.append(bool(state.loc[ts, "above200"]))
        frac.append(np.mean(vals) if vals else np.nan)
    return pd.Series(frac, index=all_index).ffill()


def regime_breakdown(trades_oos: pd.DataFrame, breadth: pd.Series) -> dict:
    if trades_oos.empty:
        return {}
    entry_breadth = trades_oos["entry_time"].map(lambda t: breadth.asof(t))
    valid = entry_breadth.dropna()
    if valid.empty:
        return {}
    terciles = valid.quantile([1 / 3, 2 / 3]).values
    labels = pd.cut(entry_breadth, bins=[-np.inf, terciles[0], terciles[1], np.inf],
                     labels=["LOW-breadth", "MID-breadth", "HIGH-breadth"])
    out = {}
    for label in ["LOW-breadth", "MID-breadth", "HIGH-breadth"]:
        mask = labels == label
        sub = trades_oos[mask.values]
        if sub.empty:
            out[label] = {"n_trades": 0}
            continue
        wins = sub[sub.pnl > 0]
        out[label] = {
            "n_trades": len(sub), "win_rate": len(wins) / len(sub) * 100,
            "avg_ret_pct": float(sub["ret_pct"].mean()), "total_pnl": float(sub["pnl"].sum()),
        }
    return out


def top_winner_sensitivity(trades_oos: pd.DataFrame, start_capital: float, window_years: float) -> dict:
    """Disclosed approximation (see module docstring): a realized-pnl-only
    capital path over the OOS window, recomputed with the top-K trades'
    pnl zeroed out in exit-time order. Does not re-simulate position sizing
    under the counterfactual capital path."""
    if trades_oos.empty:
        return {}
    ordered = trades_oos.sort_values("exit_time")
    out = {"n_oos_trades": len(ordered)}
    baseline_total_pnl = float(ordered["pnl"].sum())
    baseline_total_return_pct = baseline_total_pnl / start_capital * 100
    baseline_cagr_pct = (((start_capital + baseline_total_pnl) / start_capital) ** (1 / max(window_years, 1e-9)) - 1) * 100
    out["all_trades"] = {"total_return_pct": baseline_total_return_pct, "cagr_pct": baseline_cagr_pct}
    sorted_by_pnl = ordered.sort_values("pnl", ascending=False)
    for k in TOP_K_SENSITIVITY:
        k = min(k, len(sorted_by_pnl))
        excluded_pnl_sum = float(sorted_by_pnl["pnl"].iloc[:k].sum())
        adj_total_pnl = baseline_total_pnl - excluded_pnl_sum
        adj_total_return_pct = adj_total_pnl / start_capital * 100
        adj_cagr_pct = (((start_capital + adj_total_pnl) / start_capital) ** (1 / max(window_years, 1e-9)) - 1) * 100 \
            if (start_capital + adj_total_pnl) > 0 else float("-100")
        out[f"excl_top{k}"] = {
            "total_return_pct": adj_total_return_pct, "cagr_pct": adj_cagr_pct,
            "excluded_pnl_share_of_total_pct": (excluded_pnl_sum / baseline_total_pnl * 100) if baseline_total_pnl else float("nan"),
        }
    return out


def chronological_subperiods(trades_oos: pd.DataFrame, split_ts: pd.Timestamp, data_end: pd.Timestamp) -> dict:
    if trades_oos.empty:
        return {}
    years = list(range(split_ts.year, data_end.year + 1))
    out = {}
    for yr in years:
        yr_start = max(pd.Timestamp(f"{yr}-01-01"), split_ts)
        yr_end = min(pd.Timestamp(f"{yr}-12-31"), data_end)
        sub = trades_oos[(trades_oos["entry_time"] >= yr_start) & (trades_oos["entry_time"] <= yr_end)]
        if sub.empty:
            out[str(yr)] = {"n_trades": 0}
            continue
        wins = sub[sub.pnl > 0]
        out[str(yr)] = {
            "n_trades": len(sub), "win_rate": len(wins) / len(sub) * 100,
            "total_pnl": float(sub["pnl"].sum()), "avg_ret_pct": float(sub["ret_pct"].mean()),
        }
    return out


def _trade_set(trades: pd.DataFrame) -> set:
    if trades.empty:
        return set()
    return set(zip(trades["asset"], trades["entry_time"]))


def run_candidate(states, sizing_label, median_atr_pct, median_vol20, size_multiplier=1.0,
                   entry_cost=ENTRY_COST, exit_cost=EXIT_COST):
    sizing_fn = make_sizing_fn(sizing_label, median_atr_pct, median_vol20)
    kwargs = {"vol_target_annual": VOL_TARGET_ANNUAL_PCT / 100} if sizing_label == "VOL-TARGET" else {}
    trades, nav_series, exposure_df = run_backtest_instrumented(
        states, ENTRY_SETS["BASELINE"], sizing_fn, size_multiplier=size_multiplier,
        entry_cost=entry_cost, exit_cost=exit_cost, **kwargs,
    )
    return trades, nav_series, exposure_df


def _print_summary(label, summ, exp_avg, exp_realized_vol):
    if not summ or summ.get("n_trades", 0) == 0:
        print(f"  {label:<16} no OOS trades")
        return
    print(f"  {label:<16} n={summ['n_trades']:>4}  cagr={summ['cagr_pct']:>6.2f}%  "
          f"sharpe={summ['sharpe']:>5.2f}  sortino={summ['sortino']:>5.2f}  maxdd={summ['max_dd_pct']:>6.1f}%  "
          f"worst_day={summ['worst_portfolio_day_pct']:>6.2f}%  avg_exp={exp_avg:>5.1f}%  "
          f"realized_vol={exp_realized_vol:>5.1f}%")


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    states = build_states(price)
    data_start = min(s.index.min() for s in states.values())
    data_end = max(s.index.max() for s in states.values())
    print(f"full history span: {data_start.date()} -> {data_end.date()}")
    print(f"OOS_SPLIT_DATE (frozen before viewing results): {OOS_SPLIT_DATE.date()}")
    oos_years = max((data_end - OOS_SPLIT_DATE).days / 365.25, 1e-9)
    print(f"OOS window length: ~{oos_years:.2f} years\n")

    median_atr_pct, median_vol20 = compute_calibration_before(states, OOS_SPLIT_DATE)
    print(f"calibration (IN-SAMPLE ONLY, strictly before {OOS_SPLIT_DATE.date()}): "
          f"median_atr_pct={median_atr_pct*100:.2f}%  median_vol20={median_vol20*100:.2f}%\n")

    breadth = compute_breadth_series(states)

    native_runs = {}
    oos_summaries = {}
    oos_trade_sets_with_cost = {}
    results_out = {}

    print("=" * 90)
    print("NATIVE OOS RESULTS (each candidate's own natural sizing, no rescaling)")
    print("=" * 90)
    for label in CANDIDATES:
        trades, nav_series, exposure_df = run_candidate(states, label, median_atr_pct, median_vol20)
        native_runs[label] = (trades, nav_series, exposure_df)

        oos_nav = nav_series[nav_series.index >= OOS_SPLIT_DATE]
        oos_trades = trades[trades["entry_time"] >= OOS_SPLIT_DATE] if not trades.empty else trades
        oos_exposure = exposure_df[exposure_df.index >= OOS_SPLIT_DATE]

        summ = summarize_window(oos_trades, oos_nav, window_start=OOS_SPLIT_DATE)
        exp_avg = float(oos_exposure["exposure_frac"].mean() * 100) if not oos_exposure.empty else float("nan")
        oos_summaries[label] = summ
        oos_trade_sets_with_cost[label] = _trade_set(oos_trades)
        _print_summary(label, summ, exp_avg, summ.get("realized_annual_vol_pct", float("nan")))
        results_out[label] = {"native": summ, "native_avg_exposure_pct": exp_avg}

    ref_summ = oos_summaries[REFERENCE_CANDIDATE]
    ref_vol = ref_summ.get("realized_annual_vol_pct", float("nan"))
    print(f"\nreference (for vol-matching) = {REFERENCE_CANDIDATE}: OOS realized_annual_vol={ref_vol:.2f}%")

    print("\n" + "=" * 90)
    print("VOL-MATCHED OOS RESULTS (rescaled to EQUAL-DOLLAR's own OOS realized vol)")
    print("=" * 90)
    for label in CANDIDATES:
        native_summ = oos_summaries[label]
        native_vol = native_summ.get("realized_annual_vol_pct", float("nan"))
        multiplier = (ref_vol / native_vol) if native_vol and not np.isnan(native_vol) else 1.0
        trades, nav_series, exposure_df = run_candidate(states, label, median_atr_pct, median_vol20, size_multiplier=multiplier)
        oos_nav = nav_series[nav_series.index >= OOS_SPLIT_DATE]
        oos_trades = trades[trades["entry_time"] >= OOS_SPLIT_DATE] if not trades.empty else trades
        oos_exposure = exposure_df[exposure_df.index >= OOS_SPLIT_DATE]
        summ = summarize_window(oos_trades, oos_nav, window_start=OOS_SPLIT_DATE)
        exp_avg = float(oos_exposure["exposure_frac"].mean() * 100) if not oos_exposure.empty else float("nan")
        _print_summary(f"{label} (x{multiplier:.2f})", summ, exp_avg, summ.get("realized_annual_vol_pct", float("nan")))
        results_out[label]["vol_matched"] = summ
        results_out[label]["vol_matched_multiplier"] = multiplier
        results_out[label]["vol_matched_avg_exposure_pct"] = exp_avg

    print("\n" + "=" * 90)
    print("COST DRAG & CAPITAL-CONSTRAINT DIVERGENCE (OOS-entry trades only)")
    print("=" * 90)
    for label in CANDIDATES:
        trades, nav_series, _ = native_runs[label]
        trades0, nav0, _ = run_candidate(states, label, median_atr_pct, median_vol20, entry_cost=0.0, exit_cost=0.0)
        oos_trades = trades[trades["entry_time"] >= OOS_SPLIT_DATE] if not trades.empty else trades
        oos_trades0 = trades0[trades0["entry_time"] >= OOS_SPLIT_DATE] if not trades0.empty else trades0
        oos_nav0 = nav0[nav0.index >= OOS_SPLIT_DATE]
        zc_summ = summarize_window(oos_trades0, oos_nav0, window_start=OOS_SPLIT_DATE)
        wc_cagr = oos_summaries[label].get("cagr_pct", float("nan"))
        zc_cagr = zc_summ.get("cagr_pct", float("nan"))
        wc_set = oos_trade_sets_with_cost[label]
        zc_set = _trade_set(oos_trades0)
        divergence = len(wc_set.symmetric_difference(zc_set))
        print(f"  {label:<16} with_cost cagr={wc_cagr:>6.2f}%  zero_cost cagr={zc_cagr:>6.2f}%  "
              f"drag={zc_cagr - wc_cagr:>5.2f}pp  trade_set_divergence={divergence} "
              f"(with_cost n={len(wc_set)}, zero_cost n={len(zc_set)})")
        results_out[label]["cost_drag_pp"] = zc_cagr - wc_cagr
        results_out[label]["trade_set_divergence"] = divergence

    print("\n" + "=" * 90)
    print("CONCENTRATION (OOS-entry trades only, native sizing)")
    print("=" * 90)
    for label in CANDIDATES:
        trades, _, _ = native_runs[label]
        oos_trades = trades[trades["entry_time"] >= OOS_SPLIT_DATE] if not trades.empty else trades
        conc = concentration_table(oos_trades)
        if conc:
            print(f"  {label:<16} top1%={conc['top1pct_share_of_total_pnl']:>5.1f}% ({conc['top1pct_n_trades']} trades)  "
                  f"top5%={conc['top5pct_share_of_total_pnl']:>5.1f}% ({conc['top5pct_n_trades']})  "
                  f"top10%={conc['top10pct_share_of_total_pnl']:>5.1f}% ({conc['top10pct_n_trades']})")
        else:
            print(f"  {label:<16} no OOS trades to compute concentration")
        results_out[label]["concentration"] = conc

    print("\n" + "=" * 90)
    print("PERFORMANCE WITH / WITHOUT THE LARGEST FEW WINNERS (OOS-entry trades, native sizing)")
    print("(disclosed approximation -- see module docstring)")
    print("=" * 90)
    for label in CANDIDATES:
        trades, _, _ = native_runs[label]
        oos_trades = trades[trades["entry_time"] >= OOS_SPLIT_DATE] if not trades.empty else trades
        start_cap_ref = oos_summaries[label].get("start_capital_ref", STARTING_CAPITAL)
        sens = top_winner_sensitivity(oos_trades, start_cap_ref, oos_years)
        if sens:
            print(f"  {label}: all_trades cagr={sens['all_trades']['cagr_pct']:.2f}%")
            for k in TOP_K_SENSITIVITY:
                key = f"excl_top{min(k, sens['n_oos_trades'])}"
                if key in sens:
                    d = sens[key]
                    print(f"    excl_top{k}: cagr={d['cagr_pct']:.2f}%  "
                          f"(those {k} trade(s) were {d['excluded_pnl_share_of_total_pct']:.1f}% of total OOS pnl)")
        results_out[label]["top_winner_sensitivity"] = sens

    print("\n" + "=" * 90)
    print("RESULTS BY MARKET REGIME (golden-cross breadth terciles, OOS window, native sizing)")
    print("=" * 90)
    for label in CANDIDATES:
        trades, _, _ = native_runs[label]
        oos_trades = trades[trades["entry_time"] >= OOS_SPLIT_DATE] if not trades.empty else trades
        rb = regime_breakdown(oos_trades, breadth)
        print(f"  {label}:")
        for regime_label, stats in rb.items():
            if stats.get("n_trades", 0) == 0:
                print(f"    {regime_label:<14} no trades")
            else:
                print(f"    {regime_label:<14} n={stats['n_trades']:>4}  win={stats['win_rate']:>5.1f}%  "
                      f"avg_ret={stats['avg_ret_pct']:>6.2f}%  total_pnl=${stats['total_pnl']:>12,.0f}")
        results_out[label]["regime_breakdown"] = rb

    print("\n" + "=" * 90)
    print("CHRONOLOGICAL SUBPERIODS (calendar year, OOS window, native sizing)")
    print("=" * 90)
    for label in CANDIDATES:
        trades, _, _ = native_runs[label]
        oos_trades = trades[trades["entry_time"] >= OOS_SPLIT_DATE] if not trades.empty else trades
        sub = chronological_subperiods(oos_trades, OOS_SPLIT_DATE, data_end)
        print(f"  {label}:")
        for yr, stats in sub.items():
            if stats.get("n_trades", 0) == 0:
                print(f"    {yr}: no trades")
            else:
                print(f"    {yr}: n={stats['n_trades']:>4}  win={stats['win_rate']:>5.1f}%  "
                      f"total_pnl=${stats['total_pnl']:>12,.0f}  avg_ret={stats['avg_ret_pct']:>6.2f}%")
        results_out[label]["chronological_subperiods"] = sub

    print("\n" + "=" * 90)
    print("PREDETERMINED DECISION RULE (thresholds fixed before viewing these results:")
    print(f"  Sharpe/Sortino 'material' delta = {SHARPE_SORTINO_MATERIAL_DELTA}, "
          f"CAGR 'similar' band = {CAGR_SIMILAR_BAND_PP}pp)")
    print("=" * 90)
    base = oos_summaries[REFERENCE_CANDIDATE]
    base_cagr, base_sharpe, base_sortino = base.get("cagr_pct", float("nan")), base.get("sharpe", float("nan")), base.get("sortino", float("nan"))
    verdicts = {}
    for label in CANDIDATES:
        if label == REFERENCE_CANDIDATE:
            continue
        s = oos_summaries[label]
        cagr_delta = s.get("cagr_pct", float("nan")) - base_cagr
        sharpe_delta = s.get("sharpe", float("nan")) - base_sharpe
        sortino_delta = s.get("sortino", float("nan")) - base_sortino
        similar_cagr = abs(cagr_delta) <= CAGR_SIMILAR_BAND_PP
        sharpe_or_sortino_improved = (sharpe_delta > SHARPE_SORTINO_MATERIAL_DELTA) or (sortino_delta > SHARPE_SORTINO_MATERIAL_DELTA)
        sharpe_and_sortino_worse = (sharpe_delta < -SHARPE_SORTINO_MATERIAL_DELTA) and (sortino_delta < -SHARPE_SORTINO_MATERIAL_DELTA)
        if label == "VOL-TARGET" and cagr_delta > 0 and sharpe_and_sortino_worse:
            verdict = "REJECT added complexity -- wins on CAGR only, materially worse OOS risk-adjusted performance"
        elif label in ("EQUAL-RISK-ATR", "INVERSE-VOL") and sharpe_or_sortino_improved and similar_cagr:
            verdict = "STRONGEST ALLOCATION CANDIDATE -- retains Sharpe/Sortino edge with similar CAGR OOS"
        elif cagr_delta <= 0 and not sharpe_or_sortino_improved:
            verdict = "FAILS TO BEAT BASELINE -- keep EQUAL-DOLLAR"
        else:
            verdict = "MIXED -- EQUAL-DOLLAR stays the live control, treat as a research variant"
        verdicts[label] = verdict
        print(f"  {label:<16} cagr_delta={cagr_delta:+6.2f}pp  sharpe_delta={sharpe_delta:+5.2f}  "
              f"sortino_delta={sortino_delta:+5.2f}  ->  {verdict}")
    results_out["decision_rule_verdicts"] = verdicts

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "oos_split_date": str(OOS_SPLIT_DATE.date()),
        "data_span": [str(data_start.date()), str(data_end.date())],
        "calibration": {"median_atr_pct": median_atr_pct, "median_vol20": median_vol20},
        "results": results_out,
    }, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
