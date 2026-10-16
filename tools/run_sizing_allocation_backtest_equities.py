"""Position sizing / capital allocation study -- entries and exits held
EXACTLY as validated (immediate entry on golden cross, death-cross OR fixed
-12% stop exit, unchanged from run_stoploss_tail_risk_test_equities.py).
The only thing that varies across the runs below is HOW MUCH CAPITAL each
signal gets. This follows directly from the entry-filter diagnostic's
conclusion: since filters were shown to mostly gate exposure rather than
improve trade selection, the next place an improvement could plausibly
come from is portfolio construction, not entry timing -- so this keeps the
already-validated signal untouched and studies allocation on its own.

Five sizing methods, all documented here so the formulas are auditable
rather than implied by a name:

  EQUAL-DOLLAR   Baseline's own existing rule, unchanged: notional = 2% of
                 current capital per new position (cash-constrained). This
                 is the control -- literally what's live today.

  EQUAL-RISK-ATR Size so each position contributes similar DOLLAR risk,
                 using each instrument's own ATR14-as-%-of-price at entry
                 as the cross-sectional risk proxy (NOT the actual stop,
                 which stays fixed at -12% for every position exactly as
                 before -- this only changes position SIZE, never the
                 exit rule). notional = capital * 2% * (median_atr_pct /
                 this_instrument's_atr_pct), so an instrument with exactly
                 the universe's median ATR%% gets baseline's own 2% size;
                 a choppier instrument gets less, a quieter one gets more.

  INVERSE-VOL    Same idea, using each instrument's trailing 20-day
                 realized daily-return volatility instead of ATR --
                 the standard "inverse volatility weighting" definition,
                 calibrated the same way against the universe median so
                 average sizing is comparable to baseline's 2%.

  VOL-TARGET     Baseline per-position sizing (2% flat), globally
                 throttled up or down every trading day so the STRATEGY'S
                 OWN trailing 63-day realized annualized NAV volatility
                 tracks a fixed 10% annual target (a common institutional
                 target level, chosen independently of what baseline
                 happens to realize -- not calibrated to match it, so this
                 is a genuine "what if" rather than a tautology). Only new
                 entries are throttled; open positions are not resized.
                 Throttle is clipped to [0.2x, 5x] and defaults to 1x
                 until a full 63-day trailing NAV-return window exists
                 (VOL_LOOKBACK_DAYS itself, not a shorter placeholder),
                 before which there isn't enough trailing data to
                 estimate realized volatility from.

  POSITION-CAP   Baseline's own 2% sizing, unchanged, but capped at 10
                 concurrent open positions (baseline's own average is
                 reported alongside this so the size of the real
                 constraint is visible, not just asserted). When more
                 than 10 golden crosses are eligible at once, whichever
                 symbols are reached first in a fixed iteration order get
                 the open slots and the rest are skipped for that day --
                 a simple, disclosed tie-break, not a ranking rule.

Every method is run against BOTH the full baseline entry set and the
FILTER-FULL entry set (the one filter that showed a higher profit factor
and a slightly better exposure-normalized CAGR in the prior diagnostic),
specifically to test whether FILTER-FULL's apparent capital-efficiency
edge survives once it's sized to the SAME portfolio volatility as
baseline -- if it does, that's a real capital-efficiency result; if it
disappears, the earlier result was mostly a cash-allocation artifact, as
requested.

Three comparisons, all from the same 10 (2 entry sets x 5 sizing methods)
underlying simulations:
  A. NATIVE       -- each method's own natural capital deployment. What an
                     unlevered investor starting with the same $100,000
                     would actually experience.
  B. EXPOSURE-MATCHED -- every method's sizing uniformly rescaled (a single
                     multiplier applied to every position, equivalent to
                     leveraging or de-leveraging the whole book) so its
                     OWN average exposure matches BASELINE-EQUAL-DOLLAR's.
  C. VOL-MATCHED  -- same rescaling idea, but matched to BASELINE-EQUAL-
                     DOLLAR's own realized annualized NAV volatility
                     instead of its exposure.
Both B and C are explicitly disclosed as an idealization: uniformly
rescaling notional assumes leverage/margin is frictionless, which it is
not in reality. Treat B/C as "does this method use risk capital more
efficiently," not as a promise that real leverage would deliver exactly
these numbers.

A fourth table (D) reruns every NATIVE combination with the entry/exit
cost model zeroed out (ENTRY_COST=EXIT_COST=0) to show performance before
vs. after the existing 12bps-round-trip cost assumption.

Unlike the earlier diagnostic scripts, this one tracks a real DAILY
mark-to-market NAV series (capital plus every open position's unrealized
P&L, marked at that day's close), not just capital-at-trade-close. That
is what makes Sharpe, Sortino, true max drawdown, and "worst single
portfolio day" meaningful here instead of the "per-trade, not annualized"
approximation used in earlier scripts in this project. Exposure fractions
here are also measured against mark-to-market NAV, not against
realized-only capital -- a small methodological refinement from the
prior diagnostic script's avg_exposure_pct, disclosed here rather than
silently changed.

Caveats NOT resolved by this script (disclosed, not hidden): still the
same historical sample as every earlier test in this thread, not a
genuinely untouched out-of-sample period; still the existing 12bps
round-trip cost model with no separate slippage estimate; still no
delisted-asset accounting in the 60-symbol universe. Exit-method research
(ATR trailing stops, chandelier exits, time-based exits, partial
profit-taking) is explicitly NOT covered here -- this script's whole
point is to hold entries AND exits fixed and isolate allocation mechanics
as the only variable, per the stated research order (sizing/allocation
before exits).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tools.run_cross_asset_golden_cross_backtest import ENTRY_COST, EXIT_COST
from tools.run_stoploss_tail_risk_test_equities import (
    STARTING_CAPITAL,
    PER_TRADE_ALLOCATION_PCT,
    _load_equities_universe,
)
from tools.run_pullback_entry_backtest_equities import fetch_ohlcv_batch, compute_pullback_indicators
from tools.run_entry_filter_backtest_equities import BASELINE_STOP_PCT

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "sizing_allocation_backtest_results.json"

SIZING_METHODS = ["EQUAL-DOLLAR", "EQUAL-RISK-ATR", "INVERSE-VOL", "VOL-TARGET", "POSITION-CAP"]
ENTRY_SETS = {
    "BASELINE": lambda row: True,
    "FILTER-FULL": lambda row: bool(row["ma200_rising"]) and bool(row["volume_confirm"])
    and bool(row["structure_ok"]) and bool(row["ma50_rising"]),
}
VOL_TARGET_ANNUAL_PCT = 10.0  # independent illustrative target, NOT calibrated to baseline
VOL_LOOKBACK_DAYS = 63
VOL_THROTTLE_BOUNDS = (0.2, 5.0)
POSITION_CAP_N = 10
REFERENCE_ENTRY_SIZING = ("BASELINE", "EQUAL-DOLLAR")  # what exposure/vol matching targets


def compute_sizing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Builds on the pullback script's own compute_pullback_indicators
    (already has ma200/ma50/cross_up/cross_down/atr14/ma200_rising/
    ma50_rising/structure_ok/volume_confirm) and adds the extra fields
    this study needs: atr_pct (ATR14 as a %% of price, for EQUAL-RISK-
    ATR) and vol20 (trailing 20-day realized daily-return stdev, for
    INVERSE-VOL).

    Look-ahead note: atr_pct/vol20 as computed here both use that same
    day's own close (same no-shift convention as the rest of this
    project's ma50/ma200/cross_up/cross_down -- the established,
    disclosed assumption that the decision/execution happens "at that
    day's close"). BUT the sizing formulas are a genuinely separate
    look-ahead concern from entry/exit timing: they use ATR/vol as a
    cross-sectional RISK PROXY, not as the trigger for the trade itself,
    and there is no reason the risk proxy needs same-day data the way
    the trend/cross signal does. So *_lag1 versions (shifted by one bar)
    are added below and are what make_sizing_fn / compute_calibration
    actually use for sizing decisions -- entry/exit timing (cross_up,
    cross_down, stop check) is untouched and still uses the unshifted
    row exactly as before, preserving comparability with baseline."""
    out = compute_pullback_indicators(df)
    daily_ret = out["close"].pct_change()
    out["vol20"] = daily_ret.rolling(20).std()
    out["atr_pct"] = out["atr14"] / out["close"]
    out["vol20_lag1"] = out["vol20"].shift(1)
    out["atr_pct_lag1"] = out["atr_pct"].shift(1)
    return out


def build_states(price: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {sym: compute_sizing_indicators(df) for sym, df in price.items()}


def compute_calibration(states: dict[str, pd.DataFrame]) -> tuple[float, float]:
    """Universe-median ATR%% and vol20, sampled at cross_up days only (the
    days sizing decisions are actually made), so EQUAL-RISK-ATR and
    INVERSE-VOL can be calibrated to average out to baseline's own 2%%
    sizing rather than an arbitrary, unmoored scale.

    Uses the *_lag1 (one-bar-shifted) columns, matching what make_sizing_fn
    actually sizes with -- so the calibration median is measured on the
    same data the ratios are built from, not a same-day series that's
    never actually used for sizing."""
    atr_pcts, vols = [], []
    for state in states.values():
        cross_rows = state[state["cross_up"]]
        atr_pcts.append(cross_rows["atr_pct_lag1"].dropna())
        vols.append(cross_rows["vol20_lag1"].dropna())
    atr_series = pd.concat(atr_pcts) if atr_pcts else pd.Series(dtype=float)
    vol_series = pd.concat(vols) if vols else pd.Series(dtype=float)
    median_atr_pct = float(atr_series.median()) if len(atr_series) else 0.03
    median_vol20 = float(vol_series.median()) if len(vol_series) else 0.02
    return median_atr_pct, median_vol20


def make_sizing_fn(method: str, median_atr_pct: float, median_vol20: float):
    base_pct = PER_TRADE_ALLOCATION_PCT

    if method == "EQUAL-DOLLAR":
        return lambda capital, row: capital * base_pct / 100

    if method == "EQUAL-RISK-ATR":
        def fn(capital, row):
            # lag1: sizing uses YESTERDAY's ATR%, not today's -- see the
            # look-ahead note in compute_sizing_indicators. Entry/exit
            # timing itself (cross_up/cross_down/stop) is untouched.
            atr_pct = row["atr_pct_lag1"]
            if pd.isna(atr_pct) or atr_pct <= 0:
                return capital * base_pct / 100
            ratio = median_atr_pct / atr_pct
            # Clip the ratio itself (not just the resulting notional) so an
            # extremely quiet instrument is gracefully sized UP to a bounded
            # multiple of baseline's 2%, rather than silently never entered
            # because its uncapped notional blows through the free_capital
            # backstop in run_backtest. Reuses VOL_THROTTLE_BOUNDS so every
            # ratio-based sizing rule in this script shares one disclosed
            # bound (0.2x-5x of baseline's flat sizing).
            ratio = min(max(ratio, VOL_THROTTLE_BOUNDS[0]), VOL_THROTTLE_BOUNDS[1])
            return capital * (base_pct / 100) * ratio
        return fn

    if method == "INVERSE-VOL":
        def fn(capital, row):
            v = row["vol20_lag1"]
            if pd.isna(v) or v <= 0:
                return capital * base_pct / 100
            ratio = median_vol20 / v
            ratio = min(max(ratio, VOL_THROTTLE_BOUNDS[0]), VOL_THROTTLE_BOUNDS[1])
            return capital * (base_pct / 100) * ratio
        return fn

    if method in ("VOL-TARGET", "POSITION-CAP"):
        # Base per-position sizing for both is plain equal-dollar; the
        # extra behavior (global vol throttle / concurrency cap) is
        # applied by the engine itself, not the sizing function.
        return lambda capital, row: capital * base_pct / 100

    raise ValueError(f"unknown sizing method: {method}")


def run_backtest(
    states: dict[str, pd.DataFrame],
    entry_filter_fn,
    sizing_fn,
    stop_pct: float = BASELINE_STOP_PCT,
    entry_cost: float = ENTRY_COST,
    exit_cost: float = EXIT_COST,
    max_concurrent: int | None = None,
    size_multiplier: float = 1.0,
    vol_target_annual: float | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Baseline's own cash-constrained event loop (immediate entry on
    cross_up subject to entry_filter_fn, fixed-pct stop checked via that
    day's low, death-cross signal exit -- unchanged), plus: (1) a daily
    mark-to-market NAV series, (2) an optional concurrency cap, (3) an
    optional daily vol-targeting throttle on new entries, (4) a uniform
    size_multiplier for the exposure/vol-matched comparisons."""
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions: dict[str, dict | None] = {sym: None for sym in states}
    trade_log = []
    nav_history: list[tuple] = []
    daily_nav_returns: list[float] = []
    exposure_sum = 0.0
    exposure_steps = 0
    max_exposure_pct = 0.0
    concurrent_sum = 0
    max_concurrent_count = 0

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
                # Position-cap semantics (disclosed, not just implied by the
                # name): open_count is checked/updated LIVE within this same
                # per-symbol loop over `states.items()`, so a symbol whose
                # exit was processed earlier in this SAME ts's iteration (see
                # the `open_count -= 1` above) can free a slot that a
                # later-iterated symbol's entry uses on the very same day.
                # The tie-break for which symbol gets a freed/contested slot
                # is therefore the iteration order of the `states` dict --
                # i.e. whatever order build_states()/fetch_ohlcv_batch()
                # happened to populate it in -- not an economically motivated
                # ranking (not "best signal first", not alphabetical by
                # design). Treat POSITION-CAP's results as reflecting "a cap
                # exists" rather than "the best N candidates are chosen".
                if max_concurrent is not None and open_count >= max_concurrent:
                    continue
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

        # Non-circularity (verified by construction, not just by reasoning
        # about it after the fact): `throttle` for THIS ts was computed
        # above from `daily_nav_returns` as it stood BEFORE this ts's own
        # return exists -- the append below is the first point at which
        # this ts's return enters the list, strictly after every entry
        # decision for this ts has already used `throttle`. So no entry
        # ever gets sized using a NAV move it itself caused.
        nav_t = capital + nav_unrealized_sum
        if nav_history:
            prev_nav = nav_history[-1][1]
            if prev_nav > 0:
                daily_nav_returns.append(nav_t / prev_nav - 1)
        nav_history.append((ts, nav_t))

        deployed_notional_now = sum(p["notional"] for p in positions.values() if p is not None)
        exposure_frac = deployed_notional_now / nav_t if nav_t > 0 else 0.0
        exposure_sum += exposure_frac
        exposure_steps += 1
        max_exposure_pct = max(max_exposure_pct, exposure_frac * 100)
        open_count_now = sum(1 for p in positions.values() if p is not None)
        concurrent_sum += open_count_now
        max_concurrent_count = max(max_concurrent_count, open_count_now)

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    nav_series = pd.Series({t: v for t, v in nav_history}).sort_index() if nav_history else pd.Series(dtype=float)
    exposure_info = {
        "avg_exposure_pct": (exposure_sum / exposure_steps * 100) if exposure_steps else float("nan"),
        "max_exposure_pct": max_exposure_pct,
        "avg_concurrent": (concurrent_sum / exposure_steps) if exposure_steps else float("nan"),
        "max_concurrent": max_concurrent_count,
    }
    return trades, nav_series, exposure_info


def summarize_nav(trades: pd.DataFrame, nav_series: pd.Series) -> dict:
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
    }


def concentration_table(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    total_pnl = trades["pnl"].sum()
    sorted_trades = trades.sort_values("pnl", ascending=False)
    n = len(sorted_trades)
    out = {}
    for pct in (1, 5, 10):
        k = max(1, int(np.ceil(n * pct / 100)))
        top_pnl = sorted_trades["pnl"].iloc[:k].sum()
        out[f"top{pct}pct_share_of_total_pnl"] = float(top_pnl / total_pnl * 100) if total_pnl else float("nan")
        out[f"top{pct}pct_n_trades"] = k
    return out


def run_variant(
    states, entry_label: str, sizing_label: str, median_atr_pct: float, median_vol20: float,
    size_multiplier: float = 1.0, entry_cost: float = ENTRY_COST, exit_cost: float = EXIT_COST,
    return_trades: bool = False,
):
    entry_filter_fn = ENTRY_SETS[entry_label]
    sizing_fn = make_sizing_fn(sizing_label, median_atr_pct, median_vol20)
    kwargs: dict = {}
    if sizing_label == "VOL-TARGET":
        kwargs["vol_target_annual"] = VOL_TARGET_ANNUAL_PCT / 100
    if sizing_label == "POSITION-CAP":
        kwargs["max_concurrent"] = POSITION_CAP_N

    trades, nav_series, exposure_info = run_backtest(
        states, entry_filter_fn, sizing_fn, size_multiplier=size_multiplier,
        entry_cost=entry_cost, exit_cost=exit_cost, **kwargs,
    )
    summ = summarize_nav(trades, nav_series)
    conc = concentration_table(trades)
    norm_cagr = (
        summ["cagr_pct"] / (exposure_info["avg_exposure_pct"] / 100)
        if summ.get("cagr_pct") is not None and exposure_info.get("avg_exposure_pct")
        else float("nan")
    )
    result = {"entry": entry_label, "sizing": sizing_label, **summ, **exposure_info,
              "cagr_per_exposure": norm_cagr, **conc}
    if return_trades:
        return result, trades
    return result


def _trade_set(trades: pd.DataFrame) -> set:
    """(asset, entry_time) identity of every trade in a run, for comparing
    whether two cost regimes produced the SAME set of trades. Notional and
    costs are computed independently of each other per-trade, but capital
    compounds differently under different cost regimes and free_capital /
    notional checks are a function of that compounding capital path -- so
    near a capital-constraint boundary the with-cost and zero-cost runs
    could in principle accept or reject a different entry. This makes that
    measurable instead of assumed away."""
    if trades.empty:
        return set()
    return set(zip(trades["asset"], trades["entry_time"]))


def _print_row(label: str, r: dict) -> None:
    if not r or r.get("n_trades", 0) == 0:
        print(f"{label:<28} no trades")
        return
    print(f"{label:<28} n={r['n_trades']:>5}  win={r['win_rate']:>5.1f}%  pf={r['pf']:>5.2f}  "
          f"cagr={r['cagr_pct']:>6.2f}%  total_ret={r['total_return_pct']:>9.1f}%  "
          f"maxdd={r['max_dd_pct']:>6.1f}%  sharpe={r['sharpe']:>5.2f}  sortino={r['sortino']:>5.2f}")
    print(f"{'':<28} avg_exp={r['avg_exposure_pct']:>5.1f}%  max_exp={r['max_exposure_pct']:>5.1f}%  "
          f"avg_conc={r['avg_concurrent']:>5.1f}  max_conc={r['max_concurrent']:>3}  "
          f"cagr/exp={r['cagr_per_exposure']:>6.2f}  worst_trade={r['worst_trade_pct']:>6.1f}%  "
          f"worst_day={r['worst_portfolio_day_pct']:>6.2f}%")
    if "top1pct_share_of_total_pnl" in r:
        print(f"{'':<28} concentration -- top1%={r['top1pct_share_of_total_pnl']:>5.1f}% of pnl "
              f"({r['top1pct_n_trades']} trades)  top5%={r['top5pct_share_of_total_pnl']:>5.1f}% "
              f"({r['top5pct_n_trades']})  top10%={r['top10pct_share_of_total_pnl']:>5.1f}% ({r['top10pct_n_trades']})")


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")

    states = build_states(price)
    median_atr_pct, median_vol20 = compute_calibration(states)
    print(f"calibration -- universe median ATR%%(at cross): {median_atr_pct*100:.2f}%  "
          f"median 20d realized vol: {median_vol20*100:.2f}%\n")

    combos = [(e, s) for e in ENTRY_SETS for s in SIZING_METHODS]

    print("=" * 78)
    print("TABLE A -- NATIVE (same nominal starting capital, no rescaling)")
    print("=" * 78)
    native_results = {}
    native_trade_sets = {}
    for entry_label, sizing_label in combos:
        r, trades = run_variant(states, entry_label, sizing_label, median_atr_pct, median_vol20, return_trades=True)
        native_results[(entry_label, sizing_label)] = r
        native_trade_sets[(entry_label, sizing_label)] = _trade_set(trades)
        _print_row(f"{entry_label} / {sizing_label}", r)

    ref = native_results[REFERENCE_ENTRY_SIZING]
    ref_avg_exposure = ref.get("avg_exposure_pct")
    ref_vol = ref.get("realized_annual_vol_pct")
    print(f"\nreference for matching = {REFERENCE_ENTRY_SIZING}: "
          f"avg_exposure={ref_avg_exposure:.1f}%  realized_annual_vol={ref_vol:.1f}%  "
          f"avg_concurrent={ref['avg_concurrent']:.1f}  (position cap of {POSITION_CAP_N} is "
          f"{POSITION_CAP_N / ref['avg_concurrent'] * 100:.0f}% of baseline's own average concurrency)")

    print("\n" + "=" * 78)
    print("TABLE B -- EXPOSURE-MATCHED (uniformly rescaled to baseline-equal-dollar's")
    print("own average exposure -- an idealized leverage/de-leverage assumption)")
    print("=" * 78)
    exposure_matched_results = {}
    for entry_label, sizing_label in combos:
        native = native_results[(entry_label, sizing_label)]
        native_exp = native.get("avg_exposure_pct")
        multiplier = (ref_avg_exposure / native_exp) if native_exp else 1.0
        r = run_variant(states, entry_label, sizing_label, median_atr_pct, median_vol20, size_multiplier=multiplier)
        exposure_matched_results[(entry_label, sizing_label)] = r
        _print_row(f"{entry_label} / {sizing_label} (x{multiplier:.2f})", r)

    print("\n" + "=" * 78)
    print("TABLE C -- VOL-MATCHED (uniformly rescaled to baseline-equal-dollar's own")
    print("realized annualized NAV volatility)")
    print("=" * 78)
    vol_matched_results = {}
    for entry_label, sizing_label in combos:
        native = native_results[(entry_label, sizing_label)]
        native_vol = native.get("realized_annual_vol_pct")
        multiplier = (ref_vol / native_vol) if native_vol else 1.0
        r = run_variant(states, entry_label, sizing_label, median_atr_pct, median_vol20, size_multiplier=multiplier)
        vol_matched_results[(entry_label, sizing_label)] = r
        _print_row(f"{entry_label} / {sizing_label} (x{multiplier:.2f})", r)

    print("\n" + "=" * 78)
    print("TABLE D -- BEFORE vs. AFTER transaction costs (NATIVE sizing, cost")
    print("zeroed out: ENTRY_COST=EXIT_COST=0, vs. the existing 12bps round trip)")
    print("=" * 78)
    cost_results = {}
    for entry_label, sizing_label in combos:
        r0, trades0 = run_variant(states, entry_label, sizing_label, median_atr_pct, median_vol20,
                                   entry_cost=0.0, exit_cost=0.0, return_trades=True)
        with_cost = native_results[(entry_label, sizing_label)]
        with_cost_set = native_trade_sets[(entry_label, sizing_label)]
        zero_cost_set = _trade_set(trades0)
        # Cost-monotonicity is NOT assumed: notional/accept-reject decisions
        # don't reference entry_cost/exit_cost directly, but costs still
        # shape capital's compounding path, which feeds every subsequent
        # notional = f(capital) calculation and free_capital gate -- so the
        # exact trade SET can in principle diverge between cost regimes.
        # Measure it directly via symmetric set difference rather than
        # assert zero_cost_cagr >= with_cost_cagr is automatic.
        divergent = with_cost_set.symmetric_difference(zero_cost_set)
        cost_results[(entry_label, sizing_label)] = {
            "with_cost": with_cost, "zero_cost": r0,
            "n_trades_with_cost": len(with_cost_set), "n_trades_zero_cost": len(zero_cost_set),
            "n_divergent_trades": len(divergent),
        }
        wc_cagr = with_cost.get("cagr_pct", float("nan"))
        zc_cagr = r0.get("cagr_pct", float("nan"))
        print(f"{entry_label} / {sizing_label:<16} with_cost cagr={wc_cagr:>6.2f}%  "
              f"zero_cost cagr={zc_cagr:>6.2f}%  drag={zc_cagr - wc_cagr:>5.2f}pp  "
              f"trade_set_divergence={len(divergent)} "
              f"(with_cost n={len(with_cost_set)}, zero_cost n={len(zero_cost_set)})")

    def _key(entry_label, sizing_label):
        return f"{entry_label}|{sizing_label}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "calibration": {"median_atr_pct": median_atr_pct, "median_vol20": median_vol20},
        "params": {
            "vol_target_annual_pct": VOL_TARGET_ANNUAL_PCT, "vol_lookback_days": VOL_LOOKBACK_DAYS,
            "position_cap_n": POSITION_CAP_N, "reference_entry_sizing": list(REFERENCE_ENTRY_SIZING),
        },
        "table_a_native": {_key(*k): v for k, v in native_results.items()},
        "table_b_exposure_matched": {_key(*k): v for k, v in exposure_matched_results.items()},
        "table_c_vol_matched": {_key(*k): v for k, v in vol_matched_results.items()},
        "table_d_cost_comparison": {_key(*k): v for k, v in cost_results.items()},
    }, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
