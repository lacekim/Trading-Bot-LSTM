"""Out-of-sample holdout + paired moving-block-bootstrap validation of the
two exit-logic families that showed ANY improvement over the control in the
full four-part in-sample sequence (ATR, Chandelier, time-based, partial
profit-taking) -- TIME_STAGNATION and PARTIAL_PROFIT_TAKING -- run at the
user's explicit request after reviewing that closing synthesis, together
with their explicit instruction to extend both grids first:

  "The two variants worth retaining as risk-management alternatives are:
   * 100-day stagnation ...
   * +100% partial profit-taking ...
   Before adopting either alternative, the next useful checks are:
   1. Evaluate them on untouched out-of-sample data or walk-forward splits.
   2. Bootstrap or block-bootstrap the return series to assess whether the
      risk-metric differences are robust.
   3. Test nearby parameters around 100 days and +100% to distinguish a
      broad plateau from a single favorable point.
   ...
   The key decision rule should be based on the trade-off frontier, not the
   top individual metric: retain a variant only if its OOS drawdown
   improvement is persistent and economically meaningful, while its CAGR
   and winner-P&L retention remain acceptable."

Per the user's explicit choice when asked which OOS scheme to build first
("both, single holdout first"): this script implements the SINGLE FROZEN
HOLDOUT scheme now; a rolling walk-forward study is explicit follow-on
work, not attempted here.

ATR and Chandelier are NOT retested here -- both failed decisively in-sample
on both CAGR and risk-adjusted metrics (see their own scripts/results), so
there is nothing "worth retaining as a risk-management alternative" to
validate for them. Only the two families with a genuine trade-off (better
risk-adjusted / lower drawdown, at some CAGR and winner-retention cost) are
carried forward to this OOS + bootstrap stage.

GRIDS VALIDATED (per the user's explicit grid-extension request, addressed
in run_time_based_exit_backtest_equities.py and
run_partial_profit_taking_backtest_equities.py before this script was
written):
  TIME_STAGNATION_DAYS      = (40, 60, 100, 150, 200)
  PROFIT_TAKE_THRESHOLDS_PCT = (25, 50, 100, 150, 200)
Every point in both extended grids is validated here, not just the two
in-sample standouts (100d, +100%) -- this directly serves check #3's "broad
stable region vs. single favorable point" question by showing where OOS
performance sits across the WHOLE grid, not only at its in-sample peak.

FROZEN BEFORE VIEWING THESE OOS RESULTS (nothing below was tuned after
seeing the numbers):

  OOS_SPLIT_DATE = 2023-01-01 -- reused UNCHANGED from
  run_sizing_oos_validation_equities.py's own frozen split, deliberately,
  rather than choosing a new one: reusing an already-frozen, already-used
  split avoids introducing a fresh researcher degree of freedom (picking a
  split date that happens to look favorable) at exactly the design stage
  where that risk is highest.

  BLOCK_LENGTH_DAYS = 21 (approximately one trading month) -- the moving-
  block-bootstrap block length for resampling OOS daily NAV returns. Chosen
  to preserve short-horizon autocorrelation and volatility clustering
  (single-day iid resampling would destroy exactly the drawdown-duration
  and drawdown-depth structure this validation cares about); not tuned to
  the data.
  N_BOOTSTRAP_DRAWS = 2000.
  BOOTSTRAP_SEED = 20260101 -- fixed, so every re-run of this exact script
  reproduces bit-identical bootstrap draws.

  PAIRED DESIGN: every bootstrap draw uses ONE shared sequence of resampled
  block start-positions, applied to the control's OOS daily-return series
  AND every variant's OOS daily-return series simultaneously (all share an
  identical trading-calendar index by construction, since every variant is
  run over the same `states`). This is a paired comparison, not two
  independently-resampled distributions: each draw represents "the same
  resampled slice of market history, as experienced by the control vs. this
  variant," which isolates the variant's OWN effect from shared market-
  regime noise far more powerfully than resampling each series separately.

  MAX-DRAWDOWN-DURATION (a NEW diagnostic, not present in any prior script
  in this thread): the longest run of consecutive trading days a NAV series
  stays strictly below its own running peak. An episode still open at the
  end of the series counts through the last available day (right-censored,
  not treated as "recovered"); this is disclosed per-variant, not hidden.
  Reported for both the OOS point estimate and, in bootstrap-resampled
  form, per draw.

  DECISION RULE -- operationalizes the user's own frontier framing
  literally, as four independent, predetermined gates. A variant is
  RETAINED AS A RISK-MANAGEMENT ALTERNATIVE only if ALL FOUR hold on the OOS
  window; otherwise it is REJECTED and the control remains the sole
  conclusion for that variant. The magnitude gates (DD improvement, CAGR
  giveup) use the ACTUAL OOS point estimate, not a bootstrap mean, so the
  decision reflects what was really observed; the bootstrap is used
  separately to characterize how PERSISTENT that observed effect is across
  resampled histories:
    MIN_DD_IMPROVEMENT_PP     = 3.0   ("economically meaningful": OOS
                                        max-drawdown improvement over
                                        control, in percentage points)
    PERSISTENCE_THRESHOLD_PCT = 90.0  ("persistent": fraction of paired
                                        bootstrap draws in which the
                                        variant's max-drawdown is better
                                        than the control's on that SAME
                                        resampled history)
    MAX_CAGR_GIVEUP_PP        = 2.0   ("CAGR remains acceptable": OOS CAGR
                                        must not trail control's OOS CAGR
                                        by more than this many points)
    MIN_WINNER_RETENTION_PCT  = 50.0  ("winner-P&L retention remains
                                        acceptable": pct_control_winner_
                                        pnl_retained, computed ONLY on
                                        OOS-entered matched trades, must be
                                        at least this)
  These four thresholds are judgment calls, disclosed in full rather than
  tuned after seeing the OOS numbers below. A different, equally reasonable
  set of thresholds could shift a borderline verdict; the underlying
  numbers (OOS CAGR/Sharpe/Sortino/max-DD/max-DD-duration deltas, bootstrap
  persistence rates, winner-retention) are reported in full so the reader
  can apply their own bar if they disagree with these ones.

ENGINE REUSE: every backtest here is run via run_backtest_exit_variant,
IMPORTED UNCHANGED from run_time_based_exit_backtest_equities.py (for the
control and all 5 TIME_STAGNATION_DAYS variants -- time_stagnation_days=None
reproduces the control exactly, byte-for-byte, as already verified) and
from run_partial_profit_taking_backtest_equities.py (for all 5
PROFIT_TAKE_THRESHOLDS_PCT variants). No engine logic is re-implemented;
only the OOS-window slicing, max-drawdown-duration diagnostic, paired
block-bootstrap, and the decision rule are new code in this script, and
are the only things this script's own dry-run test suite needs to verify
from scratch (the underlying engines already carry their own verification).

METHODOLOGY CAVEATS carried forward, disclosed exactly as in
run_sizing_oos_validation_equities.py: same 60-symbol universe and 12bps
cost model as every earlier script in this thread; no delisted-asset
accounting; ONE holdout split, not (yet) a walk-forward study (explicit
follow-on, not attempted here); the OOS window's length is whatever data
happens to exist after 2023-01-01 (~3-4 years), not independently chosen
for statistical power -- with this few years of OOS data and long average
holding periods (control's own average holding period is ~325 days), the
number of OOS-window trade-level observations is small, so all trade-level
diagnostics (winner-retention especially) should be read with that in mind;
the block bootstrap partially compensates for this by resampling the
DAILY NAV path (hundreds of observations) rather than the trade log
(dozens), but it inherits whatever the underlying trade-driven return
process looked like in this one particular ~3-4 year window and cannot
manufacture regime diversity that period didn't contain -- exactly the gap
a future walk-forward study is meant to close.
"""
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
    run_backtest_exit_variant as run_time_stagnation_backtest,
    TIME_STAGNATION_DAYS,
)
from tools.run_partial_profit_taking_backtest_equities import (
    run_backtest_exit_variant as run_partial_profit_backtest,
    _aggregate_to_position_level,
    PROFIT_TAKE_THRESHOLDS_PCT,
    PARTIAL_SALE_FRACTION,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "exit_logic_oos_bootstrap_validation_results.json"

OOS_SPLIT_DATE = pd.Timestamp("2023-01-01")
BLOCK_LENGTH_DAYS = 21
N_BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260101

MIN_DD_IMPROVEMENT_PP = 3.0
PERSISTENCE_THRESHOLD_PCT = 90.0
MAX_CAGR_GIVEUP_PP = 2.0
MIN_WINNER_RETENTION_PCT = 50.0


def _position_level(trades: pd.DataFrame) -> pd.DataFrame:
    """A trade log with an 'original_notional' column (partial-profit-taking
    style) may contain 2 rows for a split position -- collapse it via that
    script's own _aggregate_to_position_level. A trade log WITHOUT that
    column (control / time-stagnation, which never split a position) is
    already position-level and is returned unchanged. This is the only
    branching needed to treat both trade-log schemas uniformly below."""
    if trades.empty or "original_notional" not in trades.columns:
        return trades
    return _aggregate_to_position_level(trades)


def max_drawdown_duration(nav: pd.Series) -> tuple[int, bool]:
    """Longest run of consecutive observations nav stays STRICTLY below its
    own running peak. is_ongoing is True only when the run that ACHIEVES the
    reported max_len duration is ITSELF still open at the series' last
    observation -- i.e. the reported duration IS the presently-unresolved
    episode. This is deliberately NOT "was the last observation underwater,"
    which could describe a shorter, more recent drawdown while a longer,
    already-recovered one is what max_len is actually reporting -- that
    would mislabel a recovered episode as ongoing. An episode still open at
    the end counts through the last available day (right-censored, not
    treated as recovered). Returns (duration_in_observations, is_ongoing)."""
    if len(nav) == 0:
        return 0, False
    running_max = nav.cummax()
    underwater = (nav < running_max).to_numpy()
    n = len(underwater)
    max_len = 0
    max_run_end = -1
    cur_len = 0
    for i, u in enumerate(underwater):
        if u:
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
                max_run_end = i
        else:
            cur_len = 0
    is_ongoing = bool(max_len > 0 and max_run_end == n - 1)
    return int(max_len), is_ongoing


def summarize_oos(nav_oos: pd.Series) -> dict:
    """CAGR/Sharpe/Sortino/max_dd/worst_day math is copied verbatim from
    run_sizing_oos_validation_equities.py's summarize_window (same
    established OOS convention: start_capital is the FIRST value of the
    already-OOS-sliced nav series, i.e. what an investor who started
    watching on OOS_SPLIT_DATE actually experienced), plus the NEW
    max_dd_duration diagnostic this script adds."""
    if nav_oos.empty or len(nav_oos) < 2:
        return {}
    start_capital = float(nav_oos.iloc[0])
    nav_end = float(nav_oos.iloc[-1])
    total_return = (nav_end / start_capital - 1) * 100
    years = max((nav_oos.index[-1] - nav_oos.index[0]).days / 365.25, 1e-9)
    cagr = (nav_end / start_capital) ** (1 / years) - 1
    daily_ret = nav_oos.pct_change().dropna()
    ann_factor = np.sqrt(252)
    mean_d, std_d = (float(daily_ret.mean()), float(daily_ret.std(ddof=1))) if len(daily_ret) > 1 else (float("nan"), float("nan"))
    sharpe = (mean_d / std_d) * ann_factor if std_d and std_d > 0 else float("nan")
    downside = daily_ret[daily_ret < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = (mean_d / downside_std) * ann_factor if downside_std and downside_std > 0 else float("nan")
    running_max = nav_oos.cummax()
    drawdown = (nav_oos / running_max - 1) * 100
    max_dd = float(drawdown.min())
    worst_day = float(daily_ret.min() * 100) if len(daily_ret) else float("nan")
    dd_dur, dd_ongoing = max_drawdown_duration(nav_oos)
    return {
        "cagr_pct": cagr * 100, "total_return_pct": total_return, "max_dd_pct": max_dd,
        "worst_day_pct": worst_day, "sharpe": sharpe, "sortino": sortino,
        "max_dd_duration_days": dd_dur, "max_dd_duration_ongoing": dd_ongoing,
        "n_days": len(nav_oos), "daily_ret": daily_ret,
        "start_capital_ref": start_capital,
    }


def oos_winner_preservation(control_trades: pd.DataFrame, variant_trades: pd.DataFrame,
                             oos_split: pd.Timestamp) -> dict:
    """Winner-preservation diagnostics restricted to OOS-ENTERED trades only
    (entry_time >= oos_split -- decisions actually made during the test
    period, mirroring run_sizing_oos_validation_equities.py's own OOS-entry
    convention for trade-level metrics), after collapsing any split position
    to one row per (asset, entry_time) via _position_level. Matched to the
    intersection of (asset, entry_time); the unmatched count is reported,
    not hidden."""
    c_all = _position_level(control_trades)
    v_all = _position_level(variant_trades)
    c_oos = c_all[c_all["entry_time"] >= oos_split] if not c_all.empty else c_all
    v_oos = v_all[v_all["entry_time"] >= oos_split] if not v_all.empty else v_all
    if c_oos.empty or v_oos.empty:
        return {"n_matched": 0, "n_control_oos_positions": len(c_oos)}
    c = c_oos.set_index(["asset", "entry_time"])
    v = v_oos.set_index(["asset", "entry_time"])
    common = c.index.intersection(v.index)
    n_matched = len(common)
    n_control = len(c)
    if n_matched == 0:
        return {"n_matched": 0, "n_control_oos_positions": n_control}
    c_m = c.loc[common]
    v_m = v.loc[common]
    control_winner_mask = c_m["ret_pct"] > 0
    n_control_winners = int(control_winner_mask.sum())
    control_winner_pnl_total = float(c_m.loc[control_winner_mask, "pnl"].sum())
    variant_pnl_on_control_winners = float(v_m.loc[control_winner_mask, "pnl"].sum())
    pct_retained = (variant_pnl_on_control_winners / control_winner_pnl_total * 100) if control_winner_pnl_total else float("nan")
    cut_short_mask = control_winner_mask & (v_m["ret_pct"].to_numpy() < c_m["ret_pct"].to_numpy())
    n_cut_short = int(cut_short_mask.sum())
    return {
        "n_matched": n_matched, "n_control_oos_positions": n_control,
        "n_control_winners": n_control_winners, "n_cut_short": n_cut_short,
        "pct_winners_cut_short": (n_cut_short / n_control_winners * 100) if n_control_winners else float("nan"),
        "control_winner_pnl_total": control_winner_pnl_total,
        "variant_pnl_on_control_winners": variant_pnl_on_control_winners,
        "pct_control_winner_pnl_retained": pct_retained,
    }


def block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap: draws ceil(n/block_len) block start
    positions uniformly from [0, n), wrapping around the end of the series
    (so every start position is equally likely regardless of proximity to
    the series' end), concatenates the resulting blocks, and truncates to
    exactly n. Returns an array of integer positions into the ORIGINAL
    (unresampled) series -- the same array is used to index BOTH the
    control's and a variant's return series for a single paired draw."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    blocks = [(np.arange(block_len) + s) % n for s in starts]
    idx = np.concatenate(blocks)
    return idx[:n]


def metrics_from_returns(returns: np.ndarray) -> dict:
    """Given a raw array of daily returns (no dates -- as produced by
    indexing an original return series with block-bootstrap positions),
    reconstructs a NAV path starting at 1.0 and computes CAGR/Sharpe/
    Sortino/max_dd/max_dd_duration with the SAME formulas as summarize_oos,
    annualizing on len(returns)/252 (one trading day per observation, this
    codebase's standing convention)."""
    n = len(returns)
    nav = np.cumprod(1.0 + returns)
    years = max(n / 252.0, 1e-9)
    cagr = (nav[-1]) ** (1 / years) - 1 if nav[-1] > 0 else -1.0
    ann_factor = np.sqrt(252)
    mean_d = float(np.mean(returns)) if n > 0 else float("nan")
    std_d = float(np.std(returns, ddof=1)) if n > 1 else float("nan")
    sharpe = (mean_d / std_d) * ann_factor if std_d and std_d > 0 else float("nan")
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else float("nan")
    sortino = (mean_d / downside_std) * ann_factor if downside_std and downside_std > 0 else float("nan")
    running_max = np.maximum.accumulate(nav)
    drawdown = nav / running_max - 1
    max_dd = float(np.min(drawdown)) * 100
    underwater = nav < running_max
    max_len = cur_len = 0
    for u in underwater:
        if u:
            cur_len += 1
            max_len = max(max_len, cur_len)
        else:
            cur_len = 0
    return {"cagr_pct": cagr * 100, "sharpe": sharpe, "sortino": sortino,
            "max_dd_pct": max_dd, "max_dd_duration_days": int(max_len)}


def paired_bootstrap(control_returns: np.ndarray, variant_returns: np.ndarray,
                      block_len: int, n_draws: int, seed: int) -> dict:
    """Runs n_draws PAIRED block-bootstrap draws (the SAME resampled index
    sequence applied to both series each draw) and returns, per metric, the
    array of (variant - control) deltas across draws (max_dd_pct and
    max_dd_duration_days deltas are defined as control-minus-variant instead,
    so that a POSITIVE delta always means "the variant is better" for every
    metric here: higher CAGR/Sharpe/Sortino is better, and a SMALLER
    drawdown magnitude / SHORTER drawdown duration is better)."""
    assert len(control_returns) == len(variant_returns), (
        "paired_bootstrap requires control and variant return series of identical length -- "
        "they must share the same OOS trading-day calendar"
    )
    n = len(control_returns)
    rng = np.random.default_rng(seed)
    deltas = {"cagr_pct": [], "sharpe": [], "sortino": [], "max_dd_pct": [], "max_dd_duration_days": []}
    for _ in range(n_draws):
        idx = block_bootstrap_indices(n, block_len, rng)
        c_m = metrics_from_returns(control_returns[idx])
        v_m = metrics_from_returns(variant_returns[idx])
        deltas["cagr_pct"].append(v_m["cagr_pct"] - c_m["cagr_pct"])
        deltas["sharpe"].append(v_m["sharpe"] - c_m["sharpe"])
        deltas["sortino"].append(v_m["sortino"] - c_m["sortino"])
        deltas["max_dd_pct"].append(v_m["max_dd_pct"] - c_m["max_dd_pct"])  # variant less negative => positive delta => better
        deltas["max_dd_duration_days"].append(c_m["max_dd_duration_days"] - v_m["max_dd_duration_days"])  # variant shorter => positive => better
    out = {}
    for k, vals in deltas.items():
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            out[k] = {"mean_delta": float("nan"), "p05": float("nan"), "p95": float("nan"), "pct_draws_favorable": float("nan")}
            continue
        out[k] = {
            "mean_delta": float(np.mean(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "pct_draws_favorable": float(np.mean(arr > 0) * 100),
        }
    return out


def decide(oos_control: dict, oos_variant: dict, wp: dict) -> dict:
    """The four predetermined gates (see module docstring), applied to the
    ACTUAL OOS point estimates (not bootstrap means)."""
    dd_delta_pp = oos_variant.get("max_dd_pct", float("nan")) - oos_control.get("max_dd_pct", float("nan"))
    cagr_delta_pp = oos_variant.get("cagr_pct", float("nan")) - oos_control.get("cagr_pct", float("nan"))
    retained_pct = wp.get("pct_control_winner_pnl_retained", float("nan"))
    economically_meaningful = np.isfinite(dd_delta_pp) and dd_delta_pp >= MIN_DD_IMPROVEMENT_PP
    cagr_acceptable = np.isfinite(cagr_delta_pp) and cagr_delta_pp >= -MAX_CAGR_GIVEUP_PP
    retention_acceptable = np.isfinite(retained_pct) and retained_pct >= MIN_WINNER_RETENTION_PCT
    return {
        "dd_delta_pp": dd_delta_pp, "cagr_delta_pp": cagr_delta_pp, "retained_pct": retained_pct,
        "economically_meaningful": bool(economically_meaningful),
        "cagr_acceptable": bool(cagr_acceptable),
        "retention_acceptable": bool(retention_acceptable),
    }


def finalize_decision(gates: dict, persistence_pct: float) -> dict:
    persistent = np.isfinite(persistence_pct) and persistence_pct >= PERSISTENCE_THRESHOLD_PCT
    gates = dict(gates)
    gates["persistent"] = bool(persistent)
    gates["persistence_pct"] = persistence_pct
    retain = gates["economically_meaningful"] and gates["persistent"] and gates["cagr_acceptable"] and gates["retention_acceptable"]
    if retain:
        verdict = "RETAIN AS RISK-MANAGEMENT ALTERNATIVE"
    else:
        failed = [g for g in ("economically_meaningful", "persistent", "cagr_acceptable", "retention_acceptable") if not gates[g]]
        verdict = f"REJECT (failed: {', '.join(failed)})"
    gates["verdict"] = verdict
    return gates


def _print_oos_summary(label: str, s: dict) -> None:
    if not s:
        print(f"  {label:<28} no OOS data")
        return
    print(f"  {label:<28} n={s['n_days']:>4}d  cagr={s['cagr_pct']:>7.2f}%  sharpe={s['sharpe']:>5.2f}  "
          f"sortino={s['sortino']:>5.2f}  max_dd={s['max_dd_pct']:>7.2f}%  "
          f"dd_dur={s['max_dd_duration_days']:>3}d{'*' if s['max_dd_duration_ongoing'] else ' '}")


def main() -> None:
    symbols = _load_equities_universe()
    print(f"universe: {len(symbols)} robust non-crypto instruments (same as live)")
    price = fetch_ohlcv_batch(symbols)
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments")

    states = build_states(price)
    data_start = min(s.index.min() for s in states.values())
    data_end = max(s.index.max() for s in states.values())
    print(f"full history span: {data_start.date()} -> {data_end.date()}")
    print(f"OOS_SPLIT_DATE (frozen, reused from run_sizing_oos_validation_equities.py): {OOS_SPLIT_DATE.date()}")
    oos_years = max((data_end - OOS_SPLIT_DATE).days / 365.25, 1e-9)
    print(f"OOS window length: ~{oos_years:.2f} years\n")

    entry_filter_fn = ENTRY_SETS["BASELINE"]
    sizing_fn = make_sizing_fn("EQUAL-DOLLAR", 0.0, 0.0)

    # ---- run control + every grid point of both extended families ----
    all_trades: dict[str, pd.DataFrame] = {}
    all_nav: dict[str, pd.Series] = {}

    control_trades, control_nav = run_time_stagnation_backtest(states, entry_filter_fn, sizing_fn, stop_mode="FIXED")
    all_trades["CONTROL"] = control_trades
    all_nav["CONTROL"] = control_nav

    time_labels, profit_labels = [], []
    for n in TIME_STAGNATION_DAYS:
        label = f"TIME_STAGNATION_{n}d"
        trades, nav = run_time_stagnation_backtest(states, entry_filter_fn, sizing_fn, stop_mode="FIXED", time_stagnation_days=n)
        all_trades[label], all_nav[label] = trades, nav
        time_labels.append(label)
    for t in PROFIT_TAKE_THRESHOLDS_PCT:
        label = f"PARTIAL_PROFIT_{t}pct"
        trades, nav = run_partial_profit_backtest(states, entry_filter_fn, sizing_fn, stop_mode="FIXED", profit_take_threshold_pct=t)
        all_trades[label], all_nav[label] = trades, nav
        profit_labels.append(label)
    variant_labels = time_labels + profit_labels

    # ---- OOS summaries ----
    print("=" * 100)
    print("OOS WINDOW SUMMARY (NAV-based; start_capital_ref = NAV on first OOS day, matching the established "
          "run_sizing_oos_validation_equities.py convention)")
    print("=" * 100)
    oos_summ: dict[str, dict] = {}
    for label in ["CONTROL"] + variant_labels:
        nav_oos = all_nav[label][all_nav[label].index >= OOS_SPLIT_DATE]
        oos_summ[label] = summarize_oos(nav_oos)
        _print_oos_summary(label, oos_summ[label])

    # sanity: every variant's OOS daily-return series must be date-aligned with control's (same length,
    # same index) for the paired bootstrap to be valid.
    control_ret = oos_summ["CONTROL"]["daily_ret"]
    for label in variant_labels:
        vr = oos_summ[label]["daily_ret"]
        assert len(vr) == len(control_ret), f"{label}: OOS daily-return series length differs from control's"
        assert (vr.index == control_ret.index).all(), f"{label}: OOS daily-return series dates differ from control's"

    # ---- OOS winner preservation (OOS-entered trades only) ----
    print("\n" + "=" * 100)
    print("OOS WINNER PRESERVATION (OOS-entered matched trades only, position-level)")
    print("=" * 100)
    wp_results: dict[str, dict] = {}
    for label in variant_labels:
        wp = oos_winner_preservation(control_trades, all_trades[label], OOS_SPLIT_DATE)
        wp_results[label] = wp
        if wp.get("n_matched", 0) == 0:
            print(f"  {label:<28} no matched OOS-entered trades (n_control_oos_positions="
                  f"{wp.get('n_control_oos_positions', 0)}) -- winner-retention gate cannot be evaluated")
        else:
            print(f"  {label:<28} matched={wp['n_matched']:>3}  control_winners={wp['n_control_winners']:>3}  "
                  f"cut_short={wp['n_cut_short']:>3} ({wp['pct_winners_cut_short']:.1f}%)  "
                  f"pnl_retained={wp['pct_control_winner_pnl_retained']:.1f}%")

    # ---- paired block bootstrap ----
    print("\n" + "=" * 100)
    print(f"PAIRED BLOCK BOOTSTRAP (block_len={BLOCK_LENGTH_DAYS}d, draws={N_BOOTSTRAP_DRAWS}, seed={BOOTSTRAP_SEED})")
    print("positive delta = variant better than control on that SAME resampled history")
    print("=" * 100)
    boot_results: dict[str, dict] = {}
    for label in variant_labels:
        vr = oos_summ[label]["daily_ret"].to_numpy()
        cr = control_ret.to_numpy()
        boot = paired_bootstrap(cr, vr, BLOCK_LENGTH_DAYS, N_BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
        boot_results[label] = boot
        dd = boot["max_dd_pct"]
        print(f"  {label:<28} max_dd_delta: mean={dd['mean_delta']:>+6.2f}pp  "
              f"[p05={dd['p05']:>+6.2f}, p95={dd['p95']:>+6.2f}]  favorable={dd['pct_draws_favorable']:>5.1f}%  |  "
              f"cagr_delta mean={boot['cagr_pct']['mean_delta']:>+6.2f}pp favorable={boot['cagr_pct']['pct_draws_favorable']:>5.1f}%  |  "
              f"sharpe_delta mean={boot['sharpe']['mean_delta']:>+5.2f} favorable={boot['sharpe']['pct_draws_favorable']:>5.1f}%")

    # ---- decision rule ----
    print("\n" + "=" * 100)
    print("PREDETERMINED DECISION RULE (fixed before viewing these OOS numbers -- see module docstring):")
    print(f"  economically_meaningful: OOS max_dd improvement >= {MIN_DD_IMPROVEMENT_PP}pp")
    print(f"  persistent:              max_dd improvement holds in >= {PERSISTENCE_THRESHOLD_PCT}% of paired bootstrap draws")
    print(f"  cagr_acceptable:         OOS CAGR does not trail control by more than {MAX_CAGR_GIVEUP_PP}pp")
    print(f"  retention_acceptable:    OOS-entered pct_control_winner_pnl_retained >= {MIN_WINNER_RETENTION_PCT}%")
    print("  RETAIN only if all four hold; otherwise REJECT (control remains the sole conclusion for that variant).")
    print("=" * 100)
    decisions: dict[str, dict] = {}
    for label in variant_labels:
        gates = decide(oos_summ["CONTROL"], oos_summ[label], wp_results[label])
        persistence_pct = boot_results[label]["max_dd_pct"]["pct_draws_favorable"]
        final = finalize_decision(gates, persistence_pct)
        decisions[label] = final
        print(f"  {label:<28} dd_delta={final['dd_delta_pp']:>+6.2f}pp (meaningful={final['economically_meaningful']})  "
              f"persistence={final['persistence_pct']:>5.1f}% (persistent={final['persistent']})  "
              f"cagr_delta={final['cagr_delta_pp']:>+6.2f}pp (acceptable={final['cagr_acceptable']})  "
              f"retained={final['retained_pct']:>5.1f}% (acceptable={final['retention_acceptable']})")
        print(f"      -> {final['verdict']}")

    retained_any = [label for label, d in decisions.items() if d["verdict"].startswith("RETAIN")]
    print("\n" + "=" * 100)
    if retained_any:
        print(f"SUMMARY: {len(retained_any)} of {len(variant_labels)} variants clear all four OOS gates: "
              f"{', '.join(retained_any)}.")
        print("These are candidates worth carrying into a rolling walk-forward study (explicit follow-on, not run "
              "here) before any live adoption; every other tested variant in both extended grids is REJECTED on "
              "this OOS window and the control remains the sole conclusion for it.")
    else:
        print("SUMMARY: no variant in either extended grid clears all four predetermined OOS gates. The unchanged "
              "control remains the correct conclusion; neither TIME_STAGNATION nor PARTIAL_PROFIT_TAKING is "
              "retained as a risk-management alternative on this single OOS holdout.")
    print("=" * 100)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "oos_split_date": str(OOS_SPLIT_DATE.date()),
        "data_span": [str(data_start.date()), str(data_end.date())],
        "oos_summary": {k: {kk: vv for kk, vv in v.items() if kk != "daily_ret"} for k, v in oos_summ.items()},
        "winner_preservation": wp_results,
        "bootstrap": boot_results,
        "decisions": decisions,
    }
    OUT_PATH.write_text(json.dumps(serializable, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
