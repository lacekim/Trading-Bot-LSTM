"""Tests whether a real intermarket macro-regime filter improves the actual
combined-portfolio golden-cross backtest (tools/run_ma200_cross_strategy.py)
-- not a standalone strategy, a gate on top of the existing one, evaluated
against the SAME real backtest engine (15%-per-asset fixed-fractional sizing,
real fees/slippage) that run_ma200_cross_strategy.py already uses. This is
deliberately NOT run against a full-capital-compounding baseline -- that
methodology mismatch is what made the earlier LightGBM-filter comparison
misleading (removing trades from a 100%-compounding backtest can gut total
return even when trade *quality* improves, since a few outsized winners drive
most of the compounded total). Fixed-fractional sizing does not have that
distortion, so trade-count changes here mean what they look like they mean.

Regime construction (real, standard intermarket signals, not fabricated):
  - Credit risk appetite: HYG/IEF ratio (high-yield vs Treasuries). Rising =
    credit markets pricing more risk-on; standard credit-spread-proxy read.
  - Dollar pressure: UUP vs its own 200-day MA. A dollar in a strong uptrend
    is a well-documented headwind for commodities/EM/crypto (tighter global
    dollar liquidity), so UUP > 200MA counts against risk-on.
  - Regime = RISK_ON only when HYG/IEF ratio > its 200-day MA AND UUP <= its
    200-day MA. Anything else (dollar strength, or credit deteriorating) =
    filtered out, no new entries that day.
  - EVENT window: +/-1 calendar day around every real FOMC decision date,
    2014-2027, sourced directly from federalreserve.gov (fomccalendars.htm +
    fomchistorical{year}.htm archive pages) on 2026-09-12 -- not guessed from
    memory. New entries are skipped inside this window regardless of regime.

Requires network + yfinance (for IEF/UUP/HYG history) and the same conda env
as the rest of this project -- run on the actual machine, not in a sandbox
with no network access.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yfinance as yf

from trading_bot_v4.signals.ma200_trend import compute_trend_state
from tools.run_ma200_cross_strategy import ASSETS, ENTRY_COST, EXIT_COST, PER_ASSET_ALLOCATION_PCT, STARTING_CAPITAL, load_full_history

# Real FOMC decision dates, 2014-2027, verified against federalreserve.gov
# (fomccalendars.htm for 2021-2027, fomchistorical{year}.htm for 2014-2020)
# on 2026-09-12. Multi-day meetings: the announcement is the LAST day listed.
FOMC_DATES = pd.to_datetime([
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30",
    "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29",
    "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-04", "2019-10-30", "2019-12-11",
    # 2020 (includes emergency/unscheduled decisions)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
], utc=True)


def fetch_macro_daily(ticker: str, start: str = "2013-06-01") -> pd.Series:
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index, utc=True)
    return close.dropna()


def build_regime_series() -> pd.Series:
    """Returns a daily boolean series: True = RISK_ON (entries allowed)."""
    hyg = fetch_macro_daily("HYG")
    ief = fetch_macro_daily("IEF")
    uup = fetch_macro_daily("UUP")

    ratio = (hyg / ief).dropna()
    ratio_ma200 = ratio.rolling(200).mean()
    credit_risk_on = ratio > ratio_ma200

    uup_ma200 = uup.rolling(200).mean()
    dollar_not_headwind = uup <= uup_ma200

    idx = credit_risk_on.index.intersection(dollar_not_headwind.index)
    risk_on = (credit_risk_on.reindex(idx) & dollar_not_headwind.reindex(idx)).dropna()

    event_days = pd.DatetimeIndex([])
    for d in FOMC_DATES:
        event_days = event_days.append(pd.date_range(d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)))
    is_event = risk_on.index.normalize().isin(event_days.normalize())

    final = risk_on.copy()
    final[is_event] = False
    return final


def run_filtered_backtest(regime: pd.Series) -> pd.DataFrame:
    price = {}
    for label, (pair, path) in ASSETS.items():
        df = load_full_history(pair, path)
        state = compute_trend_state(df["close"])
        price[label] = state

    regime_daily = regime.copy()
    regime_daily.index = regime_daily.index.normalize()

    def regime_ok(ts) -> bool:
        day = pd.Timestamp(ts).normalize()
        pos = regime_daily.index.searchsorted(day, side="right") - 1
        if pos < 0:
            return False
        return bool(regime_daily.iloc[pos])

    all_index = sorted(set().union(*[s.index for s in price.values()]))
    capital = STARTING_CAPITAL
    positions = {label: None for label in ASSETS}
    trade_log = []
    rejected_for_cash = 0

    for ts in all_index:
        # Same real available-cash check as run_ma200_cross_strategy.py --
        # see that file's comment for the full story (verify_capital_
        # constrained_sizing.py confirmed the sweep's 45%-allocation result
        # implicitly assumed free leverage; real constrained drawdown at
        # that size is -60.6%, not the -33.0% the unconstrained sweep
        # reported). This file has its own copy of the entry loop rather
        # than calling run_ma200_cross_strategy.run_backtest() directly
        # (it needs the regime_ok(ts) gate threaded through), so the fix
        # has to be applied here too, not just upstream.
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional

        for label, state in price.items():
            if ts not in state.index:
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                continue
            pos = positions[label]

            if pos is None and row["cross_up"] and regime_ok(ts):
                notional = capital * PER_ASSET_ALLOCATION_PCT / 100.0
                if notional > free_capital:
                    rejected_for_cash += 1
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + ENTRY_COST)
                    positions[label] = {"entry_price": fill, "notional": notional, "entry_time": ts}
                    free_capital -= notional

            elif pos is not None and row["cross_down"]:
                exit_price = row["close"] * (1 - EXIT_COST)
                ret = (exit_price / pos["entry_price"]) - 1
                pnl = pos["notional"] * ret
                capital += pnl
                trade_log.append({
                    "asset": label, "entry_time": pos["entry_time"], "exit_time": ts,
                    "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital,
                })
                positions[label] = None

    if rejected_for_cash:
        print(f"  [capital constraint] {rejected_for_cash} entries rejected for insufficient free cash "
              f"at {PER_ASSET_ALLOCATION_PCT:.0f}% allocation")
    return pd.DataFrame(trade_log).sort_values("exit_time")


def summarize(trades: pd.DataFrame, label: str) -> None:
    print(f"\n===== {label} =====")
    if trades.empty:
        print("no trades")
        return
    wins = trades[trades.pnl > 0]
    gross_win = wins.pnl.sum()
    gross_loss = -trades[trades.pnl <= 0].pnl.sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    final_capital = trades.capital_after.iloc[-1]
    print(f"trades: {len(trades)}  win_rate: {len(wins)/len(trades)*100:.1f}%  "
          f"profit_factor: {pf:.2f}  final_capital: ${final_capital:,.0f}  "
          f"total_return: {(final_capital/STARTING_CAPITAL - 1)*100:.1f}%")

    peak, max_dd = STARTING_CAPITAL, 0.0
    for c in trades.capital_after:
        peak = max(peak, c)
        max_dd = min(max_dd, (c/peak - 1) * 100)
    print(f"max_drawdown: {max_dd:.1f}%")

    # --- CAGR: exact, uses real elapsed calendar time, no approximation ---
    start_dt = trades["entry_time"].min()
    end_dt = trades["exit_time"].max()
    years = max((end_dt - start_dt).days / 365.25, 1e-9)
    cagr = (final_capital / STARTING_CAPITAL) ** (1 / years) - 1
    print(f"elapsed: {years:.2f} years ({start_dt.date()} to {end_dt.date()})  CAGR: {cagr*100:.2f}%")

    # --- Sharpe/Sortino: APPROXIMATION, flagged as such. True daily-equity
    # Sharpe would mark-to-market all 6 overlapping positions every day;
    # this instead uses per-trade returns annualized by trade frequency,
    # which is a reasonable read on risk-adjusted trade quality but is NOT
    # the same rigor as a daily-equity-curve Sharpe (it ignores time actually
    # spent in each trade and cross-asset overlap). Treat as directional.
    trades_per_year = len(trades) / years
    r = trades["ret_pct"].to_numpy() / 100.0
    mean_r, std_r = r.mean(), r.std(ddof=1)
    sharpe_approx = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else float("nan")
    downside = r[r < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else float("nan")
    sortino_approx = (mean_r / downside_std) * np.sqrt(trades_per_year) if downside_std and downside_std > 0 else float("nan")
    print(f"[APPROX, per-trade not daily-equity] Sharpe: {sharpe_approx:.2f}  Sortino: {sortino_approx:.2f}  "
          f"(mean trade ret: {mean_r*100:.2f}%  std: {std_r*100:.2f}%  n={len(r)}  trades/yr: {trades_per_year:.1f})")


def sweep_allocation(regime: pd.Series, target_dd_pct: float, allocations=(15, 20, 25, 30, 35, 40, 45, 50)):
    """Reruns the FILTERED backtest at increasing per-asset allocation sizes
    to find where its max drawdown comes back up to a target (e.g. the
    baseline's own drawdown), so 'the filter frees up risk budget' is an
    empirically tested claim, not a linear-scaling assumption."""
    import tools.run_ma200_cross_strategy as base_mod

    print(f"\nsweeping allocation size on the REGIME-FILTERED backtest "
          f"(target: max_drawdown near {target_dd_pct:.1f}%)...")
    results = []
    original_alloc = base_mod.PER_ASSET_ALLOCATION_PCT
    try:
        for alloc in allocations:
            base_mod.PER_ASSET_ALLOCATION_PCT = alloc
            globals()["PER_ASSET_ALLOCATION_PCT"] = alloc
            trades = run_filtered_backtest(regime)
            if trades.empty:
                continue
            final_capital = trades.capital_after.iloc[-1]
            peak, max_dd = STARTING_CAPITAL, 0.0
            for c in trades.capital_after:
                peak = max(peak, c)
                max_dd = min(max_dd, (c / peak - 1) * 100)
            years = max((trades["exit_time"].max() - trades["entry_time"].min()).days / 365.25, 1e-9)
            cagr = (final_capital / STARTING_CAPITAL) ** (1 / years) - 1
            results.append((alloc, max_dd, cagr * 100, final_capital))
            print(f"  alloc={alloc:>3.0f}%  max_dd={max_dd:>7.1f}%  CAGR={cagr*100:>7.2f}%  "
                  f"final=${final_capital:,.0f}")
    finally:
        base_mod.PER_ASSET_ALLOCATION_PCT = original_alloc
        globals()["PER_ASSET_ALLOCATION_PCT"] = original_alloc

    closest = min(results, key=lambda r: abs(r[1] - target_dd_pct)) if results else None
    if closest:
        print(f"\nclosest match to target drawdown: alloc={closest[0]:.0f}%  "
              f"max_dd={closest[1]:.1f}%  CAGR={closest[2]:.2f}%  final=${closest[3]:,.0f}")
    return results


if __name__ == "__main__":
    print("fetching HYG/IEF/UUP and building regime series...")
    regime = build_regime_series()
    print(f"regime series: {len(regime)} days, "
          f"{regime.sum()} RISK_ON ({regime.mean()*100:.1f}%), "
          f"{(~regime).sum()} filtered ({(1-regime.mean())*100:.1f}%)")

    print("\nrunning UNFILTERED baseline (should match run_ma200_cross_strategy.py --backtest)...")
    baseline = run_filtered_backtest(pd.Series(True, index=regime.index))
    summarize(baseline, "BASELINE (no regime filter)")

    print("\nrunning REGIME-FILTERED version...")
    filtered = run_filtered_backtest(regime)
    summarize(filtered, "REGIME-FILTERED (credit + dollar + FOMC event gate)")

    baseline_peak, baseline_max_dd = STARTING_CAPITAL, 0.0
    for c in baseline.capital_after:
        baseline_peak = max(baseline_peak, c)
        baseline_max_dd = min(baseline_max_dd, (c / baseline_peak - 1) * 100)
    sweep_allocation(regime, target_dd_pct=baseline_max_dd)

    if not baseline.empty and not filtered.empty:
        cut = baseline[~baseline["entry_time"].isin(filtered["entry_time"])]
        print(f"\n{len(cut)} trades cut by the filter. Their combined stats:")
        if not cut.empty:
            w = cut[cut.pnl > 0]
            print(f"  win_rate of CUT trades: {len(w)/len(cut)*100:.1f}%  "
                  f"total_pnl of CUT trades: ${cut.pnl.sum():,.0f}  "
                  f"median ret: {cut['ret_pct'].median():.2f}%  mean ret: {cut['ret_pct'].mean():.2f}%  "
                  f"(negative pnl = filter correctly avoided net-losing trades; "
                  f"positive = filter is cutting real winners)")
            print(f"  top 3 cut trades by return: "
                  f"{cut.nlargest(3, 'ret_pct')[['asset','entry_time','ret_pct']].to_dict('records')}")
