"""Section E: does a real hard stoploss reduce tail risk enough to be worth
its cost in PF/CAGR, or does the already-verified signal-only exit stay
correct? Not a return-boost question -- a tail-risk question, prompted by
A0 (the Alpaca account running 4x leveraged undetected) as motivation for
an independent safety net against future sizing/leverage bugs.

Reuses the exact same real data, cash-constrained entry logic (A2's fix),
costs, and 6-asset universe as tools/run_ma200_cross_strategy.py -- not a
separate, potentially-diverging implementation. The only new behavior is
an optional stop check: for an open position, if a bar's LOW breaches
entry_price * (1 + stop_pct) before the death-cross signal would have
exited it anyway, exit at the stop price intrabar instead of waiting for
the signal close. Checking the bar's low (not just its close) is the more
realistic assumption for how a real stop order actually fills.

Stop distances tested: none (baseline), -5%, -8%, -15%, -25% -- the last
three reuse ma200_position_monitor.py's own existing alert thresholds for
continuity, -5% added to see the shape of the tradeoff at a tighter
distance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from trading_bot_v4.signals.ma200_trend import compute_trend_state
from tools.run_ma200_cross_strategy import ASSETS, ENTRY_COST, EXIT_COST, PER_ASSET_ALLOCATION_PCT, STARTING_CAPITAL, load_full_history

STOP_DISTANCES_PCT = [None, -5.0, -8.0, -15.0, -25.0]  # None = baseline, signal-only exit


def run_backtest(price: dict, stop_pct: float | None) -> pd.DataFrame:
    """Same cash-constrained multi-asset loop as run_ma200_cross_strategy.py
    (A2's fix included), plus an optional intrabar stop check."""
    all_index = sorted(set().union(*[d["state"].index for d in price.values()]))
    capital = STARTING_CAPITAL
    positions = {label: None for label in ASSETS}
    trade_log = []

    for ts in all_index:
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional

        for label, df in price.items():
            state = df["state"]
            ohlc = df["ohlc"]
            if ts not in state.index:
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                continue
            pos = positions[label]

            if pos is not None:
                stop_hit = False
                if stop_pct is not None and ts in ohlc.index:
                    bar_low = ohlc.loc[ts, "low"]
                    stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                    if bar_low <= stop_price:
                        stop_hit = True

                if stop_hit:
                    exit_price = stop_price * (1 - EXIT_COST)
                    ret = (exit_price / pos["entry_price"]) - 1
                    pnl = pos["notional"] * ret
                    capital += pnl
                    trade_log.append({
                        "asset": label, "entry_time": pos["entry_time"], "exit_time": ts,
                        "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital, "exit_reason": "stop",
                    })
                    positions[label] = None
                    free_capital += pos["notional"]
                    continue  # can't also signal-exit the same bar

                if row["cross_down"]:
                    exit_price = row["close"] * (1 - EXIT_COST)
                    ret = (exit_price / pos["entry_price"]) - 1
                    pnl = pos["notional"] * ret
                    capital += pnl
                    trade_log.append({
                        "asset": label, "entry_time": pos["entry_time"], "exit_time": ts,
                        "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital, "exit_reason": "signal",
                    })
                    positions[label] = None
                    free_capital += pos["notional"]

            elif pos is None and row["cross_up"]:
                notional = capital * PER_ASSET_ALLOCATION_PCT / 100.0
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + ENTRY_COST)
                    positions[label] = {"entry_price": fill, "notional": notional, "entry_time": ts}
                    free_capital -= notional

    return pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()


def summarize(trades: pd.DataFrame, label: str) -> dict:
    print(f"\n===== {label} =====")
    if trades.empty:
        print("no trades")
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
    stop_exits = (trades["exit_reason"] == "stop").sum() if "exit_reason" in trades else 0

    print(f"trades: {len(trades)}  (stop exits: {stop_exits})  win_rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"profit_factor: {pf:.2f}  CAGR: {cagr*100:.2f}%  total_return: {total_return:,.1f}%")
    print(f"max_drawdown: {max_dd:.1f}%  worst_single_trade: {worst_trade:.1f}%  "
          f"Sharpe (per-trade, not annualized): {sharpe:.2f}")

    return {
        "n_trades": len(trades), "stop_exits": int(stop_exits), "win_rate": len(wins)/len(trades)*100,
        "pf": pf, "cagr_pct": cagr * 100, "total_return_pct": total_return,
        "max_dd_pct": max_dd, "worst_trade_pct": worst_trade, "sharpe": sharpe,
    }


def main() -> None:
    price = {}
    for label, (pair, path) in ASSETS.items():
        df = load_full_history(pair, path)
        state = compute_trend_state(df["close"])
        price[label] = {"state": state, "ohlc": df}
        print(f"{label}: {len(state)} bars, {state.index.min().date()} to {state.index.max().date()}")

    results = {}
    for stop_pct in STOP_DISTANCES_PCT:
        label = "BASELINE (signal-only exit, no stop)" if stop_pct is None else f"STOP at {stop_pct:.0f}%"
        trades = run_backtest(price, stop_pct)
        results[label] = summarize(trades, label)

    baseline_key = "BASELINE (signal-only exit, no stop)"
    baseline = results[baseline_key]
    print(f"\n{'='*78}\nSUMMARY vs. baseline (worst_trade {baseline.get('worst_trade_pct', float('nan')):.1f}%, "
          f"max_dd {baseline.get('max_dd_pct', float('nan')):.1f}%, CAGR {baseline.get('cagr_pct', float('nan')):.1f}%, "
          f"PF {baseline.get('pf', float('nan')):.2f})\n{'='*78}")
    for label, r in results.items():
        if label == baseline_key or not r:
            continue
        print(f"{label:20s} worst_trade {r['worst_trade_pct']:+7.1f}% (Δ{r['worst_trade_pct']-baseline['worst_trade_pct']:+6.1f}pp)  "
              f"max_dd {r['max_dd_pct']:+7.1f}% (Δ{r['max_dd_pct']-baseline['max_dd_pct']:+6.1f}pp)  "
              f"CAGR {r['cagr_pct']:+7.1f}% (Δ{r['cagr_pct']-baseline['cagr_pct']:+6.1f}pp)  "
              f"PF {r['pf']:.2f} (Δ{r['pf']-baseline['pf']:+.2f})")


if __name__ == "__main__":
    main()
