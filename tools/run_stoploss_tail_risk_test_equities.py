"""Section E5: the E1-E3b hard-stop test, but for the equities/bonds/
commodities universe -- NOT a copy of crypto's -8% answer.

E1-E3b (tools/run_stoploss_tail_risk_test.py) validated -8% against the 4h
Kraken crypto golden-cross qualifier only. That universe's volatility
profile (crypto, 4h bars) has nothing to do with daily-bar equities/bonds/
commodities, so that -8% number carries zero evidence for this universe.
Section B is putting both on the same Alpaca account, so both need their
own independently-found stop, or the crypto side gets a real safety net
while the equities side (the bulk of Alpaca's book) still has none.

Reuses the exact real backtest this project already trusts for this
universe -- tools/run_cross_asset_golden_cross_backtest.py's own
compute_trend_state, cost model (12bps round trip), and per-trade sizing
(2%, not crypto's 15% -- this universe can have far more concurrent open
positions) -- plus A2's cash-constrained-entry fix (reject an entry that
would exceed capital minus what's already deployed), which that script's
own event-driven simulate_portfolio() deliberately does NOT apply (see its
own docstring: "an accepted simplification, not a bug" for measuring raw
edge -- but not the right assumption for a tail-risk/stop test, same
reasoning as A2's crypto fix).

Universe: the 60 non-crypto symbols already marked robust in
runtime/cross_asset_golden_cross_results.json (58 equities, IEF, and the
2 commodity futures GC=F/PA=F -- fetched under their own backtested
tickers, not the GLD/PPLT proxies only used for live Alpaca execution).

Stop distances: baseline (none), -3%, -5%, -8%, -12% -- deliberately NOT
reusing crypto's -5/-8/-15/-25% set; equities move less per day than
crypto per 4h bar, so a tighter starting grid is the right shape to test,
per the handoff's own instruction not to assume the crypto thresholds
apply here.
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

RESULTS_PATH = Path(__file__).resolve().parents[1] / "runtime" / "cross_asset_golden_cross_results.json"
STARTING_CAPITAL = 100_000.0
PER_TRADE_ALLOCATION_PCT = 2.0  # matches run_cross_asset_golden_cross_backtest.py's own sizing
STOP_DISTANCES_PCT = [None, -3.0, -5.0, -8.0, -12.0]
BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 2.0


def _load_equities_universe() -> list[str]:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return sorted({
        r["symbol"] for r in data["results"]
        if r.get("robust") and r.get("asset_class") != "Crypto"
    })


def fetch_ohlc_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    session = cffi_requests.Session(impersonate="chrome")
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        try:
            data = yf.download(batch, period="max", interval="1d", session=session,
                                progress=False, group_by="ticker", threads=False)
        except Exception as exc:
            print(f"batch failed: {exc}")
            time.sleep(BATCH_DELAY_SECONDS)
            continue
        for sym in batch:
            try:
                df = data[sym] if len(batch) > 1 else data
                df = df[["High", "Low", "Close"]].dropna()
                df.columns = ["high", "low", "close"]
            except (KeyError, TypeError):
                continue
            if len(df) < 210:
                continue
            out[sym] = df
        time.sleep(BATCH_DELAY_SECONDS)
    return out


def run_backtest(price: dict[str, pd.DataFrame], stop_pct: float | None) -> pd.DataFrame:
    """Same cash-constrained multi-asset loop as
    tools/run_stoploss_tail_risk_test.py, adapted to daily bars and this
    universe's own 2% sizing."""
    states = {sym: compute_trend_state(df["close"]) for sym, df in price.items()}
    all_index = sorted(set().union(*[s.index for s in states.values()]))
    capital = STARTING_CAPITAL
    positions: dict[str, dict | None] = {sym: None for sym in price}
    trade_log = []

    for ts in all_index:
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional

        for sym, state in states.items():
            if ts not in state.index:
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                continue
            pos = positions[sym]

            if pos is not None:
                stop_hit = False
                if stop_pct is not None and ts in price[sym].index:
                    bar_low = price[sym].loc[ts, "low"]
                    stop_price = pos["entry_price"] * (1 + stop_pct / 100)
                    if bar_low <= stop_price:
                        stop_hit = True

                if stop_hit:
                    exit_price = stop_price * (1 - EXIT_COST)
                    ret = (exit_price / pos["entry_price"]) - 1
                    pnl = pos["notional"] * ret
                    capital += pnl
                    trade_log.append({
                        "asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                        "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital, "exit_reason": "stop",
                    })
                    positions[sym] = None
                    free_capital += pos["notional"]
                    continue

                if row["cross_down"]:
                    exit_price = row["close"] * (1 - EXIT_COST)
                    ret = (exit_price / pos["entry_price"]) - 1
                    pnl = pos["notional"] * ret
                    capital += pnl
                    trade_log.append({
                        "asset": sym, "entry_time": pos["entry_time"], "exit_time": ts,
                        "ret_pct": ret * 100, "pnl": pnl, "capital_after": capital, "exit_reason": "signal",
                    })
                    positions[sym] = None
                    free_capital += pos["notional"]

            elif pos is None and row["cross_up"]:
                notional = capital * PER_TRADE_ALLOCATION_PCT / 100.0
                if notional > free_capital:
                    continue
                if notional > 100:
                    fill = row["close"] * (1 + ENTRY_COST)
                    positions[sym] = {"entry_price": fill, "notional": notional, "entry_time": ts}
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


def e3b_check(trades: pd.DataFrame, label: str) -> None:
    """Exclude each asset's single worst loss and recompute PF -- the same
    house standard applied to crypto's -8% (run_stoploss_tail_risk_test.py)
    and to this project's own robustness bar elsewhere (PF excl. top-3)."""
    if trades.empty:
        return
    baseline_pf = summarize(trades, f"{label} (before E3b exclusion)").get("pf", float("nan"))

    worst_idx = trades.loc[trades.groupby("asset")["ret_pct"].idxmin()]
    print(f"\nexcluded trades (single worst loss per asset, {label}):")
    print(worst_idx[["asset", "entry_time", "exit_time", "ret_pct", "exit_reason"]].to_string(index=False))

    remaining = trades.drop(worst_idx.index)
    wins = remaining[remaining.pnl > 0]
    gross_win = wins.pnl.sum()
    gross_loss = -remaining[remaining.pnl <= 0].pnl.sum()
    pf_excl = gross_win / gross_loss if gross_loss > 0 else float("inf")
    print(f"\n{label} PF EXCLUDING each asset worst loss "
          f"({len(remaining)} of {len(trades)} trades): {pf_excl:.2f}  (was {baseline_pf:.2f})")


def main() -> None:
    symbols = _load_equities_universe()
    print(f"E5 universe: {len(symbols)} non-crypto robust instruments")
    price = fetch_ohlc_batch(symbols)
    print(f"usable OHLC history for {len(price)} of {len(symbols)} instruments\n")

    results = {}
    trades_by_label = {}
    for stop_pct in STOP_DISTANCES_PCT:
        label = "BASELINE (signal-only exit, no stop)" if stop_pct is None else f"STOP at {stop_pct:.0f}%"
        trades = run_backtest(price, stop_pct)
        trades_by_label[label] = trades
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

    # E3b -- run automatically against every stop variant that both improved
    # PF and cut max drawdown vs. baseline (the candidates worth trusting),
    # not just whichever looks best on paper before checking it's real.
    print(f"\n{'='*78}\nE3b robustness check (each candidate stop that beat baseline on PF+maxDD)\n{'='*78}")
    for label, r in results.items():
        if label == baseline_key or not r:
            continue
        if r["pf"] > baseline["pf"] and r["max_dd_pct"] > baseline["max_dd_pct"]:
            e3b_check(trades_by_label[label], label)


if __name__ == "__main__":
    main()
