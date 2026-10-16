"""Same honest FVG test as run_fvg_strategy_test.py, run on the real 72-
stock equity universe (see cross_asset_tracker.py's UNIVERSE) instead of
the 6 crypto assets -- reuses the exact same FVG detection and trade
simulation logic (imported directly, not reimplemented) so the two tests
are apples-to-apples. Daily bars (Yahoo Finance via yfinance) instead of
4h, since free intraday equity history isn't available the way crypto's
is; trend filter is the same 50/200 SMA golden-cross state, just computed
on daily closes instead of 4h closes.

2% per-trade allocation (not crypto test's 15%) matches the convention
already used for the cross-asset equities backtest earlier this session --
72 simultaneous candidates makes 15%-per-asset absurd.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading_bot_v4.signals.ma200_trend import compute_trend_state

from run_fvg_strategy_test import detect_fvgs, simulate_asset, R_MULTIPLES_TO_TEST, STARTING_CAPITAL

PER_ASSET_ALLOCATION_PCT = 2.0
MAX_STALE_DAYS = 5

EQUITIES = [
    "SPY", "QQQ", "DIA", "IWM", "VTI",
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "ADBE", "AMD",
    "JPM", "BAC", "WFC", "GS", "MS", "BRK-B", "V", "MA", "AXP",
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "BKNG",
    "WMT", "PG", "KO", "PEP", "COST", "PM",
    "CAT", "BA", "HON", "UPS", "GE", "LMT", "RTX",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "NEE", "DUK", "SO",
    "LIN", "SHW", "FCX", "NEM",
    "PLD", "AMT", "SPG",
    "NFLX", "DIS", "CMCSA", "T", "VZ",
]


def fetch_daily(symbol: str, session) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period="max", interval="1d", session=session,
                          progress=False, auto_adjust=True)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    if len(df) < 210:
        return None
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def main() -> None:
    session = cffi_requests.Session(impersonate="chrome")
    price, trends, failures = {}, {}, []
    for symbol in EQUITIES:
        df = fetch_daily(symbol, session)
        if df is None:
            failures.append(symbol)
            continue
        price[symbol] = df
        trends[symbol] = compute_trend_state(df["close"])
        print(f"{symbol}: {len(df)} bars, {df.index.min().date()} to {df.index.max().date()}")

    if failures:
        print(f"\n{len(failures)} symbols failed to fetch, skipped: {failures}")
    print(f"\n{len(price)} of {len(EQUITIES)} equities loaded successfully")

    for r_multiple in R_MULTIPLES_TO_TEST:
        print(f"\n{'='*70}\nR MULTIPLE = {r_multiple}\n{'='*70}")
        all_trades = []
        for symbol, df in price.items():
            trades = simulate_asset(symbol, df, trends[symbol], r_multiple)
            for t in trades:
                t["pnl"] = t["pnl"] / 15.0 * PER_ASSET_ALLOCATION_PCT  # rescale from imported 15% sizing to 2%
            all_trades.extend(trades)

        if not all_trades:
            print("no trades at all -- zero FVG setups triggered under these rules")
            continue

        trades_df = pd.DataFrame(all_trades).sort_values("exit_time")
        wins = trades_df[trades_df.pnl > 0]
        losses = trades_df[trades_df.pnl <= 0]
        gross_win = wins.pnl.sum()
        gross_loss = -losses.pnl.sum()
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        top3 = trades_df.nlargest(3, "pnl")
        trades_excl_top3 = trades_df.drop(top3.index)
        wins_excl = trades_excl_top3[trades_excl_top3.pnl > 0]
        losses_excl = trades_excl_top3[trades_excl_top3.pnl <= 0]
        gross_win_excl = wins_excl.pnl.sum()
        gross_loss_excl = -losses_excl.pnl.sum()
        pf_excl = gross_win_excl / gross_loss_excl if gross_loss_excl > 0 else float("inf")

        print(f"total trades: {len(trades_df)}  win rate: {len(wins)/len(trades_df)*100:.1f}%")
        print(f"profit factor: {pf:.2f}   profit factor excl. top-3 winners: {pf_excl:.2f}")
        print(f"total pnl: ${trades_df.pnl.sum():,.0f}")

        print("\nby side:")
        for side, g in trades_df.groupby("side"):
            w = g[g.pnl > 0]
            wr = len(w) / len(g) * 100 if len(g) else 0
            gw = w.pnl.sum()
            gl = -g[g.pnl <= 0].pnl.sum()
            side_pf = gw / gl if gl > 0 else float("inf")
            print(f"  {side:5} n={len(g):5} win_rate={wr:5.1f}% pf={side_pf:5.2f} total_pnl=${g.pnl.sum():>12,.0f}")

        print("\ntop 10 assets by total pnl:")
        by_asset = trades_df.groupby("asset").pnl.agg(["sum", "count"]).sort_values("sum", ascending=False)
        for symbol, row in by_asset.head(10).iterrows():
            print(f"  {symbol:6} n={int(row['count']):4} total_pnl=${row['sum']:>10,.0f}")

        print("\nbottom 10 assets by total pnl:")
        for symbol, row in by_asset.tail(10).iterrows():
            print(f"  {symbol:6} n={int(row['count']):4} total_pnl=${row['sum']:>10,.0f}")


if __name__ == "__main__":
    main()
