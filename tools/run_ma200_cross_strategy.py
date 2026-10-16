"""The real, buildable system: BTC + ETH + XMR, golden cross (50MA/200MA)
entry/exit, run as one combined portfolio with real position sizing -- not
isolated 100%-capital backtests per asset (which double-count capital if
multiple signals can be on at once).

Verified separately (see trading_bot_v4/signals/ma200_trend.py) that all
three hold up even with their 3 biggest winning trades removed under the
golden cross rule -- BTC PF 1.85, ETH PF 2.61, XMR PF 1.18, excluding top 3.
The plain 200MA-only version is real too (BTC/ETH) but fails on XMR and has
a much lower win rate; golden cross is the current, active rule. SOL and
DOGE did NOT hold up under either version and are deliberately excluded.

Usage:
  --backtest   run the combined portfolio backtest over full real history
  --live       fetch current live signal for BTC, ETH, and XMR
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from trading_bot_v4.signals.ma200_trend import compute_trend_state, current_signal

FEE_BPS = 5
SLIP_BPS = 7
ENTRY_COST = (FEE_BPS + SLIP_BPS) / 10000
EXIT_COST = (FEE_BPS + SLIP_BPS) / 10000

STARTING_CAPITAL = 100_000.0
PER_ASSET_ALLOCATION_PCT = 15.0  # % of total capital deployed per asset when long
# Not 100%: most individual trades are small losses, so full-capital
# deployment on every signal compounds unnecessary drag. 15% per asset
# (90% max combined if all 6 assets are long at once, which is plausible
# during a broad market rally like the one underway right now) keeps a
# real cash buffer while still giving the rare large winners room to matter.

ASSETS = {
    "BTC": ("XBTUSD", "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/BTC_USD-1h.feather"),
    "ETH": ("ETHUSD", "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/ETH_USD-1h.feather"),
    "XMR": ("XMRUSD", "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/XMR_USD-1h.feather"),
    "PAXG": ("PAXGUSD", "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/PAXG_USD-1h.feather"),
    "SOL": ("SOLUSD", "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/SOL_USD-1h.feather"),
    "TRX": ("TRXUSD", "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/TRX_USD-1h.feather"),
}
# All six verified under the golden-cross rule specifically (PF excl. top 3
# winners: BTC 1.85, ETH 2.61, XMR 1.18, PAXG 1.56, SOL 1.03, TRX 1.15) --
# none of them passed the same robustness bar under the plain 200MA-only
# cross except BTC/ETH. PAXG is the tokenized-gold commodity representative;
# XMR has a structurally different (privacy/regulatory) demand driver, not
# just a different backtest number.


def fetch_fresh(pair: str, since_ts: int) -> pd.DataFrame:
    all_rows = []
    since = since_ts
    for _ in range(6):
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240&since={since}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        keys = [k for k in data["result"].keys() if k != "last"]
        if not keys:
            break
        rows = data["result"][keys[0]]
        if not rows:
            break
        all_rows.extend(rows)
        new_since = int(data["result"]["last"])
        if new_since <= since:
            break
        since = new_since
        time.sleep(1)
    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    fresh = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"]).drop_duplicates("ts")
    fresh["dt"] = pd.to_datetime(fresh["ts"].astype(float), unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        fresh[c] = fresh[c].astype(float)
    return fresh.set_index("dt")[["open", "high", "low", "close", "volume"]]


def load_full_history(pair: str, local_path: str) -> pd.DataFrame:
    h1 = pd.read_feather(local_path)
    h1["date"] = pd.to_datetime(h1["date"], utc=True)
    h1 = h1.set_index("date")
    hist4h = h1.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    fresh = fetch_fresh(pair, int(hist4h.index.max().timestamp()))
    df = pd.concat([hist4h, fresh[fresh.index > hist4h.index.max()]]).sort_index()
    return df[~df.index.duplicated()]


def run_backtest() -> None:
    price = {}
    for label, (pair, path) in ASSETS.items():
        df = load_full_history(pair, path)
        state = compute_trend_state(df["close"])
        price[label] = state
        print(f"{label}: {len(state)} bars, {state.index.min().date()} to {state.index.max().date()}")

    all_index = sorted(set().union(*[s.index for s in price.values()]))
    capital = STARTING_CAPITAL
    positions = {label: None for label in ASSETS}
    trade_log = []
    rejected_for_cash = 0

    for ts in all_index:
        # Real available-cash check -- sizing every entry off total capital
        # with no regard for what's already deployed silently assumes free
        # leverage once allocation gets large enough for concurrent positions
        # to sum past 100%. At the live 15% allocation this never binds (max
        # 90% gross with all 6 assets long), which is exactly why the bug
        # went unnoticed here -- but verify_capital_constrained_sizing.py
        # confirmed it's real: at 45% allocation, unconstrained sizing
        # deploys up to 270% of capital, and the real constrained result is
        # drastically worse (153 trades / -60.6% max DD / 134 rejected vs.
        # the unconstrained 285 trades / -76.3% DD / 0 rejected). Any future
        # allocation experiment needs this check to report something
        # actually achievable in an unlevered spot account.
        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional

        for label, state in price.items():
            if ts not in state.index:
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                continue
            pos = positions[label]

            if pos is None and row["cross_up"]:
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

    trades = pd.DataFrame(trade_log).sort_values("exit_time")
    print(f"\ntotal trades: {len(trades)}")
    print(f"rejected for insufficient free cash: {rejected_for_cash}  "
          f"(0 is expected at the live {PER_ASSET_ALLOCATION_PCT:.0f}% allocation; "
          f"nonzero at higher allocations means positions could not actually "
          f"all be taken in an unlevered account)")
    print(f"final capital: ${capital:,.0f}  (started ${STARTING_CAPITAL:,.0f})")
    print(f"total return: {(capital/STARTING_CAPITAL - 1)*100:.1f}%")
    wins = trades[trades.pnl > 0]
    print(f"win rate: {len(wins)/len(trades)*100:.1f}%  ({len(wins)}/{len(trades)})")

    peak = STARTING_CAPITAL
    max_dd = 0.0
    for c in trades["capital_after"]:
        peak = max(peak, c)
        max_dd = min(max_dd, (c/peak - 1) * 100)
    print(f"max drawdown (at trade-close resolution): {max_dd:.1f}%")

    print("\nby asset:")
    for label, g in trades.groupby("asset"):
        w = g[g.pnl > 0]
        print(f"  {label}: n={len(g)} win_rate={len(w)/len(g)*100:.1f}% total_pnl=${g.pnl.sum():,.0f}")


def run_live() -> None:
    for label, (pair, path) in ASSETS.items():
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        key = [k for k in data["result"].keys() if k != "last"][0]
        rows = data["result"][key]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"])
        df["dt"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
        df["close"] = df["close"].astype(float)
        df = df.set_index("dt")

        sig = current_signal(df["close"])
        print(f"\n{label}:")
        for k, v in sig.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    if args.live:
        run_live()
    if not args.backtest and not args.live:
        run_live()
