"""Verify: (1) how often multiple assets are concurrently long in the real
golden-cross backtest, and (2) what happens to the reported 45%-allocation
"match baseline drawdown" result once entries are actually capital-constrained
(can't deploy more notional than free cash allows) instead of always sizing
off total capital regardless of what's already deployed.

Uses ONLY local feather history (no live fetch -- this box has no network
egress to Kraken), which is fine: we're checking structural concurrency and
capital feasibility, not the freshest last few candles.
"""
import os
import sys
from pathlib import Path

BASE = os.environ.get("PROJECT_BASE", "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera")
sys.path.insert(0, BASE)

import pandas as pd
from trading_bot_v4.signals.ma200_trend import compute_trend_state

FEE_BPS = 5
SLIP_BPS = 7
ENTRY_COST = (FEE_BPS + SLIP_BPS) / 10000
EXIT_COST = (FEE_BPS + SLIP_BPS) / 10000
STARTING_CAPITAL = 100_000.0

ASSETS = {
    "BTC": f"{BASE}/freqtrade-develop/user_data/data/kraken/BTC_USD-1h.feather",
    "ETH": f"{BASE}/freqtrade-develop/user_data/data/kraken/ETH_USD-1h.feather",
    "XMR": f"{BASE}/freqtrade-develop/user_data/data/kraken/XMR_USD-1h.feather",
    "PAXG": f"{BASE}/freqtrade-develop/user_data/data/kraken/PAXG_USD-1h.feather",
    "SOL": f"{BASE}/freqtrade-develop/user_data/data/kraken/SOL_USD-1h.feather",
    "TRX": f"{BASE}/freqtrade-develop/user_data/data/kraken/TRX_USD-1h.feather",
}


def load_local(path):
    h1 = pd.read_feather(path)
    h1["date"] = pd.to_datetime(h1["date"], utc=True)
    h1 = h1.set_index("date")
    return h1.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def run(allocation_pct, capital_constrained):
    price = {}
    for label, path in ASSETS.items():
        df = load_local(path)
        state = compute_trend_state(df["close"])
        price[label] = state

    all_index = sorted(set().union(*[s.index for s in price.values()]))
    capital = STARTING_CAPITAL
    positions = {label: None for label in ASSETS}
    trade_log = []
    max_concurrent = 0
    rejected_for_cash = 0
    concurrent_counts = []

    for ts in all_index:
        open_now = sum(1 for p in positions.values() if p is not None)
        max_concurrent = max(max_concurrent, open_now)
        if open_now > 0:
            concurrent_counts.append(open_now)

        deployed_notional = sum(p["notional"] for p in positions.values() if p is not None)
        free_capital = capital - deployed_notional  # only meaningful if capital_constrained

        for label, state in price.items():
            if ts not in state.index:
                continue
            row = state.loc[ts]
            if pd.isna(row["ma200"]):
                continue
            pos = positions[label]

            if pos is None and row["cross_up"]:
                notional = capital * allocation_pct / 100.0
                if capital_constrained:
                    if notional > free_capital:
                        rejected_for_cash += 1
                        continue
                    free_capital -= notional
                if notional > 100:
                    fill = row["close"] * (1 + ENTRY_COST)
                    positions[label] = {"entry_price": fill, "notional": notional, "entry_time": ts}

            elif pos is not None and row["cross_down"]:
                exit_price = row["close"] * (1 - EXIT_COST)
                ret = (exit_price / pos["entry_price"]) - 1
                pnl = pos["notional"] * ret
                capital += pnl
                trade_log.append({"asset": label, "exit_time": ts, "pnl": pnl, "capital_after": capital})
                positions[label] = None

    trades = pd.DataFrame(trade_log).sort_values("exit_time") if trade_log else pd.DataFrame()
    peak = STARTING_CAPITAL
    max_dd = 0.0
    for c in trades["capital_after"] if len(trades) else []:
        peak = max(peak, c)
        max_dd = min(max_dd, (c / peak - 1) * 100)

    return {
        "allocation_pct": allocation_pct,
        "capital_constrained": capital_constrained,
        "n_trades": len(trades),
        "final_capital": capital,
        "total_return_pct": (capital / STARTING_CAPITAL - 1) * 100,
        "max_dd_pct": max_dd,
        "max_concurrent_positions": max_concurrent,
        "rejected_for_cash": rejected_for_cash,
        "pct_time_with_any_position_open": len(concurrent_counts) / len(all_index) * 100,
        "avg_concurrent_when_any_open": (sum(concurrent_counts) / len(concurrent_counts)) if concurrent_counts else 0,
    }


if __name__ == "__main__":
    print("=== Baseline (unfiltered), 15% per-asset, UNCONSTRAINED capital (matches original script) ===")
    r = run(15.0, capital_constrained=False)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== Baseline (unfiltered), 15% per-asset, CAPITAL-CONSTRAINED (realistic) ===")
    r = run(15.0, capital_constrained=True)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== 45% per-asset, UNCONSTRAINED capital (reproduces the sweep's reported allocation) ===")
    r = run(45.0, capital_constrained=False)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== 45% per-asset, CAPITAL-CONSTRAINED (what actually happens if you can't over-deploy) ===")
    r = run(45.0, capital_constrained=True)
    for k, v in r.items():
        print(f"  {k}: {v}")
