"""Traces the real intra-trade price path for every historical golden-cross
trade -- not just entry price -> exit price, but what actually happened to
price in between, normalized to % return from entry and % of trade duration
so trades of very different lengths (4 hours to 6 months in this dataset)
overlay comparably.

Reuses the exact same entry/exit logic already verified in
run_ma200_cross_strategy.py's run_backtest() (golden cross, 50MA/200MA,
same ASSETS/cost assumptions) -- this only adds path capture on top, it
doesn't change what counts as a trade.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_ma200_cross_strategy import ASSETS, load_full_history, ENTRY_COST, EXIT_COST
from trading_bot_v4.signals.ma200_trend import compute_trend_state

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "trade_tracer_data.json"
PATH_POINTS = 40  # resampled points per trade path, evenly spaced across % of duration


def _resample_path(pct_path: np.ndarray) -> list[float]:
    """Resample a trade's raw bar-by-bar % path onto PATH_POINTS evenly
    spaced points across 0-100% of duration, so trades with wildly
    different bar counts (a 1-bar trade vs. a 300-bar trade) can be
    overlaid on the same x-axis honestly."""
    if len(pct_path) == 1:
        return [float(pct_path[0])] * PATH_POINTS
    x_orig = np.linspace(0, 100, len(pct_path))
    x_target = np.linspace(0, 100, PATH_POINTS)
    return np.interp(x_target, x_orig, pct_path).tolist()


def compute_trades_for_asset(label: str, pair: str, path: str) -> list[dict]:
    df = load_full_history(pair, path)
    state = compute_trend_state(df["close"])
    close = state["close"]

    trades = []
    pos = None
    for ts, row in state.iterrows():
        if pd.isna(row["ma200"]):
            continue
        if pos is None and row["cross_up"]:
            pos = {"entry_time": ts, "entry_price": row["close"]}
        elif pos is not None and row["cross_down"]:
            entry_time, entry_price = pos["entry_time"], pos["entry_price"]
            segment = close.loc[entry_time:ts]
            raw_pct = ((segment / entry_price) - 1) * 100
            fill_entry = entry_price * (1 + ENTRY_COST)
            fill_exit = row["close"] * (1 - EXIT_COST)
            net_ret_pct = ((fill_exit / fill_entry) - 1) * 100
            trades.append({
                "asset": label,
                "entry_time": entry_time.isoformat(),
                "exit_time": ts.isoformat(),
                "duration_days": round((ts - entry_time).total_seconds() / 86400, 2),
                "entry_price": float(entry_price),
                "exit_price": float(row["close"]),
                "net_ret_pct": round(net_ret_pct, 2),
                "max_drawdown_pct": round(float(raw_pct.min()), 2),
                "max_runup_pct": round(float(raw_pct.max()), 2),
                "path": [round(v, 3) for v in _resample_path(raw_pct.to_numpy())],
            })
            pos = None
    return trades


def main() -> None:
    all_trades = []
    for label, (pair, path) in ASSETS.items():
        trades = compute_trades_for_asset(label, pair, path)
        all_trades.extend(trades)
        wins = [t for t in trades if t["net_ret_pct"] > 0]
        print(f"{label}: {len(trades)} trades, {len(wins)} wins ({len(wins)/len(trades)*100:.0f}%)" if trades else f"{label}: 0 trades")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "path_points": PATH_POINTS,
        "trades": all_trades,
    }), encoding="utf-8")
    print(f"\ntotal: {len(all_trades)} trades -> {OUT_PATH}")


if __name__ == "__main__":
    main()
