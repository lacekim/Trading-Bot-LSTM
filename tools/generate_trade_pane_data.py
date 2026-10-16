"""Exports real OHLC candles + real historical trades for the 6 golden-cross
crypto assets, for the FreqUI-style "Trade" pane (candlestick chart with
entry/exit markers, sidebar pair list, closed trades table) -- this is the
piece the earlier trade-tracer work skipped by using normalized % paths
instead of actual price candles. Real freqtrade shows real candles; this
does too.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from run_ma200_cross_strategy import ASSETS, load_full_history, ENTRY_COST, EXIT_COST
from trading_bot_v4.signals.ma200_trend import compute_trend_state

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "trade_pane_data.json"
CHART_BARS = 400  # most recent N 4h bars shown by default, same idea as freqtrade's default chart window


def compute_trades(state: pd.DataFrame) -> list[dict]:
    trades = []
    pos = None
    for ts, row in state.iterrows():
        if pd.isna(row["ma200"]):
            continue
        if pos is None and row["cross_up"]:
            pos = {"entry_time": ts, "entry_price": row["close"]}
        elif pos is not None and row["cross_down"]:
            fill_entry = pos["entry_price"] * (1 + ENTRY_COST)
            fill_exit = row["close"] * (1 - EXIT_COST)
            net_ret_pct = (fill_exit / fill_entry - 1) * 100
            trades.append({
                "entry_time": pos["entry_time"].isoformat(), "exit_time": ts.isoformat(),
                "entry_price": float(pos["entry_price"]), "exit_price": float(row["close"]),
                "net_ret_pct": float(net_ret_pct),
            })
            pos = None
    return trades


def main() -> None:
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "assets": {}}
    for label, (pair, path) in ASSETS.items():
        df = load_full_history(pair, path)
        state = compute_trend_state(df["close"])
        trades = compute_trades(state)

        recent = df.tail(CHART_BARS)
        recent_state = state.loc[recent.index]
        candles = []
        for ts in recent.index:
            row = recent.loc[ts]
            srow = recent_state.loc[ts]
            candles.append({
                "time": ts.isoformat(), "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]), "volume": float(row["volume"]),
                "ma50": None if pd.isna(srow["ma50"]) else float(srow["ma50"]),
                "ma200": None if pd.isna(srow["ma200"]) else float(srow["ma200"]),
            })

        out["assets"][label] = {
            "pair": f"{label}/USD", "candles": candles, "trades": trades,
            "n_trades": len(trades),
        }
        print(f"{label}: {len(candles)} candles exported, {len(trades)} real historical trades")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
