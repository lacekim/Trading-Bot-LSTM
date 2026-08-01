"""Research-only grid for consolidating consecutive legacy VVV signals."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def simulate(frame, threshold, stop_atr, target_atr, max_hold, cost_bps=0.0):
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    capital = peak = 100000.0
    minimum = 0.0
    trades = wins = 0
    position = None
    pnls = []
    for i in range(len(frame) - 1):
        row, nxt = frame.iloc[i], frame.iloc[i + 1]
        prob = float(row.model_probability)
        signal = 1 if prob > threshold else (-1 if prob < 1.0 - threshold else 0)
        if position is None and signal:
            entry, atr = float(row.close), float(row.atr)
            distance = atr * stop_atr
            units = min(capital * 0.01 / distance, capital / entry * 0.95)
            position = {"side": signal, "entry": entry, "units": units, "atr": atr, "bars": 0}
        if position is None:
            continue
        position["bars"] += 1
        side, entry, atr = position["side"], position["entry"], position["atr"]
        stop = entry - side * atr * stop_atr
        target = entry + side * atr * target_atr
        high, low = float(nxt.high), float(nxt.low)
        exit_price = reason = None
        if side == 1 and low <= stop or side == -1 and high >= stop:
            exit_price, reason = stop, "stop"
        elif side == 1 and high >= target or side == -1 and low <= target:
            exit_price, reason = target, "target"
        elif signal == -side:
            exit_price, reason = float(row.close), "reversal"
        elif position["bars"] >= max_hold:
            exit_price, reason = float(nxt.close), "window"
        if exit_price is None:
            continue
        gross = side * (exit_price - entry) * position["units"]
        cost = entry * position["units"] * cost_bps / 10000.0
        pnl = gross - cost
        capital += pnl
        pnls.append(pnl)
        trades += 1
        wins += pnl > 0
        peak = max(peak, capital)
        minimum = min(minimum, (capital / peak - 1.0) * 100.0)
        position = None
    if position is not None:
        exit_price = float(frame.iloc[-1].close)
        pnl = position["side"] * (exit_price - position["entry"]) * position["units"]
        capital += pnl
        pnls.append(pnl)
        trades += 1
    gains = sum(value for value in pnls if value > 0)
    losses = abs(sum(value for value in pnls if value < 0))
    return {"return_pct": (capital / 100000.0 - 1) * 100, "trades": trades,
            "win_rate_pct": wins / trades * 100 if trades else 0,
            "max_drawdown_pct": minimum, "profit_factor": gains / losses if losses else float("inf")}


def main():
    cache = pd.read_pickle("data/cache/v5_original_long_research_signals.pkl")
    frame = cache["signals"]["VVV"]
    rows = []
    for values in itertools.product((0.58, 0.60, 0.62, 0.64, 0.66),
                                    (0.5, 0.75, 1.0, 1.5, 2.0),
                                    (1.5, 3.0, 5.0, 8.0, 12.0, 20.0),
                                    (4, 8, 12, 24, 48, 96, 240)):
        result = simulate(frame, *values, cost_bps=24.0)
        rows.append({"threshold": values[0], "stop_atr": values[1], "target_atr": values[2],
                     "max_hold": values[3], **result})
    results = pd.DataFrame(rows).sort_values(["return_pct", "trades"], ascending=[False, True])
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/v5_vvv_position_persistence_grid.csv", index=False)
    passing = results[(results.return_pct >= 200) & (results.trades <= 1000)]
    print(json.dumps({"passing_candidates": len(passing),
                      "best": results.head(10).to_dict("records"),
                      "best_passing": passing.head(10).to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
