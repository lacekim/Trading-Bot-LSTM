"""Causal shared-capital selector over production-equivalent persistent trades."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(path="reports/v5_production_backtest_trades.csv"):
    frame = pd.read_csv(path)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["trade_return"] = pd.to_numeric(frame["net_pnl"], errors="coerce") / pd.to_numeric(frame["notional"], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["entry_time", "exit_time", "trade_return"])


def simulate(frame, lookback, minimum, pf_min, selected_assets, start=None, end=None):
    first = frame.entry_time.min().floor("D") if start is None else pd.Timestamp(start)
    last = frame.entry_time.max().ceil("D") if end is None else pd.Timestamp(end)
    equity = peak = 100000.0
    drawdown = 0.0
    trades = 0
    choices = []
    for day in pd.date_range(first, last, freq="D", tz="UTC"):
        known = frame[(frame.exit_time < day) & (frame.exit_time >= day - pd.Timedelta(days=lookback))]
        grouped = known.groupby("symbol").trade_return
        stats = grouped.agg(["count", "mean", "sum"])
        wins = known[known.trade_return > 0].groupby("symbol").trade_return.sum()
        losses = -known[known.trade_return < 0].groupby("symbol").trade_return.sum()
        stats["pf"] = wins / losses
        stats["pf"] = stats["pf"].replace([np.inf], 99).fillna(0)
        stats["score"] = stats["mean"] * np.log1p(stats["count"])
        qualified = stats[(stats["count"] >= minimum) & (stats["pf"] >= pf_min) & (stats["sum"] > 0)]
        selected = list(qualified.nlargest(selected_assets, "score").index)
        today = frame[(frame.entry_time >= day) & (frame.entry_time < day + pd.Timedelta(days=1)) & frame.symbol.isin(selected)]
        for _, row in today.sort_values("exit_time").iterrows():
            equity *= max(0.0, 1.0 + float(row.trade_return) / max(1, selected_assets))
            peak = max(peak, equity)
            drawdown = min(drawdown, equity / peak - 1)
            trades += 1
        choices.append(",".join(selected))
    return {"return_pct": (equity / 100000 - 1) * 100, "trades": trades,
            "max_drawdown_pct": drawdown * 100, "selection_days": sum(bool(x) for x in choices)}


def main():
    frame = prepare()
    split = frame.entry_time.min() + (frame.entry_time.max() - frame.entry_time.min()) * .60
    development = frame[frame.entry_time < split]
    validation = frame[frame.entry_time >= split]
    rows = []
    for params in itertools.product((7, 14, 21, 30, 45, 60), (5, 10, 20, 30), (1.0, 1.1, 1.2, 1.3), (1, 2, 3)):
        result = simulate(development, *params)
        rows.append({"lookback": params[0], "minimum": params[1], "pf_min": params[2], "selected_assets": params[3], **result})
    grid = pd.DataFrame(rows).sort_values(["return_pct", "trades"], ascending=[False, True])
    chosen = grid.iloc[0]
    params = (int(chosen.lookback), int(chosen.minimum), float(chosen.pf_min), int(chosen.selected_assets))
    result = {"chosen_on_development": chosen.to_dict(), "validation": simulate(frame, *params, start=split),
              "full_causal": simulate(frame, *params)}
    Path("reports").mkdir(exist_ok=True)
    grid.to_csv("reports/v5_causal_persistent_selector_grid.csv", index=False)
    Path("reports/v5_causal_persistent_selector.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
