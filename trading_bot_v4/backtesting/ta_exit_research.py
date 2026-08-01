"""Non-overlapping temporal diagnostics for the confirmed TA entry structure."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_bot_v4.backtesting.production_backtest import SIGNAL_CACHE_DIR, simulate_production_symbol
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.utils.macd_confirmation import macd_entry_confirmation


REPORT_PATH = Path("reports/v5_ta_exit_temporal_diagnostic.json")
HOLDS = (4, 8, 12, 24)
WINDOWS = ((.40, .60), (.60, .80), (.80, 1.00))


def _latest_signal_files() -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in SIGNAL_CACHE_DIR.glob("*_1h_*.pkl"):
        symbol = path.name.split("_1h_", 1)[0]
        if symbol not in selected or path.stat().st_mtime_ns > selected[symbol].stat().st_mtime_ns:
            selected[symbol] = path
    return selected


def run_ta_exit_temporal_diagnostic(ma_slope_lag: int | None = None) -> dict:
    frames = {symbol: pd.read_pickle(path) for symbol, path in _latest_signal_files().items()}
    if not frames:
        raise RuntimeError("No cached production directional signals are available")
    all_times = pd.concat([pd.to_datetime(frame["timestamp"], utc=True) for frame in frames.values()])
    times = pd.Series(all_times.unique()).sort_values().reset_index(drop=True)
    results = []
    original_hold = Config.PAPER_MAX_HOLD_CANDLES
    try:
        for hold in HOLDS:
            Config.PAPER_MAX_HOLD_CANDLES = hold
            for number, (start_fraction, end_fraction) in enumerate(WINDOWS, 1):
                start = times.iloc[min(int(len(times) * start_fraction), len(times) - 1)]
                end = times.iloc[min(int(len(times) * end_fraction), len(times) - 1)]
                returns, trades = [], []
                for symbol, source in frames.items():
                    frame = source.copy()
                    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
                    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].copy()
                    if frame.empty:
                        continue
                    frame["_entry_eligible"] = macd_entry_confirmation(frame)
                    if ma_slope_lag:
                        close = pd.to_numeric(frame["close"], errors="coerce")
                        ma200 = close.rolling(200).mean()
                        slope = ma200 / ma200.shift(ma_slope_lag) - 1.0
                        direction = frame["model_direction"].astype(str).str.upper()
                        slope_confirmed = (direction.eq("LONG") & slope.gt(0)) | (
                            direction.eq("SHORT") & slope.lt(0)
                        )
                        frame["_entry_eligible"] &= slope_confirmed
                    frame.loc[frame["timestamp"].idxmax(), "_entry_eligible"] = False
                    summary, asset_trades = simulate_production_symbol(frame, symbol, 100_000.0, True)
                    returns.append(float(summary["return_pct"]))
                    if not asset_trades.empty:
                        trades.append(asset_trades)
                combined = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
                wins = combined.loc[combined["net_pnl"] > 0, "net_pnl"].sum() if not combined.empty else 0.0
                losses = abs(combined.loc[combined["net_pnl"] < 0, "net_pnl"].sum()) if not combined.empty else 0.0
                results.append({
                    "hold_candles": hold, "window": number, "start": str(start), "end": str(end),
                    "assets": len(returns), "trades": int(len(combined)),
                    "equal_weight_return_pct": float(pd.Series(returns).mean()) if returns else 0.0,
                    "profit_factor": float(wins / losses) if losses else (float("inf") if wins else 0.0),
                    "win_rate_pct": float((combined["net_pnl"] > 0).mean() * 100) if not combined.empty else 0.0,
                })
    finally:
        Config.PAPER_MAX_HOLD_CANDLES = original_hold
    report = {
        "policy": "identical cached model predictions; MACD crossover + 200MA entry; three non-overlapping temporal windows; configured costs",
        "ma200_slope_lag_hours": ma_slope_lag,
        "limitation": "temporal diagnostic only: deployed model artifacts may have been trained on observations inside these windows",
        "results": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_ta_exit_temporal_diagnostic()
