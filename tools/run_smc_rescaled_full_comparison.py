"""One-off experiment driver: run the rescaled SMC model (models/experiments/) across
the full liquidity-qualified symbol universe and simulate trades with the same
production engine used everywhere else, so results are directly comparable to
logs/v4_paper_model_performance_constrained.csv's original_*/smc_* columns.
"""
from __future__ import annotations

import pickle
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tensorflow.keras.models import load_model

from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol
from trading_bot_v4.execution.paper_model_performance import (
    _build_production_execution_frame,
    _production_metrics,
)
from trading_bot_v4.execution.smc_model_paper import _predict_smc_model_signals

EXPERIMENT_MODEL_PATH = Path("models/experiments/lstm_smc_model_rescaled.h5")
EXPERIMENT_SCALER_PATH = Path("models/experiments/scaler_smc_rescaled.pkl")
OUTPUT_PATH = Path("logs/v4_smc_rescaled_full_comparison.csv")
STARTING_CAPITAL = 100_000.0
TIMEFRAME = "1h"


def main() -> None:
    audit = pd.read_csv("logs/v4_go_asset_selection_audit.csv")
    symbols = sorted(audit.loc[audit["readiness_decision"] == "GO", "symbol"].astype(str).str.upper().tolist())
    print(f"{len(symbols)} liquidity-qualified symbols to evaluate", flush=True)

    model = load_model(EXPERIMENT_MODEL_PATH)
    with open(EXPERIMENT_SCALER_PATH, "rb") as handle:
        scaler = pickle.load(handle)

    rows = []
    started = time.monotonic()
    for index, symbol in enumerate(symbols, start=1):
        try:
            signals = _predict_smc_model_signals(model, scaler, symbol, TIMEFRAME)
            execution = _build_production_execution_frame(
                symbol, TIMEFRAME, signals["timestamp"]
            ).reset_index(drop=True)
            base = signals.reset_index(drop=True).join(execution)
            frame = base[["timestamp", "model_direction", "Open", "High", "Low", "Close", "ATR"]].copy()
            frame = frame.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "price", "ATR": "atr",
            })
            frame["symbol"] = symbol
            frame = frame.dropna(subset=["open", "high", "low", "price", "atr"])

            summary, trades = simulate_production_symbol(frame, symbol, STARTING_CAPITAL)
            metrics = _production_metrics(summary, trades)
            rows.append({"symbol": symbol, "status": "ok", **metrics})
            print(f"[{index}/{len(symbols)}] {symbol}: return={metrics['return_pct']:.2f}% "
                  f"dd={metrics['max_drawdown_pct']:.2f}% trades={metrics['trade_count']}", flush=True)
        except Exception as exc:
            rows.append({"symbol": symbol, "status": f"error: {exc}"})
            print(f"[{index}/{len(symbols)}] {symbol}: SKIPPED ({exc})", flush=True)
            traceback.print_exc()

    elapsed = time.monotonic() - started
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Done in {elapsed / 60.0:.1f} minutes. Wrote {OUTPUT_PATH} ({len(result)} rows).", flush=True)


if __name__ == "__main__":
    main()
