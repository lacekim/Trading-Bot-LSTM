"""Compare simulate_production_symbol vs simulate_production_symbol_freqtrade_style
using the real combined long+short production signal (not the long-only file used
in the earlier comparison), across the full liquidity-qualified universe.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trading_bot_v4.backtesting.production_backtest import _cached_directional_signals, simulate_production_symbol
from trading_bot_v4.backtesting.production_backtest_freqtrade_style import simulate_production_symbol_freqtrade_style
from trading_bot_v4.execution.bearish_model_paper import load_bearish_calibration
from trading_bot_v4.execution.paper_model_performance import _production_metrics
from trading_bot_v4.ml.bearish_trainer import BEARISH_MODEL_PATH, BEARISH_SCALER_PATH
from trading_bot_v4.utils.model_cache import ModelScalerCache

STARTING_CAPITAL = 100_000.0
OUTPUT_PATH = Path("logs/v4_freqtrade_style_combined_signal_comparison.csv")


def main() -> None:
    long_model, long_scaler = ModelScalerCache().load()
    bearish_model, bearish_scaler = ModelScalerCache(BEARISH_MODEL_PATH, BEARISH_SCALER_PATH).load()
    calibration = load_bearish_calibration()
    bearish_threshold = float(calibration["threshold"])

    audit = pd.read_csv("logs/v4_go_asset_selection_audit.csv")
    symbols = sorted(audit.loc[audit["readiness_decision"] == "GO", "symbol"].astype(str).str.upper().tolist())

    rows = []
    started = time.monotonic()
    for i, symbol in enumerate(symbols, 1):
        try:
            combined = _cached_directional_signals(
                symbol, "1h", long_model, long_scaler, bearish_model, bearish_scaler,
                float(calibration.get("promoted_symbols", {}).get(symbol, bearish_threshold)),
            )
            frame = combined[["timestamp", "model_direction", "open", "high", "low", "price", "atr"]].copy()
            frame["symbol"] = symbol
            frame = frame.dropna(subset=["open", "high", "low", "price", "atr"])

            summary_old, trades_old = simulate_production_symbol(frame, symbol, STARTING_CAPITAL)
            metrics_old = _production_metrics(summary_old, trades_old)
            summary_new, trades_new = simulate_production_symbol_freqtrade_style(frame, symbol, STARTING_CAPITAL)

            rows.append({
                "symbol": symbol,
                "current_return_pct": metrics_old["return_pct"], "current_dd": metrics_old["max_drawdown_pct"],
                "current_trades": metrics_old["trade_count"], "current_pf": metrics_old["profit_factor"],
                "freq_return_pct": summary_new["return_pct"], "freq_dd": summary_new["max_drawdown_pct"],
                "freq_trades": summary_new["trades"], "freq_pf": summary_new["profit_factor"],
                "long_signals": int((frame["model_direction"] == "LONG").sum()),
                "short_signals": int((frame["model_direction"] == "SHORT").sum()),
            })
        except Exception as exc:
            rows.append({"symbol": symbol, "error": str(exc)})
        if i % 10 == 0 or i == len(symbols):
            elapsed = time.monotonic() - started
            print(f"[{i}/{len(symbols)}] elapsed={elapsed/60:.1f}min", flush=True)
            pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"Done in {(time.monotonic()-started)/60:.1f} min. Wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
