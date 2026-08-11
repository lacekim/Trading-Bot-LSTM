import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.execution.paper_model_performance import _build_symbol_performance


def _make_ohlc(periods: int = 60, start_price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="1h")
    rng = np.random.default_rng(7)
    closes = start_price + np.cumsum(rng.normal(0, 1.2, size=periods))
    highs = closes + np.abs(rng.normal(0.6, 0.3, size=periods))
    lows = closes - np.abs(rng.normal(0.6, 0.3, size=periods))
    opens = closes + rng.normal(0, 0.2, size=periods)
    frame = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=index)
    frame.index.name = "timestamp"
    return frame


class PaperModelPerformanceDebugParityTests(unittest.TestCase):
    """Regression coverage for the debug/non-debug simulator mismatch bug: --debug
    used to run trades through a different (legacy) engine than the main report,
    so the two would silently diverge for the same symbol. _build_symbol_performance
    must report identical metrics regardless of the debug flag."""

    @patch("trading_bot_v4.execution.paper_model_performance.Config.PAPER_MIN_BARS_BETWEEN_TRADES", 0)
    def test_debug_and_non_debug_report_identical_metrics_for_same_symbol(self):
        ohlc = _make_ohlc()
        timestamps = pd.Series(ohlc.index).reset_index(drop=True)
        directions = (["LONG", "HOLD", "SHORT", "HOLD"] * ((len(timestamps) // 4) + 1))[: len(timestamps)]

        original = pd.DataFrame({
            "timestamp": timestamps,
            "symbol": "TEST",
            "timeframe": "1h",
            "original_probability": 0.8,
            "original_direction": directions,
        })
        symbol_smc = pd.DataFrame({
            "timestamp": timestamps,
            "symbol": "TEST",
            "timeframe": "1h",
            "model_probability": 0.8,
            "model_direction": directions,
            "is_trade_candidate": True,
        })
        comparison_row = pd.Series({
            "aggressiveness": "same candidate count",
            "original_candidates": 5,
            "smc_candidates": 5,
        })

        def fake_atr(self, frame, period):
            return pd.Series(1.0, index=frame.index)

        with patch(
            "trading_bot_v4.execution.paper_model_performance._predict_original_model_signals",
            return_value=original,
        ), patch(
            "trading_bot_v4.execution.paper_model_performance.load_gmx_ohlc",
            return_value=ohlc,
        ), patch.object(V4DataHandler, "calculate_atr", fake_atr):
            debug_row = _build_symbol_performance(
                symbol="TEST", timeframe="1h", original_model=None, original_scaler=None,
                smc_signals=symbol_smc, comparison_row=comparison_row, starting_capital=10000.0,
                debug=True,
            )
            live_row = _build_symbol_performance(
                symbol="TEST", timeframe="1h", original_model=None, original_scaler=None,
                smc_signals=symbol_smc, comparison_row=comparison_row, starting_capital=10000.0,
                debug=False,
            )

        self.assertIsNotNone(debug_row)
        self.assertIsNotNone(live_row)
        # At least one side must have actually traded, otherwise this test would
        # trivially "pass" by comparing two no-op runs.
        self.assertGreater(debug_row["original_trade_count"] + debug_row["smc_trade_count"], 0)

        for key in (
            "original_return_pct", "smc_return_pct",
            "original_max_drawdown_pct", "smc_max_drawdown_pct",
            "original_profit_factor", "smc_profit_factor",
            "original_win_rate_pct", "smc_win_rate_pct",
            "original_trade_count", "smc_trade_count",
        ):
            self.assertAlmostEqual(debug_row[key], live_row[key], places=6, msg=key)


if __name__ == "__main__":
    unittest.main()
