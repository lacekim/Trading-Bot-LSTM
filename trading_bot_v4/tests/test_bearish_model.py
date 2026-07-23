import unittest

import pandas as pd

from trading_bot_v4.execution.bearish_model_paper import combine_directional_signals
from trading_bot_v4.ml.bearish_trainer import _forward_compounded_return, build_bearish_target
from trading_bot_v4.research.scheduler import _active_directional_signals


class BearishModelTests(unittest.TestCase):
    def test_target_only_marks_actual_downside_move(self):
        target = build_bearish_target(pd.Series([-0.02, -0.01, 0.00, 0.02]), threshold=0.01)
        self.assertEqual(target.tolist(), [1, 0, 0, 0])

    def test_combiner_selects_stronger_calibrated_direction(self):
        common = {"timestamp": "2026-01-01T00:00:00Z", "symbol": "BTC", "timeframe": "1h", "price": 100}
        long = pd.DataFrame([{**common, "model_direction": "LONG", "model_probability": .76, "threshold": .70}])
        short = pd.DataFrame([{**common, "model_direction": "SHORT", "model_probability": .69, "threshold": .60}])
        result = combine_directional_signals(long, short)
        self.assertEqual(result.iloc[0]["model_direction"], "SHORT")

    def test_combiner_normalizes_mixed_timestamp_types(self):
        long = pd.DataFrame([{
            "timestamp": "2026-01-01 00:00:00", "symbol": "BTC", "timeframe": "1h",
            "model_direction": "HOLD", "model_probability": .2, "threshold": .7,
        }])
        short = pd.DataFrame([{
            "timestamp": pd.Timestamp("2026-01-01T01:00:00Z"), "symbol": "BTC", "timeframe": "1h",
            "model_direction": "SHORT", "model_probability": .8, "threshold": .7,
        }])
        result = combine_directional_signals(long, short)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["timestamp"]))
        self.assertEqual(result.sort_values("timestamp").iloc[-1]["model_direction"], "SHORT")

    def test_forward_return_does_not_cross_symbol_boundary(self):
        frame = pd.DataFrame({"symbol": ["A", "A", "B", "B"], "returns": [0.0, -0.1, 0.5, -0.2]})
        result = _forward_compounded_return(frame, 1)
        self.assertAlmostEqual(result.iloc[0], -0.1)
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertAlmostEqual(result.iloc[2], -0.2)

    def test_directional_activation_uses_separate_long_and_short_universes(self):
        signals = pd.DataFrame([
            {"symbol": "LONG_OK", "model_direction": "LONG"},
            {"symbol": "SHORT_OK", "model_direction": "SHORT"},
            {"symbol": "SHORT_OK", "model_direction": "LONG"},
            {"symbol": "OTHER", "model_direction": "SHORT"},
        ])
        active = _active_directional_signals(signals, ["LONG_OK"], {"SHORT_OK"})
        self.assertEqual(active[["symbol", "model_direction"]].values.tolist(), [
            ["LONG_OK", "LONG"], ["SHORT_OK", "SHORT"]
        ])


if __name__ == "__main__":
    unittest.main()
