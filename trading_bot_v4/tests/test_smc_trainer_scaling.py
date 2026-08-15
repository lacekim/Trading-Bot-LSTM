import unittest

import numpy as np
import pandas as pd

from trading_bot_v4.core.smc_swings import SMC_BINARY_FEATURE_COLUMNS
from trading_bot_v4.ml.smc_trainer import _split_and_scale_by_symbol


class SmcTrainerScalingTests(unittest.TestCase):
    """Regression coverage for the SMC-flag StandardScaler fix: sparse binary event
    flags (BOS/CHoCH/order blocks/etc.) must stay raw 0/1 rather than being turned
    into multi-std-dev spikes by StandardScaler."""

    def test_binary_smc_flags_are_left_as_raw_zero_one_while_others_are_standardized(self):
        rng = np.random.default_rng(3)
        rows_per_symbol = 250
        binary_column = SMC_BINARY_FEATURE_COLUMNS[0]
        continuous_column = "some_continuous_feature"
        feature_columns = [continuous_column, binary_column]

        frames = []
        for symbol in ("AAA", "BBB"):
            frame = pd.DataFrame({
                "timestamp": pd.date_range("2025-01-01", periods=rows_per_symbol, freq="1h"),
                "symbol": symbol,
                continuous_column: rng.normal(50.0, 10.0, size=rows_per_symbol),
                # Rare ~5% fire rate, matching the real sparsity of BOS/CHoCH/etc.
                binary_column: (rng.random(rows_per_symbol) < 0.05).astype(float),
                "target": rng.integers(0, 2, size=rows_per_symbol),
            })
            frames.append(frame)
        dataset = pd.concat(frames, ignore_index=True)

        arrays, scaler, stats = _split_and_scale_by_symbol(dataset, feature_columns)

        self.assertEqual(set(arrays.keys()), {"AAA", "BBB"})
        binary_index = feature_columns.index(binary_column)
        continuous_index = feature_columns.index(continuous_column)

        for symbol_arrays in arrays.values():
            for key in ("train_features", "validation_features"):
                features = symbol_arrays[key]
                unique_binary_values = set(np.unique(features[:, binary_index]).tolist())
                self.assertTrue(unique_binary_values.issubset({0.0, 1.0}), unique_binary_values)
                # Continuous column was standardized -- should no longer sit near its raw ~50 scale.
                self.assertLess(abs(float(features[:, continuous_index].mean())), 5.0)


if __name__ == "__main__":
    unittest.main()
