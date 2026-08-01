import unittest

import pandas as pd

from trading_bot_v4.ml.cost_aware_long_trainer import FEATURES, build_cost_aware_long_target


class CostAwareLongTrainerTests(unittest.TestCase):
    def test_target_requires_return_above_round_trip_cost(self):
        result = build_cost_aware_long_target(pd.Series([0.001, 0.01, 0.011]), cost_rate=0.0024)
        self.assertEqual(result.tolist(), [0, 0, 1])

    def test_noncausal_confirmed_swing_features_are_excluded(self):
        self.assertNotIn("swing_high", FEATURES)
        self.assertNotIn("swing_low", FEATURES)


if __name__ == "__main__":
    unittest.main()
