import unittest

import pandas as pd

from trading_bot_v4.ml.multi_horizon_directional import add_horizon_return


class MultiHorizonDirectionalTests(unittest.TestCase):
    def test_horizon_return_uses_current_and_future_one_step_returns(self):
        frame = pd.DataFrame({
            "symbol": ["A"] * 4,
            "future_return": [0.10, 0.20, -0.10, 0.05],
        })
        result = add_horizon_return(frame, 2)
        self.assertAlmostEqual(result.iloc[0]["horizon_return"], 1.10 * 1.20 - 1.0)
        self.assertAlmostEqual(result.iloc[1]["horizon_return"], 1.20 * 0.90 - 1.0)
        self.assertEqual(len(result), 3)

    def test_horizon_return_does_not_cross_symbols(self):
        frame = pd.DataFrame({
            "symbol": ["A", "A", "B", "B"],
            "future_return": [0.10, 0.20, 0.50, 0.50],
        })
        result = add_horizon_return(frame, 2)
        self.assertEqual(result["symbol"].tolist(), ["A", "B"])
        self.assertAlmostEqual(result.iloc[0]["horizon_return"], 0.32)
        self.assertAlmostEqual(result.iloc[1]["horizon_return"], 1.25)


if __name__ == "__main__":
    unittest.main()
