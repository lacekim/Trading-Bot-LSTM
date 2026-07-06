import unittest

from trading_bot_v4.backtesting.comparison_engine import build_trade_metrics, run_v4_compare_original


class ComparisonEngineTests(unittest.TestCase):
    def test_build_trade_metrics_reports_risk_metrics(self):
        trades = [
            {"profit": 100.0},
            {"profit": -50.0},
            {"profit": 120.0},
            {"profit": -40.0},
        ]

        metrics = build_trade_metrics(trades, starting_capital=1000.0, final_capital=1130.0)

        self.assertIn("profit_factor", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("sortino_ratio", metrics)
        self.assertIn("average_trade_return", metrics)
        self.assertIn("expectancy", metrics)
        self.assertIn("calmar_ratio", metrics)
        self.assertGreater(metrics["profit_factor"], 0)
        self.assertGreaterEqual(metrics["average_trade_return"], 0)

    def test_compare_original_entrypoint_is_available(self):
        self.assertTrue(callable(run_v4_compare_original))


if __name__ == "__main__":
    unittest.main()
