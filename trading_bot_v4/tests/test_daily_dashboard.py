import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_bot_v4.backtesting.asset_selection_engine import _add_validated_ranking_scores
from trading_bot_v4.research.daily_research import (
    _apply_validated_oos_gate, _dashboard_decision_views,
    _forward_demotion_statistics, _forward_paper_metrics,
)
from trading_bot_v4.execution.order_manager import OrderManager


class DailyDashboardTests(unittest.TestCase):
    def test_weak_economics_cannot_receive_elite_validated_score(self):
        frame = pd.DataFrame({
            "constrained_smc_return_pct": [-1.0, 8.0],
            "constrained_smc_profit_factor": [0.95, 1.4],
            "constrained_smc_max_drawdown_pct": [-4.0, -4.0],
            "walk_forward_stability": [90.0, 90.0],
            "smc_vs_original_improvement_pct": [10.0, 10.0],
            "constrained_smc_trade_count": [100.0, 100.0],
            "shared_timestamps": [1000.0, 1000.0],
            "liquidity": [1000.0, 1000.0],
            "trend_strength": [1.0, 1.0],
            "cnn_lstm_confidence": [0.8, 0.8],
        })
        ranked = _add_validated_ranking_scores(frame)
        weak = ranked.loc[ranked["constrained_smc_return_pct"].eq(-1.0)].iloc[0]
        self.assertLessEqual(weak["validated_score"], 25.0)

    def test_no_go_universe_allocates_everything_to_cash(self):
        readiness = pd.DataFrame([{
            "symbol": "BTC", "decision": "NO-GO", "failed_conditions": "profit factor failed",
            "return_7d_pct": 1.0, "return_14d_pct": 1.0, "return_30d_pct": 1.0,
            "profit_factor_30d": 0.9, "max_drawdown_30d_pct": -2.0, "trade_count_30d": 60,
        }])
        rankings = pd.DataFrame([{
            "symbol": "BTC", "validated_score": 50.0, "cnn_lstm_confidence": 0.7,
            "smc_score": 60.0, "walk_forward_stability": 70.0,
        }])
        momentum = pd.DataFrame([{"symbol": "BTC", "momentum_score": 2.0, "return_24h_pct": 3.0}])
        status, _, allocation, _ = _dashboard_decision_views(readiness, rankings, momentum)
        self.assertEqual(status.iloc[0]["status"], "WATCHLIST")
        self.assertEqual(allocation.to_dict("records"), [{"asset": "Cash", "allocation_pct": 100.0}])

    def test_forward_dashboard_metrics_read_persistent_paper_account(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "paper.sqlite3"
            manager = OrderManager(path)
            manager.save_state()
            with patch("trading_bot_v4.research.daily_research.Config.PAPER_DB_PATH", path):
                metrics, by_asset = _forward_paper_metrics()
            manager.close()
            self.assertEqual(metrics["status"], "active")
            self.assertEqual(metrics["starting_equity"], 10000.0)
            self.assertEqual(metrics["open_positions"], 0)
            self.assertEqual(metrics["win_rate"], "insufficient data")

    def test_validated_oos_gate_demotes_weak_recent_winner_to_watch(self):
        readiness = pd.DataFrame([{
            "symbol": "PUMP", "decision": "GO", "failed_conditions": "",
            "profit_factor_30d": 1.5,
        }])
        rankings = pd.DataFrame([{
            "symbol": "PUMP", "constrained_smc_return_pct": -2.0,
            "constrained_smc_profit_factor": 0.9,
            "constrained_smc_max_drawdown_pct": -6.0,
            "constrained_smc_trade_count": 200,
            "walk_forward_stability": 80.0,
        }])
        gated = _apply_validated_oos_gate(readiness, rankings)
        self.assertEqual(gated.iloc[0]["decision"], "WATCH")
        self.assertIn("validated profit factor", gated.iloc[0]["failed_conditions"])

    def test_forward_demotion_statistics_are_sample_based(self):
        stats = _forward_demotion_statistics([10.0, -20.0, 5.0])
        self.assertEqual(stats["trades"], 3.0)
        self.assertAlmostEqual(stats["profit_factor"], 0.75)
        self.assertLess(stats["expectancy"], 0.0)


if __name__ == "__main__":
    unittest.main()
