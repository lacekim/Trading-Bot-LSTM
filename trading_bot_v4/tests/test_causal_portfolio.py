import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading_bot_v4.backtesting.causal_portfolio import run_causal_portfolio


class CausalPortfolioTests(unittest.TestCase):
    def test_selection_never_uses_same_day_outcome(self):
        rows = []
        capital = 100.0
        for day in range(4):
            profit = 10.0 if day < 3 else -10.0
            capital += profit
            rows.append({
                "timestamp": f"2026-01-0{day + 1} 12:00:00", "symbol": "A",
                "profit": profit, "capital": capital,
            })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trades.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            result = run_causal_portfolio(
                path, lookback_days=3, minimum_observations=2,
                minimum_profit_factor=1.0, risk_multiplier=1.0,
                include_costs=False,
                report_path=Path(tmp) / "report.json",
                equity_path=Path(tmp) / "equity.csv",
            )
        # No trade can be selected until day three because only completed prior
        # outcomes count toward the two-observation minimum.
        self.assertEqual(result["trades"], 2)


if __name__ == "__main__":
    unittest.main()
