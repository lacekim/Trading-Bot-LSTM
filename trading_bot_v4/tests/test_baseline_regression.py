import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from trading_bot_v4.backtesting.baseline_regression import run_baseline_regression


class BaselineRegressionTests(unittest.TestCase):
    def test_compares_all_asset_reports_without_portfolio_reframing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / "old.csv", root / "new.csv"
            pd.DataFrame({"symbol": ["A", "B"], "return_pct": [20.0, -10.0]}).to_csv(old, index=False)
            pd.DataFrame({"symbol": ["A", "B"], "return_pct": [10.0, -20.0]}).to_csv(new, index=False)
            with patch("trading_bot_v4.backtesting.baseline_regression.OUTPUT_CSV", root / "out.csv"), \
                 patch("trading_bot_v4.backtesting.baseline_regression.OUTPUT_JSON", root / "out.json"):
                result = run_baseline_regression(SimpleNamespace(reference_report=old, current_report=new))
            self.assertEqual(result["reference"]["mean_return_pct"], 5.0)
            self.assertEqual(result["current"]["mean_return_pct"], -5.0)
            self.assertEqual(result["mean_change_percentage_points"], -10.0)
            self.assertIn("not a shared-capital portfolio", result["report_format"])


if __name__ == "__main__":
    unittest.main()
