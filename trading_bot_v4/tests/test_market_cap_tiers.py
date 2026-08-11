import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading_bot_v4.risk.market_cap_tiers import classify_market_cap, evaluate_asset_risk


class MarketCapTierTests(unittest.TestCase):
    def test_tiers_scale_risk_and_block_micro_caps(self):
        self.assertEqual(classify_market_cap(20_000_000_000).tier, "PREMIUM")
        self.assertEqual(classify_market_cap(2_000_000_000).risk_pct, .75)
        self.assertEqual(classify_market_cap(500_000_000).max_position_pct, 10)
        self.assertEqual(classify_market_cap(50_000_000).risk_pct, .25)
        self.assertFalse(classify_market_cap(10_000_000).allowed)

    def test_directional_gmx_liquidity_is_mandatory(self):
        # Must stay within MAX_SNAPSHOT_AGE_HOURS of "now" -- a hardcoded past
        # date is a time bomb that fails once real time passes the freshness
        # window, independent of the risk logic under test.
        fresh = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)
        with tempfile.TemporaryDirectory() as tmp:
            caps, gmx = Path(tmp) / "caps.csv", Path(tmp) / "gmx.csv"
            pd.DataFrame([{"observed_at": fresh.isoformat(), "symbol": "TEST",
                           "market_cap_usd": 500_000_000}]).to_csv(caps, index=False)
            pd.DataFrame([{"timestamp": fresh.isoformat(), "symbol": "TEST",
                           "available_liquidity_long_usd": 2_000_000,
                           "available_liquidity_short_usd": 10_000,
                           "open_interest_total_usd": 1_000_000}]).to_csv(gmx, index=False)
            self.assertTrue(evaluate_asset_risk("TEST", "LONG", caps, gmx).allowed)
            self.assertFalse(evaluate_asset_risk("TEST", "SHORT", caps, gmx).allowed)

    def test_deprecated_contract_is_always_blocked(self):
        self.assertFalse(evaluate_asset_risk("OM", "LONG").allowed)


if __name__ == "__main__":
    unittest.main()
