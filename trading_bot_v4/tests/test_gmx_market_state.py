import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from trading_bot_v4.features.gmx_market_state import (
    fetch_historical_rates,
    normalize_markets_info,
    persist_market_state,
)


def scaled(value):
    return str(int(value * 1e30))


class GmxMarketStateTests(unittest.TestCase):
    def test_normalizes_and_aggregates_multiple_pools_without_lookahead(self):
        payload = {"markets": [
            {"isListed": True, "name": "BTC/USD [pool-a]",
             "openInterestLong": scaled(100), "openInterestShort": scaled(50),
             "availableLiquidityLong": scaled(500), "availableLiquidityShort": scaled(400),
             "fundingRateLong": scaled(.01), "fundingRateShort": scaled(.02),
             "borrowingRateLong": scaled(.03), "borrowingRateShort": scaled(.04),
             "netRateLong": scaled(.05), "netRateShort": scaled(.06)},
            {"isListed": True, "name": "BTC/USD [pool-b]",
             "openInterestLong": scaled(300), "openInterestShort": scaled(150),
             "availableLiquidityLong": scaled(700), "availableLiquidityShort": scaled(600),
             "fundingRateLong": scaled(.03), "fundingRateShort": scaled(.04),
             "borrowingRateLong": scaled(.05), "borrowingRateShort": scaled(.06),
             "netRateLong": scaled(.07), "netRateShort": scaled(.08)},
        ]}
        observed = pd.Timestamp("2026-07-29T12:00:00Z")
        row = normalize_markets_info(payload, observed).iloc[0]
        self.assertEqual(row["timestamp"], observed)
        self.assertEqual(row["symbol"], "BTC")
        self.assertAlmostEqual(row["open_interest_total_usd"], 600)
        self.assertAlmostEqual(row["open_interest_skew"], 1 / 3)
        self.assertAlmostEqual(row["available_liquidity_long_usd"], 1200)
        self.assertAlmostEqual(row["funding_rate_long"], .025)
        self.assertEqual(row["market_pool_count"], 2)

    def test_persistence_replaces_same_hour_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.csv"
            first = pd.DataFrame([{"timestamp": "2026-07-29T12:00:00Z", "symbol": "BTC", "open_interest_total_usd": 1}])
            persist_market_state(first, path)
            second = pd.DataFrame([{"timestamp": "2026-07-29T12:00:00Z", "symbol": "BTC", "open_interest_total_usd": 2}])
            result = persist_market_state(second, path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["open_interest_total_usd"], 2)

    @patch("trading_bot_v4.features.gmx_market_state.requests.get")
    def test_historical_rates_uses_peer_fallback(self, get):
        failed = Mock()
        failed.raise_for_status.side_effect = requests.HTTPError("primary unavailable")
        succeeded = Mock()
        succeeded.raise_for_status.return_value = None
        succeeded.json.return_value = [{"marketAddress": "0x1"}]
        get.side_effect = [failed, succeeded]
        self.assertEqual(fetch_historical_rates(), [{"marketAddress": "0x1"}])
        self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
