import unittest

from trading_bot_v4.features.gmx_trade_flow import aggregate_trade_flow


class GmxTradeFlowTests(unittest.TestCase):
    def test_aggregates_executed_directional_flow_and_ignores_created_orders(self):
        trades = [
            {"eventName": "OrderExecuted", "timestamp": 1785354034, "marketAddress": "0xABC",
             "orderType": 2, "isLong": True, "sizeDeltaUsd": str(int(100 * 1e30))},
            {"eventName": "OrderExecuted", "timestamp": 1785354040, "marketAddress": "0xABC",
             "orderType": 3, "isLong": False, "sizeDeltaUsd": str(int(25 * 1e30))},
            {"eventName": "OrderExecuted", "timestamp": 1785354050, "marketAddress": "0xABC",
             "orderType": 4, "isLong": True, "sizeDeltaUsd": str(int(40 * 1e30))},
            {"eventName": "OrderCreated", "timestamp": 1785354050, "marketAddress": "0xABC",
             "orderType": 2, "isLong": True, "sizeDeltaUsd": str(int(999 * 1e30))},
        ]
        row = aggregate_trade_flow(trades, {"0xabc": "BTC"}).iloc[0]
        self.assertEqual(row["symbol"], "BTC")
        self.assertAlmostEqual(row["long_increase_usd"], 100)
        self.assertAlmostEqual(row["short_increase_usd"], 25)
        self.assertAlmostEqual(row["long_decrease_usd"], 40)
        self.assertAlmostEqual(row["increase_flow_skew"], .6)
        self.assertAlmostEqual(row["net_position_flow_usd"], 35)
        self.assertEqual(row["long_increase_count"], 1)


if __name__ == "__main__":
    unittest.main()
