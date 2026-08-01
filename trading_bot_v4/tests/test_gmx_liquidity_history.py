import unittest

from trading_bot_v4.features.gmx_liquidity_history import aggregate_liquidity_history


class GmxLiquidityHistoryTests(unittest.TestCase):
    def test_aggregates_glv_allocations_by_market(self):
        payload = {"snapshots": [
            {"timestamp": 1785351600, "marketAddress": "0xA", "glvAddress": "0x1",
             "longLiquidityUsd": str(int(100 * 1e30)), "shortLiquidityUsd": str(int(50 * 1e30))},
            {"timestamp": 1785351600, "marketAddress": "0xA", "glvAddress": "0x2",
             "longLiquidityUsd": str(int(200 * 1e30)), "shortLiquidityUsd": str(int(50 * 1e30))},
        ]}
        row = aggregate_liquidity_history(payload, {"0xa": "BTC"}).iloc[0]
        self.assertAlmostEqual(row["jit_liquidity_long_usd"], 300)
        self.assertAlmostEqual(row["jit_liquidity_short_usd"], 100)
        self.assertAlmostEqual(row["jit_liquidity_skew"], .5)


if __name__ == "__main__":
    unittest.main()
