import unittest
from unittest.mock import patch

from trading_bot_v4.execution.persistent_policy import direction_for_probability, policy_for


class PersistentPolicyTests(unittest.TestCase):
    def test_vvv_candidate_contract(self):
        with patch.dict("os.environ", {"PERSISTENT_POLICY_SYMBOLS": "VVV"}, clear=False):
            policy = policy_for("VVV")
            self.assertTrue(policy.enabled)
            self.assertEqual(policy.threshold, .60)
            self.assertEqual(policy.stop_atr, .50)
            self.assertEqual(policy.target_atr, 20.0)
            self.assertEqual(policy.max_hold_candles, 240)
            self.assertEqual(direction_for_probability(.61, "VVV"), "LONG")
            self.assertEqual(direction_for_probability(.39, "VVV"), "SHORT")
            self.assertEqual(direction_for_probability(.50, "VVV"), "HOLD")

    def test_other_symbols_retain_upside_only_policy(self):
        with patch.dict("os.environ", {"PERSISTENT_POLICY_SYMBOLS": "VVV"}, clear=False):
            self.assertEqual(direction_for_probability(.20, "BTC"), "HOLD")


if __name__ == "__main__":
    unittest.main()
