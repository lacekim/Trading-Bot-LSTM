import unittest

from trading_bot_v4.backtesting.baseline_contract import ORIGINAL_BASELINE, active_execution_parity_audit


class BaselineContractTests(unittest.TestCase):
    def test_original_one_hour_contract_is_locked(self):
        self.assertEqual(ORIGINAL_BASELINE.timeframe, "1h")
        self.assertEqual(ORIGINAL_BASELINE.sequence_length, 36)
        self.assertEqual(ORIGINAL_BASELINE.signal_threshold, 0.58)
        self.assertEqual(ORIGINAL_BASELINE.atr_stop_multiplier, 0.75)
        self.assertEqual(ORIGINAL_BASELINE.atr_target_multiplier, 1.5)
        self.assertEqual(ORIGINAL_BASELINE.maximum_hold_candles, 1)

    def test_paper_overlay_is_not_falsely_reported_as_exact_parity(self):
        audit = active_execution_parity_audit()
        self.assertFalse(audit["configuration_parity"])
        self.assertNotIn("atr_stop_multiplier", audit["deviations"])
        self.assertNotIn("cooldown_candles", audit["deviations"])
        self.assertIn("fee_bps", audit["deviations"])


if __name__ == "__main__":
    unittest.main()
