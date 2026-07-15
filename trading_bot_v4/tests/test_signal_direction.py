import unittest

from trading_bot_v4.utils.signal_direction import binary_upside_direction


class SignalDirectionTests(unittest.TestCase):
    def test_confident_upside_is_long(self):
        self.assertEqual(binary_upside_direction(0.59, 0.58), "LONG")

    def test_low_upside_probability_is_not_misread_as_short(self):
        self.assertEqual(binary_upside_direction(0.20, 0.58), "HOLD")

    def test_threshold_requires_strictly_greater_probability(self):
        self.assertEqual(binary_upside_direction(0.58, 0.58), "HOLD")


if __name__ == "__main__":
    unittest.main()
