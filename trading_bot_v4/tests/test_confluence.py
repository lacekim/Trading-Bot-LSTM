import unittest

import pandas as pd

from trading_bot_v4.signals.confluence import combine_signal_frames
from trading_bot_v4.signals.sentiment_stub import load_sentiment_signal_frame


def _frame(directions: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(directions), freq="h", tz="UTC"),
        "model_direction": directions,
    })


class ConfluenceTests(unittest.TestCase):
    def test_no_confirming_signal_passes_through_unchanged(self):
        price = _frame(["LONG", "HOLD", "SHORT"])
        result = combine_signal_frames(price, None)
        pd.testing.assert_series_equal(result["model_direction"], price["model_direction"])

    def test_agreement_keeps_the_signal(self):
        price = _frame(["LONG", "SHORT", "HOLD"])
        confirming = _frame(["LONG", "SHORT", "LONG"])
        result = combine_signal_frames(price, confirming)
        self.assertEqual(result["model_direction"].tolist(), ["LONG", "SHORT", "HOLD"])

    def test_disagreement_downgrades_to_hold(self):
        price = _frame(["LONG", "SHORT"])
        confirming = _frame(["SHORT", "SHORT"])
        result = combine_signal_frames(price, confirming)
        self.assertEqual(result["model_direction"].tolist(), ["HOLD", "SHORT"])

    def test_missing_confirming_row_is_treated_as_disagreement(self):
        price = _frame(["LONG", "LONG"])
        confirming = price.iloc[:1].copy()  # only one of two timestamps has a confirming reading
        result = combine_signal_frames(price, confirming)
        self.assertEqual(result["model_direction"].tolist(), ["LONG", "HOLD"])

    def test_sentiment_stub_refuses_to_run_rather_than_fake_it(self):
        with self.assertRaises(NotImplementedError):
            load_sentiment_signal_frame("BTC", "1h", "2024-01-01")


if __name__ == "__main__":
    unittest.main()
