import unittest

import pandas as pd

from trading_bot_v4.utils.point_in_time import assert_no_lookahead


class PointInTimeGuardTests(unittest.TestCase):
    def test_passes_when_all_rows_on_or_before_cutoff(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "value": [1, 2],
        })
        assert_no_lookahead(frame, "timestamp", "2024-01-02T00:00:00Z", "test-source")

    def test_raises_when_any_row_is_after_cutoff(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-05"], utc=True),
            "value": [1, 2],
        })
        with self.assertRaises(ValueError):
            assert_no_lookahead(frame, "timestamp", "2024-01-02T00:00:00Z", "test-source")

    def test_empty_frame_never_raises(self):
        frame = pd.DataFrame({"timestamp": pd.to_datetime([], utc=True), "value": []})
        assert_no_lookahead(frame, "timestamp", "2024-01-02T00:00:00Z", "test-source")


if __name__ == "__main__":
    unittest.main()
