import unittest

import pandas as pd

from trading_bot_v4.ml.directional_sequence_trainer import HOLD, LONG, SHORT, protected_direction_labels


class DirectionalSequenceTrainerTests(unittest.TestCase):
    def test_labels_choose_best_net_direction_and_hold_without_edge(self):
        frame = pd.DataFrame({
            "long_protected_return": [0.02, -0.01, 0.001],
            "short_protected_return": [-0.01, 0.03, 0.001],
        })
        labels = protected_direction_labels(frame, minimum_net_edge=0.001)
        self.assertEqual(labels.tolist(), [LONG, SHORT, HOLD])


if __name__ == "__main__":
    unittest.main()
