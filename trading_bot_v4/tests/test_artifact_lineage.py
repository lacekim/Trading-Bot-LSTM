import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from trading_bot_v4.utils import artifact_lineage
from trading_bot_v4.utils.artifact_lineage import write_model_manifest


class ArtifactLineageTests(unittest.TestCase):
    def test_write_model_manifest_prunes_old_versions_beyond_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.bin"
            model_path.write_bytes(b"weights")
            version_dir = Path(tmp) / "model_versions"

            # One manifest filename per wall-clock second means writes within the
            # same second would collide; fake strictly-increasing timestamps so
            # pruning order is deterministic regardless of how fast this runs.
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            fake_now_values = [base + timedelta(seconds=i) for i in range(5)]

            with patch.object(artifact_lineage, "MODEL_VERSION_DIR", version_dir), \
                 patch.object(artifact_lineage, "MODEL_VERSION_RETENTION_COUNT", 3), \
                 patch.object(artifact_lineage, "datetime") as mock_datetime:
                mock_datetime.now.side_effect = fake_now_values
                mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
                for index in range(5):
                    model_path.write_bytes(f"weights-{index}".encode())
                    write_model_manifest([model_path], f"cycle-{index}")

                remaining = sorted(version_dir.glob("*.json"))
                self.assertEqual(len(remaining), 3)
                # The newest manifests survive; the oldest two were pruned.
                contents = [entry.read_text() for entry in remaining]
                self.assertTrue(any('"cycle-4"' in text for text in contents))
                self.assertFalse(any('"cycle-0"' in text for text in contents))


if __name__ == "__main__":
    unittest.main()
