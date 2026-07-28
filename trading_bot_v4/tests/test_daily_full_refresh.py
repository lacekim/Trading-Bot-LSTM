import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_bot_v4.research.scheduler import _run_complete_daily_refresh


class DailyFullRefreshTests(unittest.TestCase):
    def test_every_expensive_stage_runs_before_qualification(self):
        calls = []

        class Handler:
            def refresh_gmx_cache(self, force=False):
                calls.append(("refresh", force))
                return True

        def stage(name, result=None):
            def run(*_args, **_kwargs):
                calls.append(name)
                return result
            return run

        bundle = object()
        with patch("trading_bot_v4.research.scheduler._daily_artifact_paths", return_value=[]), \
             patch("trading_bot_v4.research.scheduler.V4DataHandler", return_value=Handler()), \
             patch("trading_bot_v4.research.scheduler.build_all_assets_smc_training_data", side_effect=stage("build")), \
             patch("trading_bot_v4.research.scheduler.train_v4_model", side_effect=stage("train_original")), \
             patch("trading_bot_v4.research.scheduler.train_smc_model", side_effect=stage("train_smc")), \
             patch("trading_bot_v4.research.scheduler.train_bearish_model", side_effect=stage("train_bearish")), \
             patch("trading_bot_v4.research.scheduler.run_walk_forward_smc_validation", side_effect=stage("walk_forward")), \
             patch("trading_bot_v4.research.scheduler._load_scheduler_models", side_effect=stage("reload", bundle)), \
             patch("trading_bot_v4.research.scheduler._run_daily_research", side_effect=stage("qualification", "daily")):
            result, refreshed = _run_complete_daily_refresh("1h", 10)

        self.assertEqual(result, "daily")
        self.assertIs(refreshed, bundle)
        self.assertEqual(calls, [
            ("refresh", True), "build", "train_original", "train_smc",
            "train_bearish", "walk_forward", "reload", "qualification",
        ])

    def test_failed_training_restores_previous_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old-model.h5"
            new = Path(directory) / "new-model.h5"
            old.write_text("validated", encoding="utf-8")

            class Handler:
                def refresh_gmx_cache(self, force=False):
                    return True

            def corrupt_then_continue(*_args, **_kwargs):
                old.write_text("partial", encoding="utf-8")
                new.write_text("partial", encoding="utf-8")

            with patch("trading_bot_v4.research.scheduler._daily_artifact_paths", return_value=[old, new]), \
                 patch("trading_bot_v4.research.scheduler.V4DataHandler", return_value=Handler()), \
                 patch("trading_bot_v4.research.scheduler.build_all_assets_smc_training_data"), \
                 patch("trading_bot_v4.research.scheduler.train_v4_model", side_effect=corrupt_then_continue), \
                 patch("trading_bot_v4.research.scheduler.train_smc_model", side_effect=RuntimeError("training failed")):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    _run_complete_daily_refresh("1h", 10)

            self.assertEqual(old.read_text(encoding="utf-8"), "validated")
            self.assertFalse(new.exists())


if __name__ == "__main__":
    unittest.main()
