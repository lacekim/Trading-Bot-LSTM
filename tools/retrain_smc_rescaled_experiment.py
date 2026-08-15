"""One-off experiment driver: retrain the SMC model with the binary-flag
StandardScaler fix (SMC_BINARY_FEATURE_COLUMNS left as raw 0/1), saving to
experimental paths instead of the live models/lstm_smc_model.h5 + scaler_smc.pkl
so this doesn't disturb anything currently reading those files.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot_v4.ml import smc_trainer

EXPERIMENT_MODEL_PATH = Path("models/experiments/lstm_smc_model_rescaled.h5")
EXPERIMENT_SCALER_PATH = Path("models/experiments/scaler_smc_rescaled.pkl")


def main() -> None:
    smc_trainer.SMC_MODEL_PATH = EXPERIMENT_MODEL_PATH
    smc_trainer.SMC_SCALER_PATH = EXPERIMENT_SCALER_PATH

    started = time.monotonic()
    print(f"Starting SMC rescaled-experiment training at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    result = smc_trainer.train_smc_model(timeframe="1h")
    elapsed = time.monotonic() - started

    print(f"Done in {elapsed / 60.0:.1f} minutes", flush=True)
    print(result, flush=True)
    with open("models/experiments/rescaled_training_result.pkl", "wb") as handle:
        pickle.dump(result, handle)


if __name__ == "__main__":
    main()
