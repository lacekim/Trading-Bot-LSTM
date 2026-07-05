"""Prediction wrapper around the original model loading and inference logic."""

from __future__ import annotations

from pathlib import Path

from tensorflow.keras.models import load_model

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_predictor")


def predict_with_v4_model(model_path: str | Path | None = None, features=None):
    model_path = Path(model_path or (Config.MODEL_DIR / Config.MODEL_NAME))
    model = load_model(model_path)
    if features is None:
        logger.info("No feature batch provided; returning the loaded model handle.")
        return model
    return model.predict(features, verbose=0)
