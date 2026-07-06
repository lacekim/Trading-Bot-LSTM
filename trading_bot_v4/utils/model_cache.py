"""Shared model/scaler cache for V4 backtesting and comparison modes."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from tensorflow.keras.models import load_model

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_model_cache")


class ModelScalerCache:
    def __init__(self, model_path: str | Path | None = None, scaler_path: str | Path | None = None):
        self.model_path = Path(model_path or (Config.MODEL_DIR / Config.MODEL_NAME))
        self.scaler_path = Path(scaler_path or (Config.MODEL_DIR / Config.SCALER_NAME))
        self._model: Any | None = None
        self._scaler: Any | None = None

    def load(self) -> tuple[Any, Any]:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            logger.info(f"Loading model from {self.model_path}")
            self._model = load_model(self.model_path)

        if self._scaler is None:
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
            logger.info(f"Loading scaler from {self.scaler_path}")
            with self.scaler_path.open("rb") as handle:
                self._scaler = pickle.load(handle)

        return self._model, self._scaler
