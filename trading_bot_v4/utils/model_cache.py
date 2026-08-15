"""Shared model/scaler cache for V4 backtesting and comparison modes."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Literal

from tensorflow.keras.models import load_model

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_model_cache")

Direction = Literal["long", "short"]

PER_ASSET_MODEL_ROOT: dict[Direction, Path] = {
    "long": Path("models/long"),
    "short": Path("models/short"),
}


def per_asset_model_root(direction: Direction, horizon: int = 1) -> Path:
    """horizon=1 (the default/production prediction_horizon) keeps today's
    models/long, models/short paths unchanged. A different horizon gets its
    own parallel namespace (models/long_h6, ...) so horizon experiments never
    overwrite the production models or their calibration."""
    base = PER_ASSET_MODEL_ROOT[direction]
    return base if horizon == 1 else base.with_name(f"{base.name}_h{horizon}")


def per_asset_model_path(direction: Direction, symbol: str, horizon: int = 1) -> Path:
    return per_asset_model_root(direction, horizon) / f"lstm_{symbol.upper()}_model.h5"


def per_asset_scaler_path(direction: Direction, symbol: str, horizon: int = 1) -> Path:
    return per_asset_model_root(direction, horizon) / f"scaler_{symbol.upper()}.pkl"


def per_asset_calibration_path(direction: Direction, horizon: int = 1) -> Path:
    return per_asset_model_root(direction, horizon) / "calibration.json"


def load_per_asset_calibration(direction: Direction, horizon: int = 1) -> dict[str, Any]:
    path = per_asset_calibration_path(direction, horizon)
    if not path.exists():
        return {"promoted": False, "promoted_symbols": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def newest_per_asset_mtime(direction: Direction, horizon: int = 1) -> float:
    root = per_asset_model_root(direction, horizon)
    if not root.exists():
        return 0.0
    mtimes = [p.stat().st_mtime for p in root.glob("*.h5")]
    return max(mtimes) if mtimes else 0.0


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


class PerAssetModelCache:
    """Lazily loads and caches one (model, scaler) pair per (direction, symbol).

    Unlike ModelScalerCache (one fixed global model), this resolves a distinct
    model file per GMX symbol, so callers looping over many symbols get the
    model actually trained on that symbol's own data instead of one model
    reused for every asset.
    """

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon
        self._caches: dict[tuple[Direction, str], ModelScalerCache] = {}

    def has(self, direction: Direction, symbol: str) -> bool:
        return (
            per_asset_model_path(direction, symbol, self.horizon).exists()
            and per_asset_scaler_path(direction, symbol, self.horizon).exists()
        )

    def get(self, direction: Direction, symbol: str) -> tuple[Any, Any] | None:
        """Returns (model, scaler) for this symbol, or None if no per-asset
        model has been trained/promoted for it yet. Callers should skip the
        symbol rather than fall back to a different symbol's model."""
        key = (direction, symbol.upper())
        if key not in self._caches:
            if not self.has(direction, symbol):
                return None
            self._caches[key] = ModelScalerCache(
                per_asset_model_path(direction, symbol, self.horizon),
                per_asset_scaler_path(direction, symbol, self.horizon),
            )
        try:
            return self._caches[key].load()
        except FileNotFoundError:
            return None
