"""V4 configuration wrapper that preserves the original Config behavior and adds optional SMC toggles."""

import os
from pathlib import Path

from config import Config as LegacyConfig


class V4Config(LegacyConfig):
    """Compatibility layer for the modular V4 package."""

    # Optional SMC features are disabled by default and do not alter the current training/live path.
    ENABLE_SMC = os.getenv("V4_ENABLE_SMC", "false").lower() in {"1", "true", "yes", "on"}
    SMC_MIN_CONFIDENCE = float(os.getenv("V4_SMC_MIN_CONFIDENCE", "0.65"))
    SMC_USE_ORDER_BLOCKS = os.getenv("V4_USE_ORDER_BLOCKS", "false").lower() in {"1", "true", "yes", "on"}
    SMC_USE_FVG = os.getenv("V4_USE_FVG", "false").lower() in {"1", "true", "yes", "on"}
    SMC_USE_LIQUIDITY_SWEEPS = os.getenv("V4_USE_LIQUIDITY_SWEEPS", "false").lower() in {"1", "true", "yes", "on"}


Config = V4Config
