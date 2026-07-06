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
    SMC_SWING_WINDOW = int(os.getenv("V4_SMC_SWING_WINDOW", "5"))
    SMC_MIN_SWING_DISTANCE_ATR = float(os.getenv("V4_SMC_MIN_SWING_DISTANCE_ATR", "0.5"))
    USE_SMC_FILTER = False

    # Paper-only execution constraints for analysis reports. These do not affect live trading.
    PAPER_MAX_RISK_PER_TRADE = 0.25
    PAPER_MIN_BARS_BETWEEN_TRADES = 4
    PAPER_FEE_BPS = 5
    PAPER_SLIPPAGE_BPS = 5
    PAPER_MAX_TRADES_PER_DAY = 3
    PAPER_MAX_DAILY_LOSS_PCT = 3
    PAPER_USE_COMPOUNDING = False

    # Paper-only scheduler scope. Daily research still refreshes and evaluates all assets.
    HOURLY_REFRESH_SYMBOLS = ["AIXBT", "DYDX"]


Config = V4Config
