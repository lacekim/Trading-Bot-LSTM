"""Per-symbol lower-turnover policy shared by inference, backtest, and paper."""
from dataclasses import dataclass
import os

from trading_bot_v4.config_v4 import V4Config as Config


@dataclass(frozen=True)
class PersistentPolicy:
    enabled: bool
    threshold: float
    stop_atr: float
    target_atr: float
    max_hold_candles: int
    use_tier_sizing: bool
    require_macd: bool


def persistent_symbols() -> set[str]:
    return {x.strip().upper() for x in os.getenv("PERSISTENT_POLICY_SYMBOLS", "").split(",") if x.strip()}


def policy_for(symbol: str) -> PersistentPolicy:
    if str(symbol).upper() in persistent_symbols():
        return PersistentPolicy(True, float(os.getenv("PERSISTENT_SIGNAL_THRESHOLD", "0.60")),
            float(os.getenv("PERSISTENT_STOP_ATR", "0.50")), float(os.getenv("PERSISTENT_TARGET_ATR", "20.0")),
            int(os.getenv("PERSISTENT_MAX_HOLD_CANDLES", "240")), True, True)
    return PersistentPolicy(False, Config.MIN_SIGNAL_THRESHOLD, Config.ATR_SL_MULTIPLIER,
        Config.ATR_TP_MULTIPLIER, Config.PAPER_MAX_HOLD_CANDLES, True, True)


def direction_for_probability(probability: float, symbol: str) -> str:
    policy = policy_for(symbol)
    if probability > policy.threshold:
        return "LONG"
    if policy.enabled and probability < 1.0 - policy.threshold:
        return "SHORT"
    return "HOLD"
