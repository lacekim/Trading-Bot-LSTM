"""The protected original 1-hour research strategy contract.

This contract makes differences visible.  Paper safety overlays are allowed,
but must never be described as exact baseline parity while this audit reports
deviations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from trading_bot_v4.config_v4 import V4Config as Config


@dataclass(frozen=True)
class OriginalBaselineContract:
    timeframe: str = "1h"
    sequence_length: int = 36
    signal_threshold: float = 0.58
    risk_percentage: float = 1.0
    atr_stop_multiplier: float = 0.75
    atr_target_multiplier: float = 1.5
    maximum_hold_candles: int = 1
    cooldown_candles: int = 0
    maximum_trades_per_day: int | None = None
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    compounding: bool = True


ORIGINAL_BASELINE = OriginalBaselineContract()


def active_execution_parity_audit(fee_bps: float | None = None,
                                  slippage_bps: float | None = None,
                                  symbol: str = "") -> dict[str, object]:
    from trading_bot_v4.execution.persistent_policy import policy_for
    policy = policy_for(symbol)
    active = {
        "timeframe": Config.TIMEFRAME,
        "sequence_length": Config.SEQUENCE_LENGTH,
        "signal_threshold": policy.threshold,
        "risk_percentage": Config.RISK_PERCENTAGE,
        "atr_stop_multiplier": policy.stop_atr,
        "atr_target_multiplier": policy.target_atr,
        "maximum_hold_candles": policy.max_hold_candles,
        "cooldown_candles": Config.PAPER_MIN_BARS_BETWEEN_TRADES,
        "maximum_trades_per_day": Config.PAPER_MAX_TRADES_PER_DAY or None,
        "fee_bps": Config.PAPER_FEE_BPS if fee_bps is None else float(fee_bps),
        "slippage_bps": (
            Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS
            if slippage_bps is None else float(slippage_bps)
        ),
        "compounding": True,
        "fixed_stop_loss_pct": None,
        "fixed_take_profit_pct": None,
    }
    expected = asdict(ORIGINAL_BASELINE)
    deviations = {
        key: {"baseline": expected[key], "active": active.get(key)}
        for key in expected if active.get(key) != expected[key]
    }
    return {
        "symbol": str(symbol).upper() or "DEFAULT",
        "persistent_policy": policy.enabled,
        "configuration_parity": not deviations,
        "baseline": expected,
        "active": active,
        "deviations": deviations,
    }
