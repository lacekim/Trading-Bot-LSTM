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
    PAPER_MAX_RISK_PER_TRADE = float(os.getenv("PAPER_MAX_RISK_PER_TRADE", "0.25"))
    PAPER_MIN_BARS_BETWEEN_TRADES = int(os.getenv("PAPER_MIN_BARS_BETWEEN_TRADES", "4"))
    PAPER_FEE_BPS = float(os.getenv("PAPER_FEE_BPS", "5"))
    PAPER_SLIPPAGE_BPS = float(os.getenv("PAPER_SLIPPAGE_BPS", "5"))
    PAPER_MAX_TRADES_PER_DAY = int(os.getenv("PAPER_MAX_TRADES_PER_DAY", "3"))
    PAPER_MAX_DAILY_LOSS_PCT = float(os.getenv("PAPER_MAX_DAILY_LOSS_PCT", "3"))
    PAPER_USE_COMPOUNDING = os.getenv("PAPER_USE_COMPOUNDING", "false").lower() in {"1", "true", "yes", "on"}

    # Daily discovery scans the whole market; hourly work follows only that day's GO list.
    MARKET_MOMENTUM_VALIDATION_CANDIDATES = 10

    # Explicit execution modes.  LIVE is intentionally unavailable until a
    # separate live adapter is configured and armed.
    EXECUTION_MODE = os.getenv("EXECUTION_MODE", "PAPER").upper()
    SHUTDOWN_MODE = os.getenv("SHUTDOWN_MODE", "GRACEFUL").upper()
    PAPER_DB_PATH = Path(os.getenv("PAPER_DB_PATH", "data/v5_paper_trading.sqlite3"))
    PAPER_STARTING_BALANCE = float(os.getenv("PAPER_STARTING_BALANCE", "10000"))
    PAPER_LEVERAGE = float(os.getenv("PAPER_LEVERAGE", "1"))
    PAPER_STOP_LOSS_PCT = float(os.getenv("PAPER_STOP_LOSS_PCT", "2"))
    PAPER_TAKE_PROFIT_PCT = float(os.getenv("PAPER_TAKE_PROFIT_PCT", "4"))
    PAPER_MAX_OPEN_POSITIONS = int(os.getenv("PAPER_MAX_OPEN_POSITIONS", "5"))
    PAPER_MAX_POSITION_PCT = float(os.getenv("PAPER_MAX_POSITION_PCT", "20"))
    PAPER_MAX_PORTFOLIO_EXPOSURE_PCT = float(os.getenv("PAPER_MAX_PORTFOLIO_EXPOSURE_PCT", "80"))
    PAPER_MIN_ORDER_USD = float(os.getenv("PAPER_MIN_ORDER_USD", "30"))
    PAPER_CASH_BUFFER_PCT = float(os.getenv("PAPER_CASH_BUFFER_PCT", "10"))
    PAPER_PRICE_IMPACT_BPS = float(os.getenv("PAPER_PRICE_IMPACT_BPS", "2"))
    PAPER_FUNDING_BPS_PER_DAY = float(os.getenv("PAPER_FUNDING_BPS_PER_DAY", "0"))
    PAPER_BORROWING_BPS_PER_DAY = float(os.getenv("PAPER_BORROWING_BPS_PER_DAY", "0"))
    PAPER_MAX_EQUITY_LOSS_PCT = float(os.getenv("PAPER_MAX_EQUITY_LOSS_PCT", "3"))
    PAPER_MAX_DRAWDOWN_PCT = float(os.getenv("PAPER_MAX_DRAWDOWN_PCT", "15"))
    PAPER_MAX_CONSECUTIVE_LOSSES = int(os.getenv("PAPER_MAX_CONSECUTIVE_LOSSES", "5"))
    PAPER_MAX_FAILED_ORDERS = int(os.getenv("PAPER_MAX_FAILED_ORDERS", "3"))
    PAPER_MAX_CANDLE_AGE_MULTIPLIER = float(os.getenv("PAPER_MAX_CANDLE_AGE_MULTIPLIER", "2.5"))
    PAPER_STOP_TARGET_PRIORITY = os.getenv("PAPER_STOP_TARGET_PRIORITY", "STOP_FIRST").upper()
    POSITION_MONITOR_INTERVAL_SECONDS = int(os.getenv("POSITION_MONITOR_INTERVAL_SECONDS", "10"))
    POSITION_MONITOR_FAILURE_THRESHOLD = int(os.getenv("POSITION_MONITOR_FAILURE_THRESHOLD", "3"))
    POSITION_MONITOR_MAX_PRICE_AGE_SECONDS = int(os.getenv("POSITION_MONITOR_MAX_PRICE_AGE_SECONDS", "180"))
    POSITION_MONITOR_REQUEST_TIMEOUT_SECONDS = int(os.getenv("POSITION_MONITOR_REQUEST_TIMEOUT_SECONDS", "5"))
    POSITION_MONITOR_WATCHDOG_SECONDS = int(os.getenv("POSITION_MONITOR_WATCHDOG_SECONDS", "45"))
    QUALIFIED_MARKET_INTERVAL_SECONDS = int(os.getenv("QUALIFIED_MARKET_INTERVAL_SECONDS", "10"))
    QUALIFIED_MARKET_MAX_SPREAD_BPS = float(os.getenv("QUALIFIED_MARKET_MAX_SPREAD_BPS", "50"))
    QUALIFIED_MARKET_SNAPSHOT_MAX_AGE_SECONDS = int(os.getenv("QUALIFIED_MARKET_SNAPSHOT_MAX_AGE_SECONDS", "30"))
    QUALIFIED_MARKET_WATCHDOG_SECONDS = int(os.getenv("QUALIFIED_MARKET_WATCHDOG_SECONDS", "45"))
    QUALIFIED_MARKET_RETENTION_HOURS = int(os.getenv("QUALIFIED_MARKET_RETENTION_HOURS", "168"))
    MONITOR_HEARTBEAT_RETENTION_HOURS = int(os.getenv("MONITOR_HEARTBEAT_RETENTION_HOURS", "168"))

    # Promotion uses recent economics plus separate validated/walk-forward evidence.
    GO_MIN_RECENT_PROFIT_FACTOR = float(os.getenv("GO_MIN_RECENT_PROFIT_FACTOR", "1.30"))
    GO_MIN_VALIDATED_PROFIT_FACTOR = float(os.getenv("GO_MIN_VALIDATED_PROFIT_FACTOR", "1.30"))
    GO_MIN_WALK_FORWARD_STABILITY = float(os.getenv("GO_MIN_WALK_FORWARD_STABILITY", "70"))
    GO_MIN_VALIDATED_TRADES = int(os.getenv("GO_MIN_VALIDATED_TRADES", "100"))
    GO_MAX_VALIDATED_DRAWDOWN_PCT = float(os.getenv("GO_MAX_VALIDATED_DRAWDOWN_PCT", "-5"))

    # Forward demotion is deliberately sample-gated to avoid reacting to a few trades.
    FORWARD_DEMOTION_MIN_TRADES = int(os.getenv("FORWARD_DEMOTION_MIN_TRADES", "20"))
    FORWARD_DEMOTION_WINDOW_TRADES = int(os.getenv("FORWARD_DEMOTION_WINDOW_TRADES", "30"))
    FORWARD_DEMOTION_MIN_PROFIT_FACTOR = float(os.getenv("FORWARD_DEMOTION_MIN_PROFIT_FACTOR", "1.0"))
    FORWARD_DEMOTION_MAX_DRAWDOWN_PCT = float(os.getenv("FORWARD_DEMOTION_MAX_DRAWDOWN_PCT", "5"))

    # Shadow-only challenger; these settings never alter baseline model outputs.
    CHALLENGER_MIN_PROBABILITY = float(os.getenv("CHALLENGER_MIN_PROBABILITY", "0.65"))
    CHALLENGER_TREND_LOOKBACK = int(os.getenv("CHALLENGER_TREND_LOOKBACK", "24"))
    CHALLENGER_MIN_CLOSE_LOCATION = float(os.getenv("CHALLENGER_MIN_CLOSE_LOCATION", "0.65"))
    GMX_PRICE_API_BASE = os.getenv("GMX_PRICE_API_BASE", "https://arbitrum-api.gmxinfra.io").rstrip("/")
    ARBITRUM_RPC_URL = os.getenv("ARBITRUM_RPC_URL", "")
    ARBITRUM_BACKUP_RPC_URL = os.getenv("ARBITRUM_BACKUP_RPC_URL", "")
    WEB3_WALLET_ADDRESS = os.getenv("WEB3_WALLET_ADDRESS", "")
    ARBITRUM_CHAIN_ID = 42161
    TELEGRAM_BOT_TOKEN = LegacyConfig.TELEGRAM_BOT_TOKEN
    TELEGRAM_ALLOWED_CHAT_IDS = set(LegacyConfig.TELEGRAM_ALLOWED_CHAT_IDS)
    # Older V5 name remains an alias so existing deployments keep working.
    TELEGRAM_AUTHORIZED_CHAT_IDS = TELEGRAM_ALLOWED_CHAT_IDS


Config = V4Config
