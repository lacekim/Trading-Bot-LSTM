"""V4 data handling module derived from the original training and trading bot logic."""

from __future__ import annotations

from typing import Any
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_model import DataHandler as TrainingDataHandler
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_data_handler")


class V4DataHandler(TrainingDataHandler):
    """Compatibility wrapper around the original training DataHandler logic."""

    def __init__(self, scaler_path: str | None = None, logger_obj: Any | None = None):
        super().__init__()
        self.scaler_path = scaler_path
        self.logger = logger_obj or logger

    def ensure_atr(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        df = df.copy()
        if "ATR" not in df.columns:
            atr = self.calculate_atr(df, period or Config.ATR_PERIOD)
            df["ATR"] = atr
        return df

    def prepare_features(self, df: pd.DataFrame, prediction_horizon: int = 1):
        df = self.ensure_atr(df)
        return super().prepare_features(df, prediction_horizon=prediction_horizon)

    def refresh_gmx_cache(self, force: bool = False) -> bool:
        """Refresh the local GMX OHLC cache from source before loading CSVs."""
        if getattr(Config, "DATA_SOURCE", "").upper() != "GMX":
            return True
        if not getattr(Config, "GMX_AUTO_REFRESH_ENABLED", True):
            self.logger.info("GMX auto-refresh disabled")
            return True

        now = time.time()
        last_refresh = getattr(self, "_last_gmx_refresh_ts", 0)
        refresh_seconds = int(getattr(Config, "GMX_AUTO_REFRESH_SECONDS", 3300))
        if not force and last_refresh and (now - last_refresh) < refresh_seconds:
            self.logger.info("GMX data refresh skipped; cache was refreshed recently")
            return True

        update_script = Path(getattr(Config, "GMX_UPDATE_SCRIPT"))
        update_config = Path(getattr(Config, "GMX_UPDATE_CONFIG"))
        if not update_script.exists():
            self.logger.warning(f"GMX update script not found: {update_script}")
            return False
        if not update_config.exists():
            self.logger.warning(f"GMX update config not found: {update_config}")
            return False

        cmd = [
            sys.executable,
            str(update_script),
            "--config",
            str(update_config),
            "--period",
            Config.TIMEFRAME,
            "--chain",
            getattr(Config, "GMX_UPDATE_CHAIN", "arbitrum"),
        ]

        try:
            self.logger.info(f"Refreshing GMX {Config.TIMEFRAME} OHLC cache before fetch")
            result = subprocess.run(
                cmd,
                cwd=update_script.parent,
                text=True,
                capture_output=True,
                timeout=int(getattr(Config, "GMX_UPDATE_TIMEOUT_SECONDS", 900)),
            )
            if result.stdout:
                self.logger.info(result.stdout.strip())
            if result.stderr:
                self.logger.warning(result.stderr.strip())
            if result.returncode != 0:
                self.logger.warning(f"GMX data refresh failed with exit code {result.returncode}")
                return False

            self._last_gmx_refresh_ts = now
            self.logger.info("GMX OHLC cache refresh complete")
            return True
        except Exception as exc:
            self.logger.warning(f"GMX data refresh failed: {exc}")
            return False

    def fetch_data(self, symbol=None, period=None, interval=None, include_sentiment=True):
        try:
            interval = interval or Config.TIMEFRAME
            if getattr(Config, "DATA_SOURCE", "").upper() == "GMX":
                if not self.refresh_gmx_cache():
                    self.logger.warning("Proceeding with existing GMX cache after failed refresh")
                from trading_bot import load_gmx_ohlc

                asset_symbol = symbol or Config.GMX_SYMBOL
                df = load_gmx_ohlc(asset_symbol, interval)
                if include_sentiment:
                    df = self.merge_sentiment_data(df, interval, symbol=asset_symbol)
                return df

            from kraken_data import load_kraken_ohlc

            df = load_kraken_ohlc(Config.KRAKEN_PAIR, period, interval, prefer_live=True, logger=self.logger)
            if include_sentiment:
                df = self.merge_sentiment_data(df, interval, symbol=Config.SYMBOL)
            return df
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning("V4DataHandler.fetch_data fallback: %s", exc)
            return None

    def normalize_data(self, data, fit=True):
        return super().normalize_data(data, fit=fit)

    def prepare_sequences(self, data, target, seq_length):
        return super().prepare_sequences(data, target, seq_length)

    def merge_sentiment_data(self, df, interval, symbol=None):
        """Best-effort sentiment merge using the original bot logic if available."""
        try:
            from trading_bot import DataHandler as TradingBotDataHandler

            legacy_handler = TradingBotDataHandler(scaler_path=self.scaler_path or "")
            return legacy_handler.merge_sentiment_data(df, interval, symbol=symbol)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.debug("Sentiment merge unavailable: %s", exc)
            return df


DataHandler = V4DataHandler
