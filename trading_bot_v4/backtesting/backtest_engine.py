"""Backtesting engine wrapper around the original trading_bot.py backtest implementation."""

from __future__ import annotations

from trading_bot import parse_args as legacy_parse_args
from trading_bot import run_backtest as legacy_run_backtest
from trading_bot import run_lstm_backtest_for_symbol as legacy_run_lstm_backtest_for_symbol
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler


def run_v4_backtest(args=None):
    if args is None:
        args = legacy_parse_args()

    if getattr(Config, "DATA_SOURCE", "").upper() == "GMX":
        handler = V4DataHandler()
        handler.refresh_gmx_cache()

    return legacy_run_backtest(args)


def run_symbol_backtest(model, data_handler, symbol, timeframe, starting_capital):
    if getattr(Config, "DATA_SOURCE", "").upper() == "GMX":
        handler = V4DataHandler()
        handler.refresh_gmx_cache()
    return legacy_run_lstm_backtest_for_symbol(model, data_handler, symbol, timeframe, starting_capital)
