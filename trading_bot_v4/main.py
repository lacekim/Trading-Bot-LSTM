"""Entry point for the modular V4 trading bot."""

import argparse
import sys
from pathlib import Path

from trading_bot_v4.config_v4 import V4Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.ml.trainer import train_v4_model
from trading_bot_v4.ml.predictor import predict_with_v4_model
from trading_bot_v4.backtesting.backtest_engine import run_v4_backtest
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_main")


def parse_args():
    parser = argparse.ArgumentParser(description="Modular V4 trading bot")
    parser.add_argument("--train", action="store_true", help="Train the original CNN/LSTM model via V4 modules")
    parser.add_argument("--predict", action="store_true", help="Load the saved model and generate a sample prediction")
    parser.add_argument("--backtest", action="store_true", help="Run the original backtest logic via V4 modules")
    parser.add_argument("--refresh", action="store_true", help="Refresh the GMX OHLC cache before doing anything else")
    parser.add_argument("--symbol", default=V4Config.GMX_SYMBOL, help="GMX symbol to backtest")
    parser.add_argument("--timeframe", default=V4Config.TIMEFRAME, help="GMX data timeframe")
    parser.add_argument("--capital", type=float, default=100000.0, help="Starting capital for backtest")
    parser.add_argument("--all-assets", action="store_true", help="Backtest every GMX asset")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.refresh:
        handler = V4DataHandler()
        refreshed = handler.refresh_gmx_cache(force=True)
        if refreshed:
            print("GMX OHLC cache refreshed")
            return 0
        print("GMX refresh failed; local cache remains in place")
        return 1

    if args.train:
        train_v4_model(send_telegram=False)
        return 0
    if args.predict:
        predict_with_v4_model()
        return 0
    if args.backtest:
        run_v4_backtest(args)
        return 0

    print("V4 bot scaffold ready. Use --train, --predict, --backtest, or --refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
