"""Entry point for the modular V4 trading bot."""

import argparse
import sys
from pathlib import Path

from trading_bot_v4.config_v4 import V4Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.ml.trainer import train_v4_model
from trading_bot_v4.ml.predictor import predict_with_v4_model
from trading_bot_v4.backtesting.backtest_engine import run_v4_backtest
from trading_bot_v4.backtesting.comparison_engine import run_v4_compare_original
from trading_bot_v4.backtesting.ranking_engine import run_v4_backtest_ranking
from trading_bot_v4.core.smc_swings import analyze_gmx_smc_swings
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_main")


def parse_args():
    parser = argparse.ArgumentParser(description="Modular V4 trading bot")
    parser.add_argument("--train", action="store_true", help="Train the original CNN/LSTM model via V4 modules")
    parser.add_argument("--predict", action="store_true", help="Load the saved model and generate a sample prediction")
    parser.add_argument("--backtest", action="store_true", help="Run a V4 backtest over one asset or all assets")
    parser.add_argument("--backtest-rank", action="store_true", help="Backtest every GMX asset and rank them by risk-adjusted performance")
    parser.add_argument("--compare", action="store_true", dest="compare_original", help="Compare the original bot and V4 on the same asset")
    parser.add_argument("--compare-original", action="store_true", dest="compare_original", help="Compare the original bot and V4 on the same asset")
    parser.add_argument("--analyze-smc", action="store_true", help="Generate standalone V4 SMC swing high/low features")
    parser.add_argument("--swing-window", type=int, default=V4Config.SMC_SWING_WINDOW, help="Bars on each side used to confirm SMC swings")
    parser.add_argument(
        "--min-swing-distance-atr",
        type=float,
        default=V4Config.SMC_MIN_SWING_DISTANCE_ATR,
        help="Minimum distance from the previous same-side swing, measured in ATR",
    )
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
    if args.backtest_rank:
        run_v4_backtest_ranking(args)
        return 0
    if args.compare_original:
        run_v4_compare_original(args)
        return 0
    if args.analyze_smc:
        output_path, validation = analyze_gmx_smc_swings(
            args.symbol,
            args.timeframe,
            swing_window=args.swing_window,
            min_swing_distance_atr=args.min_swing_distance_atr,
        )
        print(f"V4 SMC swing features saved to {output_path}")
        print(f"total swing highs: {validation.total_swing_highs}")
        print(f"total swing lows: {validation.total_swing_lows}")
        print(f"total bullish BOS: {validation.total_bullish_bos}")
        print(f"total bearish BOS: {validation.total_bearish_bos}")
        print(f"total bullish CHOCH: {validation.total_bullish_choch}")
        print(f"total bearish CHOCH: {validation.total_bearish_choch}")
        print(f"current structure_trend: {validation.current_structure_trend}")
        print(f"latest BOS/CHOCH signal: {validation.latest_structure_signal}")
        print(f"total bullish FVG: {validation.total_bullish_fvg}")
        print(f"total bearish FVG: {validation.total_bearish_fvg}")
        print(f"open FVGs: {validation.open_fvgs}")
        print(f"filled FVGs: {validation.filled_fvgs}")
        nearest_fvg_distance = "none"
        if validation.nearest_fvg_distance is not None:
            nearest_fvg_distance = f"{validation.nearest_fvg_distance:.10f}"
        print(f"nearest FVG distance: {nearest_fvg_distance}")
        print(f"total bullish order blocks: {validation.total_bullish_order_blocks}")
        print(f"total bearish order blocks: {validation.total_bearish_order_blocks}")
        print(f"open order blocks: {validation.open_order_blocks}")
        print(f"mitigated order blocks: {validation.mitigated_order_blocks}")
        nearest_ob_distance = "none"
        if validation.nearest_ob_distance is not None:
            nearest_ob_distance = f"{validation.nearest_ob_distance:.10f}"
        print(f"nearest OB distance: {nearest_ob_distance}")
        return 0

    print("V4 bot scaffold ready. Use --train, --predict, --backtest, --backtest-rank, --compare-original, --analyze-smc, or --refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
