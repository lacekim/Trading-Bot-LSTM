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
from trading_bot_v4.backtesting.smc_shadow_backtest import run_smc_shadow_backtest
from trading_bot_v4.backtesting.walk_forward import WALK_FORWARD_REPORT_PATH, WALK_FORWARD_SUMMARY_PATH, run_walk_forward_smc_validation
from trading_bot_v4.core.smc_swings import analyze_gmx_smc_swings
from trading_bot_v4.execution.paper_smc_filter import run_paper_trade_smc_filter
from trading_bot_v4.features.smc_feature_builder import build_all_assets_smc_training_data, build_smc_training_data
from trading_bot_v4.ml.smc_trainer import train_smc_model
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_main")


def format_optional_metric(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "none"
    if numeric != numeric:
        return "none"
    return f"{numeric:.6f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Modular V4 trading bot")
    parser.add_argument("--train", action="store_true", help="Train the original CNN/LSTM model via V4 modules")
    parser.add_argument("--predict", action="store_true", help="Load the saved model and generate a sample prediction")
    parser.add_argument("--backtest", action="store_true", help="Run a V4 backtest over one asset or all assets")
    parser.add_argument("--backtest-rank", action="store_true", help="Backtest every GMX asset and rank them by risk-adjusted performance")
    parser.add_argument("--smc-shadow-backtest", action="store_true", help="Compare baseline model behavior against selected SMC filters")
    parser.add_argument("--walk-forward-smc", action="store_true", help="Run walk-forward baseline vs SMC-filter validation for every GMX asset")
    parser.add_argument("--paper-trade-smc", action="store_true", help="Run paper-only model signal evaluation with the optional SMC filter")
    parser.add_argument("--build-smc-training-data", action="store_true", help="Build an optional SMC-enhanced training dataset")
    parser.add_argument("--train-smc-model", action="store_true", help="Train a separate optional SMC-enhanced model")
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
    if args.smc_shadow_backtest:
        result = run_smc_shadow_backtest(args)
        if args.all_assets:
            summary_df = result["summary_df"]
            print(f"V4 SMC shadow backtest summary saved to {result['summary_path']}")
            print(f"SMC filter features: {', '.join(result['smc_filter_features'])}")
            print(f"SMC filter lookback candles: {result['smc_filter_lookback']}")
            display_columns = [
                "ranking",
                "symbol",
                "smc_filtered_return_pct",
                "drawdown_improvement_pct",
                "profit_factor_improvement",
                "smc_return_drawdown_ratio",
                "baseline_return_pct",
                "baseline_max_drawdown_pct",
                "smc_filtered_max_drawdown_pct",
                "baseline_profit_factor",
                "smc_filtered_profit_factor",
            ]
            if not summary_df.empty:
                print(summary_df[display_columns].head(20).to_string(index=False))
            return 0

        baseline = result["baseline_summary"]
        filtered = result["filtered_summary"]
        print(f"SMC shadow backtest: {baseline['symbol']} {baseline['timeframe']}")
        print(f"SMC filter features: {', '.join(result['smc_filter_features'])}")
        print(f"SMC filter lookback candles: {result['smc_filter_lookback']}")
        print(f"baseline return: {format_optional_metric(baseline.get('return_pct'))}%")
        print(f"SMC-filtered return: {format_optional_metric(filtered.get('return_pct'))}%")
        print(f"baseline trades: {int(baseline.get('signals_traded', 0))}")
        print(f"SMC-filtered trades: {int(filtered.get('signals_traded', 0))}")
        print(f"baseline win rate: {format_optional_metric(baseline.get('win_rate_pct'))}%")
        print(f"SMC-filtered win rate: {format_optional_metric(filtered.get('win_rate_pct'))}%")
        print(f"baseline max drawdown: {format_optional_metric(baseline.get('max_drawdown_pct'))}%")
        print(f"SMC-filtered max drawdown: {format_optional_metric(filtered.get('max_drawdown_pct'))}%")
        print(f"baseline profit factor: {format_optional_metric(baseline.get('profit_factor'))}")
        print(f"SMC-filtered profit factor: {format_optional_metric(filtered.get('profit_factor'))}")
        return 0
    if args.walk_forward_smc:
        summary_df = run_walk_forward_smc_validation(args)
        print(f"V4 walk-forward summary saved to {WALK_FORWARD_SUMMARY_PATH}")
        print(f"V4 walk-forward HTML report saved to {WALK_FORWARD_REPORT_PATH}")
        if not summary_df.empty:
            display_columns = [
                "symbol",
                "window",
                "strategy",
                "return_pct",
                "max_drawdown_pct",
                "profit_factor",
                "sharpe_ratio",
                "trades",
                "win_rate_pct",
            ]
            print(summary_df[display_columns].head(30).to_string(index=False))
        return 0
    if args.paper_trade_smc:
        result = run_paper_trade_smc_filter(args)
        if result.get("all_assets"):
            summary_df = result.get("summary_df")
            print(f"V4 paper SMC filter summary saved to {result['summary_path']}")
            print("live trading: disabled")
            print(f"assets evaluated: {result['assets']}")
            print(f"predictions evaluated: {result['predictions']}")
            print(f"paper trade candidates: {result['paper_candidates']}")
            print(f"SMC allowed candidates: {result['allowed']}")
            print(f"SMC blocked candidates: {result['blocked']}")
            print(f"blocked trade log: {result['blocked_log_path']}")
            if summary_df is not None and not summary_df.empty:
                display_columns = [
                    "symbol",
                    "candidates",
                    "allowed",
                    "blocked",
                    "block_rate",
                    "latest_allowed_signal",
                    "latest_blocked_signal",
                ]
                print(summary_df[display_columns].head(30).to_string(index=False))
            return 0

        latest = result.get("latest") or {}
        print(f"V4 paper SMC filter: {result['symbol']} {result['timeframe']}")
        print("live trading: disabled")
        print(f"predictions evaluated: {result['predictions']}")
        print(f"paper trade candidates: {result['paper_candidates']}")
        print(f"SMC allowed candidates: {result['allowed']}")
        print(f"SMC blocked candidates: {result['blocked']}")
        print(f"blocked trade log: {result['blocked_log_path']}")
        if latest:
            print(f"latest model probability: {format_optional_metric(latest.get('model_probability'))}")
            print(f"latest model direction: {latest.get('model_direction', 'none')}")
            print(f"latest SMC context: {latest.get('smc_context', 'none')}")
            print(f"latest price: {format_optional_metric(latest.get('price'))}")
        return 0
    if args.build_smc_training_data:
        if args.all_assets:
            result = build_all_assets_smc_training_data(args.timeframe)
            print(f"assets processed: {result.assets_processed}")
            print(f"total rows: {result.total_rows}")
            print(f"original feature count: {result.original_feature_count}")
            print(f"SMC feature count: {result.smc_feature_count}")
            print(f"total feature count: {result.total_feature_count}")
            print(f"output path: {result.output_path}")
            return 0

        result = build_smc_training_data(args.symbol, args.timeframe)
        print(f"symbol: {result.symbol}")
        print(f"timeframe: {result.timeframe}")
        print(f"original feature count: {result.original_feature_count}")
        print(f"SMC feature count: {result.smc_feature_count}")
        print(f"total feature count: {result.total_feature_count}")
        print(f"rows written: {result.rows_written}")
        print(f"output path: {result.output_path}")
        return 0
    if args.train_smc_model:
        result = train_smc_model(args.timeframe)
        print(f"rows used: {result.rows_used}")
        print(f"feature count: {result.feature_count}")
        print(
            "train/validation split: "
            f"{result.train_rows}/{result.validation_rows} rows "
            f"({result.train_sequences}/{result.validation_sequences} sequences)"
        )
        print(f"validation accuracy: {format_optional_metric(result.validation_accuracy)}")
        print(f"validation loss: {format_optional_metric(result.validation_loss)}")
        print(f"model path: {result.model_path}")
        print(f"scaler path: {result.scaler_path}")
        return 0
    if args.compare_original:
        run_v4_compare_original(args)
        return 0
    if args.analyze_smc:
        output_path, summary_path, diagnostics, ranking, validation = analyze_gmx_smc_swings(
            args.symbol,
            args.timeframe,
            swing_window=args.swing_window,
            min_swing_distance_atr=args.min_swing_distance_atr,
        )
        print(f"V4 SMC swing features saved to {output_path}")
        print(f"V4 SMC summary saved to {summary_path}")
        print(f"V4 SMC feature diagnostics saved to {diagnostics.output_path}")
        print(f"V4 SMC feature ranking saved to {ranking.output_path}")
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
        print(f"total bullish liquidity sweeps: {validation.total_bullish_liquidity_sweeps}")
        print(f"total bearish liquidity sweeps: {validation.total_bearish_liquidity_sweeps}")
        print(f"latest liquidity sweep type: {validation.latest_liquidity_sweep_type}")
        bars_since_sweep = "none"
        if validation.bars_since_latest_liquidity_sweep is not None:
            bars_since_sweep = str(validation.bars_since_latest_liquidity_sweep)
        print(f"bars since latest liquidity sweep: {bars_since_sweep}")
        print(f"current regime: {validation.current_regime}")
        print(f"trending candles count: {validation.trending_candles_count}")
        print(f"ranging candles count: {validation.ranging_candles_count}")
        print(f"high volatility count: {validation.high_volatility_count}")
        print(f"low volatility count: {validation.low_volatility_count}")
        print("top 10 SMC feature correlations by absolute value:")
        for rank, row in enumerate(diagnostics.top_correlations, start=1):
            print(
                f"{rank}. {row['feature']} "
                f"(next {row['future_return_horizon']}): "
                f"corr={row['correlation']:.6f}, "
                f"abs={row['abs_correlation']:.6f}"
            )
        print("top 20 SMC features by combined ranking score:")
        for row in ranking.top_features:
            print(
                f"{row['ranking']}. {row['feature']}: "
                f"score={format_optional_metric(row['combined_score'])}, "
                f"Pearson={format_optional_metric(row['Pearson'])}, "
                f"Spearman={format_optional_metric(row['Spearman'])}, "
                f"MI={format_optional_metric(row['Mutual Information'])}, "
                f"RF={format_optional_metric(row['RF importance'])}, "
                f"XGB={format_optional_metric(row['XGBoost importance'])}"
            )
        return 0

    print("V4 bot scaffold ready. Use --train, --predict, --backtest, --backtest-rank, --smc-shadow-backtest, --walk-forward-smc, --compare-original, --analyze-smc, or --refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
