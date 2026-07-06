"""V4 backtesting engine that preserves the original strategy rules while adding caching and rich research reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot import list_gmx_symbols
from trading_bot_v4.backtesting.ranking_engine import run_symbol_v4_backtest_ranking
from trading_bot_v4.backtesting.reporting import write_v4_backtest_html_report
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache

logger = build_logger("v4_backtest_engine")


def run_v4_backtest(args: Any | None = None) -> pd.DataFrame:
    if args is None:
        args = type("Args", (), {"all_assets": True, "symbol": Config.GMX_SYMBOL, "timeframe": Config.TIMEFRAME, "capital": 100000.0})()

    timeframe = getattr(args, "timeframe", Config.TIMEFRAME)
    starting_capital = float(getattr(args, "capital", 100000.0))
    use_all_assets = bool(getattr(args, "all_assets", False))
    symbol = getattr(args, "symbol", Config.GMX_SYMBOL)

    if getattr(Config, "DATA_SOURCE", "").upper() == "GMX":
        handler = V4DataHandler()
        handler.refresh_gmx_cache(force=False)

    cache = ModelScalerCache()
    model, scaler = cache.load()
    data_handler = V4DataHandler(str(cache.scaler_path), scaler=scaler)

    symbols = list_gmx_symbols(timeframe) if use_all_assets else [symbol]
    if not symbols:
        raise FileNotFoundError(f"No GMX {timeframe} files found in {Config.GMX_OHLC_DIR}")

    summaries = []
    trades_by_symbol = {}
    for asset_symbol in symbols:
        try:
            summary, trades = run_symbol_v4_backtest_ranking(model, data_handler, asset_symbol, timeframe, starting_capital)
            summaries.append(summary)
            trades_by_symbol[asset_symbol] = trades
        except Exception as exc:
            logger.warning("Backtest failed for %s: %s", asset_symbol, exc)

    summary_df = pd.DataFrame(summaries)
    summary_path = Path(f"trading_bot_gmx_{timeframe}_all_assets_summary.csv") if use_all_assets else Path(f"trading_bot_gmx_{timeframe}_{symbol}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    report_path = Path("v4_backtest_research_report.html")
    write_v4_backtest_html_report(summary_df, trades_by_symbol, report_path, title="V4 Backtest Research Report")

    print(f"V4 backtest summary saved to {summary_path}")
    print(f"V4 HTML report saved to {report_path}")
    if not summary_df.empty:
        print(summary_df[["symbol", "return_pct", "final_capital", "signals_traded", "win_rate_pct", "max_drawdown_pct", "profit_factor", "sharpe_ratio", "sortino_ratio", "average_trade_return", "expectancy"]].head(20).to_string(index=False))

    return summary_df


def run_symbol_backtest(model, data_handler, symbol, timeframe, starting_capital):
    if getattr(Config, "DATA_SOURCE", "").upper() == "GMX":
        handler = V4DataHandler()
        handler.refresh_gmx_cache(force=False)
    return run_symbol_v4_backtest_ranking(model, data_handler, symbol, timeframe, starting_capital)
