"""Walk-forward SMC validation for V4 model signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols
from trading_bot_v4.backtesting.ranking_engine import _compute_trade_metrics
from trading_bot_v4.backtesting.smc_shadow_backtest import (
    SMC_FILTER_LOOKBACK,
    SMC_SHADOW_FEATURES,
    _add_smc_filters,
    _build_prediction_signals,
    _run_filtered_signal_backtest,
)
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache


logger = build_logger("v4_walk_forward")

WALK_FORWARD_SUMMARY_PATH = Path("v4_walk_forward_summary.csv")
WALK_FORWARD_REPORT_PATH = Path("v4_walk_forward_report.html")
WINDOW_SPLITS = [
    ("train", 0.0, 0.70),
    ("validation", 0.70, 0.85),
    ("test", 0.85, 1.0),
]


def _summarize_window_trades(
    symbol: str,
    timeframe: str,
    window_name: str,
    strategy: str,
    trades: pd.DataFrame,
    equity: list[tuple[pd.Timestamp, float]],
    starting_capital: float,
    final_capital: float,
) -> dict[str, float | int | str]:
    if trades.empty:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "window": window_name,
            "strategy": strategy,
            "start_time": "",
            "end_time": "",
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "starting_capital": starting_capital,
            "final_capital": starting_capital,
        }

    anchored_equity = list(equity)
    if anchored_equity:
        first_timestamp = pd.Timestamp(anchored_equity[0][0])
        anchored_equity.insert(0, (first_timestamp, float(starting_capital)))
    equity_df = pd.DataFrame(anchored_equity, columns=["timestamp", "equity"]).set_index("timestamp")
    drawdown = (equity_df["equity"] / equity_df["equity"].cummax() - 1).min() * 100 if not equity_df.empty else 0.0
    metrics = _compute_trade_metrics(trades, starting_capital, final_capital, drawdown)
    wins = trades[trades["profit"] > 0]
    return_pct = ((final_capital / starting_capital) - 1) * 100 if starting_capital else 0.0
    timestamps = pd.to_datetime(trades["timestamp"], errors="coerce")
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "window": window_name,
        "strategy": strategy,
        "start_time": str(timestamps.min()) if not timestamps.empty else "",
        "end_time": str(timestamps.max()) if not timestamps.empty else "",
        "return_pct": return_pct,
        "max_drawdown_pct": float(drawdown),
        "profit_factor": float(metrics["profit_factor"]),
        "sharpe_ratio": float(metrics["sharpe_ratio"]),
        "trades": int(len(trades)),
        "win_rate_pct": (len(wins) / len(trades) * 100) if len(trades) else 0.0,
        "starting_capital": starting_capital,
        "final_capital": final_capital,
    }


def _run_window_signal_backtest(
    signals: pd.DataFrame,
    symbol: str,
    timeframe: str,
    window_name: str,
    strategy: str,
    starting_capital: float,
) -> dict[str, float | int | str]:
    if signals.empty:
        return _summarize_window_trades(symbol, timeframe, window_name, strategy, pd.DataFrame(), [], starting_capital, starting_capital)

    working = signals.copy()
    if strategy == "baseline":
        working["smc_allow_long"] = True
        working["smc_allow_short"] = True

    summary, trades = _run_filtered_signal_backtest(
        working,
        symbol=symbol,
        timeframe=timeframe,
        candles=len(working),
        starting_capital=starting_capital,
    )
    if trades.empty:
        return _summarize_window_trades(symbol, timeframe, window_name, strategy, trades, [], starting_capital, starting_capital)

    equity = list(zip(pd.to_datetime(trades["timestamp"]), trades["capital"]))
    return _summarize_window_trades(
        symbol=symbol,
        timeframe=timeframe,
        window_name=window_name,
        strategy=strategy,
        trades=trades,
        equity=equity,
        starting_capital=starting_capital,
        final_capital=float(summary["final_capital"]),
    )


def _window_slices(signals: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    total = len(signals)
    windows = []
    for name, start_pct, end_pct in WINDOW_SPLITS:
        start = int(np.floor(total * start_pct))
        end = int(np.floor(total * end_pct)) if end_pct < 1.0 else total
        windows.append((name, signals.iloc[start:end].copy()))
    return windows


def run_symbol_walk_forward_smc(
    model: Any,
    data_handler: Any,
    symbol: str,
    timeframe: str,
    starting_capital: float,
) -> list[dict[str, float | int | str]]:
    signals, _ = _build_prediction_signals(model, data_handler, symbol, timeframe)
    signals = _add_smc_filters(signals, symbol, timeframe)

    rows = []
    for window_name, window_signals in _window_slices(signals):
        rows.append(
            _run_window_signal_backtest(
                window_signals,
                symbol=symbol,
                timeframe=timeframe,
                window_name=window_name,
                strategy="baseline",
                starting_capital=starting_capital,
            )
        )
        rows.append(
            _run_window_signal_backtest(
                window_signals,
                symbol=symbol,
                timeframe=timeframe,
                window_name=window_name,
                strategy="smc_filtered",
                starting_capital=starting_capital,
            )
        )
    return rows


def _write_walk_forward_html(summary_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary_df.copy()

    pivot = summary.pivot_table(
        index=["symbol", "window"],
        columns="strategy",
        values=["return_pct", "max_drawdown_pct", "profit_factor", "sharpe_ratio", "trades", "win_rate_pct"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{strategy}" for metric, strategy in pivot.columns]
    pivot = pivot.reset_index()
    if {"return_pct_smc_filtered", "return_pct_baseline"}.issubset(pivot.columns):
        pivot["return_improvement_pct"] = pivot["return_pct_smc_filtered"] - pivot["return_pct_baseline"]
    if {"max_drawdown_pct_smc_filtered", "max_drawdown_pct_baseline"}.issubset(pivot.columns):
        pivot["drawdown_improvement_pct"] = pivot["max_drawdown_pct_baseline"].abs() - pivot["max_drawdown_pct_smc_filtered"].abs()
    if {"profit_factor_smc_filtered", "profit_factor_baseline"}.issubset(pivot.columns):
        pivot["profit_factor_improvement"] = pivot["profit_factor_smc_filtered"] - pivot["profit_factor_baseline"]

    top_test = pivot[pivot["window"].eq("test")].copy()
    if "return_improvement_pct" in top_test.columns:
        top_test = top_test.sort_values(
            by=["return_pct_smc_filtered", "drawdown_improvement_pct", "profit_factor_improvement"],
            ascending=[False, False, False],
            na_position="last",
        )

    strategy_stats = summary.groupby(["window", "strategy"])[
        ["return_pct", "max_drawdown_pct", "profit_factor", "sharpe_ratio", "trades", "win_rate_pct"]
    ].mean(numeric_only=True).reset_index()

    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>V4 Walk-Forward SMC Validation</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
          h1, h2 {{ color: #1f2937; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 12px; }}
          th, td {{ border: 1px solid #d1d5db; padding: 6px; text-align: right; }}
          th {{ background: #f3f4f6; }}
          td:first-child, th:first-child {{ text-align: left; }}
          .meta {{ color: #4b5563; margin-bottom: 20px; }}
        </style>
      </head>
      <body>
        <h1>V4 Walk-Forward SMC Validation</h1>
        <p class="meta">Model is fixed; no retraining. Windows: train 70%, validation 15%, test 15%.</p>
        <p class="meta">SMC filter features: {", ".join(SMC_SHADOW_FEATURES)}. Lookback candles: {SMC_FILTER_LOOKBACK}.</p>
        <h2>Average Metrics By Window</h2>
        {strategy_stats.to_html(index=False, escape=False)}
        <h2>Top Test Windows</h2>
        {top_test.head(30).to_html(index=False, escape=False)}
        <h2>All Window Results</h2>
        {summary.to_html(index=False, escape=False)}
      </body>
    </html>
    """
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_walk_forward_smc_validation(args: Any | None = None) -> pd.DataFrame:
    if args is None:
        args = type("Args", (), {"timeframe": Config.TIMEFRAME, "capital": 100000.0})()

    timeframe = getattr(args, "timeframe", Config.TIMEFRAME)
    starting_capital = float(getattr(args, "capital", 100000.0))

    if getattr(Config, "DATA_SOURCE", "").upper() == "GMX":
        handler = V4DataHandler()
        handler.refresh_gmx_cache(force=False)

    cache = ModelScalerCache()
    model, scaler = cache.load()
    data_handler = V4DataHandler(str(cache.scaler_path), scaler=scaler)

    symbols = list_gmx_symbols(timeframe)
    if not symbols:
        raise FileNotFoundError(f"No GMX {timeframe} files found in {Config.GMX_OHLC_DIR}")

    rows = []
    for symbol in symbols:
        try:
            rows.extend(run_symbol_walk_forward_smc(model, data_handler, symbol, timeframe, starting_capital))
        except Exception as exc:
            logger.warning("Walk-forward SMC validation failed for %s: %s", symbol, exc)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(WALK_FORWARD_SUMMARY_PATH, index=False)
    _write_walk_forward_html(summary_df, WALK_FORWARD_REPORT_PATH)
    return summary_df


def run_walk_forward(*args, **kwargs):
    return run_walk_forward_smc_validation(*args, **kwargs)
