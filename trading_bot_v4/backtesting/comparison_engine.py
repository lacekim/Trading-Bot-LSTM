"""Comparison mode for the original bot versus the V4 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import load_gmx_ohlc, run_lstm_backtest_for_symbol as legacy_run_lstm_backtest_for_symbol
from trading_bot_v4.backtesting.ranking_engine import run_symbol_v4_backtest_ranking
from trading_bot_v4.backtesting.reporting import write_v4_backtest_html_report
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler, build_legacy_data_handler
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache

logger = build_logger("v4_comparison")


def _prepare_prediction_frame(handler: Any, raw_df: pd.DataFrame) -> pd.DataFrame:
    prepared = handler.prepare_features(raw_df.copy())
    prepared["ATR"] = handler.calculate_atr(raw_df, Config.ATR_PERIOD)
    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna()
    return prepared


def _build_prediction_probabilities(model: Any, handler: Any, raw_df: pd.DataFrame, scaler: Any) -> np.ndarray:
    handler.scaler = scaler
    prepared = _prepare_prediction_frame(handler, raw_df)
    if len(prepared) <= Config.SEQUENCE_LENGTH:
        raise ValueError(f"Insufficient data for prediction: {len(prepared)} rows")

    feature_values = prepared[Config.FEATURE_COLUMNS].values
    feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        scaled = handler.normalize_data(feature_values, fit=False)
    except TypeError:
        scaled = handler.normalize_data(feature_values)
    seq_len = Config.SEQUENCE_LENGTH
    X = np.array([scaled[i - seq_len:i] for i in range(seq_len, len(scaled))])
    return model.predict(X, verbose=0).reshape(-1)


def _normalize_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_trade_metrics(trades: list[dict[str, Any]] | pd.DataFrame | None, starting_capital: float, final_capital: float, return_pct: float = 0.0, max_drawdown_pct: float = 0.0) -> dict[str, float]:
    if trades is None:
        trades = []
    if isinstance(trades, pd.DataFrame):
        trade_rows = trades.to_dict("records")
    else:
        trade_rows = list(trades)

    trade_returns = []
    for trade in trade_rows:
        if isinstance(trade, dict):
            entry_price = trade.get("entry_price")
            exit_price = trade.get("exit_price")
            direction = str(trade.get("direction", "LONG")).upper()
            if entry_price is not None and exit_price is not None:
                entry_price = float(entry_price)
                exit_price = float(exit_price)
                if direction == "LONG":
                    trade_returns.append(((exit_price / entry_price) - 1) * 100.0)
                else:
                    trade_returns.append(((entry_price / exit_price) - 1) * 100.0)
            else:
                trade_returns.append(float(trade.get("profit", 0.0)) / max(float(starting_capital), 1.0) * 100.0)

    if not trade_returns:
        trade_returns = [0.0]

    arr = np.array(trade_returns, dtype=float)
    mean_return = float(arr.mean())
    std_return = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0
    downside = arr[arr < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 1 else 0.0
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    win_rate = len(wins) / len(arr) if len(arr) else 0.0

    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
    sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss) if len(arr) else 0.0
    calmar_ratio = (return_pct / max(abs(float(max_drawdown_pct)), 1e-9)) if max_drawdown_pct != 0 else 0.0

    return {
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "average_trade_return": mean_return,
        "expectancy": expectancy,
        "calmar_ratio": calmar_ratio,
        "win_rate_pct": win_rate * 100.0,
    }


def run_v4_asset_comparison(symbol: str, timeframe: str, starting_capital: float, model_cache: ModelScalerCache | None = None) -> pd.DataFrame:
    if model_cache is None:
        model_cache = ModelScalerCache()

    model, scaler = model_cache.load()

    legacy_handler = build_legacy_data_handler(str(model_cache.scaler_path), scaler=scaler)
    v4_handler = V4DataHandler(str(model_cache.scaler_path), scaler=scaler)

    raw_df = load_gmx_ohlc(symbol, timeframe)
    legacy_probs = _build_prediction_probabilities(model, legacy_handler, raw_df, scaler)
    v4_probs = _build_prediction_probabilities(model, v4_handler, raw_df, scaler)

    shared_length = min(len(legacy_probs), len(v4_probs))
    if shared_length > 0:
        legacy_probs = legacy_probs[:shared_length]
        v4_probs = v4_probs[:shared_length]

    prediction_match = bool(np.allclose(legacy_probs, v4_probs, atol=1e-6, rtol=1e-6))
    max_prediction_diff = float(np.max(np.abs(legacy_probs - v4_probs))) if len(legacy_probs) else 0.0

    legacy_summary, legacy_trades = legacy_run_lstm_backtest_for_symbol(model, legacy_handler, symbol, timeframe, starting_capital)
    v4_summary, v4_trades = run_symbol_v4_backtest_ranking(
        model=model,
        data_handler=v4_handler,
        symbol=symbol,
        timeframe=timeframe,
        starting_capital=starting_capital,
    )

    legacy_metrics = build_trade_metrics(
        legacy_trades,
        starting_capital=starting_capital,
        final_capital=legacy_summary.get("final_capital", starting_capital),
        return_pct=legacy_summary.get("return_pct", 0.0),
        max_drawdown_pct=legacy_summary.get("max_drawdown_pct", 0.0),
    )
    v4_metrics = build_trade_metrics(
        v4_trades,
        starting_capital=starting_capital,
        final_capital=v4_summary.get("final_capital", starting_capital),
        return_pct=v4_summary.get("return_pct", 0.0),
        max_drawdown_pct=v4_summary.get("max_drawdown_pct", 0.0),
    )

    report = pd.DataFrame([
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "starting_capital": starting_capital,
            "legacy_return_pct": legacy_summary.get("return_pct", 0.0),
            "v4_return_pct": v4_summary.get("return_pct", 0.0),
            "return_pct_match": bool(np.isclose(legacy_summary.get("return_pct", 0.0), v4_summary.get("return_pct", 0.0), atol=1e-6, rtol=1e-6)),
            "legacy_final_capital": legacy_summary.get("final_capital", starting_capital),
            "v4_final_capital": v4_summary.get("final_capital", starting_capital),
            "final_capital_match": bool(np.isclose(legacy_summary.get("final_capital", starting_capital), v4_summary.get("final_capital", starting_capital), atol=1e-6, rtol=1e-6)),
            "legacy_signals_traded": legacy_summary.get("signals_traded", 0),
            "v4_signals_traded": v4_summary.get("signals_traded", 0),
            "signals_traded_match": int(legacy_summary.get("signals_traded", 0)) == int(v4_summary.get("signals_traded", 0)),
            "legacy_win_rate_pct": legacy_summary.get("win_rate_pct", 0.0),
            "v4_win_rate_pct": v4_summary.get("win_rate_pct", 0.0),
            "win_rate_pct_match": bool(np.isclose(legacy_summary.get("win_rate_pct", 0.0), v4_summary.get("win_rate_pct", 0.0), atol=1e-6, rtol=1e-6)),
            "legacy_max_drawdown_pct": legacy_summary.get("max_drawdown_pct", 0.0),
            "v4_max_drawdown_pct": v4_summary.get("max_drawdown_pct", 0.0),
            "max_drawdown_pct_match": bool(np.isclose(legacy_summary.get("max_drawdown_pct", 0.0), v4_summary.get("max_drawdown_pct", 0.0), atol=1e-6, rtol=1e-6)),
            "legacy_profit_factor": legacy_metrics["profit_factor"],
            "v4_profit_factor": v4_metrics["profit_factor"],
            "legacy_sharpe_ratio": legacy_metrics["sharpe_ratio"],
            "v4_sharpe_ratio": v4_metrics["sharpe_ratio"],
            "legacy_sortino_ratio": legacy_metrics["sortino_ratio"],
            "v4_sortino_ratio": v4_metrics["sortino_ratio"],
            "legacy_average_trade_return": legacy_metrics["average_trade_return"],
            "v4_average_trade_return": v4_metrics["average_trade_return"],
            "legacy_expectancy": legacy_metrics["expectancy"],
            "v4_expectancy": v4_metrics["expectancy"],
            "legacy_calmar_ratio": legacy_metrics["calmar_ratio"],
            "v4_calmar_ratio": v4_metrics["calmar_ratio"],
            "prediction_match": prediction_match,
            "max_prediction_diff": max_prediction_diff,
            "backtest_stats_match": bool(
                np.isclose(legacy_summary.get("return_pct", 0.0), v4_summary.get("return_pct", 0.0), atol=1e-6, rtol=1e-6)
                and np.isclose(legacy_summary.get("final_capital", starting_capital), v4_summary.get("final_capital", starting_capital), atol=1e-6, rtol=1e-6)
                and int(legacy_summary.get("signals_traded", 0)) == int(v4_summary.get("signals_traded", 0))
                and np.isclose(legacy_summary.get("win_rate_pct", 0.0), v4_summary.get("win_rate_pct", 0.0), atol=1e-6, rtol=1e-6)
                and np.isclose(legacy_summary.get("max_drawdown_pct", 0.0), v4_summary.get("max_drawdown_pct", 0.0), atol=1e-6, rtol=1e-6)
            ),
        }
    ])

    return report


def run_v4_comparison(args: Any | None = None) -> pd.DataFrame:
    if args is None:
        args = type("Args", (), {"symbol": Config.GMX_SYMBOL, "timeframe": Config.TIMEFRAME, "capital": 100000.0})()

    symbol = getattr(args, "symbol", Config.GMX_SYMBOL)
    timeframe = getattr(args, "timeframe", Config.TIMEFRAME)
    starting_capital = float(getattr(args, "capital", 100000.0))

    cache = ModelScalerCache()
    report = run_v4_asset_comparison(symbol=symbol, timeframe=timeframe, starting_capital=starting_capital, model_cache=cache)

    output_path = Path("v4_comparison_report.csv")
    report.to_csv(output_path, index=False)
    print(f"V4 comparison report saved to {output_path}")
    print(report.to_string(index=False))
    return report


def run_v4_compare_original(args: Any | None = None) -> pd.DataFrame:
    return run_v4_comparison(args)
