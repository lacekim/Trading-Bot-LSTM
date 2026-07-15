"""Standalone SMC shadow backtest experiment for V4 model signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols, load_gmx_ohlc
from trading_bot_v4.backtesting.ranking_engine import _compute_trade_metrics, run_symbol_v4_backtest_ranking
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.core.smc_swings import (
    add_fvg_features,
    add_liquidity_sweep_features,
    add_market_regime_features,
    add_order_block_features,
    add_structure_features,
    add_swing_features,
    load_gmx_ohlcv,
)
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache


logger = build_logger("v4_smc_shadow_backtest")

SMC_SHADOW_FEATURES = [
    "swing_low",
    "ob_size",
    "bullish_order_block",
    "bearish_order_block",
    "swept_swing_high",
]
SMC_FILTER_LOOKBACK = 24
SMC_SHADOW_SUMMARY_PATH = Path("v4_smc_shadow_backtest_summary.csv")


def _rolling_activity(series: pd.Series, lookback: int = SMC_FILTER_LOOKBACK) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric.rolling(window=lookback, min_periods=1).max().gt(0)


def _rolling_notna(series: pd.Series, lookback: int = SMC_FILTER_LOOKBACK) -> pd.Series:
    return series.notna().astype(int).rolling(window=lookback, min_periods=1).max().gt(0)


def _build_smc_feature_frame(symbol: str, timeframe: str) -> pd.DataFrame:
    smc = load_gmx_ohlcv(symbol, timeframe)
    smc = add_swing_features(smc)
    smc = add_structure_features(smc)
    smc = add_liquidity_sweep_features(smc)
    smc = add_order_block_features(smc)
    smc = add_fvg_features(smc)
    smc = add_market_regime_features(smc)
    return smc


def _build_prediction_signals(model: Any, data_handler: Any, symbol: str, timeframe: str) -> tuple[pd.DataFrame, int]:
    df_raw = load_gmx_ohlc(symbol, timeframe)
    df = data_handler.prepare_features(df_raw)
    df["ATR"] = data_handler.calculate_atr(df_raw, Config.ATR_PERIOD)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) <= Config.SEQUENCE_LENGTH + 1:
        raise ValueError(f"Insufficient data for {symbol}: {len(df)} clean candles")

    features = df[Config.FEATURE_COLUMNS].values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        scaled = data_handler.normalize_data(features, fit=False)
    except TypeError:
        scaled = data_handler.normalize_data(features)

    seq_len = Config.SEQUENCE_LENGTH
    X = np.array([scaled[i - seq_len:i] for i in range(seq_len, len(scaled))])
    probs = model.predict(X, verbose=0).reshape(-1)
    signal_index = df.index[seq_len:]

    signals = pd.DataFrame(index=signal_index)
    signals["prob"] = probs
    signals["Close"] = df.loc[signal_index, "Close"]
    signals["High_next"] = df["High"].shift(-1).loc[signal_index]
    signals["Low_next"] = df["Low"].shift(-1).loc[signal_index]
    signals["Close_next"] = df["Close"].shift(-1).loc[signal_index]
    signals["ATR"] = df.loc[signal_index, "ATR"]
    return signals.dropna(), len(df)


def _add_smc_filters(signals: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    smc = _build_smc_feature_frame(symbol, timeframe)
    smc = smc.reindex(signals.index)
    result = signals.copy()

    bullish_ob = _rolling_activity(smc["bullish_order_block"]) & _rolling_activity(smc["ob_size"])
    bearish_ob = _rolling_activity(smc["bearish_order_block"]) & _rolling_activity(smc["ob_size"])
    recent_swing_low = _rolling_activity(smc["swing_low"])
    recent_swept_high = _rolling_notna(smc["swept_swing_high"])

    result["smc_allow_long"] = recent_swing_low | bullish_ob
    result["smc_allow_short"] = bearish_ob | recent_swept_high
    for column in SMC_SHADOW_FEATURES:
        result[column] = smc[column]
    return result


def _summarize_trades(
    symbol: str,
    timeframe: str,
    candles: int,
    predictions: int,
    trades: pd.DataFrame,
    equity: list[tuple[pd.Timestamp, float]],
    starting_capital: float,
    final_capital: float,
) -> dict[str, float | int | str]:
    if trades.empty:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles,
            "predictions": predictions,
            "signals_traded": 0,
            "starting_capital": starting_capital,
            "final_capital": starting_capital,
            "total_profit": 0.0,
            "return_pct": 0.0,
            "win_rate_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "average_trade_return": 0.0,
            "return_to_drawdown_ratio": 0.0,
        }

    equity_df = pd.DataFrame(equity, columns=["timestamp", "equity"]).set_index("timestamp")
    drawdown = (equity_df["equity"] / equity_df["equity"].cummax() - 1).min() * 100 if not equity_df.empty else 0.0
    wins = trades[trades["profit"] > 0]
    metrics = _compute_trade_metrics(trades, starting_capital, final_capital, drawdown)
    return_pct = ((final_capital / starting_capital) - 1) * 100 if starting_capital else 0.0
    return_to_drawdown_ratio = return_pct / abs(drawdown) if drawdown < 0 else (float("inf") if return_pct > 0 else 0.0)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "predictions": predictions,
        "signals_traded": len(trades),
        "starting_capital": starting_capital,
        "final_capital": final_capital,
        "total_profit": final_capital - starting_capital,
        "return_pct": return_pct,
        "win_rate_pct": (len(wins) / len(trades) * 100) if len(trades) else 0.0,
        "max_drawdown_pct": drawdown,
        "profit_factor": metrics["profit_factor"],
        "average_trade_return": metrics["average_trade_return"],
        "return_to_drawdown_ratio": return_to_drawdown_ratio,
    }


def _run_filtered_signal_backtest(
    signals: pd.DataFrame,
    symbol: str,
    timeframe: str,
    candles: int,
    starting_capital: float,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    capital = float(starting_capital)
    trade_log = []
    equity = []
    threshold = Config.MIN_SIGNAL_THRESHOLD

    for timestamp, row in signals.iterrows():
        probability = float(row["prob"])
        entry_price = float(row["Close"])
        atr = float(row["ATR"])

        if probability > threshold:
            if not bool(row["smc_allow_long"]):
                equity.append((timestamp, capital))
                continue
            direction = "LONG"
            stop_loss = entry_price - atr * Config.ATR_SL_MULTIPLIER
            take_profit = entry_price + atr * Config.ATR_TP_MULTIPLIER
        else:
            equity.append((timestamp, capital))
            continue

        stop_distance = abs(entry_price - stop_loss)
        risk_amount = capital * (Config.RISK_PERCENTAGE / 100)
        risk_size = risk_amount / stop_distance if stop_distance else 0
        max_affordable = capital / entry_price
        units = min(risk_size, max_affordable * 0.95)
        if units <= 0:
            equity.append((timestamp, capital))
            continue

        high_next = float(row["High_next"])
        low_next = float(row["Low_next"])
        close_next = float(row["Close_next"])

        if direction == "LONG":
            if low_next <= stop_loss:
                exit_price = stop_loss
                reason = "Stop Loss"
            elif high_next >= take_profit:
                exit_price = take_profit
                reason = "Take Profit"
            else:
                exit_price = close_next
                reason = "Next Candle Close"
            profit = (exit_price - entry_price) * units
        else:
            if high_next >= stop_loss:
                exit_price = stop_loss
                reason = "Stop Loss"
            elif low_next <= take_profit:
                exit_price = take_profit
                reason = "Take Profit"
            else:
                exit_price = close_next
                reason = "Next Candle Close"
            profit = (entry_price - exit_price) * units

        capital += profit
        trade_log.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "direction": direction,
                "probability": probability,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "units": units,
                "profit": profit,
                "capital": capital,
                "reason": reason,
                "smc_filter": True,
            }
        )
        equity.append((timestamp, capital))

    trades = pd.DataFrame(trade_log)
    summary = _summarize_trades(
        symbol=symbol,
        timeframe=timeframe,
        candles=candles,
        predictions=len(signals),
        trades=trades,
        equity=equity,
        starting_capital=starting_capital,
        final_capital=capital,
    )
    return summary, trades


def run_symbol_smc_shadow_backtest(
    model: Any,
    data_handler: Any,
    symbol: str,
    timeframe: str,
    starting_capital: float,
) -> dict[str, Any]:
    baseline_summary, baseline_trades = run_symbol_v4_backtest_ranking(
        model,
        data_handler,
        symbol,
        timeframe,
        starting_capital,
    )
    signals, candles = _build_prediction_signals(model, data_handler, symbol, timeframe)
    filtered_signals = _add_smc_filters(signals, symbol, timeframe)
    filtered_summary, filtered_trades = _run_filtered_signal_backtest(
        filtered_signals,
        symbol,
        timeframe,
        candles,
        starting_capital,
    )
    return {
        "baseline_summary": baseline_summary,
        "filtered_summary": filtered_summary,
        "baseline_trades": baseline_trades,
        "filtered_trades": filtered_trades,
        "smc_filter_features": SMC_SHADOW_FEATURES,
        "smc_filter_lookback": SMC_FILTER_LOOKBACK,
    }


def _build_summary_row(result: dict[str, Any]) -> dict[str, float | int | str]:
    baseline = result["baseline_summary"]
    filtered = result["filtered_summary"]
    baseline_drawdown = float(baseline.get("max_drawdown_pct", 0.0) or 0.0)
    filtered_drawdown = float(filtered.get("max_drawdown_pct", 0.0) or 0.0)
    baseline_profit_factor = float(baseline.get("profit_factor", 0.0) or 0.0)
    filtered_profit_factor = float(filtered.get("profit_factor", 0.0) or 0.0)

    return {
        "symbol": baseline["symbol"],
        "timeframe": baseline["timeframe"],
        "baseline_return_pct": float(baseline.get("return_pct", 0.0) or 0.0),
        "smc_filtered_return_pct": float(filtered.get("return_pct", 0.0) or 0.0),
        "return_improvement_pct": float(filtered.get("return_pct", 0.0) or 0.0) - float(baseline.get("return_pct", 0.0) or 0.0),
        "baseline_trades": int(baseline.get("signals_traded", 0) or 0),
        "smc_filtered_trades": int(filtered.get("signals_traded", 0) or 0),
        "baseline_win_rate_pct": float(baseline.get("win_rate_pct", 0.0) or 0.0),
        "smc_filtered_win_rate_pct": float(filtered.get("win_rate_pct", 0.0) or 0.0),
        "baseline_max_drawdown_pct": baseline_drawdown,
        "smc_filtered_max_drawdown_pct": filtered_drawdown,
        "drawdown_improvement_pct": abs(baseline_drawdown) - abs(filtered_drawdown),
        "baseline_profit_factor": baseline_profit_factor,
        "smc_filtered_profit_factor": filtered_profit_factor,
        "profit_factor_improvement": filtered_profit_factor - baseline_profit_factor,
        "smc_return_drawdown_ratio": float(filtered.get("return_to_drawdown_ratio", 0.0) or 0.0),
    }


def _rank_shadow_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    ranked = summary_df.copy()
    ranking_columns = [
        "smc_filtered_return_pct",
        "drawdown_improvement_pct",
        "profit_factor_improvement",
        "smc_return_drawdown_ratio",
    ]
    for column in ranking_columns:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce").fillna(0.0)

    ranked = ranked.sort_values(
        by=ranking_columns,
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    ranked["ranking"] = ranked.index + 1
    return ranked


def run_all_assets_smc_shadow_backtest(
    model: Any,
    data_handler: Any,
    timeframe: str,
    starting_capital: float,
) -> pd.DataFrame:
    symbols = list_gmx_symbols(timeframe)
    if not symbols:
        raise FileNotFoundError(f"No GMX {timeframe} files found in {Config.GMX_OHLC_DIR}")

    rows = []
    for symbol in symbols:
        try:
            result = run_symbol_smc_shadow_backtest(model, data_handler, symbol, timeframe, starting_capital)
            rows.append(_build_summary_row(result))
        except Exception as exc:
            logger.warning("SMC shadow backtest failed for %s: %s", symbol, exc)

    ranked = _rank_shadow_summary(pd.DataFrame(rows))
    ranked.to_csv(SMC_SHADOW_SUMMARY_PATH, index=False)
    return ranked


def run_smc_shadow_backtest(args: Any | None = None) -> dict[str, Any]:
    if args is None:
        args = type("Args", (), {"symbol": Config.GMX_SYMBOL, "timeframe": Config.TIMEFRAME, "capital": 100000.0, "all_assets": False})()

    symbol = str(getattr(args, "symbol", Config.GMX_SYMBOL)).upper()
    timeframe = getattr(args, "timeframe", Config.TIMEFRAME)
    starting_capital = float(getattr(args, "capital", 100000.0))
    use_all_assets = bool(getattr(args, "all_assets", False))

    if getattr(Config, "DATA_SOURCE", "").upper() == "GMX":
        handler = V4DataHandler()
        handler.refresh_gmx_cache(force=False)

    cache = ModelScalerCache()
    model, scaler = cache.load()
    data_handler = V4DataHandler(str(cache.scaler_path), scaler=scaler)

    if use_all_assets:
        summary_df = run_all_assets_smc_shadow_backtest(model, data_handler, timeframe, starting_capital)
        return {
            "summary_df": summary_df,
            "summary_path": SMC_SHADOW_SUMMARY_PATH,
            "smc_filter_features": SMC_SHADOW_FEATURES,
            "smc_filter_lookback": SMC_FILTER_LOOKBACK,
        }

    return run_symbol_smc_shadow_backtest(model, data_handler, symbol, timeframe, starting_capital)
