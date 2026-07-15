"""Paper-only SMC filter for V4 model signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols, load_gmx_ohlc
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
from trading_bot_v4.utils.signal_direction import binary_upside_direction
from trading_bot_v4.utils.model_cache import ModelScalerCache


logger = build_logger("v4_paper_smc_filter")

SMC_PAPER_BLOCK_LOG_PATH = Path("logs/v4_smc_blocked_trades.csv")
SMC_PAPER_SUMMARY_PATH = Path("logs/v4_smc_paper_summary.csv")
SMC_CONTEXT_LOOKBACK = 24


def _rolling_active(series: pd.Series, lookback: int = SMC_CONTEXT_LOOKBACK) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric.rolling(window=lookback, min_periods=1).max().gt(0)


def _build_prediction_signals(model: Any, data_handler: V4DataHandler, symbol: str, timeframe: str) -> pd.DataFrame:
    df_raw = load_gmx_ohlc(symbol, timeframe)
    df = data_handler.prepare_features(df_raw)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) <= Config.SEQUENCE_LENGTH:
        raise ValueError(f"Insufficient data for paper SMC evaluation: {symbol} {timeframe}")

    features = df[Config.FEATURE_COLUMNS].values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        scaled = data_handler.normalize_data(features, fit=False)
    except TypeError:
        scaled = data_handler.normalize_data(features)

    seq_len = Config.SEQUENCE_LENGTH
    X = np.array([scaled[i - seq_len:i] for i in range(seq_len, len(scaled))])
    probabilities = model.predict(X, verbose=0).reshape(-1)
    signal_index = df.index[seq_len:]

    signals = pd.DataFrame(index=signal_index)
    signals["model_probability"] = probabilities
    signals["price"] = pd.to_numeric(df.loc[signal_index, "Close"], errors="coerce")
    return signals.dropna()


def _build_smc_context(symbol: str, timeframe: str, signal_index: pd.Index) -> pd.DataFrame:
    smc = load_gmx_ohlcv(symbol, timeframe)
    smc = add_swing_features(smc)
    smc = add_structure_features(smc)
    smc = add_liquidity_sweep_features(smc)
    smc = add_order_block_features(smc)
    smc = add_fvg_features(smc)
    smc = add_market_regime_features(smc)
    smc = smc.reindex(signal_index)

    bullish_score = (
        _rolling_active(smc["bullish_bos"]).astype(int)
        + _rolling_active(smc["bullish_choch"]).astype(int)
        + _rolling_active(smc["bullish_liquidity_sweep"]).astype(int)
        + _rolling_active(smc["bullish_order_block"]).astype(int)
        + _rolling_active(smc["bullish_fvg"]).astype(int)
        + _rolling_active(smc["swing_low"]).astype(int)
        + pd.to_numeric(smc["structure_trend"], errors="coerce").fillna(0).gt(0).astype(int)
    )
    bearish_score = (
        _rolling_active(smc["bearish_bos"]).astype(int)
        + _rolling_active(smc["bearish_choch"]).astype(int)
        + _rolling_active(smc["bearish_liquidity_sweep"]).astype(int)
        + _rolling_active(smc["bearish_order_block"]).astype(int)
        + _rolling_active(smc["bearish_fvg"]).astype(int)
        + _rolling_active(smc["swing_high"]).astype(int)
        + pd.to_numeric(smc["structure_trend"], errors="coerce").fillna(0).lt(0).astype(int)
    )

    context = pd.DataFrame(index=signal_index)
    context["smc_bullish_score"] = bullish_score
    context["smc_bearish_score"] = bearish_score
    context["smc_context"] = "neutral"
    context.loc[bullish_score.gt(bearish_score), "smc_context"] = "bullish"
    context.loc[bearish_score.gt(bullish_score), "smc_context"] = "bearish"
    context.loc[bullish_score.eq(bearish_score) & bullish_score.gt(0), "smc_context"] = "supportive"
    context["smc_reason"] = (
        "context="
        + context["smc_context"]
        + "; bullish_score="
        + context["smc_bullish_score"].astype(str)
        + "; bearish_score="
        + context["smc_bearish_score"].astype(str)
        + f"; lookback={SMC_CONTEXT_LOOKBACK}"
    )
    return context


def _direction_from_probability(probability: float) -> str:
    return binary_upside_direction(probability, Config.MIN_SIGNAL_THRESHOLD)


def _smc_allows(direction: str, context: str) -> bool:
    if direction == "LONG":
        return context in {"bullish", "supportive"}
    if direction == "SHORT":
        return context in {"bearish", "supportive"}
    return True


def _append_blocked_trade_log(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    SMC_PAPER_BLOCK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    blocked = pd.DataFrame(rows)
    blocked.to_csv(
        SMC_PAPER_BLOCK_LOG_PATH,
        mode="a",
        header=not SMC_PAPER_BLOCK_LOG_PATH.exists(),
        index=False,
    )


def _format_signal(timestamp: Any, row: pd.Series | dict[str, Any]) -> str:
    if row is None:
        return ""
    probability = float(row["model_probability"])
    price = float(row["price"])
    direction = str(row["model_direction"])
    context = str(row.get("smc_context", "none"))
    return f"{timestamp} {direction} p={probability:.6f} price={price:.10f} context={context}"


def _evaluate_symbol_paper_smc(
    model: Any,
    data_handler: V4DataHandler,
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    signals = _build_prediction_signals(model, data_handler, symbol, timeframe)
    contexts = _build_smc_context(symbol, timeframe, signals.index)
    paper = signals.join(contexts, how="left").dropna(subset=["price"])
    paper["model_direction"] = paper["model_probability"].apply(_direction_from_probability)
    candidates = paper[paper["model_direction"].isin(["LONG", "SHORT"])].copy()

    allowed_rows = []
    blocked_rows = []
    latest_allowed_signal = ""
    latest_blocked_signal = ""
    for timestamp, row in candidates.iterrows():
        direction = str(row["model_direction"])
        context = str(row["smc_context"])
        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "timeframe": timeframe,
            "model_probability": float(row["model_probability"]),
            "model_direction": direction,
            "smc_reason": str(row["smc_reason"]),
            "price": float(row["price"]),
        }
        if _smc_allows(direction, context):
            allowed_rows.append(record)
            latest_allowed_signal = _format_signal(timestamp, row)
            continue

        blocked_rows.append(record)
        latest_blocked_signal = _format_signal(timestamp, row)
        logger.info(
            "SMC paper trade blocked timestamp=%s probability=%.6f direction=%s smc_reason=%s price=%.10f",
            timestamp,
            record["model_probability"],
            direction,
            record["smc_reason"],
            record["price"],
        )

    candidates_count = int(len(candidates))
    blocked_count = int(len(blocked_rows))
    block_rate = blocked_count / candidates_count if candidates_count else 0.0
    latest = paper.iloc[-1].to_dict() if not paper.empty else {}
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "predictions": int(len(paper)),
        "paper_candidates": candidates_count,
        "candidates": candidates_count,
        "allowed": int(len(allowed_rows)),
        "blocked": blocked_count,
        "block_rate": block_rate,
        "latest_allowed_signal": latest_allowed_signal,
        "latest_blocked_signal": latest_blocked_signal,
        "blocked_rows": blocked_rows,
        "latest": latest,
    }


def _write_summary(results: list[dict[str, Any]]) -> pd.DataFrame:
    summary_rows = [
        {
            "symbol": result["symbol"],
            "timeframe": result["timeframe"],
            "candidates": result["candidates"],
            "allowed": result["allowed"],
            "blocked": result["blocked"],
            "block_rate": result["block_rate"],
            "latest_allowed_signal": result["latest_allowed_signal"],
            "latest_blocked_signal": result["latest_blocked_signal"],
        }
        for result in results
    ]
    summary = pd.DataFrame(summary_rows)
    SMC_PAPER_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SMC_PAPER_SUMMARY_PATH, index=False)
    return summary


def run_paper_trade_smc_filter(args: Any) -> dict[str, Any]:
    """Evaluate paper-only model signals with the optional SMC filter enabled."""
    symbol = str(getattr(args, "symbol", Config.GMX_SYMBOL)).upper()
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    use_all_assets = bool(getattr(args, "all_assets", False))

    Config.USE_SMC_FILTER = True
    cache = ModelScalerCache()
    model, scaler = cache.load()
    data_handler = V4DataHandler(str(cache.scaler_path), scaler=scaler)

    if use_all_assets:
        symbols = [str(item).upper() for item in list_gmx_symbols(timeframe)]
        results = []
        all_blocked_rows = []
        for asset in symbols:
            try:
                result = _evaluate_symbol_paper_smc(model, data_handler, asset, timeframe)
            except Exception as exc:
                logger.warning("Skipping %s %s during paper SMC evaluation: %s", asset, timeframe, exc)
                continue
            results.append(result)
            all_blocked_rows.extend(result["blocked_rows"])

        _append_blocked_trade_log(all_blocked_rows)
        summary = _write_summary(results)
        return {
            "all_assets": True,
            "timeframe": timeframe,
            "assets": int(len(results)),
            "predictions": int(sum(result["predictions"] for result in results)),
            "paper_candidates": int(sum(result["paper_candidates"] for result in results)),
            "candidates": int(sum(result["candidates"] for result in results)),
            "allowed": int(sum(result["allowed"] for result in results)),
            "blocked": int(sum(result["blocked"] for result in results)),
            "blocked_log_path": SMC_PAPER_BLOCK_LOG_PATH,
            "summary_path": SMC_PAPER_SUMMARY_PATH,
            "summary_df": summary,
        }

    result = _evaluate_symbol_paper_smc(model, data_handler, symbol, timeframe)
    _append_blocked_trade_log(result["blocked_rows"])
    result["blocked_log_path"] = SMC_PAPER_BLOCK_LOG_PATH
    result["all_assets"] = False
    return result
