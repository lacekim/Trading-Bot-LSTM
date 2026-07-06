"""Paper-only SMC filter for V4 model signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import load_gmx_ohlc
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


logger = build_logger("v4_paper_smc_filter")

SMC_PAPER_BLOCK_LOG_PATH = Path("logs/v4_smc_blocked_trades.csv")
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
    threshold = float(Config.MIN_SIGNAL_THRESHOLD)
    if probability > threshold:
        return "LONG"
    if probability < (1 - threshold):
        return "SHORT"
    return "HOLD"


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


def run_paper_trade_smc_filter(args: Any) -> dict[str, Any]:
    """Evaluate paper-only model signals with the optional SMC filter enabled."""
    symbol = str(getattr(args, "symbol", Config.GMX_SYMBOL)).upper()
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))

    Config.USE_SMC_FILTER = True
    cache = ModelScalerCache()
    model, scaler = cache.load()
    data_handler = V4DataHandler(str(cache.scaler_path), scaler=scaler)

    signals = _build_prediction_signals(model, data_handler, symbol, timeframe)
    contexts = _build_smc_context(symbol, timeframe, signals.index)
    paper = signals.join(contexts, how="left").dropna(subset=["price"])
    paper["model_direction"] = paper["model_probability"].apply(_direction_from_probability)
    candidates = paper[paper["model_direction"].isin(["LONG", "SHORT"])].copy()

    allowed_rows = []
    blocked_rows = []
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
            continue

        blocked_rows.append(record)
        logger.info(
            "SMC paper trade blocked timestamp=%s probability=%.6f direction=%s smc_reason=%s price=%.10f",
            timestamp,
            record["model_probability"],
            direction,
            record["smc_reason"],
            record["price"],
        )

    _append_blocked_trade_log(blocked_rows)
    latest = paper.iloc[-1].to_dict() if not paper.empty else {}
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "predictions": int(len(paper)),
        "paper_candidates": int(len(candidates)),
        "allowed": int(len(allowed_rows)),
        "blocked": int(len(blocked_rows)),
        "blocked_log_path": SMC_PAPER_BLOCK_LOG_PATH,
        "latest": latest,
    }
