"""Inference for the independently trained bearish SMC model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.smc_swings import SMC_FEATURE_COLUMNS
from trading_bot_v4.execution.smc_model_paper import _build_smc_model_feature_frame, _timeframe_delta
from trading_bot_v4.ml.bearish_trainer import (
    BEARISH_CALIBRATION_PATH, BEARISH_MODEL_PATH, BEARISH_SCALER_PATH,
)
from trading_bot_v4.utils.model_cache import ModelScalerCache


BEARISH_SIGNALS_PATH = Path("logs/v5_bearish_model_paper_signals.csv")
BEARISH_SUMMARY_PATH = Path("logs/v5_bearish_model_paper_summary.csv")


def load_bearish_calibration() -> dict[str, Any]:
    if not BEARISH_CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"Bearish calibration not found: {BEARISH_CALIBRATION_PATH}")
    return json.loads(BEARISH_CALIBRATION_PATH.read_text(encoding="utf-8"))


def predict_bearish_signals(model: Any, scaler: Any, symbol: str, timeframe: str,
                            threshold: float) -> pd.DataFrame:
    features = _build_smc_model_feature_frame(symbol, timeframe)
    columns = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS]
    scaled = scaler.transform(features[columns].to_numpy(dtype=np.float32)).astype(np.float32)
    length = Config.SEQUENCE_LENGTH
    if len(scaled) <= length:
        raise ValueError(f"Insufficient bearish model feature rows for {symbol} {timeframe}: {len(scaled)}")
    sequences = np.array([scaled[start:start + length] for start in range(len(scaled) - length)], dtype=np.float32)
    probability = model.predict(sequences, verbose=0).reshape(-1)
    frame = features.iloc[length:].copy()
    closed = pd.to_datetime(frame.index, utc=True) + _timeframe_delta(timeframe) <= pd.Timestamp.now(tz="UTC")
    frame, probability = frame.loc[closed], probability[closed]
    result = pd.DataFrame({
        "timestamp": frame.index, "symbol": symbol.upper(), "timeframe": timeframe,
        "model_probability": probability,
        "model_direction": np.where(probability >= threshold, "SHORT", "HOLD"),
        "price": frame["price"].to_numpy(float), "open": frame["open"].to_numpy(float),
        "high": frame["high"].to_numpy(float), "low": frame["low"].to_numpy(float),
        "close": frame["close"].to_numpy(float),
        "candle_gap_seconds": frame["candle_gap_seconds"].to_numpy(float),
    })
    result["is_trade_candidate"] = result["model_direction"].eq("SHORT")
    result["threshold"] = threshold
    result["feature_count"] = len(columns)
    result["model_side"] = "BEARISH"
    return result


def run_bearish_paper_signals(symbols: list[str], timeframe: str, model: Any | None = None,
                              scaler: Any | None = None) -> dict[str, Any]:
    calibration = load_bearish_calibration()
    if not calibration.get("promoted", False):
        raise RuntimeError("Bearish model failed validation promotion and cannot emit SHORT candidates")
    if model is None or scaler is None:
        model, scaler = ModelScalerCache(BEARISH_MODEL_PATH, BEARISH_SCALER_PATH).load()
    symbol_thresholds = {str(key).upper(): float(value) for key, value in calibration.get("promoted_symbols", {}).items()}
    frames, summary = [], []
    for symbol in symbols:
        if symbol.upper() not in symbol_thresholds:
            continue
        threshold = symbol_thresholds[symbol.upper()]
        signals = predict_bearish_signals(model, scaler, symbol, timeframe, threshold)
        frames.append(signals)
        latest = signals.iloc[-1] if not signals.empty else None
        summary.append({
            "symbol": symbol, "predictions": len(signals),
            "short_candidates": int(signals["is_trade_candidate"].sum()) if not signals.empty else 0,
            "latest_direction": str(latest["model_direction"]) if latest is not None else "",
            "latest_probability": float(latest["model_probability"]) if latest is not None else np.nan,
        })
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    BEARISH_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(BEARISH_SIGNALS_PATH, index=False)
    pd.DataFrame(summary).to_csv(BEARISH_SUMMARY_PATH, index=False)
    return {"signals": combined, "signals_path": BEARISH_SIGNALS_PATH, "summary_path": BEARISH_SUMMARY_PATH}


def combine_directional_signals(long_signals: pd.DataFrame, short_signals: pd.DataFrame) -> pd.DataFrame:
    """Select one direction per symbol/candle using confidence above its calibrated threshold."""
    frames = []
    for source, frame in (("UPSIDE", long_signals), ("DOWNSIDE", short_signals)):
        if frame is None or frame.empty:
            continue
        candidate = frame.copy()
        candidate["model_side"] = source
        candidate["confidence_margin"] = (
            pd.to_numeric(candidate["model_probability"], errors="coerce")
            - pd.to_numeric(candidate["threshold"], errors="coerce")
        )
        frames.append(candidate)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["_candidate"] = combined["model_direction"].isin(["LONG", "SHORT"]).astype(int)
    combined = combined.sort_values(
        ["symbol", "timestamp", "_candidate", "confidence_margin"],
        ascending=[True, True, False, False],
    )
    return combined.drop_duplicates(["symbol", "timestamp"], keep="first").drop(columns="_candidate")
