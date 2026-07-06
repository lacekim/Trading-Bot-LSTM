"""Prediction wrapper that runs the full original inference pipeline from the bot."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import build_legacy_data_handler
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_predictor")


def predict_with_v4_model(model_path: str | Path | None = None, scaler_path: str | Path | None = None, symbol: str | None = None):
    model_path = Path(model_path or (Config.MODEL_DIR / Config.MODEL_NAME))
    scaler_path = Path(scaler_path or (Config.MODEL_DIR / Config.SCALER_NAME))

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)

    logger.info(f"Loading scaler from {scaler_path}")
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    resolved_symbol = symbol or (Config.GMX_SYMBOL if getattr(Config, "DATA_SOURCE", "").upper() == "GMX" else Config.SYMBOL)
    logger.info(f"Fetching data for symbol={resolved_symbol} timeframe={Config.TIMEFRAME}")

    data_handler = build_legacy_data_handler(str(scaler_path), scaler=scaler)

    df = data_handler.fetch_data(
        resolved_symbol,
        period="60d",
        interval=Config.TIMEFRAME,
        include_sentiment=True,
    )

    if df is None or df.empty:
        raise ValueError("No OHLC data available for prediction")

    df = data_handler.prepare_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < Config.SEQUENCE_LENGTH:
        raise ValueError(f"Insufficient data after cleaning NaNs: {len(df)}")

    feature_frame = df[Config.FEATURE_COLUMNS].values[-Config.SEQUENCE_LENGTH:]
    feature_frame = np.nan_to_num(feature_frame, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = data_handler.normalize_data(feature_frame)

    X = scaled.reshape(1, Config.SEQUENCE_LENGTH, len(Config.FEATURE_COLUMNS))
    probability = float(model.predict(X, verbose=0)[0][0])

    current_price = float(df["Close"].iloc[-1])
    atr = float(data_handler.calculate_atr(df, Config.ATR_PERIOD).iloc[-1])

    result = {
        "symbol": resolved_symbol,
        "current_price": current_price,
        "probability_up": probability,
        "atr": atr,
        "feature_count": len(Config.FEATURE_COLUMNS),
    }

    print(f"symbol: {result['symbol']}")
    print(f"current price: {result['current_price']:.4f}")
    print(f"probability of upward move: {result['probability_up']:.6f}")
    print(f"ATR: {result['atr']:.6f}")
    print(f"feature count: {result['feature_count']}")

    return result
