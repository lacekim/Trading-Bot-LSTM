"""Inference for the per-asset SHORT models (trading_bot_v4/ml/per_asset_trainer.py).

Mirrors paper_model_comparison.py's _predict_original_model_signals feature
build exactly (same 18 Config.FEATURE_COLUMNS, same SEQUENCE_LENGTH, same
auxiliary columns downstream execution needs) since the per-asset LONG and
SHORT models share that feature pipeline -- the only difference is how a high
probability gets labeled (LONG for the long model's "price rises" positive
class, SHORT here for the short model's own "price falls" positive class).

predict_bearish_signals (bearish_model_paper.py) is NOT reusable for this: it
is hardcoded to the old pooled bearish model's 52-feature SMC input, while
per-asset models use the base 18 features only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_bot import load_gmx_ohlc
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.execution.smc_model_paper import _timeframe_delta
from trading_bot_v4.utils.macd_confirmation import macd_components


def predict_per_asset_short_signals(model: Any, scaler: Any, symbol: str, timeframe: str,
                                    threshold: float, closed_only: bool = False) -> pd.DataFrame:
    raw = load_gmx_ohlc(symbol, timeframe)
    handler = V4DataHandler()
    features = handler.prepare_features(raw.copy())
    missing = [column for column in Config.FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Missing per-asset feature columns for {symbol}: {missing}")

    feature_frame = features[Config.FEATURE_COLUMNS].copy()
    feature_frame[Config.FEATURE_COLUMNS] = feature_frame[Config.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=Config.FEATURE_COLUMNS)
    prices = pd.to_numeric(raw["Close"], errors="coerce").reindex(feature_frame.index)
    feature_frame["price"] = prices
    feature_frame["atr"] = handler.calculate_atr(raw, Config.ATR_PERIOD).reindex(feature_frame.index)
    feature_frame = feature_frame.join(macd_components(raw["Close"]).reindex(feature_frame.index))
    feature_frame["macd_histogram_previous"] = feature_frame["macd_histogram"].shift(1)
    feature_frame["price_vs_ma200"] = prices / prices.rolling(200).mean() - 1.0
    for source, target in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")):
        feature_frame[target] = pd.to_numeric(raw[source], errors="coerce").reindex(feature_frame.index)
    timestamps = pd.Series(pd.to_datetime(feature_frame.index), index=feature_frame.index)
    feature_frame["candle_gap_seconds"] = timestamps.diff().dt.total_seconds().fillna(0.0)
    feature_frame = feature_frame.dropna(subset=["price", "open", "high", "low", "close", "atr"])

    seq_len = Config.SEQUENCE_LENGTH
    if len(feature_frame) <= seq_len:
        raise ValueError(f"Insufficient per-asset short feature rows for {symbol} {timeframe}: {len(feature_frame)}")

    values = feature_frame[Config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    scaled = scaler.transform(values).astype(np.float32)
    sequences = np.array([scaled[start:start + seq_len] for start in range(0, len(scaled) - seq_len)], dtype=np.float32)
    probability = model.predict(sequences, verbose=0).reshape(-1)
    frame = feature_frame.iloc[seq_len:].copy()

    if closed_only:
        row_timestamps = pd.to_datetime(frame.index, utc=True)
        closed_mask = (row_timestamps + _timeframe_delta(timeframe) <= pd.Timestamp.now(tz="UTC")).to_numpy()
        frame, probability = frame.loc[closed_mask], probability[closed_mask]

    result = pd.DataFrame({
        "timestamp": frame.index, "symbol": symbol.upper(), "timeframe": timeframe,
        "model_probability": probability,
        "model_direction": np.where(probability >= threshold, "SHORT", "HOLD"),
        "price": frame["price"].to_numpy(float), "open": frame["open"].to_numpy(float),
        "high": frame["high"].to_numpy(float), "low": frame["low"].to_numpy(float),
        "close": frame["close"].to_numpy(float),
        "candle_gap_seconds": frame["candle_gap_seconds"].to_numpy(float),
        "atr": frame["atr"].to_numpy(float),
        "macd_line": frame["macd_line"].to_numpy(float),
        "macd_signal": frame["macd_signal"].to_numpy(float),
        "macd_histogram": frame["macd_histogram"].to_numpy(float),
        "macd_histogram_previous": frame["macd_histogram_previous"].to_numpy(float),
        "price_vs_ma200": frame["price_vs_ma200"].to_numpy(float),
    })
    result["is_trade_candidate"] = result["model_direction"].eq("SHORT")
    result["threshold"] = threshold
    result["feature_count"] = len(Config.FEATURE_COLUMNS)
    result["model_side"] = "PER_ASSET_SHORT"
    return result
