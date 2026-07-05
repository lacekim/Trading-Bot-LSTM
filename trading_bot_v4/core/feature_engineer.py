"""Feature engineering entry points that preserve the original 18-feature pipeline."""

from __future__ import annotations

import pandas as pd

from trading_bot_v4.core.data_handler import V4DataHandler


def prepare_features(df: pd.DataFrame, prediction_horizon: int = 1, logger=None):
    handler = V4DataHandler(logger_obj=logger)
    return handler.prepare_features(df, prediction_horizon=prediction_horizon)


def prepare_sequences(data, target, seq_length: int):
    handler = V4DataHandler()
    return handler.prepare_sequences(data, target, seq_length)


def build_feature_frame(df: pd.DataFrame, prediction_horizon: int = 1, logger=None):
    return prepare_features(df, prediction_horizon=prediction_horizon, logger=logger)
