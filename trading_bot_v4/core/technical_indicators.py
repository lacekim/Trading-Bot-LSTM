"""Technical indicator helpers reusing the original ATR logic from the existing bot."""

from __future__ import annotations

import pandas as pd

from trading_bot_v4.core.data_handler import V4DataHandler


def calculate_atr(df: pd.DataFrame, period: int = 14):
    handler = V4DataHandler()
    return handler.calculate_atr(df, period=period)
