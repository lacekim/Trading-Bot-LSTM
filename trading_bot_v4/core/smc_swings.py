"""Standalone SMC swing-high and swing-low features for V4 analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading_bot_v4.config_v4 import V4Config as Config


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
SWING_COLUMNS = [
    "swing_high",
    "swing_low",
    "last_swing_high",
    "last_swing_low",
    "distance_to_swing_high",
    "distance_to_swing_low",
]


@dataclass(frozen=True)
class SwingValidationSummary:
    total_swing_highs: int
    total_swing_lows: int
    both_swing_high_and_low: int
    swing_candle_percentage: float


def calculate_atr(df: pd.DataFrame, period: int | None = None) -> pd.Series:
    """Calculate ATR locally so standalone SMC analysis does not touch the live pipeline."""
    atr_period = period or Config.ATR_PERIOD
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=atr_period, min_periods=1).mean()


def _filter_by_atr_distance(
    candidates: pd.Series,
    prices: pd.Series,
    atr: pd.Series,
    min_swing_distance_atr: float,
) -> pd.Series:
    accepted = pd.Series(False, index=candidates.index)
    previous_swing_price: float | None = None

    for index, is_candidate in candidates.items():
        if not is_candidate:
            continue

        price = prices.loc[index]
        if pd.isna(price):
            continue

        if previous_swing_price is None:
            accepted.loc[index] = True
            previous_swing_price = float(price)
            continue

        threshold = atr.loc[index] * min_swing_distance_atr
        if pd.isna(threshold):
            continue
        if abs(float(price) - previous_swing_price) >= float(threshold):
            accepted.loc[index] = True
            previous_swing_price = float(price)

    return accepted


def add_swing_features(
    df: pd.DataFrame,
    swing_window: int | None = None,
    min_swing_distance_atr: float | None = None,
) -> pd.DataFrame:
    """Add swing high/low features without changing the original feature pipeline."""
    window = Config.SMC_SWING_WINDOW if swing_window is None else swing_window
    min_distance = Config.SMC_MIN_SWING_DISTANCE_ATR if min_swing_distance_atr is None else min_swing_distance_atr

    if window < 1:
        raise ValueError("swing_window must be >= 1")
    if min_distance < 0:
        raise ValueError("min_swing_distance_atr must be >= 0")

    missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    result = df.copy()
    for column in OHLCV_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    rolling_window = (window * 2) + 1
    rolling_high = result["High"].rolling(window=rolling_window, center=True, min_periods=rolling_window).max()
    rolling_low = result["Low"].rolling(window=rolling_window, center=True, min_periods=rolling_window).min()
    high_candidates = result["High"].eq(rolling_high).fillna(False)
    low_candidates = result["Low"].eq(rolling_low).fillna(False)

    atr = calculate_atr(result)
    swing_high = _filter_by_atr_distance(high_candidates, result["High"], atr, min_distance)
    swing_low = _filter_by_atr_distance(low_candidates, result["Low"], atr, min_distance)

    result["swing_high"] = swing_high.astype(int)
    result["swing_low"] = swing_low.astype(int)

    swing_high_price = result["High"].where(result["swing_high"].eq(1))
    swing_low_price = result["Low"].where(result["swing_low"].eq(1))

    result["last_swing_high"] = swing_high_price.ffill()
    result["last_swing_low"] = swing_low_price.ffill()
    result["distance_to_swing_high"] = result["last_swing_high"] - result["Close"]
    result["distance_to_swing_low"] = result["Close"] - result["last_swing_low"]

    return result


def summarize_swing_features(df: pd.DataFrame) -> SwingValidationSummary:
    """Summarize standalone SMC swing labels for CLI validation output."""
    missing = [column for column in ["swing_high", "swing_low"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing swing feature columns: {missing}")

    swing_high = df["swing_high"].eq(1)
    swing_low = df["swing_low"].eq(1)
    total_rows = len(df)
    swing_rows = (swing_high | swing_low).sum()
    swing_percentage = (swing_rows / total_rows * 100.0) if total_rows else 0.0

    return SwingValidationSummary(
        total_swing_highs=int(swing_high.sum()),
        total_swing_lows=int(swing_low.sum()),
        both_swing_high_and_low=int((swing_high & swing_low).sum()),
        swing_candle_percentage=float(swing_percentage),
    )


def load_gmx_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load local GMX OHLCV data for SMC analysis."""
    path = Path(Config.GMX_OHLC_DIR) / f"gmx_arbitrum_{symbol.upper()}_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"GMX OHLCV file not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "open": "Open",
            "open_time.1": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "open_time": "Date",
        }
    )

    for column in ["Date", *OHLCV_COLUMNS]:
        if column in df.columns and isinstance(df[column], pd.DataFrame):
            values = df.loc[:, df.columns == column].bfill(axis=1).iloc[:, 0]
            df = df.loc[:, df.columns != column]
            df[column] = values

    missing = [column for column in ["Date", *OHLCV_COLUMNS] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    for column in OHLCV_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", *OHLCV_COLUMNS])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[OHLCV_COLUMNS]


def analyze_gmx_smc_swings(
    symbol: str,
    timeframe: str,
    output_path: str | Path | None = None,
    swing_window: int | None = None,
    min_swing_distance_atr: float | None = None,
) -> tuple[Path, SwingValidationSummary]:
    """Generate the first V4 SMC feature file for a GMX symbol/timeframe."""
    normalized_symbol = symbol.upper()
    df = load_gmx_ohlcv(normalized_symbol, timeframe)
    features = add_swing_features(
        df,
        swing_window=swing_window,
        min_swing_distance_atr=min_swing_distance_atr,
    )
    validation = summarize_swing_features(features)

    output = Path(output_path or f"v4_smc_features_{normalized_symbol}_{timeframe}.csv")
    features.to_csv(output, index_label="Date", float_format="%.10f")
    return output, validation
