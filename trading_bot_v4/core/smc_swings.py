"""Standalone SMC swing-high and swing-low features for V4 analysis."""

from __future__ import annotations

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


def add_swing_features(df: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    """Add swing high/low features without changing the original feature pipeline."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    result = df.copy()
    for column in OHLCV_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    swing_high = pd.Series(True, index=result.index)
    swing_low = pd.Series(True, index=result.index)
    for offset in range(1, lookback + 1):
        swing_high &= result["High"].gt(result["High"].shift(offset))
        swing_high &= result["High"].gt(result["High"].shift(-offset))
        swing_low &= result["Low"].lt(result["Low"].shift(offset))
        swing_low &= result["Low"].lt(result["Low"].shift(-offset))

    result["swing_high"] = swing_high.fillna(False).astype(int)
    result["swing_low"] = swing_low.fillna(False).astype(int)

    swing_high_price = result["High"].where(result["swing_high"].eq(1))
    swing_low_price = result["Low"].where(result["swing_low"].eq(1))

    result["last_swing_high"] = swing_high_price.ffill()
    result["last_swing_low"] = swing_low_price.ffill()
    result["distance_to_swing_high"] = result["last_swing_high"] - result["Close"]
    result["distance_to_swing_low"] = result["Close"] - result["last_swing_low"]

    return result


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


def analyze_gmx_smc_swings(symbol: str, timeframe: str, output_path: str | Path | None = None) -> Path:
    """Generate the first V4 SMC feature file for a GMX symbol/timeframe."""
    normalized_symbol = symbol.upper()
    df = load_gmx_ohlcv(normalized_symbol, timeframe)
    features = add_swing_features(df)

    output = Path(output_path or f"v4_smc_features_{normalized_symbol}_{timeframe}.csv")
    features.to_csv(output, index_label="Date", float_format="%.10f")
    return output
