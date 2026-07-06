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
STRUCTURE_COLUMNS = [
    "bullish_bos",
    "bearish_bos",
    "bullish_choch",
    "bearish_choch",
    "structure_trend",
]
FVG_COLUMNS = [
    "bullish_fvg",
    "bearish_fvg",
    "fvg_top",
    "fvg_bottom",
    "fvg_size",
    "fvg_midpoint",
    "fvg_filled",
    "distance_to_nearest_fvg",
]
ORDER_BLOCK_COLUMNS = [
    "bullish_order_block",
    "bearish_order_block",
    "ob_top",
    "ob_bottom",
    "ob_midpoint",
    "ob_size",
    "ob_mitigated",
    "distance_to_nearest_ob",
]
LIQUIDITY_SWEEP_COLUMNS = [
    "bullish_liquidity_sweep",
    "bearish_liquidity_sweep",
    "swept_swing_high",
    "swept_swing_low",
    "sweep_distance",
    "bars_since_liquidity_sweep",
]
REGIME_COLUMNS = [
    "regime_trending",
    "regime_ranging",
    "regime_high_volatility",
    "regime_low_volatility",
    "regime_score",
]
SMC_FEATURE_COLUMNS = [
    *SWING_COLUMNS,
    *STRUCTURE_COLUMNS,
    *LIQUIDITY_SWEEP_COLUMNS,
    *ORDER_BLOCK_COLUMNS,
    *FVG_COLUMNS,
    *REGIME_COLUMNS,
]


@dataclass(frozen=True)
class SwingValidationSummary:
    total_swing_highs: int
    total_swing_lows: int
    both_swing_high_and_low: int
    swing_candle_percentage: float
    total_bullish_bos: int
    total_bearish_bos: int
    total_bullish_choch: int
    total_bearish_choch: int
    current_structure_trend: int
    latest_structure_signal: str
    total_bullish_fvg: int
    total_bearish_fvg: int
    open_fvgs: int
    filled_fvgs: int
    nearest_fvg_distance: float | None
    total_bullish_order_blocks: int
    total_bearish_order_blocks: int
    open_order_blocks: int
    mitigated_order_blocks: int
    nearest_ob_distance: float | None
    total_bullish_liquidity_sweeps: int
    total_bearish_liquidity_sweeps: int
    latest_liquidity_sweep_type: str
    bars_since_latest_liquidity_sweep: int | None
    current_regime: str
    trending_candles_count: int
    ranging_candles_count: int
    high_volatility_count: int
    low_volatility_count: int


@dataclass(frozen=True)
class SmcDiagnosticResult:
    output_path: Path
    top_correlations: list[dict[str, float | int | str]]


@dataclass(frozen=True)
class SmcFeatureRankingResult:
    output_path: Path
    top_features: list[dict[str, float | int | str]]


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


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BOS/CHOCH structure labels from confirmed swing features."""
    required = ["Close", *SWING_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required SMC feature columns: {missing}")

    result = df.copy()
    for column in ["Close", "last_swing_high", "last_swing_low"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    prior_swing_high = result["last_swing_high"].shift(1)
    prior_swing_low = result["last_swing_low"].shift(1)
    trend = 0
    last_broken_high: float | None = None
    last_broken_low: float | None = None

    for column in STRUCTURE_COLUMNS:
        result[column] = 0

    for index, close in result["Close"].items():
        if pd.isna(close):
            result.loc[index, "structure_trend"] = trend
            continue

        high_level = prior_swing_high.loc[index]
        low_level = prior_swing_low.loc[index]
        break_high = (
            pd.notna(high_level)
            and float(close) > float(high_level)
            and (last_broken_high is None or float(high_level) != last_broken_high)
        )
        break_low = (
            pd.notna(low_level)
            and float(close) < float(low_level)
            and (last_broken_low is None or float(low_level) != last_broken_low)
        )

        if break_high:
            result.loc[index, "bullish_bos"] = 1
            if trend == -1:
                result.loc[index, "bullish_choch"] = 1
            trend = 1
            last_broken_high = float(high_level)

        if break_low:
            result.loc[index, "bearish_bos"] = 1
            if trend == 1:
                result.loc[index, "bearish_choch"] = 1
            trend = -1
            last_broken_low = float(low_level)

        result.loc[index, "structure_trend"] = trend

    return result


def _distance_to_gap(close: float, bottom: float, top: float) -> float:
    if bottom <= close <= top:
        return 0.0
    if close < bottom:
        return bottom - close
    return close - top


def add_fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Fair Value Gap labels and open-gap distance features."""
    missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    result = df.copy()
    for column in ["High", "Low", "Close"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in FVG_COLUMNS:
        result[column] = 0
    for column in ["fvg_top", "fvg_bottom", "fvg_size", "fvg_midpoint", "distance_to_nearest_fvg"]:
        result[column] = pd.NA

    open_gaps: list[dict[str, float | pd.Timestamp | str]] = []

    for position, (index, row) in enumerate(result.iterrows()):
        high = row["High"]
        low = row["Low"]
        close = row["Close"]
        if pd.isna(high) or pd.isna(low) or pd.isna(close):
            continue

        still_open = []
        for gap in open_gaps:
            direction = gap["direction"]
            top = float(gap["top"])
            bottom = float(gap["bottom"])
            created_at = gap["created_at"]

            filled = (direction == "bullish" and float(low) <= bottom) or (
                direction == "bearish" and float(high) >= top
            )
            if filled:
                result.loc[created_at, "fvg_filled"] = 1
            else:
                still_open.append(gap)
        open_gaps = still_open

        if position >= 2:
            high_two_back = result["High"].iloc[position - 2]
            low_two_back = result["Low"].iloc[position - 2]

            if pd.notna(high_two_back) and float(low) > float(high_two_back):
                bottom = float(high_two_back)
                top = float(low)
                result.loc[index, "bullish_fvg"] = 1
                result.loc[index, "fvg_top"] = top
                result.loc[index, "fvg_bottom"] = bottom
                result.loc[index, "fvg_size"] = top - bottom
                result.loc[index, "fvg_midpoint"] = (top + bottom) / 2.0
                open_gaps.append(
                    {
                        "direction": "bullish",
                        "top": top,
                        "bottom": bottom,
                        "created_at": index,
                    }
                )

            if pd.notna(low_two_back) and float(high) < float(low_two_back):
                bottom = float(high)
                top = float(low_two_back)
                result.loc[index, "bearish_fvg"] = 1
                result.loc[index, "fvg_top"] = top
                result.loc[index, "fvg_bottom"] = bottom
                result.loc[index, "fvg_size"] = top - bottom
                result.loc[index, "fvg_midpoint"] = (top + bottom) / 2.0
                open_gaps.append(
                    {
                        "direction": "bearish",
                        "top": top,
                        "bottom": bottom,
                        "created_at": index,
                    }
                )

        if open_gaps:
            distances = [
                _distance_to_gap(float(close), float(gap["bottom"]), float(gap["top"]))
                for gap in open_gaps
            ]
            result.loc[index, "distance_to_nearest_fvg"] = min(distances)

    result["bullish_fvg"] = result["bullish_fvg"].astype(int)
    result["bearish_fvg"] = result["bearish_fvg"].astype(int)
    result["fvg_filled"] = result["fvg_filled"].astype(int)
    return result


def add_order_block_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add order block labels created by BOS events and track later mitigation."""
    required = ["Open", "High", "Low", "Close", "bullish_bos", "bearish_bos"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required order block columns: {missing}")

    result = df.copy()
    for column in ["Open", "High", "Low", "Close"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in ORDER_BLOCK_COLUMNS:
        result[column] = 0
    for column in ["ob_top", "ob_bottom", "ob_midpoint", "ob_size", "distance_to_nearest_ob"]:
        result[column] = pd.NA

    open_order_blocks: list[dict[str, float | pd.Timestamp | str]] = []
    index_list = list(result.index)
    used_ob_indexes: set[pd.Timestamp] = set()

    for position, (index, row) in enumerate(result.iterrows()):
        high = row["High"]
        low = row["Low"]
        close = row["Close"]
        if pd.isna(high) or pd.isna(low) or pd.isna(close):
            continue

        still_open = []
        for order_block in open_order_blocks:
            top = float(order_block["top"])
            bottom = float(order_block["bottom"])
            created_at = order_block["created_at"]
            mitigated = float(low) <= top and float(high) >= bottom
            if mitigated:
                result.loc[created_at, "ob_mitigated"] = 1
            else:
                still_open.append(order_block)
        open_order_blocks = still_open

        if row["bullish_bos"] == 1:
            ob_index = None
            for candidate_position in range(position - 1, -1, -1):
                candidate_index = index_list[candidate_position]
                candidate = result.loc[candidate_index]
                if candidate_index in used_ob_indexes:
                    continue
                if candidate["Close"] < candidate["Open"]:
                    ob_index = candidate_index
                    break

            if ob_index is not None:
                ob_bottom = float(result.loc[ob_index, "Low"])
                ob_top = float(result.loc[ob_index, "High"])
                result.loc[ob_index, "bullish_order_block"] = 1
                result.loc[ob_index, "ob_top"] = ob_top
                result.loc[ob_index, "ob_bottom"] = ob_bottom
                result.loc[ob_index, "ob_midpoint"] = (ob_top + ob_bottom) / 2.0
                result.loc[ob_index, "ob_size"] = ob_top - ob_bottom
                used_ob_indexes.add(ob_index)
                open_order_blocks.append(
                    {
                        "direction": "bullish",
                        "top": ob_top,
                        "bottom": ob_bottom,
                        "created_at": ob_index,
                    }
                )

        if row["bearish_bos"] == 1:
            ob_index = None
            for candidate_position in range(position - 1, -1, -1):
                candidate_index = index_list[candidate_position]
                candidate = result.loc[candidate_index]
                if candidate_index in used_ob_indexes:
                    continue
                if candidate["Close"] > candidate["Open"]:
                    ob_index = candidate_index
                    break

            if ob_index is not None:
                ob_bottom = float(result.loc[ob_index, "Low"])
                ob_top = float(result.loc[ob_index, "High"])
                result.loc[ob_index, "bearish_order_block"] = 1
                result.loc[ob_index, "ob_top"] = ob_top
                result.loc[ob_index, "ob_bottom"] = ob_bottom
                result.loc[ob_index, "ob_midpoint"] = (ob_top + ob_bottom) / 2.0
                result.loc[ob_index, "ob_size"] = ob_top - ob_bottom
                used_ob_indexes.add(ob_index)
                open_order_blocks.append(
                    {
                        "direction": "bearish",
                        "top": ob_top,
                        "bottom": ob_bottom,
                        "created_at": ob_index,
                    }
                )

        if open_order_blocks:
            distances = [
                _distance_to_gap(float(close), float(order_block["bottom"]), float(order_block["top"]))
                for order_block in open_order_blocks
            ]
            result.loc[index, "distance_to_nearest_ob"] = min(distances)

    result["bullish_order_block"] = result["bullish_order_block"].astype(int)
    result["bearish_order_block"] = result["bearish_order_block"].astype(int)
    result["ob_mitigated"] = result["ob_mitigated"].astype(int)
    return result


def add_liquidity_sweep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add liquidity sweep labels from prior confirmed swing highs/lows."""
    required = ["High", "Low", "Close", "last_swing_high", "last_swing_low"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required liquidity sweep columns: {missing}")

    result = df.copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in LIQUIDITY_SWEEP_COLUMNS:
        result[column] = 0
    for column in ["swept_swing_high", "swept_swing_low", "sweep_distance"]:
        result[column] = pd.NA

    prior_swing_high = result["last_swing_high"].shift(1)
    prior_swing_low = result["last_swing_low"].shift(1)
    latest_sweep_position: int | None = None

    for position, (index, row) in enumerate(result.iterrows()):
        high = row["High"]
        low = row["Low"]
        close = row["Close"]

        high_level = prior_swing_high.loc[index]
        low_level = prior_swing_low.loc[index]

        bearish_sweep = pd.notna(high_level) and pd.notna(high) and pd.notna(close)
        bearish_sweep = bearish_sweep and float(high) > float(high_level) and float(close) < float(high_level)
        bullish_sweep = pd.notna(low_level) and pd.notna(low) and pd.notna(close)
        bullish_sweep = bullish_sweep and float(low) < float(low_level) and float(close) > float(low_level)

        if bearish_sweep:
            result.loc[index, "bearish_liquidity_sweep"] = 1
            result.loc[index, "swept_swing_high"] = float(high_level)
            result.loc[index, "sweep_distance"] = float(high) - float(high_level)
            latest_sweep_position = position

        if bullish_sweep:
            result.loc[index, "bullish_liquidity_sweep"] = 1
            result.loc[index, "swept_swing_low"] = float(low_level)
            result.loc[index, "sweep_distance"] = float(low_level) - float(low)
            latest_sweep_position = position

        if latest_sweep_position is not None:
            result.loc[index, "bars_since_liquidity_sweep"] = position - latest_sweep_position

    result["bullish_liquidity_sweep"] = result["bullish_liquidity_sweep"].astype(int)
    result["bearish_liquidity_sweep"] = result["bearish_liquidity_sweep"].astype(int)
    result["bars_since_liquidity_sweep"] = result["bars_since_liquidity_sweep"].astype(int)
    return result


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Classify market regime from ATR, EMA slope, and recent price range."""
    missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    result = df.copy()
    for column in OHLCV_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    atr = calculate_atr(result)
    close = result["Close"]
    atr_pct = atr / close.replace(0, pd.NA)
    ema = close.ewm(span=50, adjust=False).mean()
    ema_slope_pct = (ema - ema.shift(5)) / close.replace(0, pd.NA)
    recent_range_pct = (
        result["High"].rolling(window=20, min_periods=1).max()
        - result["Low"].rolling(window=20, min_periods=1).min()
    ) / close.replace(0, pd.NA)

    trend_threshold = (atr_pct * 0.35).fillna(0)
    range_threshold = (atr_pct * 3.0).fillna(0)
    volatility_baseline = atr_pct.rolling(window=100, min_periods=20).median().bfill().ffill()

    trending = ema_slope_pct.abs().gt(trend_threshold) & recent_range_pct.gt(range_threshold)
    high_volatility = atr_pct.gt(volatility_baseline)
    low_volatility = atr_pct.le(volatility_baseline)

    result["regime_trending"] = trending.fillna(False).astype(int)
    result["regime_ranging"] = (~trending.fillna(False)).astype(int)
    result["regime_high_volatility"] = high_volatility.fillna(False).astype(int)
    result["regime_low_volatility"] = low_volatility.fillna(False).astype(int)

    trend_component = (ema_slope_pct.abs() / atr_pct.replace(0, pd.NA)).clip(upper=5).fillna(0)
    range_component = (recent_range_pct / atr_pct.replace(0, pd.NA)).clip(upper=10).fillna(0)
    result["regime_score"] = trend_component + (range_component / 10.0)
    return result


def summarize_swing_features(df: pd.DataFrame) -> SwingValidationSummary:
    """Summarize standalone SMC swing labels for CLI validation output."""
    missing = [
        column
        for column in [
            "swing_high",
            "swing_low",
            *STRUCTURE_COLUMNS,
            *FVG_COLUMNS,
            *ORDER_BLOCK_COLUMNS,
            *LIQUIDITY_SWEEP_COLUMNS,
            *REGIME_COLUMNS,
        ]
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing swing feature columns: {missing}")

    swing_high = df["swing_high"].eq(1)
    swing_low = df["swing_low"].eq(1)
    total_rows = len(df)
    swing_rows = (swing_high | swing_low).sum()
    swing_percentage = (swing_rows / total_rows * 100.0) if total_rows else 0.0
    signal_columns = ["bullish_choch", "bearish_choch", "bullish_bos", "bearish_bos"]
    signal_rows = df[df[signal_columns].eq(1).any(axis=1)]
    latest_signal = "none"
    if not signal_rows.empty:
        latest_row = signal_rows.iloc[-1]
        for column in signal_columns:
            if latest_row[column] == 1:
                latest_signal = column
                break
    fvg_created = df["bullish_fvg"].eq(1) | df["bearish_fvg"].eq(1)
    filled_fvgs = df.loc[fvg_created, "fvg_filled"].eq(1)
    nearest_fvg_distance = df["distance_to_nearest_fvg"].dropna()
    order_blocks = df["bullish_order_block"].eq(1) | df["bearish_order_block"].eq(1)
    mitigated_order_blocks = df.loc[order_blocks, "ob_mitigated"].eq(1)
    nearest_ob_distance = df["distance_to_nearest_ob"].dropna()
    liquidity_sweeps = df["bullish_liquidity_sweep"].eq(1) | df["bearish_liquidity_sweep"].eq(1)
    latest_liquidity_sweep_type = "none"
    bars_since_latest_liquidity_sweep = None
    if liquidity_sweeps.any():
        latest_sweep_row = df.loc[liquidity_sweeps].iloc[-1]
        if latest_sweep_row["bullish_liquidity_sweep"] == 1:
            latest_liquidity_sweep_type = "bullish"
        elif latest_sweep_row["bearish_liquidity_sweep"] == 1:
            latest_liquidity_sweep_type = "bearish"
        latest_bars_since = df["bars_since_liquidity_sweep"].iloc[-1]
        bars_since_latest_liquidity_sweep = int(latest_bars_since)
    current_regime = "unknown"
    if total_rows:
        latest = df.iloc[-1]
        direction_regime = "trending" if latest["regime_trending"] == 1 else "ranging"
        volatility_regime = "high_volatility" if latest["regime_high_volatility"] == 1 else "low_volatility"
        current_regime = f"{direction_regime}_{volatility_regime}"

    return SwingValidationSummary(
        total_swing_highs=int(swing_high.sum()),
        total_swing_lows=int(swing_low.sum()),
        both_swing_high_and_low=int((swing_high & swing_low).sum()),
        swing_candle_percentage=float(swing_percentage),
        total_bullish_bos=int(df["bullish_bos"].eq(1).sum()),
        total_bearish_bos=int(df["bearish_bos"].eq(1).sum()),
        total_bullish_choch=int(df["bullish_choch"].eq(1).sum()),
        total_bearish_choch=int(df["bearish_choch"].eq(1).sum()),
        current_structure_trend=int(df["structure_trend"].iloc[-1]) if total_rows else 0,
        latest_structure_signal=latest_signal,
        total_bullish_fvg=int(df["bullish_fvg"].eq(1).sum()),
        total_bearish_fvg=int(df["bearish_fvg"].eq(1).sum()),
        open_fvgs=int(fvg_created.sum() - filled_fvgs.sum()),
        filled_fvgs=int(filled_fvgs.sum()),
        nearest_fvg_distance=float(nearest_fvg_distance.iloc[-1]) if not nearest_fvg_distance.empty else None,
        total_bullish_order_blocks=int(df["bullish_order_block"].eq(1).sum()),
        total_bearish_order_blocks=int(df["bearish_order_block"].eq(1).sum()),
        open_order_blocks=int(order_blocks.sum() - mitigated_order_blocks.sum()),
        mitigated_order_blocks=int(mitigated_order_blocks.sum()),
        nearest_ob_distance=float(nearest_ob_distance.iloc[-1]) if not nearest_ob_distance.empty else None,
        total_bullish_liquidity_sweeps=int(df["bullish_liquidity_sweep"].eq(1).sum()),
        total_bearish_liquidity_sweeps=int(df["bearish_liquidity_sweep"].eq(1).sum()),
        latest_liquidity_sweep_type=latest_liquidity_sweep_type,
        bars_since_latest_liquidity_sweep=bars_since_latest_liquidity_sweep,
        current_regime=current_regime,
        trending_candles_count=int(df["regime_trending"].eq(1).sum()),
        ranging_candles_count=int(df["regime_ranging"].eq(1).sum()),
        high_volatility_count=int(df["regime_high_volatility"].eq(1).sum()),
        low_volatility_count=int(df["regime_low_volatility"].eq(1).sum()),
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


def _format_optional_float(value: float | None) -> str:
    return "none" if value is None else f"{value:.10f}"


def build_smc_summary_text(
    symbol: str,
    timeframe: str,
    validation: SwingValidationSummary,
    total_smc_feature_columns: int,
) -> str:
    """Build a compact text summary for standalone SMC analysis."""
    bars_since_sweep = "none"
    if validation.bars_since_latest_liquidity_sweep is not None:
        bars_since_sweep = str(validation.bars_since_latest_liquidity_sweep)

    return "\n".join(
        [
            f"SMC Validation Summary: {symbol.upper()} {timeframe}",
            "",
            "Counts",
            f"total swing highs: {validation.total_swing_highs}",
            f"total swing lows: {validation.total_swing_lows}",
            f"both high/low rows: {validation.both_swing_high_and_low}",
            f"percentage of candles marked as swings: {validation.swing_candle_percentage:.2f}%",
            f"total bullish BOS: {validation.total_bullish_bos}",
            f"total bearish BOS: {validation.total_bearish_bos}",
            f"total bullish CHOCH: {validation.total_bullish_choch}",
            f"total bearish CHOCH: {validation.total_bearish_choch}",
            f"total bullish FVG: {validation.total_bullish_fvg}",
            f"total bearish FVG: {validation.total_bearish_fvg}",
            f"open FVGs: {validation.open_fvgs}",
            f"filled FVGs: {validation.filled_fvgs}",
            f"total bullish order blocks: {validation.total_bullish_order_blocks}",
            f"total bearish order blocks: {validation.total_bearish_order_blocks}",
            f"open order blocks: {validation.open_order_blocks}",
            f"mitigated order blocks: {validation.mitigated_order_blocks}",
            f"total bullish liquidity sweeps: {validation.total_bullish_liquidity_sweeps}",
            f"total bearish liquidity sweeps: {validation.total_bearish_liquidity_sweeps}",
            f"trending candles count: {validation.trending_candles_count}",
            f"ranging candles count: {validation.ranging_candles_count}",
            f"high volatility count: {validation.high_volatility_count}",
            f"low volatility count: {validation.low_volatility_count}",
            "",
            "Latest State",
            f"latest structure trend: {validation.current_structure_trend}",
            f"latest BOS/CHOCH: {validation.latest_structure_signal}",
            f"latest liquidity sweep: {validation.latest_liquidity_sweep_type}",
            f"bars since latest liquidity sweep: {bars_since_sweep}",
            f"current regime: {validation.current_regime}",
            f"latest open FVG distance: {_format_optional_float(validation.nearest_fvg_distance)}",
            f"latest open OB distance: {_format_optional_float(validation.nearest_ob_distance)}",
            f"total generated SMC feature columns: {total_smc_feature_columns}",
            "",
        ]
    )


def build_smc_feature_diagnostics(
    features: pd.DataFrame,
    output_path: str | Path,
    horizons: tuple[int, ...] = (1, 3, 6, 12),
) -> SmcDiagnosticResult:
    """Compare standalone SMC features against future candle returns."""
    missing = [column for column in ["Close", *SMC_FEATURE_COLUMNS] if column not in features.columns]
    if missing:
        raise ValueError(f"Missing columns for SMC diagnostics: {missing}")

    diagnostic_rows = []
    close = pd.to_numeric(features["Close"], errors="coerce")

    for horizon in horizons:
        future_return = close.shift(-horizon) / close - 1.0
        for feature_name in SMC_FEATURE_COLUMNS:
            feature = pd.to_numeric(features[feature_name], errors="coerce")
            pair = pd.concat([feature, future_return], axis=1).dropna()
            pair.columns = ["feature", "future_return"]

            correlation = pd.NA
            if len(pair) >= 2 and pair["feature"].nunique() > 1 and pair["future_return"].nunique() > 1:
                correlation = pair["feature"].corr(pair["future_return"])

            diagnostic_rows.append(
                {
                    "feature": feature_name,
                    "future_return_horizon": horizon,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation) if pd.notna(correlation) else pd.NA,
                    "non_null_rows": int(len(pair)),
                    "feature_mean": float(pair["feature"].mean()) if len(pair) else pd.NA,
                    "future_return_mean": float(pair["future_return"].mean()) if len(pair) else pd.NA,
                }
            )

    diagnostics = pd.DataFrame(diagnostic_rows)
    output = Path(output_path)
    diagnostics.to_csv(output, index=False, float_format="%.10f")
    top = (
        diagnostics.dropna(subset=["abs_correlation"])
        .sort_values("abs_correlation", ascending=False)
        .head(10)
    )
    return SmcDiagnosticResult(output_path=output, top_correlations=top.to_dict("records"))


def _normalize_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").abs()
    max_value = numeric.max(skipna=True)
    if pd.isna(max_value) or max_value == 0:
        return pd.Series(pd.NA, index=series.index, dtype="Float64")
    return numeric / max_value


def build_smc_feature_ranking(
    features: pd.DataFrame,
    output_path: str | Path,
    horizon: int = 1,
) -> SmcFeatureRankingResult:
    """Rank SMC features against next-horizon returns without touching model inputs."""
    missing = [column for column in ["Close", *SMC_FEATURE_COLUMNS] if column not in features.columns]
    if missing:
        raise ValueError(f"Missing columns for SMC feature ranking: {missing}")

    close = pd.to_numeric(features["Close"], errors="coerce")
    target = close.shift(-horizon) / close - 1.0
    raw_features = features[SMC_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    dataset = raw_features.copy()
    dataset["target"] = target
    dataset = dataset.dropna(subset=["target"])
    x_raw = dataset[SMC_FEATURE_COLUMNS].replace([float("inf"), float("-inf")], pd.NA)
    y = pd.to_numeric(dataset["target"], errors="coerce")

    rows = []
    for feature_name in SMC_FEATURE_COLUMNS:
        pair = pd.concat([x_raw[feature_name], y], axis=1).dropna()
        pair.columns = ["feature", "target"]
        pearson = pd.NA
        spearman = pd.NA
        if len(pair) >= 2 and pair["feature"].nunique() > 1 and pair["target"].nunique() > 1:
            pearson = pair["feature"].corr(pair["target"], method="pearson")
            spearman = pair["feature"].corr(pair["target"], method="spearman")
        rows.append(
            {
                "feature": feature_name,
                "Pearson": pearson,
                "Spearman": spearman,
                "Mutual Information": pd.NA,
                "RF importance": pd.NA,
                "XGBoost importance": pd.NA,
            }
        )

    ranking = pd.DataFrame(rows)
    x_model = x_raw.fillna(x_raw.median()).fillna(0.0).astype(float)
    y_model = y.astype(float)

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression

        mi_scores = mutual_info_regression(x_model, y_model, random_state=42)
        rf = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=5)
        rf.fit(x_model, y_model)
        ranking["Mutual Information"] = mi_scores
        ranking["RF importance"] = rf.feature_importances_
    except ImportError:
        pass

    try:
        from xgboost import XGBRegressor

        xgb = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        xgb.fit(x_model, y_model)
        ranking["XGBoost importance"] = xgb.feature_importances_
    except ImportError:
        pass

    score_columns = [
        "Pearson",
        "Spearman",
        "Mutual Information",
        "RF importance",
        "XGBoost importance",
    ]
    normalized_scores = pd.concat(
        [_normalize_score(ranking[column]) for column in score_columns],
        axis=1,
    )
    ranking["combined_score"] = normalized_scores.mean(axis=1, skipna=True)
    ranking = ranking.sort_values("combined_score", ascending=False, na_position="last").reset_index(drop=True)
    ranking["ranking"] = range(1, len(ranking) + 1)

    output = Path(output_path)
    ranking.to_csv(output, index=False, float_format="%.10f")
    return SmcFeatureRankingResult(output_path=output, top_features=ranking.head(20).to_dict("records"))


def analyze_gmx_smc_swings(
    symbol: str,
    timeframe: str,
    output_path: str | Path | None = None,
    summary_output_path: str | Path | None = None,
    diagnostics_output_path: str | Path | None = None,
    ranking_output_path: str | Path | None = None,
    swing_window: int | None = None,
    min_swing_distance_atr: float | None = None,
) -> tuple[Path, Path, SmcDiagnosticResult, SmcFeatureRankingResult, SwingValidationSummary]:
    """Generate the first V4 SMC feature file for a GMX symbol/timeframe."""
    normalized_symbol = symbol.upper()
    df = load_gmx_ohlcv(normalized_symbol, timeframe)
    features = add_swing_features(
        df,
        swing_window=swing_window,
        min_swing_distance_atr=min_swing_distance_atr,
    )
    features = add_structure_features(features)
    features = add_liquidity_sweep_features(features)
    features = add_order_block_features(features)
    features = add_fvg_features(features)
    features = add_market_regime_features(features)
    validation = summarize_swing_features(features)

    output = Path(output_path or f"v4_smc_features_{normalized_symbol}_{timeframe}.csv")
    features.to_csv(output, index_label="Date", float_format="%.10f")

    summary_output = Path(summary_output_path or f"v4_smc_summary_{normalized_symbol}_{timeframe}.txt")
    summary_output.write_text(
        build_smc_summary_text(
            normalized_symbol,
            timeframe,
            validation,
            total_smc_feature_columns=len(SMC_FEATURE_COLUMNS),
        ),
        encoding="utf-8",
    )
    diagnostics = build_smc_feature_diagnostics(
        features,
        diagnostics_output_path or f"v4_smc_feature_diagnostics_{normalized_symbol}_{timeframe}.csv",
    )
    ranking = build_smc_feature_ranking(
        features,
        ranking_output_path or "v4_smc_feature_ranking.csv",
    )
    return output, summary_output, diagnostics, ranking, validation
