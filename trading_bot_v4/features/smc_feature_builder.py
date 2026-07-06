"""Build optional SMC-enhanced training datasets for V4 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot import load_gmx_ohlc
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.core.smc_swings import (
    SMC_FEATURE_COLUMNS,
    add_fvg_features,
    add_liquidity_sweep_features,
    add_market_regime_features,
    add_order_block_features,
    add_structure_features,
    add_swing_features,
    load_gmx_ohlcv,
)


@dataclass(frozen=True)
class SmcTrainingDataResult:
    symbol: str
    timeframe: str
    original_feature_count: int
    smc_feature_count: int
    total_feature_count: int
    rows_written: int
    output_path: Path
    dataset: pd.DataFrame


def _build_original_training_frame(raw: pd.DataFrame, prediction_horizon: int = 1) -> pd.DataFrame:
    handler = V4DataHandler()
    raw_with_atr = handler.ensure_atr(raw.copy(), Config.ATR_PERIOD)
    prepared = super(V4DataHandler, handler).prepare_features(
        raw_with_atr,
        prediction_horizon=prediction_horizon,
    )

    required = [*Config.FEATURE_COLUMNS, "future_return", "target"]
    missing = [column for column in required if column not in prepared.columns]
    if missing:
        raise ValueError(f"Missing original training columns: {missing}")

    original = prepared[required].copy()
    original.index.name = "timestamp"
    return original


def _build_smc_feature_frame(symbol: str, timeframe: str) -> pd.DataFrame:
    smc = load_gmx_ohlcv(symbol, timeframe)
    smc = add_swing_features(smc)
    smc = add_structure_features(smc)
    smc = add_liquidity_sweep_features(smc)
    smc = add_order_block_features(smc)
    smc = add_fvg_features(smc)
    smc = add_market_regime_features(smc)

    missing = [column for column in SMC_FEATURE_COLUMNS if column not in smc.columns]
    if missing:
        raise ValueError(f"Missing SMC feature columns: {missing}")

    features = smc[SMC_FEATURE_COLUMNS].copy()
    features.index.name = "timestamp"
    return features


def build_smc_training_data(
    symbol: str,
    timeframe: str,
    output_path: str | Path | None = None,
    prediction_horizon: int = 1,
) -> SmcTrainingDataResult:
    """Build one timestamp-aligned dataset with original model features, SMC features, and target."""
    symbol = symbol.upper()
    raw = load_gmx_ohlc(symbol, timeframe)
    original = _build_original_training_frame(raw, prediction_horizon=prediction_horizon)
    smc = _build_smc_feature_frame(symbol, timeframe)

    smc_aligned = smc.reindex(original.index)
    dataset = original.join(smc_aligned, how="inner")

    numeric_columns = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS, "future_return", "target"]
    dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors="coerce")
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset[SMC_FEATURE_COLUMNS] = dataset[SMC_FEATURE_COLUMNS].ffill().fillna(0.0)
    dataset = dataset.dropna(subset=[*Config.FEATURE_COLUMNS, "future_return", "target"])
    dataset["target"] = dataset["target"].astype(int)
    dataset = dataset[numeric_columns]

    path = Path(output_path) if output_path is not None else Path("models") / f"training_data_smc_{symbol}_{timeframe}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=True)

    original_feature_count = len(Config.FEATURE_COLUMNS)
    smc_feature_count = len(SMC_FEATURE_COLUMNS)
    return SmcTrainingDataResult(
        symbol=symbol,
        timeframe=timeframe,
        original_feature_count=original_feature_count,
        smc_feature_count=smc_feature_count,
        total_feature_count=original_feature_count + smc_feature_count,
        rows_written=int(len(dataset)),
        output_path=path,
        dataset=dataset,
    )
