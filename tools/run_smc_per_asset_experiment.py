"""Real per-asset SMC models, replacing the single pooled one.

The SMC model (models/lstm_smc_model.h5) that dominates 72% of the asset-
ranking score that gates "picker -> smart money" progression turned out to
be a single model trained on training_data_smc_all_assets_{timeframe}.csv --
literally every asset pooled into one training set, the same root problem
(one model, not asset-specific) this whole project was built to fix for the
LONG side, just never applied here. bearish_trainer.py inherits the same
pooled sequence generator, so SHORT has the identical issue.

This trains a genuinely separate model per (symbol, direction) instead,
reusing the already-validated GBM + triple-barrier + real-execution
calibration pipeline from tonight rather than the slow pooled CNN/LSTM path
-- GBM has been faster and at least as effective everywhere else tonight.
Feature set: the base 18 + all 39 SMC swing/structure/liquidity-sweep/
order-block/FVG/regime columns. Pilots on a handful of symbols first before
any full sweep, same discipline as every other experiment tonight.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.smc_swings import SMC_FEATURE_COLUMNS
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from trading_bot_v4.features.smc_feature_builder import _build_smc_feature_frame
from trading_bot_v4.ml.per_asset_trainer import (
    MIN_CALIBRATION_TRADES,
    MIN_WINDOW_TRADES,
    _ENDPOINT_COLUMNS,
    _select_threshold,
    _simulate_threshold,
    _walk_forward_rows,
)
from trading_bot_v4.utils.logger import build_logger
from tools.run_gbm_relative_strength_experiment import _new_model
from tools.run_triple_barrier_target_experiment import _triple_barrier_target
from train_model import DataHandler

logger = build_logger("v4_smc_per_asset_experiment")

TIMEFRAME = "1h"
FEATURE_COLUMNS = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS]
MODEL_ROOT = Path("models/experiments/smc_per_asset")
MIN_TOTAL_ROWS = 1500
TRAIN_END, CALIBRATION_END = 0.70, 0.85
MIN_SEGMENT_ROWS = 150


def _load_symbol_frame(symbol: str, direction: str) -> pd.DataFrame | None:
    df = load_gmx_ohlc(symbol, TIMEFRAME)
    if df is None or df.empty:
        return None
    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df = df.copy()
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr
    df = handler.prepare_features(df, prediction_horizon=1)
    if df.empty:
        return None

    smc = _build_smc_feature_frame(symbol, TIMEFRAME)
    df = df.join(smc, how="inner")
    if df.empty:
        return None
    # Several SMC columns (fvg_top, ob_size, swept_swing_low, ...) are only
    # defined on rows where that pattern just occurred -- NaN elsewhere by
    # design, not missing data. A blanket dropna would drop nearly every
    # row. Matches smc_feature_builder.py's own handling: forward-fill the
    # last known pattern level, zero only before any pattern has ever formed.
    df[SMC_FEATURE_COLUMNS] = df[SMC_FEATURE_COLUMNS].ffill().fillna(0.0)

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["target"] = _triple_barrier_target(df, direction)
    df = df.dropna(subset=[*FEATURE_COLUMNS, "target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    n = len(df)
    train_end = int(n * TRAIN_END)
    calibration_end = int(n * CALIBRATION_END)
    boundaries = (train_end, calibration_end - train_end, n - calibration_end)
    if any(size < MIN_SEGMENT_ROWS for size in boundaries):
        return None
    return df.iloc[:train_end], df.iloc[train_end:calibration_end], df.iloc[calibration_end:]


def _model_path(direction: str, symbol: str) -> Path:
    return MODEL_ROOT / direction / f"gbm_{symbol.upper()}.pkl"


def train_one(symbol: str, direction: str, force: bool = False) -> dict:
    symbol = symbol.upper()
    result_base = {"symbol": symbol, "direction": direction}
    if not force and _model_path(direction, symbol).exists():
        return {**result_base, "trained": True, "reason": "already trained (cached)"}

    df = _load_symbol_frame(symbol, direction)
    if df is None or len(df) < MIN_TOTAL_ROWS:
        rows = 0 if df is None else len(df)
        return {**result_base, "trained": False, "promoted": False,
                "reason": f"insufficient data: {rows} rows (< {MIN_TOTAL_ROWS})"}

    split = _split(df)
    if split is None:
        return {**result_base, "trained": False, "promoted": False,
                "reason": "insufficient rows per split segment", "train_rows": len(df)}
    train, calibration, holdout = split

    model = _new_model()
    model.fit(train[FEATURE_COLUMNS].to_numpy(np.float32), train["target"].to_numpy(int))

    calibration_probability = model.predict_proba(calibration[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    calibration_frame = calibration[_ENDPOINT_COLUMNS].reset_index(drop=True)
    threshold, _ = _select_threshold(calibration_frame, calibration_probability, direction, symbol)

    holdout_probability = model.predict_proba(holdout[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    holdout_frame = holdout[_ENDPOINT_COLUMNS].reset_index(drop=True)
    holdout_target = holdout_frame["target"].to_numpy(int)
    holdout_auc = (
        float(roc_auc_score(holdout_target, holdout_probability))
        if len(np.unique(holdout_target)) > 1 else float("nan")
    )
    holdout_summary = _simulate_threshold(holdout_frame, holdout_probability, threshold, direction, symbol)
    windows = _walk_forward_rows(holdout_frame, holdout_probability, threshold, direction, symbol)

    promoted = bool(
        holdout_summary["trades"] >= MIN_CALIBRATION_TRADES
        and all(row["trades"] >= MIN_WINDOW_TRADES and row["profit_factor"] >= 1.30 and row["return_pct"] > 0
                for row in windows)
    )

    _model_path(direction, symbol).parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with _model_path(direction, symbol).open("wb") as handle:
        pickle.dump(model, handle)

    result = {
        **result_base, "trained": True, "promoted": promoted,
        "threshold": float(threshold), "reason": "promoted" if promoted else "did not clear promotion gate",
        "train_rows": len(df), "holdout_trades": int(holdout_summary["trades"]),
        "holdout_return_pct": float(holdout_summary["return_pct"]),
        "holdout_profit_factor": float(holdout_summary["profit_factor"]),
        "holdout_auc": holdout_auc, "windows": windows,
    }
    logger.info(f"{direction}/{symbol}: promoted={promoted} threshold={threshold:.2f} "
                f"holdout_trades={result['holdout_trades']} holdout_return_pct={result['holdout_return_pct']:.2f} "
                f"holdout_profit_factor={result['holdout_profit_factor']:.2f} holdout_auc={holdout_auc:.3f}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=["BTC", "ETH", "ARB"])
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    directions = ["long", "short"] if args.direction == "both" else [args.direction]
    for direction in directions:
        for symbol in args.symbols:
            result = train_one(symbol, direction, force=args.force)
            print(result)
