"""Experiment: per-asset gradient-boosted-tree models with cross-asset
relative-strength features, on the triple-barrier target.

Two prior experiments both concluded "no real edge" for the per-asset
CNN-LSTM (18 generic TA features, binary threshold target, then triple-barrier
target): 0/95 and 0/8 promotions respectively, with model confidence never
tracking real outcomes at any threshold. That failure signature -- raising the
decision threshold never improves results -- points at the *inputs*, not the
target definition: a 13k-parameter sequence model trained on ~50-70k noisy
rows of a single asset's own generic TA indicators (RSI/MACD/BB/momentum) has
very little to work with, and those indicators are well known to carry weak
standalone signal in liquid, arbitraged crypto markets.

Two changes here, both grounded in established results rather than another
target permutation:

1. Model class: HistGradientBoostingClassifier instead of a CNN-LSTM. Tree
   ensembles are consistently the stronger choice for small/medium tabular
   data (see e.g. Grinsztajn et al. 2022, "Why do tree-based models still
   outperform deep learning on tabular data?"); this codebase already has
   prior art in ml/cost_aware_long_trainer.py using the same estimator
   (pooled across symbols there -- here it's refit per-asset, preserving the
   per-asset-only-sees-its-own-data design this session already fixed for).
   Trees also don't need SEQUENCE_LENGTH-sized windows, so per-asset datasets
   too short for the LSTM's split requirements become usable.
2. Features: add cross-asset relative-strength features versus BTC (excess
   return same candle, its rolling mean, and BTC's own 24h return as a
   regime signal). Much of an altcoin's price action is just beta to BTC;
   relative-strength/cross-sectional momentum is one of the more robust,
   literature-backed signals, and unlike GMX funding/open-interest (checked
   and ruled out: only ~40 sparse points per symbol over 13 days, far too
   little history) this is computable over each symbol's full OHLC history
   with no new data collection needed.

Target stays triple-barrier (run_triple_barrier_target_experiment's
_triple_barrier_target, imported directly) -- still the more economically
honest label. Split/calibration/simulation reuse per_asset_trainer's already
-validated real-engine scoring machinery. Trains/evaluates the liquidity-
cleared symbols only, so results are directly comparable to both prior
baselines. Writes to models/experiments/gbm_relative_strength/, never
touches production models/{long,short}/ artifacts.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from trading_bot import list_gmx_symbols
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from trading_bot_v4.ml.per_asset_trainer import (
    MIN_CALIBRATION_TRADES,
    MIN_WINDOW_TRADES,
    SIMULATION_STARTING_CAPITAL,
    _ENDPOINT_COLUMNS,
    _select_threshold,
    _simulate_threshold,
    _walk_forward_rows,
)
from trading_bot_v4.risk.market_cap_tiers import liquid_symbols
from trading_bot_v4.utils.logger import build_logger
from train_model import DataHandler
from tools.run_triple_barrier_target_experiment import _triple_barrier_target

logger = build_logger("v4_gbm_relative_strength_experiment")

Direction = Literal["long", "short"]

TIMEFRAME = "1h"
EXPERIMENT_ROOT = Path("models/experiments/gbm_relative_strength")
RELATIVE_STRENGTH_COLUMNS = ["btc_relative_return", "btc_relative_return_ma20", "btc_return_24h"]
FEATURE_COLUMNS = [*Config.FEATURE_COLUMNS, *RELATIVE_STRENGTH_COLUMNS]

MIN_TOTAL_ROWS = 500
MIN_SEGMENT_ROWS = 50
TRAIN_END, CALIBRATION_END = 0.70, 0.85


@lru_cache(maxsize=4)
def _btc_return_frame(timeframe: str) -> pd.DataFrame:
    df = load_gmx_ohlc("BTC", timeframe)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    close = df["Close"].astype(float)
    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "btc_return_1h": np.log(close / close.shift(1)),
        "btc_return_24h": close.pct_change(24),
    })


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.10, random_state=42,
    )


def _load_symbol_frame(symbol: str, timeframe: str, direction: Direction) -> pd.DataFrame | None:
    df = load_gmx_ohlc(symbol, timeframe)
    if df is None or df.empty:
        return None

    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df = df.copy()
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr
    df = handler.prepare_features(df, prediction_horizon=1)
    if df.empty:
        return None

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})

    btc = _btc_return_frame(timeframe)
    df = df.merge(btc, on="timestamp", how="left")
    df[["btc_return_1h", "btc_return_24h"]] = df[["btc_return_1h", "btc_return_24h"]].ffill()
    df["btc_relative_return"] = df["returns"] - df["btc_return_1h"]
    df["btc_relative_return_ma20"] = df["btc_relative_return"].rolling(20).mean()
    df = df.dropna(subset=RELATIVE_STRENGTH_COLUMNS).reset_index(drop=True)
    if df.empty:
        return None

    df["target"] = _triple_barrier_target(df, direction)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    n = len(df)
    train_end = int(n * TRAIN_END)
    calibration_end = int(n * CALIBRATION_END)
    boundaries = (train_end, calibration_end - train_end, n - calibration_end)
    if any(size < MIN_SEGMENT_ROWS for size in boundaries):
        return None
    return df.iloc[:train_end], df.iloc[train_end:calibration_end], df.iloc[calibration_end:]


def _model_path(direction: Direction, symbol: str) -> Path:
    return EXPERIMENT_ROOT / direction / f"gbm_{symbol.upper()}.pkl"


def train_one(symbol: str, direction: Direction, timeframe: str = TIMEFRAME, force: bool = False) -> dict[str, Any]:
    symbol = symbol.upper()
    result_base: dict[str, Any] = {"symbol": symbol, "direction": direction}

    if not force and _model_path(direction, symbol).exists():
        logger.info(f"{direction}/{symbol}: experiment model already trained, skipping")
        return {**result_base, "trained": True, "reason": "already trained (cached)"}

    try:
        df = _load_symbol_frame(symbol, timeframe, direction)
    except FileNotFoundError:
        df = None

    if df is None or len(df) < MIN_TOTAL_ROWS:
        rows = 0 if df is None else len(df)
        return {**result_base, "trained": False, "promoted": False,
                "reason": f"insufficient data: {rows} rows (< {MIN_TOTAL_ROWS})"}

    split = _chronological_split(df)
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
        and all(
            row["trades"] >= MIN_WINDOW_TRADES and row["profit_factor"] >= 1.30 and row["return_pct"] > 0
            for row in windows
        )
    )

    _model_path(direction, symbol).parent.mkdir(parents=True, exist_ok=True)
    with _model_path(direction, symbol).open("wb") as handle:
        pickle.dump(model, handle)

    result = {
        **result_base, "trained": True, "promoted": promoted,
        "threshold": float(threshold), "reason": "promoted" if promoted else "did not clear promotion gate",
        "train_rows": len(df), "holdout_trades": int(holdout_summary["trades"]),
        "holdout_return_pct": float(holdout_summary["return_pct"]),
        "holdout_profit_factor": float(holdout_summary["profit_factor"]),
        "holdout_win_rate_pct": float(holdout_summary["win_rate_pct"]),
        "holdout_auc": holdout_auc, "windows": windows,
    }
    logger.info(
        f"{direction}/{symbol}: promoted={promoted} threshold={threshold:.2f} "
        f"holdout_trades={result['holdout_trades']} holdout_return_pct={result['holdout_return_pct']:.2f} "
        f"holdout_profit_factor={result['holdout_profit_factor']:.2f} holdout_auc={holdout_auc:.3f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None, help="explicit symbol subset (default: all liquid symbols)")
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    directions: list[Direction] = ["long", "short"] if args.direction == "both" else [args.direction]
    all_symbols = [s.upper() for s in (args.symbols or list_gmx_symbols(TIMEFRAME))]

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for direction in directions:
        symbols = liquid_symbols(direction, all_symbols) if not args.symbols else all_symbols
        logger.info(f"{direction}: {len(symbols)} symbols to train ({'explicit' if args.symbols else 'liquidity-filtered'})")
        for i, symbol in enumerate(symbols, start=1):
            started = time.time()
            try:
                result = train_one(symbol, direction, force=args.force)
            except Exception as exc:  # noqa: BLE001 -- keep the sweep alive past one symbol's failure
                logger.error(f"{direction}/{symbol}: FAILED: {exc}\n{traceback.format_exc()}")
                result = {"symbol": symbol, "direction": direction, "trained": False,
                          "promoted": False, "reason": f"exception: {exc}"}
            result["elapsed_sec"] = round(time.time() - started, 1)
            results.append(result)
            logger.info(f"[{i}/{len(symbols)}] {direction}/{symbol} done in {result['elapsed_sec']}s")
            (EXPERIMENT_ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    promoted = [r for r in results if r.get("promoted")]
    logger.info(f"DONE: {len(promoted)}/{len(results)} symbol/direction combos promoted")
    for r in promoted:
        logger.info(f"  PROMOTED {r['direction']}/{r['symbol']}: threshold={r['threshold']:.2f} "
                    f"return_pct={r['holdout_return_pct']:.2f} profit_factor={r['holdout_profit_factor']:.2f}")


if __name__ == "__main__":
    main()
