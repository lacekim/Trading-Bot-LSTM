"""Ablation experiment: train the SMC CNN/LSTM with only a subset of SMC feature
groups (instead of all 18 binary flags + their continuous companions), and run
each variant through the same full liquidity-qualified comparison used for the
rescaled-fix experiment. All variants use the StandardScaler binary-passthrough
fix already in ml/smc_trainer.py.

Variants:
  regime_only  - Config.FEATURE_COLUMNS + REGIME_COLUMNS
  fvg_only     - Config.FEATURE_COLUMNS + FVG_COLUMNS
  regime_fvg   - Config.FEATURE_COLUMNS + REGIME_COLUMNS + FVG_COLUMNS

Nothing here touches the live models/lstm_smc_model.h5 + scaler_smc.pkl, or the
rescaled-fix experiment files from the previous run.
"""
from __future__ import annotations

import pickle
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.smc_swings import FVG_COLUMNS, REGIME_COLUMNS
from trading_bot_v4.execution.paper_model_performance import _production_metrics
from trading_bot_v4.execution.smc_model_paper import (
    _build_smc_model_feature_frame,
    _direction_from_probability,
    _timeframe_delta,
)
from trading_bot_v4.ml import smc_trainer

TIMEFRAME = "1h"
STARTING_CAPITAL = 100_000.0
EXPERIMENT_ROOT = Path("models/experiments")

VARIANTS: dict[str, list[str]] = {
    "regime_only": [*Config.FEATURE_COLUMNS, *REGIME_COLUMNS],
    "fvg_only": [*Config.FEATURE_COLUMNS, *FVG_COLUMNS],
    "regime_fvg": [*Config.FEATURE_COLUMNS, *REGIME_COLUMNS, *FVG_COLUMNS],
}


def _predict_with_features(model, scaler, symbol: str, timeframe: str, feature_columns: list[str]) -> pd.DataFrame:
    """Mirrors execution/smc_model_paper.py::_predict_smc_model_signals, parameterized
    by an explicit feature_columns subset instead of the hardcoded full SMC set, and
    returns columns already matching simulate_production_symbol's expected schema."""
    features = _build_smc_model_feature_frame(symbol, timeframe)
    if len(features) <= Config.SEQUENCE_LENGTH:
        raise ValueError(f"Insufficient feature rows for {symbol} {timeframe}: {len(features)}")

    feature_values = features[feature_columns].to_numpy(dtype=np.float32)
    scaled = scaler.transform(feature_values).astype(np.float32)
    seq_len = Config.SEQUENCE_LENGTH
    sequences = np.array([scaled[start:start + seq_len] for start in range(0, len(scaled) - seq_len)], dtype=np.float32)
    probabilities = model.predict(sequences, verbose=0).reshape(-1)

    signal_frame = features.iloc[seq_len:].copy()
    timestamps = pd.to_datetime(signal_frame.index, utc=True)
    closed_mask = timestamps + _timeframe_delta(timeframe) <= pd.Timestamp.now(tz="UTC")
    signal_frame = signal_frame.loc[closed_mask].copy()
    probabilities = probabilities[closed_mask]

    return pd.DataFrame({
        "timestamp": signal_frame.index,
        "symbol": symbol,
        "model_direction": [_direction_from_probability(float(p)) for p in probabilities],
        "open": signal_frame["open"].to_numpy(dtype=float),
        "high": signal_frame["high"].to_numpy(dtype=float),
        "low": signal_frame["low"].to_numpy(dtype=float),
        "price": signal_frame["close"].to_numpy(dtype=float),
        "atr": signal_frame["atr"].to_numpy(dtype=float),
    })


def run_variant(name: str, feature_columns: list[str]) -> None:
    variant_dir = EXPERIMENT_ROOT / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    model_path = variant_dir / "lstm_smc_model.h5"
    scaler_path = variant_dir / "scaler_smc.pkl"

    smc_trainer.SMC_MODEL_PATH = model_path
    smc_trainer.SMC_SCALER_PATH = scaler_path

    print(f"\n===== [{name}] training on {len(feature_columns)} features "
          f"({len(feature_columns) - len(Config.FEATURE_COLUMNS)} SMC) =====", flush=True)
    started = time.monotonic()
    result = smc_trainer.train_smc_model(timeframe=TIMEFRAME, feature_columns=feature_columns)
    elapsed = time.monotonic() - started
    print(f"[{name}] trained in {elapsed / 60.0:.1f} min. {result}", flush=True)

    print(f"[{name}] running full liquidity-qualified comparison", flush=True)
    model = load_model(model_path)
    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)

    audit = pd.read_csv("logs/v4_go_asset_selection_audit.csv")
    symbols = sorted(audit.loc[audit["readiness_decision"] == "GO", "symbol"].astype(str).str.upper().tolist())

    rows = []
    started_eval = time.monotonic()
    for index, symbol in enumerate(symbols, start=1):
        try:
            base = _predict_with_features(model, scaler, symbol, TIMEFRAME, feature_columns)
            base = base.dropna(subset=["open", "high", "low", "price", "atr"])

            summary, trades = simulate_production_symbol(base, symbol, STARTING_CAPITAL)
            metrics = _production_metrics(summary, trades)
            rows.append({"symbol": symbol, "status": "ok", **metrics})
            if index % 20 == 0 or index == len(symbols):
                print(f"[{name}] [{index}/{len(symbols)}] ...", flush=True)
        except Exception as exc:
            rows.append({"symbol": symbol, "status": f"error: {exc}"})
            traceback.print_exc()

    eval_elapsed = time.monotonic() - started_eval
    output_path = Path("logs") / f"v4_smc_ablation_{name}_comparison.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[{name}] comparison done in {eval_elapsed / 60.0:.1f} min. Wrote {output_path}", flush=True)


def main() -> None:
    overall_started = time.monotonic()
    for name, feature_columns in VARIANTS.items():
        run_variant(name, feature_columns)
    print(f"\nAll variants done in {(time.monotonic() - overall_started) / 60.0:.1f} minutes total.", flush=True)


if __name__ == "__main__":
    main()
