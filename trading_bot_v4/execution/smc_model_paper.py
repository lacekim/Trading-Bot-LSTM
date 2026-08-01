"""Paper-only signal generation for the optional SMC-enhanced model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import timedelta

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols, load_gmx_ohlc
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.core.smc_swings import SMC_FEATURE_COLUMNS
from trading_bot_v4.features.smc_feature_builder import _build_smc_feature_frame
from trading_bot_v4.ml.smc_trainer import SMC_MODEL_PATH, SMC_SCALER_PATH
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache
from trading_bot_v4.utils.signal_direction import binary_upside_direction


logger = build_logger("v4_smc_model_paper")

SMC_MODEL_PAPER_SUMMARY_PATH = Path("logs/v4_smc_model_paper_summary.csv")
SMC_MODEL_PAPER_SIGNALS_PATH = Path("logs/v4_smc_model_paper_signals.csv")
VALIDATED_RANKINGS_PATH = Path("models/asset_rankings_validated.csv")
VALIDATED_WHITELIST_PAPER_SUMMARY_PATH = Path("logs/v4_validated_whitelist_paper_summary.csv")
VALIDATED_WHITELIST_PAPER_SIGNALS_PATH = Path("logs/v4_validated_whitelist_paper_signals.csv")


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    units = {"m": "minutes", "h": "hours", "d": "days"}
    if unit not in units or value <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return pd.Timedelta(**{units[unit]: value})


def _direction_from_probability(probability: float) -> str:
    return binary_upside_direction(probability, Config.MIN_SIGNAL_THRESHOLD)


def _format_latest_signal(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return (
        f"{row['timestamp']} {row['model_direction']} "
        f"p={float(row['model_probability']):.6f} "
        f"price={float(row['price']):.10f}"
    )


def _build_smc_model_feature_frame(symbol: str, timeframe: str) -> pd.DataFrame:
    raw = load_gmx_ohlc(symbol, timeframe)
    handler = V4DataHandler()
    original = handler.prepare_features(raw.copy())

    missing_original = [column for column in Config.FEATURE_COLUMNS if column not in original.columns]
    if missing_original:
        raise ValueError(f"Missing original model feature columns for {symbol}: {missing_original}")

    original_features = original[Config.FEATURE_COLUMNS].copy()
    original_features.index.name = "timestamp"
    smc_features = _build_smc_feature_frame(symbol, timeframe)
    combined = original_features.join(smc_features.reindex(original_features.index), how="inner")
    feature_columns = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS]
    combined[feature_columns] = combined[feature_columns].apply(pd.to_numeric, errors="coerce")
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined[SMC_FEATURE_COLUMNS] = combined[SMC_FEATURE_COLUMNS].ffill().fillna(0.0)
    combined = combined.dropna(subset=Config.FEATURE_COLUMNS)

    prices = pd.to_numeric(raw["Close"], errors="coerce").reindex(combined.index)
    combined["price"] = prices
    combined["atr"] = handler.calculate_atr(raw, Config.ATR_PERIOD).reindex(combined.index)
    for source, target in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")]:
        combined[target] = pd.to_numeric(raw[source], errors="coerce").reindex(combined.index)
    timestamp_series = pd.Series(pd.to_datetime(combined.index), index=combined.index)
    combined["candle_gap_seconds"] = timestamp_series.diff().dt.total_seconds().fillna(0.0)
    combined = combined.dropna(subset=["price", "open", "high", "low", "close", "atr"])
    return combined


def _predict_smc_model_signals(model: Any, scaler: Any, symbol: str, timeframe: str) -> pd.DataFrame:
    features = _build_smc_model_feature_frame(symbol, timeframe)
    feature_columns = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS]
    if len(features) <= Config.SEQUENCE_LENGTH:
        raise ValueError(f"Insufficient SMC model feature rows for {symbol} {timeframe}: {len(features)}")

    feature_values = features[feature_columns].to_numpy(dtype=np.float32)
    scaled = scaler.transform(feature_values).astype(np.float32)
    seq_len = Config.SEQUENCE_LENGTH
    sequences = np.array([scaled[start : start + seq_len] for start in range(0, len(scaled) - seq_len)], dtype=np.float32)
    probabilities = model.predict(sequences, verbose=0).reshape(-1)
    signal_frame = features.iloc[seq_len:].copy()
    timestamps = pd.to_datetime(signal_frame.index, utc=True)
    closed_mask = timestamps + _timeframe_delta(timeframe) <= pd.Timestamp.now(tz="UTC")
    signal_frame = signal_frame.loc[closed_mask].copy()
    probabilities = probabilities[closed_mask]

    signals = pd.DataFrame(
        {
            "timestamp": signal_frame.index,
            "symbol": symbol,
            "timeframe": timeframe,
            "model_probability": probabilities,
            "model_direction": [_direction_from_probability(float(probability)) for probability in probabilities],
            "price": signal_frame["price"].to_numpy(dtype=float),
            "open": signal_frame["open"].to_numpy(dtype=float),
            "high": signal_frame["high"].to_numpy(dtype=float),
            "low": signal_frame["low"].to_numpy(dtype=float),
            "close": signal_frame["close"].to_numpy(dtype=float),
            "candle_gap_seconds": signal_frame["candle_gap_seconds"].to_numpy(dtype=float),
            "atr": signal_frame["atr"].to_numpy(dtype=float),
        }
    )
    signals["is_trade_candidate"] = signals["model_direction"].isin(["LONG", "SHORT"])
    signals["threshold"] = float(Config.MIN_SIGNAL_THRESHOLD)
    signals["feature_count"] = len(feature_columns)
    return signals


def _summarize_asset(signals: pd.DataFrame, symbol: str, timeframe: str) -> dict[str, Any]:
    candidates = signals.loc[signals["is_trade_candidate"]].copy()
    latest_candidate = candidates.iloc[-1] if not candidates.empty else None
    latest_prediction = signals.iloc[-1] if not signals.empty else None
    latest_signal = _format_latest_signal(latest_candidate if latest_candidate is not None else latest_prediction)
    latest_direction = "" if latest_prediction is None else str(latest_prediction["model_direction"])
    latest_probability = float("nan") if latest_prediction is None else float(latest_prediction["model_probability"])
    latest_price = float("nan") if latest_prediction is None else float(latest_prediction["price"])

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "predictions": int(len(signals)),
        "trade_candidates": int(len(candidates)),
        "latest_direction": latest_direction,
        "latest_probability": latest_probability,
        "latest_price": latest_price,
        "latest_signal": latest_signal,
    }


def _load_validated_whitelist(timeframe: str) -> list[str]:
    if not VALIDATED_RANKINGS_PATH.exists():
        raise FileNotFoundError(f"Validated asset rankings not found: {VALIDATED_RANKINGS_PATH}")

    rankings = pd.read_csv(VALIDATED_RANKINGS_PATH)
    if rankings.empty:
        raise ValueError(f"Validated asset rankings are empty: {VALIDATED_RANKINGS_PATH}")

    rankings["symbol"] = rankings["symbol"].astype(str).str.upper()
    if "timeframe" in rankings.columns:
        rankings = rankings.loc[rankings["timeframe"].astype(str).eq(str(timeframe))]

    required = [
        "constrained_smc_return_pct",
        "constrained_smc_profit_factor",
        "constrained_smc_max_drawdown_pct",
        "trade_count_sanity_score",
    ]
    missing = [column for column in required if column not in rankings.columns]
    if missing:
        raise ValueError(f"Validated rankings missing whitelist columns: {missing}")

    whitelist = rankings.loc[
        (pd.to_numeric(rankings["constrained_smc_return_pct"], errors="coerce") > 0.0)
        & (pd.to_numeric(rankings["constrained_smc_profit_factor"], errors="coerce") >= 1.0)
        & (pd.to_numeric(rankings["constrained_smc_max_drawdown_pct"], errors="coerce") >= -15.0)
        & (pd.to_numeric(rankings["trade_count_sanity_score"], errors="coerce") >= 70.0)
    ].sort_values("rank" if "rank" in rankings.columns else "validated_score")

    symbols = whitelist["symbol"].drop_duplicates().tolist()
    if not symbols:
        raise ValueError("Validated whitelist is empty; rerun --rank-assets --validated and inspect filters")
    return symbols


def _write_outputs(
    signals: pd.DataFrame,
    summaries: list[dict[str, Any]],
    summary_path: Path,
    signals_path: Path,
) -> pd.DataFrame:
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(signals_path, index=False)

    summary = pd.DataFrame(summaries)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


def run_smc_model_paper_trading(args: Any) -> dict[str, Any]:
    """Generate paper-only signals from the separate SMC-enhanced model."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    use_validated_whitelist = bool(getattr(args, "validated_whitelist", False))
    use_all_assets = bool(getattr(args, "all_assets", False))
    selected_symbols = getattr(args, "symbols", None)
    selected_symbol = str(getattr(args, "symbol", Config.GMX_SYMBOL)).upper()
    if selected_symbols:
        symbols = [str(symbol).upper() for symbol in selected_symbols]
        summary_path = SMC_MODEL_PAPER_SUMMARY_PATH
        signals_path = SMC_MODEL_PAPER_SIGNALS_PATH
    elif use_validated_whitelist:
        symbols = _load_validated_whitelist(timeframe)
        summary_path = VALIDATED_WHITELIST_PAPER_SUMMARY_PATH
        signals_path = VALIDATED_WHITELIST_PAPER_SIGNALS_PATH
    else:
        symbols = [str(symbol).upper() for symbol in list_gmx_symbols(timeframe)] if use_all_assets else [selected_symbol]
        summary_path = SMC_MODEL_PAPER_SUMMARY_PATH
        signals_path = SMC_MODEL_PAPER_SIGNALS_PATH

    model = getattr(args, "model", None)
    scaler = getattr(args, "scaler", None)
    if model is None or scaler is None:
        cache = ModelScalerCache(model_path=SMC_MODEL_PATH, scaler_path=SMC_SCALER_PATH)
        model, scaler = cache.load()

    signal_frames = []
    summaries = []
    for symbol in symbols:
        try:
            signals = _predict_smc_model_signals(model, scaler, symbol, timeframe)
        except Exception as exc:
            logger.warning("Skipping %s %s during SMC model paper generation: %s", symbol, timeframe, exc)
            continue
        signal_frames.append(signals)
        summaries.append(_summarize_asset(signals, symbol, timeframe))

    all_signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame(
        columns=[
            "timestamp",
            "symbol",
            "timeframe",
            "model_probability",
            "model_direction",
            "price",
            "is_trade_candidate",
            "threshold",
            "feature_count",
        ]
    )
    summary = _write_outputs(all_signals, summaries, summary_path, signals_path)
    return {
        "all_assets": use_all_assets,
        "validated_whitelist": use_validated_whitelist,
        "symbols": symbols,
        "assets_evaluated": int(len(summaries)),
        "predictions_evaluated": int(len(all_signals)),
        "trade_candidates": int(all_signals["is_trade_candidate"].sum()) if not all_signals.empty else 0,
        "summary_path": summary_path,
        "signals_path": signals_path,
        "summary_df": summary,
    }
