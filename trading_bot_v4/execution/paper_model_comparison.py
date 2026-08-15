"""Paper-only comparison of original model signals against SMC model signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols, load_gmx_ohlc
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.execution.smc_model_paper import _predict_smc_model_signals
from trading_bot_v4.execution.smc_model_paper import _timeframe_delta
from trading_bot_v4.ml.smc_trainer import SMC_MODEL_PATH, SMC_SCALER_PATH
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache
from trading_bot_v4.utils.artifact_lineage import artifact_version
from trading_bot_v4.utils.signal_direction import binary_upside_direction
from trading_bot_v4.utils.macd_confirmation import macd_components
from trading_bot_v4.execution.persistent_policy import direction_for_probability, policy_for


logger = build_logger("v4_paper_model_comparison")

ORIGINAL_BASELINE_SIGNALS_PATH = Path("logs/v5_original_long_paper_signals.csv")
ORIGINAL_BASELINE_SUMMARY_PATH = Path("logs/v5_original_long_paper_summary.csv")

PAPER_MODEL_COMPARISON_CSV_PATH = Path("logs/v4_paper_model_comparison.csv")
PAPER_MODEL_COMPARISON_HTML_PATH = Path("logs/v4_paper_model_comparison.html")


def _direction_from_probability(probability: float, symbol: str = "") -> str:
    return direction_for_probability(probability, symbol) if symbol else binary_upside_direction(probability, Config.MIN_SIGNAL_THRESHOLD)


def _predict_original_model_signals(model: Any, scaler: Any, symbol: str, timeframe: str,
                                    closed_only: bool = True) -> pd.DataFrame:
    raw = load_gmx_ohlc(symbol, timeframe)
    handler = V4DataHandler()
    features = handler.prepare_features(raw.copy())
    missing = [column for column in Config.FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Missing original feature columns for {symbol}: {missing}")

    feature_frame = features[Config.FEATURE_COLUMNS].copy()
    feature_frame[Config.FEATURE_COLUMNS] = feature_frame[Config.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=Config.FEATURE_COLUMNS)
    prices = pd.to_numeric(raw["Close"], errors="coerce").reindex(feature_frame.index)
    feature_frame["price"] = prices
    feature_frame["atr"] = handler.calculate_atr(raw, Config.ATR_PERIOD).reindex(feature_frame.index)
    feature_frame = feature_frame.join(macd_components(raw["Close"]).reindex(feature_frame.index))
    feature_frame["macd_histogram_previous"] = feature_frame["macd_histogram"].shift(1)
    feature_frame["price_vs_ma200"] = prices / prices.rolling(200).mean() - 1.0
    for source, target in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")):
        feature_frame[target] = pd.to_numeric(raw[source], errors="coerce").reindex(feature_frame.index)
    timestamps = pd.Series(pd.to_datetime(feature_frame.index), index=feature_frame.index)
    feature_frame["candle_gap_seconds"] = timestamps.diff().dt.total_seconds().fillna(0.0)
    feature_frame = feature_frame.dropna(subset=["price", "open", "high", "low", "close", "atr"])

    if len(feature_frame) <= Config.SEQUENCE_LENGTH:
        raise ValueError(f"Insufficient original model feature rows for {symbol} {timeframe}: {len(feature_frame)}")

    values = feature_frame[Config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    scaled = scaler.transform(values).astype(np.float32)
    seq_len = Config.SEQUENCE_LENGTH
    sequences = np.array([scaled[start : start + seq_len] for start in range(0, len(scaled) - seq_len)], dtype=np.float32)
    probabilities = model.predict(sequences, verbose=0).reshape(-1)
    signal_frame = feature_frame.iloc[seq_len:].copy()
    if closed_only:
        timestamps = pd.to_datetime(signal_frame.index, utc=True)
        closed_mask = timestamps + _timeframe_delta(timeframe) <= pd.Timestamp.now(tz="UTC")
        signal_frame = signal_frame.loc[closed_mask].copy()
        probabilities = probabilities[closed_mask]

    signals = pd.DataFrame(
        {
            "timestamp": signal_frame.index,
            "symbol": symbol,
            "timeframe": timeframe,
            "original_probability": probabilities,
            "original_direction": [_direction_from_probability(float(probability), symbol) for probability in probabilities],
            "original_price": signal_frame["price"].to_numpy(dtype=float),
            "open": signal_frame["open"].to_numpy(dtype=float),
            "high": signal_frame["high"].to_numpy(dtype=float),
            "low": signal_frame["low"].to_numpy(dtype=float),
            "close": signal_frame["close"].to_numpy(dtype=float),
            "candle_gap_seconds": signal_frame["candle_gap_seconds"].to_numpy(dtype=float),
            "atr": signal_frame["atr"].to_numpy(dtype=float),
            "macd_line": signal_frame["macd_line"].to_numpy(dtype=float),
            "macd_signal": signal_frame["macd_signal"].to_numpy(dtype=float),
            "macd_histogram": signal_frame["macd_histogram"].to_numpy(dtype=float),
            "macd_histogram_previous": signal_frame["macd_histogram_previous"].to_numpy(dtype=float),
            "price_vs_ma200": signal_frame["price_vs_ma200"].to_numpy(dtype=float),
        }
    )
    signals["original_candidate"] = signals["original_direction"].isin(["LONG", "SHORT"])
    return signals


def predict_original_baseline_signals(model: Any, scaler: Any, symbol: str, timeframe: str,
                                      closed_only: bool = True) -> pd.DataFrame:
    """Return original-model inference in the scheduler's standard signal schema."""
    signals = _predict_original_model_signals(
        model, scaler, symbol, timeframe, closed_only=closed_only
    ).rename(columns={
        "original_probability": "model_probability",
        "original_direction": "model_direction",
        "original_price": "price",
        "original_candidate": "is_trade_candidate",
    })
    signals["threshold"] = float(policy_for(symbol).threshold)
    signals["feature_count"] = len(Config.FEATURE_COLUMNS)
    signals["model_side"] = "ORIGINAL_PERSISTENT" if policy_for(symbol).enabled else "ORIGINAL_UPSIDE"
    return signals


def run_original_baseline_paper_signals(symbols: list[str], timeframe: str, model: Any = None,
                                        scaler: Any = None, signals_path: Path | None = None,
                                        summary_path: Path | None = None,
                                        per_asset_cache: Any = None) -> dict[str, Any]:
    """Generate the active baseline LONG signals and persist daily-analysis inputs.

    Pass per_asset_cache (a PerAssetModelCache) to resolve one independently
    trained model per symbol instead of reusing a single model/scaler for
    every symbol -- symbols without a trained LONG model are skipped rather
    than falling back to some other symbol's model.
    """
    frames, summaries = [], []
    version = (
        "per-asset" if per_asset_cache is not None else
        artifact_version([Config.MODEL_DIR / Config.MODEL_NAME, Config.MODEL_DIR / Config.SCALER_NAME])
    )
    for symbol in symbols:
        try:
            if per_asset_cache is not None:
                pair = per_asset_cache.get("long", str(symbol).upper())
                if pair is None:
                    continue
                symbol_model, symbol_scaler = pair
            else:
                symbol_model, symbol_scaler = model, scaler
            signals = predict_original_baseline_signals(symbol_model, symbol_scaler, str(symbol).upper(), timeframe)
        except Exception as exc:
            logger.warning("Skipping %s %s during original baseline inference: %s", symbol, timeframe, exc)
            continue
        frames.append(signals)
        signals["model_version"] = version
        latest = signals.iloc[-1] if not signals.empty else None
        summaries.append({
            "symbol": str(symbol).upper(), "timeframe": timeframe,
            "predictions": len(signals),
            "trade_candidates": int(signals["is_trade_candidate"].sum()) if not signals.empty else 0,
            "latest_direction": str(latest["model_direction"]) if latest is not None else "",
            "latest_probability": float(latest["model_probability"]) if latest is not None else np.nan,
            "latest_price": float(latest["price"]) if latest is not None else np.nan,
        })
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    signals_path = Path(signals_path or ORIGINAL_BASELINE_SIGNALS_PATH)
    summary_path = Path(summary_path or ORIGINAL_BASELINE_SUMMARY_PATH)
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(signals_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    return {"signals": combined, "signals_path": signals_path,
            "summary_path": summary_path, "assets_evaluated": len(summaries)}


def _prepare_smc_signals(smc_signals: pd.DataFrame) -> pd.DataFrame:
    prepared = smc_signals.rename(
        columns={
            "model_probability": "smc_probability",
            "model_direction": "smc_direction",
            "price": "smc_price",
            "is_trade_candidate": "smc_candidate",
        }
    )
    return prepared[
        [
            "timestamp",
            "symbol",
            "timeframe",
            "smc_probability",
            "smc_direction",
            "smc_price",
            "smc_candidate",
        ]
    ].copy()


def _agreement_pct(mask: pd.Series, agreement: pd.Series) -> float:
    denominator = int(mask.sum())
    if denominator == 0:
        return 0.0
    return float((agreement & mask).sum() / denominator * 100.0)


def _latest_difference(merged: pd.DataFrame) -> str:
    if merged.empty:
        return ""
    latest = merged.iloc[-1]
    return (
        f"{latest['timestamp']} "
        f"original={latest['original_direction']} p={float(latest['original_probability']):.6f}; "
        f"SMC={latest['smc_direction']} p={float(latest['smc_probability']):.6f}"
    )


def _compare_symbol(
    original_model: Any,
    original_scaler: Any,
    smc_model: Any,
    smc_scaler: Any,
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    original = _predict_original_model_signals(original_model, original_scaler, symbol, timeframe)
    smc = _prepare_smc_signals(_predict_smc_model_signals(smc_model, smc_scaler, symbol, timeframe))
    merged = original.merge(
        smc,
        on=["timestamp", "symbol", "timeframe"],
        how="inner",
    ).sort_values("timestamp")
    if merged.empty:
        raise ValueError(f"No shared timestamps for {symbol} {timeframe}")

    same_direction = merged["original_direction"].eq(merged["smc_direction"])
    long_union = merged["original_direction"].eq("LONG") | merged["smc_direction"].eq("LONG")
    short_union = merged["original_direction"].eq("SHORT") | merged["smc_direction"].eq("SHORT")
    long_agreement = merged["original_direction"].eq("LONG") & merged["smc_direction"].eq("LONG")
    short_agreement = merged["original_direction"].eq("SHORT") & merged["smc_direction"].eq("SHORT")

    original_candidates = int(merged["original_candidate"].sum())
    smc_candidates = int(merged["smc_candidate"].sum())
    if smc_candidates > original_candidates:
        aggressiveness = "SMC more aggressive"
    elif smc_candidates < original_candidates:
        aggressiveness = "SMC more selective"
    else:
        aggressiveness = "same candidate count"

    latest = merged.iloc[-1]
    latest_signal_difference = _latest_difference(merged)
    latest_differs = bool(latest["original_direction"] != latest["smc_direction"])
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "shared_timestamps": int(len(merged)),
        "original_candidates": original_candidates,
        "smc_candidates": smc_candidates,
        "candidate_delta_smc_minus_original": int(smc_candidates - original_candidates),
        "signal_agreement_pct": float(same_direction.mean() * 100.0),
        "long_agreement_pct": _agreement_pct(long_union, long_agreement),
        "short_agreement_pct": _agreement_pct(short_union, short_agreement),
        "aggressiveness": aggressiveness,
        "latest_timestamp": latest["timestamp"],
        "latest_original_direction": latest["original_direction"],
        "latest_original_probability": float(latest["original_probability"]),
        "latest_smc_direction": latest["smc_direction"],
        "latest_smc_probability": float(latest["smc_probability"]),
        "latest_signal_differs": latest_differs,
        "latest_signal_difference": latest_signal_difference,
    }


def _write_html_report(report: pd.DataFrame, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    display = report.copy()
    numeric_columns = display.select_dtypes(include=[np.number]).columns
    display[numeric_columns] = display[numeric_columns].round(6)
    aggressive = int((display["aggressiveness"] == "SMC more aggressive").sum()) if not display.empty else 0
    selective = int((display["aggressiveness"] == "SMC more selective").sum()) if not display.empty else 0
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>V4 Paper Model Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f8; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>V4 Paper Model Comparison</h1>
  <p>Paper-only comparison of original model signals and optional SMC model signals.</p>
  <ul>
    <li>Assets compared: {len(display)}</li>
    <li>Assets where SMC is more aggressive: {aggressive}</li>
    <li>Assets where SMC is more selective: {selective}</li>
  </ul>
  {display.to_html(index=False, classes="comparison", border=0)}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def run_paper_model_comparison(args: Any) -> dict[str, Any]:
    """Compare original and SMC paper model signals without placing orders."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    use_all_assets = bool(getattr(args, "all_assets", False))
    selected_symbol = str(getattr(args, "symbol", Config.GMX_SYMBOL)).upper()
    symbols = [str(symbol).upper() for symbol in list_gmx_symbols(timeframe)] if use_all_assets else [selected_symbol]

    original_model, original_scaler = ModelScalerCache().load()
    smc_model, smc_scaler = ModelScalerCache(model_path=SMC_MODEL_PATH, scaler_path=SMC_SCALER_PATH).load()

    rows = []
    for symbol in symbols:
        try:
            rows.append(_compare_symbol(original_model, original_scaler, smc_model, smc_scaler, symbol, timeframe))
        except Exception as exc:
            logger.warning("Skipping %s %s during paper model comparison: %s", symbol, timeframe, exc)

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["aggressiveness", "candidate_delta_smc_minus_original", "symbol"], ascending=[True, False, True])

    PAPER_MODEL_COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(PAPER_MODEL_COMPARISON_CSV_PATH, index=False)
    _write_html_report(report, PAPER_MODEL_COMPARISON_HTML_PATH)

    aggressive_assets = report.loc[report["aggressiveness"] == "SMC more aggressive", "symbol"].astype(str).tolist() if not report.empty else []
    selective_assets = report.loc[report["aggressiveness"] == "SMC more selective", "symbol"].astype(str).tolist() if not report.empty else []
    return {
        "assets_compared": int(len(report)),
        "csv_path": PAPER_MODEL_COMPARISON_CSV_PATH,
        "html_path": PAPER_MODEL_COMPARISON_HTML_PATH,
        "report_df": report,
        "smc_more_aggressive_assets": aggressive_assets,
        "smc_more_selective_assets": selective_assets,
    }
