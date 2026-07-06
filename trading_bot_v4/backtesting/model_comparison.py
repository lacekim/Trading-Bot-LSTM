"""Analysis-only comparison between the original and optional SMC V4 models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from tensorflow.keras.models import load_model

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.smc_swings import SMC_FEATURE_COLUMNS
from trading_bot_v4.utils.logger import build_logger


logger = build_logger("v4_model_comparison")

ORIGINAL_MODEL_PATH = Path("models/lstm_ada_model.h5")
ORIGINAL_SCALER_PATH = Path("models/scaler_ada.pkl")
SMC_MODEL_PATH = Path("models/lstm_smc_model.h5")
SMC_SCALER_PATH = Path("models/scaler_smc.pkl")
TRAINING_DATA_TEMPLATE = "training_data_smc_all_assets_{timeframe}.csv"
COMPARISON_CSV_PATH = Path("models/model_comparison.csv")
COMPARISON_HTML_PATH = Path("models/model_comparison.html")

PREDICTION_METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "log_loss"]
TRADING_METRICS = [
    "return_pct",
    "final_capital",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown_pct",
    "trade_count",
    "win_rate_pct",
    "average_trade_pct",
    "expectancy",
]


@dataclass(frozen=True)
class ModelComparisonResult:
    assets_compared: int
    smc_better_assets: list[str]
    original_better_assets: list[str]
    average_improvement: float
    median_improvement: float
    csv_path: Path
    html_path: Path
    comparison: pd.DataFrame


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def _load_comparison_data(timeframe: str) -> pd.DataFrame:
    path = Path("models") / TRAINING_DATA_TEMPLATE.format(timeframe=timeframe)
    _require_file(path, "SMC all-assets training data")

    required = [
        "timestamp",
        "symbol",
        *Config.FEATURE_COLUMNS,
        *SMC_FEATURE_COLUMNS,
        "future_return",
        "target",
    ]
    dataset = pd.read_csv(path, usecols=required)
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"Missing comparison data columns: {missing}")

    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], errors="coerce")
    dataset["symbol"] = dataset["symbol"].astype(str).str.upper()
    numeric_columns = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS, "future_return", "target"]
    dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors="coerce")
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=["timestamp", "symbol", *numeric_columns])
    dataset["target"] = dataset["target"].astype(int)
    return dataset.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _make_sequences(features: np.ndarray) -> np.ndarray:
    seq_len = Config.SEQUENCE_LENGTH
    sequence_count = len(features) - seq_len
    if sequence_count <= 0:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32)
    return np.array([features[start : start + seq_len] for start in range(sequence_count)], dtype=np.float32)


def _predict_symbol(
    model: Any,
    scaler: Any,
    group: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    features = group[feature_columns].to_numpy(dtype=np.float32)
    scaled = scaler.transform(features).astype(np.float32)
    sequences = _make_sequences(scaled)
    if len(sequences) == 0:
        return np.array([], dtype=np.float32)
    return model.predict(sequences, verbose=0).reshape(-1).astype(np.float32)


def _prediction_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    clipped = np.clip(probabilities.astype(float), 1e-7, 1 - 1e-7)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, clipped))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _max_drawdown_pct(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity / running_max - 1.0) * 100.0
    return float(drawdown.min())


def _trading_metrics(future_returns: np.ndarray, probabilities: np.ndarray, starting_capital: float) -> dict[str, float]:
    if len(future_returns) == 0:
        return {
            "return_pct": 0.0,
            "final_capital": float(starting_capital),
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "average_trade_pct": 0.0,
            "expectancy": 0.0,
        }

    threshold = float(Config.MIN_SIGNAL_THRESHOLD)
    long_mask = probabilities > threshold
    short_mask = probabilities < (1.0 - threshold)
    trade_mask = long_mask | short_mask
    if not bool(trade_mask.any()):
        return {
            "return_pct": 0.0,
            "final_capital": float(starting_capital),
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "average_trade_pct": 0.0,
            "expectancy": 0.0,
        }

    active_future_returns = future_returns[trade_mask]
    active_probabilities = probabilities[trade_mask]
    directions = np.where(active_probabilities > threshold, 1.0, -1.0)
    trade_returns_pct = active_future_returns.astype(float) * directions * 100.0
    equity_multipliers = 1.0 + (trade_returns_pct / 100.0)
    equity_multipliers = np.maximum(equity_multipliers, 0.0)
    equity = float(starting_capital) * np.cumprod(equity_multipliers)
    final_capital = float(equity[-1]) if len(equity) else float(starting_capital)

    winning = trade_returns_pct[trade_returns_pct > 0]
    losing = trade_returns_pct[trade_returns_pct < 0]
    gross_profit = float(winning.sum()) if len(winning) else 0.0
    gross_loss = abs(float(losing.sum())) if len(losing) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    mean_return = float(trade_returns_pct.mean())
    std_return = float(trade_returns_pct.std(ddof=0)) if len(trade_returns_pct) > 1 else 0.0
    downside_std = float(losing.std(ddof=0)) if len(losing) > 1 else 0.0
    return_pct = ((final_capital / float(starting_capital)) - 1.0) * 100.0 if starting_capital else 0.0
    max_drawdown = _max_drawdown_pct(equity)
    win_rate = float(len(winning) / len(trade_returns_pct) * 100.0) if len(trade_returns_pct) else 0.0
    avg_win = float(winning.mean()) if len(winning) else 0.0
    avg_loss = float(losing.mean()) if len(losing) else 0.0
    win_rate_decimal = win_rate / 100.0

    return {
        "return_pct": float(return_pct),
        "final_capital": final_capital,
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(mean_return / std_return) if std_return > 0 else 0.0,
        "sortino_ratio": float(mean_return / downside_std) if downside_std > 0 else 0.0,
        "calmar_ratio": float(return_pct / abs(max_drawdown)) if max_drawdown < 0 else 0.0,
        "max_drawdown_pct": max_drawdown,
        "trade_count": int(len(trade_returns_pct)),
        "win_rate_pct": win_rate,
        "average_trade_pct": mean_return,
        "expectancy": float((win_rate_decimal * avg_win) + ((1.0 - win_rate_decimal) * avg_loss)),
    }


def _build_symbol_row(
    symbol: str,
    group: pd.DataFrame,
    original_model: Any,
    original_scaler: Any,
    smc_model: Any,
    smc_scaler: Any,
    starting_capital: float,
) -> dict[str, float | int | str] | None:
    seq_len = Config.SEQUENCE_LENGTH
    if len(group) <= seq_len:
        logger.warning("Skipping %s: insufficient rows for comparison (%s)", symbol, len(group))
        return None

    original_feature_columns = list(Config.FEATURE_COLUMNS)
    smc_feature_columns = [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS]
    original_probabilities = _predict_symbol(original_model, original_scaler, group, original_feature_columns)
    smc_probabilities = _predict_symbol(smc_model, smc_scaler, group, smc_feature_columns)

    row_count = min(len(original_probabilities), len(smc_probabilities))
    if row_count == 0:
        logger.warning("Skipping %s: no comparison predictions produced", symbol)
        return None

    endpoint_frame = group.iloc[seq_len : seq_len + row_count]
    y_true = endpoint_frame["target"].to_numpy(dtype=int)
    future_returns = endpoint_frame["future_return"].to_numpy(dtype=float)
    original_probabilities = original_probabilities[:row_count]
    smc_probabilities = smc_probabilities[:row_count]

    original_prediction = _prediction_metrics(y_true, original_probabilities)
    smc_prediction = _prediction_metrics(y_true, smc_probabilities)
    original_trading = _trading_metrics(future_returns, original_probabilities, starting_capital)
    smc_trading = _trading_metrics(future_returns, smc_probabilities, starting_capital)

    row: dict[str, float | int | str] = {
        "symbol": symbol,
        "rows_evaluated": int(row_count),
        "original_feature_count": len(original_feature_columns),
        "smc_feature_count": len(SMC_FEATURE_COLUMNS),
        "total_smc_feature_count": len(smc_feature_columns),
    }
    for metric in PREDICTION_METRICS:
        row[f"original_{metric}"] = original_prediction[metric]
        row[f"smc_{metric}"] = smc_prediction[metric]
    for metric in TRADING_METRICS:
        row[f"original_{metric}"] = original_trading[metric]
        row[f"smc_{metric}"] = smc_trading[metric]

    row["smc_improvement"] = float(smc_trading["return_pct"] - original_trading["return_pct"])
    row["better_model"] = "SMC" if row["smc_improvement"] > 0 else ("Original" if row["smc_improvement"] < 0 else "Tie")
    return row


def _write_html_report(comparison: pd.DataFrame, result: ModelComparisonResult) -> None:
    COMPARISON_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    display = comparison.copy()
    numeric_columns = display.select_dtypes(include=[np.number]).columns
    display[numeric_columns] = display[numeric_columns].round(6)

    summary_html = f"""
    <h1>V4 Model Comparison</h1>
    <p>Analysis-only comparison of the original CNN/LSTM model and optional SMC-enhanced model.</p>
    <ul>
      <li>Assets compared: {result.assets_compared}</li>
      <li>SMC better: {len(result.smc_better_assets)}</li>
      <li>Original better: {len(result.original_better_assets)}</li>
      <li>Average SMC improvement: {result.average_improvement:.6f}%</li>
      <li>Median SMC improvement: {result.median_improvement:.6f}%</li>
    </ul>
    """
    table_html = display.to_html(index=False, classes="comparison", border=0)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>V4 Model Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ position: sticky; top: 0; background: #f0f4f8; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
  </style>
</head>
<body>
{summary_html}
{table_html}
</body>
</html>
"""
    COMPARISON_HTML_PATH.write_text(html, encoding="utf-8")


def run_model_comparison(timeframe: str = "1h", starting_capital: float = 100000.0) -> ModelComparisonResult:
    """Run an analysis-only comparison of the original model and optional SMC model."""
    _require_file(ORIGINAL_MODEL_PATH, "Original model")
    _require_file(ORIGINAL_SCALER_PATH, "Original scaler")
    _require_file(SMC_MODEL_PATH, "SMC model")
    _require_file(SMC_SCALER_PATH, "SMC scaler")

    dataset = _load_comparison_data(timeframe)
    original_model = load_model(ORIGINAL_MODEL_PATH)
    smc_model = load_model(SMC_MODEL_PATH)
    original_scaler = _load_pickle(ORIGINAL_SCALER_PATH)
    smc_scaler = _load_pickle(SMC_SCALER_PATH)

    rows = []
    for symbol, group in dataset.groupby("symbol", sort=True):
        try:
            row = _build_symbol_row(
                symbol=symbol,
                group=group.sort_values("timestamp").reset_index(drop=True),
                original_model=original_model,
                original_scaler=original_scaler,
                smc_model=smc_model,
                smc_scaler=smc_scaler,
                starting_capital=starting_capital,
            )
        except Exception as exc:
            logger.warning("Model comparison failed for %s: %s", symbol, exc)
            continue
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError("No assets produced model comparison results")

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("smc_improvement", ascending=False).reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))

    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_CSV_PATH, index=False)

    improvements = comparison["smc_improvement"].astype(float)
    smc_better_assets = comparison.loc[comparison["smc_improvement"] > 0, "symbol"].astype(str).tolist()
    original_better_assets = comparison.loc[comparison["smc_improvement"] < 0, "symbol"].astype(str).tolist()
    result = ModelComparisonResult(
        assets_compared=int(len(comparison)),
        smc_better_assets=smc_better_assets,
        original_better_assets=original_better_assets,
        average_improvement=float(improvements.mean()),
        median_improvement=float(improvements.median()),
        csv_path=COMPARISON_CSV_PATH,
        html_path=COMPARISON_HTML_PATH,
        comparison=comparison,
    )
    _write_html_report(comparison, result)
    return result
