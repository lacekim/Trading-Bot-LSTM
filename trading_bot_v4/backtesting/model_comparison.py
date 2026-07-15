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

from trading_bot import load_gmx_ohlc
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
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
COMPARISON_TEST_ONLY_CSV_PATH = Path("models/model_comparison_test_only.csv")
DEBUG_TEMPLATE = "model_comparison_debug_{symbol}.csv"
TEST_ONLY_FRACTION = 0.15

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
    debug_path: Path | None
    debug_sample: pd.DataFrame
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


def _empty_trading_metrics(starting_capital: float) -> dict[str, float]:
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


def _build_execution_frame(symbol: str, timeframe: str, timestamps: pd.Series) -> pd.DataFrame:
    raw = load_gmx_ohlc(symbol, timeframe)
    handler = V4DataHandler()
    raw = raw.copy()
    raw["ATR"] = handler.calculate_atr(raw, Config.ATR_PERIOD)
    execution = raw[["Close", "High", "Low", "ATR"]].copy()
    execution["High_next"] = execution["High"].shift(-1)
    execution["Low_next"] = execution["Low"].shift(-1)
    execution["Close_next"] = execution["Close"].shift(-1)
    execution = execution.reindex(pd.DatetimeIndex(timestamps))
    execution = execution.replace([np.inf, -np.inf], np.nan)
    return execution


def _fee_rate() -> float:
    return float(getattr(Config, "FEE_RATE", getattr(Config, "TRADING_FEE_RATE", getattr(Config, "COMMISSION_RATE", 0.0))))


def _slippage_rate() -> float:
    return float(getattr(Config, "SLIPPAGE_RATE", getattr(Config, "SLIPPAGE_PCT", 0.0)))


def _trade_return_pct(entry_price: float, exit_price: float, direction: str) -> float:
    if direction == "LONG":
        return ((exit_price / entry_price) - 1.0) * 100.0
    return ((entry_price / exit_price) - 1.0) * 100.0


def _trading_metrics(signals: pd.DataFrame, probabilities: np.ndarray, starting_capital: float) -> dict[str, float]:
    if signals.empty or len(probabilities) == 0:
        return _empty_trading_metrics(starting_capital)

    threshold = float(Config.MIN_SIGNAL_THRESHOLD)
    fee_rate = _fee_rate()
    slippage_rate = _slippage_rate()
    capital = float(starting_capital)
    trade_returns_pct: list[float] = []
    equity: list[float] = []

    for (_, row), probability in zip(signals.iterrows(), probabilities):
        if pd.isna(row.get("Close")) or pd.isna(row.get("ATR")):
            equity.append(capital)
            continue

        probability = float(probability)
        entry_price = float(row["Close"])
        atr = float(row["ATR"])
        if probability > threshold:
            direction = "LONG"
            effective_entry_price = entry_price * (1.0 + slippage_rate)
            stop_loss = effective_entry_price - atr * Config.ATR_SL_MULTIPLIER
            take_profit = effective_entry_price + atr * Config.ATR_TP_MULTIPLIER
        else:
            equity.append(capital)
            continue

        if pd.isna(row.get("High_next")) or pd.isna(row.get("Low_next")) or pd.isna(row.get("Close_next")):
            equity.append(capital)
            continue

        stop_distance = abs(effective_entry_price - stop_loss)
        risk_amount = capital * (Config.RISK_PERCENTAGE / 100.0)
        risk_size = risk_amount / stop_distance if stop_distance else 0.0
        max_affordable = capital / effective_entry_price if effective_entry_price else 0.0
        units = min(risk_size, max_affordable * 0.95)
        if units <= 0:
            equity.append(capital)
            continue

        high_next = float(row["High_next"])
        low_next = float(row["Low_next"])
        close_next = float(row["Close_next"])
        if direction == "LONG":
            if low_next <= stop_loss:
                exit_price = stop_loss
            elif high_next >= take_profit:
                exit_price = take_profit
            else:
                exit_price = close_next * (1.0 - slippage_rate)
            profit = (exit_price - effective_entry_price) * units
        else:
            if high_next >= stop_loss:
                exit_price = stop_loss
            elif low_next <= take_profit:
                exit_price = take_profit
            else:
                exit_price = close_next * (1.0 + slippage_rate)
            profit = (effective_entry_price - exit_price) * units

        fees = (effective_entry_price * units * fee_rate) + (exit_price * units * fee_rate)
        profit -= fees

        capital += profit
        trade_returns_pct.append((profit / max(effective_entry_price * units, 1e-12)) * 100.0)
        equity.append(capital)

    if not trade_returns_pct:
        return _empty_trading_metrics(starting_capital)

    trade_returns_pct_array = np.array(trade_returns_pct, dtype=float)
    equity_array = np.array(equity, dtype=float) if equity else np.array([starting_capital], dtype=float)
    final_capital = float(capital)

    winning = trade_returns_pct_array[trade_returns_pct_array > 0]
    losing = trade_returns_pct_array[trade_returns_pct_array < 0]
    gross_profit = float(winning.sum()) if len(winning) else 0.0
    gross_loss = abs(float(losing.sum())) if len(losing) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    mean_return = float(trade_returns_pct_array.mean())
    std_return = float(trade_returns_pct_array.std(ddof=0)) if len(trade_returns_pct_array) > 1 else 0.0
    downside_std = float(losing.std(ddof=0)) if len(losing) > 1 else 0.0
    return_pct = ((final_capital / float(starting_capital)) - 1.0) * 100.0 if starting_capital else 0.0
    running_max = np.maximum.accumulate(equity_array)
    max_drawdown = float(((equity_array / running_max) - 1.0).min() * 100.0) if len(equity_array) else 0.0
    win_rate = float(len(winning) / len(trade_returns_pct_array) * 100.0) if len(trade_returns_pct_array) else 0.0
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
        "trade_count": int(len(trade_returns_pct_array)),
        "win_rate_pct": win_rate,
        "average_trade_pct": mean_return,
        "expectancy": float((win_rate_decimal * avg_win) + ((1.0 - win_rate_decimal) * avg_loss)),
    }


def _select_evaluation_group(group: pd.DataFrame, test_only: bool) -> pd.DataFrame:
    group = group.sort_values("timestamp").reset_index(drop=True)
    if not test_only:
        return group
    test_rows = int(np.ceil(len(group) * TEST_ONLY_FRACTION))
    return group.tail(test_rows).reset_index(drop=True)


def _build_symbol_row(
    symbol: str,
    group: pd.DataFrame,
    original_model: Any,
    original_scaler: Any,
    smc_model: Any,
    smc_scaler: Any,
    starting_capital: float,
    timeframe: str,
    test_only: bool,
) -> dict[str, float | int | str] | None:
    group = _select_evaluation_group(group, test_only=test_only)
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
    execution = _build_execution_frame(symbol, timeframe, endpoint_frame["timestamp"])
    valid_execution = execution.dropna(subset=["Close", "High_next", "Low_next", "Close_next", "ATR"])
    if valid_execution.empty:
        logger.warning("Skipping %s: no executable comparison timestamps", symbol)
        return None

    valid_positions = execution.index.get_indexer(valid_execution.index)
    endpoint_frame = endpoint_frame.iloc[valid_positions].reset_index(drop=True)
    y_true = y_true[valid_positions]
    original_probabilities = original_probabilities[:row_count]
    smc_probabilities = smc_probabilities[:row_count]
    original_probabilities = original_probabilities[valid_positions]
    smc_probabilities = smc_probabilities[valid_positions]

    original_prediction = _prediction_metrics(y_true, original_probabilities)
    smc_prediction = _prediction_metrics(y_true, smc_probabilities)
    original_trading = _trading_metrics(valid_execution, original_probabilities, starting_capital)
    smc_trading = _trading_metrics(valid_execution, smc_probabilities, starting_capital)

    row: dict[str, float | int | str] = {
        "symbol": symbol,
        "rows_evaluated": int(len(endpoint_frame)),
        "evaluation_mode": "test_only" if test_only else "all_rows",
        "evaluation_start": endpoint_frame["timestamp"].min(),
        "evaluation_end": endpoint_frame["timestamp"].max(),
        "original_feature_count": len(original_feature_columns),
        "smc_feature_count": len(SMC_FEATURE_COLUMNS),
        "total_smc_feature_count": len(smc_feature_columns),
        "threshold": float(Config.MIN_SIGNAL_THRESHOLD),
        "risk_percentage": float(Config.RISK_PERCENTAGE),
        "atr_stop_loss_multiplier": float(Config.ATR_SL_MULTIPLIER),
        "atr_take_profit_multiplier": float(Config.ATR_TP_MULTIPLIER),
        "fee_rate": _fee_rate(),
        "slippage_rate": _slippage_rate(),
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


def _build_debug_predictions(
    symbol: str,
    group: pd.DataFrame,
    original_model: Any,
    original_scaler: Any,
    smc_model: Any,
    smc_scaler: Any,
    timeframe: str,
    test_only: bool,
) -> pd.DataFrame:
    group = _select_evaluation_group(group, test_only=test_only)
    if len(group) <= Config.SEQUENCE_LENGTH:
        return pd.DataFrame(columns=["timestamp", "original_probability", "smc_probability", "target", "close"])

    original_probabilities = _predict_symbol(original_model, original_scaler, group, list(Config.FEATURE_COLUMNS))
    smc_probabilities = _predict_symbol(smc_model, smc_scaler, group, [*Config.FEATURE_COLUMNS, *SMC_FEATURE_COLUMNS])
    row_count = min(len(original_probabilities), len(smc_probabilities))
    if row_count == 0:
        return pd.DataFrame(columns=["timestamp", "original_probability", "smc_probability", "target", "close"])

    endpoints = group.iloc[Config.SEQUENCE_LENGTH : Config.SEQUENCE_LENGTH + row_count].copy()
    execution = _build_execution_frame(symbol, timeframe, endpoints["timestamp"])
    valid_execution = execution.dropna(subset=["Close"])
    valid_positions = execution.index.get_indexer(valid_execution.index)

    debug = pd.DataFrame(
        {
            "timestamp": endpoints.iloc[valid_positions]["timestamp"].to_numpy(),
            "original_probability": original_probabilities[:row_count][valid_positions],
            "smc_probability": smc_probabilities[:row_count][valid_positions],
            "target": endpoints.iloc[valid_positions]["target"].to_numpy(dtype=int),
            "close": valid_execution["Close"].to_numpy(dtype=float),
        }
    )
    return debug


def _write_html_report(comparison: pd.DataFrame, result: ModelComparisonResult, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
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
    html_path.write_text(html, encoding="utf-8")


def run_model_comparison(
    timeframe: str = "1h",
    starting_capital: float = 100000.0,
    test_only: bool = False,
    debug_symbol: str = "GMX",
    symbol: str | None = None,
    all_assets: bool = True,
) -> ModelComparisonResult:
    """Run an analysis-only comparison of the original model and optional SMC model."""
    _require_file(ORIGINAL_MODEL_PATH, "Original model")
    _require_file(ORIGINAL_SCALER_PATH, "Original scaler")
    _require_file(SMC_MODEL_PATH, "SMC model")
    _require_file(SMC_SCALER_PATH, "SMC scaler")

    dataset = _load_comparison_data(timeframe)
    if not all_assets:
        selected_symbol = str(symbol or debug_symbol).upper()
        dataset = dataset.loc[dataset["symbol"] == selected_symbol].copy()
        if dataset.empty:
            raise ValueError(f"No comparison rows found for symbol {selected_symbol}")

    original_model = load_model(ORIGINAL_MODEL_PATH)
    smc_model = load_model(SMC_MODEL_PATH)
    original_scaler = _load_pickle(ORIGINAL_SCALER_PATH)
    smc_scaler = _load_pickle(SMC_SCALER_PATH)

    rows = []
    debug_frame = pd.DataFrame()
    debug_symbol = str(debug_symbol or "GMX").upper()
    for symbol, group in dataset.groupby("symbol", sort=True):
        sorted_group = group.sort_values("timestamp").reset_index(drop=True)
        if symbol == debug_symbol:
            debug_frame = _build_debug_predictions(
                symbol=symbol,
                group=sorted_group,
                original_model=original_model,
                original_scaler=original_scaler,
                smc_model=smc_model,
                smc_scaler=smc_scaler,
                timeframe=timeframe,
                test_only=test_only,
            )
        try:
            row = _build_symbol_row(
                symbol=symbol,
                group=sorted_group,
                original_model=original_model,
                original_scaler=original_scaler,
                smc_model=smc_model,
                smc_scaler=smc_scaler,
                starting_capital=starting_capital,
                timeframe=timeframe,
                test_only=test_only,
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

    csv_path = COMPARISON_TEST_ONLY_CSV_PATH if test_only else COMPARISON_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)

    debug_path = Path("models") / DEBUG_TEMPLATE.format(symbol=debug_symbol)
    if not debug_frame.empty:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_frame.to_csv(debug_path, index=False)
    else:
        debug_path = None

    improvements = comparison["smc_improvement"].astype(float)
    smc_better_assets = comparison.loc[comparison["smc_improvement"] > 0, "symbol"].astype(str).tolist()
    original_better_assets = comparison.loc[comparison["smc_improvement"] < 0, "symbol"].astype(str).tolist()
    result = ModelComparisonResult(
        assets_compared=int(len(comparison)),
        smc_better_assets=smc_better_assets,
        original_better_assets=original_better_assets,
        average_improvement=float(improvements.mean()),
        median_improvement=float(improvements.median()),
        csv_path=csv_path,
        html_path=COMPARISON_HTML_PATH,
        debug_path=debug_path,
        debug_sample=debug_frame.head(10).copy(),
        comparison=comparison,
    )
    _write_html_report(comparison, result, COMPARISON_HTML_PATH)
    return result
