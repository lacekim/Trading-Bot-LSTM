"""Paper-only performance comparison for original and SMC model signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import load_gmx_ohlc
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.execution.paper_model_comparison import (
    PAPER_MODEL_COMPARISON_CSV_PATH,
    _predict_original_model_signals,
)
from trading_bot_v4.execution.smc_model_paper import SMC_MODEL_PAPER_SIGNALS_PATH
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache


logger = build_logger("v4_paper_model_performance")

PAPER_MODEL_PERFORMANCE_CSV_PATH = Path("logs/v4_paper_model_performance.csv")
PAPER_MODEL_PERFORMANCE_HTML_PATH = Path("logs/v4_paper_model_performance.html")


@dataclass(frozen=True)
class PaperModelPerformanceResult:
    assets_compared: int
    smc_better_assets: list[str]
    original_better_assets: list[str]
    average_return_difference: float
    average_drawdown_difference: float
    csv_path: Path
    html_path: Path
    report_df: pd.DataFrame


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _fee_rate() -> float:
    return float(getattr(Config, "FEE_RATE", getattr(Config, "TRADING_FEE_RATE", getattr(Config, "COMMISSION_RATE", 0.0))))


def _slippage_rate() -> float:
    return float(getattr(Config, "SLIPPAGE_RATE", getattr(Config, "SLIPPAGE_PCT", 0.0)))


def _empty_metrics(starting_capital: float) -> dict[str, float | int]:
    return {
        "return_pct": 0.0,
        "final_capital": float(starting_capital),
        "max_drawdown_pct": 0.0,
        "profit_factor": 0.0,
        "win_rate_pct": 0.0,
        "trade_count": 0,
        "average_trade_pct": 0.0,
        "expectancy": 0.0,
    }


def _build_execution_frame(symbol: str, timeframe: str, timestamps: pd.Series) -> pd.DataFrame:
    raw = load_gmx_ohlc(symbol, timeframe).copy()
    handler = V4DataHandler()
    raw["ATR"] = handler.calculate_atr(raw, Config.ATR_PERIOD)
    execution = raw[["Close", "High", "Low", "ATR"]].copy()
    execution["High_next"] = execution["High"].shift(-1)
    execution["Low_next"] = execution["Low"].shift(-1)
    execution["Close_next"] = execution["Close"].shift(-1)
    execution = execution.reindex(pd.DatetimeIndex(timestamps))
    return execution.replace([np.inf, -np.inf], np.nan)


def _simulate_direction_signals(
    symbol: str,
    timeframe: str,
    signals: pd.DataFrame,
    direction_column: str,
    starting_capital: float,
) -> dict[str, float | int]:
    if signals.empty:
        return _empty_metrics(starting_capital)

    signals = signals.sort_values("timestamp").reset_index(drop=True)
    execution = _build_execution_frame(symbol, timeframe, signals["timestamp"])
    signal_frame = signals.join(execution.reset_index(drop=True))

    fee_rate = _fee_rate()
    slippage_rate = _slippage_rate()
    capital = float(starting_capital)
    equity = [capital]
    trade_returns_pct: list[float] = []

    for _, row in signal_frame.iterrows():
        direction = str(row.get(direction_column, "HOLD")).upper()
        if direction not in {"LONG", "SHORT"}:
            equity.append(capital)
            continue

        if (
            pd.isna(row.get("Close"))
            or pd.isna(row.get("ATR"))
            or pd.isna(row.get("High_next"))
            or pd.isna(row.get("Low_next"))
            or pd.isna(row.get("Close_next"))
        ):
            equity.append(capital)
            continue

        entry_price = float(row["Close"])
        atr = float(row["ATR"])
        if entry_price <= 0 or atr <= 0:
            equity.append(capital)
            continue

        if direction == "LONG":
            effective_entry_price = entry_price * (1.0 + slippage_rate)
            stop_loss = effective_entry_price - atr * Config.ATR_SL_MULTIPLIER
            take_profit = effective_entry_price + atr * Config.ATR_TP_MULTIPLIER
        else:
            effective_entry_price = entry_price * (1.0 - slippage_rate)
            stop_loss = effective_entry_price + atr * Config.ATR_SL_MULTIPLIER
            take_profit = effective_entry_price - atr * Config.ATR_TP_MULTIPLIER

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
        notional = max(effective_entry_price * units, 1e-12)
        trade_returns_pct.append((profit / notional) * 100.0)
        equity.append(capital)

    if not trade_returns_pct:
        return _empty_metrics(starting_capital)

    trade_returns = np.array(trade_returns_pct, dtype=float)
    equity_array = np.array(equity, dtype=float)
    winning = trade_returns[trade_returns > 0]
    losing = trade_returns[trade_returns < 0]
    gross_profit = float(winning.sum()) if len(winning) else 0.0
    gross_loss = abs(float(losing.sum())) if len(losing) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    running_max = np.maximum.accumulate(equity_array)
    max_drawdown = float(((equity_array / running_max) - 1.0).min() * 100.0) if len(equity_array) else 0.0
    final_capital = float(capital)
    return_pct = ((final_capital / float(starting_capital)) - 1.0) * 100.0 if starting_capital else 0.0
    win_rate = float(len(winning) / len(trade_returns) * 100.0) if len(trade_returns) else 0.0
    avg_win = float(winning.mean()) if len(winning) else 0.0
    avg_loss = float(losing.mean()) if len(losing) else 0.0
    win_rate_decimal = win_rate / 100.0

    return {
        "return_pct": float(return_pct),
        "final_capital": final_capital,
        "max_drawdown_pct": max_drawdown,
        "profit_factor": float(profit_factor),
        "win_rate_pct": win_rate,
        "trade_count": int(len(trade_returns)),
        "average_trade_pct": float(trade_returns.mean()),
        "expectancy": float((win_rate_decimal * avg_win) + ((1.0 - win_rate_decimal) * avg_loss)),
    }


def _load_smc_signals() -> pd.DataFrame:
    _require_file(SMC_MODEL_PAPER_SIGNALS_PATH, "SMC model paper signals")
    signals = pd.read_csv(SMC_MODEL_PAPER_SIGNALS_PATH)
    required = ["timestamp", "symbol", "timeframe", "model_direction", "is_trade_candidate"]
    missing = [column for column in required if column not in signals.columns]
    if missing:
        raise ValueError(f"Missing SMC paper signal columns: {missing}")

    signals["timestamp"] = pd.to_datetime(signals["timestamp"], errors="coerce")
    signals["symbol"] = signals["symbol"].astype(str).str.upper()
    signals["timeframe"] = signals["timeframe"].astype(str)
    signals["model_direction"] = signals["model_direction"].astype(str).str.upper()
    return signals.dropna(subset=["timestamp", "symbol", "timeframe"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _load_signal_comparison() -> pd.DataFrame:
    _require_file(PAPER_MODEL_COMPARISON_CSV_PATH, "paper model signal comparison")
    comparison = pd.read_csv(PAPER_MODEL_COMPARISON_CSV_PATH)
    required = ["symbol", "timeframe", "original_candidates", "smc_candidates", "aggressiveness"]
    missing = [column for column in required if column not in comparison.columns]
    if missing:
        raise ValueError(f"Missing paper model comparison columns: {missing}")
    comparison["symbol"] = comparison["symbol"].astype(str).str.upper()
    comparison["timeframe"] = comparison["timeframe"].astype(str)
    return comparison


def _aggression_effect(aggressiveness: str, return_difference: float) -> str:
    if aggressiveness == "same candidate count":
        return "neutral candidate count"
    outcome = "improved performance" if return_difference > 0 else "hurt performance"
    return f"{aggressiveness} {outcome}"


def _build_symbol_performance(
    symbol: str,
    timeframe: str,
    original_model: Any,
    original_scaler: Any,
    smc_signals: pd.DataFrame,
    comparison_row: pd.Series,
    starting_capital: float,
) -> dict[str, Any] | None:
    original = _predict_original_model_signals(original_model, original_scaler, symbol, timeframe)
    symbol_smc = smc_signals.loc[(smc_signals["symbol"] == symbol) & (smc_signals["timeframe"] == timeframe)].copy()
    if symbol_smc.empty:
        logger.warning("Skipping %s %s: no saved SMC paper signals", symbol, timeframe)
        return None

    smc = symbol_smc.rename(columns={"model_direction": "smc_direction", "is_trade_candidate": "smc_candidate"})
    merged = original.merge(
        smc[["timestamp", "symbol", "timeframe", "smc_direction", "smc_candidate"]],
        on=["timestamp", "symbol", "timeframe"],
        how="inner",
    ).sort_values("timestamp")
    if merged.empty:
        logger.warning("Skipping %s %s: no shared signal timestamps", symbol, timeframe)
        return None

    original_metrics = _simulate_direction_signals(
        symbol=symbol,
        timeframe=timeframe,
        signals=merged[["timestamp", "original_direction"]],
        direction_column="original_direction",
        starting_capital=starting_capital,
    )
    smc_metrics = _simulate_direction_signals(
        symbol=symbol,
        timeframe=timeframe,
        signals=merged[["timestamp", "smc_direction"]],
        direction_column="smc_direction",
        starting_capital=starting_capital,
    )

    return_difference = float(smc_metrics["return_pct"] - original_metrics["return_pct"])
    drawdown_difference = float(smc_metrics["max_drawdown_pct"] - original_metrics["max_drawdown_pct"])
    aggressiveness = str(comparison_row["aggressiveness"])
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "shared_timestamps": int(len(merged)),
        "original_candidates": int(comparison_row["original_candidates"]),
        "smc_candidates": int(comparison_row["smc_candidates"]),
        "aggressiveness": aggressiveness,
        "original_return_pct": float(original_metrics["return_pct"]),
        "smc_return_pct": float(smc_metrics["return_pct"]),
        "return_difference_pct": return_difference,
        "original_max_drawdown_pct": float(original_metrics["max_drawdown_pct"]),
        "smc_max_drawdown_pct": float(smc_metrics["max_drawdown_pct"]),
        "drawdown_difference_pct": drawdown_difference,
        "original_profit_factor": float(original_metrics["profit_factor"]),
        "smc_profit_factor": float(smc_metrics["profit_factor"]),
        "original_win_rate_pct": float(original_metrics["win_rate_pct"]),
        "smc_win_rate_pct": float(smc_metrics["win_rate_pct"]),
        "original_trade_count": int(original_metrics["trade_count"]),
        "smc_trade_count": int(smc_metrics["trade_count"]),
        "smc_aggression_performance_effect": _aggression_effect(aggressiveness, return_difference),
    }


def _write_html_report(report: pd.DataFrame, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    display = report.copy()
    numeric_columns = display.select_dtypes(include=[np.number]).columns
    display[numeric_columns] = display[numeric_columns].round(6)
    smc_better = int((display["return_difference_pct"] > 0).sum()) if not display.empty else 0
    original_better = int((display["return_difference_pct"] < 0).sum()) if not display.empty else 0
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>V4 Paper Model Performance</title>
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
  <h1>V4 Paper Model Performance</h1>
  <p>Paper-only V4-style simulated execution of original and SMC model signals on shared timestamps.</p>
  <ul>
    <li>Assets compared: {len(display)}</li>
    <li>SMC better assets: {smc_better}</li>
    <li>Original better assets: {original_better}</li>
  </ul>
  {display.to_html(index=False, classes="performance", border=0)}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def run_paper_model_performance(args: Any) -> PaperModelPerformanceResult:
    """Compare paper signal performance without placing orders."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    starting_capital = float(getattr(args, "capital", 100000.0))
    comparison = _load_signal_comparison()
    smc_signals = _load_smc_signals()

    comparison = comparison.loc[comparison["timeframe"] == timeframe].copy()
    if comparison.empty:
        raise ValueError(f"No paper model comparison rows for timeframe {timeframe}")

    original_model, original_scaler = ModelScalerCache().load()
    rows = []
    for _, comparison_row in comparison.sort_values("symbol").iterrows():
        symbol = str(comparison_row["symbol"]).upper()
        try:
            row = _build_symbol_performance(
                symbol=symbol,
                timeframe=timeframe,
                original_model=original_model,
                original_scaler=original_scaler,
                smc_signals=smc_signals,
                comparison_row=comparison_row,
                starting_capital=starting_capital,
            )
        except Exception as exc:
            logger.warning("Skipping %s %s during paper performance comparison: %s", symbol, timeframe, exc)
            continue
        if row is not None:
            rows.append(row)

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values("return_difference_pct", ascending=False).reset_index(drop=True)

    PAPER_MODEL_PERFORMANCE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(PAPER_MODEL_PERFORMANCE_CSV_PATH, index=False)
    _write_html_report(report, PAPER_MODEL_PERFORMANCE_HTML_PATH)

    smc_better_assets = report.loc[report["return_difference_pct"] > 0, "symbol"].astype(str).tolist() if not report.empty else []
    original_better_assets = report.loc[report["return_difference_pct"] < 0, "symbol"].astype(str).tolist() if not report.empty else []
    average_return_difference = float(report["return_difference_pct"].mean()) if not report.empty else 0.0
    average_drawdown_difference = float(report["drawdown_difference_pct"].mean()) if not report.empty else 0.0
    return PaperModelPerformanceResult(
        assets_compared=int(len(report)),
        smc_better_assets=smc_better_assets,
        original_better_assets=original_better_assets,
        average_return_difference=average_return_difference,
        average_drawdown_difference=average_drawdown_difference,
        csv_path=PAPER_MODEL_PERFORMANCE_CSV_PATH,
        html_path=PAPER_MODEL_PERFORMANCE_HTML_PATH,
        report_df=report,
    )
