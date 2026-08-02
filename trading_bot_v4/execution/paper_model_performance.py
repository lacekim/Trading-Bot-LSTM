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
PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH = Path("logs/v4_paper_model_performance_constrained.csv")
PAPER_MODEL_PERFORMANCE_DEBUG_TEMPLATE = "v4_paper_model_performance_debug_{symbol}.csv"


@dataclass(frozen=True)
class PaperModelPerformanceResult:
    assets_compared: int
    smc_better_assets: list[str]
    original_better_assets: list[str]
    average_return_difference: float
    average_drawdown_difference: float
    csv_path: Path | None
    html_path: Path | None
    debug_path: Path | None
    report_df: pd.DataFrame


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _fee_rate() -> float:
    return float(getattr(Config, "PAPER_FEE_BPS", 0.0)) / 10000.0


def _slippage_rate() -> float:
    return float(getattr(Config, "PAPER_SLIPPAGE_BPS", 0.0)) / 10000.0


def _constraints_used() -> dict[str, float | int | bool]:
    return {
        "paper_max_risk_per_trade_pct": float(Config.PAPER_MAX_RISK_PER_TRADE),
        "paper_min_bars_between_trades": int(Config.PAPER_MIN_BARS_BETWEEN_TRADES),
        "paper_fee_bps": float(Config.PAPER_FEE_BPS),
        "paper_slippage_bps": float(Config.PAPER_SLIPPAGE_BPS),
        "paper_max_trades_per_day": int(Config.PAPER_MAX_TRADES_PER_DAY),
        "paper_max_daily_loss_pct": float(Config.PAPER_MAX_DAILY_LOSS_PCT),
        "paper_use_compounding": bool(Config.PAPER_USE_COMPOUNDING),
    }


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


def _trade_result(direction: str, profit: float) -> str:
    if profit > 0:
        return f"{direction}_WIN"
    if profit < 0:
        return f"{direction}_LOSS"
    return f"{direction}_FLAT"


def _simulate_prepared_signals(
    signal_frame: pd.DataFrame,
    direction_column: str,
    starting_capital: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    if signal_frame.empty:
        return _empty_metrics(starting_capital), pd.DataFrame(columns=["timestamp", "trade_result", "capital"])

    signal_frame = signal_frame.sort_values("timestamp").reset_index(drop=True)
    fee_rate = _fee_rate()
    slippage_rate = _slippage_rate()
    capital = float(starting_capital)
    equity = [capital]
    trade_returns_pct: list[float] = []
    account_trade_returns_pct: list[float] = []
    debug_rows: list[dict[str, Any]] = []
    min_bars_between_trades = int(Config.PAPER_MIN_BARS_BETWEEN_TRADES)
    max_trades_per_day = int(Config.PAPER_MAX_TRADES_PER_DAY)
    max_daily_loss_pct = float(Config.PAPER_MAX_DAILY_LOSS_PCT)
    risk_pct = float(Config.PAPER_MAX_RISK_PER_TRADE)
    use_compounding = bool(Config.PAPER_USE_COMPOUNDING)
    last_trade_index: int | None = None
    current_day = None
    day_start_capital = capital
    trades_today = 0

    for row_index, row in signal_frame.iterrows():
        timestamp = row.get("timestamp")
        timestamp_value = pd.Timestamp(timestamp)
        day = timestamp_value.date()
        if current_day != day:
            current_day = day
            day_start_capital = capital
            trades_today = 0

        direction = str(row.get(direction_column, "HOLD")).upper()
        if direction not in {"LONG", "SHORT"}:
            equity.append(capital)
            debug_rows.append({"timestamp": timestamp, "trade_result": "HOLD", "capital": capital})
            continue

        if last_trade_index is not None and (row_index - last_trade_index) < min_bars_between_trades:
            equity.append(capital)
            debug_rows.append({"timestamp": timestamp, "trade_result": "BLOCKED_COOLDOWN", "capital": capital})
            continue

        if max_trades_per_day > 0 and trades_today >= max_trades_per_day:
            equity.append(capital)
            debug_rows.append({"timestamp": timestamp, "trade_result": "BLOCKED_DAILY_TRADE_LIMIT", "capital": capital})
            continue

        daily_loss_pct = ((capital / day_start_capital) - 1.0) * 100.0 if day_start_capital else 0.0
        if daily_loss_pct <= -max_daily_loss_pct:
            equity.append(capital)
            debug_rows.append({"timestamp": timestamp, "trade_result": "BLOCKED_DAILY_LOSS", "capital": capital})
            continue

        if (
            pd.isna(row.get("Close"))
            or pd.isna(row.get("ATR"))
            or pd.isna(row.get("High_next"))
            or pd.isna(row.get("Low_next"))
            or pd.isna(row.get("Close_next"))
        ):
            equity.append(capital)
            debug_rows.append({"timestamp": timestamp, "trade_result": "SKIPPED_MISSING_EXECUTION", "capital": capital})
            continue

        entry_price = float(row["Close"])
        atr = float(row["ATR"])
        if entry_price <= 0 or atr <= 0:
            equity.append(capital)
            debug_rows.append({"timestamp": timestamp, "trade_result": "SKIPPED_INVALID_EXECUTION", "capital": capital})
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
        sizing_capital = capital if use_compounding else float(starting_capital)
        risk_amount = sizing_capital * (risk_pct / 100.0)
        risk_size = risk_amount / stop_distance if stop_distance else 0.0
        max_affordable = capital / effective_entry_price if effective_entry_price else 0.0
        units = min(risk_size, max_affordable * 0.95)
        if units <= 0:
            equity.append(capital)
            debug_rows.append({"timestamp": timestamp, "trade_result": "SKIPPED_ZERO_SIZE", "capital": capital})
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
        capital_before_trade = capital
        capital += profit
        notional = max(effective_entry_price * units, 1e-12)
        trade_returns_pct.append((profit / notional) * 100.0)
        account_trade_returns_pct.append((profit / max(capital_before_trade, 1e-12)) * 100.0)
        equity.append(capital)
        last_trade_index = int(row_index)
        trades_today += 1
        debug_rows.append({"timestamp": timestamp, "trade_result": _trade_result(direction, profit), "capital": capital})

    if not trade_returns_pct:
        return _empty_metrics(starting_capital), pd.DataFrame(debug_rows)

    trade_returns = np.array(account_trade_returns_pct, dtype=float)
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

    metrics = {
        "return_pct": float(return_pct),
        "final_capital": final_capital,
        "max_drawdown_pct": max_drawdown,
        "profit_factor": float(profit_factor),
        "win_rate_pct": win_rate,
        "trade_count": int(len(trade_returns)),
        "average_trade_pct": float(trade_returns.mean()),
        "expectancy": float((win_rate_decimal * avg_win) + ((1.0 - win_rate_decimal) * avg_loss)),
    }
    return metrics, pd.DataFrame(debug_rows)


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
    metrics, _ = _simulate_prepared_signals(signal_frame, direction_column, starting_capital)
    return metrics


def _load_smc_signals() -> pd.DataFrame:
    _require_file(SMC_MODEL_PAPER_SIGNALS_PATH, "SMC model paper signals")
    signals = pd.read_csv(SMC_MODEL_PAPER_SIGNALS_PATH)
    required = ["timestamp", "symbol", "timeframe", "model_probability", "model_direction", "is_trade_candidate"]
    missing = [column for column in required if column not in signals.columns]
    if missing:
        raise ValueError(f"Missing SMC paper signal columns: {missing}")

    signals["timestamp"] = pd.to_datetime(signals["timestamp"], errors="coerce")
    signals["symbol"] = signals["symbol"].astype(str).str.upper()
    signals["timeframe"] = signals["timeframe"].astype(str)
    signals["model_probability"] = pd.to_numeric(signals["model_probability"], errors="coerce")
    signals["model_direction"] = signals["model_direction"].astype(str).str.upper()
    return signals.dropna(subset=["timestamp", "symbol", "timeframe", "model_probability"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


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


def _prepare_merged_signals(
    original: pd.DataFrame,
    symbol_smc: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    original_duplicates = int(original.duplicated(subset=["timestamp", "symbol", "timeframe"]).sum())
    smc_duplicates = int(symbol_smc.duplicated(subset=["timestamp", "symbol", "timeframe"]).sum())
    if original_duplicates:
        logger.warning("Dropping %s duplicate original signal timestamps for %s %s", original_duplicates, symbol, timeframe)
    if smc_duplicates:
        logger.warning("Dropping %s duplicate SMC signal timestamps for %s %s", smc_duplicates, symbol, timeframe)

    original = original.drop_duplicates(subset=["timestamp", "symbol", "timeframe"], keep="last")
    smc = symbol_smc.rename(columns={"model_direction": "smc_direction", "is_trade_candidate": "smc_candidate"})
    smc = smc.drop_duplicates(subset=["timestamp", "symbol", "timeframe"], keep="last")
    merged = original.merge(
        smc[["timestamp", "symbol", "timeframe", "model_probability", "smc_direction", "smc_candidate"]],
        on=["timestamp", "symbol", "timeframe"],
        how="inner",
    ).rename(columns={"model_probability": "smc_probability"})
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    if merged["timestamp"].duplicated().any():
        raise ValueError(f"Duplicate merged timestamps for {symbol} {timeframe}")
    return merged


def _build_debug_frame(
    symbol: str,
    timeframe: str,
    merged: pd.DataFrame,
    starting_capital: float,
) -> tuple[dict[str, float | int], dict[str, float | int], pd.DataFrame]:
    execution = _build_execution_frame(symbol, timeframe, merged["timestamp"]).reset_index(drop=True)
    base = merged.reset_index(drop=True).join(execution)
    original_metrics, original_debug = _simulate_prepared_signals(
        base[["timestamp", "original_direction", "Close", "High_next", "Low_next", "Close_next", "ATR"]].copy(),
        "original_direction",
        starting_capital,
    )
    smc_metrics, smc_debug = _simulate_prepared_signals(
        base[["timestamp", "smc_direction", "Close", "High_next", "Low_next", "Close_next", "ATR"]].copy(),
        "smc_direction",
        starting_capital,
    )

    debug = pd.DataFrame(
        {
            "timestamp": base["timestamp"],
            "close": base["Close"],
            "high_next": base["High_next"],
            "low_next": base["Low_next"],
            "close_next": base["Close_next"],
            "ATR": base["ATR"],
            "original_probability": base["original_probability"],
            "smc_probability": base["smc_probability"],
            "original_signal": base["original_direction"],
            "smc_signal": base["smc_direction"],
            "original_trade_result": original_debug["trade_result"].to_numpy(),
            "smc_trade_result": smc_debug["trade_result"].to_numpy(),
            "original_capital": original_debug["capital"].to_numpy(dtype=float),
            "smc_capital": smc_debug["capital"].to_numpy(dtype=float),
        }
    )
    return original_metrics, smc_metrics, debug


def _build_symbol_performance(
    symbol: str,
    timeframe: str,
    original_model: Any,
    original_scaler: Any,
    smc_signals: pd.DataFrame,
    comparison_row: pd.Series,
    starting_capital: float,
    debug: bool = False,
) -> dict[str, Any] | None:
    original = _predict_original_model_signals(original_model, original_scaler, symbol, timeframe)
    symbol_smc = smc_signals.loc[(smc_signals["symbol"] == symbol) & (smc_signals["timeframe"] == timeframe)].copy()
    if symbol_smc.empty:
        logger.warning("Skipping %s %s: no saved SMC paper signals", symbol, timeframe)
        return None

    merged = _prepare_merged_signals(original, symbol_smc, symbol, timeframe)
    if merged.empty:
        logger.warning("Skipping %s %s: no shared signal timestamps", symbol, timeframe)
        return None

    debug_frame = pd.DataFrame()
    if debug:
        original_metrics, smc_metrics, debug_frame = _build_debug_frame(symbol, timeframe, merged, starting_capital)
    else:
        execution = _build_execution_frame(symbol, timeframe, merged["timestamp"]).reset_index(drop=True)
        base = merged.reset_index(drop=True).join(execution)
        original_metrics, _ = _simulate_prepared_signals(
            base[["timestamp", "original_direction", "Close", "High_next", "Low_next", "Close_next", "ATR"]].copy(),
            "original_direction",
            starting_capital,
        )
        smc_metrics, _ = _simulate_prepared_signals(
            base[["timestamp", "smc_direction", "Close", "High_next", "Low_next", "Close_next", "ATR"]].copy(),
            "smc_direction",
            starting_capital,
        )

    return_difference = float(smc_metrics["return_pct"] - original_metrics["return_pct"])
    drawdown_difference = float(smc_metrics["max_drawdown_pct"] - original_metrics["max_drawdown_pct"])
    aggressiveness = str(comparison_row["aggressiveness"])
    row = {
        "symbol": symbol,
        "timeframe": timeframe,
        **_constraints_used(),
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
        "original_average_trade_pct": float(original_metrics["average_trade_pct"]),
        "smc_average_trade_pct": float(smc_metrics["average_trade_pct"]),
        "original_trade_count": int(original_metrics["trade_count"]),
        "smc_trade_count": int(smc_metrics["trade_count"]),
        "smc_aggression_performance_effect": _aggression_effect(aggressiveness, return_difference),
    }
    if debug:
        row["_debug_frame"] = debug_frame
    return row


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
    debug = bool(getattr(args, "debug", False))
    selected_symbol = str(getattr(args, "symbol", Config.GMX_SYMBOL)).upper()
    comparison = _load_signal_comparison()
    smc_signals = _load_smc_signals()

    comparison = comparison.loc[comparison["timeframe"] == timeframe].copy()
    if comparison.empty:
        raise ValueError(f"No paper model comparison rows for timeframe {timeframe}")
    if debug:
        comparison = comparison.loc[comparison["symbol"] == selected_symbol].copy()
        if comparison.empty:
            raise ValueError(f"No paper model comparison row for {selected_symbol} {timeframe}")

    original_model, original_scaler = ModelScalerCache().load()
    rows = []
    debug_frame = pd.DataFrame()
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
                debug=debug and symbol == selected_symbol,
            )
        except Exception as exc:
            logger.warning("Skipping %s %s during paper performance comparison: %s", symbol, timeframe, exc)
            continue
        if row is not None:
            if "_debug_frame" in row:
                debug_frame = row.pop("_debug_frame")
            rows.append(row)

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values("return_difference_pct", ascending=False).reset_index(drop=True)

    csv_path = None
    html_path = None
    if not debug:
        PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH, index=False)
        _write_html_report(report, PAPER_MODEL_PERFORMANCE_HTML_PATH)
        csv_path = PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH
        html_path = PAPER_MODEL_PERFORMANCE_HTML_PATH

    debug_path = None
    if debug and not debug_frame.empty:
        debug_path = Path("logs") / PAPER_MODEL_PERFORMANCE_DEBUG_TEMPLATE.format(symbol=selected_symbol)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_frame.to_csv(debug_path, index=False)

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
        csv_path=csv_path,
        html_path=html_path,
        debug_path=debug_path,
        report_df=report,
    )
