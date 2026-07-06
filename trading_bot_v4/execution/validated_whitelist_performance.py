"""Performance report for validated whitelist SMC model paper signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.paper_model_performance import (
    _build_execution_frame,
    _constraints_used,
    _require_file,
    _simulate_prepared_signals,
)
from trading_bot_v4.execution.smc_model_paper import VALIDATED_WHITELIST_PAPER_SIGNALS_PATH


VALIDATED_WHITELIST_ASSETS = ["PENGU", "DYDX", "AIXBT"]
STRICT_VALIDATED_WHITELIST_ASSETS = ["AIXBT", "DYDX"]
VALIDATED_WHITELIST_PERFORMANCE_CSV_PATH = Path("logs/v4_validated_whitelist_performance.csv")
VALIDATED_WHITELIST_PERFORMANCE_HTML_PATH = Path("logs/v4_validated_whitelist_performance.html")
STRICT_VALIDATED_WHITELIST_PERFORMANCE_CSV_PATH = Path("logs/v4_strict_validated_whitelist_performance.csv")
STRICT_VALIDATED_WHITELIST_PERFORMANCE_HTML_PATH = Path("logs/v4_strict_validated_whitelist_performance.html")


@dataclass(frozen=True)
class ValidatedWhitelistPerformanceResult:
    assets_evaluated: int
    combined_portfolio_return_pct: float
    combined_final_capital: float
    combined_starting_capital: float
    csv_path: Path
    html_path: Path
    report_df: pd.DataFrame


def _load_validated_whitelist_signals(timeframe: str, assets: list[str]) -> pd.DataFrame:
    _require_file(VALIDATED_WHITELIST_PAPER_SIGNALS_PATH, "validated whitelist paper signals")
    signals = pd.read_csv(VALIDATED_WHITELIST_PAPER_SIGNALS_PATH)
    required = ["timestamp", "symbol", "timeframe", "model_probability", "model_direction", "is_trade_candidate"]
    missing = [column for column in required if column not in signals.columns]
    if missing:
        raise ValueError(f"Missing validated whitelist signal columns: {missing}")

    signals["timestamp"] = pd.to_datetime(signals["timestamp"], errors="coerce")
    signals["symbol"] = signals["symbol"].astype(str).str.upper()
    signals["timeframe"] = signals["timeframe"].astype(str)
    signals["model_probability"] = pd.to_numeric(signals["model_probability"], errors="coerce")
    signals["model_direction"] = signals["model_direction"].astype(str).str.upper()
    signals = signals.dropna(subset=["timestamp", "symbol", "timeframe", "model_probability"])
    signals = signals.loc[
        signals["timeframe"].eq(str(timeframe))
        & signals["symbol"].isin(assets)
    ].copy()
    return signals.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _daily_loss_events(debug: pd.DataFrame) -> int:
    if debug.empty or "trade_result" not in debug.columns:
        return 0
    blocked = debug.loc[debug["trade_result"].astype(str).eq("BLOCKED_DAILY_LOSS")].copy()
    if blocked.empty:
        return 0
    blocked["day"] = pd.to_datetime(blocked["timestamp"], errors="coerce").dt.date
    return int(blocked["day"].nunique())


def _build_symbol_report(symbol: str, timeframe: str, signals: pd.DataFrame, starting_capital: float) -> dict[str, Any] | None:
    symbol_signals = signals.loc[signals["symbol"].eq(symbol)].copy()
    if symbol_signals.empty:
        return None

    execution = _build_execution_frame(symbol, timeframe, symbol_signals["timestamp"]).reset_index(drop=True)
    base = symbol_signals.reset_index(drop=True).join(execution)
    metrics, debug = _simulate_prepared_signals(
        base[["timestamp", "model_direction", "Close", "High_next", "Low_next", "Close_next", "ATR"]].copy(),
        "model_direction",
        starting_capital,
    )

    candidates = int(symbol_signals["is_trade_candidate"].astype(bool).sum())
    latest = symbol_signals.iloc[-1]
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        **_constraints_used(),
        "predictions": int(len(symbol_signals)),
        "trade_candidates": candidates,
        "return_pct": float(metrics["return_pct"]),
        "final_capital": float(metrics["final_capital"]),
        "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
        "profit_factor": float(metrics["profit_factor"]),
        "win_rate_pct": float(metrics["win_rate_pct"]),
        "trade_count": int(metrics["trade_count"]),
        "average_trade_pct": float(metrics["average_trade_pct"]),
        "expectancy": float(metrics["expectancy"]),
        "daily_loss_events": _daily_loss_events(debug),
        "latest_signal": str(latest["model_direction"]),
        "latest_probability": float(latest["model_probability"]),
        "latest_price": float(latest["price"]) if "price" in latest and pd.notna(latest["price"]) else float("nan"),
    }


def _write_html_report(report: pd.DataFrame, combined_return: float, html_path: Path, assets: list[str]) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    display = report.copy()
    numeric_columns = display.select_dtypes(include=[np.number]).columns
    display[numeric_columns] = display[numeric_columns].round(6)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>V4 Validated Whitelist Performance</title>
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
  <h1>V4 Validated Whitelist Performance</h1>
  <p>Paper-only constrained V4-style execution for {", ".join(assets)}. No live orders are submitted.</p>
  <ul>
    <li>Assets evaluated: {len(display)}</li>
    <li>Combined portfolio return: {combined_return:.6f}%</li>
  </ul>
  {display.to_html(index=False, classes="performance", border=0)}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def run_validated_whitelist_performance(args: Any) -> ValidatedWhitelistPerformanceResult:
    """Run constrained paper performance for the validated whitelist only."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    starting_capital = float(getattr(args, "capital", 100000.0))
    strict = bool(getattr(args, "strict", False))
    single_asset = str(getattr(args, "asset", "") or "").upper()
    if single_asset:
        assets = [single_asset]
        csv_path = Path("logs") / f"v4_{single_asset}_validated_performance.csv"
        html_path = Path("logs") / f"v4_{single_asset}_validated_performance.html"
    else:
        assets = STRICT_VALIDATED_WHITELIST_ASSETS if strict else VALIDATED_WHITELIST_ASSETS
        csv_path = STRICT_VALIDATED_WHITELIST_PERFORMANCE_CSV_PATH if strict else VALIDATED_WHITELIST_PERFORMANCE_CSV_PATH
        html_path = STRICT_VALIDATED_WHITELIST_PERFORMANCE_HTML_PATH if strict else VALIDATED_WHITELIST_PERFORMANCE_HTML_PATH
    signals = _load_validated_whitelist_signals(timeframe, assets)
    if signals.empty:
        raise ValueError(f"No validated whitelist paper signals found for {timeframe}")

    per_asset_capital = starting_capital / len(assets)
    rows = []
    for symbol in assets:
        row = _build_symbol_report(symbol, timeframe, signals, per_asset_capital)
        if row is not None:
            rows.append(row)

    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("No validated whitelist assets could be evaluated")
    report = report.sort_values("return_pct", ascending=False).reset_index(drop=True)

    combined_starting_capital = per_asset_capital * len(report)
    combined_final_capital = float(report["final_capital"].sum())
    combined_return = ((combined_final_capital / combined_starting_capital) - 1.0) * 100.0
    report.insert(0, "portfolio_weight_pct", 100.0 / len(report))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False)
    _write_html_report(report, combined_return, html_path, assets)

    return ValidatedWhitelistPerformanceResult(
        assets_evaluated=int(len(report)),
        combined_portfolio_return_pct=float(combined_return),
        combined_final_capital=combined_final_capital,
        combined_starting_capital=float(combined_starting_capital),
        csv_path=csv_path,
        html_path=html_path,
        report_df=report,
    )
