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
from trading_bot_v4.execution.smc_model_paper import SMC_MODEL_PAPER_SIGNALS_PATH, VALIDATED_WHITELIST_PAPER_SIGNALS_PATH


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
    start_date: str
    end_date: str


@dataclass(frozen=True)
class ValidatedWhitelistRecentSweepResult:
    symbol: str
    timeframe: str
    csv_path: Path
    sweep_df: pd.DataFrame


@dataclass(frozen=True)
class PaperReadinessResult:
    symbol: str
    timeframe: str
    decision: str
    failed_conditions: list[str]
    sweep_csv_path: Path
    sweep_df: pd.DataFrame
    readiness_df: pd.DataFrame | None = None


def _load_validated_whitelist_signals(
    timeframe: str,
    assets: list[str],
    recent_days: int = 0,
    signals_path: Path = VALIDATED_WHITELIST_PAPER_SIGNALS_PATH,
) -> pd.DataFrame:
    _require_file(signals_path, "paper signals")
    signals = pd.read_csv(signals_path)
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
    if recent_days > 0 and not signals.empty:
        latest_timestamp = signals["timestamp"].max()
        start_timestamp = latest_timestamp - pd.Timedelta(days=recent_days)
        signals = signals.loc[signals["timestamp"].ge(start_timestamp)].copy()
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
        "start_date": symbol_signals["timestamp"].min(),
        "end_date": symbol_signals["timestamp"].max(),
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


def _write_html_report(report: pd.DataFrame, combined_return: float, html_path: Path, assets: list[str], recent_days: int = 0) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    display = report.copy()
    numeric_columns = display.select_dtypes(include=[np.number]).columns
    display[numeric_columns] = display[numeric_columns].round(6)
    period_note = f"Recent {recent_days} days only." if recent_days > 0 else "Full available signal history."
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
  <p>{period_note}</p>
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
    recent_days = int(getattr(args, "recent_days", 0) or 0)
    if single_asset:
        assets = [single_asset]
        if recent_days > 0:
            csv_path = Path("logs") / f"v4_{single_asset}_recent{recent_days}_validated_performance.csv"
            html_path = Path("logs") / f"v4_{single_asset}_recent{recent_days}_validated_performance.html"
        else:
            csv_path = Path("logs") / f"v4_{single_asset}_validated_performance.csv"
            html_path = Path("logs") / f"v4_{single_asset}_validated_performance.html"
    else:
        assets = STRICT_VALIDATED_WHITELIST_ASSETS if strict else VALIDATED_WHITELIST_ASSETS
        if recent_days > 0:
            prefix = "strict_validated_whitelist" if strict else "validated_whitelist"
            csv_path = Path("logs") / f"v4_{prefix}_recent{recent_days}_performance.csv"
            html_path = Path("logs") / f"v4_{prefix}_recent{recent_days}_performance.html"
        else:
            csv_path = STRICT_VALIDATED_WHITELIST_PERFORMANCE_CSV_PATH if strict else VALIDATED_WHITELIST_PERFORMANCE_CSV_PATH
            html_path = STRICT_VALIDATED_WHITELIST_PERFORMANCE_HTML_PATH if strict else VALIDATED_WHITELIST_PERFORMANCE_HTML_PATH
    signals = _load_validated_whitelist_signals(timeframe, assets, recent_days=recent_days)
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
    report["start_date"] = pd.to_datetime(report["start_date"], errors="coerce")
    report["end_date"] = pd.to_datetime(report["end_date"], errors="coerce")
    report = report.sort_values("return_pct", ascending=False).reset_index(drop=True)

    combined_starting_capital = per_asset_capital * len(report)
    combined_final_capital = float(report["final_capital"].sum())
    combined_return = ((combined_final_capital / combined_starting_capital) - 1.0) * 100.0
    report.insert(0, "portfolio_weight_pct", 100.0 / len(report))
    start_date = str(report["start_date"].min())
    end_date = str(report["end_date"].max())

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False)
    _write_html_report(report, combined_return, html_path, assets, recent_days=recent_days)

    return ValidatedWhitelistPerformanceResult(
        assets_evaluated=int(len(report)),
        combined_portfolio_return_pct=float(combined_return),
        combined_final_capital=combined_final_capital,
        combined_starting_capital=float(combined_starting_capital),
        csv_path=csv_path,
        html_path=html_path,
        report_df=report,
        start_date=start_date,
        end_date=end_date,
    )


def run_validated_whitelist_recent_sweep(args: Any) -> ValidatedWhitelistRecentSweepResult:
    """Run recent-window constrained paper performance for one asset."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    starting_capital = float(getattr(args, "capital", 100000.0))
    symbol = str(getattr(args, "asset", "") or getattr(args, "symbol", Config.GMX_SYMBOL)).upper()
    signals_path = Path(getattr(args, "signals_path", VALIDATED_WHITELIST_PAPER_SIGNALS_PATH))
    output_path = getattr(args, "sweep_output_path", None)
    windows = [7, 14, 30, 60, 90]

    rows: list[dict[str, Any]] = []
    for recent_days in windows:
        signals = _load_validated_whitelist_signals(timeframe, [symbol], recent_days=recent_days, signals_path=signals_path)
        if signals.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "recent_days": recent_days,
                    "start_date": "",
                    "end_date": "",
                    "return_pct": float("nan"),
                    "max_drawdown_pct": float("nan"),
                    "profit_factor": float("nan"),
                    "win_rate_pct": float("nan"),
                    "trade_count": 0,
                }
            )
            continue

        row = _build_symbol_report(symbol, timeframe, signals, starting_capital)
        if row is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "recent_days": recent_days,
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "return_pct": row["return_pct"],
                "max_drawdown_pct": row["max_drawdown_pct"],
                "profit_factor": row["profit_factor"],
                "win_rate_pct": row["win_rate_pct"],
                "trade_count": row["trade_count"],
            }
        )

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        raise ValueError(f"No recent sweep rows could be evaluated for {symbol} {timeframe}")

    csv_path = Path(output_path) if output_path is not None else Path("logs") / f"v4_{symbol}_recent_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(csv_path, index=False)

    return ValidatedWhitelistRecentSweepResult(
        symbol=symbol,
        timeframe=timeframe,
        csv_path=csv_path,
        sweep_df=sweep,
    )


def _window_row(sweep: pd.DataFrame, recent_days: int) -> pd.Series:
    rows = sweep.loc[sweep["recent_days"].eq(recent_days)]
    if rows.empty:
        raise ValueError(f"Missing {recent_days}d recent sweep row")
    return rows.iloc[0]


def _finite_metric(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(float(value)):
        return float("nan")
    return float(value)


def _evaluate_readiness(symbol: str, timeframe: str, sweep: pd.DataFrame) -> tuple[str, list[str], dict[str, Any]]:
    row_7 = _window_row(sweep, 7)
    row_14 = _window_row(sweep, 14)
    row_30 = _window_row(sweep, 30)

    return_7 = _finite_metric(row_7, "return_pct")
    return_14 = _finite_metric(row_14, "return_pct")
    return_30 = _finite_metric(row_30, "return_pct")
    profit_factor_30 = _finite_metric(row_30, "profit_factor")
    drawdown_30 = _finite_metric(row_30, "max_drawdown_pct")
    trade_count_30 = int(_finite_metric(row_30, "trade_count")) if np.isfinite(_finite_metric(row_30, "trade_count")) else 0

    failed: list[str] = []
    if not return_7 > 0:
        failed.append(f"7d return > 0 failed: {return_7:.6f}%")
    if not return_14 > 0:
        failed.append(f"14d return > 0 failed: {return_14:.6f}%")
    if not return_30 > 0:
        failed.append(f"30d return > 0 failed: {return_30:.6f}%")
    if not profit_factor_30 > 1.05:
        failed.append(f"30d profit factor > 1.05 failed: {profit_factor_30:.6f}")
    if not drawdown_30 > -5.0:
        failed.append(f"30d max drawdown better than -5% failed: {drawdown_30:.6f}%")
    if not trade_count_30 >= 50:
        failed.append(f"30d trade count >= 50 failed: {trade_count_30}")

    metrics = {
        "symbol": symbol,
        "timeframe": timeframe,
        "decision": "GO" if not failed else "NO-GO",
        "failed_conditions": "; ".join(failed) if failed else "",
        "return_7d_pct": return_7,
        "return_14d_pct": return_14,
        "return_30d_pct": return_30,
        "profit_factor_30d": profit_factor_30,
        "max_drawdown_30d_pct": drawdown_30,
        "trade_count_30d": trade_count_30,
    }
    return str(metrics["decision"]), failed, metrics


def _load_top_validated_symbols(timeframe: str, count: int) -> list[str]:
    path = Path("models/asset_rankings_validated.csv")
    _require_file(path, "validated asset rankings")
    rankings = pd.read_csv(path)
    required = ["symbol", "timeframe"]
    missing = [column for column in required if column not in rankings.columns]
    if missing:
        raise ValueError(f"Missing validated ranking columns: {missing}")

    rankings["symbol"] = rankings["symbol"].astype(str).str.upper()
    rankings["timeframe"] = rankings["timeframe"].astype(str)
    rankings = rankings.loc[rankings["timeframe"].eq(str(timeframe))].copy()
    if rankings.empty:
        raise ValueError(f"No validated rankings found for timeframe {timeframe}")
    if "rank" in rankings.columns:
        rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
        rankings = rankings.sort_values(["rank", "symbol"])
    elif "validated_score" in rankings.columns:
        rankings["validated_score"] = pd.to_numeric(rankings["validated_score"], errors="coerce")
        rankings = rankings.sort_values("validated_score", ascending=False)
    return rankings["symbol"].head(count).tolist()


def run_paper_readiness(args: Any) -> PaperReadinessResult:
    """Evaluate go/no-go paper readiness using recent sweep metrics."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    top_validated = int(getattr(args, "top_validated", 0) or 0)
    if top_validated > 0:
        symbols = _load_top_validated_symbols(timeframe, top_validated)
        readiness_rows: list[dict[str, Any]] = []
        sweep_rows: list[pd.DataFrame] = []
        for symbol in symbols:
            sweep_args = type(
                "Args",
                (),
                {
                    "asset": symbol,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "capital": float(getattr(args, "capital", 100000.0)),
                    "signals_path": SMC_MODEL_PAPER_SIGNALS_PATH,
                    "sweep_output_path": None,
                },
            )()
            sweep_result = run_validated_whitelist_recent_sweep(sweep_args)
            _, _, metrics = _evaluate_readiness(symbol, timeframe, sweep_result.sweep_df)
            readiness_rows.append(metrics)
            sweep_rows.append(sweep_result.sweep_df)

        readiness = pd.DataFrame(readiness_rows)
        sweep = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
        csv_path = Path("logs") / f"v4_top_validated{top_validated}_paper_readiness.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        readiness.to_csv(csv_path, index=False)
        failed_assets = readiness.loc[readiness["decision"].ne("GO"), "symbol"].tolist()
        return PaperReadinessResult(
            symbol=f"TOP_VALIDATED_{top_validated}",
            timeframe=timeframe,
            decision="GO" if not failed_assets else "NO-GO",
            failed_conditions=[f"{symbol}: NO-GO" for symbol in failed_assets],
            sweep_csv_path=csv_path,
            sweep_df=sweep,
            readiness_df=readiness,
        )

    sweep_result = run_validated_whitelist_recent_sweep(args)
    sweep = sweep_result.sweep_df
    decision, failed, _ = _evaluate_readiness(sweep_result.symbol, sweep_result.timeframe, sweep)

    return PaperReadinessResult(
        symbol=sweep_result.symbol,
        timeframe=sweep_result.timeframe,
        decision=decision,
        failed_conditions=failed,
        sweep_csv_path=sweep_result.csv_path,
        sweep_df=sweep,
    )
