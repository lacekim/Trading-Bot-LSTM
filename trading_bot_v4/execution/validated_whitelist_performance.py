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
from trading_bot_v4.execution.paper_model_comparison import ORIGINAL_BASELINE_SIGNALS_PATH


VALIDATED_WHITELIST_ASSETS = ["PENGU", "DYDX", "AIXBT"]
STRICT_VALIDATED_WHITELIST_ASSETS = ["AIXBT", "DYDX"]
VALIDATED_WHITELIST_PERFORMANCE_CSV_PATH = Path("logs/v4_validated_whitelist_performance.csv")
VALIDATED_WHITELIST_PERFORMANCE_HTML_PATH = Path("logs/v4_validated_whitelist_performance.html")
STRICT_VALIDATED_WHITELIST_PERFORMANCE_CSV_PATH = Path("logs/v4_strict_validated_whitelist_performance.csv")
STRICT_VALIDATED_WHITELIST_PERFORMANCE_HTML_PATH = Path("logs/v4_strict_validated_whitelist_performance.html")
GO_ASSET_SELECTION_AUDIT_PATH = Path("logs/v4_go_asset_selection_audit.csv")
GO_MIN_PROFIT_FACTOR = Config.GO_MIN_RECENT_PROFIT_FACTOR
GO_MIN_RETURN_PCT = 0.0
GO_MAX_DRAWDOWN_PCT = -5.0
GO_MAX_DAILY_LOSS_EVENTS = 0


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


@dataclass(frozen=True)
class GoAssetsPerformanceResult:
    selected_assets: list[str]
    timeframe: str
    combined_portfolio_return_pct: float
    combined_max_drawdown_pct: float
    combined_profit_factor: float
    combined_win_rate_pct: float
    combined_trade_count: int
    combined_daily_loss_events: int
    csv_path: Path
    html_path: Path
    report_df: pd.DataFrame


def _load_validated_whitelist_signals(
    timeframe: str,
    assets: list[str],
    recent_days: int = 0,
    signals_path: Path = ORIGINAL_BASELINE_SIGNALS_PATH,
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


def _build_symbol_report_with_debug(
    symbol: str,
    timeframe: str,
    signals: pd.DataFrame,
    starting_capital: float,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
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
    row = {
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
    debug = debug.copy()
    debug["symbol"] = symbol
    return row, debug


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
    signals_path = Path(getattr(args, "signals_path", ORIGINAL_BASELINE_SIGNALS_PATH))
    output_path = getattr(args, "sweep_output_path", None)
    preloaded_signals = getattr(args, "signals_df", None)
    windows = [7, 14, 30, 60, 90]

    rows: list[dict[str, Any]] = []
    for recent_days in windows:
        if isinstance(preloaded_signals, pd.DataFrame):
            signals = preloaded_signals.loc[
                preloaded_signals["symbol"].astype(str).str.upper().eq(symbol)
            ].copy()
            if not signals.empty:
                latest_timestamp = signals["timestamp"].max()
                signals = signals.loc[
                    signals["timestamp"].ge(latest_timestamp - pd.Timedelta(days=recent_days))
                ].copy()
        else:
            signals = _load_validated_whitelist_signals(
                timeframe, [symbol], recent_days=recent_days, signals_path=signals_path
            )
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
    win_rate_30 = _finite_metric(row_30, "win_rate_pct")
    drawdown_30 = _finite_metric(row_30, "max_drawdown_pct")
    trade_count_30 = int(_finite_metric(row_30, "trade_count")) if np.isfinite(_finite_metric(row_30, "trade_count")) else 0

    failed: list[str] = []
    if not return_7 > 0:
        failed.append(f"7d return > 0 failed: {return_7:.6f}%")
    if not return_14 > 0:
        failed.append(f"14d return > 0 failed: {return_14:.6f}%")
    if not return_30 > 0:
        failed.append(f"30d return > 0 failed: {return_30:.6f}%")
    if not profit_factor_30 >= Config.GO_MIN_RECENT_PROFIT_FACTOR:
        failed.append(
            f"30d profit factor >= {Config.GO_MIN_RECENT_PROFIT_FACTOR:.2f} failed: {profit_factor_30:.6f}"
        )
    if not win_rate_30 >= Config.GO_MIN_RECENT_WIN_RATE_PCT:
        failed.append(
            f"30d win rate >= {Config.GO_MIN_RECENT_WIN_RATE_PCT:.1f}% failed: {win_rate_30:.6f}%"
        )
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
        "win_rate_30d_pct": win_rate_30,
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
    explicit_symbols = [str(symbol).upper() for symbol in getattr(args, "symbols", []) or []]
    signals_path = Path(getattr(args, "signals_path", ORIGINAL_BASELINE_SIGNALS_PATH))
    if top_validated > 0 or explicit_symbols:
        symbols = list(dict.fromkeys(explicit_symbols or _load_top_validated_symbols(timeframe, top_validated)))
        all_signals = _load_validated_whitelist_signals(
            timeframe, symbols, signals_path=signals_path
        )
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
                    "signals_path": signals_path,
                    "signals_df": all_signals,
                    "sweep_output_path": None,
                },
            )()
            sweep_result = run_validated_whitelist_recent_sweep(sweep_args)
            _, _, metrics = _evaluate_readiness(symbol, timeframe, sweep_result.sweep_df)
            readiness_rows.append(metrics)
            sweep_rows.append(sweep_result.sweep_df)

        readiness = pd.DataFrame(readiness_rows)
        sweep = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
        # Keep the established path because downstream GO-performance code consumes it.
        csv_path = Path("logs/v4_top_validated10_paper_readiness.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        readiness.to_csv(csv_path, index=False)
        failed_assets = readiness.loc[readiness["decision"].ne("GO"), "symbol"].tolist()
        return PaperReadinessResult(
            symbol=f"MARKET_CANDIDATES_{len(symbols)}",
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


def _load_top_validated_readiness(timeframe: str) -> pd.DataFrame:
    path = Path("logs/v4_top_validated10_paper_readiness.csv")
    _require_file(path, "top validated paper readiness")
    readiness = pd.read_csv(path)
    required = ["symbol", "timeframe", "decision"]
    missing = [column for column in required if column not in readiness.columns]
    if missing:
        raise ValueError(f"Missing readiness columns: {missing}")

    readiness["symbol"] = readiness["symbol"].astype(str).str.upper()
    readiness["timeframe"] = readiness["timeframe"].astype(str)
    readiness["decision"] = readiness["decision"].astype(str).str.upper()
    return readiness.loc[readiness["timeframe"].eq(str(timeframe))].copy().reset_index(drop=True)


def _load_go_assets_from_readiness(timeframe: str) -> list[str]:
    readiness = _load_top_validated_readiness(timeframe)
    selected = readiness.loc[
        readiness["decision"].eq("GO"),
        "symbol",
    ].tolist()
    return selected


def _load_profitable_go_assets(timeframe: str) -> list[str]:
    path = Path("logs/v4_go_assets_performance.csv")
    _require_file(path, "GO assets performance")
    performance = pd.read_csv(path)
    required = ["symbol", "timeframe", "return_pct"]
    missing = [column for column in required if column not in performance.columns]
    if missing:
        raise ValueError(f"Missing GO assets performance columns: {missing}")

    performance["symbol"] = performance["symbol"].astype(str).str.upper()
    performance["timeframe"] = performance["timeframe"].astype(str)
    performance["return_pct"] = pd.to_numeric(performance["return_pct"], errors="coerce")
    selected = performance.loc[
        performance["timeframe"].eq(str(timeframe))
        & performance["return_pct"].gt(0),
        "symbol",
    ].tolist()
    if not selected:
        raise ValueError(f"No profitable GO assets found in {path} for timeframe {timeframe}")
    return selected


def _combined_capital_curve(debug_frames: list[pd.DataFrame], starting_capital_per_asset: float) -> pd.Series:
    curves = []
    for debug in debug_frames:
        if debug.empty:
            continue
        symbol = str(debug["symbol"].iloc[0])
        curve = debug[["timestamp", "capital"]].copy()
        curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce")
        curve["capital"] = pd.to_numeric(curve["capital"], errors="coerce")
        curve = curve.dropna(subset=["timestamp", "capital"]).drop_duplicates("timestamp", keep="last")
        curve = curve.set_index("timestamp").sort_index().rename(columns={"capital": symbol})
        curves.append(curve)
    if not curves:
        return pd.Series(dtype=float)
    combined = pd.concat(curves, axis=1).sort_index().ffill().fillna(float(starting_capital_per_asset))
    return combined.sum(axis=1)


def _combined_trade_metrics(debug_frames: list[pd.DataFrame], starting_capital_per_asset: float) -> dict[str, float | int]:
    trade_pnls: list[float] = []
    trade_results: list[str] = []
    daily_loss_events = 0
    for debug in debug_frames:
        if debug.empty:
            continue
        frame = debug.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["capital"] = pd.to_numeric(frame["capital"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "capital"]).sort_values("timestamp").reset_index(drop=True)
        frame["previous_capital"] = frame["capital"].shift(1).fillna(float(starting_capital_per_asset))
        frame["pnl"] = frame["capital"] - frame["previous_capital"]
        result = frame["trade_result"].astype(str)
        executed = frame.loc[result.str.contains("_WIN|_LOSS|_FLAT", regex=True)].copy()
        trade_pnls.extend(executed["pnl"].astype(float).tolist())
        trade_results.extend(executed["trade_result"].astype(str).tolist())
        daily_loss_events += _daily_loss_events(frame)

    if not trade_pnls:
        return {
            "profit_factor": 0.0,
            "win_rate_pct": 0.0,
            "trade_count": 0,
            "daily_loss_events": int(daily_loss_events),
        }

    pnls = np.array(trade_pnls, dtype=float)
    gross_profit = float(pnls[pnls > 0].sum()) if np.any(pnls > 0) else 0.0
    gross_loss = abs(float(pnls[pnls < 0].sum())) if np.any(pnls < 0) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    wins = sum(1 for item in trade_results if item.endswith("_WIN"))
    return {
        "profit_factor": float(profit_factor),
        "win_rate_pct": float(wins / len(trade_results) * 100.0) if trade_results else 0.0,
        "trade_count": int(len(trade_results)),
        "daily_loss_events": int(daily_loss_events),
    }


def _write_go_assets_html(
    report: pd.DataFrame,
    selected_assets: list[str],
    combined_return: float,
    combined_drawdown: float,
    combined_profit_factor: float,
    combined_win_rate: float,
    combined_trade_count: int,
    combined_daily_loss_events: int,
    html_path: Path,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    display = report.copy()
    numeric_columns = display.select_dtypes(include=[np.number]).columns
    display[numeric_columns] = display[numeric_columns].round(6)
    assets = ", ".join(selected_assets) if selected_assets else "none"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>V4 GO Assets Performance</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f8; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>V4 GO Assets Performance</h1>
  <p>Paper-only constrained V4-style execution for {assets}. No live orders are submitted.</p>
  <ul>
    <li>Combined portfolio return: {combined_return:.6f}%</li>
    <li>Combined max drawdown: {combined_drawdown:.6f}%</li>
    <li>Combined profit factor: {combined_profit_factor:.6f}</li>
    <li>Combined win rate: {combined_win_rate:.6f}%</li>
    <li>Combined trade count: {combined_trade_count}</li>
    <li>Combined daily loss events: {combined_daily_loss_events}</li>
  </ul>
  {display.to_html(index=False, classes="performance", border=0)}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def _empty_go_assets_report(csv_path: Path, html_path: Path, timeframe: str) -> GoAssetsPerformanceResult:
    columns = [
        "portfolio_weight_pct",
        "symbol",
        "timeframe",
        "start_date",
        "end_date",
        "return_pct",
        "final_capital",
        "max_drawdown_pct",
        "profit_factor",
        "win_rate_pct",
        "trade_count",
        "daily_loss_events",
    ]
    report = pd.DataFrame(columns=columns)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False)
    _write_go_assets_html(
        report,
        [],
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        html_path,
    )
    return GoAssetsPerformanceResult(
        selected_assets=[],
        timeframe=timeframe,
        combined_portfolio_return_pct=0.0,
        combined_max_drawdown_pct=0.0,
        combined_profit_factor=0.0,
        combined_win_rate_pct=0.0,
        combined_trade_count=0,
        combined_daily_loss_events=0,
        csv_path=csv_path,
        html_path=html_path,
        report_df=report,
    )


def _passes_strict_go_performance(row: pd.Series) -> bool:
    return_pct = pd.to_numeric(pd.Series([row.get("return_pct")]), errors="coerce").iloc[0]
    profit_factor = pd.to_numeric(pd.Series([row.get("profit_factor")]), errors="coerce").iloc[0]
    max_drawdown = pd.to_numeric(pd.Series([row.get("max_drawdown_pct")]), errors="coerce").iloc[0]
    daily_loss_events = pd.to_numeric(pd.Series([row.get("daily_loss_events")]), errors="coerce").iloc[0]
    return bool(
        pd.notna(return_pct)
        and pd.notna(profit_factor)
        and pd.notna(max_drawdown)
        and pd.notna(daily_loss_events)
        and float(return_pct) > GO_MIN_RETURN_PCT
        and float(profit_factor) > GO_MIN_PROFIT_FACTOR
        and float(max_drawdown) > GO_MAX_DRAWDOWN_PCT
        and int(daily_loss_events) == GO_MAX_DAILY_LOSS_EVENTS
    )


def _go_rule_failures(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    readiness_decision = str(row.get("readiness_decision", "")).upper()
    if readiness_decision != "GO":
        reasons.append("paper readiness decision != GO")

    return_pct = pd.to_numeric(pd.Series([row.get("constrained_return_pct")]), errors="coerce").iloc[0]
    profit_factor = pd.to_numeric(pd.Series([row.get("constrained_profit_factor")]), errors="coerce").iloc[0]
    max_drawdown = pd.to_numeric(pd.Series([row.get("constrained_max_drawdown_pct")]), errors="coerce").iloc[0]
    daily_loss_events = pd.to_numeric(pd.Series([row.get("daily_loss_events")]), errors="coerce").iloc[0]

    if pd.isna(return_pct):
        reasons.append("constrained return unavailable")
    elif not float(return_pct) > GO_MIN_RETURN_PCT:
        reasons.append(f"constrained return > 0 failed: {float(return_pct):.6f}%")

    if pd.isna(profit_factor):
        reasons.append("constrained profit factor unavailable")
    elif not float(profit_factor) > GO_MIN_PROFIT_FACTOR:
        reasons.append(f"constrained profit factor > {GO_MIN_PROFIT_FACTOR:.2f} failed: {float(profit_factor):.6f}")

    if pd.isna(max_drawdown):
        reasons.append("constrained max drawdown unavailable")
    elif not float(max_drawdown) > GO_MAX_DRAWDOWN_PCT:
        reasons.append(f"constrained max drawdown > {GO_MAX_DRAWDOWN_PCT:.2f}% failed: {float(max_drawdown):.6f}%")

    if pd.isna(daily_loss_events):
        reasons.append("daily loss events unavailable")
    elif int(daily_loss_events) != GO_MAX_DAILY_LOSS_EVENTS:
        reasons.append(f"daily loss events == 0 failed: {int(daily_loss_events)}")

    return reasons


def _write_go_asset_selection_audit(readiness: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    audit = readiness.copy()
    if audit.empty:
        audit = pd.DataFrame(
            columns=[
                "symbol",
                "readiness_decision",
                "constrained_return_pct",
                "constrained_profit_factor",
                "constrained_max_drawdown_pct",
                "daily_loss_events",
                "final_decision",
                "failed_rule_reasons",
            ]
        )
        GO_ASSET_SELECTION_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(GO_ASSET_SELECTION_AUDIT_PATH, index=False)
        return audit

    audit["symbol"] = audit["symbol"].astype(str).str.upper()
    audit["readiness_decision"] = audit["decision"].astype(str).str.upper()

    perf_columns = [
        "symbol",
        "return_pct",
        "profit_factor",
        "max_drawdown_pct",
        "daily_loss_events",
    ]
    if performance.empty:
        perf = pd.DataFrame(columns=perf_columns)
    else:
        perf = performance[[column for column in perf_columns if column in performance.columns]].copy()
        perf["symbol"] = perf["symbol"].astype(str).str.upper()

    audit = audit.merge(perf, on="symbol", how="left")
    audit = audit.rename(
        columns={
            "return_pct": "constrained_return_pct",
            "profit_factor": "constrained_profit_factor",
            "max_drawdown_pct": "constrained_max_drawdown_pct",
        }
    )
    for column in [
        "constrained_return_pct",
        "constrained_profit_factor",
        "constrained_max_drawdown_pct",
        "daily_loss_events",
    ]:
        if column not in audit.columns:
            audit[column] = np.nan

    audit["failed_rule_reasons"] = audit.apply(lambda row: "; ".join(_go_rule_failures(row)), axis=1)
    audit["final_decision"] = np.where(audit["failed_rule_reasons"].astype(str).eq(""), "GO", "NO-GO")
    output_columns = [
        "symbol",
        "readiness_decision",
        "constrained_return_pct",
        "constrained_profit_factor",
        "constrained_max_drawdown_pct",
        "daily_loss_events",
        "final_decision",
        "failed_rule_reasons",
    ]
    GO_ASSET_SELECTION_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit[output_columns].to_csv(GO_ASSET_SELECTION_AUDIT_PATH, index=False)
    return audit[output_columns]


def run_go_assets_performance(args: Any) -> GoAssetsPerformanceResult:
    """Run constrained paper performance for assets marked GO by top-validated readiness."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    starting_capital = float(getattr(args, "capital", 100000.0))
    profitable_only = bool(getattr(args, "profitable_only", False))
    signals_path = Path(getattr(args, "signals_path", ORIGINAL_BASELINE_SIGNALS_PATH))
    csv_path = Path("logs/v4_profitable_go_assets_performance.csv") if profitable_only else Path("logs/v4_go_assets_performance.csv")
    html_path = Path("logs/v4_profitable_go_assets_performance.html") if profitable_only else Path("logs/v4_go_assets_performance.html")

    readiness = pd.DataFrame()
    if profitable_only:
        candidate_assets = _load_profitable_go_assets(timeframe)
    else:
        readiness = _load_top_validated_readiness(timeframe)
        candidate_assets = readiness["symbol"].astype(str).str.upper().tolist()

    if not candidate_assets:
        if not profitable_only:
            _write_go_asset_selection_audit(readiness, pd.DataFrame())
        return _empty_go_assets_report(csv_path, html_path, timeframe)

    signals = _load_validated_whitelist_signals(timeframe, candidate_assets, signals_path=signals_path)
    if signals.empty:
        if not profitable_only:
            _write_go_asset_selection_audit(readiness, pd.DataFrame())
        return _empty_go_assets_report(csv_path, html_path, timeframe)

    candidate_capital = starting_capital / len(candidate_assets)
    candidate_rows: list[dict[str, Any]] = []
    for symbol in candidate_assets:
        candidate_result = _build_symbol_report_with_debug(symbol, timeframe, signals, candidate_capital)
        if candidate_result is None:
            continue
        row, _ = candidate_result
        candidate_rows.append(row)

    candidate_report = pd.DataFrame(candidate_rows)
    if candidate_report.empty:
        if not profitable_only:
            _write_go_asset_selection_audit(readiness, pd.DataFrame())
        return _empty_go_assets_report(csv_path, html_path, timeframe)

    if profitable_only:
        selected_assets = candidate_report.loc[
            pd.to_numeric(candidate_report["return_pct"], errors="coerce").gt(0),
            "symbol",
        ].astype(str).tolist()
    else:
        audit = _write_go_asset_selection_audit(readiness, candidate_report)
        selected_assets = audit.loc[
            audit["final_decision"].astype(str).str.upper().eq("GO"),
            "symbol",
        ].astype(str).tolist()

    if not selected_assets:
        return _empty_go_assets_report(csv_path, html_path, timeframe)

    per_asset_capital = starting_capital / len(selected_assets)
    signals = _load_validated_whitelist_signals(timeframe, selected_assets, signals_path=signals_path)
    rows: list[dict[str, Any]] = []
    debug_frames: list[pd.DataFrame] = []
    for symbol in selected_assets:
        result = _build_symbol_report_with_debug(symbol, timeframe, signals, per_asset_capital)
        if result is None:
            continue
        row, debug = result
        rows.append(row)
        debug_frames.append(debug)

    report = pd.DataFrame(rows)
    if report.empty:
        return _empty_go_assets_report(csv_path, html_path, timeframe)
    report["start_date"] = pd.to_datetime(report["start_date"], errors="coerce")
    report["end_date"] = pd.to_datetime(report["end_date"], errors="coerce")
    report = report.sort_values("return_pct", ascending=False).reset_index(drop=True)
    report.insert(0, "portfolio_weight_pct", 100.0 / len(report))

    combined_starting_capital = per_asset_capital * len(report)
    combined_final_capital = float(report["final_capital"].sum())
    combined_return = ((combined_final_capital / combined_starting_capital) - 1.0) * 100.0
    combined_curve = _combined_capital_curve(debug_frames, per_asset_capital)
    if combined_curve.empty:
        combined_drawdown = 0.0
    else:
        equity = np.concatenate([[combined_starting_capital], combined_curve.to_numpy(dtype=float)])
        running_max = np.maximum.accumulate(equity)
        combined_drawdown = float(((equity / running_max) - 1.0).min() * 100.0)
    combined_trade_metrics = _combined_trade_metrics(debug_frames, per_asset_capital)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False)
    _write_go_assets_html(
        report,
        selected_assets,
        combined_return,
        combined_drawdown,
        float(combined_trade_metrics["profit_factor"]),
        float(combined_trade_metrics["win_rate_pct"]),
        int(combined_trade_metrics["trade_count"]),
        int(combined_trade_metrics["daily_loss_events"]),
        html_path,
    )

    return GoAssetsPerformanceResult(
        selected_assets=selected_assets,
        timeframe=timeframe,
        combined_portfolio_return_pct=float(combined_return),
        combined_max_drawdown_pct=float(combined_drawdown),
        combined_profit_factor=float(combined_trade_metrics["profit_factor"]),
        combined_win_rate_pct=float(combined_trade_metrics["win_rate_pct"]),
        combined_trade_count=int(combined_trade_metrics["trade_count"]),
        combined_daily_loss_events=int(combined_trade_metrics["daily_loss_events"]),
        csv_path=csv_path,
        html_path=html_path,
        report_df=report,
    )
