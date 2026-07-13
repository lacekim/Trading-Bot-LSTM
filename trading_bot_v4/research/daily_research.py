"""Daily paper-only research pipeline for V4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot_v4.backtesting.asset_selection_engine import run_asset_ranking
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.execution.smc_model_paper import run_smc_model_paper_trading
from trading_bot_v4.execution.validated_whitelist_performance import (
    _load_top_validated_symbols,
    run_go_assets_performance,
    run_paper_readiness,
)
from trading_bot_v4.features.smc_feature_builder import _build_smc_feature_frame
from trading_bot_v4.utils.logger import build_logger


logger = build_logger("v4_daily_research")

DAILY_DASHBOARD_PATH = Path("reports/v4_daily_dashboard.html")
DAILY_GO_STATUS_PATH = Path("reports/v4_daily_go_status.csv")
DAILY_SMC_FEATURE_DIR = Path("reports/smc_features")


@dataclass(frozen=True)
class DailyResearchResult:
    dashboard_path: Path
    refresh_succeeded: bool
    smc_features_updated: int
    rankings_path: Path
    readiness_path: Path
    performance_csv_path: Path
    performance_html_path: Path
    selected_assets: list[str]
    go_assets: list[str]
    no_go_assets: list[str]
    alerts: list[str]
    combined_return: float
    readiness_df: pd.DataFrame
    performance_df: pd.DataFrame


def _format_metric(value: Any, suffix: str = "") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "none"
    if not np.isfinite(numeric):
        return "none"
    return f"{numeric:.6f}{suffix}"


def _daily_args(**kwargs: Any) -> Any:
    defaults = {
        "timeframe": Config.TIMEFRAME,
        "capital": 100000.0,
        "validated": False,
        "top_validated": 0,
        "profitable_only": False,
    }
    defaults.update(kwargs)
    return type("Args", (), defaults)()


def _safe_top_validated_symbols(timeframe: str, count: int) -> list[str]:
    try:
        return _load_top_validated_symbols(timeframe, count)
    except Exception as exc:
        logger.warning("Top validated symbols unavailable before ranking refresh: %s", exc)
        return [str(getattr(Config, "GMX_SYMBOL", "GMX")).upper()]


def _update_smc_features(symbols: list[str], timeframe: str) -> list[Path]:
    DAILY_SMC_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for symbol in symbols:
        try:
            features = _build_smc_feature_frame(symbol, timeframe)
            output = DAILY_SMC_FEATURE_DIR / f"v4_smc_features_{symbol}_{timeframe}.csv"
            features.to_csv(output, index_label="Date", float_format="%.10f")
            outputs.append(output)
        except Exception as exc:
            logger.warning("Skipping SMC feature update for %s %s: %s", symbol, timeframe, exc)
    return outputs


def _status_alerts(current: pd.DataFrame) -> list[str]:
    current_status = current[["symbol", "decision"]].copy()
    current_status["symbol"] = current_status["symbol"].astype(str).str.upper()
    current_status["decision"] = current_status["decision"].astype(str).str.upper()
    current_status["updated_at"] = pd.Timestamp.utcnow().isoformat()

    alerts: list[str] = []
    if DAILY_GO_STATUS_PATH.exists():
        previous = pd.read_csv(DAILY_GO_STATUS_PATH)
        if {"symbol", "decision"}.issubset(previous.columns):
            previous["symbol"] = previous["symbol"].astype(str).str.upper()
            previous["decision"] = previous["decision"].astype(str).str.upper()
            merged = current_status.merge(
                previous[["symbol", "decision"]].rename(columns={"decision": "previous_decision"}),
                on="symbol",
                how="left",
            )
            changed = merged.loc[
                merged["previous_decision"].notna()
                & merged["decision"].ne(merged["previous_decision"])
            ]
            for row in changed.itertuples(index=False):
                alerts.append(f"{row.symbol}: {row.previous_decision} -> {row.decision}")

    DAILY_GO_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    current_status.to_csv(DAILY_GO_STATUS_PATH, index=False)
    return alerts


def _readiness_recent_metrics(readiness: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "decision",
        "return_7d_pct",
        "return_14d_pct",
        "return_30d_pct",
        "profit_factor_30d",
        "max_drawdown_30d_pct",
        "trade_count_30d",
        "failed_conditions",
    ]
    return readiness[[column for column in columns if column in readiness.columns]].copy()


def _apply_strict_go_status(readiness: pd.DataFrame, selected_assets: list[str]) -> pd.DataFrame:
    strict = readiness.copy()
    if strict.empty or "symbol" not in strict.columns or "decision" not in strict.columns:
        return strict

    selected = {symbol.upper() for symbol in selected_assets}
    strict["symbol"] = strict["symbol"].astype(str).str.upper()
    strict["decision"] = strict["decision"].astype(str).str.upper()
    readiness_go = strict["decision"].eq("GO")
    strict_fail = readiness_go & ~strict["symbol"].isin(selected)
    strict.loc[strict_fail, "decision"] = "NO-GO"
    if "failed_conditions" not in strict.columns:
        strict["failed_conditions"] = ""
    strict.loc[strict_fail, "failed_conditions"] = strict.loc[strict_fail, "failed_conditions"].apply(
        lambda value: (
            f"{value}; strict GO performance filter failed"
            if isinstance(value, str) and value.strip()
            else "strict GO performance filter failed"
        )
    )
    return strict


def _write_dashboard(
    result: DailyResearchResult,
    rankings: pd.DataFrame,
    readiness: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    DAILY_DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    top_rankings = rankings.head(20).copy()
    readiness_display = _readiness_recent_metrics(readiness)
    performance_display = performance.copy()

    for frame in [top_rankings, readiness_display, performance_display]:
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_columns] = frame[numeric_columns].round(6)

    alert_items = "\n".join(f"<li>{alert}</li>" for alert in result.alerts) or "<li>No GO status changes detected.</li>"
    whitelist = ", ".join(result.go_assets) if result.go_assets else "none"
    no_go = ", ".join(result.no_go_assets) if result.no_go_assets else "none"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>V4 Daily Research Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 6px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f8; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .metric {{ border: 1px solid #d9e2ec; padding: 12px; border-radius: 6px; }}
    .label {{ color: #52606d; font-size: 12px; }}
    .value {{ font-size: 20px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>V4 Daily Research Dashboard</h1>
  <p>Paper-only research report. No live orders are submitted.</p>
  <div class="summary">
    <div class="metric"><div class="label">Market data refresh</div><div class="value">{result.refresh_succeeded}</div></div>
    <div class="metric"><div class="label">SMC feature files updated</div><div class="value">{result.smc_features_updated}</div></div>
    <div class="metric"><div class="label">Current whitelist</div><div class="value">{whitelist}</div></div>
    <div class="metric"><div class="label">Portfolio return</div><div class="value">{_format_metric(result.combined_return, "%")}</div></div>
    <div class="metric"><div class="label">GO assets</div><div class="value">{len(result.go_assets)}</div></div>
    <div class="metric"><div class="label">NO-GO assets</div><div class="value">{len(result.no_go_assets)}</div></div>
  </div>

  <h2>Alerts</h2>
  <ul>{alert_items}</ul>

  <h2>Current Whitelist</h2>
  <p>{whitelist}</p>

  <h2>NO-GO Assets</h2>
  <p>{no_go}</p>

  <h2>GO / NO-GO and Recent Metrics</h2>
  {readiness_display.to_html(index=False, border=0)}

  <h2>Paper Performance</h2>
  {performance_display.to_html(index=False, border=0)}

  <h2>Top Validated Rankings</h2>
  {top_rankings.to_html(index=False, border=0)}
</body>
</html>
"""
    DAILY_DASHBOARD_PATH.write_text(html, encoding="utf-8")


def run_daily_research(args: Any) -> DailyResearchResult:
    """Run the complete daily paper-only research pipeline."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    top_count = int(getattr(args, "top_validated", 10) or 10)

    handler = V4DataHandler()
    refresh_succeeded = handler.refresh_gmx_cache(force=True)

    pre_rank_symbols = _safe_top_validated_symbols(timeframe, top_count)
    smc_outputs = _update_smc_features(pre_rank_symbols, timeframe)

    ranking_result = run_asset_ranking(
        _daily_args(
            timeframe=timeframe,
            validated=True,
            model=getattr(args, "model", None),
            scaler=getattr(args, "scaler", None),
        )
    )
    rankings = ranking_result.rankings

    run_smc_model_paper_trading(
        _daily_args(
            timeframe=timeframe,
            all_assets=True,
            model=getattr(args, "smc_model", None),
            scaler=getattr(args, "smc_scaler", None),
        )
    )

    readiness_result = run_paper_readiness(_daily_args(timeframe=timeframe, top_validated=top_count))
    readiness = readiness_result.readiness_df if readiness_result.readiness_df is not None else pd.DataFrame()
    if readiness.empty:
        raise ValueError("Daily research readiness report is empty")

    performance_result = run_go_assets_performance(_daily_args(timeframe=timeframe))
    performance = performance_result.report_df

    strict_readiness = _apply_strict_go_status(readiness, performance_result.selected_assets)
    go_assets = strict_readiness.loc[
        strict_readiness["decision"].astype(str).str.upper().eq("GO"),
        "symbol",
    ].astype(str).tolist()
    no_go_assets = strict_readiness.loc[
        strict_readiness["decision"].astype(str).str.upper().ne("GO"),
        "symbol",
    ].astype(str).tolist()
    alerts = _status_alerts(strict_readiness)

    result = DailyResearchResult(
        dashboard_path=DAILY_DASHBOARD_PATH,
        refresh_succeeded=bool(refresh_succeeded),
        smc_features_updated=len(smc_outputs),
        rankings_path=ranking_result.output_path,
        readiness_path=readiness_result.sweep_csv_path,
        performance_csv_path=performance_result.csv_path,
        performance_html_path=performance_result.html_path,
        selected_assets=performance_result.selected_assets,
        go_assets=go_assets,
        no_go_assets=no_go_assets,
        alerts=alerts,
        combined_return=performance_result.combined_portfolio_return_pct,
        readiness_df=strict_readiness,
        performance_df=performance,
    )
    _write_dashboard(result, rankings, strict_readiness, performance)
    return result
