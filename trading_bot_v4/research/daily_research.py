"""Daily paper-only research pipeline for V4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols
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
from trading_bot_v4.research.market_scanner import scan_market_momentum


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
    momentum_df: pd.DataFrame


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
                if row.decision == "GO":
                    alerts.append(f"NEW GO — {row.symbol}")
                elif row.previous_decision == "GO":
                    alerts.append(f"REMOVED — {row.symbol}")
                else:
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


def _dashboard_decision_views(
    readiness: pd.DataFrame,
    rankings: pd.DataFrame,
    momentum: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build user-facing status, opportunity, allocation, and regime views."""
    view = _readiness_recent_metrics(readiness)
    if view.empty:
        return view, pd.DataFrame(), pd.DataFrame([{"asset": "Cash", "allocation_pct": 100.0}]), {
            "label": "UNKNOWN", "risk": "Unknown", "breadth_pct": 0.0, "trend_strength": 0.0
        }

    view["failed_conditions"] = view["failed_conditions"].fillna("").astype(str)
    view["failed_count"] = view["failed_conditions"].apply(lambda value: 0 if not value.strip() else len(value.split(";")))
    view["status"] = np.where(
        view["decision"].astype(str).str.upper().eq("GO"),
        "GO",
        np.where(view["failed_count"].le(1), "WATCHLIST", "NO-GO"),
    )
    view["criteria"] = view.apply(
        lambda row: (
            "Passed all validation criteria"
            if row["status"] == "GO"
            else (
                f"Almost GO — {row['failed_conditions']}"
                if row["status"] == "WATCHLIST"
                else row["failed_conditions"]
            )
        ),
        axis=1,
    )

    rank_columns = [column for column in [
        "symbol", "validated_score", "cnn_lstm_confidence", "smc_score", "walk_forward_stability"
    ] if column in rankings.columns]
    joined = view.merge(rankings[rank_columns], on="symbol", how="left")
    momentum_columns = [column for column in ["symbol", "momentum_score", "return_24h_pct"] if column in momentum.columns]
    if momentum_columns:
        joined = joined.merge(momentum[momentum_columns], on="symbol", how="left")
    for column in ["validated_score", "smc_score", "walk_forward_stability", "momentum_score"]:
        values = joined[column] if column in joined.columns else pd.Series(0.0, index=joined.index)
        joined[column] = pd.to_numeric(values, errors="coerce").fillna(0.0)
    confidence_values = joined["cnn_lstm_confidence"] if "cnn_lstm_confidence" in joined.columns else pd.Series(0.0, index=joined.index)
    model_confidence = pd.to_numeric(confidence_values, errors="coerce").fillna(0.0)
    if "return_24h_pct" not in joined.columns:
        joined["return_24h_pct"] = np.nan
    if model_confidence.max() <= 1.0:
        model_confidence *= 100.0
    momentum_rank_score = joined["momentum_score"].rank(pct=True).fillna(0.0) * 100.0
    joined["confidence_pct"] = (
        joined["validated_score"] * 0.30
        + model_confidence * 0.25
        + joined["smc_score"] * 0.20
        + joined["walk_forward_stability"] * 0.15
        + momentum_rank_score * 0.10
    ).clip(0.0, 100.0)
    joined["today_opportunity_score"] = (
        momentum_rank_score * 0.55 + joined["confidence_pct"] * 0.45
    ).clip(0.0, 100.0)
    opportunity = joined.sort_values("today_opportunity_score", ascending=False).head(10)[
        ["symbol", "status", "today_opportunity_score", "confidence_pct", "return_24h_pct"]
    ].copy()

    go = joined.loc[joined["status"].eq("GO")].copy()
    if go.empty:
        allocation = pd.DataFrame([{"asset": "Cash", "allocation_pct": 100.0}])
    else:
        weights = go["confidence_pct"].clip(lower=1.0)
        go_allocation = weights / weights.sum() * 90.0
        allocation = pd.DataFrame({"asset": go["symbol"], "allocation_pct": go_allocation.round(2)})
        allocation = pd.concat([allocation, pd.DataFrame([{"asset": "Cash", "allocation_pct": 10.0}])], ignore_index=True)

    market = momentum.copy()
    returns_24h = pd.to_numeric(market.get("return_24h_pct", pd.Series(dtype=float)), errors="coerce").dropna()
    breadth = float((returns_24h > 0.0).mean() * 100.0) if len(returns_24h) else 0.0
    median_return = float(returns_24h.median()) if len(returns_24h) else 0.0
    regime = {
        "label": "BULLISH" if breadth >= 60.0 and median_return > 0.0 else ("BEARISH" if breadth <= 40.0 else "MIXED"),
        "risk": "Low" if breadth >= 70.0 else ("High" if breadth <= 35.0 else "Medium"),
        "breadth_pct": breadth,
        "trend_strength": float(np.clip(abs(median_return) * 20.0, 0.0, 100.0)),
    }
    display_columns = [
        "symbol", "status", "confidence_pct", "return_7d_pct", "return_14d_pct", "return_30d_pct",
        "profit_factor_30d", "max_drawdown_30d_pct", "trade_count_30d", "criteria",
    ]
    return joined[[column for column in display_columns if column in joined.columns]], opportunity, allocation, regime


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
    readiness_display, opportunity, allocation, regime = _dashboard_decision_views(
        readiness, rankings, result.momentum_df
    )
    performance_display = performance.copy()

    for frame in [top_rankings, readiness_display, performance_display, opportunity, allocation]:
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_columns] = frame[numeric_columns].round(6)

    alert_items = "\n".join(f"<li>{alert}</li>" for alert in result.alerts) or "<li>No GO status changes detected.</li>"
    whitelist = ", ".join(result.go_assets) if result.go_assets else "none"
    no_go = ", ".join(result.no_go_assets) if result.no_go_assets else "none"
    watch_count = int(readiness_display.get("status", pd.Series(dtype=str)).eq("WATCHLIST").sum())
    no_go_count = int(readiness_display.get("status", pd.Series(dtype=str)).eq("NO-GO").sum())
    trade_status = "SAFE TO PAPER TRADE" if result.go_assets else "DO NOT TRADE TODAY"
    trade_class = "safe" if result.go_assets else "stop"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>V5 Daily Decision Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #e5e7eb; background: #0b1220; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #263449; padding: 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #172033; }}
    tr:nth-child(even) {{ background: #111a2b; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .metric {{ border: 1px solid #263449; background:#111a2b; padding: 12px; border-radius: 8px; }}
    .label {{ color: #9ca3af; font-size: 12px; }}
    .value {{ font-size: 20px; margin-top: 4px; }}
    .decision {{ padding:24px; border-radius:10px; font-size:30px; font-weight:700; margin:16px 0; }}
    .safe {{ background:#064e3b; color:#a7f3d0; }} .stop {{ background:#7f1d1d; color:#fecaca; }}
    .regime {{ display:flex; gap:16px; padding:14px; background:#111a2b; border-radius:8px; }}
  </style>
</head>
<body>
  <h1>V5 Daily Decision Dashboard</h1>
  <p>Paper-only research report. No live orders are submitted.</p>
  <div class="decision {trade_class}">{trade_status}</div>
  <div class="summary">
    <div class="metric"><div class="label">Market data refresh</div><div class="value">{result.refresh_succeeded}</div></div>
    <div class="metric"><div class="label">SMC feature files updated</div><div class="value">{result.smc_features_updated}</div></div>
    <div class="metric"><div class="label">Current whitelist</div><div class="value">{whitelist}</div></div>
    <div class="metric"><div class="label">Portfolio return</div><div class="value">{_format_metric(result.combined_return, "%")}</div></div>
    <div class="metric"><div class="label">GO assets</div><div class="value">{len(result.go_assets)}</div></div>
    <div class="metric"><div class="label">NO-GO assets</div><div class="value">{no_go_count}</div></div>
    <div class="metric"><div class="label">WATCHLIST assets</div><div class="value">{watch_count}</div></div>
  </div>

  <h2>Market Regime</h2>
  <div class="regime"><b>{regime['label']}</b><span>Risk: {regime['risk']}</span><span>Positive breadth: {regime['breadth_pct']:.1f}%</span><span>Trend strength: {regime['trend_strength']:.0f}</span></div>

  <h2>Today's Opportunity</h2>
  {opportunity.to_html(index=False, border=0)}

  <h2>Today's Paper Portfolio</h2>
  {allocation.to_html(index=False, border=0)}

  <h2>Alerts</h2>
  <ul>{alert_items}</ul>

  <h2>Current Whitelist</h2>
  <p>{whitelist}</p>

  <h2>NO-GO Assets</h2>
  <p>{no_go}</p>

  <h2>GO / NO-GO and Recent Metrics</h2>
  {readiness_display.to_html(index=False, border=0)}

  <h2>Current Market Momentum Leaders</h2>
  {result.momentum_df.to_html(index=False, border=0)}

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

    momentum = scan_market_momentum(
        timeframe,
        limit=int(getattr(Config, "MARKET_MOMENTUM_VALIDATION_CANDIDATES", 10)),
    )
    momentum_leaders = momentum["symbol"].astype(str).tolist() if not momentum.empty else []

    pre_rank_symbols = _safe_top_validated_symbols(timeframe, top_count)
    discovery_symbols = list(dict.fromkeys([*momentum_leaders, *pre_rank_symbols]))
    smc_outputs = _update_smc_features(discovery_symbols, timeframe)

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

    validation_symbols = sorted({str(symbol).upper() for symbol in list_gmx_symbols(timeframe)})
    readiness_result = run_paper_readiness(
        _daily_args(timeframe=timeframe, top_validated=top_count, symbols=validation_symbols)
    )
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
        momentum_df=momentum,
    )
    _write_dashboard(result, rankings, strict_readiness, performance)
    return result
