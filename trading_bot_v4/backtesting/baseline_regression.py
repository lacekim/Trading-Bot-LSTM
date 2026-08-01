"""Direct comparison of two all-assets baseline reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REFERENCE_PATH = Path("v4_ranked_backtest_summary.csv")
CURRENT_PATH = Path("trading_bot_gmx_1h_all_assets_summary.csv")
OUTPUT_CSV = Path("reports/v5_baseline_regression_comparison.csv")
OUTPUT_JSON = Path("reports/v5_baseline_regression_summary.json")


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(frame["return_pct"], errors="coerce").dropna()
    return {
        "assets": int(len(returns)), "mean_return_pct": float(returns.mean()),
        "median_return_pct": float(returns.median()),
        "profitable_assets": int((returns > 0).sum()),
        "losing_assets": int((returns < 0).sum()), "flat_assets": int((returns == 0).sum()),
        "best_return_pct": float(returns.max()), "worst_return_pct": float(returns.min()),
    }


def run_baseline_regression(args: Any | None = None) -> dict[str, Any]:
    reference_path = Path(getattr(args, "reference_report", REFERENCE_PATH)) if args else REFERENCE_PATH
    current_path = Path(getattr(args, "current_report", CURRENT_PATH)) if args else CURRENT_PATH
    reference, current = pd.read_csv(reference_path), pd.read_csv(current_path)
    for name, frame in (("reference", reference), ("current", current)):
        if not {"symbol", "return_pct"}.issubset(frame.columns):
            raise ValueError(f"{name} report lacks symbol/return_pct columns")
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    comparison = reference[["symbol", "return_pct"]].rename(columns={"return_pct": "reference_return_pct"}).merge(
        current[["symbol", "return_pct"]].rename(columns={"return_pct": "current_return_pct"}),
        on="symbol", how="inner",
    )
    comparison["change_percentage_points"] = comparison["current_return_pct"] - comparison["reference_return_pct"]
    comparison = comparison.sort_values("change_percentage_points")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_CSV, index=False)
    reference_metrics, current_metrics = _metrics(reference), _metrics(current)
    payload = {
        "reference_report": str(reference_path), "current_report": str(current_path),
        "report_format": "independent per-asset backtests; not a shared-capital portfolio",
        "reference": reference_metrics, "current": current_metrics,
        "mean_change_percentage_points": current_metrics["mean_return_pct"] - reference_metrics["mean_return_pct"],
        "median_change_percentage_points": current_metrics["median_return_pct"] - reference_metrics["median_return_pct"],
        "profitable_asset_change": current_metrics["profitable_assets"] - reference_metrics["profitable_assets"],
        "matched_assets": int(len(comparison)),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Per-asset comparison: {OUTPUT_CSV}")
    return payload
