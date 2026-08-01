"""Causal rolling asset selection over original one-candle trade outcomes.

The selector may only use outcomes timestamped before each selection day.  It
is a shared-capital diagnostic, unlike the independent-capital asset reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot_v4.config_v4 import V4Config as Config


REPORT_PATH = Path("reports/v5_causal_portfolio.json")
EQUITY_PATH = Path("reports/v5_causal_portfolio_equity.csv")


def _trade_returns(trades: pd.DataFrame, include_costs: bool = True) -> pd.DataFrame:
    frame = trades.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    prior_capital = pd.to_numeric(frame["capital"], errors="coerce") - pd.to_numeric(
        frame["profit"], errors="coerce"
    )
    frame["trade_return"] = pd.to_numeric(frame["profit"], errors="coerce") / prior_capital.replace(0, np.nan)
    if include_costs:
        notional_fraction = (
            pd.to_numeric(frame["entry_price"], errors="coerce")
            * pd.to_numeric(frame["units"], errors="coerce")
            / prior_capital.replace(0, np.nan)
        )
        round_trip_bps = (
            Config.PAPER_FEE_BPS * 2.0
            + (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) * 2.0
        )
        frame["trade_return"] -= notional_fraction * round_trip_bps / 10_000.0
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["timestamp", "symbol", "trade_return"]
    ).sort_values("timestamp")


def run_causal_portfolio(
    trades_path: Path = Path("trading_bot_gmx_1h_all_assets_trades.csv"),
    starting_capital: float = 100_000.0,
    lookback_days: int = 14,
    minimum_observations: int = 50,
    minimum_profit_factor: float = 1.25,
    selected_assets: int = 1,
    risk_multiplier: float = 1.5,
    include_costs: bool = True,
    report_path: Path = REPORT_PATH,
    equity_path: Path = EQUITY_PATH,
) -> dict[str, object]:
    if not trades_path.exists():
        raise FileNotFoundError(f"Original all-asset trades are required: {trades_path}")
    frame = _trade_returns(pd.read_csv(trades_path), include_costs=include_costs)
    if frame.empty:
        raise ValueError("Original all-asset trade report contains no usable trades")

    first_day = frame["timestamp"].min().floor("D")
    last_day = frame["timestamp"].max().floor("D")
    days = pd.date_range(first_day, last_day, freq="D", tz="UTC")
    equity = float(starting_capital)
    peak = equity
    maximum_drawdown = 0.0
    executed = 0
    rows: list[dict[str, object]] = []

    for day in days:
        history = frame.loc[
            (frame["timestamp"] < day)
            & (frame["timestamp"] >= day - pd.Timedelta(days=lookback_days))
        ]
        stats = history.groupby("symbol")["trade_return"].agg(["count", "sum", "mean"])
        positive = history.loc[history["trade_return"] > 0].groupby("symbol")["trade_return"].sum()
        negative = -history.loc[history["trade_return"] < 0].groupby("symbol")["trade_return"].sum()
        stats["profit_factor"] = positive / negative
        no_losses = negative.reindex(stats.index).fillna(0.0).eq(0.0)
        has_wins = positive.reindex(stats.index).fillna(0.0).gt(0.0)
        stats.loc[no_losses & has_wins, "profit_factor"] = float("inf")
        stats["profit_factor"] = stats["profit_factor"].fillna(0.0)
        stats["score"] = stats["mean"] * np.log1p(stats["count"])
        qualified = stats.loc[
            (stats["count"] >= minimum_observations)
            & (stats["profit_factor"] >= minimum_profit_factor)
            & (stats["sum"] > 0)
        ].nlargest(selected_assets, "score")
        selected = set(qualified.index.astype(str))

        today = frame.loc[
            (frame["timestamp"] >= day)
            & (frame["timestamp"] < day + pd.Timedelta(days=1))
            & frame["symbol"].astype(str).isin(selected)
        ]
        for _timestamp, signals in today.groupby("timestamp"):
            portfolio_return = (
                float(signals["trade_return"].sum())
                * float(risk_multiplier)
                / max(1, int(selected_assets))
            )
            equity *= 1.0 + portfolio_return
            executed += len(signals)
            peak = max(peak, equity)
            maximum_drawdown = min(maximum_drawdown, equity / peak - 1.0)
        rows.append({
            "date": day, "equity": equity, "selected_symbols": ",".join(sorted(selected)),
            "rolling_candidates": int(len(stats)), "trades": int(len(today)),
        })

    curve = pd.DataFrame(rows)
    split_points = [max(1, int(len(curve) * fraction)) - 1 for fraction in (0.50, 0.75, 1.0)]
    first, second, final = [float(curve.iloc[index]["equity"]) for index in split_points]
    payload: dict[str, object] = {
        "strategy": "causal_rolling_original_one_candle_top_asset",
        "starting_capital": float(starting_capital),
        "final_capital": final,
        "total_return_pct": (final / starting_capital - 1.0) * 100.0,
        "first_half_return_pct": (first / starting_capital - 1.0) * 100.0,
        "next_quarter_return_pct": (second / first - 1.0) * 100.0,
        "final_quarter_return_pct": (final / second - 1.0) * 100.0,
        "maximum_drawdown_pct": maximum_drawdown * 100.0,
        "trades": int(executed),
        "lookback_days": int(lookback_days),
        "minimum_observations": int(minimum_observations),
        "minimum_profit_factor": float(minimum_profit_factor),
        "selected_assets": int(selected_assets),
        "risk_multiplier": float(risk_multiplier),
        "costs_included": bool(include_costs),
        "round_trip_cost_bps": float(
            Config.PAPER_FEE_BPS * 2.0
            + (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) * 2.0
        ) if include_costs else 0.0,
        "causality": "Each daily selection uses only trade outcomes timestamped before that day.",
        "limitation": (
            "The loaded model may have been trained on candles inside this period. "
            "The risk multiplier and selector parameters require frozen-artifact forward validation before promotion."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    equity_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(equity_path, index=False)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload
