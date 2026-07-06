"""HTML reporting helpers for V4 backtest research output."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _trade_returns(trades: pd.DataFrame | None) -> list[float]:
    if trades is None or trades.empty:
        return []
    returns = []
    for _, trade in trades.iterrows():
        direction = str(trade.get("direction", "LONG")).upper()
        entry = trade.get("entry_price")
        exit_price = trade.get("exit_price")
        if entry is None or exit_price is None:
            continue
        entry = float(entry)
        exit_price = float(exit_price)
        if direction == "LONG":
            returns.append(((exit_price / entry) - 1) * 100.0)
        else:
            returns.append(((entry / exit_price) - 1) * 100.0)
    return returns


def _equity_series(trades: pd.DataFrame | None, starting_capital: float) -> pd.Series:
    if trades is None or trades.empty:
        return pd.Series([starting_capital], index=[pd.Timestamp.utcnow().tz_localize(None)])
    equity = [float(starting_capital)]
    capital = float(starting_capital)
    raw_timestamps = [pd.Timestamp.utcnow().tz_localize(None)]
    for _, trade in trades.iterrows():
        capital += float(trade.get("profit", 0.0))
        timestamp = pd.Timestamp(trade.get("timestamp"))
        if getattr(timestamp, "tzinfo", None) is not None:
            try:
                timestamp = timestamp.tz_convert(None)
            except Exception:
                timestamp = timestamp.tz_localize(None)
        raw_timestamps.append(timestamp)
        equity.append(capital)
    normalized_timestamps = pd.to_datetime(raw_timestamps, utc=True).tz_convert(None)
    return pd.Series(equity, index=normalized_timestamps)


def _figure_to_base64(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def write_v4_backtest_html_report(summary_df: pd.DataFrame, trades_by_symbol: dict[str, pd.DataFrame], output_path: str | Path, title: str = "V4 Backtest Research Report") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summary_df.copy()
    summary = summary.sort_values(by=["return_pct", "return_to_drawdown_ratio"], ascending=[False, False], na_position="last")

    top_assets = summary.head(10)
    bottom_assets = summary.tail(10)

    chart_assets = [symbol for symbol in trades_by_symbol if trades_by_symbol[symbol] is not None and not trades_by_symbol[symbol].empty][:8]
    if chart_assets:
        fig, ax = plt.subplots(figsize=(10, 4))
        for symbol in chart_assets:
            trades = trades_by_symbol[symbol]
            starting_capital = float(summary.loc[summary["symbol"] == symbol, "starting_capital"].iloc[0]) if (summary["symbol"] == symbol).any() else 100000.0
            equity = _equity_series(trades, starting_capital)
            if len(equity) > 1:
                ax.plot(equity.index, equity.values, label=symbol)
        ax.set_title("Equity Curves")
        ax.set_ylabel("Capital")
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
        equity_img = _figure_to_base64(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        for symbol in chart_assets:
            trades = trades_by_symbol[symbol]
            starting_capital = float(summary.loc[summary["symbol"] == symbol, "starting_capital"].iloc[0]) if (summary["symbol"] == symbol).any() else 100000.0
            equity = _equity_series(trades, starting_capital)
            if len(equity) > 1:
                peak = equity.cummax()
                drawdown = (equity / peak - 1) * 100
                ax.plot(drawdown.index, drawdown.values, label=symbol)
        ax.set_title("Drawdown Curves")
        ax.set_ylabel("Drawdown %")
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
        drawdown_img = _figure_to_base64(fig)
    else:
        equity_img = ""
        drawdown_img = ""

    combined_returns = []
    for symbol, trades in trades_by_symbol.items():
        combined_returns.extend(_trade_returns(trades))

    fig, ax = plt.subplots(figsize=(8, 4))
    if combined_returns:
        ax.hist(combined_returns, bins=20, color="#1f77b4")
    ax.set_title("Trade Return Distribution")
    ax.set_xlabel("Trade Return %")
    ax.set_ylabel("Count")
    fig.tight_layout()
    distribution_img = _figure_to_base64(fig)

    monthly_frames = []
    for symbol, trades in trades_by_symbol.items():
        if trades is None or trades.empty:
            continue
        starting_capital = float(summary.loc[summary["symbol"] == symbol, "starting_capital"].iloc[0]) if (summary["symbol"] == symbol).any() else 100000.0
        equity = _equity_series(trades, starting_capital)
        monthly = equity.resample("ME").last().pct_change().dropna() * 100
        monthly = monthly.rename(symbol)
        monthly_frames.append(monthly)
    if monthly_frames:
        monthly_df = pd.concat(monthly_frames, axis=1)
        monthly_df = monthly_df.fillna(0.0)
        if not monthly_df.empty and not monthly_df.columns.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            monthly_df.plot(ax=ax, linewidth=1.0)
            ax.set_title("Monthly Returns")
            ax.set_ylabel("Return %")
            fig.tight_layout()
            monthly_img = _figure_to_base64(fig)
        else:
            monthly_img = ""
    else:
        monthly_img = ""

    summary_html = summary[[
        "symbol",
        "return_pct",
        "final_capital",
        "total_profit",
        "signals_traded",
        "win_rate_pct",
        "max_drawdown_pct",
        "profit_factor",
        "sharpe_ratio",
        "sortino_ratio",
        "average_trade_return",
        "expectancy",
        "average_win",
        "average_loss",
        "largest_win",
        "largest_loss",
        "consecutive_wins",
        "consecutive_losses",
    ]].head(20).to_html(index=False, escape=False)

    top_html = top_assets[[
        "symbol",
        "return_pct",
        "final_capital",
        "total_profit",
        "signals_traded",
        "win_rate_pct",
        "max_drawdown_pct",
        "profit_factor",
        "return_to_drawdown_ratio",
    ]].head(10).to_html(index=False, escape=False)

    bottom_html = bottom_assets[[
        "symbol",
        "return_pct",
        "final_capital",
        "total_profit",
        "signals_traded",
        "win_rate_pct",
        "max_drawdown_pct",
        "profit_factor",
        "return_to_drawdown_ratio",
    ]].head(10).to_html(index=False, escape=False)

    summary_stats = summary[["return_pct", "max_drawdown_pct", "profit_factor", "sharpe_ratio", "sortino_ratio", "expectancy"]].describe().T
    summary_stats_html = summary_stats.to_html()

    html = f"""
    <html>
      <head>
        <meta charset='utf-8'>
        <title>{title}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          h1, h2 {{ color: #1f2937; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 12px; }}
          th, td {{ border: 1px solid #d1d5db; padding: 6px; text-align: left; }}
          th {{ background: #f3f4f6; }}
          img {{ width: 100%; max-width: 1000px; height: auto; margin-bottom: 20px; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        <h2>Summary Statistics</h2>
        {summary_stats_html}
        <h2>Equity Curve</h2>
        {f'<img src="data:image/png;base64,{equity_img}">' if equity_img else '<p>No equity curve data available.</p>'}
        <h2>Drawdown Curve</h2>
        {f'<img src="data:image/png;base64,{drawdown_img}">' if drawdown_img else '<p>No drawdown data available.</p>'}
        <h2>Monthly Returns</h2>
        {f'<img src="data:image/png;base64,{monthly_img}">' if monthly_img else '<p>No monthly return data available.</p>'}
        <h2>Trade Return Distribution</h2>
        {f'<img src="data:image/png;base64,{distribution_img}">' if distribution_img else '<p>No trade return data available.</p>'}
        <h2>Performance Ranking Table</h2>
        {summary_html}
        <h2>Top Assets</h2>
        {top_html}
        <h2>Bottom Assets</h2>
        {bottom_html}
      </body>
    </html>
    """
    output_path.write_text(html, encoding="utf-8")
    return output_path
