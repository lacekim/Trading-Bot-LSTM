"""Breaks down the MACD+200MA+RSI+StochRSI rule signal's actual trades by two
market-context factors instead of treating every trade as interchangeable:

  - Calendar quarter (Q1-Q4) the trade was entered in -- tests for real
    seasonality rather than assuming it.
  - Market-wide activity regime at entry time -- BTC's own rolling volume and
    volatility relative to their trailing baseline, as a proxy for "how
    active is the market as a whole" (BTC leads/dominates), classified HIGH
    when short-term activity exceeds the long-term baseline, LOW otherwise.
    Both rolling windows are strictly backward-looking, so no lookahead.

Reuses the exact rule/backtest logic already validated
(run_macd_200ma_signal_experiment.py) -- only difference is capturing the
per-trade record instead of just the aggregate summary, so trades can be
grouped by these two factors.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import list_gmx_symbols, load_gmx_ohlc
from trading_bot_v4.risk.market_cap_tiers import liquid_symbols
from trading_bot_v4.utils.logger import build_logger
from tools.run_macd_200ma_signal_experiment import STARTING_CAPITAL, TIMEFRAME, _load_signal_frame

logger = build_logger("v4_quarter_activity_analysis")

SHORT_WINDOW = 24 * 30   # 30 days of hourly candles
LONG_WINDOW = 24 * 180   # 180 days of hourly candles


def _btc_activity_regime() -> pd.DataFrame:
    """BTC's own rolling volume/ATR relative to their trailing baseline, as a
    market-wide activity proxy. Both windows only look backward from each
    bar, so this is safe to join onto any other symbol's trade timestamps."""
    df = load_gmx_ohlc("BTC", TIMEFRAME)
    df = df.copy()
    from train_model import DataHandler
    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr

    volume_short = df["Volume"].rolling(SHORT_WINDOW).mean()
    volume_long = df["Volume"].rolling(LONG_WINDOW).mean()
    atr_short = df["ATR"].rolling(SHORT_WINDOW).mean()
    atr_long = df["ATR"].rolling(LONG_WINDOW).mean()

    activity_score = 0.5 * (volume_short / volume_long) + 0.5 * (atr_short / atr_long)
    regime = np.where(activity_score > 1.0, "HIGH", "LOW")

    out = df.reset_index()[[df.reset_index().columns[0]]].copy()
    out.columns = ["timestamp"]
    out["activity_regime"] = regime
    out = out.dropna(subset=["activity_regime"]) if out["activity_regime"].isna().any() else out
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.set_index("timestamp").sort_index()


def _tag_regime(entry_time: pd.Timestamp, btc_regime: pd.DataFrame) -> str:
    idx = btc_regime.index.searchsorted(entry_time, side="right") - 1
    if idx < 0:
        return "UNKNOWN"
    return btc_regime.iloc[idx]["activity_regime"]


def collect_all_trades(direction: str, symbols: list[str], btc_regime: pd.DataFrame) -> pd.DataFrame:
    all_trades = []
    for symbol in symbols:
        df = _load_signal_frame(symbol, direction)
        if df is None:
            continue
        _, trades = simulate_production_symbol(
            df, symbol, STARTING_CAPITAL, entry_eligible=True,
            fee_bps=Config.PAPER_FEE_BPS,
            slippage_bps=Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS,
        )
        if trades is None or trades.empty:
            continue
        trades = trades.copy()
        trades["symbol"] = symbol
        trades["quarter"] = pd.to_datetime(trades["entry_time"], utc=True).dt.quarter
        trades["year"] = pd.to_datetime(trades["entry_time"], utc=True).dt.year
        trades["activity_regime"] = [
            _tag_regime(pd.Timestamp(t, tz="UTC") if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t), btc_regime)
            for t in trades["entry_time"]
        ]
        all_trades.append(trades)
        logger.info(f"{direction}/{symbol}: {len(trades)} trades collected")
    if not all_trades:
        return pd.DataFrame()
    return pd.concat(all_trades, ignore_index=True)


def _summarize(group: pd.DataFrame) -> dict:
    gains = group.loc[group["net_pnl"] > 0, "net_pnl"].sum()
    losses = -group.loc[group["net_pnl"] < 0, "net_pnl"].sum()
    pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
    win_rate = float((group["net_pnl"] > 0).mean() * 100.0)
    return {
        "trades": len(group), "total_net_pnl": float(group["net_pnl"].sum()),
        "profit_factor": pf, "win_rate_pct": win_rate,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    args = parser.parse_args()
    directions = ["long", "short"] if args.direction == "both" else [args.direction]

    logger.info("computing BTC market-activity regime...")
    btc_regime = _btc_activity_regime()

    all_symbols = list_gmx_symbols(TIMEFRAME)
    all_trades = []
    for direction in directions:
        symbols = liquid_symbols(direction.upper(), all_symbols)
        trades = collect_all_trades(direction, symbols, btc_regime)
        if not trades.empty:
            trades["direction"] = direction
            all_trades.append(trades)

    pooled = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if pooled.empty:
        print("no trades collected")
        sys.exit(0)

    pooled.to_csv("logs/v4_quarter_activity_trades.csv", index=False)
    print(f"\n{len(pooled)} total trades collected across {pooled['symbol'].nunique()} symbols\n")

    print("=== By calendar quarter ===")
    for q, group in pooled.groupby("quarter"):
        s = _summarize(group)
        print(f"Q{q}: trades={s['trades']:>5} net_pnl=${s['total_net_pnl']:>12,.2f} "
              f"profit_factor={s['profit_factor']:.2f} win_rate={s['win_rate_pct']:.1f}%")

    print("\n=== By market activity regime (BTC-derived) ===")
    for regime, group in pooled.groupby("activity_regime"):
        s = _summarize(group)
        print(f"{regime:>7}: trades={s['trades']:>5} net_pnl=${s['total_net_pnl']:>12,.2f} "
              f"profit_factor={s['profit_factor']:.2f} win_rate={s['win_rate_pct']:.1f}%")

    print("\n=== By quarter x activity regime ===")
    for (q, regime), group in pooled.groupby(["quarter", "activity_regime"]):
        s = _summarize(group)
        print(f"Q{q}/{regime:>7}: trades={s['trades']:>5} net_pnl=${s['total_net_pnl']:>12,.2f} "
              f"profit_factor={s['profit_factor']:.2f} win_rate={s['win_rate_pct']:.1f}%")
