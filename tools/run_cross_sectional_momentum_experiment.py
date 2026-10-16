"""Cross-sectional momentum: one of the most robustly replicated findings in
empirical finance (Jegadeesh & Titman 1993 for equities; separately
confirmed across futures, currencies, and crypto specifically), and
structurally different from everything tested earlier tonight.

Every prior experiment asked an absolute question about one asset in
isolation: "will BTC go up." This asks a relative question across the whole
universe at once: "which assets are strongest/weakest *right now, relative
to each other*, and does that ranking persist." Go long the strongest
decile, short the weakest, hold, rebalance. No single-asset direction
prediction at all -- the edge (if any) is in the ranking, not in any one
asset's absolute forecast.

Doesn't depend on any new data source (funding data collector is dead,
sentiment data covers only 17 assets) -- runs entirely on the OHLC history
already fully available for the whole GMX universe. Scored through the
same real-execution backtest engine (fees, slippage) used all night, walked
forward across multiple non-overlapping periods rather than a single split.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_cross_sectional_momentum_experiment")

TIMEFRAME = "1h"
LOOKBACK_CANDLES = 24 * 14   # rank by trailing 14-day return
HOLD_CANDLES = 24 * 7        # hold each ranked position for 7 days before rebalancing
TOP_N = 8                    # long the N strongest
BOTTOM_N = 8                 # short the N weakest
MIN_HISTORY_CANDLES = 24 * 400  # ~400 days minimum, so ranking has a real trailing window
STARTING_CAPITAL = 100_000.0


def _load_close_series(symbol: str) -> pd.Series | None:
    df = load_gmx_ohlc(symbol, TIMEFRAME)
    if df is None or df.empty or len(df) < MIN_HISTORY_CANDLES:
        return None
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    series = df.set_index("timestamp")["Close"].astype(float)
    return series[~series.index.duplicated(keep="last")]


def build_universe(symbols: list[str]) -> pd.DataFrame:
    closes = {}
    for symbol in symbols:
        series = _load_close_series(symbol)
        if series is not None:
            closes[symbol] = series
    frame = pd.DataFrame(closes)
    return frame.sort_index().ffill(limit=6)  # small gap tolerance only, no long forward-fills


def backtest(prices: pd.DataFrame) -> dict:
    trailing_return = prices.pct_change(LOOKBACK_CANDLES)
    rebalance_points = range(LOOKBACK_CANDLES, len(prices) - HOLD_CANDLES, HOLD_CANDLES)

    capital = STARTING_CAPITAL
    peak = capital
    max_drawdown = 0.0
    trades: list[dict] = []

    fee_rate = Config.PAPER_FEE_BPS / 10000.0
    slip = (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) / 10000.0
    round_trip_cost = 2 * (fee_rate + slip)

    for i in rebalance_points:
        ranks = trailing_return.iloc[i].dropna().sort_values()
        if len(ranks) < TOP_N + BOTTOM_N:
            continue
        longs = ranks.index[-TOP_N:]
        shorts = ranks.index[:BOTTOM_N]

        entry_prices = prices.iloc[i]
        exit_prices = prices.iloc[i + HOLD_CANDLES]

        # A symbol can rank cleanly at entry (valid trailing return) and still
        # go missing by the exit candle (delisted, deprecated pair, data gap).
        # One NaN return must not poison every subsequent period's capital --
        # drop only that leg for this rebalance, don't let it propagate.
        valid_longs = [s for s in longs if np.isfinite(entry_prices.get(s, np.nan)) and np.isfinite(exit_prices.get(s, np.nan))]
        valid_shorts = [s for s in shorts if np.isfinite(entry_prices.get(s, np.nan)) and np.isfinite(exit_prices.get(s, np.nan))]
        dropped = (len(longs) - len(valid_longs)) + (len(shorts) - len(valid_shorts))
        if dropped:
            logger.info(f"rebalance {prices.index[i]}: dropped {dropped} leg(s) with missing exit price")
        if not valid_longs and not valid_shorts:
            continue

        leg_notional = capital / (len(valid_longs) + len(valid_shorts))
        period_pnl = 0.0
        for symbol in valid_longs:
            ret = exit_prices[symbol] / entry_prices[symbol] - 1.0
            period_pnl += leg_notional * (ret - round_trip_cost)
        for symbol in valid_shorts:
            ret = exit_prices[symbol] / entry_prices[symbol] - 1.0
            period_pnl += leg_notional * (-ret - round_trip_cost)

        capital += period_pnl
        trades.append({
            "timestamp": prices.index[i], "pnl": period_pnl,
            "longs": list(longs), "shorts": list(shorts),
        })
        peak = max(peak, capital)
        max_drawdown = min(max_drawdown, (capital / peak - 1.0) * 100.0)

    trade_frame = pd.DataFrame(trades)
    wins = trade_frame.loc[trade_frame["pnl"] > 0] if not trade_frame.empty else trade_frame
    losses = trade_frame.loc[trade_frame["pnl"] < 0] if not trade_frame.empty else trade_frame
    gross_wins = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_losses = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
    return {
        "rebalances": len(trade_frame),
        "final_capital": capital,
        "return_pct": (capital / STARTING_CAPITAL - 1.0) * 100.0,
        "win_rate_pct": len(wins) / len(trade_frame) * 100.0 if len(trade_frame) else 0.0,
        "max_drawdown_pct": max_drawdown,
        "profit_factor": gross_wins / gross_losses if gross_losses else (float("inf") if gross_wins else 0.0),
        "trades": trade_frame,
    }


def main() -> None:
    symbols = list_gmx_symbols(TIMEFRAME)
    logger.info(f"loading OHLC for {len(symbols)} symbols (requires >= {MIN_HISTORY_CANDLES} candles of history)...")
    prices = build_universe(symbols)
    logger.info(f"universe: {prices.shape[1]} symbols with sufficient history, "
                f"{prices.index.min()} to {prices.index.max()}")

    result = backtest(prices)
    logger.info(f"FULL PERIOD: rebalances={result['rebalances']} return_pct={result['return_pct']:.2f} "
                f"win_rate_pct={result['win_rate_pct']:.2f} profit_factor={result['profit_factor']:.2f} "
                f"max_drawdown_pct={result['max_drawdown_pct']:.2f}")

    # walk-forward across 4 non-overlapping eras instead of trusting one full-period number
    n = len(prices)
    edges = np.linspace(0, n, 5).astype(int)
    for era in range(4):
        era_prices = prices.iloc[edges[era]:edges[era + 1]]
        if len(era_prices) < LOOKBACK_CANDLES + HOLD_CANDLES * 2:
            logger.info(f"era {era + 1}: insufficient rows, skipping")
            continue
        era_result = backtest(era_prices)
        logger.info(f"era {era + 1} ({era_prices.index.min()} to {era_prices.index.max()}): "
                    f"rebalances={era_result['rebalances']} return_pct={era_result['return_pct']:.2f} "
                    f"win_rate_pct={era_result['win_rate_pct']:.2f} profit_factor={era_result['profit_factor']:.2f} "
                    f"max_drawdown_pct={era_result['max_drawdown_pct']:.2f}")


if __name__ == "__main__":
    main()
