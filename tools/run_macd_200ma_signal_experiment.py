"""Fixed, non-learned rule: trade with the 200-period trend, confirmed by
MACD momentum -- LONG when Close > MA200 and macd > macd_signal, SHORT when
Close < MA200 and macd < macd_signal, else HOLD.

This is deliberately NOT a model: nothing here is fit to historical data, so
none of the failure modes found and fixed all session (lookahead bias,
test-set cherry-picking, promotion-gate small-sample luck) can apply. Every
ML approach tonight already had macd_norm as an input feature and still
found no edge -- this tests whether a fixed, un-learned combination of the
same underlying signal behaves differently, not whether the information was
available.

Same real-execution costs (simulate_production_symbol: ATR stop/target, fees,
slippage) and liquid-symbols-only universe as everything else this session.
Reports both the full-history aggregate AND an 8-fold walk-forward binomial
significance test, same discipline as run_smc_significance_test.py /
run_momentum_significance_test.py -- a good-looking aggregate number is still
one look.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import list_gmx_symbols, load_gmx_ohlc
from trading_bot_v4.risk.market_cap_tiers import liquid_symbols
from trading_bot_v4.utils.logger import build_logger
from train_model import DataHandler

logger = build_logger("v4_macd_200ma_experiment")

TIMEFRAME = "1h"
STARTING_CAPITAL = 100_000.0
N_FOLDS = 8
MIN_ROWS = 250  # need at least 200 for the MA plus room for signals


def _load_signal_frame(symbol: str, direction: str) -> pd.DataFrame | None:
    df = load_gmx_ohlc(symbol, TIMEFRAME)
    if df is None or len(df) < MIN_ROWS:
        return None
    df = df.copy()

    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr

    df["ma200"] = df["Close"].rolling(window=200).mean()
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI(14), same formula as train_model.py
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Stochastic RSI(14, 3, 3): stochastic oscillator applied to RSI itself,
    # then %K/%D smoothed -- the actual entry trigger (pullback recovery
    # within the broader trend), not the trend/momentum filters above.
    rsi_min = df["rsi"].rolling(window=14).min()
    rsi_max = df["rsi"].rolling(window=14).max()
    stoch_rsi_raw = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    df["stoch_k"] = stoch_rsi_raw.rolling(window=3).mean()
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    df = df.dropna(subset=["ma200", "macd", "macd_signal", "rsi", "stoch_k", "stoch_d", "ATR"]).reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    if df.empty:
        return None

    if direction == "long":
        label = "LONG"
        trend_ok = (df["Close"] > df["ma200"]) & (df["macd"] > df["macd_signal"]) & (df["rsi"] < 70)
        # Entry trigger: Stoch RSI %K crosses up through %D while still in/near
        # oversold territory -- a pullback bottoming out inside an established
        # uptrend, not a bare oscillator wiggle.
        stoch_trigger = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) & (df["stoch_k"] < 40)
    else:
        label = "SHORT"
        trend_ok = (df["Close"] < df["ma200"]) & (df["macd"] < df["macd_signal"]) & (df["rsi"] > 30)
        stoch_trigger = (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) & (df["stoch_k"] > 60)

    condition = trend_ok & stoch_trigger
    df["model_direction"] = np.where(condition, label, "HOLD")

    df["open"] = df["Open"].astype(float)
    df["high"] = df["High"].astype(float)
    df["low"] = df["Low"].astype(float)
    df["close"] = df["Close"].astype(float)
    df["price"] = df["Close"].astype(float)
    df["atr"] = df["ATR"].astype(float)
    df["_entry_eligible"] = True
    df["_risk_pct"] = float(Config.RISK_PERCENTAGE)
    df["_max_position_pct"] = float(Config.PAPER_MAX_POSITION_PCT)
    return df


def evaluate(symbol: str, direction: str) -> dict | None:
    df = _load_signal_frame(symbol, direction)
    if df is None:
        return None

    summary, _ = simulate_production_symbol(
        df, symbol, STARTING_CAPITAL, entry_eligible=True,
        fee_bps=Config.PAPER_FEE_BPS,
        slippage_bps=Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS,
    )

    fold_results = []
    edges = np.linspace(0, len(df), N_FOLDS + 1).astype(int)
    for fold in range(N_FOLDS):
        frame = df.iloc[edges[fold]:edges[fold + 1]]
        if len(frame) < 20:
            continue
        fold_summary, _ = simulate_production_symbol(
            frame, symbol, STARTING_CAPITAL, entry_eligible=True,
            fee_bps=Config.PAPER_FEE_BPS,
            slippage_bps=Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS,
        )
        if fold_summary["trades"] == 0:
            continue
        fold_results.append(fold_summary)

    n_folds = len(fold_results)
    n_profitable = sum(1 for r in fold_results if r["return_pct"] > 0)
    binom_p = (
        stats.binomtest(n_profitable, n_folds, p=0.5, alternative="greater").pvalue
        if n_folds > 0 else float("nan")
    )

    result = {
        "symbol": symbol, "direction": direction,
        "total_trades": summary["trades"], "return_pct": summary["return_pct"],
        "profit_factor": summary["profit_factor"], "win_rate_pct": summary["win_rate_pct"],
        "n_folds": n_folds, "n_profitable_folds": n_profitable, "binom_p": binom_p,
    }
    logger.info(
        f"{direction}/{symbol}: trades={result['total_trades']} return_pct={result['return_pct']:.2f} "
        f"profit_factor={result['profit_factor']:.2f} folds={n_profitable}/{n_folds} "
        f"binom_p={binom_p:.3f}"
    )
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    args = parser.parse_args()
    directions = ["long", "short"] if args.direction == "both" else [args.direction]

    all_symbols = list_gmx_symbols(TIMEFRAME)
    results = []
    for direction in directions:
        symbols = [s.upper() for s in args.symbols] if args.symbols else liquid_symbols(direction.upper(), all_symbols)
        # This is a trend-following rule meant to ride a position for as long
        # as the trend holds, not the ~8-candle horizon production_backtest.py
        # defaults to for the short-horizon ML models used all session --
        # enable the existing PersistentPolicy (240-candle/10-day max hold)
        # instead of building a new mechanism, but override its 0.5 ATR stop
        # (tuned for a different, tighter strategy) with a conventional
        # trend-following stop wide enough to survive normal noise while a
        # real trend develops.
        os.environ["PERSISTENT_POLICY_SYMBOLS"] = ",".join(symbols)
        os.environ["PERSISTENT_STOP_ATR"] = "3.0"
        # Config.PAPER_MIN_BARS_BETWEEN_TRADES is read once at class-body
        # execution (import time), so the env var alone wouldn't take effect
        # here -- set the class attribute directly instead.
        Config.PAPER_MIN_BARS_BETWEEN_TRADES = 24
        logger.info(f"{direction}: evaluating {len(symbols)} liquid symbols (persistent policy enabled)")
        for symbol in symbols:
            result = evaluate(symbol, direction)
            if result:
                results.append(result)

    import json
    out = Path("logs/v4_macd_200ma_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}, {len(results)} symbol/direction results")
