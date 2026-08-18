"""Demonstration: does a good price-level MAPE actually mean a model has
trading edge?

Prompted by a paper (Sun 2024, "Cryptocurrency price prediction based on
Xgboost, LightGBM and BNN") that trains models to predict next-period BTC/
ETH/LTC *price level* and reports ~8.8% MAPE for BTC as evidence the
approach works -- but never checks directional accuracy or runs the
predictions through an actual backtest.

This replicates that exact approach on our own BTC data to check the claim
directly instead of arguing about it abstractly:
  1. Train a regressor to predict next-hour Close from the same kind of TA
     features used tonight (point-in-time, no lookahead).
  2. Score it with MAPE, like the paper does.
  3. Score a naive "next price = current price" baseline with the same
     MAPE, on the same test set -- if the trained model isn't much better
     than doing nothing, the "good" MAPE is mostly measuring price
     autocorrelation, not predictive skill.
  4. Check directional accuracy separately (did it call up/down correctly)
     -- the thing MAPE never measures but trading actually needs.
  5. Convert the model's predicted price into LONG/HOLD signals and run
     them through the same real-execution simulator (fees, slippage, ATR
     stop/target) used for every other experiment tonight, to see whether
     a "good MAPE" model is actually profitable.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from train_model import DataHandler

SYMBOL = "BTC"
TIMEFRAME = "1h"
TRAIN_FRACTION = 0.75  # matches the paper's stated 75/25 split


def main() -> None:
    df = load_gmx_ohlc(SYMBOL, TIMEFRAME)
    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df = df.copy()
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr
    df = handler.prepare_features(df, prediction_horizon=1)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})

    df["next_close"] = df["Close"].shift(-1)
    df = df.dropna(subset=["next_close"]).reset_index(drop=True)

    # The paper's dataset explicitly includes raw Open/High/Low/Close as
    # columns (Section 2), so a fair replication needs a raw-price anchor
    # too -- Config.FEATURE_COLUMNS alone is all normalized/relative
    # indicators with no absolute price scale in it, which would make price-
    # level prediction meaningless as a comparison, not a fair analogue.
    feature_columns = [*Config.FEATURE_COLUMNS, "Close"]

    n = len(df)
    split = int(n * TRAIN_FRACTION)
    train, test = df.iloc[:split], df.iloc[split:]
    print(f"{SYMBOL} {TIMEFRAME}: {n} rows, train={len(train)}, test={len(test)}")

    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.10, random_state=42,
    )
    model.fit(train[feature_columns].to_numpy(np.float32), train["next_close"].to_numpy(np.float32))

    predicted = model.predict(test[feature_columns].to_numpy(np.float32))
    actual = test["next_close"].to_numpy(np.float64)
    current = test["Close"].to_numpy(np.float64)

    def mape(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.mean(np.abs((true - pred) / true)) * 100.0)

    model_mape = mape(predicted, actual)
    naive_mape = mape(current, actual)  # "tomorrow's price = today's price"

    predicted_up = predicted > current
    actual_up = actual > current
    naive_directional_accuracy = 0.5  # a persistence model has literally no directional opinion
    model_directional_accuracy = float((predicted_up == actual_up).mean()) * 100.0

    print("\n=== Step 1-3: does the model beat a do-nothing baseline on MAPE? ===")
    print(f"  Trained model MAPE:  {model_mape:.3f}%")
    print(f"  Naive persistence MAPE ('tomorrow = today'): {naive_mape:.3f}%")
    print(f"  Model improvement over naive: {naive_mape - model_mape:+.3f} percentage points "
          f"({(1 - model_mape / naive_mape) * 100:+.1f}% relative)")

    print("\n=== Step 4: directional accuracy (what MAPE never measures) ===")
    print(f"  Model correctly called up/down: {model_directional_accuracy:.2f}% of test candles")
    print(f"  Coin-flip baseline: 50.00%")

    print("\n=== Step 5: does this translate into real trading profit? ===")
    threshold_pct = 0.0  # any predicted increase at all triggers a LONG, matching the paper's optimism literally
    signals = test.copy().reset_index(drop=True)
    signals["predicted_next_close"] = predicted
    signals["model_direction"] = np.where(
        signals["predicted_next_close"] > signals["Close"] * (1 + threshold_pct / 100.0), "LONG", "HOLD"
    )
    signals["price"] = signals["Close"].astype(float)
    signals["open"] = signals["Open"].astype(float)
    signals["high"] = signals["High"].astype(float)
    signals["low"] = signals["Low"].astype(float)
    signals["close"] = signals["Close"].astype(float)
    signals["atr"] = signals["ATR"].astype(float)
    signals["_entry_eligible"] = True

    summary, _ = simulate_production_symbol(
        signals, SYMBOL, 100_000.0, entry_eligible=True,
        fee_bps=Config.PAPER_FEE_BPS,
        slippage_bps=Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS,
    )
    print(f"  Signals generated: {(signals['model_direction'] == 'LONG').sum()} / {len(signals)} candles")
    print(f"  Real backtest trades: {summary['trades']}")
    print(f"  Return: {summary['return_pct']:.2f}%")
    print(f"  Win rate: {summary['win_rate_pct']:.2f}%")
    print(f"  Profit factor: {summary['profit_factor']:.2f}")

    print("\n=== Verdict ===")
    print(f"  A price-level MAPE of {model_mape:.2f}% looks similar to the paper's reported 8.8% for BTC.")
    print(f"  It beats the naive persistence baseline by only {naive_mape - model_mape:.3f} points.")
    print(f"  Directional accuracy ({model_directional_accuracy:.1f}%) is what actually matters for trading, "
          f"and it's the thing the paper never measured.")
    print(f"  Backtested with real fees/slippage/ATR stops: {summary['return_pct']:.2f}% return, "
          f"profit factor {summary['profit_factor']:.2f}.")


if __name__ == "__main__":
    main()
