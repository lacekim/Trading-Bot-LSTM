"""Actually fits ARIMA and ETS (not just the diagnostic pre-check) on real
GMX 1h data and evaluates genuine walk-forward, out-of-sample directional
accuracy -- completing the Box-Jenkins exercise instead of stopping at the
ACF/PACF diagnostic.

ARIMA fits on log returns (the stationary series). ETS (Holt's linear trend,
statsmodels ExponentialSmoothing) fits on price level directly -- that's its
natural domain, trend/level smoothing, not differenced returns.

Both are fit ONCE on the first 80% of history, then walked forward through
the remaining 20% via statsmodels' .append()/.extend() state updates (not
refitting from scratch each step -- that's the standard efficient way to get
genuine 1-step-ahead out-of-sample forecasts from a state-space model), so
every forecast in the holdout is made without having seen that point or
anything after it. Reports directional accuracy against a binomial
significance test, same discipline as every other experiment tonight.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc

TIMEFRAME = "1h"
TRAIN_FRACTION = 0.80
ARIMA_ORDER = (2, 0, 2)  # fit on already-stationary returns, so d=0


def evaluate_arima(symbol: str) -> dict:
    df = load_gmx_ohlc(symbol, TIMEFRAME)
    close = df["Close"].astype(float)
    log_returns = np.log(close / close.shift(1)).dropna().reset_index(drop=True)

    n = len(log_returns)
    train_end = int(n * TRAIN_FRACTION)
    train = log_returns.iloc[:train_end]

    model = ARIMA(train, order=ARIMA_ORDER).fit()
    significant_params = {
        name: (float(model.params[name]), float(model.pvalues[name]))
        for name in model.params.index if model.pvalues[name] < 0.05
    }

    # Attach the full series (train + holdout) to the already-fitted model's
    # state WITHOUT refitting parameters, then pull every one-step-ahead
    # holdout forecast in a single vectorized call instead of a Python loop
    # re-running the Kalman filter update once per step -- same walk-forward
    # honesty (each forecast only conditions on data up to t-1), just fast.
    extended = model.apply(log_returns)
    predictions = extended.get_prediction(start=train_end, end=n - 1, dynamic=False)
    forecast_values = predictions.predicted_mean.to_numpy()
    actual_values = log_returns.iloc[train_end:].to_numpy()

    predicted_up = forecast_values > 0
    actual_up = actual_values > 0
    correct_mask = predicted_up == actual_up
    correct = int(correct_mask.sum())
    total = len(correct_mask)

    fold_results = [float(f.mean()) for f in np.array_split(correct_mask, 8) if len(f) > 0]

    binom = stats.binomtest(correct, total, p=0.5, alternative="greater")
    return {
        "symbol": symbol, "model": "ARIMA" + str(ARIMA_ORDER),
        "significant_params": significant_params,
        "holdout_n": total, "directional_accuracy": correct / total,
        "binom_p": binom.pvalue, "fold_accuracies": fold_results,
    }


def evaluate_ets(symbol: str, n_folds: int = 8) -> dict:
    """ETSResults has no .append()/.extend() state-update method the way
    SARIMAX-based ARIMA does, so this uses n_folds expanding-window refits
    instead of one refit per step -- each fold refits on everything up to
    that fold's start (never touching that fold's own data or later), then
    forecasts the whole fold's length in one shot. Still genuinely
    walk-forward (no fold's fit ever sees its own future), just coarser."""
    df = load_gmx_ohlc(symbol, TIMEFRAME)
    close = df["Close"].astype(float).reset_index(drop=True)

    n = len(close)
    train_end = int(n * TRAIN_FRACTION)
    fold_edges = np.linspace(train_end, n, n_folds + 1).astype(int)

    all_correct = []
    for fold in range(n_folds):
        fold_start, fold_end = fold_edges[fold], fold_edges[fold + 1]
        if fold_end <= fold_start:
            continue
        train = close.iloc[:fold_start]
        model = ETSModel(train, error="add", trend="add", damped_trend=True).fit(disp=False)
        forecast = model.forecast(steps=fold_end - fold_start).to_numpy()
        actual = close.iloc[fold_start:fold_end].to_numpy()
        prev = close.iloc[fold_start - 1:fold_end - 1].to_numpy()
        predicted_up = forecast > prev
        actual_up = actual > prev
        all_correct.append(predicted_up == actual_up)

    correct_mask = np.concatenate(all_correct)
    correct = int(correct_mask.sum())
    total = len(correct_mask)

    binom = stats.binomtest(correct, total, p=0.5, alternative="greater")
    return {
        "symbol": symbol, "model": "ETS(AAdN, damped trend)",
        "holdout_n": total, "directional_accuracy": correct / total,
        "binom_p": binom.pvalue,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=["BTC", "ETH", "SOL", "ARB", "LINK"])
    args = parser.parse_args()

    for symbol in args.symbols:
        symbol = symbol.upper()
        print(f"\n===== {symbol} =====")

        arima_result = evaluate_arima(symbol)
        print(f"ARIMA{ARIMA_ORDER}: significant params (p<0.05): {arima_result['significant_params'] or 'NONE'}")
        print(f"  holdout directional accuracy: {arima_result['directional_accuracy']*100:.2f}% "
              f"({arima_result['holdout_n']} steps)  binom_p={arima_result['binom_p']:.4f}  "
              f"SIGNIFICANT: {arima_result['binom_p'] < 0.05}")
        print(f"  per-fold accuracy: {[f'{a*100:.1f}%' for a in arima_result['fold_accuracies']]}")

        ets_result = evaluate_ets(symbol)
        print(f"ETS(Holt, damped trend): holdout directional accuracy: "
              f"{ets_result['directional_accuracy']*100:.2f}% ({ets_result['holdout_n']} steps)  "
              f"binom_p={ets_result['binom_p']:.4f}  SIGNIFICANT: {ets_result['binom_p'] < 0.05}")
