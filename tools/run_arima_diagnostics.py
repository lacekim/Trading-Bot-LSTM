"""Classical time-series diagnostics (Box-Jenkins methodology) on real GMX
1h price data: Augmented Dickey-Fuller stationarity test on price level vs.
log returns, then ACF/PACF on returns to check for genuine autocorrelation
structure before fitting anything -- ARIMA/ETS only make sense to fit if
there's actual structure to model; fitting them blind on white noise just
produces an overfit-to-nothing model, same failure mode as everything else
tested tonight.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, pacf

from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc

TIMEFRAME = "1h"
MAX_LAG = 40


def diagnose(symbol: str) -> None:
    df = load_gmx_ohlc(symbol, TIMEFRAME)
    close = df["Close"].astype(float)
    log_returns = np.log(close / close.shift(1)).dropna()

    print(f"\n===== {symbol} ({len(close)} candles) =====")

    adf_price = adfuller(close, autolag="AIC")
    print(f"ADF on price level:   stat={adf_price[0]:.3f}  p-value={adf_price[1]:.4f}  "
          f"{'STATIONARY' if adf_price[1] < 0.05 else 'NON-stationary (expected -- price is a random walk / has a unit root)'}")

    adf_ret = adfuller(log_returns, autolag="AIC")
    print(f"ADF on log returns:   stat={adf_ret[0]:.3f}  p-value={adf_ret[1]:.4f}  "
          f"{'STATIONARY' if adf_ret[1] < 0.05 else 'NON-stationary'}")

    acf_vals, acf_confint = acf(log_returns, nlags=MAX_LAG, alpha=0.05, fft=True)
    pacf_vals, pacf_confint = pacf(log_returns, nlags=MAX_LAG, alpha=0.05)

    def significant_lags(values, confint):
        sig = []
        for lag in range(1, len(values)):
            lo, hi = confint[lag]
            if values[lag] < lo or values[lag] > hi:
                sig.append((lag, float(values[lag])))
        return sig

    acf_sig = significant_lags(acf_vals, acf_confint)
    pacf_sig = significant_lags(pacf_vals, pacf_confint)

    print(f"ACF significant lags (of {MAX_LAG}, 95% band):  {acf_sig if acf_sig else 'NONE -- looks like white noise'}")
    print(f"PACF significant lags (of {MAX_LAG}, 95% band): {pacf_sig if pacf_sig else 'NONE -- looks like white noise'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=["BTC", "ETH", "SOL", "ARB", "LINK"])
    args = parser.parse_args()
    for symbol in args.symbols:
        diagnose(symbol.upper())
