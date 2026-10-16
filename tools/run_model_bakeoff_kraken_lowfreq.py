"""Re-run of tools/run_model_bakeoff_kraken.py at 1h/4h instead of 30s bars
-- same 6 models (ARIMA, LSTM, GRU, LightGBM, Prophet, Prophet+order-flow)
plus the order-flow-augmented ARIMAX/LightGBM variants, same honest
walk-forward one-step-ahead methodology, same real Kraken BTC/USD data,
just resampled to a frequency where a typical predicted move is actually
large relative to trading costs.

Why this exists: the 30s bakeoff (logs/v4_model_bakeoff_kraken*.json) found
real, statistically significant directional edge (56-58% on ARIMA/LightGBM/
Prophet+flow, p<1e-20) -- but RMSE there was ~0.00029, i.e. ~3bps per
30s bar, against a real Kraken round-trip cost of ~24-26bps (see this
project's own ENTRY_COST/EXIT_COST convention: 12bps one-way). A 57%
directional call on a 3bp move cannot survive a 24bp cost. This script
checks whether the SAME edge survives at 1h/4h, where a typical move is
a much larger multiple of the round-trip cost.

Data: real Kraken OHLC feather (BTC_USD-1h.feather, resampled to 4h),
resampled to log-returns at the target timeframe. Order-flow: real
trade-by-trade Kraken data (BTC_USD-trades.feather, confirmed to span
2025-07-04 to 2026-01-05, ~8M real trades -- NOT the 14-day window the
30s version used, since 1h/4h bars need much more history for a
comparable holdout sample size), signed by side, resampled to match.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.models import Sequential

OHLC_PATH = "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/BTC_USD-1h.feather"
TRADES_PATH = "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/BTC_USD-trades.feather"
TRAIN_FRACTION = 0.80
WINDOW = 36

FEE_BPS_ROUND_TRIP = 24  # this project's own established convention (12bps x2)


def load_returns_and_flow(timeframe: str) -> tuple[pd.Series, pd.Series]:
    h1 = pd.read_feather(OHLC_PATH)
    h1["date"] = pd.to_datetime(h1["date"], utc=True)
    h1 = h1.set_index("date")
    bars = h1.resample(timeframe).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    close = bars["close"]
    log_returns = np.log(close / close.shift(1)).dropna()

    trades = pd.read_feather(TRADES_PATH)
    trades["dt"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    trades = trades.set_index("dt")
    trades["signed_amount"] = np.where(trades["side"] == "buy", trades["amount"], -trades["amount"])
    order_flow = trades["signed_amount"].resample(timeframe).sum().reindex(log_returns.index).fillna(0.0)

    # Restrict to the window real trade-level data actually covers -- log
    # returns exist over the full multi-year OHLC history, but order_flow
    # is only real where real trades exist (2025-07-04 to 2026-01-05);
    # outside that window order_flow is fabricated zeros, not real silence.
    trade_start, trade_end = trades.index.min(), trades.index.max()
    in_range = (log_returns.index >= trade_start) & (log_returns.index <= trade_end)
    return log_returns[in_range], order_flow[in_range]


def _directional_score(forecast: np.ndarray, actual: np.ndarray) -> dict:
    mask = (forecast != 0) & (actual != 0)
    f, a = forecast[mask], actual[mask]
    correct = int((np.sign(f) == np.sign(a)).sum())
    total = len(a)
    if total == 0:
        return {"directional_accuracy": float("nan"), "binom_p": float("nan"), "n": 0}
    binom_p = stats.binomtest(correct, total, p=0.5, alternative="greater").pvalue
    return {"directional_accuracy": correct / total, "binom_p": binom_p, "n": total}


def _report(name: str, forecast: np.ndarray, actual: np.ndarray, train_time: float, infer_time: float) -> dict:
    rmse = float(np.sqrt(mean_squared_error(actual, forecast)))
    mae = float(mean_absolute_error(actual, forecast))
    dir_score = _directional_score(forecast, actual)
    typical_move_bps = rmse * 10000
    survives_costs = typical_move_bps > FEE_BPS_ROUND_TRIP
    result = {
        "model": name, "rmse": rmse, "mae": mae,
        "train_time_sec": train_time, "infer_time_sec": infer_time,
        "typical_move_bps": typical_move_bps, "survives_round_trip_cost": survives_costs,
        **dir_score,
    }
    print(f"{name:32s} RMSE={rmse:.6f} ({typical_move_bps:.1f}bps) dir_acc={dir_score['directional_accuracy']*100:.2f}% "
          f"(n={dir_score['n']}) binom_p={dir_score['binom_p']:.2e} "
          f"{'SURVIVES' if survives_costs else 'below'} {FEE_BPS_ROUND_TRIP}bps cost "
          f"train={train_time:.1f}s")
    return result


def run_arima(train: pd.Series, full: pd.Series, train_end: int) -> dict:
    t0 = time.monotonic()
    scale = 10000.0
    model = ARIMA(train * scale, order=(1, 0, 1)).fit()
    train_time = time.monotonic() - t0

    t0 = time.monotonic()
    extended = model.apply(full * scale)
    predictions = extended.get_prediction(start=train_end, end=len(full) - 1, dynamic=False)
    forecast = predictions.predicted_mean.to_numpy() / scale
    infer_time = time.monotonic() - t0

    actual = full.iloc[train_end:].to_numpy()
    return _report("ARIMA(1,0,1)", forecast, actual, train_time, infer_time)


def _make_sequences(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
    return np.array(X)[..., np.newaxis], np.array(y)


def run_rnn(name: str, cell, full: pd.Series, train_end: int) -> dict:
    values = full.to_numpy()
    X_all, y_all = _make_sequences(values, WINDOW)
    split = train_end - WINDOW
    X_train, y_train = X_all[:split], y_all[:split]
    X_hold, y_hold = X_all[split:], y_all[split:]

    t0 = time.monotonic()
    model = Sequential([cell(32, input_shape=(WINDOW, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=256, verbose=0,
              validation_split=0.1, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    train_time = time.monotonic() - t0

    t0 = time.monotonic()
    forecast = model.predict(X_hold, verbose=0).reshape(-1)
    infer_time = time.monotonic() - t0

    return _report(name, forecast, y_hold, train_time, infer_time)


def run_lightgbm(full: pd.Series, train_end: int) -> dict:
    lags = [1, 2, 3, 5, 10, 20]
    values = full.to_numpy()
    X, y = [], []
    for i in range(max(lags), len(values)):
        X.append([values[i - lag] for lag in lags])
        y.append(values[i])
    X, y = np.array(X), np.array(y)
    split = train_end - max(lags)
    X_train, y_train = X[:split], y[:split]
    X_hold, y_hold = X[split:], y[split:]

    t0 = time.monotonic()
    model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, verbosity=-1, random_state=0)
    model.fit(X_train, y_train)
    train_time = time.monotonic() - t0

    t0 = time.monotonic()
    forecast = model.predict(X_hold)
    infer_time = time.monotonic() - t0

    return _report("LightGBM (lag features)", forecast, y_hold, train_time, infer_time)


def run_prophet(train: pd.Series, full: pd.Series, train_end: int, regressor: pd.Series | None = None) -> dict:
    from prophet import Prophet

    train_df = pd.DataFrame({"ds": train.index.tz_localize(None), "y": train.to_numpy()})
    name = "Prophet"
    if regressor is not None:
        train_df["reg"] = regressor.iloc[:train_end].to_numpy()
        name = "Prophet + lagged order-flow"

    t0 = time.monotonic()
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    if regressor is not None:
        m.add_regressor("reg")
    m.fit(train_df)
    train_time = time.monotonic() - t0

    hold_index = full.index[train_end:]
    future = pd.DataFrame({"ds": hold_index.tz_localize(None)})
    if regressor is not None:
        future["reg"] = regressor.iloc[train_end:].to_numpy()

    t0 = time.monotonic()
    forecast_df = m.predict(future)
    infer_time = time.monotonic() - t0

    forecast = forecast_df["yhat"].to_numpy()
    actual = full.iloc[train_end:].to_numpy()
    return _report(name, forecast, actual, train_time, infer_time)


def run_arima_with_flow(full_y: pd.Series, full_exog: pd.Series, train_end: int) -> dict:
    scale = 10000.0
    t0 = time.monotonic()
    model = SARIMAX(full_y.iloc[:train_end] * scale, exog=full_exog.iloc[:train_end], order=(1, 0, 1)).fit(disp=False)
    train_time = time.monotonic() - t0

    t0 = time.monotonic()
    extended = model.apply(full_y * scale, exog=full_exog)
    predictions = extended.get_prediction(start=train_end, end=len(full_y) - 1, dynamic=False)
    forecast = predictions.predicted_mean.to_numpy() / scale
    infer_time = time.monotonic() - t0

    actual = full_y.iloc[train_end:].to_numpy()
    return _report("ARIMAX(1,0,1) + lagged order-flow", forecast, actual, train_time, infer_time)


def run_lightgbm_with_flow(full_y: pd.Series, full_exog: pd.Series, train_end: int) -> dict:
    lags = [1, 2, 3, 5, 10, 20]
    values = full_y.to_numpy()
    flow = full_exog.to_numpy()
    X, y = [], []
    for i in range(max(lags), len(values)):
        row = [values[i - lag] for lag in lags] + [flow[i - lag] for lag in lags]
        X.append(row)
        y.append(values[i])
    X, y = np.array(X), np.array(y)
    split = train_end - max(lags)
    X_train, y_train = X[:split], y[:split]
    X_hold, y_hold = X[split:], y[split:]

    t0 = time.monotonic()
    model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, verbosity=-1, random_state=0)
    model.fit(X_train, y_train)
    train_time = time.monotonic() - t0

    t0 = time.monotonic()
    forecast = model.predict(X_hold)
    infer_time = time.monotonic() - t0

    return _report("LightGBM + lagged order-flow", forecast, y_hold, train_time, infer_time)


def run_all(timeframe: str) -> list[dict]:
    print(f"\n{'='*78}\nTIMEFRAME = {timeframe}\n{'='*78}")
    log_returns, order_flow = load_returns_and_flow(timeframe)
    n = len(log_returns)
    train_end = int(n * TRAIN_FRACTION)
    train = log_returns.iloc[:train_end]
    print(f"n={n} (real trade-data window only), train={train_end}, holdout={n - train_end}")
    print(f"round-trip cost assumption: {FEE_BPS_ROUND_TRIP}bps\n")

    order_flow_lagged = order_flow.shift(1).fillna(0.0)

    results = []
    results.append(run_arima(train, log_returns, train_end))
    results.append(run_rnn("LSTM", LSTM, log_returns, train_end))
    results.append(run_rnn("GRU", GRU, log_returns, train_end))
    results.append(run_lightgbm(log_returns, train_end))
    results.append(run_prophet(train, log_returns, train_end))
    results.append(run_prophet(train, log_returns, train_end, regressor=order_flow_lagged))
    results.append(run_arima_with_flow(log_returns, order_flow_lagged, train_end))
    results.append(run_lightgbm_with_flow(log_returns, order_flow_lagged, train_end))

    for r in results:
        r["timeframe"] = timeframe
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", choices=["1h", "4h", "both"], default="both")
    args = parser.parse_args()

    all_results = []
    timeframes = ["1h", "4h"] if args.timeframe == "both" else [args.timeframe]
    for tf in timeframes:
        all_results.extend(run_all(tf))

    import json
    out = Path("/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/logs/v4_model_bakeoff_kraken_lowfreq.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
