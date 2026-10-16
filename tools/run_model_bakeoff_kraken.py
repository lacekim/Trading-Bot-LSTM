"""Individual, standalone test of each of the 6 models from the Jena Climate
article (ARIMA, LSTM, GRU, LightGBM, Prophet, Prophet+regressors) on real
Kraken BTC/USD data, each evaluated separately and honestly the way the
article did -- same walk-forward, one-step-ahead methodology (feeding true
recent values, not compounding predictions), same metrics (RMSE, MAE,
training/inference time), plus directional accuracy + binomial significance
since that's what actually matters for a trading signal.

Target is log returns, not raw price level -- price level is a proven random
walk (verified via ADF earlier tonight), so forecasting it directly would let
every model "cheat" via trivial persistence, the exact trap the article's own
author flagged for LSTM/GRU on temperature. Returns are the correct,
stationary target.

Data: same real, liquid Kraken 30s BTC/USD bars already validated (0.04%
lookaheadmpty bars, real trade-derived OHLC) -- not Binance.US, which was
too thin to trust.
"""
from __future__ import annotations

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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.models import Sequential

DATA_PATH = "/private/tmp/claude-501/-Volumes-Extreme-SSD-Javier-Santiago-Gaston-de-Iriarte-canrera/b7bd7210-c6b4-4186-a385-1c69707c40ed/scratchpad/kraken_btc_30s.csv"
TRADES_PATH = "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/BTC_USD-trades.feather"
TRAIN_FRACTION = 0.80
WINDOW = 36  # matches SEQUENCE_LENGTH used elsewhere tonight


def load_returns_and_flow() -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(DATA_PATH, index_col=0)
    close = df["close"].ffill()
    log_returns = np.log(close / close.shift(1)).dropna()
    log_returns.index = pd.to_datetime(log_returns.index, utc=True)

    trades = pd.read_feather(TRADES_PATH)
    trades["dt"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    recent = trades[trades["dt"] >= trades["dt"].max() - pd.Timedelta(days=14)].copy()
    recent = recent.set_index("dt")
    recent["signed_amount"] = np.where(recent["side"] == "buy", recent["amount"], -recent["amount"])
    order_flow = recent["signed_amount"].resample("30s").sum().reindex(log_returns.index).fillna(0.0)

    return log_returns, order_flow


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
    result = {
        "model": name, "rmse": rmse, "mae": mae,
        "train_time_sec": train_time, "infer_time_sec": infer_time,
        **dir_score,
    }
    print(f"{name:25s} RMSE={rmse:.6f} MAE={mae:.6f} dir_acc={dir_score['directional_accuracy']*100:.2f}% "
          f"(n={dir_score['n']}) binom_p={dir_score['binom_p']:.4f} "
          f"train={train_time:.1f}s infer={infer_time:.2f}s")
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


def run_rnn(name: str, cell, train: pd.Series, full: pd.Series, train_end: int) -> dict:
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

    # Rolling one-step-ahead: each window uses the TRUE past values (same
    # methodology the article used for LSTM/GRU), fully vectorizable since
    # there's no recursive state to update -- just batch-predict every
    # holdout window at once.
    t0 = time.monotonic()
    forecast = model.predict(X_hold, verbose=0).reshape(-1)
    infer_time = time.monotonic() - t0

    return _report(name, forecast, y_hold, train_time, infer_time)


def run_lightgbm(train: pd.Series, full: pd.Series, train_end: int) -> dict:
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

    # regressor, if given, must already be LAGGED (shifted forward) by the
    # caller -- using bar t's own order flow to predict bar t's own return
    # isn't a valid forecast, since flow and return within the same bar are
    # mechanically linked (measured correlation 0.31) and flow isn't known
    # until the bar has already closed, same moment the return is known.
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


def run_arima_with_flow(train_y: pd.Series, full_y: pd.Series, train_exog: pd.Series, full_exog: pd.Series, train_end: int) -> dict:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    scale = 10000.0
    t0 = time.monotonic()
    model = SARIMAX(train_y * scale, exog=train_exog.iloc[:train_end], order=(1, 0, 1)).fit(disp=False)
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


if __name__ == "__main__":
    print("Loading real Kraken 30s BTC/USD data (14 days, liquid, verified)...")
    log_returns, order_flow = load_returns_and_flow()
    n = len(log_returns)
    train_end = int(n * TRAIN_FRACTION)
    train = log_returns.iloc[:train_end]
    print(f"n={n}, train={train_end}, holdout={n - train_end}\n")

    # Shifted forward by 1 bar: order_flow_lagged[t] = order_flow[t-1], so
    # every model below only ever sees flow from BEFORE the bar it's
    # predicting -- the honest, causal version of the earlier (flawed) test.
    order_flow_lagged = order_flow.shift(1).fillna(0.0)

    print("=== Corrected: lagged order-flow added to ARIMA, LightGBM, Prophet ===")
    results_flow = []
    results_flow.append(run_arima_with_flow(train, log_returns, order_flow_lagged, order_flow_lagged, train_end))
    results_flow.append(run_lightgbm_with_flow(log_returns, order_flow_lagged, train_end))
    results_flow.append(run_prophet(train, log_returns, train_end, regressor=order_flow_lagged))

    import json
    out_flow = Path("logs/v4_model_bakeoff_kraken_with_flow.json")
    out_flow.write_text(json.dumps(results_flow, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_flow}")
    import sys
    sys.exit(0)

    results = []
    results.append(run_arima(train, log_returns, train_end))
    results.append(run_rnn("LSTM", LSTM, train, log_returns, train_end))
    results.append(run_rnn("GRU", GRU, train, log_returns, train_end))
    results.append(run_lightgbm(train, log_returns, train_end))
    results.append(run_prophet(train, log_returns, train_end))
    results.append(run_prophet(train, log_returns, train_end, regressor=order_flow))

    import json
    out = Path("logs/v4_model_bakeoff_kraken.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
