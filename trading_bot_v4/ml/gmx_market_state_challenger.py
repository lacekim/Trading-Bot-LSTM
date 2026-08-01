"""Leakage-controlled GMX liquidity/order-flow LONG/SHORT challenger research."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from trading_bot import load_gmx_ohlc
from trading_bot_v4.features.gmx_liquidity_history import LIQUIDITY_HISTORY_PATH
from trading_bot_v4.features.gmx_trade_flow import TRADE_FLOW_PATH
from trading_bot_v4.ml.cost_aware_long_trainer import FEATURES, _load
from trading_bot_v4.ml.multi_horizon_directional import add_horizon_return, portfolio_metrics
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler


REPORT_PATH = Path("reports/v5_gmx_market_state_walk_forward.json")
HORIZONS = (4, 8)
FOLDS = ((.40, .50, .65), (.55, .65, .80), (.70, .80, 1.00))
MIN_CALIBRATION_TRADES = 30
MIN_TEST_TRADES = 20


def add_trailing_outcomes(data: pd.DataFrame, horizon: int, timeframe: str = "1h") -> pd.DataFrame:
    """Initial 2ATR stop, +1ATR breakeven, 2ATR trailing stop, max hold."""
    frames = []
    for symbol in sorted(data["symbol"].unique()):
        try:
            raw = load_gmx_ohlc(symbol, timeframe)
        except Exception:
            continue
        close = raw["Close"].astype(float)
        atr = V4DataHandler().calculate_atr(raw, Config.ATR_PERIOD).astype(float)
        long_exit = close.shift(-horizon).copy()
        short_exit = close.shift(-horizon).copy()
        long_stop = close - 2 * atr; short_stop = close + 2 * atr
        long_active = pd.Series(True, index=raw.index); short_active = pd.Series(True, index=raw.index)
        long_peak = close.copy(); short_trough = close.copy()
        for step in range(1, horizon + 1):
            high, low = raw["High"].shift(-step), raw["Low"].shift(-step)
            hit_long = long_active & low.le(long_stop)
            hit_short = short_active & high.ge(short_stop)
            long_exit.loc[hit_long] = long_stop.loc[hit_long]
            short_exit.loc[hit_short] = short_stop.loc[hit_short]
            long_active &= ~hit_long; short_active &= ~hit_short
            # Current-bar extremes only adjust protection for the next bar.
            long_peak = pd.concat([long_peak, high], axis=1).max(axis=1)
            short_trough = pd.concat([short_trough, low], axis=1).min(axis=1)
            long_breakeven = long_peak.ge(close + atr)
            short_breakeven = short_trough.le(close - atr)
            long_candidate = pd.concat([long_stop, long_peak - 2 * atr,
                                        close.where(long_breakeven)], axis=1).max(axis=1)
            short_candidate = pd.concat([short_stop, short_trough + 2 * atr,
                                         close.where(short_breakeven)], axis=1).min(axis=1)
            long_stop = long_stop.where(~long_active, long_candidate)
            short_stop = short_stop.where(~short_active, short_candidate)
        frames.append(pd.DataFrame({
            "timestamp": raw.index, "symbol": symbol,
            "long_protected_return": long_exit / close - 1,
            "short_protected_return": 1 - short_exit / close,
        }))
    outcomes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return data.merge(outcomes, on=["timestamp", "symbol"], how="left").dropna(
        subset=["long_protected_return", "short_protected_return"]
    )


def _external_features() -> pd.DataFrame:
    liquidity = pd.read_csv(LIQUIDITY_HISTORY_PATH)
    liquidity["timestamp"] = pd.to_datetime(liquidity["timestamp"], utc=True, errors="coerce", format="mixed")
    liquidity["symbol"] = liquidity["symbol"].astype(str).str.upper()
    liquidity = liquidity.sort_values(["symbol", "timestamp"])
    generated = []
    for symbol, group in liquidity.groupby("symbol", sort=False):
        group = group.copy()
        # Shift one complete hour: no snapshot from the decision hour can leak
        # into a decision made immediately after its candle closes.
        for column in ("jit_liquidity_total_usd", "jit_liquidity_skew"):
            group[f"{column}_lag1"] = group[column].shift(1)
        group["jit_liquidity_log_lag1"] = np.log1p(group["jit_liquidity_total_usd_lag1"].clip(lower=0))
        for lag in (1, 6, 24):
            group[f"jit_liquidity_change_{lag}h"] = group["jit_liquidity_total_usd"].pct_change(lag).shift(1)
            group[f"jit_skew_change_{lag}h"] = group["jit_liquidity_skew"].diff(lag).shift(1)
        generated.append(group)
    result = pd.concat(generated, ignore_index=True)
    if TRADE_FLOW_PATH.exists():
        flow = pd.read_csv(TRADE_FLOW_PATH)
        flow["timestamp"] = pd.to_datetime(flow["timestamp"], utc=True, errors="coerce", format="mixed")
        flow["symbol"] = flow["symbol"].astype(str).str.upper()
        flow_columns = [column for column in ("increase_flow_skew", "net_position_flow_usd") if column in flow]
        for column in flow_columns:
            flow[column] = flow.groupby("symbol")[column].shift(1)
        result = result.merge(flow[["timestamp", "symbol", *flow_columns]], on=["timestamp", "symbol"], how="left")
    return result.replace([np.inf, -np.inf], np.nan)


def build_dataset(timeframe: str = "1h", horizon: int = 8) -> tuple[pd.DataFrame, list[str]]:
    price = _load(timeframe)
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True, errors="coerce")
    data = price.merge(_external_features(), on=["timestamp", "symbol"], how="inner")
    btc = load_gmx_ohlc("BTC", timeframe)[["Close"]].copy()
    btc.index = pd.to_datetime(btc.index, utc=True)
    btc = btc.sort_index()
    btc["btc_return_24h"] = btc["Close"].pct_change(24)
    btc["btc_return_72h"] = btc["Close"].pct_change(72)
    btc["btc_price_vs_ma200"] = btc["Close"] / btc["Close"].rolling(200).mean() - 1.0
    btc["btc_volatility_24h"] = btc["Close"].pct_change().rolling(24).std()
    # Completed BTC candle features are known at the same hourly decision.
    btc = btc.drop(columns="Close").rename_axis("timestamp").reset_index()
    data = data.merge(btc, on="timestamp", how="inner")
    # Existing candle files use timezone-naive UTC indices; normalize only
    # after the external as-of merge so protective-outcome joins are exact.
    data["timestamp"] = data["timestamp"].dt.tz_localize(None)
    data = add_trailing_outcomes(add_horizon_return(data, horizon), horizon, timeframe)
    market_features = [
        "jit_liquidity_log_lag1", "jit_liquidity_skew_lag1",
        "jit_liquidity_change_1h", "jit_liquidity_change_6h", "jit_liquidity_change_24h",
        "jit_skew_change_1h", "jit_skew_change_6h", "jit_skew_change_24h",
        "btc_return_24h", "btc_return_72h", "btc_price_vs_ma200", "btc_volatility_24h",
    ]
    # A prospective flow collector must not collapse a historical experiment
    # to its first day. Admit a source only after it covers most joined rows.
    market_features += [
        column for column in ("increase_flow_skew", "net_position_flow_usd")
        if column in data and data[column].notna().mean() >= .50
    ]
    columns = [*FEATURES, *market_features]
    data[columns] = data[columns].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=[*FEATURES, *market_features]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return data, columns


def _chronological_split(data: pd.DataFrame, boundaries: tuple[float, float, float]):
    times = np.array(sorted(data["timestamp"].unique()))
    first, second, third = (times[min(int(len(times) * point), len(times) - 1)] for point in boundaries)
    return (
        data[data["timestamp"] < first].copy(),
        data[(data["timestamp"] >= first) & (data["timestamp"] < second)].copy(),
        data[(data["timestamp"] >= second) & (data["timestamp"] < third)].copy(),
    )


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=.04, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=2.0, early_stopping=True,
        validation_fraction=.10, random_state=42,
    )


def _score(frame: pd.DataFrame, long_prediction: np.ndarray, short_prediction: np.ndarray) -> np.ndarray:
    del frame  # Kept in the signature for future causal portfolio-level gates.
    return np.where(long_prediction >= short_prediction, long_prediction, -short_prediction)


def _threshold(frame: pd.DataFrame, score: np.ndarray, horizon: int, side: str) -> float:
    values = score[score > 0] if side == "long" else -score[score < 0]
    if not len(values):
        return float("inf")
    candidates = []
    for threshold in np.unique(np.quantile(values, np.arange(.70, .991, .02))):
        pair = (float(threshold), float("inf")) if side == "long" else (float("inf"), float(threshold))
        metrics = portfolio_metrics(frame, score, pair, horizon)
        if metrics["trades"] >= MIN_CALIBRATION_TRADES and metrics["return_pct"] > 0 and metrics["profit_factor"] >= 1.30:
            candidates.append((float(threshold), metrics))
    return max(candidates, key=lambda item: (item[1]["profit_factor"], item[1]["return_pct"]))[0] if candidates else float("inf")


def run_gmx_market_state_walk_forward(timeframe: str = "1h") -> dict:
    results = []
    for horizon in HORIZONS:
        data, columns = build_dataset(timeframe, horizon)
        for fold, boundaries in enumerate(FOLDS, 1):
            train, calibration, test = _chronological_split(data, boundaries)
            long_model = _model().fit(train[columns].to_numpy(np.float32), train["long_protected_return"])
            short_model = _model().fit(train[columns].to_numpy(np.float32), train["short_protected_return"])
            calibration_score = _score(calibration,
                long_model.predict(calibration[columns].to_numpy(np.float32)),
                short_model.predict(calibration[columns].to_numpy(np.float32)),
            )
            thresholds = (
                _threshold(calibration, calibration_score, horizon, "long"),
                _threshold(calibration, calibration_score, horizon, "short"),
            )
            test_score = _score(test,
                long_model.predict(test[columns].to_numpy(np.float32)),
                short_model.predict(test[columns].to_numpy(np.float32)),
            )
            results.append({
                "horizon": horizon, "fold": fold,
                "train_range": [str(train.timestamp.min()), str(train.timestamp.max())],
                "calibration_range": [str(calibration.timestamp.min()), str(calibration.timestamp.max())],
                "test_range": [str(test.timestamp.min()), str(test.timestamp.max())],
                "long_threshold": thresholds[0], "short_threshold": thresholds[1],
                "calibration": portfolio_metrics(calibration, calibration_score, thresholds, horizon),
                "test": portfolio_metrics(test, test_score, thresholds, horizon),
            })
    qualified = []
    for horizon in HORIZONS:
        rows = [row for row in results if row["horizon"] == horizon]
        if all(row["test"]["trades"] >= MIN_TEST_TRADES and row["test"]["return_pct"] > 0
               and row["test"]["profit_factor"] >= 1.30 for row in rows):
            qualified.append(horizon)
    report = {
        "promoted": bool(qualified), "qualified_horizons": qualified,
        "data_policy": "GMX hourly directional liquidity lagged one complete hour plus completed-candle BTC 200MA/return regime; global chronological non-overlapping tests",
        "execution_policy": "shared capital, after configured fees/slippage/impact, initial 2ATR stop, +1ATR breakeven, causal 2ATR trail, max hold",
        "promotion_policy": "every unseen fold positive, PF>=1.30, and >=20 trades",
        "results": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_gmx_market_state_walk_forward()
