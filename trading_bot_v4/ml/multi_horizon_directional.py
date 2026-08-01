"""Cost-aware hourly-entry LONG/SHORT research with multi-hour holds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot import load_gmx_ohlc
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.ml.cost_aware_long_trainer import FEATURES, _fractional_split, _load, round_trip_cost_rate


REPORT_PATH = Path("reports/v5_multi_horizon_directional_walk_forward.json")
SHORT_CLASSIFIER_REPORT_PATH = Path("reports/v5_8h_short_classifier_walk_forward.json")
ASSET_GATED_REPORT_PATH = Path("reports/v5_8h_asset_gated_walk_forward.json")
VOL_NORMALIZED_REPORT_PATH = Path("reports/v5_8h_vol_normalized_walk_forward.json")
AGREEMENT_REPORT_PATH = Path("reports/v5_8h_9h_agreement_walk_forward.json")
PROTECTED_TARGET_REPORT_PATH = Path("reports/v5_8h_protected_target_walk_forward.json")
PROTECTED_CLASSIFIER_REPORT_PATH = Path("reports/v5_8h_protected_classifier_walk_forward.json")
HORIZONS = (4, 8, 12)
FOLDS = ((0.55, 0.70, 0.80), (0.65, 0.80, 0.90), (0.75, 0.90, 1.00))
REGRESSION_LOSS = "absolute_error"
REGRESSION_QUANTILE = None
CHALLENGER_STOP_ATR = 2.0
CHALLENGER_TARGET_ATR = 3.0


def add_horizon_return(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    result = data.copy()
    compounded = pd.Series(1.0, index=result.index)
    grouped = result.groupby("symbol", sort=False)["future_return"]
    for step in range(horizon):
        compounded *= 1.0 + grouped.shift(-step)
    result["horizon_return"] = compounded - 1.0
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["horizon_return"])


def add_protective_outcomes(data: pd.DataFrame, horizon: int, timeframe: str = "1h") -> pd.DataFrame:
    frames = []
    for symbol in sorted(data["symbol"].unique()):
        try:
            raw = load_gmx_ohlc(symbol, timeframe)
        except Exception:
            continue
        close = raw["Close"].astype(float)
        atr = V4DataHandler().calculate_atr(raw, Config.ATR_PERIOD)
        stop_pct = atr / close * CHALLENGER_STOP_ATR
        target_pct = atr / close * CHALLENGER_TARGET_ATR
        long_result = close.shift(-horizon) / close - 1.0
        short_result = 1.0 - close.shift(-horizon) / close
        long_open = pd.Series(True, index=raw.index); short_open = pd.Series(True, index=raw.index)
        for step in range(1, horizon + 1):
            high, low = raw["High"].shift(-step), raw["Low"].shift(-step)
            long_stop = long_open & low.le(close * (1.0 - stop_pct))
            long_target = long_open & ~long_stop & high.ge(close * (1.0 + target_pct))
            long_result.loc[long_stop] = -stop_pct.loc[long_stop]
            long_result.loc[long_target] = target_pct.loc[long_target]
            long_open &= ~(long_stop | long_target)
            short_stop = short_open & high.ge(close * (1.0 + stop_pct))
            short_target = short_open & ~short_stop & low.le(close * (1.0 - target_pct))
            short_result.loc[short_stop] = -stop_pct.loc[short_stop]
            short_result.loc[short_target] = target_pct.loc[short_target]
            short_open &= ~(short_stop | short_target)
        frames.append(pd.DataFrame({
            "symbol": symbol, "timestamp": raw.index,
            "long_protected_return": long_result, "short_protected_return": short_result,
        }))
    outcomes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = data.merge(outcomes, on=["symbol", "timestamp"], how="left")
    return result.dropna(subset=["long_protected_return", "short_protected_return"])


def portfolio_metrics(frame: pd.DataFrame, prediction: np.ndarray, threshold: float | tuple[float, float],
                      horizon: int) -> dict:
    long_threshold, short_threshold = threshold if isinstance(threshold, tuple) else (threshold, threshold)
    direction = np.where(prediction > long_threshold, 1.0, np.where(prediction < -short_threshold, -1.0, 0.0))
    selected = direction != 0
    columns = ["timestamp", "symbol", "horizon_return", "atr_norm"]
    columns += [column for column in ("long_protected_return", "short_protected_return") if column in frame.columns]
    trades = frame.loc[selected, columns].copy()
    trades["direction"], trades["strength"] = direction[selected], np.abs(prediction[selected])
    blacklist = {str(symbol).upper() for symbol in Config.GMX_SYMBOL_BLACKLIST}
    invalid = trades["symbol"].isin(blacklist) | trades["symbol"].str.contains("DEPRECATED", regex=False) | trades["symbol"].str.startswith("GLV ")
    trades = trades.loc[~invalid].copy()
    if {"long_protected_return", "short_protected_return"}.issubset(trades.columns):
        directional_return = np.where(trades["direction"] > 0, trades["long_protected_return"], trades["short_protected_return"])
        trades["trade_return"] = directional_return - round_trip_cost_rate()
    else:
        trades["trade_return"] = trades["horizon_return"] * trades["direction"] - round_trip_cost_rate()
    capital, open_positions, pnls = 1.0, [], []
    long_count = short_count = 0
    for timestamp, candidates in trades.groupby("timestamp", sort=True):
        still_open = []
        for position in open_positions:
            if position[0] <= timestamp:
                capital += position[1]; pnls.append(position[1])
            else:
                still_open.append(position)
        open_positions = still_open
        exposure = sum(position[2] for position in open_positions)
        capacity = max(0, Config.PAPER_MAX_OPEN_POSITIONS - len(open_positions))
        for _, trade in candidates.sort_values("strength", ascending=False).head(capacity).iterrows():
            remaining = Config.PAPER_MAX_PORTFOLIO_EXPOSURE_PCT / 100.0 - exposure
            stop_pct = float(trade["atr_norm"]) * CHALLENGER_STOP_ATR
            desired = min((Config.RISK_PERCENTAGE / 100.0) / max(stop_pct, 1e-12), Config.PAPER_MAX_POSITION_PCT / 100.0)
            allocation = min(desired, remaining)
            if allocation <= 0:
                continue
            pnl = capital * allocation * float(trade["trade_return"])
            open_positions.append((timestamp + pd.Timedelta(hours=horizon), pnl, allocation))
            exposure += allocation
            long_count += int(trade["direction"] > 0); short_count += int(trade["direction"] < 0)
    for _, pnl, _ in open_positions:
        capital += pnl; pnls.append(pnl)
    values = pd.Series(pnls, dtype=float); wins, losses = values[values > 0], values[values < 0]
    return {
        "trades": len(values), "long_trades": long_count, "short_trades": short_count,
        "return_pct": (capital - 1.0) * 100.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else 0.0),
        "win_rate_pct": float((values > 0).mean() * 100.0) if len(values) else 0.0,
    }


def _side_threshold(frame: pd.DataFrame, prediction: np.ndarray, horizon: int, side: str) -> float:
    side_values = prediction[prediction > 0] if side == "long" else -prediction[prediction < 0]
    if not len(side_values):
        return float("inf")
    candidates = []; sampled = []
    for threshold in np.unique(np.quantile(side_values, np.arange(0.70, 0.991, 0.02))):
        pair = (float(threshold), float("inf")) if side == "long" else (float("inf"), float(threshold))
        metrics = portfolio_metrics(frame, prediction, pair, horizon)
        if metrics["trades"] >= 50:
            sampled.append((float(threshold), metrics))
            if metrics["return_pct"] > 0 and metrics["profit_factor"] >= 1.20:
                candidates.append((float(threshold), metrics))
    pool = candidates or sampled
    return max(pool, key=lambda item: (item[1]["profit_factor"], item[1]["return_pct"]))[0] if pool else float("inf")


def choose_threshold(frame: pd.DataFrame, prediction: np.ndarray, horizon: int) -> tuple[tuple[float, float], dict]:
    pair = (_side_threshold(frame, prediction, horizon, "long"), _side_threshold(frame, prediction, horizon, "short"))
    return pair, portfolio_metrics(frame, prediction, pair, horizon)


def run_multi_horizon_walk_forward(timeframe: str = "1h") -> dict:
    source = _load(timeframe); results = []
    for horizon in HORIZONS:
        data = add_protective_outcomes(add_horizon_return(source, horizon), horizon, timeframe)
        for fold, boundaries in enumerate(FOLDS, start=1):
            train, calibration, test = _fractional_split(data, *boundaries)
            model = HistGradientBoostingRegressor(
                loss=REGRESSION_LOSS, quantile=REGRESSION_QUANTILE,
                learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
                min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                validation_fraction=0.10, random_state=42,
            ).fit(train[FEATURES].to_numpy(np.float32), train["horizon_return"].to_numpy(float))
            cp = model.predict(calibration[FEATURES].to_numpy(np.float32))
            threshold, calibration_metrics = choose_threshold(calibration, cp, horizon)
            tp = model.predict(test[FEATURES].to_numpy(np.float32))
            results.append({"horizon": horizon, "fold": fold,
                            "long_threshold": threshold[0], "short_threshold": threshold[1],
                            "calibration": calibration_metrics, "test": portfolio_metrics(test, tp, threshold, horizon)})
    qualified = []
    for horizon in HORIZONS:
        rows = [row for row in results if row["horizon"] == horizon]
        if all(row["test"]["trades"] >= 50 and row["test"]["return_pct"] > 0 and row["test"]["profit_factor"] >= 1.20 for row in rows):
            qualified.append(horizon)
    report = {
        "promoted": bool(qualified), "qualified_horizons": qualified,
        "stop_atr": CHALLENGER_STOP_ATR, "target_atr": CHALLENGER_TARGET_ATR,
        "position_policy": "shared capital; at most 5 positions; 95% portfolio exposure; 1% ATR risk sizing",
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report


def run_8h_short_classifier_walk_forward(timeframe: str = "1h") -> dict:
    data = add_horizon_return(_load(timeframe), 8)
    data["short_target"] = data["horizon_return"].lt(-round_trip_cost_rate()).astype(int)
    results = []
    for fold, boundaries in enumerate(FOLDS, start=1):
        train, calibration, test = _fractional_split(data, *boundaries)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=31, min_samples_leaf=100,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.10, random_state=42,
        ).fit(train[FEATURES].to_numpy(np.float32), train["short_target"].to_numpy(int))
        cp = model.predict_proba(calibration[FEATURES].to_numpy(np.float32))[:, 1]
        threshold = _side_threshold(calibration, -cp, 8, "short")
        calibration_metrics = portfolio_metrics(calibration, -cp, (float("inf"), threshold), 8)
        tp = model.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]
        test_metrics = portfolio_metrics(test, -tp, (float("inf"), threshold), 8)
        results.append({"fold": fold, "threshold": threshold, "calibration": calibration_metrics, "test": test_metrics})
    stable = all(row["test"]["trades"] >= 50 and row["test"]["return_pct"] > 0 and row["test"]["profit_factor"] >= 1.20 for row in results)
    report = {"stable": stable, "promoted": False, "horizon": 8, "results": results}
    SHORT_CLASSIFIER_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report


def run_8h_asset_gated_walk_forward(timeframe: str = "1h") -> dict:
    """Select symbols on calibration only, then freeze them for unseen test."""
    data = add_horizon_return(_load(timeframe), 8); results = []
    for fold, boundaries in enumerate(FOLDS, start=1):
        train, calibration, test = _fractional_split(data, *boundaries)
        model = HistGradientBoostingRegressor(
            loss=REGRESSION_LOSS, learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
            min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.10, random_state=42,
        ).fit(train[FEATURES].to_numpy(np.float32), train["horizon_return"].to_numpy(float))
        cp = model.predict(calibration[FEATURES].to_numpy(np.float32))
        short_threshold = _side_threshold(calibration, cp, 8, "short")
        qualified = []
        for symbol, indices in calibration.groupby("symbol").groups.items():
            positions = calibration.index.get_indexer(indices)
            metrics = portfolio_metrics(calibration.loc[indices], cp[positions], (float("inf"), short_threshold), 8)
            if metrics["trades"] >= 10 and metrics["return_pct"] > 0 and metrics["profit_factor"] >= 1.30:
                qualified.append(symbol)
        calibration_mask = calibration["symbol"].isin(qualified).to_numpy()
        test_mask = test["symbol"].isin(qualified).to_numpy()
        calibration_metrics = portfolio_metrics(calibration.loc[calibration_mask], cp[calibration_mask], (float("inf"), short_threshold), 8)
        tp = model.predict(test[FEATURES].to_numpy(np.float32))
        test_metrics = portfolio_metrics(test.loc[test_mask], tp[test_mask], (float("inf"), short_threshold), 8)
        results.append({"fold": fold, "short_threshold": short_threshold, "qualified_symbols": qualified,
                        "calibration": calibration_metrics, "test": test_metrics})
    stable = all(row["test"]["trades"] >= 50 and row["test"]["return_pct"] > 0 and row["test"]["profit_factor"] >= 1.20 for row in results)
    report = {"stable": stable, "promoted": False, "horizon": 8, "results": results}
    ASSET_GATED_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report


def run_8h_vol_normalized_walk_forward(timeframe: str = "1h") -> dict:
    data = add_horizon_return(_load(timeframe), 8); results = []
    for fold, boundaries in enumerate(FOLDS, start=1):
        train, calibration, test = _fractional_split(data, *boundaries)
        model = HistGradientBoostingRegressor(
            loss=REGRESSION_LOSS, learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
            min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.10, random_state=42,
        ).fit(train[FEATURES].to_numpy(np.float32), train["horizon_return"].to_numpy(float))
        raw_calibration = model.predict(calibration[FEATURES].to_numpy(np.float32))
        calibration_score = raw_calibration / calibration["atr_norm"].clip(lower=1e-6).to_numpy(float)
        threshold = _side_threshold(calibration, calibration_score, 8, "short")
        raw_test = model.predict(test[FEATURES].to_numpy(np.float32))
        test_score = raw_test / test["atr_norm"].clip(lower=1e-6).to_numpy(float)
        results.append({
            "fold": fold, "short_threshold_atr_units": threshold,
            "calibration": portfolio_metrics(calibration, calibration_score, (float("inf"), threshold), 8),
            "test": portfolio_metrics(test, test_score, (float("inf"), threshold), 8),
        })
    stable = all(row["test"]["trades"] >= 50 and row["test"]["return_pct"] > 0 and row["test"]["profit_factor"] >= 1.20 for row in results)
    report = {"stable": stable, "promoted": False, "horizon": 8, "score": "predicted_return / atr_norm", "results": results}
    VOL_NORMALIZED_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report


def run_8h_9h_agreement_walk_forward(timeframe: str = "1h") -> dict:
    base = _load(timeframe)
    eight = add_horizon_return(base, 8).rename(columns={"horizon_return": "return_8h"})
    nine = add_horizon_return(base, 9)[["symbol", "timestamp", "horizon_return"]].rename(columns={"horizon_return": "return_9h"})
    data = eight.merge(nine, on=["symbol", "timestamp"], how="inner")
    data["horizon_return"] = data["return_8h"]
    results = []
    for fold, boundaries in enumerate(FOLDS, start=1):
        train, calibration, test = _fractional_split(data, *boundaries)
        predictions = {}
        for label in ("return_8h", "return_9h"):
            model = HistGradientBoostingRegressor(
                loss=REGRESSION_LOSS, learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
                min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                validation_fraction=0.10, random_state=42,
            ).fit(train[FEATURES].to_numpy(np.float32), train[label].to_numpy(float))
            predictions[label] = (
                model.predict(calibration[FEATURES].to_numpy(np.float32)),
                model.predict(test[FEATURES].to_numpy(np.float32)),
            )
        c8, t8 = predictions["return_8h"]; c9, t9 = predictions["return_9h"]
        threshold8 = _side_threshold(calibration, c8, 8, "short")
        threshold9 = _side_threshold(calibration, c9, 8, "short")
        calibration_score = np.where((c8 < -threshold8) & (c9 < -threshold9), -(np.abs(c8) + np.abs(c9)) / 2, 0.0)
        test_score = np.where((t8 < -threshold8) & (t9 < -threshold9), -(np.abs(t8) + np.abs(t9)) / 2, 0.0)
        results.append({
            "fold": fold, "threshold_8h": threshold8, "threshold_9h": threshold9,
            "calibration": portfolio_metrics(calibration, calibration_score, (float("inf"), 0.0), 8),
            "test": portfolio_metrics(test, test_score, (float("inf"), 0.0), 8),
        })
    stable = all(row["test"]["trades"] >= 50 and row["test"]["return_pct"] > 0 and row["test"]["profit_factor"] >= 1.20 for row in results)
    report = {"stable": stable, "promoted": False, "policy": "8h and 9h SHORT agreement; 8h exit", "results": results}
    AGREEMENT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report


def run_8h_protected_target_walk_forward(timeframe: str = "1h") -> dict:
    """Train directly on stop/target-aware directional payoffs."""
    data = add_protective_outcomes(add_horizon_return(_load(timeframe), 8), 8, timeframe)
    results = []
    for fold, boundaries in enumerate(FOLDS, start=1):
        train, calibration, test = _fractional_split(data, *boundaries)
        predictions = {}
        for target in ("long_protected_return", "short_protected_return"):
            model = HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
                min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                validation_fraction=0.10, random_state=42,
            ).fit(train[FEATURES].to_numpy(np.float32), train[target].to_numpy(float))
            predictions[target] = (
                model.predict(calibration[FEATURES].to_numpy(np.float32)),
                model.predict(test[FEATURES].to_numpy(np.float32)),
            )
        cl, tl = predictions["long_protected_return"]
        cs, ts = predictions["short_protected_return"]
        long_threshold = _side_threshold(calibration, cl, 8, "long")
        short_threshold = _side_threshold(calibration, -cs, 8, "short")

        def combined(long_prediction, short_prediction):
            long_ok, short_ok = long_prediction > long_threshold, short_prediction > short_threshold
            return np.where(long_ok & (~short_ok | (long_prediction >= short_prediction)), long_prediction,
                            np.where(short_ok, -short_prediction, 0.0))

        calibration_score, test_score = combined(cl, cs), combined(tl, ts)
        results.append({
            "fold": fold, "long_threshold": long_threshold, "short_threshold": short_threshold,
            "calibration": portfolio_metrics(calibration, calibration_score, (0.0, 0.0), 8),
            "test": portfolio_metrics(test, test_score, (0.0, 0.0), 8),
        })
    stable = all(row["test"]["trades"] >= 50 and row["test"]["return_pct"] > 0 and row["test"]["profit_factor"] >= 1.20 for row in results)
    report = {
        "stable": stable, "promoted": False, "horizon": 8,
        "stop_atr": CHALLENGER_STOP_ATR, "target_atr": CHALLENGER_TARGET_ATR,
        "target_policy": "separate realized protected LONG and SHORT returns", "results": results,
    }
    PROTECTED_TARGET_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report


def run_8h_protected_classifier_walk_forward(timeframe: str = "1h") -> dict:
    data = add_protective_outcomes(add_horizon_return(_load(timeframe), 8), 8, timeframe)
    data["long_win"] = data["long_protected_return"].gt(round_trip_cost_rate()).astype(int)
    data["short_win"] = data["short_protected_return"].gt(round_trip_cost_rate()).astype(int)
    results = []
    for fold, boundaries in enumerate(FOLDS, start=1):
        train, calibration, test = _fractional_split(data, *boundaries); predictions = {}
        for target in ("long_win", "short_win"):
            model = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=250, max_leaf_nodes=31, min_samples_leaf=100,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.10, random_state=42,
            ).fit(train[FEATURES].to_numpy(np.float32), train[target].to_numpy(int))
            predictions[target] = (model.predict_proba(calibration[FEATURES].to_numpy(np.float32))[:, 1],
                                   model.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1])
        cl, tl = predictions["long_win"]; cs, ts = predictions["short_win"]
        long_threshold = _side_threshold(calibration, cl, 8, "long")
        short_threshold = _side_threshold(calibration, -cs, 8, "short")
        def score(lp, sp):
            long_ok, short_ok = lp > long_threshold, sp > short_threshold
            return np.where(long_ok & (~short_ok | (lp >= sp)), lp, np.where(short_ok, -sp, 0.0))
        results.append({
            "fold": fold, "long_threshold": long_threshold, "short_threshold": short_threshold,
            "calibration": portfolio_metrics(calibration, score(cl, cs), (0.0, 0.0), 8),
            "test": portfolio_metrics(test, score(tl, ts), (0.0, 0.0), 8),
        })
    stable = all(row["test"]["trades"] >= 50 and row["test"]["return_pct"] > 0 and row["test"]["profit_factor"] >= 1.20 for row in results)
    report = {"stable": stable, "promoted": False, "horizon": 8, "stop_atr": CHALLENGER_STOP_ATR,
              "target_atr": CHALLENGER_TARGET_ATR, "target_policy": "net-profitable protected outcome by direction", "results": results}
    PROTECTED_CLASSIFIER_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report
