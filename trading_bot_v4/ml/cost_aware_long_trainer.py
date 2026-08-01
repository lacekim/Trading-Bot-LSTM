"""All-asset cost-aware LONG challenger with untouched chronological holdout."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from trading_bot_v4.config_v4 import V4Config as Config


MODEL_PATH = Path("models/cost_aware_long_model.pkl")
CALIBRATION_PATH = Path("models/cost_aware_long_calibration.json")
VALIDATION_PATH = Path("reports/v5_cost_aware_long_validation.csv")
WALK_FORWARD_PATH = Path("reports/v5_cost_aware_long_walk_forward.json")
DIRECTIONAL_WALK_FORWARD_PATH = Path("reports/v5_cost_aware_directional_walk_forward.json")
# SMC swing columns are intentionally excluded: the current standalone swing
# builder confirms pivots with centered windows and is not causal at the signal
# timestamp. Only the original trailing indicators are eligible here.
FEATURES = [*Config.FEATURE_COLUMNS]
TRAIN_END, CALIBRATION_END = 0.70, 0.85
MIN_CALIBRATION_TRADES = 1000
MIN_HOLDOUT_TRADES = 200
MIN_SYMBOL_CALIBRATION_TRADES = 20


@dataclass(frozen=True)
class CostAwareLongResult:
    promoted: bool
    threshold: float
    calibration_trades: int
    holdout_trades: int
    holdout_return_pct: float
    holdout_profit_factor: float
    holdout_precision: float
    holdout_auc: float


def round_trip_cost_rate() -> float:
    one_way_bps = Config.PAPER_FEE_BPS + Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS
    return 2.0 * one_way_bps / 10000.0


def build_cost_aware_long_target(future_return: pd.Series, cost_rate: float | None = None) -> pd.Series:
    cost = round_trip_cost_rate() if cost_rate is None else float(cost_rate)
    required_return = max(cost, float(Config.MOVEMENT_THRESHOLD))
    return pd.to_numeric(future_return, errors="coerce").gt(required_return).astype(int)


def _load(timeframe: str) -> pd.DataFrame:
    path = Path("models") / f"training_data_smc_all_assets_{timeframe}.csv"
    source_features = [*Config.FEATURE_COLUMNS]
    usecols = ["timestamp", "symbol", "future_return", *source_features]
    data = pd.read_csv(path, usecols=usecols)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.upper()
    numeric = ["future_return", *source_features]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "symbol", *numeric])
    # Reject obvious hourly data discontinuities instead of allowing a handful
    # of bad prints/listing jumps to manufacture enormous compounded returns.
    data = data.loc[data["future_return"].abs().le(0.20)].copy()
    data["target"] = build_cost_aware_long_target(data["future_return"])
    return data.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, calibration, holdout = [], [], []
    for _, group in data.groupby("symbol", sort=True):
        first, second = int(len(group) * TRAIN_END), int(len(group) * CALIBRATION_END)
        if first < 100 or second <= first or len(group) <= second:
            continue
        train.append(group.iloc[:first]); calibration.append(group.iloc[first:second]); holdout.append(group.iloc[second:])
    return tuple(pd.concat(parts, ignore_index=True) for parts in (train, calibration, holdout))


def _fractional_split(data: pd.DataFrame, train_end: float, calibration_end: float,
                      test_end: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, calibration, test = [], [], []
    for _, group in data.groupby("symbol", sort=True):
        first = int(len(group) * train_end)
        second = int(len(group) * calibration_end)
        third = int(len(group) * test_end)
        if first < 100 or second <= first or third <= second:
            continue
        train.append(group.iloc[:first]); calibration.append(group.iloc[first:second]); test.append(group.iloc[second:third])
    return tuple(pd.concat(parts, ignore_index=True) for parts in (train, calibration, test))


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.10, random_state=42,
    )


def _trade_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    selected = frame.loc[np.asarray(probability) >= threshold].copy()
    net = pd.to_numeric(selected["future_return"], errors="coerce") - round_trip_cost_rate()
    wins, losses = net[net > 0], net[net < 0]
    compounded = float((1.0 + net.clip(lower=-0.99)).prod() - 1.0) if len(net) else 0.0
    return {
        "trades": int(len(net)), "return_pct": compounded * 100.0,
        "mean_trade_pct": float(net.mean() * 100.0) if len(net) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else 0.0),
        "win_rate_pct": float((net > 0).mean() * 100.0) if len(net) else 0.0,
    }


def _choose_threshold(frame: pd.DataFrame, probability: np.ndarray) -> tuple[float, dict]:
    candidates = []
    # Probability rank is meaningful for an imbalanced classifier; 0.50 is
    # not a universal decision boundary. Include empirical upper quantiles so
    # every fold can evaluate adequately sized selective candidates.
    thresholds = set(np.arange(0.10, 0.951, 0.01).round(4).tolist())
    thresholds.update(float(value) for value in np.quantile(probability, np.arange(0.80, 0.991, 0.01)))
    for threshold in sorted(thresholds):
        metrics = _trade_metrics(frame, probability, float(threshold))
        if metrics["trades"] >= MIN_CALIBRATION_TRADES:
            candidates.append((float(threshold), metrics))
    viable = [item for item in candidates if item[1]["profit_factor"] >= 1.20 and item[1]["return_pct"] > 0]
    pool = viable or candidates
    if not pool:
        return 1.0, _trade_metrics(frame, probability, 1.0)
    return max(pool, key=lambda item: (item[1]["profit_factor"], item[1]["return_pct"]))


def train_cost_aware_long_model(timeframe: str = "1h") -> CostAwareLongResult:
    data = _load(timeframe)
    train, calibration, holdout = _split(data)
    model = _new_model()
    model.fit(train[FEATURES].to_numpy(np.float32), train["target"].to_numpy(int))
    calibration_probability = model.predict_proba(calibration[FEATURES].to_numpy(np.float32))[:, 1]
    threshold, calibration_metrics = _choose_threshold(calibration, calibration_probability)
    symbol_calibration = []
    for symbol, indices in calibration.groupby("symbol").groups.items():
        positions = calibration.index.get_indexer(indices)
        metrics = _trade_metrics(calibration.loc[indices], calibration_probability[positions], threshold)
        symbol_calibration.append({"symbol": symbol, **metrics})
    qualified_symbols = {
        row["symbol"] for row in symbol_calibration
        if row["trades"] >= MIN_SYMBOL_CALIBRATION_TRADES
        and row["profit_factor"] >= 1.30 and row["return_pct"] > 0
    }
    holdout_probability = model.predict_proba(holdout[FEATURES].to_numpy(np.float32))[:, 1]
    unrestricted_holdout_metrics = _trade_metrics(holdout, holdout_probability, threshold)
    qualified_mask = holdout["symbol"].isin(qualified_symbols).to_numpy()
    qualified_holdout = holdout.loc[qualified_mask].copy()
    qualified_probability = holdout_probability[qualified_mask]
    holdout_metrics = _trade_metrics(qualified_holdout, qualified_probability, threshold)
    predicted = qualified_probability >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        qualified_holdout["target"].to_numpy(int), predicted.astype(int), average="binary", zero_division=0
    )
    auc = roc_auc_score(qualified_holdout["target"], qualified_probability) if len(qualified_holdout) else float("nan")
    windows = []
    chronological = qualified_holdout.copy()
    chronological["_probability"] = qualified_probability
    chronological = chronological.sort_values("timestamp").reset_index(drop=True)
    for window, indices in enumerate(np.array_split(np.arange(len(chronological)), 3), start=1):
        section = chronological.iloc[indices]
        metrics = _trade_metrics(section, section["_probability"].to_numpy(), threshold)
        windows.append({"window": window, **metrics})
    promoted = bool(
        calibration_metrics["trades"] >= MIN_CALIBRATION_TRADES
        and calibration_metrics["profit_factor"] >= 1.20
        and holdout_metrics["trades"] >= MIN_HOLDOUT_TRADES
        and holdout_metrics["profit_factor"] >= 1.20
        and holdout_metrics["return_pct"] > 0
        and precision >= 0.55
        and all(row["trades"] >= 30 and row["profit_factor"] >= 1.0 and row["return_pct"] > 0 for row in windows)
    )
    artifact = {
        "promoted": promoted, "timeframe": timeframe, "threshold": threshold,
        "target": "next_candle_return > max(configured round_trip_cost, meaningful movement threshold)",
        "round_trip_cost_rate": round_trip_cost_rate(),
        "split_policy": "70% train / 15% calibration / 15% untouched holdout per asset",
        "features": FEATURES, "calibration": calibration_metrics, "qualified_symbols": sorted(qualified_symbols),
        "symbol_calibration": symbol_calibration,
        "unrestricted_holdout": unrestricted_holdout_metrics,
        "holdout": holdout_metrics,
        "holdout_precision": float(precision), "holdout_recall": float(recall),
        "holdout_f1": float(f1), "holdout_auc": float(auc), "windows": windows,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(model, handle)
    CALIBRATION_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(windows).to_csv(VALIDATION_PATH, index=False)
    print(json.dumps(artifact, indent=2))
    return CostAwareLongResult(
        promoted, threshold, int(calibration_metrics["trades"]), int(holdout_metrics["trades"]),
        float(holdout_metrics["return_pct"]), float(holdout_metrics["profit_factor"]), float(precision), float(auc),
    )


def run_cost_aware_long_walk_forward(timeframe: str = "1h") -> dict:
    """Repeated expanding-window validation; no fold model is deployed."""
    data = _load(timeframe)
    folds = (
        (0.55, 0.70, 0.80),
        (0.65, 0.80, 0.90),
        (0.75, 0.90, 1.00),
    )
    results = []
    for number, (train_end, calibration_end, test_end) in enumerate(folds, start=1):
        train, calibration, test = _fractional_split(data, train_end, calibration_end, test_end)
        model = _new_model()
        model.fit(train[FEATURES].to_numpy(np.float32), train["target"].to_numpy(int))
        calibration_probability = model.predict_proba(calibration[FEATURES].to_numpy(np.float32))[:, 1]
        threshold, calibration_metrics = _choose_threshold(calibration, calibration_probability)
        test_probability = model.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]
        test_metrics = _trade_metrics(test, test_probability, threshold)
        results.append({
            "fold": number, "train_end": train_end, "calibration_end": calibration_end,
            "test_end": test_end, "threshold": threshold,
            "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
            "calibration": calibration_metrics, "test": test_metrics,
        })
    stable = bool(all(
        row["test"]["trades"] >= 100
        and row["test"]["return_pct"] > 0
        and row["test"]["profit_factor"] >= 1.20
        for row in results
    ))
    report = {
        "stable": stable, "promoted": False,
        "policy": "expanding train / separate calibration / unseen next segment; >=100 trades, positive return and PF>=1.20 every fold",
        "features": FEATURES, "round_trip_cost_rate": round_trip_cost_rate(), "folds": results,
        "note": "Walk-forward stability alone does not deploy a model; the final artifact must also pass its untouched holdout.",
    }
    WALK_FORWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    WALK_FORWARD_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def _directional_metrics(frame: pd.DataFrame, prediction: np.ndarray, threshold: float) -> dict[str, float | int]:
    direction = np.where(prediction > threshold, 1.0, np.where(prediction < -threshold, -1.0, 0.0))
    selected = direction != 0
    trades = frame.loc[selected, ["timestamp", "symbol", "future_return", "atr_norm"]].copy()
    trades["direction"] = direction[selected]
    trades["strength"] = np.abs(prediction[selected])
    trades["trade_return"] = trades["future_return"] * trades["direction"] - round_trip_cost_rate()
    capital = 1.0
    realized_pnl: list[float] = []
    executed_long = executed_short = 0
    for _, candidates in trades.groupby("timestamp", sort=True):
        candidates = candidates.sort_values("strength", ascending=False).head(Config.PAPER_MAX_OPEN_POSITIONS)
        remaining_exposure = Config.PAPER_MAX_PORTFOLIO_EXPOSURE_PCT / 100.0
        period_pnl = 0.0
        for _, trade in candidates.iterrows():
            stop_pct = float(trade["atr_norm"]) * Config.ATR_SL_MULTIPLIER
            desired = min(
                (Config.RISK_PERCENTAGE / 100.0) / max(stop_pct, 1e-12),
                Config.PAPER_MAX_POSITION_PCT / 100.0,
            )
            allocation = min(desired, remaining_exposure)
            if allocation <= 0:
                continue
            pnl = capital * allocation * float(trade["trade_return"])
            period_pnl += pnl; realized_pnl.append(pnl)
            executed_long += int(trade["direction"] > 0); executed_short += int(trade["direction"] < 0)
            remaining_exposure -= allocation
        capital += period_pnl
    net = pd.Series(realized_pnl, dtype=float)
    wins, losses = net[net > 0], net[net < 0]
    return {
        "trades": int(len(net)), "long_trades": executed_long, "short_trades": executed_short,
        "return_pct": float((capital - 1.0) * 100.0),
        "mean_trade_pct": float(trades["trade_return"].mean() * 100.0) if len(trades) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else 0.0),
        "win_rate_pct": float((net > 0).mean() * 100.0) if len(net) else 0.0,
    }


def _choose_directional_threshold(frame: pd.DataFrame, prediction: np.ndarray) -> tuple[float, dict]:
    thresholds = np.unique(np.quantile(np.abs(prediction), np.arange(0.80, 0.991, 0.01)))
    candidates = [(float(value), _directional_metrics(frame, prediction, float(value))) for value in thresholds]
    viable = [item for item in candidates if item[1]["trades"] >= 200 and item[1]["profit_factor"] >= 1.20 and item[1]["return_pct"] > 0]
    if not viable:
        adequately_sampled = [item for item in candidates if item[1]["trades"] >= 200]
        if adequately_sampled:
            # Diagnostic only; callers still reject it through their explicit
            # profitability gates. Returning it exposes the least-bad result.
            return max(adequately_sampled, key=lambda item: (item[1]["profit_factor"], item[1]["return_pct"]))
        return float("inf"), _directional_metrics(frame, prediction, float("inf"))
    return max(viable, key=lambda item: (item[1]["profit_factor"], item[1]["return_pct"]))


def run_cost_aware_directional_walk_forward(timeframe: str = "1h") -> dict:
    """Evaluate abstaining LONG/SHORT regression over expanding unseen folds."""
    data = _load(timeframe)
    folds = ((0.55, 0.70, 0.80), (0.65, 0.80, 0.90), (0.75, 0.90, 1.00))
    results = []
    for number, boundaries in enumerate(folds, start=1):
        train, calibration, test = _fractional_split(data, *boundaries)
        model = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
            min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.10, random_state=42,
        )
        model.fit(train[FEATURES].to_numpy(np.float32), train["future_return"].to_numpy(float))
        calibration_prediction = model.predict(calibration[FEATURES].to_numpy(np.float32))
        threshold, calibration_metrics = _choose_directional_threshold(calibration, calibration_prediction)
        test_prediction = model.predict(test[FEATURES].to_numpy(np.float32))
        results.append({
            "fold": number, "threshold": threshold, "calibration": calibration_metrics,
            "test": _directional_metrics(test, test_prediction, threshold),
        })
    stable = bool(all(
        row["test"]["trades"] >= 100 and row["test"]["return_pct"] > 0
        and row["test"]["profit_factor"] >= 1.20 for row in results
    ))
    report = {
        "stable": stable, "promoted": False,
        "policy": "causal all-asset return regression; LONG/SHORT only beyond separately calibrated symmetric margin",
        "round_trip_cost_rate": round_trip_cost_rate(), "folds": results,
    }
    DIRECTIONAL_WALK_FORWARD_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
