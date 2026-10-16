"""LightGBM on the exact same meta-labeling pooled dataset as
run_meta_label_experiment.py (same trigger-filtered events, same 22 features
-- base 18 + quarter/activity/ARIMA context -- same 36-bar history window,
same chronological train/val/calibration/holdout split), just flattened to a
flat feature vector instead of fed to a CNN/LSTM as a sequence.

This is a genuine architecture comparison, not a repeat: identical
information, identical labels, identical split -- only the model class
differs. Motivated directly by the Jena Climate article's finding that
LightGBM was the honest strong performer there (good accuracy without the
suspicious one-step-ahead evaluation advantage neural nets got), and that it
natively handles engineered/lag features better than a sequence model.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

import tools.run_meta_label_experiment as meta
from trading_bot_v4.execution.gmx_adapter import list_gmx_symbols
from trading_bot_v4.risk.market_cap_tiers import liquid_symbols
from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_meta_label_lightgbm")
MODEL_ROOT = Path("models/experiments/meta_label_lightgbm")


def run_direction(direction: str, symbols: list[str]) -> dict:
    index_frame, feature_history = meta.build_pooled_dataset(direction, symbols)
    n = len(index_frame)
    logger.info(f"{direction}: {n} pooled triggered events across {len(feature_history)} symbols")
    if n < 200:
        return {"direction": direction, "trained": False, "reason": f"too few pooled events: {n}"}

    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    cal_end = int(n * 0.90)
    train_idx = index_frame.iloc[:train_end]
    val_idx = index_frame.iloc[train_end:val_end]
    cal_idx = index_frame.iloc[val_end:cal_end]
    hold_idx = index_frame.iloc[cal_end:]

    handler = meta.DataHandler()
    train_feature_rows = np.vstack([
        feature_history[row.symbol][row.row_pos] for row in train_idx.itertuples()
    ])
    handler.normalize_data(train_feature_rows, fit=True)
    scaler = handler.scaler

    # Same 36-bar sequences the CNN/LSTM sees, just flattened to a flat
    # vector for the tree model -- identical information, different model.
    X_train = meta._build_sequences(train_idx, feature_history, scaler).reshape(len(train_idx), -1)
    X_val = meta._build_sequences(val_idx, feature_history, scaler).reshape(len(val_idx), -1)
    X_cal = meta._build_sequences(cal_idx, feature_history, scaler).reshape(len(cal_idx), -1)
    X_hold = meta._build_sequences(hold_idx, feature_history, scaler).reshape(len(hold_idx), -1)
    y_train = train_idx["meta_label"].to_numpy(int)
    y_val = val_idx["meta_label"].to_numpy(int)

    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=5, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
        random_state=0, verbosity=-1,
    )
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        eval_metric="auc", callbacks=[],
    )

    cal_probability = model.predict_proba(X_cal)[:, 1]
    hold_probability = model.predict_proba(X_hold)[:, 1]
    hold_target = hold_idx["meta_label"].to_numpy(int)
    holdout_auc = (
        float(roc_auc_score(hold_target, hold_probability))
        if len(np.unique(hold_target)) > 1 else float("nan")
    )

    cal_returns = cal_idx["net_return"].to_numpy(np.float64)
    hold_returns = hold_idx["net_return"].to_numpy(np.float64)

    best_threshold, best_pf = 0.5, -1.0
    for threshold in np.arange(0.40, 0.91, 0.02):
        mask = cal_probability >= threshold
        if mask.sum() < meta.MIN_CALIBRATION_TRADES:
            continue
        selected = cal_returns[mask]
        gains = selected[selected > 0].sum()
        losses = -selected[selected < 0].sum()
        pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
        if selected.sum() > 0 and pf > best_pf:
            best_pf, best_threshold = pf, float(threshold)

    hold_mask = hold_probability >= best_threshold
    hold_selected = hold_returns[hold_mask]
    promoted, windows = meta._promotion_check(hold_selected)

    gains = hold_selected[hold_selected > 0].sum()
    losses = -hold_selected[hold_selected < 0].sum()
    holdout_pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with (MODEL_ROOT / f"model_{direction}.pkl").open("wb") as f:
        pickle.dump(model, f)
    with (MODEL_ROOT / f"scaler_{direction}.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    result = {
        "direction": direction, "trained": True, "promoted": promoted,
        "threshold": best_threshold, "pooled_events": int(n),
        "holdout_events": len(hold_idx), "holdout_trades_selected": int(hold_mask.sum()),
        "holdout_return_pct": float(hold_selected.sum() * 100.0),
        "holdout_profit_factor": holdout_pf, "holdout_auc": holdout_auc,
        "windows": windows,
    }
    logger.info(
        f"{direction}: promoted={promoted} threshold={best_threshold:.2f} "
        f"holdout_trades={result['holdout_trades_selected']}/{result['holdout_events']} "
        f"holdout_return_pct={result['holdout_return_pct']:.2f} "
        f"holdout_profit_factor={holdout_pf:.2f} holdout_auc={holdout_auc:.3f}"
    )
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    args = parser.parse_args()
    directions = ["long", "short"] if args.direction == "both" else [args.direction]

    all_symbols = list_gmx_symbols("1h")
    results = []
    for direction in directions:
        symbols = liquid_symbols(direction.upper(), all_symbols)
        results.append(run_direction(direction, symbols))

    out = Path("logs/v4_meta_label_lightgbm_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
