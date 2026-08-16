"""Diagnostic: does the ~0.50 AUC found tonight hold across every market
regime, or only in the most recent holdout window?

Every promotion experiment tonight used a chronological split, so the
holdout was always the most recent ~15% of each symbol's history. For
BTC/ETH that window (Mar 2025-present) turned out to be an unusually adverse
one -- BTC -27.5% with a -54% max drawdown, ETH -10.2% with a -69% max
drawdown. That raises a fair question: is "no edge" a property of the
feature set, or an artifact of testing on one hard window?

This splits full history into contiguous ~1.5-year eras and does an
expanding-window walk-forward: train on everything up to era N, test AUC on
era N+1, for each era in sequence. If AUC stays ~0.50 across bull, bear, and
chop regimes alike, that rules out "unlucky test window" as the explanation.
If some eras show real AUC while only the most recent one doesn't, that
supports the regime-masking hypothesis. Reuses the exact target (triple-
barrier) and feature set (18 base + BTC relative-strength) already built and
validated tonight -- diagnostic only, no models saved, no promotion gating.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tools.run_gbm_relative_strength_experiment import FEATURE_COLUMNS, _load_symbol_frame, _new_model

N_ERAS = 6


def check_symbol(symbol: str, direction: str) -> None:
    df = _load_symbol_frame(symbol, "1h", direction)
    if df is None:
        print(f"{symbol}/{direction}: no data")
        return

    n = len(df)
    edges = np.linspace(0, n, N_ERAS + 1).astype(int)
    eras = [df.iloc[edges[i]:edges[i + 1]] for i in range(N_ERAS)]

    print(f"\n=== {symbol}/{direction} ({n} rows, {N_ERAS} eras) ===")
    for i in range(1, N_ERAS):
        train = pd.concat(eras[:i], ignore_index=True)
        test = eras[i]
        if len(test) < 200 or train["target"].nunique() < 2:
            continue
        model = _new_model()
        model.fit(train[FEATURE_COLUMNS].to_numpy(np.float32), train["target"].to_numpy(int))
        proba = model.predict_proba(test[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
        target = test["target"].to_numpy(int)
        auc = roc_auc_score(target, proba) if len(np.unique(target)) > 1 else float("nan")
        close = test["Close"].astype(float)
        price_change_pct = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
        max_dd_pct = ((close / close.cummax()) - 1.0).min() * 100.0
        start, end = test["timestamp"].iloc[0], test["timestamp"].iloc[-1]
        print(f"  era {i}: {str(start)[:10]} to {str(end)[:10]} | "
              f"price {price_change_pct:+7.1f}% | max_dd {max_dd_pct:6.1f}% | "
              f"train_rows={len(train):>6} test_rows={len(test):>5} | AUC={auc:.3f}")


if __name__ == "__main__":
    for symbol in ["BTC", "ETH"]:
        for direction in ["long", "short"]:
            check_symbol(symbol, direction)
