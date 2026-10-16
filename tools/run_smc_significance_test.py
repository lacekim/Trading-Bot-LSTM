"""Statistically honest re-test of the ARB/short SMC promotion.

Same reasoning as run_momentum_significance_test.py: a single calibration
-> 3-window holdout pass, however disciplined, is still one pass. ARB/short
promoted on 15 total holdout trades over a ~5-week window, out of only
~8.2 months of total SMC feature history -- thin enough that it deserves
the same finer-grained, many-fold check that caught the momentum
strategy's illusion earlier tonight, before trusting it.

Expanding-window walk-forward across the full available SMC history: train
on everything up to fold N, calibrate a threshold on a slice of it, test on
fold N+1, repeat. Reports whether the fraction of profitable folds is
statistically distinguishable from chance via a one-sided binomial test.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats

from trading_bot_v4.ml.per_asset_trainer import MIN_CALIBRATION_TRADES
from trading_bot_v4.utils.logger import build_logger
from tools.run_gbm_relative_strength_experiment import _new_model
from tools.run_smc_per_asset_experiment import FEATURE_COLUMNS, _load_symbol_frame, _ENDPOINT_COLUMNS
from tools.run_triple_barrier_target_experiment import _select_threshold, _simulate_threshold

logger = build_logger("v4_smc_significance_test")

N_FOLDS = 8
CALIBRATION_FRACTION = 0.5  # within each fold's own held-back window, before testing forward


def evaluate(symbol: str, direction: str) -> None:
    df = _load_symbol_frame(symbol, direction)
    if df is None:
        logger.info(f"{symbol}/{direction}: no data")
        return
    n = len(df)
    edges = np.linspace(0, n, N_FOLDS + 2).astype(int)  # first slice is pure warm-up training data

    fold_results = []
    for fold in range(1, N_FOLDS + 1):
        train = df.iloc[:edges[fold]]
        test = df.iloc[edges[fold]:edges[fold + 1]]
        if len(test) < 20 or train["target"].nunique() < 2:
            continue

        cal_split = int(len(train) * (1 - CALIBRATION_FRACTION * 0.3))  # last ~15% of train as calibration
        fit_rows, calibration = train.iloc[:cal_split], train.iloc[cal_split:]
        if calibration["target"].nunique() < 2 or len(calibration) < 50:
            continue

        model = _new_model()
        model.fit(fit_rows[FEATURE_COLUMNS].to_numpy(np.float32), fit_rows["target"].to_numpy(int))

        calibration_proba = model.predict_proba(calibration[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
        calibration_frame = calibration[_ENDPOINT_COLUMNS].reset_index(drop=True)
        threshold, _ = _select_threshold(calibration_frame, calibration_proba, direction, symbol)

        test_proba = model.predict_proba(test[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
        test_frame = test[_ENDPOINT_COLUMNS].reset_index(drop=True)
        summary = _simulate_threshold(test_frame, test_proba, threshold, direction, symbol)
        if summary["trades"] == 0:
            continue
        fold_results.append(summary)
        logger.info(f"{symbol}/{direction} fold {fold}: "
                    f"{test['timestamp'].min()} to {test['timestamp'].max()} | "
                    f"trades={summary['trades']:>3} return_pct={summary['return_pct']:>7.2f} "
                    f"profit_factor={summary['profit_factor']:.2f}")

    if not fold_results:
        logger.info(f"{symbol}/{direction}: no fold produced any trades, insufficient data for a significance test")
        return

    n_folds = len(fold_results)
    n_profitable = sum(1 for r in fold_results if r["return_pct"] > 0)
    total_trades = sum(r["trades"] for r in fold_results)
    mean_return = sum(r["return_pct"] for r in fold_results) / n_folds
    binom_result = stats.binomtest(n_profitable, n_folds, p=0.5, alternative="greater")

    logger.info(f"\n{symbol}/{direction}: {n_folds} folds ({total_trades} total trades), "
                f"{n_profitable} profitable ({n_profitable/n_folds*100:.1f}%)")
    logger.info(f"{symbol}/{direction}: mean fold return {mean_return:.2f}%  "
                f"binomial p-value={binom_result.pvalue:.4f}  "
                f"SIGNIFICANT (p<0.05): {binom_result.pvalue < 0.05}")


if __name__ == "__main__":
    evaluate("ARB", "short")
