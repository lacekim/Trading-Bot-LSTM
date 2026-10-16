"""Statistically honest re-test of cross-sectional momentum.

Every failed result tonight -- and the pre-existing gmx_market_state_challenger
and multi_horizon_directional walk-forwards already in this repo -- share one
signature: good on calibration, bad on held-out test. That's the classic
footprint of overfitting a threshold sweep to a small calibration window,
not genuine edge. Sweeping many candidates and keeping the best one will
usually look good by chance alone; a single calibration/test split or even
3-4 walk-forward windows doesn't correct for that, it just delays where the
illusion breaks.

This applies a real correction: run momentum on MANY more non-overlapping
folds (not 3-4), then run a one-sided binomial test on the fraction of
profitable folds against the 50% you'd expect under a true null of no edge.
No threshold is swept or cherry-picked here -- momentum's own ranking IS
the signal, with fixed, pre-chosen parameters (same lookback/hold used
earlier tonight), so there's nothing here to overfit to a calibration
window in the first place.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy import stats

from trading_bot import list_gmx_symbols
from trading_bot_v4.utils.logger import build_logger
from tools.run_cross_sectional_momentum_experiment import (
    HOLD_CANDLES,
    LOOKBACK_CANDLES,
    TIMEFRAME,
    backtest,
    build_universe,
)

logger = build_logger("v4_momentum_significance_test")

N_FOLDS = 20


def main() -> None:
    symbols = list_gmx_symbols(TIMEFRAME)
    prices = build_universe(symbols)
    n = len(prices)
    fold_size = n // N_FOLDS

    fold_results = []
    for fold in range(N_FOLDS):
        start = fold * fold_size
        end = n if fold == N_FOLDS - 1 else (fold + 1) * fold_size
        fold_prices = prices.iloc[start:end]
        if len(fold_prices) < LOOKBACK_CANDLES + HOLD_CANDLES * 2:
            continue
        result = backtest(fold_prices)
        if result["rebalances"] == 0:
            continue
        fold_results.append(result)
        logger.info(f"fold {fold + 1:>2} ({fold_prices.index.min()} to {fold_prices.index.max()}): "
                    f"rebalances={result['rebalances']:>3} return_pct={result['return_pct']:>7.2f} "
                    f"profit_factor={result['profit_factor']:.2f}")

    n_folds = len(fold_results)
    n_profitable = sum(1 for r in fold_results if r["return_pct"] > 0)
    mean_return = sum(r["return_pct"] for r in fold_results) / n_folds
    mean_pf = sum(r["profit_factor"] for r in fold_results if r["profit_factor"] != float("inf")) / n_folds

    # one-sided binomial test: is the observed fraction of profitable folds
    # significantly better than the 50% you'd expect by chance under no edge?
    binom_result = stats.binomtest(n_profitable, n_folds, p=0.5, alternative="greater")

    logger.info(f"\n{n_folds} independent folds, {n_profitable} profitable ({n_profitable/n_folds*100:.1f}%)")
    logger.info(f"mean fold return: {mean_return:.2f}%  mean profit factor: {mean_pf:.2f}")
    logger.info(f"one-sided binomial test vs 50% chance: p-value={binom_result.pvalue:.4f}")
    logger.info(f"STATISTICALLY SIGNIFICANT (p<0.05): {binom_result.pvalue < 0.05}")


if __name__ == "__main__":
    main()
