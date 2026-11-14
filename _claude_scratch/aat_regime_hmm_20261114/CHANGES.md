# Advanced Algorithmic Trading, regime detection with a hidden Markov model, on top of the golden cross

What the book does (regime_hmm_train.py, regime_hmm_risk_manager.py, regime_hmm_strategy.py, regime_hmm_backtest.py):
a moving average crossover strategy, plus a risk manager that fits a 2-state Gaussian HMM to the traded instrument's
returns on data BEFORE the backtest period and then, bar by bar, blocks new long entries while the HMM says the
market is in the undesirable (high-volatility) regime. Exits are always allowed. A blocked entry is skipped until
the next crossover.
What is done here: the same filter put on the user's golden cross, in the user's own engines, Kraken and Alpaca.
Every step is one commit. The user said on 2026-11-14: "Do whats next, write everything down."

### AAT regime HMM step 1: two-state Gaussian HMM in numpy (hmmlearn is not installed) and its checks
Files: hmm_np.py (Baum-Welch fit, bar-by-bar last-Viterbi-state, calm-state picker), test_hmm_np.py, test_hmm_np_output.txt.
Why: the book calls hmmlearn.hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=1000) and hmm_model.predict(returns)[-1]; hmmlearn is not installed where the code runs and there is no network, so the same model is written out. Stated differences from hmmlearn: deterministic starting means/variances (split of returns by absolute size) instead of seeded k-means; hmmlearn's small covariance priors are not applied. The book's code assumes state 0 is the desirable regime, which hmmlearn does not guarantee; here the desirable regime is defined as the lower-variance state.
Checks, all passed: on 6,000 simulated two-regime returns the fit recovers the volatilities (0.00608 vs 0.006; 0.02483 vs 0.025) and the bar-by-bar regime matches the truth 94.7% of the time; EM log-likelihood never falls; the log-likelihood from a separate log-space forward pass agrees to 1.35e-10; the bar-by-bar state equals a brute-force Viterbi over all paths on 12 prefixes; removing later data never changes an earlier state (no look-ahead).

### AAT regime HMM step 2: fix a bug in my HMM found on the first Kraken run (emission underflow on extreme returns)
Found: the first Kraken run fitted ETH in 2 iterations with nonsense volatilities (calm 5.06%, wild 6.94% per 4h bar) and the log-likelihood FELL from 38841.4 to 31898.4, which EM must never do. Cause: ETH's first weeks (August 2015) contain 4h returns of +700% and +175%; my emission densities were floored at 1e-300, both states hit the floor on those bars, and the floor broke the EM update. The first Kraken numbers are therefore discarded and are not reported as a result.
Fix (fix_01_emission_underflow.patch): emission densities are computed in log space and each bar is scaled by its larger entry, which is exact and needs no floor; the stop rule uses the absolute log-likelihood change. New regression test with +700%, +175% and -92% returns inserted: 41 iterations, log-likelihood never falls. All earlier checks still pass (test_hmm_np_output.txt).
