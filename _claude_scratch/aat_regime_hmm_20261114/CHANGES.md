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

### AAT regime HMM step 3: the HMM regime filter on the golden cross, Kraken
Script: regime_filter_kraken.py (the user's run(); the one changed line is 'if want:' -> 'if want and bool(r.reg_ok):'; one HMM per asset fitted on 4h returns before 2024-01-01 only; regime at each bar uses returns up to that bar). Saved: kraken_regime_off.json (filter off: reproduces the bar exactly, 145 trades +107.5% and 34 trades +25.0%), kraken_regime_hmm.json (with fitted model parameters per asset); trade lists and equity curves are CSV on disk.
Result, filter off vs HMM filter. LIVE: from 2024-01-01 +107.5% (max DD -21.5%, PF 1.82, 145 trades) vs +111.2% (max DD -20.3%, PF 1.98, 132 trades); last 365 days +25.0% (max DD -14.2%) vs +25.2% (max DD -14.2%). VALIDATED (a blocked entry waits for the next cross, which is the book's behaviour): from 2024-01-01 +124.8% (max DD -17.7%, PF 2.48, 89 trades) vs +106.4% (max DD -17.9%, PF 2.39, 81 trades); last 365 days +28.2% vs +24.7% (PF 2.39 vs 2.01).
Fitted regimes: calm share of bars from 2024: BTC 83%, XMR 83%, PAXG 83%, SOL 87%, TRX 95%, ETH 99%. ETH's wild state has a 23% per-bar volatility because it is fitted to the August 2015 listing bars, so the filter almost never blocks ETH. Not tuned.
Verdict on Kraken: no clear improvement. Slightly better in the LIVE variant (3.7 points and 1.2 points of drawdown from 2024), worse in the VALIDATED variant (18.4 points), flat over the last 365 days.
