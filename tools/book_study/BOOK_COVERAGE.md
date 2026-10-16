# Book coverage ledger: AFML extract, every numbered snippet (Ch.2-21)

Source: the 218-page extract the user supplied. Every snippet below is a port of the printed code in `tools/book_study/book_mp.py` and `book_port_a.py` ... `book_port_e.py` (the file headers list every change from the printed text), unit-tested in `test_book_port_*.py`, and applied to each venue's dollar bars by `run_book_ports_bars.py` (section named in the last column). This replaces the previous ledger, whose 'built, in bar runs' rows rested on the project's earlier implementations written from memory; those were compared with the book code (differences are listed in the sections' text).

- Kraken run: `book_ports_kraken_bars/20261014T130136Z`; 51,002 events; seconds per section in `summary_numbers.json`; errors: none; independent checker (verify_book_ports_bars.py): 224 of 224.
- Alpaca run: `book_ports_alpaca_bars/20261014T130136Z`; 351,040 events; seconds per section in `summary_numbers.json`; errors: none; independent checker (verify_book_ports_bars.py): 224 of 224.

Counts (97 snippets): 87 ported verbatim; run on both venues' dollar bars; 2 ported verbatim and unit-tested; not applicable to these data; 6 ported with a stated substitution; run on both venues' dollar bars; 1 ported verbatim; run writes tables, not the plot; 1 ported; a Python 2 recipe that Python 3 does not need.

Not a snippet, in the text, NOT built: Ch.19 Kyle / Amihud / Hasbrouck lambdas, VPIN and PIN (need trade-level data; the bars carry no trade flags). The real CLA code of Ch.16 is not in the extract (stand-in above). Chapter 22 is not in the extract.

Run-size reductions against the printed examples (trees, Monte Carlo iterations, samples) are stated in each section of SUMMARY.md and in `summary_numbers.json` (`meta.config`).

| Snippet | Page | Title | Status | Code | Run section | Note |
|---|---|---|---|---|---|---|
| 2.1 | 21 | Pca Weights From A Risk Distribution R | ported verbatim; run on both venues' dollar bars | book_port_a.py: pcaWeights | ch2 | applied to the cross-section of daily bar returns |
| 2.2 | 22 | Form A Gaps Series, Detract It From Prices | ported verbatim and unit-tested; not applicable to these data | book_port_a.py: rollGaps, getRolledSeries, getRolledSeries_df | ch2 | futures roll gaps: no futures on Kraken spot or Alpaca equities; ported and unit-tested only |
| 2.3 | 22 | Non-Negative Rolled Price Series | ported verbatim and unit-tested; not applicable to these data | book_port_a.py: rolled_prices | ch2 | futures rolled price series: same; ported and unit-tested only |
| 2.4 | 23 | The Symmetric Cusum Filter | ported verbatim; run on both venues' dollar bars | book_port_e.py: getTEvents | ch3 | compared with the project's CUSUM (book_stats): identical event sets |
| 3.1 | 26 | Daily Volatility Estimates | ported verbatim; run on both venues' dollar bars | book_port_e.py: getDailyVol | ch3 | differs from the project's daily vol only through the tie rule for a bar exactly one day earlier (see ch3 text) |
| 3.2 | 27 | Triple-Barrier Labeling Method | ported verbatim; run on both venues' dollar bars | book_port_e.py: applyPtSlOnT1 | ch3 |  |
| 3.3 | 29 | Getting The Time Of First Touch | ported verbatim; run on both venues' dollar bars | book_port_e.py: getEvents_3_3 | ch3 |  |
| 3.4 | 30 | Adding A Vertical Barrier | ported verbatim; run on both venues' dollar bars | book_port_e.py: addVerticalBarrier | ch3 |  |
| 3.5 | 30 | Labeling For Side And Size | ported verbatim; run on both venues' dollar bars | book_port_e.py: getBins_3_5 | ch3 |  |
| 3.6 | 31 | Expanding Getevents To Incorporate Meta-Labeling | ported verbatim; run on both venues' dollar bars | book_port_e.py: getEvents | ch3 | the printed body uses an undefined `inst['Close']`; the function's own `close` argument is used |
| 3.7 | 32 | Expanding Getbins To Incorporate Meta-Labeling | ported verbatim; run on both venues' dollar bars | book_port_e.py: getBins | ch3 |  |
| 3.8 | 34 | Dropping Under-Populated Labels | ported verbatim; run on both venues' dollar bars | book_port_a.py: dropLabels | ch3 | printed signature says minPtc, body uses minPct (NameError as printed) |
| 4.1 | 35 | Estimating The Uniqueness Of A Label | ported verbatim; run on both venues' dollar bars | book_port_e.py: mpNumCoEvents | ch4 |  |
| 4.2 | 37 | Estimating The Average Uniqueness Of A Label | ported verbatim; run on both venues' dollar bars | book_port_e.py: mpSampleTW, getTW | ch4 | compared with the project's average_uniqueness |
| 4.3 | 38 | Build An Indicator Matrix | ported verbatim; run on both venues' dollar bars | book_port_a.py: getIndMatrix | ch4 |  |
| 4.4 | 39 | Compute Average Uniqueness | ported verbatim; run on both venues' dollar bars | book_port_a.py: getAvgUniqueness | ch4 |  |
| 4.5 | 39 | Return Sample From Sequential Bootstrap | ported verbatim; run on both venues' dollar bars | book_port_a.py: seqBootstrap | ch4 |  |
| 4.6 | 40 | Example Of Sequential Bootstrap | ported verbatim; run on both venues' dollar bars | book_port_a.py: example_seq_bootstrap | ch4 | the book's own 3-label example, in the unit tests |
| 4.7 | 40 | Generating A Random T1 Series | ported verbatim; run on both venues' dollar bars | book_port_a.py: getRndT1 | ch4 |  |
| 4.8 | 41 | Uniqueness From Standard And Sequential Bootstraps | ported verbatim; run on both venues' dollar bars | book_port_a.py: auxMC | ch4 |  |
| 4.9 | 41 | Multi-Threaded Monte Carlo | ported verbatim; run on both venues' dollar bars | book_port_a.py: mainMC | ch4 | iterations cut from the printed 1,000,000 (the number is stated in the section) |
| 4.10 | 43 | Determination Of Sample Weight By Absolute Return Attribution | ported verbatim; run on both venues' dollar bars | book_port_e.py: mpSampleW, getSampleW | ch4 |  |
| 4.11 | 44 | Implementation Of Time-Decay Factors | ported verbatim; run on both venues' dollar bars | book_port_e.py: getTimeDecay | ch4 |  |
| 5.1 | 48 | Weighting Function | ported verbatim; run on both venues' dollar bars | book_port_a.py: getWeights | ch5 |  |
| 5.2 | 50 | Standard Fracdiff (Expanding Window) | ported verbatim; run on both venues' dollar bars | book_port_a.py: fracDiff | ch5 | expanding window, compared with FFD on real bar series |
| 5.3 | 51 | The New Fixed-Width Window Fracdiff Method | ported with a stated substitution; run on both venues' dollar bars | book_port_e.py: fracDiff_FFD, getWeights_FFD | ch5 | the printed code calls getWeights_FFD, whose text is not in the extract: written from the book's definition |
| 5.4 | 53 | Finding The Minimum D Value That Passes The Adf Test | ported with a stated substitution; run on both venues' dollar bars | book_port_e.py: minFFD_table, adfuller_c1, adf_crit5_mackinnon | ch5 | statsmodels is not installable here: the same ADF regression written directly; MacKinnon response-surface constants are from memory, checked only against the asymptotic value and the project's own ADF |
| 6.1 | 58 | Accuracy Of The Bagging Classifier | ported verbatim; run on both venues' dollar bars | book_port_a.py: bagging_accuracy | ch6 |  |
| 6.2 | 60 | Three Ways Of Setting Up An Rf | ported verbatim; run on both venues' dollar bars | book_port_a.py: three_rf_setups | ch6 | all three setups (bootstrap RF, subsample RF, bagged tree with max_samples = average uniqueness) |
| 7.1 | 63 | Purging Observation In The Training Set | ported verbatim; run on both venues' dollar bars | book_port_b.py: getTrainTimes | ch7 |  |
| 7.2 | 65 | Embargo On Training Observations | ported verbatim; run on both venues' dollar bars | book_port_b.py: getEmbargoTimes, embargo_usage | ch7 |  |
| 7.3 | 66 | Cross-Validation Class When Observations Overlap | ported verbatim; run on both venues' dollar bars | book_port_b.py: PurgedKFold | ch7 | `self.t1[test_indices]` -> `.iloc` (pandas 3 treats integers as labels) |
| 7.4 | 67 | Using The Purgedkfold Class | ported verbatim; run on both venues' dollar bars | book_port_b.py: cvScore | ch7 | used by 8.x and 9.x |
| 8.2 | 68 | Mdi Feature Importance | ported verbatim; run on both venues' dollar bars | book_port_b.py: featImpMDI | ch8 |  |
| 8.3 | 69 | Mda Feature Importance | ported verbatim; run on both venues' dollar bars | book_port_b.py: featImpMDA | ch8 |  |
| 8.4 | 70 | Implementation Of Sfi | ported verbatim; run on both venues' dollar bars | book_port_b.py: auxFeatImpSFI | ch8 |  |
| 8.5 | 71 | Computation Of Orthogonal Features | ported verbatim; run on both venues' dollar bars | book_port_b.py: get_eVec, orthoFeats | ch8 |  |
| 8.6 | 73 | Computation Of Weighted Kendall’S Tau Between Feature Importance And I | ported verbatim; run on both venues' dollar bars | book_port_b.py: weighted_tau_example | ch8 |  |
| 8.7 | 74 | Creating A Synthetic Dataset | ported verbatim; run on both venues' dollar bars | book_port_b.py: getTestData | ch8 | the printed generator returns rows grouped by cluster (recorded in the section text and in a unit test) |
| 8.8 | 75 | Calling Feature Importance For Any Method | ported verbatim; run on both venues' dollar bars | book_port_b.py: featImportance | ch8 | trees cut from the printed 1,000 (the number is stated in the section); max_samples as an integer (scikit-learn >= 1.7) |
| 8.9 | 75 | Calling All Components | ported verbatim; run on both venues' dollar bars | book_port_b.py: testFunc | ch8 |  |
| 8.10 | 76 | Feature Importance Plotting Function | ported verbatim; run writes tables, not the plot | book_port_b.py: plotFeatImportance | ch8 | ported and imported; the run writes the importance tables (CSV), it does not call the plotting function |
| 9.1 | 79 | Grid Search With Purged K-Fold Cross-Validation | ported verbatim; run on both venues' dollar bars | book_port_b.py: clfHyperFit | ch9 | grid search |
| 9.2 | 80 | An Enhanced Pipeline Class | ported verbatim; run on both venues' dollar bars | book_port_b.py: MyPipeline | ch9 |  |
| 9.3 | 80 | Randomized Search With Purged K-Fold Cv | ported verbatim; run on both venues' dollar bars | book_port_b.py: clfHyperFit | ch9 | randomized search (same function, rndSearchIter > 0); draws cut from the book's |
| 9.4 | 82 | The Loguniform_Gen Class | ported verbatim; run on both venues' dollar bars | book_port_b.py: logUniform_gen, logUniform, test_logUniform | ch9 |  |
| 10.1 | 87 | From Probabilities To Bet Size | ported verbatim; run on both venues' dollar bars | book_port_c.py: getSignal | ch10 |  |
| 10.2 | 88 | Bets Are Averaged As Long As They Are Still Active | ported verbatim; run on both venues' dollar bars | book_port_c.py: avgActiveSignals, mpAvgActiveSignals | ch10 | the averaging is compared with the project's bet_averaging.py |
| 10.3 | 89 | Size Discretization To Prevent Overtrading | ported verbatim; run on both venues' dollar bars | book_port_c.py: discreteSignal | ch10 |  |
| 10.4 | 91 | Dynamic Position Size And Limit Price | ported verbatim; run on both venues' dollar bars | book_port_c.py: betSize, getTPos, invPrice, limitPrice, getW, main_10_4 | ch10 |  |
| 13.1 | 103 | Python Code For The Determination Of Optimal Trading Rules | ported verbatim; run on both venues' dollar bars | book_port_c.py: main_13_1 | ch13 | iterations cut from the printed values (stated in the section) |
| 13.2 | 103 | Python Code For The Determination Of Optimal Trading Rules | ported verbatim; run on both venues' dollar bars | book_port_c.py: batch | ch13 | the printed layout puts three lines outside the node loop; inside it here (listed in the file header) |
| 14.1 | 120 | Deriving The Timing Of Bets From A Series Of Target Positions | ported verbatim; run on both venues' dollar bars | book_port_c.py: getBets | ch14 |  |
| 14.2 | 120 | Implementation Of A Holding Period Estimator | ported verbatim; run on both venues' dollar bars | book_port_c.py: getHoldingPeriod | ch14 |  |
| 14.3 | 124 | Algorithm For Deriving Hhi Concentration | ported verbatim; run on both venues' dollar bars | book_port_c.py: getHHI, getHHIs | ch14 |  |
| 14.4 | 124 | Deriving The Sequence Of Dd And Tuw | ported verbatim; run on both venues' dollar bars | book_port_c.py: computeDD_TuW | ch14 |  |
| 15.1 | 131 | Targeting A Sharpe Ratio As A Function Of The Number Of Bets | ported verbatim; run on both venues' dollar bars | book_port_c.py: sharpe_of_binomial_draws | ch15 | draws cut from the printed 1,000,000 where stated |
| 15.2 | 132 | Using The Sympy Library For Symbolic Operations | ported with a stated substitution; run on both venues' dollar bars | book_port_c.py: factor_variance_check | ch15 | SymPy is not installable here: the identity the snippet factors is verified numerically on a grid |
| 15.3 | 132 | Computing The Implied Precision | ported with a stated substitution; run on both venues' dollar bars | book_port_c.py: binHR | ch15 | binSR is called by the printed code but its text is not in the extract: Equation 56 from the same chapter |
| 15.4 | 133 | Computing The Implied Betting Frequency | ported with a stated substitution; run on both venues' dollar bars | book_port_c.py: binFreq, binSR | ch15 | same |
| 15.5 | 135 | Calculating The Strategy Risk In Practice | ported verbatim; run on both venues' dollar bars | book_port_c.py: mixGaussians, probFailure, main_15_5 | ch15 | as printed the failure probability uses a standard deviation of p(1-p); the section text says so |
| 16.1 | 141 | Tree Clustering Using Scipy Functionality | ported verbatim; run on both venues' dollar bars | book_port_d.py: treeCluster, getIVP, getClusterVar | ch16 |  |
| 16.2 | 142 | Quasi-Diagonalization | ported verbatim; run on both venues' dollar bars | book_port_d.py: getQuasiDiag | ch16 |  |
| 16.3 | 143 | Recursive Bisection | ported verbatim; run on both venues' dollar bars | book_port_d.py: getRecBipart | ch16 |  |
| 16.4 | 151 | Full Implementation Of The Hrp Algorithm | ported verbatim; run on both venues' dollar bars | book_port_d.py: generateData, main_16_4, getHRP, correlDist, plotCorrMatrix | ch16 | Table 16.1 reproduced exactly with the draw read from the Figure 16.5 caption |
| 16.5 | 153 | Monte Carlo Experiment On Hrp Out-Of-Sample Performance | ported with a stated substitution; run on both venues' dollar bars | book_port_d.py: hrpMC, generateData_MC, getCLA, minVarLongOnly | ch16 | the CLA module is not in the extract: the long-only minimum-variance portfolio (the CLA end point) is solved by own code; the Monte Carlo's iterations are cut from the printed number |
| 17.1 | 164 | Sadf’S Inner Loop | ported verbatim; run on both venues' dollar bars | book_port_c.py: get_bsadf | ch17_19 | recomputed values equal the stored SADF feature |
| 17.2 | 164 | Preparing The Datasets | ported verbatim; run on both venues' dollar bars | book_port_c.py: getYX | ch17_19 |  |
| 17.3 | 165 | Apply Lags To Dataframe | ported verbatim; run on both venues' dollar bars | book_port_c.py: lagDF | ch17_19 |  |
| 17.4 | 165 | Fitting The Adf Specification | ported verbatim; run on both venues' dollar bars | book_port_c.py: getBetas | ch17_19 |  |
| 18.1 | 168 | Plug-In Entropy Estimator | ported verbatim; run on both venues' dollar bars | book_port_c.py: plugIn, pmf1 | ch17_19 | recomputed values equal the stored plug-in entropy |
| 18.2 | 169 | A Library Built Using The Lz Algorithm | ported verbatim; run on both venues' dollar bars | book_port_c.py: lempelZiv_lib | ch17_19 |  |
| 18.3 | 170 | Function That Computes The Length Of The Longest Match | ported verbatim; run on both venues' dollar bars | book_port_c.py: matchLength | ch17_19 |  |
| 18.4 | 172 | Implementation Of Algorithms Discussed In Gao Et Al. [2008] | ported verbatim; run on both venues' dollar bars | book_port_c.py: konto | ch17_19 | recomputed values equal the stored Kontoyiannis estimate |
| 19.1 | 180 | Implementation Of The Corwin-Schultz Algorithm | ported verbatim; run on both venues' dollar bars | book_port_e.py: corwinSchultz, getBeta, getGamma, getAlpha | ch17_19 | equals the project's implementation to 1e-11 |
| 19.2 | 181 | Estimating Volatility For High-Low Prices | ported verbatim; run on both venues' dollar bars | book_port_e.py: getSigma | ch17_19 |  |
| 20.1 | 187 | Un-Vectorized Cartesian Product | ported verbatim; run on both venues' dollar bars | book_mp.py: cartesian_unvectorized | ch20 |  |
| 20.2 | 187 | Vectorized Cartesian Product | ported verbatim; run on both venues' dollar bars | book_mp.py: cartesian_vectorized | ch20 |  |
| 20.3 | 188 | Single-Thread Implementation Of A One-Touch Double Barrier | ported verbatim; run on both venues' dollar bars | book_mp.py: barrier_touch_as_printed, barrierTouch, barrier_touch_vectorised | ch20 | as printed the loop records the LAST touch, not the first; both versions are kept |
| 20.4 | 188 | Multiprocessing Implementation Of A One-Touch Double Barrier | ported verbatim; run on both venues' dollar bars | book_mp.py: main1 | ch20 | the printed code slices columns using the row count; the path count is used |
| 20.5 | 189 | The Linpartsfunction | ported verbatim; run on both venues' dollar bars | book_mp.py: linParts | ch20 |  |
| 20.6 | 191 | The Nestedpartsfunction | ported verbatim; run on both venues' dollar bars | book_mp.py: nestedParts | ch20 |  |
| 20.7 | 193 | The Mppandasobj, Used At Various Points In The Book | ported verbatim; run on both venues' dollar bars | book_mp.py: mpPandasObj | ch4, ch10 (used throughout) | the printed code uses an undefined name argList; pdObj is used |
| 20.8 | 194 | Single-Thread Execution, For Debugging | ported verbatim; run on both venues' dollar bars | book_mp.py: processJobs_ | ch20 | single-thread debugging path |
| 20.9 | 195 | Example Of Asynchronous Call To Python | ported verbatim; run on both venues' dollar bars | book_mp.py: processJobs, reportProgress | ch4, ch10 | asynchronous pool call with progress report |
| 20.10 | 195 | Passing The Job (Molecule) To The Callback Function | ported verbatim; run on both venues' dollar bars | book_mp.py: expandCall | ch4, ch10 |  |
| 20.11 | 196 | Place This Code At The Beginning Of Your Engine | ported; a Python 2 recipe that Python 3 does not need | book_mp.py: _pickle_method, _unpickle_method | - | the Python 2 copy_reg recipe is unnecessary in Python 3 (bound methods pickle); registered so the snippet is present, not exercised by a run |
| 20.12 | 197 | Enhancing Processjobsto Perform On-The-Fly Output Reduction | ported verbatim; run on both venues' dollar bars | book_mp.py: processJobsRedux | ch20 |  |
| 20.13 | 198 | Enhancing Mppandasobj To Perform On-The-Fly Output Reduction | ported verbatim; run on both venues' dollar bars | book_mp.py: mpJobList | ch20 |  |
| 20.14 | 199 | Principal Components For A Subset Of The Columns | ported verbatim; run on both venues' dollar bars | book_mp.py: getPCs | ch20 |  |
| 21.1 | 201 | Partitions Of K Objects Into N Slots | ported verbatim; run on both venues' dollar bars | book_port_d.py: pigeonHole | ch21 |  |
| 21.2 | 202 | Set 𝛀 Of All Vectors Associated With All Partitions | ported verbatim; run on both venues' dollar bars | book_port_d.py: getAllWeights | ch21 | as printed a zero entry is multiplied by both signs, so columns repeat (3 assets: 80 columns, 38 distinct); both counts are in the section text |
| 21.3 | 203 | Evaluating All Trajectories | ported verbatim; run on both venues' dollar bars | book_port_d.py: evalTCosts, evalSR, dynOptPort | ch21 |  |
| 21.4 | 204 | Produce A Random Matrix Of A Given Rank | ported verbatim; run on both venues' dollar bars | book_port_d.py: rndMatWithRank | ch21 |  |
| 21.5 | 205 | Generate The Problem’S Parameters | ported verbatim; run on both venues' dollar bars | book_port_d.py: genMean, genParams | ch21 |  |
| 21.6 | 205 | Compute And Evaluate The Static Solution | ported verbatim; run on both venues' dollar bars | book_port_d.py: statOptPortf, staticSolution | ch21 |  |
| 21.7 | 206 | Compute And Evaluate The Dynamic Solution | ported verbatim; run on both venues' dollar bars | book_port_d.py: dynamicSolution | ch21 |  |
