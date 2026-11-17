# AFML ports vs BlackArbsCEO/Adv_Fin_ML_Exercises (2026-11-16)

User: "make sure your stuff matches this /Users/mike/Documents/GitHub/Adv_Fin_ML_Exercises".
Compared: tools/book_study/book_port_a.py, book_port_b.py, book_port_e.py, book_mp.py against
that repo's src/features/snippets.py and the function definitions in its notebooks (commit b1f980e).
Method: compare_with_blackarbs.py parses both sides, strips comments/docstrings/formatting (ast.unparse)
and diffs each function present in both. Referee for every difference: the AFML PDF text (/tmp/afml.txt line numbers).
The BlackArbs repo is a third-party transcription, not the book; its code is not copied here.

39 functions in common. 13 identical: _unpickle_method, expandCall, featImpMDI, getAvgUniqueness, getBins,
getWeights, getWeights_FFD, get_eVec, linParts, nestedParts, processJobs, processJobs_, reportProgress.
(getWeights_FFD is not in the PDF extract; the port matches BlackArbs' transcription exactly.)

## Where the port matches the book and BlackArbs does not (their own edits)
getTEvents (they use log-abs diff + tqdm; book p.782: gRaw.diff()), getDailyVol (they add try/except, rename),
fracDiff / fracDiff_FFD (they add try/except, resample, .any()), cvScore (they add pd.IndexSlice; not in book),
featImpMDA (they raise ValueError, typo pd.SEries, and dedent the permutation loop out of the fold loop; book p.2300/2370),
orthoFeats (they rename dfX->dfx; book p.2428 dfX), getTestData (typo n_feautres), plotFeatImportance (lw/ls/'k'; book p.2587/2591 linewidth/linestyle/'black').

## Where the port differs from the book because the printed code fails on Python 3 / pandas 2 (forced; open items, rule 6)
iteritems -> items (getIndMatrix, getTrainTimes, mpNumCoEvents, mpSampleTW, mpSampleW, applyPtSlOnT1);
Series.append -> pd.concat (getEmbargoTimes, mpPandasObj); im_func/im_self/im_class -> __func__/__self__/type (_pickle_method);
pd.DatetimeIndex(periods=,end=pd.datetime.today()) -> pd.bdate_range (getTestData; book p.2481);
self.t1[test_indices] -> self.t1.iloc[test_indices] (PurgedKFold; book p.2281);
argmin -> idxmin (dropLabels; pandas >=1 argmin returns a position, the book's code means the label);
out['method',...] -> out[['method',...]] (testFunc; as printed raises KeyError, book p.2571);
fillna(method='ffill') -> ffill() (fracDiff, fracDiff_FFD); print statement -> function; izip -> zip.

## Where the port differs from the book and the change was NOT forced (Claude's additions; to take out on the user's yes)
1. dropLabels: added `verbose=False` and made the print conditional (book p.1028-1034 prints always).
2. getTimeDecay: returns (clfW, const, slope); book p.1384-1385 prints const,slope and returns clfW.
3. applyPtSlOnT1: added out['sl']=pd.NaT and out['pt']=pd.NaT (not in book).
4. getIndMatrix: loop variable renamed t1 -> t1_ (book p.1133 rebinds t1; works as printed).
5. testFunc: added pathOut, numThreads, plot arguments, os.makedirs, `return out` (book p.2538-2574: return nothing, writes stats.csv).
6. plotFeatImportance: 'tag='+str(tag) (book p.2592 'tag='+tag; works for the book's own call with tag='testFunc').
7. featImpMDA: np.random.permutation in place of np.random.shuffle(X1_[j].values) (book p.2370). Whether shuffle still
   works in place on the installed pandas has to be tested before deciding forced or not.
8. dtype=float added to empty pd.Series/DataFrame constructors (mpSampleTW, mpSampleW, seqBootstrap, auxFeatImpSFI,
   featImpMDA, fracDiff, fracDiff_FFD, mpPandasObj, applyPtSlOnT1): pandas 2 runs the printed form; not forced.
9. addVerticalBarrier: the book prints only the three body lines (p.926), no def; the port wraps them as
   addVerticalBarrier(close, tEvents, numDays). A wrapper is needed to call it; the argument order is Claude's.
10. getEvents: calls MP.mpPandasObj (module prefix) where the book calls mpPandasObj; import layout only.
Nothing changed yet; each item waits for the user's ruling, one at a time.
