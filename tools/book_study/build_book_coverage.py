#!/usr/bin/env python
"""Writes BOOK_COVERAGE.md (the 97 numbered snippets of Chapters 2-21 of the user's AFML extract) from the verbatim ports.

Every row names the module and function(s) that hold the snippet; this script imports the modules and stops if a named function does not exist.
Statuses are about the PORT and where it was RUN; they say nothing about whether a method works.  The run folders and checker results are passed in, not guessed.

    python tools/book_study/build_book_coverage.py --old <previous BOOK_COVERAGE.md> --out <new BOOK_COVERAGE.md> \
        --run kraken=<folder> --run alpaca=<folder> [--checked kraken="177 of 177" ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import book_mp, book_port_a, book_port_b, book_port_c, book_port_d, book_port_e  # noqa: E401,E402

MOD = {"mp": book_mp, "a": book_port_a, "b": book_port_b, "c": book_port_c, "d": book_port_d, "e": book_port_e}
RUN, NA, SUB, TAB, PY2 = "RUN", "NA", "SUB", "TAB", "PY2"

# id: (category, module letter, [functions], section of run_book_ports_bars.py, note)
M = {
    "2.1": (RUN, "a", ["pcaWeights"], "ch2", "applied to the cross-section of daily bar returns"),
    "2.2": (NA, "a", ["rollGaps", "getRolledSeries", "getRolledSeries_df"], "ch2", "futures roll gaps: no futures on Kraken spot or Alpaca equities; ported and unit-tested only"),
    "2.3": (NA, "a", ["rolled_prices"], "ch2", "futures rolled price series: same; ported and unit-tested only"),
    "2.4": (RUN, "e", ["getTEvents"], "ch3", "compared with the project's CUSUM (book_stats): identical event sets"),
    "3.1": (RUN, "e", ["getDailyVol"], "ch3", "differs from the project's daily vol only through the tie rule for a bar exactly one day earlier (see ch3 text)"),
    "3.2": (RUN, "e", ["applyPtSlOnT1"], "ch3", ""),
    "3.3": (RUN, "e", ["getEvents_3_3"], "ch3", ""),
    "3.4": (RUN, "e", ["addVerticalBarrier"], "ch3", ""),
    "3.5": (RUN, "e", ["getBins_3_5"], "ch3", ""),
    "3.6": (RUN, "e", ["getEvents"], "ch3", "the printed body uses an undefined `inst['Close']`; the function's own `close` argument is used"),
    "3.7": (RUN, "e", ["getBins"], "ch3", ""),
    "3.8": (RUN, "a", ["dropLabels"], "ch3", "printed signature says minPtc, body uses minPct (NameError as printed)"),
    "4.1": (RUN, "e", ["mpNumCoEvents"], "ch4", ""),
    "4.2": (RUN, "e", ["mpSampleTW", "getTW"], "ch4", "compared with the project's average_uniqueness"),
    "4.3": (RUN, "a", ["getIndMatrix"], "ch4", ""),
    "4.4": (RUN, "a", ["getAvgUniqueness"], "ch4", ""),
    "4.5": (RUN, "a", ["seqBootstrap"], "ch4", ""),
    "4.6": (RUN, "a", ["example_seq_bootstrap"], "ch4", "the book's own 3-label example, in the unit tests"),
    "4.7": (RUN, "a", ["getRndT1"], "ch4", ""),
    "4.8": (RUN, "a", ["auxMC"], "ch4", ""),
    "4.9": (RUN, "a", ["mainMC"], "ch4", "iterations cut from the printed 1,000,000 (the number is stated in the section)"),
    "4.10": (RUN, "e", ["mpSampleW", "getSampleW"], "ch4", ""),
    "4.11": (RUN, "e", ["getTimeDecay"], "ch4", ""),
    "5.1": (RUN, "a", ["getWeights"], "ch5", ""),
    "5.2": (RUN, "a", ["fracDiff"], "ch5", "expanding window, compared with FFD on real bar series"),
    "5.3": (SUB, "e", ["fracDiff_FFD", "getWeights_FFD"], "ch5", "the printed code calls getWeights_FFD, whose text is not in the extract: written from the book's definition"),
    "5.4": (SUB, "e", ["minFFD_table", "adfuller_c1", "adf_crit5_mackinnon"], "ch5", "statsmodels is not installable here: the same ADF regression written directly; MacKinnon response-surface constants are from memory, checked only against the asymptotic value and the project's own ADF"),
    "6.1": (RUN, "a", ["bagging_accuracy"], "ch6", ""),
    "6.2": (RUN, "a", ["three_rf_setups"], "ch6", "all three setups (bootstrap RF, subsample RF, bagged tree with max_samples = average uniqueness)"),
    "7.1": (RUN, "b", ["getTrainTimes"], "ch7", ""),
    "7.2": (RUN, "b", ["getEmbargoTimes", "embargo_usage"], "ch7", ""),
    "7.3": (RUN, "b", ["PurgedKFold"], "ch7", "`self.t1[test_indices]` -> `.iloc` (pandas 3 treats integers as labels)"),
    "7.4": (RUN, "b", ["cvScore"], "ch7", "used by 8.x and 9.x"),
    "8.2": (RUN, "b", ["featImpMDI"], "ch8", ""),
    "8.3": (RUN, "b", ["featImpMDA"], "ch8", ""),
    "8.4": (RUN, "b", ["auxFeatImpSFI"], "ch8", ""),
    "8.5": (RUN, "b", ["get_eVec", "orthoFeats"], "ch8", ""),
    "8.6": (RUN, "b", ["weighted_tau_example"], "ch8", ""),
    "8.7": (RUN, "b", ["getTestData"], "ch8", "the printed generator returns rows grouped by cluster (recorded in the section text and in a unit test)"),
    "8.8": (RUN, "b", ["featImportance"], "ch8", "trees cut from the printed 1,000 (the number is stated in the section); max_samples as an integer (scikit-learn >= 1.7)"),
    "8.9": (RUN, "b", ["testFunc"], "ch8", ""),
    "8.10": (TAB, "b", ["plotFeatImportance"], "ch8", "ported and imported; the run writes the importance tables (CSV), it does not call the plotting function"),
    "9.1": (RUN, "b", ["clfHyperFit"], "ch9", "grid search"),
    "9.2": (RUN, "b", ["MyPipeline"], "ch9", ""),
    "9.3": (RUN, "b", ["clfHyperFit"], "ch9", "randomized search (same function, rndSearchIter > 0); draws cut from the book's"),
    "9.4": (RUN, "b", ["logUniform_gen", "logUniform", "test_logUniform"], "ch9", ""),
    "10.1": (RUN, "c", ["getSignal"], "ch10", ""),
    "10.2": (RUN, "c", ["avgActiveSignals", "mpAvgActiveSignals"], "ch10", "the averaging is compared with the project's bet_averaging.py"),
    "10.3": (RUN, "c", ["discreteSignal"], "ch10", ""),
    "10.4": (RUN, "c", ["betSize", "getTPos", "invPrice", "limitPrice", "getW", "main_10_4"], "ch10", ""),
    "13.1": (RUN, "c", ["main_13_1"], "ch13", "iterations cut from the printed values (stated in the section)"),
    "13.2": (RUN, "c", ["batch"], "ch13", "the printed layout puts three lines outside the node loop; inside it here (listed in the file header)"),
    "14.1": (RUN, "c", ["getBets"], "ch14", ""),
    "14.2": (RUN, "c", ["getHoldingPeriod"], "ch14", ""),
    "14.3": (RUN, "c", ["getHHI", "getHHIs"], "ch14", ""),
    "14.4": (RUN, "c", ["computeDD_TuW"], "ch14", ""),
    "15.1": (RUN, "c", ["sharpe_of_binomial_draws"], "ch15", "draws cut from the printed 1,000,000 where stated"),
    "15.2": (SUB, "c", ["factor_variance_check"], "ch15", "SymPy is not installable here: the identity the snippet factors is verified numerically on a grid"),
    "15.3": (SUB, "c", ["binHR"], "ch15", "binSR is called by the printed code but its text is not in the extract: Equation 56 from the same chapter"),
    "15.4": (SUB, "c", ["binFreq", "binSR"], "ch15", "same"),
    "15.5": (RUN, "c", ["mixGaussians", "probFailure", "main_15_5"], "ch15", "as printed the failure probability uses a standard deviation of p(1-p); the section text says so"),
    "16.1": (RUN, "d", ["treeCluster", "getIVP", "getClusterVar"], "ch16", ""),
    "16.2": (RUN, "d", ["getQuasiDiag"], "ch16", ""),
    "16.3": (RUN, "d", ["getRecBipart"], "ch16", ""),
    "16.4": (RUN, "d", ["generateData", "main_16_4", "getHRP", "correlDist", "plotCorrMatrix"], "ch16", "Table 16.1 reproduced exactly with the draw read from the Figure 16.5 caption"),
    "16.5": (SUB, "d", ["hrpMC", "generateData_MC", "getCLA", "minVarLongOnly"], "ch16", "the CLA module is not in the extract: the long-only minimum-variance portfolio (the CLA end point) is solved by own code; the Monte Carlo's iterations are cut from the printed number"),
    "17.1": (RUN, "c", ["get_bsadf"], "ch17_19", "recomputed values equal the stored SADF feature"),
    "17.2": (RUN, "c", ["getYX"], "ch17_19", ""),
    "17.3": (RUN, "c", ["lagDF"], "ch17_19", ""),
    "17.4": (RUN, "c", ["getBetas"], "ch17_19", ""),
    "18.1": (RUN, "c", ["plugIn", "pmf1"], "ch17_19", "recomputed values equal the stored plug-in entropy"),
    "18.2": (RUN, "c", ["lempelZiv_lib"], "ch17_19", ""),
    "18.3": (RUN, "c", ["matchLength"], "ch17_19", ""),
    "18.4": (RUN, "c", ["konto"], "ch17_19", "recomputed values equal the stored Kontoyiannis estimate"),
    "19.1": (RUN, "e", ["corwinSchultz", "getBeta", "getGamma", "getAlpha"], "ch17_19", "equals the project's implementation to 1e-11"),
    "19.2": (RUN, "e", ["getSigma"], "ch17_19", ""),
    "20.1": (RUN, "mp", ["cartesian_unvectorized"], "ch20", ""),
    "20.2": (RUN, "mp", ["cartesian_vectorized"], "ch20", ""),
    "20.3": (RUN, "mp", ["barrier_touch_as_printed", "barrierTouch", "barrier_touch_vectorised"], "ch20", "as printed the loop records the LAST touch, not the first; both versions are kept"),
    "20.4": (RUN, "mp", ["main1"], "ch20", "the printed code slices columns using the row count; the path count is used"),
    "20.5": (RUN, "mp", ["linParts"], "ch20", ""),
    "20.6": (RUN, "mp", ["nestedParts"], "ch20", ""),
    "20.7": (RUN, "mp", ["mpPandasObj"], "ch4, ch10 (used throughout)", "the printed code uses an undefined name argList; pdObj is used"),
    "20.8": (RUN, "mp", ["processJobs_"], "ch20", "single-thread debugging path"),
    "20.9": (RUN, "mp", ["processJobs", "reportProgress"], "ch4, ch10", "asynchronous pool call with progress report"),
    "20.10": (RUN, "mp", ["expandCall"], "ch4, ch10", ""),
    "20.11": (PY2, "mp", ["_pickle_method", "_unpickle_method"], "-", "the Python 2 copy_reg recipe is unnecessary in Python 3 (bound methods pickle); registered so the snippet is present, not exercised by a run"),
    "20.12": (RUN, "mp", ["processJobsRedux"], "ch20", ""),
    "20.13": (RUN, "mp", ["mpJobList"], "ch20", ""),
    "20.14": (RUN, "mp", ["getPCs"], "ch20", ""),
    "21.1": (RUN, "d", ["pigeonHole"], "ch21", ""),
    "21.2": (RUN, "d", ["getAllWeights"], "ch21", "as printed a zero entry is multiplied by both signs, so columns repeat (3 assets: 80 columns, 38 distinct); both counts are in the section text"),
    "21.3": (RUN, "d", ["evalTCosts", "evalSR", "dynOptPort"], "ch21", ""),
    "21.4": (RUN, "d", ["rndMatWithRank"], "ch21", ""),
    "21.5": (RUN, "d", ["genMean", "genParams"], "ch21", ""),
    "21.6": (RUN, "d", ["statOptPortf", "staticSolution"], "ch21", ""),
    "21.7": (RUN, "d", ["dynamicSolution"], "ch21", ""),
}
LABEL = {RUN: "ported verbatim; run on both venues' dollar bars", NA: "ported verbatim and unit-tested; not applicable to these data",
         SUB: "ported with a stated substitution; run on both venues' dollar bars", TAB: "ported verbatim; run writes tables, not the plot", PY2: "ported; a Python 2 recipe that Python 3 does not need"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[], help="venue=run folder")
    ap.add_argument("--checked", action="append", default=[], help='venue="177 of 177"')
    a = ap.parse_args()
    old = Path(a.old).read_text(encoding="utf-8").splitlines()
    rows = {}
    for line in old:
        m = re.match(r"\| (\d+\.\d+) \| (\d+) \| (.*?) \| (.*?) \| (.*) \|$", line)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(3), m.group(4))
    assert len(rows) == 97 and set(rows) == set(M), (len(rows), set(rows) ^ set(M))
    for k, (cat, mod, funcs, sec, note) in M.items():
        for f in funcs:
            if not hasattr(MOD[mod], f):
                raise SystemExit(f"snippet {k}: {MOD[mod].__name__} has no {f}")
    cnt = {c: sum(1 for v in M.values() if v[0] == c) for c in LABEL}
    runs = dict(r.split("=", 1) for r in a.run)
    chk = dict(r.split("=", 1) for r in a.checked)
    L = ["# Book coverage ledger: AFML extract, every numbered snippet (Ch.2-21)", "",
         "Source: the 218-page extract the user supplied. Every snippet below is a port of the printed code in `tools/book_study/book_mp.py` and `book_port_a.py` ... `book_port_e.py` (the file headers list every change from the printed text), "
         "unit-tested in `test_book_port_*.py`, and applied to each venue's dollar bars by `run_book_ports_bars.py` (section named in the last column). This replaces the previous ledger, whose 'built, in bar runs' rows rested on the project's earlier "
         "implementations written from memory; those were compared with the book code (differences are listed in the sections' text).", ""]
    for v, p in runs.items():
        meta = json.loads((Path(p) / "summary_numbers.json").read_text(encoding="utf-8"))["meta"]
        L.append(f"- {v.capitalize()} run: `{Path(p).parent.name}/{Path(p).name}`; {meta['events']:,} events; seconds per section in `summary_numbers.json`; errors: {meta['errors'] or 'none'}; "
                 f"independent checker (verify_book_ports_bars.py): {chk.get(v, 'not recorded')}.")
    L += ["", "Counts (97 snippets): " + "; ".join(f"{n} {LABEL[c]}" for c, n in cnt.items() if n) + ".", "",
          "Not a snippet, in the text, NOT built: Ch.19 Kyle / Amihud / Hasbrouck lambdas, VPIN and PIN (need trade-level data; the bars carry no trade flags). The real CLA code of Ch.16 is not in the extract (stand-in above). Chapter 22 is not in the extract.", "",
          "Run-size reductions against the printed examples (trees, Monte Carlo iterations, samples) are stated in each section of SUMMARY.md and in `summary_numbers.json` (`meta.config`).", "",
          "| Snippet | Page | Title | Status | Code | Run section | Note |", "|---|---|---|---|---|---|---|"]
    for k in sorted(rows, key=lambda s: tuple(int(x) for x in s.split("."))):
        cat, mod, funcs, sec, note = M[k]
        page, title, _ = rows[k]
        mname = MOD[mod].__name__ + ".py"
        L.append(f"| {k} | {page} | {title} | {LABEL[cat]} | {mname}: {', '.join(funcs)} | {sec} | {note} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"written {a.out}; {cnt}")


if __name__ == "__main__":
    main()
