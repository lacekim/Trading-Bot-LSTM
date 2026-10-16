"""Write-up items for Chapters 20, 21 and 22 of the book (multiprocessing, brute-force dynamic portfolios, HPC survey).  New file; reads saved data, changes nothing audited."""
from __future__ import annotations

import json
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import book_mp as MP
import book_port_d as D
import writeup_core as W
import writeup_panel as WP
from writeup_core import VENUES, SYM, item, tbl, fmt, save


def _ss(molecule, obj):
    """Squares of the rows of `obj` named in `molecule` (a function the Chapter 20 machinery can run in parallel; the concatenation over molecules does not depend on the split)."""
    return obj.loc[molecule] ** 2


def _job_sum(molecule, obj):
    return obj.loc[molecule].sum()


def ch20():
    out = []
    d0 = {"a": ["1", "2"], "b": ["+", "*"], "c": ["!", "@"]}
    u, v = MP.cartesian_unvectorized(d0), MP.cartesian_vectorized(d0)
    out.append(item("Snippet 20.1", "Un-vectorised Cartesian product of a dictionary of lists", "code", table=tbl(["#", "combination"], [[i + 1, str(x)] for i, x in enumerate(u)]),
                    text=f"book_mp.cartesian_unvectorized: {len(u)} combinations (the book's example dictionary)."))
    out.append(item("Snippet 20.2", "Vectorised Cartesian product (any number of keys)", "code", text=f"book_mp.cartesian_vectorized gives the same {len(v)} combinations: " + ("identical list to Snippet 20.1." if u == v else "DIFFERENT from Snippet 20.1.")))

    rng = np.random.RandomState(20)
    r = rng.normal(0, 0.01, size=(400, 2000))
    t0 = time.time(); printed = MP.barrier_touch_as_printed(r); t_printed = time.time() - t0
    t0 = time.time(); first = MP.barrierTouch(r); t_first = time.time() - t0
    vec = MP.barrier_touch_vectorised(r)
    differ = sum(1 for j in first if printed.get(j) != first[j])
    out.append(item("Snippet 20.3", "Single-thread one-touch double barrier on 2,000 simulated paths of 400 steps (width 0.5)", "code",
                    table=tbl(["Version", "Paths that touch", "Seconds", "Same as vectorised reference"], [["as printed (records the last touching row)", fmt(len(printed)), fmt(t_printed, 3), str(printed == vec)], ["earliest touch (break)", fmt(len(first)), fmt(t_first, 3), str(first == vec)]]),
                    text=f"The printed inner `continue` does not stop the scan, so it records the LAST row beyond the barrier; for {differ} of {len(first)} touching paths it differs from the earliest touch the comment describes.  Both are kept in book_mp."))
    rows = []
    t0 = time.time(); one = MP.main1(r, numThreads=1); t1 = time.time() - t0
    t0 = time.time(); two = MP.main1(r, numThreads=2); t2 = time.time() - t0
    rows.append(["1 process", fmt(t1, 3), str(one == first)]); rows.append(["2 processes", fmt(t2, 3), str(two == first)])
    out.append(item("Snippet 20.4", "Multiprocessing implementation of the same job (columns split across a pool)", "code", table=tbl(["Pool", "Seconds", "Same dictionary as the single-thread earliest touch"], rows),
                    text=f"This machine has {os.cpu_count()} cores; the book's example uses 24 threads.  The printed code splits the rows and then slices columns; the port splits the columns (the paths)."))

    parts = MP.linParts(20, 6)
    out.append(item("Snippet 20.5", "The linParts function: a linear partition of atoms into molecules", "code", table=tbl(["Molecule", "Atoms"], [[i, f"{parts[i - 1] + 1} to {parts[i]}"] for i in range(1, len(parts))]),
                    text=f"20 atoms, 6 molecules: boundaries {parts.tolist()} (np.ceil of the equally spaced points)."))
    fig, ax = plt.subplots(figsize=(6.4, 1.6))
    for i in range(1, len(parts)):
        ax.barh(0, parts[i] - parts[i - 1], left=parts[i - 1], color=str(0.25 + 0.1 * i), edgecolor="white")
        ax.text((parts[i] + parts[i - 1]) / 2, 0, f"M{i}", ha="center", va="center", color="white", fontsize=7)
    ax.set_xlabel("Task #"); ax.set_yticks([]); ax.set_xlim(0, 20)
    out.append(item("Figure 20.1", "A linear partition of 20 atomic tasks into 6 molecules", "formula", fig=save(fig, "f20_1"), note="Drawn from linParts(20, 6), the same call as the book's example."))

    N, M = 20, 6
    r_ = [0.0]
    rows = []
    for m in range(1, M + 1):
        r_.append((-1 + (1 + 4 * (r_[-1] ** 2 + r_[-1] + N * (N + 1) / M)) ** .5) / 2)
    np_parts = MP.nestedParts(N, M)
    rows = [[m, fmt(r_[m], 4), fmt(int(np.round(r_[m])))] for m in range(1, M + 1)]
    out.append(item("Equation 93", "First molecule boundary of a nested loop: r1 = (-1 + sqrt(1 + 4 N(N+1) M^-1)) / 2", "formula",
                    table=tbl(["m", "r_m (Eqs 93-95)", "rounded"], rows), text="N = 20 atoms, M = 6 molecules."))
    out.append(item("Equation 94", "Second boundary r2 = (-1 + sqrt(1 + 4 (r1^2 + r1 + N(N+1) M^-1))) / 2", "formula", text="Table under Equation 93."))
    out.append(item("Equation 95", "General boundary r_m = (-1 + sqrt(1 + 4 (r_(m-1)^2 + r_(m-1) + N(N+1) M^-1))) / 2", "formula",
                    text=f"book_mp.nestedParts(20, 6) = {np_parts.tolist()}; the recursion above rounded = {[int(np.round(x)) for x in r_]}: " + ("identical." if np_parts.tolist() == [int(np.round(x)) for x in r_] else "DIFFERENT.")))
    out.append(item("Snippet 20.6", "The nestedParts function: partition of atoms for a two-nested loop", "code",
                    table=tbl(["Variant", "Boundaries"], [["lower triangle", str(MP.nestedParts(20, 6).tolist())], ["upper triangle (heaviest rows first)", str(MP.nestedParts(20, 6, upperTriang=True).tolist())]])))
    parts = MP.nestedParts(20, 6)
    work = [sum(i for i in range(parts[k - 1] + 1, parts[k] + 1)) for k in range(1, len(parts))]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for k in range(1, len(parts)):
        for i in range(parts[k - 1], parts[k]):
            ax.barh(i, i + 1, color=str(0.15 + 0.12 * k), height=0.9)
    ax.invert_yaxis(); ax.set_xlabel("inner-loop length"); ax.set_ylabel("outer-loop row")
    out.append(item("Figure 20.2", "A two-nested loops partition of 20 atoms into 6 molecules (shading = molecule)", "formula", fig=save(fig, "f20_2"),
                    text=f"Work per molecule (sum of row lengths): {work} against a linear split's {[sum(range(MP.linParts(20, 6)[k - 1] + 1, MP.linParts(20, 6)[k] + 1)) for k in range(1, 7)]}: the nested split balances the load."))

    idx = pd.RangeIndex(0, 4000)
    obj = pd.Series(np.random.RandomState(7).normal(size=4000), index=idx)
    t0 = time.time(); a1 = MP.mpPandasObj(_ss, ("molecule", idx), numThreads=1, obj=obj); ta = time.time() - t0
    t0 = time.time(); a2 = MP.mpPandasObj(_ss, ("molecule", idx), numThreads=2, obj=obj); tb = time.time() - t0
    out.append(item("Snippet 20.7", "mpPandasObj: run a function over the molecules of a pandas object, in parallel, and concatenate the outputs", "code",
                    table=tbl(["numThreads", "Seconds", "Output rows", "Same as 1 thread"], [[1, fmt(ta, 3), fmt(len(a1)), "-"], [2, fmt(tb, 3), fmt(len(a2)), str(bool(np.array_equal(a1.sort_index().to_numpy(), a2.sort_index().to_numpy())))]]),
                    text="A job that squares 4,000 values; each molecule returns its rows, so the concatenated output must not depend on the number of processes."))
    mk = lambda: [{"func": _job_sum, "molecule": list(range(i, i + 4)), "obj": obj} for i in range(0, 16, 4)]  # expandCall removes the 'func' key, so every run gets fresh dicts
    jobs = mk()
    s1 = MP.processJobs_(mk())
    out.append(item("Snippet 20.8", "Single-thread execution, for debugging (processJobs_)", "code", text=f"{len(s1)} jobs run in order; first values {[round(float(x), 5) for x in s1[:3]]}."))
    out.append(item("Snippet 20.9", "Example of an asynchronous call to python's multiprocessing library (processJobs)", "code", text="book_mp.processJobs runs the jobs through pool.imap_unordered and reports progress to the screen; timed under Snippet 20.13 below."))
    out.append(item("Snippet 20.10", "Passing the job (molecule) to the callback function (expandCall)", "code", text="book_mp.expandCall pops the 'func' key and calls it with the remaining keyword arguments; used by every job above (it is how _ss received `molecule` and `obj`)."))
    out.append(item("Snippet 20.11", "Placing the pickling of methods at the beginning of the code", "code", text="Python 3 pickles bound methods itself, so the Python 2 copy_reg / im_func registration is not needed; book_mp keeps a copyreg registration for __func__ / __self__ so that the snippet exists."))
    t0 = time.time(); rd = MP.processJobsRedux(mk(), numThreads=2, redux=lambda a, b: a + b); tc = time.time() - t0
    out.append(item("Snippet 20.12", "Enhancing processJobs to perform on-the-fly output reduction (processJobsRedux)", "code",
                    text=f"book_mp.processJobsRedux adds each job's output to a running total as it arrives instead of keeping all outputs: {len(jobs)} jobs, 2 processes, {fmt(tc, 3)} s, total {rd:.6f}; sum of the single-thread outputs {float(np.sum(s1)):.6f}."))
    t0 = time.time(); p1 = MP.processJobs(mk(), numThreads=2); td = time.time() - t0
    mj = MP.mpJobList(_job_sum, ("molecule", list(range(16))), numThreads=2, redux=lambda a, b: a + b, obj=obj)
    out.append(item("Snippet 20.13", "Enhancing mpPandasObj to perform on-the-fly output reduction (mpJobList)", "code",
                    text=f"book_mp.mpJobList splits a list of arguments into molecules and reduces on the fly: sum over 16 values = {float(mj):.6f}; the same sum computed directly = {float(obj.loc[list(range(16))].sum()):.6f}.  (Snippet 20.9's processJobs on the 4 jobs above: {fmt(td, 3)} s, outputs {'identical to' if sorted(np.round(p1, 8).tolist()) == sorted(np.round([float(x) for x in s1], 8).tolist()) else 'DIFFERENT from'} the single-thread run.)"))
    # Equation 96 / Snippet 20.14: principal components by summing per-file products, on our own panel
    rows = []
    for vname in VENUES:
        pan = WP.panel(vname).tail(500).dropna(axis=1, how="any")
        cols = list(pan.columns[:24])
        pan = pan[cols].astype(float)
        cov = pan.cov()
        lam, V = np.linalg.eigh(cov.to_numpy())
        eVec = pd.DataFrame(V[:, ::-1][:, :5], index=cols, columns=[f"PC{i}" for i in range(1, 6)])
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for b in range(4):
                fn = f"block{b}.csv"
                pan[cols[b * 6:(b + 1) * 6]].to_csv(os.path.join(tmp, fn))
                files.append(fn)
            res = MP.mpJobList(MP.getPCs, ("molecules", files), numThreads=2, mpBatches=1, path=tmp + "/", eVec=eVec, redux=pd.DataFrame.add)
        full = pd.DataFrame(pan.to_numpy() @ eVec.to_numpy(), index=pan.index, columns=eVec.columns)
        rows.append([vname, len(cols), fmt(len(pan)), f"{float(np.abs(res.to_numpy() - full.to_numpy()).max()):.1e}"])
    out.append(item("Equation 96", "Principal components as a sum over blocks of columns, P = sum_b Z_b W~_b", "data",
                    table=tbl(["Venue", "Symbols (in 4 files)", "Days", "max |sum over files - one-shot product|"], rows),
                    text="Z_b is the block of 6 symbols' daily returns held in one file, W~_b the matching rows of the eigenvector matrix (top 5 components of the 24-symbol covariance)."))
    out.append(item("Snippet 20.14", "Principal components for a subset of the columns: getPCs loads one file at a time and adds the products", "code",
                    table=tbl(["Venue", "Symbols", "Days", "max |difference|"], rows),
                    text="book_mp.getPCs run through mpJobList with pd.DataFrame.add as the reduction on our 5-year daily return panels, 2 processes."))
    return out


def ch21():
    out = []
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.0), sharey=True)
    for ax, part in zip(axs, ((1, 2, 3), (3, 2, 1))):
        for i, u in enumerate(part):
            ax.bar(i, u, color="0.35")
        ax.set_xticks(range(3)); ax.set_xticklabels(["Asset 1", "Asset 2", "Asset 3"]); ax.set_title(f"partition {part}"); ax.set_ylabel("Units of capital")
    fig.tight_layout()
    out.append(item("Figure 21.1", "Partitions (1, 2, 3) and (3, 2, 1) must be treated as different", "diagram", fig=save(fig, "f21_1"), note="Drawn here from the caption; the same units of capital in different assets are different portfolios."))
    rows = [[k, n, fmt(len(list(D.pigeonHole(k, n))))] for k, n in ((3, 3), (4, 3), (5, 4), (10, 3))]
    out.append(item("Snippet 21.1", "Partitions of k objects into n slots (pigeonHole)", "code", table=tbl(["k", "n", "partitions"], rows), text="The count is C(k + n - 1, n - 1): 10 for (3, 3), as the book's example."))
    w = D.getAllWeights(3, 3)
    out.append(item("Snippet 21.2", "Set Omega of all vectors associated with all partitions (getAllWeights)", "code",
                    text=f"k = 3 units over n = 3 assets with signs: {w.shape[1]} columns of {w.shape[0]} weights.  As printed, a zero entry is multiplied by both signs, so identical columns repeat ({w.shape[1] - len({tuple(c) for c in w.T})} repeats); the port keeps it as printed."))
    np.random.seed(21)
    params = D.genParams(size=3, horizon=2, nSamples=1000, seed=21)
    w_stat, sr_stat = D.staticSolution(params)
    t0 = time.time(); w_dyn, sr_dyn = D.dynamicSolution(params); tdyn = time.time() - t0
    out.append(item("Snippet 21.3", "Evaluating all trajectories (evalTCosts, evalSR, dynOptPort)", "code",
                    text=f"Trajectories evaluated: {len(D.getAllWeights(3, 3).T) ** 2:,} (3 assets, 2 horizons); {fmt(tdyn, 2)} s."))
    x = D.rndMatWithRank(1000, 3, 3, 0., seed=21)
    out.append(item("Snippet 21.4", "Produce a random matrix of a given rank (rndMatWithRank)", "code", text=f"1000 x 3 matrix of rank 3; numerical rank {np.linalg.matrix_rank(x)}; as printed it is unseeded (a seed argument was added so a run repeats)."))
    out.append(item("Snippet 21.5", "Generate the problem's parameters (means, covariances, transaction-cost vectors)", "code",
                    table=tbl(["h", "mean", "diag(cov)", "c"], [[h + 1, np.array2string(p["mean"].ravel(), precision=3), np.array2string(np.diag(p["cov"]), precision=4), np.array2string(p["c"], precision=4)] for h, p in enumerate(params)]),
                    text="Seed 21; 3 assets, 2 horizons."))
    out.append(item("Snippet 21.6", "Compute and evaluate the static solution", "code", text=f"Static Sharpe ratio {sr_stat:.5f}; weights by horizon {np.array2string(w_stat.T, precision=4)}."))
    out.append(item("Snippet 21.7", "Compute and evaluate the dynamic solution", "code", text=f"Dynamic Sharpe ratio {sr_dyn:.5f} (static {sr_stat:.5f}); weights by horizon {np.array2string(w_dyn.T, precision=4)}."))
    # our assets
    rows = []
    for vname in VENUES:
        pan = WP.panel(vname).tail(504).dropna(axis=1, how="any")
        pref = {"Kraken": ["ETH", "SOL", "XRP", "ADA", "DOGE", "LINK"], "Alpaca": ["AAPL", "MSFT", "NVDA"]}[vname]
        cols = [c for c in pref if c in pan.columns][:3]
        if len(cols) < 3:
            cols = list(pan.columns[:3])
        c_side = {"Kraken": 12, "Alpaca": 10}[vname] / 1e4
        pr = []
        for h in range(2):
            x = pan[cols].iloc[h * 252:(h + 1) * 252]
            pr.append({"mean": x.mean().to_numpy().reshape(-1, 1), "cov": x.cov().to_numpy(), "c": np.full(3, c_side)})
        ws, srs = D.staticSolution(pr)
        wd, srd = D.dynamicSolution(pr)
        rows.append([vname, ", ".join(cols), fmt(srs, 4), fmt(srd, 4), fmt(float(np.abs(wd).sum(0).min()), 3)])
    out.append(item("Equation 97", "Sharpe ratio over H horizons, SR[r] = sum_h (mu_h' w_h - tau_h[w]) / sqrt(sum_h w_h' V_h w_h)", "data",
                    table=tbl(["Venue", "Assets", "Static SR (Snippet 21.6)", "Dynamic SR (Snippet 21.7)", "min sum |w| per horizon"], rows),
                    text="Our data: the two 252-day halves of the last 504 trading days are the two horizons (means and covariances of daily returns), tau_h = c |dw|^(1/2) with c = the venue's per-side cost (12 bp Kraken, 10 bp Alpaca; assumption).  The SR is per day, in the units of the means."))
    out.append(item("Equation 98", "Constraint: maximise SR[r] over w subject to sum_i |w_(i,h)| = 1 for every horizon h", "formula",
                    text="Every column of the weights in Snippets 21.6 and 21.7 has sum |w| = " + f"{float(np.abs(w_dyn).sum(0).min()):.3f} to {float(np.abs(w_dyn).sum(0).max()):.3f} (dynamic) and {float(np.abs(w_stat).sum(0).min()):.3f} to {float(np.abs(w_stat).sum(0).max()):.3f} (static)."))
    return out


def ch22():
    out = []
    items = [("Figure 22.1", "Schematic of the Magellan cluster (circa 2010), an example of an HPC cluster", "A hardware schematic from a 2010 US-government cloud study."),
             ("Figure 22.2", "The cloud ran scientific applications considerably slower than on HPC systems (circa 2010)", "A bar chart of run times from the same 2010 study."),
             ("Figure 22.3", "As the number of cores increases, the virtualisation overhead becomes much more significant (circa 2010)", "A plot of overhead against core count from the same study."),
             ("Figure 22.4", "Supernova SN 2011fe was discovered 11 hours after first evidence of explosion, thanks to extensive automation", "A timeline from an astronomy pipeline."),
             ("Figure 22.5", "A distributed workflow for studying fusion plasma dynamics", "A workflow diagram from a fusion-science project."),
             ("Figure 22.6", "Gradient tree boosting follows recent usage too closely; LTAP predicts baseline usage better (a-f, three groups)", "Six time-series panels from a power-grid usage study.")]
    for i, t, what in items:
        out.append(item(i, t, "diagram", text=what + "  It reports another group's experiment, not a computation on a dataset in this project, so there is nothing to recompute on Kraken or Alpaca data.",
                        note="Not reproducible here: the figure's data and code are not in the extract."))
    out.append(item("Figure 22.7", "Apple stock price on May 6, 2010 with HHI and VPIN computed every 5 minutes", "diagram",
                    text="The figure needs trade-level quote data of one day in May 2010 and VPIN's buy / sell split; neither is in this project (our Alpaca bars are hourly and our Kraken source rows are 1 minute, from 2026 only for trades).",
                    note="Not reproducible here: no trade-by-trade data for that day."))
    # Figure 22.8 analogue: ASCII versus binary storage of our own bars
    import writeup_core as WC
    p = WC.BARS["Kraken"] / "bars" / f"{SYM['Kraken']}.csv"
    d = pd.read_csv(p)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        for _ in range(5):
            pd.read_csv(p)
        t_ascii = (time.time() - t0) / 5
        num = d.select_dtypes("number")
        np.save(os.path.join(tmp, "x.npy"), num.to_numpy())
        t0 = time.time()
        for _ in range(5):
            np.load(os.path.join(tmp, "x.npy"))
        t_bin = (time.time() - t0) / 5
        size_a, size_b = p.stat().st_size, os.path.getsize(os.path.join(tmp, "x.npy"))
    out.append(item("Figure 22.8", "Time to read the same data stored as ASCII versus binary: the book's 10 years of S&P 500 quotes in HDF5 vs ASCII (21 times faster in binary)", "data",
                    table=tbl(["Format", "Seconds to read (mean of 5)", "File size (bytes)"], [["ASCII csv (ETH dollar bars)", fmt(t_ascii, 5), fmt(size_a)], ["binary .npy (numeric columns)", fmt(t_bin, 5), fmt(size_b)], ["ratio csv / binary", fmt(t_ascii / t_bin, 1), fmt(size_a / size_b, 2)]]),
                    note="Our analogue on our own bar file (the book's data is a 10-year quote archive); the read-time ratio is machine-dependent."))
    out.append(item("Figure 22.9", "The average false positive rates (alpha) of different classes of futures contracts, ordered by their average", "diagram",
                    text="A bar chart from another group's study of futures-contract data.  It reports an experiment on data that is not in this project, so there is nothing to recompute on Kraken or Alpaca data.", note="Not reproducible here: the figure's data and code are not in the extract."))
    # Figure 22.10 analogue: spectrum of our own hourly volume (Kraken trades 24 hours a day)
    sym = SYM["Kraken"]
    r = pd.read_csv(WC.BARS["Kraken"] / "ref_1h" / f"{sym}.csv", usecols=["date", "volume"], keep_default_na=False, na_values=[""]).dropna()
    r["date"] = pd.to_datetime(r["date"])
    r = r.set_index("date")["volume"].astype(float)
    r = r[r.index >= r.index.max() - pd.Timedelta(days=365)]
    r = r.reindex(pd.date_range(r.index.min(), r.index.max(), freq="h")).fillna(0.0)
    x = np.log1p(r.to_numpy())
    x = x - x.mean()
    sp = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), d=1.0 / (365 * 24))            # cycles per year
    top = np.argsort(sp[1:])[::-1][:5] + 1
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.semilogy(fr[1:], sp[1:], color="0.2", lw=0.5)
    ax.set_xlabel("frequency (cycles per year)"); ax.set_ylabel("power"); ax.set_xlim(0, 2500)
    out.append(item("Figure 22.10", "Fourier spectrum of trading activity: the book's natural-gas futures prices (peaks at once and twice per day and once per minute), and ours: hourly volume of ETH on Kraken", "data", fig=save(fig, "f22_10"),
                    table=tbl(["Rank", "Frequency (cycles per year)", "Cycles per day", "Power"], [[i + 1, fmt(float(fr[k]), 1), fmt(float(fr[k]) / 365, 3), f"{sp[k]:.3e}"] for i, k in enumerate(top)]),
                    note="Our analogue uses the last 365 days of the hourly reference volume (log(1 + volume), regular FFT on a gap-filled hourly grid; the book uses a non-uniform FFT on trade times).  Alpaca's hourly rows exist only in trading hours, so a regular FFT is not valid for it and is not shown."))
    return out


def build():
    items = []
    for f in (ch20, ch21, ch22):
        print("building", f.__name__, flush=True)
        items += f()
    return items


if __name__ == "__main__":
    its = build()
    json.dump(its, open(W.OUT / "items_ch20_22.json", "w"), indent=1, default=str)
    print(len(its), "items")
