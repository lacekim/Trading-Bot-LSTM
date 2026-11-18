"""Chapter 20 of the book (Snippets 20.1-20.14), ported to Python 3 / pandas 3 with the book's names.

Where the printed code cannot run as printed, the change is the smallest one and is listed here:
  20.1 / 20.2  print statements -> print(); izip -> zip.
  20.3         barrierTouch is ported as printed (`barrier_touch_as_printed`): its `continue` only continues the inner loop, so it records the LAST row that touches a barrier, not the earliest as the
               comment says.  `barrierTouch` is the version that returns the earliest touch (a `break`); both are kept and tested against each other and against a vectorised reference.
  20.4         xrange -> range; the printed code splits the ROWS (r.shape[0]) into parts and then slices COLUMNS (r[:, a:b]); the columns are the paths, so the parts here are taken over r.shape[1].
  20.5 / 20.6  as printed (xrange -> range; Python 3 true division is what the printed code assumes).
  20.7         mpPandasObj: the printed code uses a name `argList` that is not defined there (the argument is called pdObj); pdObj is used.  DataFrame.append (removed from pandas) -> pd.concat.
  20.8 - 20.10 as printed.
  20.11        the Python 2 copy_reg / im_func pickling of methods is not needed in Python 3 (bound methods pickle); a copyreg registration with __func__ / __self__ is kept so the snippet is present.
  20.12 / 20.13 as printed; `copy` and `pd` imports added.
  20.14        getPCs as printed; used with pd.DataFrame.add as the redux function.
"""
from __future__ import annotations

import copy
import copyreg
import datetime as dt
import itertools
import multiprocessing as mp
import sys
import time
import types

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------------------------------
# 20.1 / 20.2 Cartesian product of a dictionary of lists
# ---------------------------------------------------------------------------------------------------------------------
def cartesian_unvectorized(dict0):
    """Snippet 20.1 for a dictionary with exactly the keys 'a', 'b', 'c' (as in the book's example); returns the list instead of printing it."""
    out = []
    for a in dict0["a"]:
        for b in dict0["b"]:
            for c in dict0["c"]:
                out.append({"a": a, "b": b, "c": c})
    return out


def cartesian_vectorized(dict0):
    """Snippet 20.2: any number of keys."""
    return [dict(zip(dict0, i)) for i in itertools.product(*dict0.values())]


# ---------------------------------------------------------------------------------------------------------------------
# 20.3 / 20.4 one-touch double barrier
# ---------------------------------------------------------------------------------------------------------------------
def barrier_touch_as_printed(r, width=0.5):
    """Snippet 20.3 as printed: the inner `continue` does not stop the scan of a column, so t[j] is the LAST row whose log path is at or beyond +-width."""
    t, p = {}, np.log((1 + r).cumprod(axis=0))
    for j in range(r.shape[1]):
        for i in range(r.shape[0]):
            if p[i, j] >= width or p[i, j] <= -width:
                t[j] = i
                continue
    return t


def barrierTouch(r, width=0.5):
    """Snippet 20.3 with the comment's intent: the index of the EARLIEST barrier touch of each column."""
    t, p = {}, np.log((1 + r).cumprod(axis=0))
    for j in range(r.shape[1]):
        for i in range(r.shape[0]):
            if p[i, j] >= width or p[i, j] <= -width:
                t[j] = i
                break
    return t


def barrier_touch_vectorised(r, width=0.5):
    """Reference for the tests: earliest touch per column, no loops over rows."""
    p = np.log((1 + r).cumprod(axis=0))
    hit = (p >= width) | (p <= -width)
    first = hit.argmax(axis=0)
    return {j: int(first[j]) for j in range(r.shape[1]) if hit[first[j], j]}


def _barrier_job(args):
    r, offset, width = args
    return {j + offset: i for j, i in barrierTouch(r, width).items()}


def main1(r=None, numThreads=24, width=0.5):
    """Snippet 20.4: the columns (paths) are split into parts and each part is searched in a process of a pool.  Returns the merged dictionary."""
    if r is None:
        r = np.random.normal(0, 0.01, size=(1000, 10000))
    parts = np.linspace(0, r.shape[1], min(numThreads, r.shape[1]) + 1)
    parts, jobs = np.ceil(parts).astype(int), []
    for i in range(1, len(parts)):
        jobs.append((r[:, parts[i - 1]:parts[i]], int(parts[i - 1]), width))
    pool, out = mp.Pool(processes=numThreads), {}
    outputs = pool.imap_unordered(_barrier_job, jobs)
    for out_ in outputs:
        out.update(out_)
    pool.close()
    pool.join()
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 20.5 / 20.6 partitions of atoms into molecules
# ---------------------------------------------------------------------------------------------------------------------
def linParts(numAtoms, numThreads):
    # partition of atoms with a single loop
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nestedParts(numAtoms, numThreads, upperTriang=False):
    # partition of atoms with an inner loop
    parts, numThreads_ = [0], min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:  # the first rows are the heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


# ---------------------------------------------------------------------------------------------------------------------
# 20.8 - 20.10 job execution
# ---------------------------------------------------------------------------------------------------------------------
def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func = kargs["func"]
    del kargs["func"]
    out = func(**kargs)
    return out


def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out


def reportProgress(jobNum, numJobs, time0, task):
    # Report progress as asynch jobs are completed
    msg = [float(jobNum) / numJobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp + " " + str(round(msg[0] * 100, 2)) + "% " + task + " done after " + str(round(msg[1], 2)) + " minutes. Remaining " + str(round(msg[2], 2)) + " minutes."
    if jobNum < numJobs:
        sys.stderr.write(msg + "\r")
    else:
        sys.stderr.write(msg + "\n")
    return


def processJobs(jobs, task=None, numThreads=24):
    # Run in parallel.  jobs must contain a 'func' callback, for expandCall
    if task is None:
        task = jobs[0]["func"].__name__
    pool = mp.Pool(processes=numThreads)
    outputs, out, time0 = pool.imap_unordered(expandCall, jobs), [], time.time()
    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 20.11 pickling of methods (Python 2 recipe kept; unnecessary in Python 3)
# ---------------------------------------------------------------------------------------------------------------------
def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = type(obj)
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


copyreg.pickle(types.MethodType, _pickle_method)


# ---------------------------------------------------------------------------------------------------------------------
# 20.7 mpPandasObj, 20.12 processJobsRedux, 20.13 mpJobList, 20.14 getPCs
# ---------------------------------------------------------------------------------------------------------------------
def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    """
    Parallelize jobs, return a DataFrame or Series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kargs: any other argument needed by func
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kargs)
    """
    if linMols:
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)
    jobs = []
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i - 1]:parts[i]], "func": func}
        job.update(kargs)
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    df0 = pd.concat([df0] + list(out)) if len(df0) else pd.concat(list(out))
    df0 = df0.sort_index()
    return df0


def processJobsRedux(jobs, task=None, numThreads=24, redux=None, reduxArgs=None, reduxInPlace=False):
    """
    Run in parallel
    jobs must contain a 'func' callback, for expandCall
    redux prevents wasting memory by reducing output on the fly
    """
    reduxArgs = {} if reduxArgs is None else reduxArgs
    if task is None:
        task = jobs[0]["func"].__name__
    pool = mp.Pool(processes=numThreads)
    imap, out, time0 = pool.imap_unordered(expandCall, jobs), None, time.time()
    # Process asynchronous output, report progress
    for i, out_ in enumerate(imap, 1):
        if out is None:
            if redux is None:
                out, redux, reduxInPlace = [out_], list.append, True
            else:
                out = copy.deepcopy(out_)
        else:
            if reduxInPlace:
                redux(out, out_, **reduxArgs)
            else:
                out = redux(out, out_, **reduxArgs)
        reportProgress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    if isinstance(out, (pd.Series, pd.DataFrame)):
        out = out.sort_index()
    return out


def mpJobList(func, argList, numThreads=24, mpBatches=1, linMols=True, redux=None, reduxArgs=None, reduxInPlace=False, **kargs):
    reduxArgs = {} if reduxArgs is None else reduxArgs
    if linMols:
        parts = linParts(len(argList[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(argList[1]), numThreads * mpBatches)
    jobs = []
    for i in range(1, len(parts)):
        job = {argList[0]: argList[1][parts[i - 1]:parts[i]], "func": func}
        job.update(kargs)
        jobs.append(job)
    out = processJobsRedux(jobs, redux=redux, reduxArgs=reduxArgs, reduxInPlace=reduxInPlace, numThreads=numThreads)
    return out


def getPCs(path, molecules, eVec):
    # get principal components by loading one file at a time
    pcs = None
    for i in molecules:
        df0 = pd.read_csv(path + i, index_col=0, parse_dates=True)
        if pcs is None:
            pcs = np.dot(df0.values, eVec.loc[df0.columns].values)
        else:
            pcs += np.dot(df0.values, eVec.loc[df0.columns].values)
    pcs = pd.DataFrame(pcs, index=df0.index, columns=eVec.columns)
    return pcs
