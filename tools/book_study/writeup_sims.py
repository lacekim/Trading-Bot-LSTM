"""Simulations behind the write-up figures that the book draws from simulated inputs (not from market data).  New file.
  fig13   Table 13.1 / Figures 13.1-13.25: heat-maps of the Sharpe ratio over the (profit-taking, stop-loss) mesh of Snippet 13.1 (linspace(0, 10, 21)), maxHP = 100, 100,000 paths per figure
          (Snippet 13.2's rule: exit when p - seed > pt or < -sl, or on the step after hp exceeds maxHP, i.e. after 101 steps; paths are generated vectorised with a fixed seed, so the numbers are a sample of the
          book's, not a copy: the book's runs are unseeded).
  cscv    Figures 11.1 / 11.2: combinatorially symmetric cross-validation on 1,000 pure-noise strategies, S = 16 (12,870 combinations; the book's text prints 12,780).
  synth8  Figures 8.2-8.4: MDI / MDA / SFI on the book's synthetic dataset (Snippet 8.7 inputs), reduced size (see the note in the output json).
Usage: python -W ignore tools/book_study/writeup_sims.py fig13|cscv|synth8|all
"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

import book_stats as B
import writeup_core as W

MESH21 = tuple(np.linspace(0, 10, 21))


def fig13(n_paths=100000, max_hp=100, seed=13):
    out = {}
    rng = np.random.default_rng(seed)
    steps = max_hp + 1
    for k, (f, hl) in enumerate(itertools.product([0, 5, 10, -5, -10], [5, 10, 25, 50, 100]), 1):
        t0 = time.time()
        phi = 2 ** (-1.0 / hl)
        eps = rng.standard_normal((n_paths, steps)).astype(np.float64)
        paths = B.simulate_ou_paths((1 - phi) * f, phi, eps)
        st = B.mesh_exit_stats(paths.astype(np.float32), pts=MESH21, sls=MESH21)
        out[(f, hl)] = {"sharpe": st["sharpe"], "mean": st["mean"], "std": st["std"]}
        print(f"fig13 {k}/25 forecast={f} hl={hl} {time.time() - t0:.0f}s best={np.nanmax(st['sharpe']):.3f}", flush=True)
    np.savez_compressed(W.OUT / "fig13.npz", **{f"f{f}_h{h}": v["sharpe"] for (f, h), v in out.items()})
    return out


def cscv(N=1000, T=1000, S=16, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N))
    L = T // S
    blocks = X[: L * S].reshape(S, L, N)
    A, Q = blocks.sum(1), (blocks ** 2).sum(1)
    combos = list(itertools.combinations(range(S), S // 2))
    sr_is, sr_oos, logit = [], [], []
    ann = np.sqrt(252.0)
    n = (S // 2) * L
    for c in combos:
        ci = list(c)
        co = [i for i in range(S) if i not in c]
        mi, mo = A[ci].sum(0) / n, A[co].sum(0) / n
        vi, vo = Q[ci].sum(0) / n - mi ** 2, Q[co].sum(0) / n - mo ** 2
        s_is, s_oos = mi / np.sqrt(vi), mo / np.sqrt(vo)
        j = int(np.argmax(s_is))
        rank = (s_oos < s_oos[j]).sum() + 1
        w = rank / (N + 1.0)
        sr_is.append(s_is[j] * ann); sr_oos.append(s_oos[j] * ann); logit.append(np.log(w / (1 - w)))
    sr_is, sr_oos, logit = map(np.array, (sr_is, sr_oos, logit))
    slope, icpt = np.polyfit(sr_is, sr_oos, 1)
    r2 = np.corrcoef(sr_is, sr_oos)[0, 1] ** 2
    res = {"combinations": len(combos), "N": N, "T": T, "S": S, "seed": seed, "intercept": float(icpt), "slope": float(slope), "adj_r2": float(1 - (1 - r2) * (len(sr_is) - 1) / (len(sr_is) - 2)),
           "prob_sr_oos_lt_0": float((sr_oos < 0).mean()), "pbo": float((logit <= 0).mean())}
    np.savez_compressed(W.OUT / "cscv.npz", sr_is=sr_is, sr_oos=sr_oos, logit=logit)
    json.dump(res, open(W.OUT / "cscv.json", "w"), indent=1)
    print("cscv", res, flush=True)
    return res


def synth8(n_estimators=100, n_samples=3000, cv=5):
    import book_port_b as PB
    X, cont = PB.getTestData(40, 10, 10, n_samples)
    res = {"settings": {"n_features": 40, "n_informative": 10, "n_redundant": 10, "n_estimators": n_estimators, "n_samples": n_samples, "cv": cv, "book": "n_estimators=1000, n_samples=10000, cv=10"}}
    for m in ("MDI", "MDA", "SFI"):
        t0 = time.time()
        imp, oob, oos = PB.featImportance(X, cont, n_estimators=n_estimators, cv=cv, numThreads=1 if m == "SFI" else 1, pctEmbargo=0, scoring="accuracy", method=m, minWLeaf=0.0)
        imp.to_csv(W.OUT / f"synth8_{m}.csv")
        res[m] = {"oob": float(oob), "oos": float(oos)}
        print("synth8", m, res[m], f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(W.OUT / "synth8.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    W.OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("cscv", "all"):
        cscv()
    if what in ("synth8", "all"):
        synth8()
    if what in ("fig13", "all"):
        fig13()
