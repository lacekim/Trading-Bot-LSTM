"""Snippet 16.5 Monte Carlo (Table of Section 16.5 text / Appendix 16.A.4): IVP, HRP, CLA out-of-sample variance.  New file."""
import json, random, sys, numpy as np
import book_port_d as D
import writeup_core as W

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
np.random.seed(16); random.seed(16)
stats, rep = D.hrpMC(numIters=n)
res = {"iterations": n, "seed": 16, "report": {k: {c: float(v) for c, v in rep.loc[k].items()} for k in rep.index}}
json.dump(res, open(W.OUT / "hrp_mc.json", "w"), indent=1)
print(res)
