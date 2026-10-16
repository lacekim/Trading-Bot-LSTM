"""Reproduction ledger: every numerical result PRINTED in the book that can be recomputed from what the book itself supplies (its formulas, its seeds,
its stated inputs), set against what this project's code returns.  New file; reads nothing from any run, changes nothing.

Not reproducible here (the book's data is not in the project): Table 5.1 (ADF on FFD(d) of ~50 futures series), Figures 13.x heat-maps (they depend on
1e5 random paths with no seed), Figure 18.1 (random messages, no seed), Figure 11.x, Figure 14.x (curves).  They are listed as such, not simulated.
Usage:  python -W ignore tools/book_study/book_reproduce.py <out.txt>
"""
from __future__ import annotations

import json
import sys
from math import comb

import numpy as np

import book_port_c as C
import book_port_d as D
import book_stats as B

BOOK_COLS = [2, 0, 4, 1, 1]  # Figure 16.5 caption (labels 6<-3, 7<-1, 8<-5, 9<-2, 10<-2)
TABLE_16_1 = {  # book p.146
    "CLA": [14.44, 19.93, 19.73, 19.87, 18.68, 0.00, 5.86, 1.49, 0.00, 0.00],
    "HRP": [7.00, 7.59, 10.84, 19.03, 9.72, 10.19, 6.62, 9.10, 7.12, 12.79],
    "IVP": [10.36, 10.28, 10.36, 10.25, 10.31, 9.74, 9.80, 9.65, 9.64, 9.61]}
FIG_12_1 = {1: [1, 2, 3, 4, 5], 2: [1, 6, 7, 8, 9], 3: [2, 6, 10, 11, 12], 4: [3, 7, 10, 13, 14], 5: [4, 8, 11, 13, 15], 6: [5, 9, 12, 14, 15]}


def main(out):
    rows = []

    def add(item, book, ours, ok, note=""):
        rows.append({"item": item, "book": book, "ours": ours, "match": bool(ok), "note": note})

    # Snippet 10.4 (page 90)
    w, tPos, lP = C.main_10_4()
    add("Snippet 10.4 order size (f=115,p=100,maxPos=100)", 97, int(tPos), tPos == 97)
    add("Snippet 10.4 limit price", 112.3657, round(float(lP), 4), abs(lP - 112.3657) < 5e-4)
    t95 = C.getTPos(w, 110, 100, 100)
    add("Snippet 10.4 order size for f=110", 95, int(t95), t95 == 95)

    # Equation 37
    add("Equation 37, S=16 combinations", 12780, comb(16, 8), comb(16, 8) == 12780,
        "the book's own Equation 37 (S choose S/2) gives 12,870; 12,780 as printed is a transposition of its own formula")

    # Equation 40 / Figure 12.1
    splits = B.cpcv_splits(6, 2)
    mat = {g: [si + 1 for si, s in enumerate(splits) if g - 1 in s] for g in range(1, 7)}
    add("Figure 12.1 number of splits for N=6,k=2", 15, len(splits), len(splits) == 15)
    add("Figure 12.1 paths phi[6,2]", 5, B.cpcv_paths(6, 2)[1], B.cpcv_paths(6, 2)[1] == 5)
    add("Figure 12.1 test-group marks, all 6 rows", FIG_12_1, mat, mat == FIG_12_1)

    # Section 16.4 / Figure 16.5 / Table 16.1, on the book's own draw
    x, cols = D.generateData(10000, 5, 5, .25, cols=BOOK_COLS)
    cov, corr = x.cov(), x.corr()
    import scipy.cluster.hierarchy as sch
    link = sch.linkage(D.correlDist(corr), "single")
    order = corr.index[D.getQuasiDiag(link)].tolist()
    add("Figure 16.5 clustered order", [9, 2, 10, 1, 7, 3, 6, 4, 5, 8], order, order == [9, 2, 10, 1, 7, 3, 6, 4, 5, 8])
    got = {"HRP": D.getHRP(cov, corr).values * 100, "IVP": D.getIVP(cov) * 100, "CLA": D.getCLA(cov) * 100}
    for k, ref in TABLE_16_1.items():
        g = np.round(got[k], 2)
        add(f"Table 16.1 {k} (10 weights, %)", ref, g.tolist(), np.allclose(g, ref, atol=0.011), f"max abs diff {np.abs(g - np.array(ref)).max():.3f}")

    # Same code, the Python-3 default draw (the book's snippet draws the columns with random.randint; the sequence differs between Python 2 and 3)
    x2, cols2 = D.generateData(10000, 5, 5, .25)
    c2, r2 = x2.cov(), x2.corr()
    h2 = np.round(D.getHRP(c2, r2).values * 100, 2)
    add("Table 16.1 HRP on the default Python-3 draw (columns " + str(cols2) + ")", TABLE_16_1["HRP"], h2.tolist(), np.allclose(h2, TABLE_16_1["HRP"], atol=0.011),
        "expected NOT to match: the data differs because random.randint gives other columns in Python 3; this is the data effect, not a code effect")

    txt = []
    for r in rows:
        txt.append(f"{'MATCH   ' if r['match'] else 'DIFFERS '}{r['item']}\n    book: {r['book']}\n    ours: {r['ours']}" + (f"\n    note: {r['note']}" if r["note"] else ""))
    txt.append("\nNOT REPRODUCIBLE FROM THE PROJECT (book data / unseeded randomness not supplied): Table 5.1, Figures 13.1-13.25, Figure 18.1, Figures 11.1-11.2, Figures 14.1-14.3, Figure 15.1 (curve), Figure 12.2 values.")
    open(out, "w").write("\n".join(txt) + "\n")
    json.dump(rows, open(out + ".json", "w"), indent=1, default=str)
    print("\n".join(txt))


if __name__ == "__main__":
    main(sys.argv[1])
