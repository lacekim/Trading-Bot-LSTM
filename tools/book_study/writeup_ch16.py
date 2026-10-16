"""Write-up items for Chapter 16 (HRP) of the book.  New file; reads the saved panels / runs, changes nothing."""
from __future__ import annotations

import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.sparse.csgraph import minimum_spanning_tree

import book_port_d as D
import writeup_core as W
import writeup_panel as WP
from writeup_core import VENUES, SYM, item, tbl, fmt, save

BOOK_COLS = [2, 0, 4, 1, 1]
TABLE_16_1 = {"CLA": [14.44, 19.93, 19.73, 19.87, 18.68, 0.00, 5.86, 1.49, 0.00, 0.00], "HRP": [7.00, 7.59, 10.84, 19.03, 9.72, 10.19, 6.62, 9.10, 7.12, 12.79],
              "IVP": [10.36, 10.28, 10.36, 10.25, 10.31, 9.74, 9.80, 9.65, 9.64, 9.61]}
N_VAR = 20


def window(v, n_var=N_VAR, days=252):
    r = WP.panel(v).tail(days)
    r = r.dropna(axis=1, how="any")
    cols = list(r.columns[:n_var])
    return r[cols]


def order_of(x):
    cov, corr, dist, link = D.treeCluster(x)
    ix = corr.index[D.getQuasiDiag(link)].tolist()
    return cov, corr, dist, link, ix


def ch16():
    out = []
    win = {v: window(v) for v in VENUES}
    # Figure 16.1 ------------------------------------------------------------------------------------------------------------------------------------
    n = 40
    mats = {}
    mats["Identity"] = np.eye(n)
    for name, rho in (("Correlated", 0.5), ("Ill-conditioned", 0.99)):
        mats[name] = np.full((n, n), rho) + (1 - rho) * np.eye(n)
    mats["Singular"] = np.ones((n, n))
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.5))
    for (name, m), c, ls in zip(mats.items(), ("0.0", "0.35", "0.6", "0.0"), ("-", "-", "-", "--")):
        axs[0].plot(np.sort(np.linalg.eigvalsh(m))[::-1], color=c, ls=ls, label=name)
    axs[0].set_xlabel("Variable #"); axs[0].set_ylabel("Eigenvalues"); axs[0].set_title("Markowitz's curse (constructed)"); axs[0].legend()
    rows = []
    for ax, v in zip(axs[1:], VENUES):
        c = win[v].corr().to_numpy()
        ev = np.sort(np.linalg.eigvalsh(c))[::-1]
        ax.plot(ev, color="0.15"); ax.set_title(f"{v}: last 252 days, {len(ev)} symbols"); ax.set_xlabel("Variable #")
        rows.append([v, len(ev), fmt(ev[0], 3), fmt(ev[-1], 4), fmt(ev[0] / ev[-1], 1)])
    fig.tight_layout()
    cond = [[k, fmt(np.linalg.cond(m), 3) if k != "Singular" else "infinite"] for k, m in mats.items()]
    out.append(item("Figure 16.1", "Markowitz's curse: eigenvalue spectra of an identity, a correlated, an ill-conditioned and a singular matrix, beside ours", "formula", fig=save(fig, "f16_1"),
                    table=tbl(["Venue", "Symbols", "Largest eigenvalue", "Smallest eigenvalue", "Condition number"], rows),
                    note="The extract prints the picture, not the matrices; ours are constructed (40 variables, common correlation 0, 0.5, 0.99 and 1 = identity, correlated, ill-conditioned, singular).  Condition numbers of the constructed ones: " + "; ".join(f"{a} {b}" for a, b in cond) + "."))

    # Section 16.4.1 and Equation 58 -----------------------------------------------------------------------------------------------------------------
    rho = np.array([[1, .7, .2], [.7, 1, -.2], [.2, -.2, 1]])
    d = np.sqrt(.5 * (1 - rho))
    dt = np.sqrt(((d[:, :, None] - d[:, None, :]) ** 2).sum(0))
    out.append(item("Section 16.4.1", "Tree clustering: correlation to distance d = sqrt(1/2 (1 - rho)); the book's Example 16.1", "formula",
                    table=tbl(["", "1", "2", "3"], [[i + 1] + [fmt(x, 4) for x in d[i]] for i in range(3)]),
                    text="Matches the book's matrix for the example ({0, .3873, .6325; .3873, 0, .7746; .6325, .7746, 0}). On our data: Kraken and Alpaca distance matrices are the ones used below (last 252 trading days)."))
    rows = []
    for v in VENUES:
        dist = D.correlDist(win[v].corr()).to_numpy()
        dtil = np.sqrt(((dist[:, :, None] - dist[:, None, :]) ** 2).sum(0))
        rows.append([v, dist.shape[0], fmt(dist[np.triu_indices_from(dist, 1)].mean(), 4), fmt(dtil[np.triu_indices_from(dtil, 1)].mean(), 4)])
    out.append(item("Equation 58", "Euclidean distance between two columns of the distance matrix, d~(i,j) = sqrt(sum_n (d(n,i) - d(n,j))^2)", "data",
                    table=tbl(["Venue", "N", "mean d", "mean d~"], rows), text="Example 16.1: d~ = " + np.array2string(dt, precision=4)))

    # Figure 16.2 ------------------------------------------------------------------------------------------------------------------------------------
    def draw(ax, v, tree):
        x = win[v].iloc[:, :14]
        c = x.corr().to_numpy()
        dist = D.correlDist(pd.DataFrame(c)).to_numpy()
        n_ = len(c)
        ang = np.linspace(0, 2 * np.pi, n_, endpoint=False)
        px, py = np.cos(ang), np.sin(ang)
        if tree:
            mst = minimum_spanning_tree(dist).toarray()
            edges = [(i, j) for i in range(n_) for j in range(n_) if mst[i, j] > 0]
        else:
            edges = [(i, j) for i in range(n_) for j in range(i + 1, n_)]
        for i, j in edges:
            ax.plot([px[i], px[j]], [py[i], py[j]], color="0.6", lw=0.5)
        ax.scatter(px, py, color="0.1", s=14, zorder=3)
        for k, s in enumerate(x.columns):
            ax.text(1.12 * px[k], 1.12 * py[k], str(s), ha="center", va="center", fontsize=5)
        ax.set_aspect("equal"); ax.axis("off")
    fig, axs = plt.subplots(2, 2, figsize=(5.6, 5.2))
    for j, v in enumerate(VENUES):
        draw(axs[0, j], v, False); axs[0, j].set_title(f"{v}: complete graph")
        draw(axs[1, j], v, True); axs[1, j].set_title(f"{v}: tree graph (minimum spanning tree)")
    fig.tight_layout()
    out.append(item("Figure 16.2", "The complete-graph (top) and tree-graph (bottom) structures of 14 of our symbols' correlation matrices", "data", fig=save(fig, "f16_2"),
                    note="The book draws 40 anonymous nodes; ours are named symbols; the tree is the minimum spanning tree of the correlation distance."))

    # Figure 16.3 ------------------------------------------------------------------------------------------------------------------------------------
    rows, cover = [], {}
    for v in VENUES:
        x = win[v].iloc[:, :10]
        cov, corr, dist, link = D.treeCluster(x)
        labels = list(x.columns)
        name = {i: labels[i] for i in range(len(labels))}
        steps = []
        for k, (a, b, h, sz) in enumerate(link[:6], 1):
            a, b = int(a), int(b)
            steps.append(f"{k}: {name[a]} + {name[b]} (d={h:.3f})")
            name[len(labels) + k - 1] = f"({name[a]},{name[b]})"
        rows.append([v, " | ".join(steps)])
    out.append(item("Figure 16.3", "Sequence of cluster formation: the first six merges of 10 of our symbols (single linkage on the correlation distance)", "data",
                    table=tbl(["Venue", "Merges in order"], rows), note="The book draws the steps as pictures of its 10 synthetic variables; the single-linkage merge sequence is the same information for ours."))

    # Snippets 16.1 - 16.3 ---------------------------------------------------------------------------------------------------------------------------
    rows16 = []
    for v in VENUES:
        cov, corr, dist, link, ix = order_of(win[v])
        w = D.getRecBipart(cov, ix)
        ivp = pd.Series(D.getIVP(cov), index=cov.index)
        rows16.append((v, cov, corr, link, ix, w, ivp))
    out.append(item("Snippet 16.1", "Tree clustering with SciPy: correlation, distance, single-linkage matrix (treeCluster)", "data",
                    table=tbl(["Venue", "Symbols N", "Linkage rows", "Last merge distance"], [[r[0], len(r[4]), len(r[3]), fmt(float(r[3][-1, 2]), 4)] for r in rows16]),
                    text="book_port_d.treeCluster is the printed lines wrapped in a function.  Window: last 252 trading days of the first 20 symbols with complete data."))
    out.append(item("Snippet 16.2", "Quasi-diagonalisation: the order of leaves that puts similar items next to each other", "data",
                    table=tbl(["Venue", "Quasi-diagonal order"], [[r[0], ", ".join(map(str, r[4]))] for r in rows16]), text="book_port_d.getQuasiDiag (Series.append replaced by concat)."))
    out.append(item("Snippet 16.3", "Recursive bisection: weights by splitting the variance of the two halves, and the same window's inverse-variance weights", "data",
                    table=tbl(["Venue", "Symbol", "HRP weight %", "IVP weight %"], [[r[0], s, fmt(100 * float(r[5][s]), 3), fmt(100 * float(r[6][s]), 3)] for r in rows16 for s in r[4][:6]]),
                    text="First six symbols of each venue's quasi-diagonal order shown; weights sum to 100% over the 20 symbols (book_port_d.getRecBipart)."))

    # Figure 16.4 / 16.6 / 16.8 ----------------------------------------------------------------------------------------------------------------------
    def heatmap(ax, m, labels, title):
        im = ax.pcolor(m, cmap="gray")
        ax.set_title(title); ax.set_xticks(np.arange(.5, m.shape[0] + .5)); ax.set_yticks(np.arange(.5, m.shape[0] + .5))
        ax.set_xticklabels(labels, rotation=90, fontsize=4); ax.set_yticklabels(labels, fontsize=4)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, r in zip(axs, rows16):
        heatmap(ax, r[1].to_numpy(), list(r[1].columns), f"{r[0]}: covariance matrix, original order")
    fig.tight_layout()
    out.append(item("Figure 16.4", "Heat-map of the original covariance matrix (20 of our symbols, last 252 days)", "data", fig=save(fig, "f16_4")))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, r in zip(axs, rows16):
        heatmap(ax, r[1].loc[r[4], r[4]].to_numpy(), r[4], f"{r[0]}: covariance matrix, clustered order")
    fig.tight_layout()
    out.append(item("Figure 16.6", "Clustered covariance matrix (same data, quasi-diagonal order)", "data", fig=save(fig, "f16_6")))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
    for j, r in enumerate(rows16):
        heatmap(axs[0, j], r[2].to_numpy(), list(r[2].columns), f"{r[0]}: correlation, before clustering")
        heatmap(axs[1, j], r[2].loc[r[4], r[4]].to_numpy(), r[4], f"{r[0]}: correlation, after clustering")
    fig.tight_layout()
    out.append(item("Figure 16.8", "Correlation matrix before and after clustering (ours)", "data", fig=save(fig, "f16_8")))

    # Figure 16.5 and Table 16.1 ---------------------------------------------------------------------------------------------------------------------
    x, cols = D.generateData(10000, 5, 5, .25, cols=BOOK_COLS)
    cov, corr = x.cov(), x.corr()
    link = sch.linkage(D.correlDist(corr), "single")
    order = corr.index[D.getQuasiDiag(link)].tolist()
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
    sch.dendrogram(link, labels=list(corr.columns), ax=axs[0], color_threshold=0, above_threshold_color="0.2"); axs[0].set_title("book's 10 variables (pinned draw)")
    for ax, r in zip(axs[1:], rows16):
        sch.dendrogram(r[3], labels=list(r[1].columns), ax=ax, color_threshold=0, above_threshold_color="0.2", leaf_font_size=5)
        ax.set_title(f"{r[0]}: 20 symbols")
    fig.tight_layout()
    out.append(item("Figure 16.5", "Dendrogram of cluster formation: the book's draw (order reproduced) and ours", "data", fig=save(fig, "f16_5"),
                    text=f"Clustered order of the book's data with the columns read off the book's caption: {order}; the book prints [9, 2, 10, 1, 7, 3, 6, 4, 5, 8]: " + ("identical." if order == [9, 2, 10, 1, 7, 3, 6, 4, 5, 8] else "DIFFERENT.")))
    got = {"CLA": D.getCLA(cov) * 100, "HRP": D.getHRP(cov, corr).values * 100, "IVP": D.getIVP(cov) * 100}
    rows = []
    for i in range(10):
        rows.append([i + 1] + [f"{got[k][i]:.2f}% (book {TABLE_16_1[k][i]:.2f}%)" for k in ("CLA", "HRP", "IVP")])
    t_book = tbl(["Weight #", "CLA", "HRP", "IVP"], rows)
    ours = []
    for r in rows16:
        cov_, ix = r[1].iloc[:10, :10], None
        sub = win[r[0]].iloc[:, :10]
        c_, k_ = sub.cov(), sub.corr()
        cla, hrp, ivp = D.getCLA(c_) * 100, D.getHRP(c_, k_).values * 100, D.getIVP(c_) * 100
        for i, s in enumerate(sub.columns):
            ours.append([r[0], s, f"{cla[i]:.2f}%", f"{hrp[i]:.2f}%", f"{ivp[i]:.2f}%"])
    out.append(item("Table 16.1", "A comparison of three allocations (CLA, HRP, IVP): the book's numbers reproduced, then ours", "data", table=t_book,
                    note="Every printed weight is reproduced to 2 decimals with the draw of columns read off the figure caption; Python 3's random.randint draws other columns for the same seed, so the book's snippet run as printed gives different data (and different weights) in Python 3; CLA is the long-only minimum-variance portfolio (the CLA module is not in the extract)."))
    out[-1]["table2"] = tbl(["Venue", "Symbol", "CLA", "HRP", "IVP"], ours)
    out[-1]["table2_title"] = "The same three allocations on the first 10 symbols of each venue's window"

    # Figure 16.7 ------------------------------------------------------------------------------------------------------------------------------------
    np.random.seed(7); random.seed(7)
    xm, colsm = D.generateData_MC(520, 260, 5, 5, 0, 1e-2, .25)
    W_ = {"IVP": [], "HRP": [], "CLA": []}
    for pointer in range(260, 520, 22):
        x_ = xm[pointer - 260:pointer]
        cov_, corr_ = np.cov(x_, rowvar=0), np.corrcoef(x_, rowvar=0)
        W_["IVP"].append(D.getIVP(cov_)); W_["HRP"].append(D.getHRP(cov_, corr_).values); W_["CLA"].append(D.getCLA(cov_))
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.4), sharey=True)
    for ax, (k, ws) in zip(axs, W_.items()):
        a = np.array(ws)
        ax.stackplot(range(len(a)), a.T, colors=[str(c) for c in np.linspace(0.1, 0.9, a.shape[1])])
        ax.set_title(f"{k} allocation"); ax.set_xlabel("rebalance")
    fig.tight_layout()
    out.append(item("Figure 16.7", "Time series of allocations for IVP, HRP and CLA in one run of the Monte Carlo experiment", "formula", fig=save(fig, "f16_7"),
                    note="One simulated experiment (Snippet 16.5's generator, seed 7), 12 monthly rebalances on a 260-observation window; the book shows its own unseeded draw."))

    # Chapter 16 Appendices --------------------------------------------------------------------------------------------------------------------------
    worst = 0.0
    for r in rows16:
        dd = D.correlDist(r[2]).to_numpy()
        n_ = len(dd)
        for i in range(n_):
            worst = max(worst, float((dd[i][:, None] + dd[i][None, :] - dd).min() * -1))
    sub = x.cov()
    diag = np.diag(np.diag(sub))
    ivp = D.getIVP(diag)
    mv = np.linalg.solve(diag, np.ones(len(diag)))
    mv /= mv.sum()
    txt = (f"A.1 Correlation-based metric: the triangle inequality d[X,Z] <= d[X,Y] + d[Y,Z] holds on every triple of our two 20-symbol distance matrices (largest violation {max(0.0, worst):.2e}).  "
           f"A.2 Inverse-variance allocation: for a diagonal covariance matrix the IVP equals the minimum-variance portfolio (largest difference {np.abs(ivp - mv).max():.2e}).  "
           "A.3 Reproducing the numerical example: Snippet 16.4 is book_port_d.main_16_4 (Figure 16.5 and Table 16.1 above).  "
           "A.4 Reproducing the Monte Carlo experiment: Snippet 16.5 is book_port_d.hrpMC; see the table.")
    try:
        mc = json.load(open(W.OUT / "hrp_mc.json"))
        mtab = tbl(["Method", "std", "variance", "variance vs HRP"], [[k.replace("get", ""), fmt(v["std"], 5), fmt(v["var"], 6), f"{100 * v['var_vs_HRP']:+.1f}%"] for k, v in mc["report"].items()])
        note = f"Monte Carlo: {mc['iterations']} iterations (the book runs 10,000), seed {mc['seed']}; out-of-sample return of each allocation over 260 observations rebalanced every 22."
    except Exception:
        mtab, note = None, "The Monte Carlo run has not finished."
    out.append(item("Chapter 16 Appendices", "A.1 correlation-based metric; A.2 inverse-variance allocation; A.3 and A.4 reproducing the example and the Monte Carlo", "formula", text=txt, table=mtab, note=note))
    return out


def build():
    return ch16()


if __name__ == "__main__":
    its = build()
    json.dump(its, open(W.OUT / "items_ch16.json", "w"), indent=1, default=str)
    print(len(its), "items")
