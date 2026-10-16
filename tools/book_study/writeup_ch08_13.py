"""Write-up items for Chapters 8-13 (see writeup_core.py).  New file; the simulations it plots come from writeup_sims.py."""
from __future__ import annotations

import itertools
import json
import math
import re
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr, weightedtau

import book_port_b as PB
import book_port_c as C
import book_stats as B
import writeup_core as W
from writeup_core import VENUES, SYM, item, tbl, fmt, save

PHI = NormalDist().cdf


def meta(v):
    return W.run_json(v, "meta.json")


def runlog(v):
    return (W.RUNS[v] / "run.log").read_text()


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch8():
    out = []
    out.append(item("Snippet 8.2", "Mean decrease impurity (featImpMDI)", "code", text="book_port_b.featImpMDI (verbatim); test_book_port_b.test_mdi_mda_sfi_find_the_informative_features passes."))
    out.append(item("Snippet 8.3", "Mean decrease accuracy (featImpMDA)", "code", text="book_port_b.featImpMDA (verbatim)."))
    out.append(item("Snipper 8.4", "Single feature importance (auxFeatImpSFI), the book's contents entry is spelled 'Snipper'", "code", text="book_port_b.auxFeatImpSFI (verbatim)."))
    out.append(item("Snippet 8.5", "Computation of orthogonal features", "code", text="book_port_b.get_eVec / orthoFeats (verbatim); test_book_port_b.test_orthogonal_features_are_uncorrelated passes."))

    def draw81(ax, v):
        o = W.run_csv(v, "mdi_orthogonal.csv", index_col=0)
        x, y = o["eigenvalue"].to_numpy(), o["mean"].to_numpy()
        ax.scatter(x, y, color="0.4", s=14); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("eVal"); ax.set_ylabel("MDI feature importance")
        k, s, p = kendalltau(x, y)[0], spearmanr(x, y)[0], pearsonr(x, y)[0]
        wk = weightedtau(x, y)[0]
        ax.set_title(f"{v}: kendall {k:.4f}, spearman {s:.3f}, pearson {p:.4f}, wKendall {wk:.4f}", fontsize=6)
    out.append(item("Figure 8.1", "Eigenvalues (x) versus MDI importance (y) of the orthogonalised features, log-log (our 10 principal components, not the book's 1,000 simulated ones)", "data", fig=W.two_panel(draw81, "f8_1"),
                    note="The book scatters many simulated runs; our run has one set of components per venue."))
    taus = {v: float(re.search(r"weighted tau\(MDI, inverse PCA rank\) = ([-0-9.]+)", runlog(v)).group(1)) for v in VENUES}
    out.append(item("Snippet 8.6", "Weighted Kendall's tau between feature importance and inverse PCA rank", "code",
                    table=tbl(["Venue", "weighted tau (MDI vs inverse PCA rank), from run.log"], [[v, fmt(taus[v], 3)] for v in VENUES]),
                    text="book_port_b.weighted_tau_example reproduces the snippet's own example (featImp .55 .33 .07 .05 against ranks 1 2 4 3): " + fmt(PB.weighted_tau_example(), 4) + ". The extract prints no value for it."))
    out.append(item("Snippet 8.7", "Creating a synthetic dataset", "code", text="book_port_b.getTestData (verbatim; 40 features, 10 informative, 10 redundant)."))
    out.append(item("Snippet 8.8", "Calling feature importance for any method (featImportance)", "code", text="book_port_b.featImportance (verbatim, with the max_samples integer note in the code)."))
    out.append(item("Snippet 8.9", "Calling all components (testFunc)", "code", text="book_port_b.testFunc (verbatim)."))
    out.append(item("Snippet 8.10", "Feature importance plotting function", "code", text="book_port_b.plotFeatImportance (verbatim)."))
    s8 = json.loads((W.OUT / "synth8.json").read_text()) if (W.OUT / "synth8.json").exists() else None

    def bars(ax, imp, title, mdi=False):
        imp = imp.sort_values("mean")
        ax.barh(range(len(imp)), imp["mean"], xerr=imp["std"], color="0.7", ecolor="0.2", error_kw={"lw": 0.5}, height=0.8)
        ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index, fontsize=4 if len(imp) > 20 else 6); ax.set_title(title, fontsize=6)
        if mdi:
            ax.axvline(1 / len(imp), color="0.2", ls=":", lw=0.8)

    for fid, m, ttl in (("Figure 8.2", "MDI", "MDI feature importance"), ("Figure 8.3", "MDA", "MDA feature importance"), ("Figure 8.4", "SFI", "SFI feature importance")):
        fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.6))
        if s8:
            imp = pd.read_csv(W.OUT / f"synth8_{m}.csv", index_col=0)
            bars(axs[0], imp, f"synthetic (book's setup, reduced) oob {s8[m]['oob']:.3f} oos {s8[m]['oos']:.3f}", m == "MDI")
        for ax, v in zip(axs[1:], VENUES):
            src = {"MDI": "mdi.csv", "MDA": "mda.csv", "SFI": "sfi.csv"}[m]
            d = W.run_csv(v, src, index_col=0).dropna()
            bars(ax, d, f"{v}: {ttl}", m == "MDI")
        fig.tight_layout()
        out.append(item(fid, f"{ttl}: the book's synthetic dataset (left, reduced size) and our two venues", "data", fig=save(fig, "f8_" + m),
                        note="Synthetic panel: n_samples 3,000, 100 trees, 5 folds (book: 10,000, 1,000, 10). Informative (I_) and redundant (R_) features rise above the noise (N_) bars, as in the book."))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch9():
    out = []
    out.append(item("Snippet 9.1", "Grid search with purged k-fold CV", "code", text="book_port_b.clfHyperFit (grid branch; verbatim)."))
    out.append(item("Snippet 9.2", "An enhanced pipeline class (MyPipeline)", "code", text="book_port_b.MyPipeline: a scikit-learn Pipeline whose fit accepts sample_weight, which the plain Pipeline does not pass through to its last step."))
    out.append(item("Snippet 9.3", "Randomized search with purged k-fold CV", "code", text="book_port_b.clfHyperFit (randomized branch, with the bagging option). The bot's search: 10 random draws plus the untuned setting = 11 candidates, scored by neg_log_loss (book_ch8_ch9.py)."))
    out.append(item("Function 2", "Log loss (Equation under Log 1)", "formula", text="L = -N^-1 sum_n sum_k y_nk log p_nk (scikit-learn's neg_log_loss is its negative)."))
    ks, desc = PB.test_logUniform(seed=3)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    v = PB.logUniform(a=1e-3, b=1e3).rvs(size=10000, random_state=3)
    axs[0].hist(np.log(v), bins=20, color="0.4"); axs[1].hist(v, bins=20, color="0.4")
    out.append(item("Figure 9.1", "Result from testing the logUniform_gen class (10,000 draws between 1e-3 and 1e3: log of the draws on the left, the draws on the right)", "formula", fig=save(fig, "f9_1"),
                    text=f"Kolmogorov-Smirnov of log(draws) against the uniform: statistic {ks.statistic:.4f}, p-value {ks.pvalue:.3f} (seed 3)."))
    out.append(item("Snippet 9.4", "The logUniform_gen class", "code", text="book_port_b.logUniform_gen / logUniform / test_logUniform; test_book_port_b.test_log_uniform_passes_its_own_ks_test passes."))
    rows = []
    for v in VENUES:
        s = W.run_csv(v, "search.csv")
        for _, r in s.iterrows():
            rows.append([v, int(r["cand"]), r["min_weight_fraction_leaf"], r["max_features"], fmt(r["mean_neg_log_loss"], 5), r["note"]])
    out.append(item("Log 1", "Our randomised search: each candidate's mean purged-CV neg_log_loss (higher is better)", "data", table=tbl(["Venue", "cand", "min_weight_fraction_leaf", "max_features", "mean neg_log_loss", "note"], rows)))
    ph = np.linspace(0.5, 0.9, 80); pm = np.linspace(0.5, 0.9, 80)
    z = -(0.5 * np.log(ph[:, None]) + 0.5 * np.log(1 - pm[None, :]))
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    im = ax.imshow(z, origin="lower", extent=[0.5, 0.9, 0.5, 0.9], aspect="auto", cmap="Greys"); plt.colorbar(im)
    cs = ax.contour(pm, ph, z, colors="k", linewidths=0.4)
    ax.set_xlabel("Predicted probability of a miss"); ax.set_ylabel("Predicted probability of a hit"); ax.set_title("Log loss")
    out.append(item("Figure 9.2", "Log loss as a function of the predicted probabilities of a hit and of a miss", "formula", fig=save(fig, "f9_2"),
                    note="Closed form: L = -(1/2) ln p_hit - (1/2) ln(1 - p_miss), i.e. equal numbers of hits and misses (the extract does not state the mix; this reproduces the book's range, about 0.4 to 1.5)."))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch10():
    out = []
    out.append(item("Equation 36", "Bet size from the concurrency of long and short bets (mixture of two Gaussians)", "formula", text="An alternative the book offers in Section 10.2; not used. The bot averages active bets (Snippet 10.2) instead."))
    out.append(item("Section 10.3", "Bet sizing from predicted probabilities", "text", text="z = (p - 1/2)/sqrt(p(1-p)), m = 2 Z[z] - 1 (Snippet 10.1)."))
    z = np.linspace(-3, 3, 400)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    axs[0].plot(z, 2 * np.vectorize(PHI)(z) - 1, color="0.2"); axs[0].axvline(0, color="0.3", lw=2)
    axs[0].set_xlabel("z"); axs[0].set_ylabel("Bet size m = 2 Z[z] - 1")
    m = 2 * np.vectorize(PHI)(z) - 1
    axs[1].step(z, np.clip(np.round(m / 0.2) * 0.2, -1, 1), color="0.2"); axs[1].axvline(0, color="0.3", lw=2); axs[1].set_xlabel("z"); axs[1].set_title("discretised, d = 0.2")
    fp = save(fig, "f10_1_2")
    out.append(item("Figure 10.1", "Bet size from predicted probabilities (left panel)", "formula", fig=fp, note="Closed form, same curve as the book."))
    out.append(item("Snippet 10.1", "From probabilities to bet size (getSignal)", "code", text="book_port_c.getSignal (verbatim)."))
    out.append(item("Snippet 10.2", "Bets are averaged as long as they are open (avgActiveSignals)", "code", text="book_port_c.avgActiveSignals / mpAvgActiveSignals (verbatim); the bot's vectorised average_positions in bet_averaging.py is tested against it."))
    out.append(item("Figure 10.2", "Discretization of the bet size, d = 0.2 (right panel of the figure above)", "formula", fig=fp))
    out.append(item("Snippet 10.3", "Size discretization to prevent overtrading (discreteSignal)", "code", text="book_port_c.discreteSignal (verbatim); step 0.1 in the bot."))
    out.append(item("Section 10.6", "Dynamic bet sizes and limit prices", "text", text="Forecast price f, market price p, sigmoid size m[omega, x] with w calibrated so that a divergence x = 10 gives m = 0.95; Snippet 10.4."))
    w, tPos, lP = C.main_10_4()
    rows = [["calibrated w (divergence 10 -> 0.95)", fmt(w, 4)], ["order size, f=115, p=100, maxPos=100 (book: 97)", tPos], ["limit price (book: 112.3657)", fmt(lP, 4)], ["order size for f=110 (book: 95)", C.getTPos(w, 110, 100, 100)]]
    out.append(item("Snippet 10.4", "Dynamic position size and limit price", "code", table=tbl(["Quantity", "ours"], rows), text="book_port_c.main_10_4 and the bot's order tickets (orders_10_4.csv). The book's printed values 97, 112.3657 and 95 are reproduced."))
    x = np.linspace(-1, 1, 400)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.plot(x, np.sign(x) * np.abs(x) ** 2, color="0.2"); ax.plot(x, x * (0.1 + x ** 2) ** -0.5 / (1.1 ** -0.5), color="0.2", ls="--")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    out.append(item("Figure 10.3", "f[x] = sgn[x]|x|^2 (concave to convex, solid) and f[x] = x(.1 + x^2)^-1/2 (convex to concave, dashed, scaled to reach 1 at x = 1)", "formula", fig=save(fig, "f10_3")))
    out.append(item("Equation 37", "Number of combinations of S submatrices, S choose S/2 (S = 16)", "formula",
                    table=tbl(["", "value"], [["book's printed text", "12,780"], ["Equation 37 evaluated: C(16, 8)", "12,870"]]),
                    note="The printed 12,780 is a transposition of the equation's own value 12,870."))
    # our data: bet sizes
    def draw(ax, v):
        p = W.run_csv(v, "walk_forward_predictions.csv")
        pr = p["p"].to_numpy()
        zz = (pr - 0.5) / np.sqrt(pr * (1 - pr))
        mm = np.round((2 * np.vectorize(PHI)(zz) - 1) / 0.1) * 0.1
        ax.hist(np.abs(mm), bins=np.arange(-0.05, 1.06, 0.1), color="0.4"); ax.set_xlabel("|bet size| (step 0.1)"); ax.set_ylabel("walk-forward events")
    try:
        out.append(item("Figure 10.1 (ours)", "Distribution of our walk-forward bet sizes from Snippet 10.1 / 10.3", "data", fig=W.two_panel(draw, "f10_ours", titles={v: v for v in VENUES})))
    except Exception as ex:
        out.append(item("Figure 10.1 (ours)", "Distribution of our walk-forward bet sizes", "data", text=f"not drawn: {type(ex).__name__}: {ex}"))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch11():
    out = []
    r = json.loads((W.OUT / "cscv.json").read_text()); a = np.load(W.OUT / "cscv.npz")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axs[0].scatter(a["sr_is"], a["sr_oos"], s=2, color="0.3", alpha=0.6); xs = np.array([a["sr_is"].min(), a["sr_is"].max()]); axs[0].plot(xs, r["intercept"] + r["slope"] * xs, color="0.1", ls="--", lw=0.8)
    axs[0].set_xlabel("SR IS"); axs[0].set_ylabel("SR OOS"); axs[0].set_title(f"[SR OOS] = {r['intercept']:.2f} {r['slope']:+.2f} [SR IS] | adjR2 {r['adj_r2']:.2f} | Prob[SR OOS<0] = {r['prob_sr_oos_lt_0']:.2f}", fontsize=6)
    axs[1].hist(a["logit"], bins=40, color="0.4", density=True); axs[1].set_xlabel("Logits"); axs[1].set_ylabel("Frequency"); axs[1].set_title(f"Prob Overfit = {r['pbo']:.2f}", fontsize=7)
    fp = save(fig, "f11")
    out.append(item("Figure 11.1", "Best Sharpe ratio in-sample versus out-of-sample (left panel)", "formula", fig=fp,
                    text=f"Combinatorially symmetric cross-validation on {r['N']} pure-noise strategies, T = {r['T']}, S = {r['S']}, {r['combinations']:,} combinations (seed {r['seed']}).",
                    note="The book's trial set is not given, so this is our noise trial set run through the same procedure; its PBO is about one half, as it must be for strategies with no skill. The book's printed 0.74 comes from its own (unstated) set."))
    out.append(item("Figure 11.2", "Probability of backtest overfitting from the distribution of logits (right panel of the figure above)", "formula", fig=fp))
    t38 = []
    for v in VENUES:
        m = W.run_json(v, "meta.json")
        T, t0 = m["events"], m["events_pre_cutoff"]
        t38.append([v, fmt(T), fmt(t0), fmt(t0 / T, 4), fmt(0.25 + 0.75 * t0 / T, 4)])
    out.append(item("Equation 38", "Average data per decision in walk-forward with a warm-up of t0 of T observations: ((T-t0)/2)^-1 (t0 + (T+t0)(T-t0)/4)... = T/4 + 3 t0 / 4", "data",
                    table=tbl(["Venue", "Events T", "Events before the cutoff t0", "t0 / T", "Share of the sample used by the average decision, 1/4 + 3 t0 / (4 T)"], t38),
                    text="Half of the decisions are made on an average of T/4 + 3 t0/4 observations, i.e. a fraction 1/4 + 3 t0/(4T) of the sample.  Our walk-forward trains on events whose labels end before 2024-01-01 (t0 above) and predicts the rest; CPCV has no such warm-up."))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch12():
    out = []
    S = B.cpcv_splits(6, 2)
    mark = [[f"G{g}"] + ["x" if (g - 1) in s else "" for s in S] + [5] for g in range(1, 7)]
    out.append(item("Figure 12.1", "Paths generated for phi[6, 2] = 5: which groups are tested in which split", "formula", table=tbl(["", *[f"S{i}" for i in range(1, 16)], "Paths"], mark), note="Same matrix as the book's figure (checked cell by cell in book_reproduce.py)."))
    out.append(item("Equation 39", "Number of splits C(N, k)", "formula", text=f"C(6, 2) = {len(S)} splits."))
    assign, nP = B.cpcv_paths(6, 2)
    rows = [[f"G{g+1}"] + [assign.get((si, g), "") + 1 if (si, g) in assign else "" for si in range(15)] + [5] for g in range(6)]
    out.append(item("Equation 40", "Number of backtest paths phi[N, k] = k/N C(N, N-k)", "formula", text=f"phi[6, 2] = {nP}."))
    out.append(item("Figure 12.2", "Assignment of testing groups to each of the 5 paths", "formula", table=tbl(["", *[f"S{i}" for i in range(1, 16)], "Paths"], rows)))
    rows2 = []
    for v in VENUES:
        sm = W.run_csv(v, "summary.csv")
        sm = sm[sm["scope"].str.startswith("cpcv") & (sm["span"] == "last_365_days")]
        for _, r in sm.iterrows():
            rows2.append([v, r["scope"], fmt(r["gain_pct"], 2), fmt(r["max_drawdown_pct"], 2), fmt(r["daily_sharpe_x_sqrt365"], 2)])
        m = meta(v)
        rows2.append([v, "AUC by path", ", ".join(fmt(x, 3) for x in m["cpcv"]["auc_by_path"]), "", ""])
    out.append(item("Equation 40 (our paths)", "Our five backtest paths, last 365 days: gain %, max drawdown %, daily Sharpe x sqrt(365)", "data", table=tbl(["Venue", "path", "gain %", "max drawdown %", "Sharpe"], rows2)))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch13():
    out = []
    ng = lambda p: NormalDist().inv_cdf(p)
    out.append(item("Equation 41", "Expected maximum of I IID standard normals: (1-g) Z^-1[1-1/I] + g Z^-1[1-1/(I e)] <= sqrt(2 log I)", "formula",
                    table=tbl(["I", "value"], [[i, fmt((1 - B.EULER_GAMMA) * ng(1 - 1 / i) + B.EULER_GAMMA * ng(1 - 1 / (i * math.e)), 4)] for i in (2, 11, 100, 1000)]), text="Evaluated with book_stats.expected_max_sharpe's formula."))
    out.append(item("Equation 42", "Expected maximum Sharpe of I backtests on a martingale: E[max x] x sigma[y]", "formula", text="Used for SR* in the deflated Sharpe ratio (Equation 53)."))
    out.append(item("Equation 43a", "Variance of the CPCV path mean: phi^-1 sigma^2 (1 + (phi - 1) rho)", "formula", text="Our var_across_cpcv_paths is saved in meta.json per span (dsr)."))
    out.append(item("Equation 43b", "Bounds: phi^-1 sigma^2 <= V[mean] < sigma^2", "formula", text="Holds by construction for rho in [0, 1)."))
    out.append(item("Equation 44", "Optimal trading rule R* = argmax Sharpe over the mesh", "formula", text="book_stats.best_node; ours is reported below."))
    out.append(item("Definition 2", "Overfit trading rule: the chosen rule's out-of-sample Sharpe is below the median out-of-sample Sharpe over the mesh", "formula", text="Evaluated on events after the cutoff for each venue (table below)."))
    out.append(item("Section 13.4", "Our framework: the discrete O-U process P_t = (1-phi) E0[P_T] + phi P_{t-1} + sigma eps (Equations 13.2-13.4)", "text",
                    text="Stationary iff phi in (-1, 1); half-life = -log 2 / log phi; phi = 2^(-1/half-life)."))
    fits = []
    for v in VENUES:
        j = W.run_json(v, "ch13.json")
        fits.append([v, fmt(j["phi"], 4), fmt(j["half_life_bars"], 1) if j["half_life_bars"] is not None else "none (phi >= 1)", fmt(j["forecast_level_sigma"], 3), fmt(j["sigma_fit"], 3), j["fit_events"], j["chosen_pt"], j["chosen_sl"], j["definition_2"]["overfit"]])
    out.append(item("Section 13.5.1", "The algorithm, Steps 1-5: our fit (Equations 13.5-13.7) and chosen rule per venue", "data",
                    table=tbl(["Venue", "phi", "half-life (bars)", "forecast (sigma)", "sigma fitted", "paths fitted", "chosen pt", "chosen sl", "Definition 2 overfit"], fits),
                    text="Out-of-sample Sharpe of the chosen rule versus the median of the mesh (Definition 2): " + "; ".join(f"{v} {fmt(W.run_json(v, 'ch13.json')['definition_2']['oos_sharpe_chosen'], 4)} vs {fmt(W.run_json(v, 'ch13.json')['definition_2']['oos_sharpe_mesh_median'], 4)}" for v in VENUES) + "."))
    out.append(item("Snippet 13.1", "Optimal trading rules, driver (main)", "code", text="book_port_c.main_13_1 (verbatim loops over forecasts {10, 5, 0, -5, -10} and half-lives {5, 10, 25, 50, 100})."))
    out.append(item("Snippet 13.2", "Optimal trading rules, batch (the verbatim loop)", "code", text="book_port_c.batch is the printed loop; the bot's vectorised simulation (book_stats) is cross-checked against it in test_book_port_c.test_mesh_shape_and_seeded_noise and in the run's ch13 step."))
    rows = []
    for k, (f, hl) in enumerate(itertools.product([0, 5, 10, -5, -10], [5, 10, 25, 50, 100]), 1):
        rows.append([f"16.{k}  (= Figure 13.{k})", f, hl, 1, 100])
    out.append(item("Table 13.1", "Input parameter combinations used in the simulations", "formula", table=tbl(["Figure (as printed)", "Forecast", "Half-life", "Sigma", "maxHP"], rows),
                    note="The book's table labels its rows 16.1-16.25; they are Figures 13.1-13.25 (a misprint in the book)."))
    npz = np.load(W.OUT / "fig13.npz")
    grid = np.linspace(0, 10, 21)
    for k, (f, hl) in enumerate(itertools.product([0, 5, 10, -5, -10], [5, 10, 25, 50, 100]), 1):
        z = npz[f"f{f}_h{hl}"]
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        im = ax.imshow(z.T, origin="lower", extent=[-0.25, 10.25, -0.25, 10.25], aspect="auto", cmap="Greys_r"); plt.colorbar(im)
        ax.set_xlabel("Profit-Taking"); ax.set_ylabel("Stop-Loss"); ax.set_title(f"Forecast={f} | H-L={hl} | Sigma=1")
        b = np.unravel_index(np.nanargmax(z), z.shape)
        out.append(item(f"Figure 13.{k}", f"Heat-map of Sharpe over the profit-taking x stop-loss mesh, {{E0[P], tau, sigma}} = {{{f}, {hl}, 1}}", "formula", fig=save(fig, f"f13_{k}"),
                        text=f"Best node: profit-taking {grid[b[0]]:.1f}, stop-loss {grid[b[1]]:.1f}, Sharpe {z[b]:.3f}.",
                        note="100,000 simulated paths, seed 13 (the book's runs are unseeded, so cell values differ by sampling noise)." if k == 1 else None))
    # our own heat-maps
    def draw(ax, v, name):
        d = pd.read_csv(W.RUNS[v] / name, header=None, keep_default_na=False, na_values=[""]).to_numpy(float)
        im = ax.imshow(d.T, origin="lower", extent=[0.25, 10.25, 0.25, 10.25], aspect="auto", cmap="Greys_r"); plt.colorbar(im, ax=ax)
        ax.set_xlabel("Profit-Taking"); ax.set_ylabel("Stop-Loss")
    try:
        out.append(item("Figure 13.26 (ours, in-sample simulation)", "Our fitted process: Sharpe over the 20 x 20 mesh on simulated paths from each venue's fitted O-U", "data", fig=W.two_panel(lambda ax, v: draw(ax, v, "ch13_mesh_sharpe_simulated.csv"), "f13_ours_sim")))
        out.append(item("Figure 13.27 (ours, out-of-sample)", "Our mesh: Sharpe of exits on real post-cutoff paths (events after 2024-01-01)", "data", fig=W.two_panel(lambda ax, v: draw(ax, v, "ch13_mesh_sharpe_oos.csv"), "f13_ours_oos")))
    except Exception as ex:
        out.append(item("Figure 13.26 (ours)", "Our mesh heat-maps", "data", text=f"not drawn: {type(ex).__name__}: {ex}"))
    return out


def build():
    items = []
    for f in (ch8, ch9, ch10, ch11, ch12, ch13):
        print("building", f.__name__, flush=True)
        items += f()
    return items


if __name__ == "__main__":
    its = build()
    json.dump(its, open(W.OUT / "items_ch08_13.json", "w"), indent=1, default=str)
    print(len(its), "items")
