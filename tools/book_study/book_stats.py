"""Pure helpers for the book-procedure study (numpy / pandas / stdlib only; no project imports).

Every function names the book item it implements (Advances in Financial Machine Learning, condensed edition):
  Ch.2  Snippet 2.4   CUSUM event sampling                       -> cusum_events
  Ch.4  Snippets 4.1-4.2  concurrency and average uniqueness      -> average_uniqueness
  Ch.7  Snippets 7.1-7.2  purging and embargo                     -> train_mask
  Ch.12 Eq.39-40      combinatorial purged cross-validation       -> cpcv_splits, cpcv_paths
  Ch.13 Eq.13.5-13.7, Snippet 13.2  O-U calibration, simulation, mesh, Sharpe  -> estimate_ou, simulate_ou_paths, mesh_exit_stats
  Ch.13 Definition 2  overfit trading rule                        -> (flag built in the runner from mesh_exit_stats)
  Ch.14 Eq.52-53      PSR and DSR                                 -> psr, dsr, expected_max_sharpe
  Ch.15 Eq.56-57, Snippets 15.3-15.4  Sharpe/precision/frequency  -> bin_sr, bin_hr, bin_freq
"""
from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
MESH = tuple(0.5 * i for i in range(1, 21))     # 0.5, 1.0, ... 10.0 (in sigma units): the book's 20 x 20 mesh


# ----------------------------------------------------------------------------------------------------------
# Ch.13: O-U process, simulation and mesh
# ----------------------------------------------------------------------------------------------------------
def simulate_ou_paths(a: float, phi: float, eps: np.ndarray) -> np.ndarray:
    """P_t = a + phi * P_{t-1} + eps_t with P_0 = 0 and sigma = 1 (price units are sigma units).

    Snippet 13.2 writes p = (1-phi)*forecast + phi*p + sigma*gauss, so a = (1 - phi) * forecast.
    eps has shape (n_paths, steps); the result has the same shape, column t-1 holding P_t."""
    n, steps = eps.shape
    out = np.empty((n, steps), dtype=np.float64)
    p = np.zeros(n, dtype=np.float64)
    for t in range(steps):
        p = a + phi * p + eps[:, t]
        out[:, t] = p
    return out


def first_exit(paths: np.ndarray, pt: float, sl: float):
    """Exit per path under profit-taking pt and stop-loss sl (both positive, sigma units).

    The position leaves at the first column where the path is above +pt or below -sl (strict, like Snippet 13.2),
    otherwise at the last column (the vertical barrier). Returns (exit_value, step, kind) with step counted from 1
    and kind 1 = profit-taking, -1 = stop-loss, 0 = vertical barrier."""
    n, steps = paths.shape
    hit_pt = paths > pt
    hit_sl = paths < -sl
    out = hit_pt | hit_sl
    any_out = out.any(axis=1)
    idx = np.where(any_out, out.argmax(axis=1), steps - 1)
    rows = np.arange(n)
    value = paths[rows, idx]
    kind = np.where(any_out, np.where(hit_pt[rows, idx], 1, -1), 0)
    return value, idx + 1, kind


def mesh_exit_stats(paths: np.ndarray, pts=MESH, sls=MESH, cost: np.ndarray | None = None) -> dict:
    """Mean, standard deviation and Sharpe of the exit value for every (profit-taking, stop-loss) node.

    Equivalent to calling first_exit on every node, but computes the first-passage step of each level once
    (running maximum / minimum are monotone, so the first step above a level is the count of steps at or below it).
    cost, if given, is a per-path amount (round-trip cost in sigma units); the result then also carries
    mean_net / std_net / sharpe_net, the same statistics after subtracting cost from every exit value."""
    n, steps = paths.shape
    runmax = np.maximum.accumulate(paths, axis=1)
    runmin = np.minimum.accumulate(paths, axis=1)
    t_pt = np.stack([(runmax <= pt).sum(axis=1) for pt in pts])      # (n_pt, n); == steps when never reached
    t_sl = np.stack([(runmin >= -sl).sum(axis=1) for sl in sls])
    del runmax, runmin
    rows = np.arange(n)
    shape = (len(pts), len(sls))
    mean, std, pt_share = np.empty(shape), np.empty(shape), np.empty(shape)
    mean_n, std_n = np.empty(shape), np.empty(shape)
    for i in range(len(pts)):
        for j in range(len(sls)):
            first_pt, first_sl = t_pt[i], t_sl[j]
            first = np.minimum(first_pt, first_sl)
            tau = np.minimum(first, steps - 1)
            x = paths[rows, tau].astype(np.float64)
            mean[i, j] = x.mean()
            std[i, j] = x.std()
            if cost is not None:
                xn = x - cost
                mean_n[i, j] = xn.mean()
                std_n[i, j] = xn.std()
            barrier = first < steps
            nb = int(barrier.sum())
            pt_share[i, j] = float(((first_pt < first_sl) & barrier).sum()) / nb if nb else float("nan")

    def ratio(m, s):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, m / s, np.nan)

    out = {"mean": mean, "std": std, "sharpe": ratio(mean, std), "pt_share_of_barrier_exits": pt_share,
           "pts": tuple(pts), "sls": tuple(sls)}
    if cost is not None:
        out.update({"mean_net": mean_n, "std_net": std_n, "sharpe_net": ratio(mean_n, std_n)})
    return out


def best_node(sharpe: np.ndarray, pts=MESH, sls=MESH):
    """(pt, sl, sharpe) of the highest-Sharpe node (Equation 44). Ties go to the first node in (pt, sl) order."""
    if not np.isfinite(sharpe).any():
        return None
    flat = int(np.nanargmax(sharpe))
    i, j = divmod(flat, sharpe.shape[1])
    return float(pts[i]), float(sls[j]), float(sharpe[i, j])


def estimate_ou(paths: np.ndarray, weights: np.ndarray | None = None, chunk: int = 20000) -> dict:
    """Step 1 of the Chapter 13 algorithm on observed paths (Equations 13.5-13.7), pooled over events and steps.

    X = P_{t-1} (P_0 = 0), Y = P_t. phi = cov(Y, X) / cov(X, X) as in Equation 13.7; the intercept a = mean(Y) - phi*mean(X)
    is the per-step drift, equal to (1 - phi) * E0[P_T] in the book's notation; sigma is the residual standard deviation.
    weights (per path, e.g. Chapter 4 average uniqueness) make the pooled moments weighted. Computed in chunks of
    `chunk` paths so that large event sets need little memory."""
    n, steps = paths.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = float(w.sum()) * steps
    if sw <= 0:
        raise ValueError("weights sum to zero")

    def blocks():
        for lo in range(0, n, chunk):
            y = np.asarray(paths[lo:lo + chunk], dtype=np.float64)
            x = np.concatenate([np.zeros((len(y), 1)), y[:, :-1]], axis=1)
            yield x, y, w[lo:lo + chunk][:, None]

    sx = sy = 0.0
    for x, y, wc in blocks():
        sx += float((wc * x).sum())
        sy += float((wc * y).sum())
    mx, my = sx / sw, sy / sw
    sxx = sxy = 0.0
    for x, y, wc in blocks():
        sxx += float((wc * (x - mx) ** 2).sum())
        sxy += float((wc * (y - my) * (x - mx)).sum())
    vxx, cyx = sxx / sw, sxy / sw
    phi = cyx / vxx if vxx > 0 else float("nan")
    a = my - phi * mx
    srr = 0.0
    for x, y, wc in blocks():
        srr += float((wc * (y - a - phi * x) ** 2).sum())
    return {"phi": phi, "a": a, "sigma": math.sqrt(srr / sw), "n_paths": int(n), "steps": int(steps)}


def half_life(phi: float) -> float:
    """Half-life of the O-U process, tau = -log 2 / log phi, defined for phi in (0, 1)."""
    return -math.log(2.0) / math.log(phi) if 0.0 < phi < 1.0 else float("inf")


# ----------------------------------------------------------------------------------------------------------
# Ch.2 / Ch.4 / Ch.7 / Ch.12: sampling, uniqueness, purging, combinatorial CV
# ----------------------------------------------------------------------------------------------------------
def cusum_events(returns: np.ndarray, h) -> np.ndarray:
    """Symmetric CUSUM filter (Snippet 2.4) with a per-bar threshold h (scalar or array); returns bar positions.

    Like the snippet, only the side that fires is reset. Non-finite returns count as zero."""
    r = np.asarray(returns, dtype=np.float64)
    thr = np.broadcast_to(np.asarray(h, dtype=np.float64), r.shape)
    s_pos = s_neg = 0.0
    events = []
    for i in range(1, len(r)):
        x = r[i] if np.isfinite(r[i]) else 0.0
        t = thr[i]
        s_pos, s_neg = max(0.0, s_pos + x), min(0.0, s_neg + x)
        if not np.isfinite(t):
            continue
        if s_neg < -t:
            s_neg = 0.0
            events.append(i)
        elif s_pos > t:
            s_pos = 0.0
            events.append(i)
    return np.asarray(events, dtype=np.int64)


def average_uniqueness(t0: np.ndarray, t1: np.ndarray, n_bars: int) -> np.ndarray:
    """Average uniqueness of each event (Snippets 4.1-4.2): mean over its bars t0..t1 (inclusive) of 1 / concurrency."""
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    diff = np.zeros(n_bars + 1, dtype=np.float64)
    np.add.at(diff, t0, 1.0)
    np.add.at(diff, np.minimum(t1 + 1, n_bars), -1.0)
    conc = np.cumsum(diff)[:n_bars]
    inv = np.where(conc > 0, 1.0 / np.where(conc > 0, conc, 1.0), 0.0)
    pref = np.concatenate([[0.0], np.cumsum(inv)])
    return (pref[np.minimum(t1 + 1, n_bars)] - pref[t0]) / (t1 - t0 + 1)


def cpcv_splits(n_groups: int, k: int) -> list:
    """All ways to choose k test groups out of n_groups (Equation 39), in lexicographic order."""
    return [tuple(c) for c in combinations(range(n_groups), k)]


def cpcv_paths(n_groups: int, k: int):
    """Assign each (split, test group) to one backtest path (Figure 12.2).

    Returns (assignment, n_paths) where assignment[(split_index, group)] = path_index. Every path contains every group
    exactly once, and n_paths = k / N * C(N, N-k) = C(N-1, k-1) (Equation 40)."""
    splits = cpcv_splits(n_groups, k)
    counter = [0] * n_groups
    assign = {}
    for si, test in enumerate(splits):
        for g in test:
            assign[(si, g)] = counter[g]
            counter[g] += 1
    n_paths = math.comb(n_groups - 1, k - 1)
    if any(c != n_paths for c in counter):
        raise AssertionError("CPCV path assignment is inconsistent")
    return assign, n_paths


def group_edges(t_min: float, t_max: float, n_groups: int) -> np.ndarray:
    """Edges of n_groups contiguous, equal-duration time groups between t_min and t_max."""
    return np.linspace(t_min, t_max, n_groups + 1)


def assign_groups(t: np.ndarray, edges: np.ndarray) -> np.ndarray:
    g = np.searchsorted(edges, t, side="right") - 1
    return np.clip(g, 0, len(edges) - 2)


def train_mask(t0: np.ndarray, t1: np.ndarray, groups: np.ndarray, test_groups, edges: np.ndarray, embargo: float) -> np.ndarray:
    """Training events for one split: not in a test group, not overlapping a test group's time span (purge, Snippet 7.1),
    and not starting within `embargo` after a test span's end (Snippet 7.2). t0/t1 are event start/end times."""
    keep = ~np.isin(groups, list(test_groups))
    for g in test_groups:
        lo, hi = edges[g], edges[g + 1]
        keep &= ~((t0 <= hi) & (t1 >= lo))
        keep &= ~((t0 > hi) & (t0 <= hi + embargo))
    return keep


# ----------------------------------------------------------------------------------------------------------
# Ch.14: PSR / DSR
# ----------------------------------------------------------------------------------------------------------
_N = NormalDist()


def sharpe_moments(x) -> dict:
    """Per-observation Sharpe ratio, skewness and (non-excess) kurtosis of a series."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 4:
        return {"n": n, "mean": float("nan"), "std": float("nan"), "sr": float("nan"), "skew": float("nan"), "kurt": float("nan")}
    m, s = float(a.mean()), float(a.std(ddof=1))
    if s <= 0:
        return {"n": n, "mean": m, "std": s, "sr": float("nan"), "skew": float("nan"), "kurt": float("nan")}
    z = (a - a.mean()) / a.std(ddof=0)
    return {"n": n, "mean": m, "std": s, "sr": m / s, "skew": float((z ** 3).mean()), "kurt": float((z ** 4).mean())}


def psr(sr: float, t: float, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Equation 52); sr and sr_star per observation, t observations, kurt non-excess."""
    if not all(np.isfinite([sr, t, skew, kurt, sr_star])) or t <= 1:
        return float("nan")
    var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if var <= 0:
        return float("nan")
    return _N.cdf((sr - sr_star) * math.sqrt(t - 1.0) / math.sqrt(var))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """SR* of Equation 53: expected maximum Sharpe ratio of n_trials unskilled strategies whose Sharpe estimates have variance var_sr."""
    if n_trials < 2 or not np.isfinite(var_sr) or var_sr < 0:
        return float("nan")
    return math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * _N.inv_cdf(1.0 - 1.0 / n_trials)
                                + EULER_GAMMA * _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e)))


def dsr(sr: float, t: float, skew: float, kurt: float, n_trials: int, var_sr: float) -> float:
    """Deflated Sharpe ratio: PSR evaluated against the expected maximum Sharpe of n_trials trials."""
    return psr(sr, t, skew, kurt, sr_star=expected_max_sharpe(n_trials, var_sr))


# ----------------------------------------------------------------------------------------------------------
# Ch.15: Sharpe, precision and frequency of a two-outcome bet (sl is NEGATIVE, as in Snippets 15.3-15.4)
# ----------------------------------------------------------------------------------------------------------
def bin_sr(sl: float, pt: float, freq: float, p: float) -> float:
    """Equation 56: Sharpe ratio of n = freq bets that win pt with probability p and lose |sl| otherwise."""
    if not 0.0 < p < 1.0:
        return float("nan")
    return ((pt - sl) * p + sl) / ((pt - sl) * math.sqrt(p * (1.0 - p))) * math.sqrt(freq)


def bin_hr(sl: float, pt: float, freq: float, t_sr: float) -> float:
    """Snippet 15.3: minimum precision p needed to reach Sharpe ratio t_sr."""
    a = (freq + t_sr ** 2) * (pt - sl) ** 2
    b = (2.0 * freq * sl - t_sr ** 2 * (pt - sl)) * (pt - sl)
    c = freq * sl ** 2
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return float("nan")
    return (-b + math.sqrt(disc)) / (2.0 * a)


def bin_freq(sl: float, pt: float, p: float, t_sr: float) -> float:
    """Snippet 15.4: number of bets needed to reach Sharpe ratio t_sr with precision p (None if the root is extraneous)."""
    f = (t_sr * (pt - sl)) ** 2 * p * (1.0 - p) / ((pt - sl) * p + sl) ** 2
    if not np.isclose(bin_sr(sl, pt, f, p), t_sr):
        return float("nan")
    return f


def break_even_precision(pt: float, sl: float, cost: float = 0.0) -> float:
    """Share of barrier exits that must be profit-takings for a zero expected net result, (sl + cost) / (pt + sl) (all positive)."""
    return (sl + cost) / (pt + sl)
