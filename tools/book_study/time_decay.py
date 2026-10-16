"""Time-decay factors for the training weights (book Ch.4, the idea of Snippet 4.11), written here from its definition.

Definition used.  For a set of events ordered by start time, let u_1..u_n be their average uniqueness (the share of its life span an event does not
share with other events) and U_j = u_1 + ... + u_j the cumulative uniqueness, U_n the total.  The decay factor of event j is a straight line in U_j:
the newest event (U_n) gets 1 and the factor falls linearly with the cumulative uniqueness that lies after the event, so that

    c >= 0 :  factor_j = c + (1 - c) * U_j / U_n          (the oldest event is near c, not exactly c: it carries its own u_1 of the total)
    -1 < c < 0 :  factor_j = 1 - (U_n - U_j) / ((1 + c) * U_n), cut at 0     (the oldest |c| share of cumulative uniqueness gets weight 0)

c = 1 means no decay (all factors 1); c = 0 means weights decline linearly to about 0 for the oldest event.
The factor multiplies the event's ordinary sample weight (absolute return attribution, Snippet 4.10); it is not rescaled afterwards.  The tree learner depends on the
weights only through their ratios, but not bit for bit: after a rescaling by a factor that is not a power of two, splits that are mathematically tied can be decided
differently by floating-point rounding (checked in test_time_decay.py), so even a pure rescaling would not reproduce the control's trees.

Deviations from the book's setting, stated where they are used (run_meta_decay.py): (1) the factors are computed per training set, on the pooled events of all
instruments ordered by event start time (ties in the order of the events file), so the 'newest' event is the newest in that training set, not the newest before the
test period; (2) in combinatorial purged CV a training set can contain events later than the test group, so 'age' there is age within the training set.
"""
from __future__ import annotations

import numpy as np


def decay_factors(u, t0, c: float) -> np.ndarray:
    """Factors aligned with the inputs.  u: average uniqueness (> 0) per event; t0: event start (any sortable number); c in (-1, 1]."""
    u = np.asarray(u, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.float64)
    if u.ndim != 1 or u.shape != t0.shape:
        raise ValueError("u and t0 must be 1-d arrays of the same length")
    if len(u) == 0:
        raise ValueError("no events")
    if not (np.isfinite(u).all() and np.isfinite(t0).all() and (u > 0).all()):
        raise ValueError("u must be finite and > 0 and t0 finite")
    if not (-1.0 < c <= 1.0):
        raise ValueError("c must satisfy -1 < c <= 1")
    order = np.argsort(t0, kind="stable")
    cum = np.cumsum(u[order])
    total = float(cum[-1])
    if c >= 0:
        f_sorted = c + (1.0 - c) * cum / total
    else:
        f_sorted = np.maximum(0.0, 1.0 - (total - cum) / ((1.0 + c) * total))
    out = np.empty_like(f_sorted)
    out[order] = f_sorted
    return out
