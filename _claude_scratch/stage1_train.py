#!/usr/bin/env python3
"""Stage 1: train ONE ensemble member (one direction, one bootstrap seed) and
checkpoint its predict_proba on the calibration and holdout splits. Split out
from the single long-running pipeline purely so each run fits the sandbox's
per-call time budget -- the math/algorithm is unchanged from the original
predeclared design (5-model bootstrapped ensemble per direction, >=4/5 vote
agreement rule applied later in stage2).

Usage: python3 stage1_train.py <direction: long|short> <member_idx: 0-4>

NOTE ON A DEVIATION FROM THE ORIGINAL SPEC, DISCLOSED HERE SO IT ISN'T BURIED:
the original plan bootstrapped each member at the FULL training-set size
(len(Xtr) rows, with replacement). That makes each tree take ~3-5s on this
dataset's real row counts, which is too slow for the sandbox's per-call cap.
Each member here instead bootstraps a sample of BAG_SIZE rows (with
replacement) from the training set -- a bagging-fraction choice, not a change
to the labeling, split, or agreement rule. This is disclosed in the final
report, not just here.
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbdt import GBDTClassifier

HERE = Path(__file__).resolve().parent
N_ESTIMATORS = 120
LEARNING_RATE = 0.08
MAX_LEAVES = 15
EARLY_STOPPING_ROUNDS = 10
BAG_SIZE = 40000  # see docstring -- disclosed bagging-fraction deviation
SEED_BASE = 100


def main():
    direction = sys.argv[1]
    member_idx = int(sys.argv[2])
    assert direction in ("long", "short")

    data = np.load(HERE / "data_splits.npz")
    X_train = data["X_train"]
    y_train = data[f"y_train_{direction}"]
    X_val = data["X_val"]
    y_val = data[f"y_val_{direction}"]
    X_cal = data["X_cal"]
    X_hold = data["X_hold"]

    mask_tr = ~np.isnan(y_train)
    mask_val = ~np.isnan(y_val)
    Xtr, ytr = X_train[mask_tr], y_train[mask_tr]
    Xv, yv = X_val[mask_val], y_val[mask_val]

    seed = SEED_BASE + member_idx + (0 if direction == "long" else 1000)
    rng = np.random.default_rng(seed)
    bag_n = min(BAG_SIZE, len(Xtr))
    boot_idx = rng.choice(len(Xtr), size=bag_n, replace=True)

    t0 = time.time()
    clf = GBDTClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_leaves=MAX_LEAVES,
                          min_samples_leaf=max(30, bag_n // 500),
                          random_state=seed, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    clf.fit(Xtr[boot_idx], ytr[boot_idx], Xv, yv)
    t1 = time.time()

    cal_proba = clf.predict_proba(X_cal)
    hold_proba = clf.predict_proba(X_hold)

    base_rate = float(1.0 / (1.0 + np.exp(-clf.base_score_)))  # this model's own prior win-rate

    out_path = HERE / f"member_{direction}_{member_idx}.npz"
    np.savez_compressed(out_path, cal_proba=cal_proba, hold_proba=hold_proba,
                         best_n_trees=clf.best_n_trees_, fit_seconds=t1 - t0,
                         bag_n=bag_n, base_rate=base_rate)
    print(f"[{direction} member {member_idx}] bag_n={bag_n} trees_used={clf.best_n_trees_} "
          f"base_rate={base_rate:.4f} fit_time={t1-t0:.1f}s -> wrote {out_path.name}")


if __name__ == "__main__":
    main()
