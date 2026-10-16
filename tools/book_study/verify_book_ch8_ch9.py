"""Independent check of what book_ch8_ch9.py can be checked on without refitting the forests: (1) the event set used (kept events whose label ends before the cutoff, rebuilt from events.csv and the bar files by timestamp),
(2) the purged 3-fold and 5-fold splits rebuilt with its own code: every training event's label interval is disjoint from the test span's interval, nothing starts inside the embargo, folds are contiguous in time and cover every event once,
(3) the search table: each mean equals the mean of its stored fold scores, the winner is the maximum, tuned_forest.json equals the winner, the untuned candidate is candidate 0,
(4) the MDI column is normalised to 1 and the SFI/MDA tables have one row per feature.  It does not refit the forests (the fits themselves are the book's ported code and the project's forest, covered by their own tests).
Usage: python verify_book_ch8_ch9.py <book_ch8_ch9 out folder>   (run from the project root)"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["ffd", "log_trgt", "sadf", "ent_plugin", "ent_konto", "cs_spread"]


def epoch(s):
    return np.array([int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()) for x in s], dtype=np.int64)


def main(out: Path) -> int:
    meta = json.loads((out / "meta.json").read_text())
    src, bar_dir = Path(meta["source_run"]), Path(meta["bar_dir"])
    problems = []
    ev0 = pd.read_csv(src / "events.csv", usecols=["symbol", "t0", "t1"])
    kept = np.zeros(len(ev0), dtype=bool)
    for s, g in ev0.groupby("symbol"):
        b = pd.read_csv(bar_dir / "bars" / f"{s}.csv", keep_default_na=False, na_values=[""], usecols=["date", "close"]).dropna(subset=["close"])
        b = b[b["close"] > 0]
        t, c = epoch(b["date"]), b["close"].to_numpy(float)
        o = np.argsort(t, kind="stable")
        t, c = t[o], c[o]
        k = np.concatenate([[True], np.diff(t) > 0])
        t, c = t[k], c[k]
        a, e = np.searchsorted(t, g["t0"].to_numpy()), np.searchsorted(t, g["t1"].to_numpy())
        kept[g.index.to_numpy()] = c[e] / c[a] - 1.0 != 0.0
    cut = int(pd.Timestamp(meta["cutoff"]).timestamp())
    ev = ev0[kept].reset_index(drop=True)
    ev = ev[ev["t1"] < cut]
    if len(ev) != meta["events_used"]:
        problems.append(f"events used {len(ev)} vs {meta['events_used']}")
    # the events are in the order of the saved table (symbol-wise as in events.csv); the splits order them by start time
    t0, t1 = ev["t0"].to_numpy(np.int64), ev["t1"].to_numpy(np.int64)
    emb = meta["embargo_pct"]
    span = float(t0.max() + 1 - t0.min())
    order = np.argsort(t0, kind="stable")
    for n_splits in (3, 5):
        parts = np.array_split(order, n_splits)
        seen = np.concatenate(parts)
        if len(set(seen.tolist())) != len(t0):
            problems.append(f"{n_splits}-fold: events not covered once")
        for q, te in enumerate(parts):
            lo, hi = t0[te].min(), t1[te].max()
            if q > 0 and t0[te].min() < t0[parts[q - 1]].max():
                problems.append(f"{n_splits}-fold {q}: test blocks not contiguous in time")
            train = np.ones(len(t0), dtype=bool)
            train[te] = False
            overlap = train & (t0 <= hi) & (t1 >= lo)
            in_emb = train & (t0 > hi) & (t0 <= hi + emb * span)
            # what the project's splitter keeps must have neither
            sys.path.insert(0, "tools/book_study")
            import meta_model as MM
            for k_, (tr_, te_) in enumerate(MM.purged_kfold(t0.astype(float), t1.astype(float), n_splits, emb)):
                if k_ != q:
                    continue
                if set(tr_.tolist()) & set(np.flatnonzero(overlap | in_emb).tolist()):
                    problems.append(f"{n_splits}-fold {q}: a training event overlaps the test span or sits in the embargo")
                if not set(np.flatnonzero(train & ~overlap & ~in_emb).tolist()) == set(tr_.tolist()):
                    problems.append(f"{n_splits}-fold {q}: training set differs from the independent rebuild")
                if set(te_.tolist()) != set(te.tolist()):
                    problems.append(f"{n_splits}-fold {q}: test set differs")
    R = pd.read_csv(out / "search.csv")
    for _, r in R.iterrows():
        m = float(np.mean(json.loads(r["fold_scores"])))
        if abs(m - r["mean_neg_log_loss"]) > 1e-12:
            problems.append(f"candidate {int(r['cand'])}: mean differs from its fold scores")
    tj = json.loads((out / "tuned_forest.json").read_text())
    best = int(R["mean_neg_log_loss"].idxmax())
    if tj["chosen_candidate"] != best or abs(tj["forest"]["min_weight_fraction_leaf"] - R.loc[best, "min_weight_fraction_leaf"]) > 1e-15:
        problems.append("tuned_forest.json is not the winner of search.csv")
    if not (R.loc[0, "min_weight_fraction_leaf"] == 0.05 and R.loc[0, "max_features"] == "sqrt"):
        problems.append("candidate 0 is not the untuned setting")
    mdi = pd.read_csv(out / "mdi.csv", index_col=0)
    if abs(float(mdi["mean"].sum()) - 1.0) > 1e-9 or list(mdi.index) != FEATURES:
        problems.append("MDI is not normalised or lacks a feature")
    for nm in ("mda.csv", "sfi.csv"):
        if list(pd.read_csv(out / nm, index_col=0).index) != FEATURES:
            problems.append(f"{nm}: features differ")
    msg = f"{meta['venue']}: ch8/ch9 structure checks ({len(ev):,} events, 3- and 5-fold purging rebuilt, {len(R)} candidates); {len(problems)} problems"
    print(msg)
    for p in problems:
        print("  PROBLEM:", p)
    (out / "checker_output.txt").write_text(msg + "\n" + "\n".join(problems) + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
