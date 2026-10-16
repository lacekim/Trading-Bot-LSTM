#!/usr/bin/env python
"""Reference for a run_hp_search.py output folder: the pooled out-of-fold weighted log loss of a CONSTANT prediction (stdlib only).

For each of the five purged folds (verify_hp_search.purged_folds, same rule as the search) the prediction for every test event is the weighted share of label 1 among that fold's
training events; the pooled loss uses the same formula and clip as the search.  Also prints the in-sample constant at the whole search set's base rate.  Writes constant_baseline.json
into the folder.  Descriptive only; it is not part of the selection.

    python tools/book_study/hp_constant_baseline.py runtime/book_hp_search/<timestamp> [source run folder, only if the recorded path does not resolve]
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_hp_search as VH  # noqa: E402


def main(folder, run_override=None):
    d = Path(folder)
    ex = json.loads((d / "hp_extra.json").read_text(encoding="utf-8"))
    run = Path(run_override) if run_override else Path(ex["source_run"])
    summ = json.loads((run / "summary_numbers.json").read_text(encoding="utf-8"))
    cut = float(VH.V.ts_key(summ["params"]["cutoff"]))
    ev = list(csv.DictReader(open(run / "events.csv", newline="", encoding="utf-8")))
    t0, t1 = [int(float(r["t0"])) for r in ev], [int(float(r["t1"])) for r in ev]
    y, w = [int(float(r["bin"])) for r in ev], [float(r["w"]) for r in ev]
    pre = [i for i in range(len(ev)) if t1[i] < cut]
    folds = VH.purged_folds(t0, t1, pre, VH.N_FOLDS, VH.EMBARGO)
    ys, ps, ws = [], [], []
    for tr, te in folds:
        pb = sum(y[i] * w[i] for i in tr) / sum(w[i] for i in tr)
        for i in te:
            ys.append(y[i]); ps.append(pb); ws.append(w[i])
    base = sum(y[i] * w[i] for i in pre) / sum(w[i] for i in pre)
    out = {"source_run": str(run), "n_search_events": len(pre), "oof_constant_log_loss_w": VH.wll(ys, ps, ws), "in_sample_base_rate_w": base,
           "in_sample_constant_log_loss_w": VH.wll([y[i] for i in pre], [base] * len(pre), [w[i] for i in pre]), "constant_half_log_loss_w": VH.wll([y[i] for i in pre], [0.5] * len(pre), [w[i] for i in pre])}
    (d / "constant_baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    sc = list(csv.DictReader(open(d / "config_scores.csv", newline="", encoding="utf-8")))
    best = min(float(r["log_loss_w"]) for r in sc)
    for k, v in out.items():
        print(f"{k}: {v}")
    print(f"configs with out-of-fold log loss below the out-of-fold constant: {sum(1 for r in sc if float(r['log_loss_w']) < out['oof_constant_log_loss_w'])} of {len(sc)} (best config {best:.6f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
