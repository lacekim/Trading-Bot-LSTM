"""Prints the headline numbers of the four dollar-bar runs from their saved summary_numbers.json (nothing is recomputed).
    python tools/book_study/collect_bar_results.py <kraken meta> <alpaca meta> <kraken 3-algo> <alpaca 3-algo>
"""
import json, sys
from pathlib import Path

def f(x, nd=1):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"

def meta_block(tag, p):
    d = json.loads((Path(p) / "summary_numbers.json").read_text())
    m = d["meta"]; print(f"\n=== {tag}: meta-labeling on dollar bars ({p}) ===")
    print(f"symbols {m['symbols_used']}, events {m['events']:,}, venue {m['venue']}, cost/side {m['cost_bp_per_side']} bp, bars manifest md5 {m['bars_manifest_md5']}, prior trials {d['params']['prior_trials']}")
    c = d["cpcv"]; print("CPCV AUC mean/min/max:", f(c.get('auc_mean'),3), f(c.get('auc_min'),3), f(c.get('auc_max'),3))
    for src, key in (("CPCV", "eval"), ("WF", "walk_forward_eval")):
        for v in ("primary", "meta", "placebo"):
            s = d[key]["summary"][v]
            print(f"  {src:4s} {v:8s} bp/event {f(s['mean_bp_per_candidate'])}  daily SR {f(s['mean_sr_daily'],4)}  ann {f(s['mean_sr_annual'],2)}  neg paths {s['paths_sr_negative']}/{s['paths']}  PSR {f(s['mean_psr'],2)}  taken {f(s['taken_share'],2)}")
        print(f"  {src} DSR meta: " + ", ".join(f"N={k}: {t['dsr']:.4f}" for k, t in d[key]["dsr"]["by_trials"].items()))
    for src, key in (("CPCV", "eval"), ("WF", "walk_forward_eval")):
        for wn in m["windows"]:
            w = d[key]["windows"]
            print(f"  window {src:4s} {wn}: events/path {w[f'primary|{wn}']['events_per_path']:.0f}  " + "  ".join(f"{v}={f(w[f'{v}|{wn}']['mean_bp_per_candidate'])}" for v in ('primary', 'meta', 'placebo')))

def three_block(tag, p):
    d = json.loads((Path(p) / "summary_numbers.json").read_text())
    m = d["meta"]; print(f"\n=== {tag}: long / short / momentum / learned side / gate on dollar bars ({p}) ===")
    print(f"events {m['events']:,}, MOM events {m['mom_events']:,} (long {m['mom_long']:,}, short {m['mom_short']:,}), prior trials {d['params']['prior_trials']}")
    c = d["cpcv"]; print("AUC dir mean/min/max", f(c['auc_dir_mean'],3), f(c['auc_dir_min'],3), f(c['auc_dir_max'],3), "| AUC mom", f(c['auc_mom_mean'],3), f(c['auc_mom_min'],3), f(c['auc_mom_max'],3))
    for src, key in (("CPCV", "eval"), ("WF", "walk_forward_eval")):
        for v, s in d[key]["summary"].items():
            print(f"  {src} {v:20s} bp/event {f(s['mean_bp_per_candidate'])}  daily SR {f(s['mean_sr_daily'],4)}  ann {f(s['mean_sr_annual'],2)}  neg paths {s.get('paths_sr_negative','-')}/{s.get('paths','-')}  PSR {f(s['mean_psr'],2)}  taken {f(s['taken_share'],2)}")
    for v, x in d["eval"]["dsr"].items():
        print(f"  DSR {v}: " + ", ".join(f"N={k}: {t['dsr']:.4f}" for k, t in x['by_trials'].items()))
    for src, key in (("CPCV", "eval"), ("WF", "walk_forward_eval")):
        for wn in m["windows"]:
            w = d[key]["windows"]
            print(f"  window {src} {wn}: events/path {d[key]['candidates_window'][wn]:.0f}  " + "  ".join(f"{v}={f(w[f'{v}|{wn}']['mean_bp_per_candidate'])}" for v in d['eval']['summary']))

if __name__ == "__main__":
    a = sys.argv[1:]
    meta_block("KRAKEN", a[0]); meta_block("ALPACA", a[1]); three_block("KRAKEN", a[2]); three_block("ALPACA", a[3])
