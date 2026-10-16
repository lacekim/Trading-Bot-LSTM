"""Baseline bot (runtime/book_bot3, Snippet 3.1 volatility) against the same bot with Natenberg's Appendix B volatility (runtime/book_bot_nat).  New file.  Every number is read from the two runs' saved meta.json / summary.csv.
Usage: python tools/book_study/nat_vs_base.py [nat_kraken_dir nat_alpaca_dir]  -> writes runtime/book_bot_nat/COMPARISON.md"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = {"kraken": ROOT / "runtime/book_bot3/kraken_20261015T034703Z", "alpaca": ROOT / "runtime/book_bot3/alpaca_20261015T034703Z"}
NAT = {v: Path(sorted(glob.glob(str(ROOT / f"runtime/book_bot_nat/{v}_*")))[-1]) for v in ("kraken", "alpaca")}
if len(sys.argv) == 3:
    NAT = {"kraken": Path(sys.argv[1]), "alpaca": Path(sys.argv[2])}


def g(m, *path):
    for p in path:
        m = m[p]
    return m


def row(label, f, fmt="{:,.4f}"):
    out = [label]
    for v in ("kraken", "alpaca"):
        for d in (BASE[v], NAT[v]):
            try:
                out.append(fmt.format(f(json.load(open(d / "meta.json")))))
            except Exception as e:
                out.append(f"n/a ({type(e).__name__})")
    return out


rows = [
    row("events kept", lambda m: m["events"], "{:,.0f}"),
    row("CUSUM events", lambda m: m["events_cusum"], "{:,.0f}"),
    row("Ch.13 chosen pt", lambda m: m["otr"]["pt"], "{:.1f}"),
    row("Ch.13 chosen sl", lambda m: m["otr"]["sl"], "{:.1f}"),
    row("CPCV AUC (mean of 5)", lambda m: m["cpcv"]["auc_mean"]),
    row("walk-forward AUC", lambda m: m["walk_forward"]["auc"]),
    row("last-365-day gain %", lambda m: m["ch14"]["gain_pct"], "{:+.2f}"),
    row("max drawdown % (14.4)", lambda m: m["ch14"]["drawdowns_14_4"]["max_pct"], "{:.2f}"),
    row("annualised Sharpe (last 365)", lambda m: m["dsr"]["last_365_days"]["sr_daily"] * 365 ** 0.5, "{:+.2f}"),
    row("PSR vs 0", lambda m: m["dsr"]["last_365_days"]["psr_vs_0"]),
    row("DSR", lambda m: m["dsr"]["last_365_days"]["dsr"]),
]
def wf_row(label, span, col, fmt):
    out = [label]
    for v in ("kraken", "alpaca"):
        for d in (BASE[v], NAT[v]):
            r = [x for x in csv.DictReader(open(d / "summary.csv")) if x["scope"] == "wf" and x["span"] == span][0]
            out.append(fmt.format(float(r[col])))
    return out


rows += [wf_row("whole walk-forward span (2024-01-01 to the end): gain %", "whole_walk_forward_span", "gain_pct", "{:+.2f}"),
         wf_row("whole walk-forward span: max drawdown %", "whole_walk_forward_span", "max_drawdown_pct", "{:.2f}"),
         wf_row("whole walk-forward span: daily Sharpe x sqrt(365)", "whole_walk_forward_span", "daily_sharpe_x_sqrt365", "{:+.2f}")]
head = ["Measure", "Kraken baseline", "Kraken Natenberg vol", "Alpaca baseline", "Alpaca Natenberg vol"]
txt = ["# Bot with Natenberg's volatility replacing Snippet 3.1 / the EWM sigma, against the baseline bot", "",
       "Only change: the volatility estimate (CUSUM threshold, barrier unit, daily target, log_trgt feature) is Natenberg Appendix B (sample standard deviation of log returns, mean 0) instead of the exponentially weighted std. Same data, seed, costs, forest, folds. Numbers are read from the saved meta.json of each run.", "",
       "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
txt += ["| " + " | ".join(r) + " |" for r in rows]
txt += ["", "Runs: " + "; ".join(f"{v} baseline {BASE[v].name}, Natenberg {NAT[v].name}" for v in ("kraken", "alpaca"))]
out = ROOT / "runtime/book_bot_nat/COMPARISON.md"
out.write_text("\n".join(txt) + "\n")
print("\n".join(txt))
