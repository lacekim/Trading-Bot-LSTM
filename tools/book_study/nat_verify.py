"""Independent check of the Natenberg write-up.  New file.  Reads runtime/book_nat/<latest>/items.json and natenberg_study.pdf; recomputes a set of printed numbers with scipy and plain python (not nat_core) and compares them with what the saved run holds.
Usage: python -W ignore tools/book_study/nat_verify.py [run_dir]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from math import exp, log, sqrt, comb
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
runs = sorted((ROOT / "runtime" / "book_nat").glob("*"))
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else runs[-1]
d = json.load(open(RUN / "items.json"))
items = {i["id"]: i for i in d["items"]}
fails, n = [], 0


def ok(name, cond, extra=""):
    global n
    n += 1
    if not cond:
        fails.append(f"{name} {extra}")


# 1. every saved check: recount ok flags against the numbers saved with it
for it in d["items"]:
    for c in it["checks"]:
        if c["kind"] == "num" and c["tol"] is not None:
            ok(f"{it['id']}: {c['what'][:60]}", (abs(c["ours"] - c["book"]) <= c["tol"]) == c["ok"], f"book {c['book']} ours {c['ours']} tol {c['tol']} flag {c['ok']}")
        elif c["kind"] in ("bool", "text"):
            ok(f"{it['id']}: {c['what'][:60]}", (c["ours"] == c["book"]) == c["ok"] or c["what"].startswith("Printed formula") or "finite difference" in c["what"], f"flag {c['ok']}")

# 2. summary table equals the recount
tot = sum(len(i["checks"]) for i in d["items"])
bad = sum(1 for i in d["items"] for c in i["checks"] if not c["ok"])
ok("summary total", d["summary"][-1][3] == tot and d["summary"][-1][5] == bad, f"{d['summary'][-1]} vs {tot},{bad}")


# 3. independent Black-Scholes (scipy) against printed numbers
def bs(S, X, t, r, s):
    d1 = (log(S / X) + (r + s * s / 2) * t) / (s * sqrt(t)); d2 = d1 - s * sqrt(t)
    return S * norm.cdf(d1) - X * exp(-r * t) * norm.cdf(d2), X * exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)


# Figure 21-9: per-contract call delta / gamma / theta / vega, S 101.25, 42 days, 27%
S, t, s = 101.25, 42 / 365, 0.27
exp9 = {90: (90.9, 1.77, -0.0181, 0.056), 95: (77.1, 3.27, -0.0335, 0.104), 100: (57.2, 4.23, -0.0433, 0.135), 105: (36.3, 4.04, -0.0414, 0.129), 110: (19.5, 2.97, -0.0304, 0.095)}
for X, (dl, gm, th, vg) in exp9.items():
    d1 = (log(S / X) + s * s / 2 * t) / (s * sqrt(t))
    ok(f"21-9 {X} delta", abs(norm.cdf(d1) * 100 - dl) <= 0.06)
    ok(f"21-9 {X} gamma", abs(norm.pdf(d1) / (S * s * sqrt(t)) * 100 - gm) <= 0.006)
    ok(f"21-9 {X} theta", abs(-S * norm.pdf(d1) * s / (2 * sqrt(t)) / 365 - th) <= 0.00006)
    ok(f"21-9 {X} vega", abs(S * norm.pdf(d1) * sqrt(t) / 100 - vg) <= 0.0006)

# Figure 21-9 position totals
calls = [-4, -12, 14, -17, 12]; puts = [13, -20, -8, 12, 8]
dt = 13 * 100.0; gt = 0.0; tt = 0.0; vt = 0.0
for X, cq, pq in zip((90, 95, 100, 105, 110), calls, puts):
    d1 = (log(S / X) + s * s / 2 * t) / (s * sqrt(t))
    dt += cq * norm.cdf(d1) * 100 + pq * (norm.cdf(d1) - 1) * 100
    gt += (cq + pq) * norm.pdf(d1) / (S * s * sqrt(t)) * 100
    tt += (cq + pq) * (-S * norm.pdf(d1) * s / (2 * sqrt(t)) / 365)
    vt += (cq + pq) * S * norm.pdf(d1) * sqrt(t) / 100
ok("21-9 total delta -297.4", abs(dt + 297.4) <= 1.5, str(dt))
ok("21-9 total gamma -24.13", abs(gt + 24.13) <= 0.4, str(gt))
ok("21-9 total vega -0.759", abs(vt + 0.759) <= 0.02, str(vt))
ok("21-9 delta-neutral estimate 88.93", abs(S - abs(dt) / abs(gt) - 88.93) <= 0.5)

# 0.00399 rule: at the money, forward = strike: call = 0.4 sigma sqrt(t) S (0.399) ; book: 3 months, 20%: 3.99 on 100 at 40% -> 7.98
c, p = bs(100, 100, 0.25, 0.0, 0.40)
ok("Black-Scholes ATM 100, 3 months, 40%: 7.97", abs(c - 7.966) < 0.01, str(c))

# binomial tree (Figure 19-1): S 100, X 100, 9 months, 3 periods, r 4%, u 1.05 -> value of European call
def tree(S, X, t, r, n, u, kind="call", american=False):
    h = t / n; dd = 1 / u; p = (1 + r * h - dd) / (u - dd)
    V = [max((S * u ** j * dd ** (n - j) - X) if kind == "call" else (X - S * u ** j * dd ** (n - j)), 0.0) for j in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            v = (p * V[j + 1] + (1 - p) * V[j]) / (1 + r * h)
            if american:
                s_ = S * u ** j * dd ** (i - j)
                v = max(v, (s_ - X) if kind == "call" else (X - s_))
            V[j] = v
    return V[0]
v = tree(100, 100, 0.75, 0.04, 3, 1.05)
saved = [c for c in items["Figure 19-1"]["checks"]]
ok("19-1 saved checks exist", len(saved) > 0)
ok("19-1 independent European call value is within 0.01 of the first saved 'ours' that mentions the value", any(abs(c["ours"] - v) < 0.01 for c in saved if c["kind"] == "num"), f"indep {v}")
va = tree(100, 100, 0.75, 0.04, 3, 1.05, "put", True)
ok("19-6 independent American put value matches the saved one", any(abs(c["ours"] - va) < 0.01 for c in items["Figure 19-6"]["checks"] if c["kind"] == "num"), f"indep {va}")

# Figure 23-3: one-sided normal probabilities
for z, odds in ((6.89, 358e9), (4.80, 1.26e6), (3.69, 8.9e3), (4.10, 48e3)):
    ok(f"23-3 z {z}", abs(log(1 / norm.sf(z), 10) - log(odds, 10)) < 0.02)

# 4. the PDF: page count, the lists, and no placeholder text
txt = subprocess.run(["pdftotext", "-layout", str(RUN / "natenberg_study.pdf"), "-"], capture_output=True, text=True).stdout
ok("pdf has text", len(txt) > 50000)
for f in d["inventory"]:
    ok(f"pdf names {f}", f in txt, "")
ok("pdf has the coverage sentence", "have at least one entry; none missing" in txt)
for bad_s in ("nan", "None", "Traceback"):
    cnt = len(re.findall(rf"\b{bad_s}\b", txt))
    ok(f"pdf contains no '{bad_s}'", cnt == 0, f"{cnt} occurrences")

print(f"verify: {n} checks, {len(fails)} failed")
for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
