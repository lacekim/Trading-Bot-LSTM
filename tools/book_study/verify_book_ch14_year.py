"""Independent re-derivation of book_ch14_year.py's statistics from the two saved series (ch14_target_position.csv, ch14_daily_return.csv) with its own code (loops, no pandas groupby): bet timing, weighted-entry holding period,
HHI of positive / negative bet returns and of bets per month, drawdown and time-under-water sequences.  It does NOT re-derive the target-position series itself (that is the account's bookkeeping, already checked number by number against the
returns in verify_book_final.py; book_ch14_year.py also asserts that its recomputed daily returns equal the saved equity curve).  Usage: python verify_book_ch14_year.py <book_final run folder>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def hhi(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 2:
        return float("nan")
    w = x / x.sum()
    return (float((w ** 2).sum()) - 1.0 / n) / (1.0 - 1.0 / n)


def main(run: Path) -> int:
    res = json.loads((run / "ch14_year.json").read_text())
    tp = pd.read_csv(run / "ch14_target_position.csv", index_col=0, parse_dates=True).iloc[:, 0]
    rt = pd.read_csv(run / "ch14_daily_return.csv", index_col=0, parse_dates=True).iloc[:, 0]
    x, d, n = tp.to_numpy(), tp.index, len(tp)
    problems = []

    def chk(name, mine, theirs, tol=1e-9):
        if isinstance(mine, float) and np.isnan(mine):
            mine = None
        if mine is None or theirs is None:
            ok = mine is None and theirs is None
        elif isinstance(mine, float) and np.isnan(mine):
            ok = theirs is None
        else:
            ok = abs(mine - theirs) <= tol * max(1.0, abs(mine))
        if not ok:
            problems.append(f"{name}: independent {mine!r} vs {theirs!r}")
    # bets: flattening (flat today, not flat yesterday) or flip (sign change), plus the last day
    bet_i = [i for i in range(1, n) if (x[i] == 0 and x[i - 1] != 0) or (x[i] * x[i - 1] < 0)]
    if n - 1 not in bet_i:
        bet_i.append(n - 1)
    chk("bets", len(bet_i), res["bets_14_1"])
    if [str(d[i].date()) for i in bet_i][:50] != res["bet_end_dates"]:
        problems.append("bet end dates differ")
    # holding period: average entry time, weight = size of the reduction
    t = np.array([(d[i] - d[0]).total_seconds() / 86400.0 for i in range(n)])
    entry, dts, ws = 0.0, [], []
    for i in range(1, n):
        diff = x[i] - x[i - 1]
        if diff * x[i - 1] >= 0:
            if x[i] != 0:
                entry = (entry * x[i - 1] + t[i] * diff) / x[i]
        elif x[i] * x[i - 1] < 0:
            dts.append(t[i] - entry)
            ws.append(abs(x[i - 1]))
            entry = t[i]
        else:
            dts.append(t[i] - entry)
            ws.append(abs(diff))
    hp = (sum(a * b for a, b in zip(dts, ws)) / sum(ws)) if sum(ws) > 0 else None
    chk("holding period", hp, res["avg_holding_period_days_14_2"])
    # bet returns: compounded daily returns after the previous bet end up to this bet end (the first bet from the start)
    rr = rt.to_numpy()
    out, start = [], 0
    for k, i in enumerate(bet_i):
        lo = 0 if k == 0 else bet_i[k - 1] + 1
        out.append(float(np.prod(1.0 + rr[lo:i + 1]) - 1.0))
    if [round(100 * v, 6) for v in out][:50] != res["bet_returns_pct"]:
        problems.append("bet returns differ")
    pos_r, neg_r = [v for v in out if v >= 0], [v for v in out if v < 0]
    chk("hhi pos", hhi(pos_r), res["hhi_positive_returns_14_3"])
    chk("hhi neg", hhi(neg_r), res["hhi_negative_returns_14_3"])
    months = {}
    for i in bet_i:
        months[(d[i].year, d[i].month)] = months.get((d[i].year, d[i].month), 0) + 1
    # calendar months with no bet inside the span count as zero, as a monthly grouper does
    y, m = d[bet_i[0]].year, d[bet_i[0]].month
    cnt = []
    while (y, m) <= (d[bet_i[-1]].year, d[bet_i[-1]].month):
        cnt.append(months.get((y, m), 0))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    chk("hhi bets per month", hhi(cnt), res["hhi_bets_per_month_14_3"])
    # drawdowns and time under water from the equity curve
    eq = np.cumprod(1.0 + rr)
    hwm_t, hwm_v, mins = [], [], []
    peak = -np.inf
    for i, v in enumerate(eq):
        if v > peak:
            peak = v
            hwm_t.append(i)
            hwm_v.append(v)
            mins.append(v)
        else:
            mins[-1] = min(mins[-1], v)
    dd = [1 - m_ / h for h, m_ in zip(hwm_v, mins) if h > m_]
    idx = [t_ for t_, h, m_ in zip(hwm_t, hwm_v, mins) if h > m_]
    tuw = [(d[idx[k + 1]] - d[idx[k]]).total_seconds() / 86400.0 / 365.25 for k in range(len(idx) - 1)]
    chk("dd count", len(dd), res["drawdowns_14_4"]["count"])
    chk("dd max", 100 * max(dd) if dd else 0.0, res["drawdowns_14_4"]["max_pct"])
    chk("dd p95", 100 * float(np.percentile(dd, 95)) if dd else 0.0, res["drawdowns_14_4"]["p95_pct"])
    chk("tuw max", max(tuw) if tuw else None, res["time_under_water_years_14_4"]["max"])
    chk("tuw p95", float(np.percentile(tuw, 95)) if tuw else None, res["time_under_water_years_14_4"]["p95"])
    chk("gain", 100 * (eq[-1] - 1.0), res["gain_pct"])
    msg = f"{res['venue']}: Chapter 14 statistics re-derived from the saved series; {len(problems)} problems"
    print(msg)
    for p_ in problems:
        print("  PROBLEM:", p_)
    (run / "ch14_checker_output.txt").write_text(msg + "\n" + "\n".join(problems) + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
