"""Independent re-derivation of book_full.py (the book's algorithms as one strategy: side-free Ch.3 labels, Ch.10 side x size from the saved predictions, Ch.13 OTR exits, Ch.16 HRP allocation, one-year account)
from the raw files (events.csv, bar csv files, the saved predictions).  Shares NO code with book_full.py, book_otr_exit.py, bet_averaging.py, book_port_d.py or the bar loader: its own bet-size function (math.erf),
its own exits (per event, numpy slice, events located on the bars by TIMESTAMP), its own per-bar position (slice adds), its own day booking (bincount), its own HRP (linkage on the book's square distance matrix,
own quasi-diagonalisation, own recursive bisection), its own quarterly schedule, its own compounding loop in plain Python.
Re-derived: kept events, walk-forward and CPCV AUC / side accuracy, EVERY HRP weight at EVERY rebalance in hrp_weights.csv, and EVERY row of summary.csv (gain, drawdown, Sharpe).
Usage: python verify_book_full.py <book_full out folder> [--plant bet_size|cost|hrp|exit]   (plant: perturbs the checker's own calculation; the checker must then report problems)
Exit code 1 when any number differs by more than 1e-9 relative (1e-9 absolute near zero).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.stats import rankdata

DAY = 86400


def epoch(s: pd.Series) -> np.ndarray:
    return np.array([int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()) for x in s], dtype=np.int64)


def load_bars(bar_dir: Path, sym: str, break_s: int):
    b = pd.read_csv(bar_dir / "bars" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "start", "close"])
    b = b.dropna(subset=["close"])
    b = b[b["close"] > 0]
    t, st = epoch(b["date"]), epoch(b["start"])
    c = b["close"].to_numpy(float)
    o = np.argsort(t, kind="stable")
    t, st, c = t[o], st[o], c[o]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, st, c = t[keep], st[keep], c[keep]
    seg = np.concatenate([[0], np.cumsum((st[1:] - t[:-1]) > break_s)]).astype(np.int64)
    return t, c, seg


def bet_size(p: float, step: float, plant: bool) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    if p <= 0.5:
        return 0.0
    z = (p - 0.5) / math.sqrt(p * (1 - p))
    s = 2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))) - 1.0
    s = round(s / step) * step
    s = min(max(s, 0.0), 1.0)
    return s * 1.01 if plant else s


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------- own HRP ----------------
def my_hrp(R: pd.DataFrame, equal: bool, plant: bool) -> dict:
    cols = list(R.columns)
    n = len(cols)
    if equal or n < 3:
        return {c: 1.0 / n for c in cols}
    cov = R.cov().to_numpy()
    corr = R.corr().to_numpy()
    dist = np.sqrt((1.0 - corr) / 2.0)
    Z = sch.linkage(dist, "single")                      # the book's Snippet 16.4 passes this square matrix as is
    Z = Z.astype(int)
    # quasi-diagonalisation: expand the root's two children left to right until every entry is an original item
    order = [int(Z[-1, 0]), int(Z[-1, 1])]
    while max(order) >= n:
        nxt = []
        for v in order:
            if v >= n:
                nxt += [int(Z[v - n, 0]), int(Z[v - n, 1])]
            else:
                nxt.append(v)
        order = nxt

    def cvar(items):
        sub = cov[np.ix_(items, items)]
        iv = 1.0 / np.diag(sub)
        iv = iv / iv.sum()
        return float(iv @ sub @ iv)

    w = np.ones(n)
    clusters = [order]
    while clusters:
        new = []
        for c in clusters:
            if len(c) > 1:
                h = len(c) // 2
                new += [c[:h], c[h:]]
        clusters = new
        for i in range(0, len(clusters), 2):
            a, b = clusters[i], clusters[i + 1]
            va, vb = cvar(a), cvar(b)
            alpha = 1.0 - va / (va + vb)
            for x in a:
                w[x] *= alpha
            for x in b:
                w[x] *= 1.0 - alpha
    out = {cols[i]: float(w[i]) for i in range(n)}
    if plant:
        k = cols[0]
        out[k] *= 1.05
    return out


def main(out: Path, plant: str | None) -> int:
    meta = json.loads((out / "meta.json").read_text())
    src, bar_dir, bf = Path(meta["source_run"]), Path(meta["bar_dir"]), Path(meta["bookfeat_run"])
    venue = meta["venue"]
    step, seed, cost_frac, cutoff = meta["step"], meta["seed"], meta["cost_frac_round_trip"], meta["cutoff"]
    pt, sl, K = float(meta["otr"]["pt"]), float(meta["otr"]["sl"]), int(meta["otr"]["K"])
    max_gap = 1 if venue == "kraken" else 5
    problems = []
    checked = 0

    def cmp(name, mine, theirs, tol=1e-9):
        nonlocal checked
        checked += 1
        if np.isnan(mine) and np.isnan(theirs):
            return
        if abs(mine - theirs) > tol * max(1.0, abs(mine)):
            problems.append(f"{name}: independent {mine!r} vs backtest {theirs!r}")

    ev0 = pd.read_csv(src / "events.csv", usecols=["symbol", "t0", "t1", "sigma"])
    idx_file = pd.read_csv(bf / "events_free_index.csv")
    break_s = int(json.loads((bar_dir / "manifest.json").read_text())["break_hours"]) * 3600
    syms_all = sorted(ev0["symbol"].unique())
    bars = {s: load_bars(bar_dir, s, break_s) for s in syms_all}
    # events located by timestamp; kept = long price return not exactly 0
    i0 = np.zeros(len(ev0), dtype=np.int64)
    keep = np.zeros(len(ev0), dtype=bool)
    ret_long = np.zeros(len(ev0))
    for s, g in ev0.groupby("symbol"):
        t, c, _ = bars[s]
        a = np.searchsorted(t, g["t0"].to_numpy())
        b = np.searchsorted(t, g["t1"].to_numpy())
        okm = (t[a] == g["t0"].to_numpy()) & (t[b] == g["t1"].to_numpy()) & (b > a)
        if not okm.all():
            problems.append(f"{s}: {int((~okm).sum())} event timestamps do not sit on bars")
        r = c[b] / c[a] - 1.0
        ii = g.index.to_numpy()
        i0[ii], ret_long[ii], keep[ii] = a, r, okm & (r != 0.0)
    ev = ev0[keep].reset_index(drop=True)
    i0, ret_long = i0[keep], ret_long[keep]
    if len(ev) != len(idx_file) or not (idx_file["src_row"].to_numpy() == np.flatnonzero(keep)).all():
        problems.append("the kept-event set differs from events_free_index.csv")
    cmp("events", len(ev), meta["events"])
    y = (ret_long > 0).astype(int)
    # ---------- model quality ----------
    wf = pd.read_csv(bf / "walk_forward_predictions.csv")
    cp = pd.read_csv(out / "cpcv_predictions.csv")
    w_idx, w_p = wf["event"].to_numpy(np.int64), wf["p"].to_numpy(float)
    cmp("wf auc", auc(y[w_idx], w_p), meta["walk_forward"]["auc"])
    cmp("wf side accuracy", float((((w_p > .5).astype(int)) == y[w_idx]).mean()), meta["walk_forward"]["side_accuracy"])
    cut = int(pd.Timestamp(cutoff).timestamp())
    if not (ev["t0"].to_numpy()[w_idx] >= cut).all():
        problems.append("a walk-forward prediction is for an event that starts before the cutoff")
    paths = {int(k): g for k, g in cp.groupby("path")}
    for k, g in sorted(paths.items()):
        e_, p_ = g["event"].to_numpy(np.int64), g["p"].to_numpy(float)
        if len(np.unique(e_)) != len(e_):
            problems.append(f"cpcv path {k}: an event appears twice")
        cmp(f"cpcv path {k} auc", auc(y[e_], p_), meta["cpcv"]["auc_by_path"][k])
        cmp(f"cpcv path {k} side accuracy", float((((p_ > .5).astype(int)) == y[e_]).mean()), meta["cpcv"]["side_accuracy_by_path"][k])
    # ---------- exits: for every event, the exit bar if the side were +1 and if it were -1 ----------
    sig_ev = ev["sigma"].to_numpy(float)
    sym_ev = ev["symbol"].to_numpy()
    exit_bar = {1.0: np.zeros(len(ev), dtype=np.int64), -1.0: np.zeros(len(ev), dtype=np.int64)}
    for sd in (1.0, -1.0):
        ptc, slc = (pt, sl) if sd == 1.0 else (pt, sl)
        for s in syms_all:
            ii = np.flatnonzero(sym_ev == s)
            if not len(ii):
                continue
            t, c, sg = bars[s]
            lc = np.log(c)
            for e in ii:
                a = int(i0[e])
                hi = min(a + K, len(t) - 1)
                seg_ok = sg[a + 1:hi + 1] == sg[a]
                m = int(seg_ok.argmin()) if not seg_ok.all() else len(seg_ok)       # number of consecutive in-segment bars after a
                if m == 0:
                    exit_bar[sd][e] = a
                    continue
                q = sd * (lc[a + 1:a + 1 + m] - lc[a]) / sig_ev[e]
                if plant == "exit":
                    q = q * 0.97
                hit = np.flatnonzero((q >= ptc) | (q <= -slc))
                exit_bar[sd][e] = a + 1 + int(hit[0]) if len(hit) else a + m
    # ---------- per-instrument daily nets for a set of events with sides and sizes ----------
    first_bar_day = min(int(bars[s][0][0] // DAY) for s in syms_all)
    last_bar_day = max(int(bars[s][0][-1] // DAY) for s in syms_all)
    D0 = first_bar_day
    ND = last_bar_day - D0 + 1
    cost_side = cost_frac / 2.0

    def nets(idx, sides, sizes, with_cost):
        """matrix [symbol index, day] of the instrument's daily result per unit of capital in it"""
        M = np.zeros((len(syms_all), ND))
        cs = cost_side if with_cost else 0.0
        if plant == "cost" and with_cost:
            cs = cs * 1.1
        for si, s in enumerate(syms_all):
            sel = np.flatnonzero(sym_ev[idx] == s)
            if not len(sel):
                continue
            t, c, _ = bars[s]
            n = len(t)
            tot, cnt = np.zeros(n), np.zeros(n)
            for j in sel:
                e = int(idx[j])
                a, b = int(i0[e]), int(exit_bar[float(sides[j])][e])
                tot[a:b] += sides[j] * sizes[j]
                cnt[a:b] += 1
            avg = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
            day = (t // DAY) - D0
            gross = avg[:-1] * (c[1:] / c[:-1] - 1.0)
            prev = np.concatenate([[0.0], avg[:-1]])
            cost = cs * np.abs(avg - prev)
            M[si] += np.bincount(day[1:], weights=gross, minlength=ND)
            M[si] -= np.bincount(day, weights=cost, minlength=ND)
        return M

    # ---------- daily returns for HRP: last close of each UTC day, days with >=80% of symbols, return only when <= max_gap days apart ----------
    lastc = {}
    for s in syms_all:
        t, c, _ = bars[s]
        d = t // DAY
        lastc[s] = pd.Series(c, index=d).groupby(level=0).last()
    DC = pd.DataFrame(lastc).sort_index()
    DC = DC[DC.notna().mean(axis=1) >= .8]
    dd = DC.index.to_numpy()
    R = DC.pct_change(fill_method=None)
    okrow = np.zeros(len(DC), dtype=bool)
    okrow[1:] = np.diff(dd) <= max_gap
    R = R[okrow]
    R_days = R.index.to_numpy()

    last_day = last_bar_day
    first_day_all = first_bar_day
    first_event_day = int(ev["t0"].min() // DAY)
    cmp("last_day", last_day, int(pd.Timestamp(meta["last_day"]).timestamp() // DAY))
    cmp("first_day", first_day_all, int(pd.Timestamp(meta["first_day"]).timestamp() // DAY))
    reb_file = pd.read_csv(out / "hrp_rebalances.csv")
    start0 = min(first_day_all, first_event_day) + int(meta["reb_window"])

    def quarter(d):
        dt = datetime.fromtimestamp(d * DAY, timezone.utc)
        return dt.year * 4 + (dt.month - 1) // 3

    keys = [start0] + [d for d in range(start0 + 1, last_day + 1) if quarter(d) != quarter(d - 1)]
    if keys != reb_file["rebalance_day"].tolist():
        problems.append("rebalance days differ from hrp_rebalances.csv")
    W = int(meta["reb_window"])
    sched = {"hrp": {}, "equal": {}}
    saved_w = pd.read_csv(out / "hrp_weights.csv", index_col=0)
    for r_ in keys:
        win = R[R_days < r_].tail(W)
        for scheme in ("hrp", "equal"):
            if len(win) < W * .8:
                sched[scheme][r_] = {}
                continue
            ok = [c for c in win.columns if win[c].notna().sum() >= .8 * len(win) and win[c].fillna(0.0).std() > 0]
            w = my_hrp(win[ok].fillna(0.0), scheme == "equal", plant == "hrp" and scheme == "hrp")
            tot = sum(w.values())
            sched[scheme][r_] = {k: v / tot for k, v in w.items()}
    # compare every HRP weight to the saved table
    lab = {pd.Timestamp(k * DAY, unit="s").strftime("%Y-%m-%d"): k for k in keys}
    nchk = 0
    for dstr, row in saved_w.iterrows():
        mine = sched["hrp"][lab[dstr]]
        for s_ in saved_w.columns:
            v = row[s_]
            mv = mine.get(s_, float("nan"))
            if np.isnan(v):
                if s_ in mine:
                    problems.append(f"hrp weight {dstr} {s_}: saved table has no weight, checker has one")
                continue
            cmp(f"hrp weight {dstr} {s_}", mv, float(v))
            nchk += 1
    # weight matrices [symbol, day] per scheme
    Wm = {}
    for scheme in ("hrp", "equal"):
        A = np.zeros((len(syms_all), ND))
        ks = sorted(sched[scheme])
        for q, st in enumerate(ks):
            en = ks[q + 1] if q + 1 < len(ks) else last_day + 1
            for si, s in enumerate(syms_all):
                A[si, st - D0:en - D0] = sched[scheme][st].get(s, 0.0)
        Wm[scheme] = A

    def stats(r):
        eq, peak, mdd = 1.0, 1.0, 0.0
        for x in r:
            eq *= 1.0 + x
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1.0)
        m = sum(r) / len(r)
        sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
        return 100.0 * (eq - 1.0), 100.0 * mdd, (m / sd * math.sqrt(365)) if sd > 0 else float("nan")

    S = pd.read_csv(out / "summary.csv")
    cutoff_day = cut // DAY
    # variants per scope
    ones_w = np.ones(len(w_idx))
    side_w = np.where(w_p > .5, 1.0, -1.0)
    size_w = np.array([bet_size(float(max(q, 1 - q)), step, plant == "bet_size") for q in w_p])
    plc_w = np.random.default_rng(seed + 7 + 99).permutation(size_w)
    rs_w = np.random.default_rng(seed + 11).choice([-1.0, 1.0], size=len(w_idx))
    scopes = {"wf": (w_idx, {"book_bet": (side_w, size_w), "pred_side_full_size": (side_w, ones_w), "placebo_sizes": (side_w, plc_w), "always_long": (ones_w, ones_w), "random_side": (rs_w, ones_w)})}
    for k, g in sorted(paths.items()):
        e_, p_ = g["event"].to_numpy(np.int64), g["p"].to_numpy(float)
        sd_ = np.where(p_ > .5, 1.0, -1.0)
        sz_ = np.array([bet_size(float(max(q, 1 - q)), step, plant == "bet_size") for q in p_])
        o_ = np.ones(len(e_))
        scopes[f"cpcv{k}"] = (e_, {"book_bet": (sd_, sz_), "pred_side_full_size": (sd_, o_), "placebo_sizes": (sd_, np.random.default_rng(seed + 7 + 99).permutation(sz_)), "always_long": (o_, o_)})
    for scope, (idx, variants) in scopes.items():
        for v, (sd, sz) in variants.items():
            for cl in ("net", "before_costs"):
                M = nets(idx, sd, sz, cl == "net")
                for scheme in ("hrp", "equal"):
                    rows = S[(S["scope"] == scope) & (S["variant"] == v) & (S["costs"] == cl) & (S["scheme"] == scheme)]
                    if scope != "wf" and scheme == "equal":
                        if len(rows):
                            problems.append(f"unexpected equal-weight rows for {scope}")
                        continue
                    if len(rows) != 2:
                        problems.append(f"{scope}/{v}/{cl}/{scheme}: {len(rows)} summary rows, expected 2")
                        continue
                    daily = (Wm[scheme] * M).sum(axis=0)
                    for _, row in rows.iterrows():
                        lo = pd.Timestamp(row["start_day"]).timestamp() // DAY
                        lo = int(lo)
                        exp_lo = last_day - 364 if row["span"] == "last_365_days" else (cutoff_day if row["span"] == "whole_walk_forward_span" else first_day_all + W + 5)
                        if lo != exp_lo:
                            problems.append(f"{scope}/{v}/{cl}/{scheme}/{row['span']}: start day {lo} vs expected {exp_lo}")
                        g_, d_, sh_ = stats(daily[lo - D0:last_day - D0 + 1].tolist())
                        nm = f"{scope}/{v}/{cl}/{scheme}/{row['span']}"
                        cmp(nm + "/gain_pct", g_, float(row["gain_pct"]))
                        cmp(nm + "/max_drawdown_pct", d_, float(row["max_drawdown_pct"]))
                        cmp(nm + "/sharpe", sh_, float(row["daily_sharpe_x_sqrt365"]))
    msg = f"{venue}: {checked} numbers re-derived independently ({nchk} HRP weights among them); {len(problems)} problems" + (f"  [planted: {plant}]" if plant else "")
    print(msg)
    for p in problems[:30]:
        print("  PROBLEM:", p)
    if not plant:
        (out / "checker_output.txt").write_text(msg + "\n" + "\n".join(problems) + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    pl = sys.argv[sys.argv.index("--plant") + 1] if "--plant" in sys.argv else None
    raise SystemExit(main(Path(sys.argv[1]), pl))
