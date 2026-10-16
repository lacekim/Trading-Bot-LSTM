#!/usr/bin/env python
"""Independent checker for a run_book_ports_bars.py output folder (plain Python loops; numpy only for the random-number stream and one 3x3 solve; imports NOTHING from book_port_*.py,
ports_*.py or run_book_ports_bars.py -- only verify_book_study.py's time-stamp parser).

What it re-derives from the raw files (the source run's events.csv, symbols.csv, cpcv_predictions.csv, summary_numbers.json, and the dollar-bar files), with its own code:
  run integrity   no section failed, not a --quick run, source run is a dollar-bar run of the same venue, bar-manifest md5, the md5 of every script used still equals the one recorded
  Ch.3            the class shares of the meta-label `bin` and of the barrier outcome `kind`
  Ch.4            average uniqueness of the first events of each studied symbol (own difference-array count), and the Monte-Carlo calibration (events per bar, median holding, numObs, maxH)
  Ch.7            every purged-k-fold fold: test block, left/right training sets, embargo, brute-force count of training labels that overlap the test span
  Ch.8            the sample (systematic every k-th event), its label share and the majority-class share
  Ch.2 / 16       the daily-return panel (data-hole days, gaps, symbol coverage) rebuilt from the bar files: symbols, days, window, eigenvalue shares (2), inverse-variance weights (16)
  Ch.15           Snippet 15.1's exact value, mean/std/ratio consistency, the bet statistics (bets, years, precision, avg win, avg loss), Equation 56 / the required-precision root /
                  the required frequency, the window rows (bets, precision, mean net return) -- from the events and the CPCV path-0 predictions
  Ch.17-19        the recorded differences to the stored features / project code are tiny (thresholds), the compared counts are plausible
  Ch.3 / 4 / 8 / 13 / 16  Snippet 2.4 on real bars (identical events), Snippet 4.6 against the EXACT expected uniqueness by enumeration, 13.1's grid and mesh, 16.4 against Table 16.1's tolerance
  Ch.20           the recorded equalities hold; linParts / nestedParts closed forms
  Ch.21           the whole exhaustive search re-done from the raw bars (daily closes, blocks, covariances, the book's cost draw, all distinct weight vectors, every trajectory)
  SUMMARY.md      the formatted numbers of the sections above appear in the text
What it does NOT re-derive (stated, not hidden): Ch.2 (PCA weights), Ch.5 minimum-d tables, Ch.6/8/9 forests and grids (random-forest accuracies), Ch.10 averaged evaluation,
Ch.13 synthetic mesh, Ch.14 HHI / drawdown / holding period, Ch.16 allocations, Ch.18 Lempel-Ziv statistics.  The book ports behind those are covered by test_book_port_*.py and by the earlier cross-checks against the project's own code.
Exit code 0 = all checks pass.

    python tools/book_study/verify_book_ports_bars.py runtime/book_ports_<venue>_bars/<timestamp>
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import verify_book_study as V  # noqa: E402  (only V.ts_key)

SCRIPTS = ("run_book_ports_bars.py", "ports_common.py", "ports_sections_a.py", "ports_sections_b.py", "ports_sections_c.py", "book_mp.py", "book_port_a.py", "book_port_b.py",
           "book_port_c.py", "book_port_d.py", "book_port_e.py")


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((bool(ok), name, detail))

    def close(self, name, got, want, rel=1e-9, ab=1e-12):
        ok = got is not None and want is not None and math.isfinite(got) and math.isfinite(want) and abs(got - want) <= ab + rel * abs(want)
        self.check(name, ok, f"recorded {got!r} vs recomputed {want!r}")

    def done(self):
        bad = [r for r in self.rows if not r[0]]
        for ok, name, detail in self.rows:
            print(("PASS " if ok else "FAIL ") + name + (f"   [{detail}]" if (not ok and detail) else ""))
        print(f"{len(self.rows) - len(bad)} of {len(self.rows)} checks pass")
        return 0 if not bad else 1


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load(folder):
    folder = Path(folder).resolve()
    j = json.loads((folder / "summary_numbers.json").read_text(encoding="utf-8"))
    src = Path(j["meta"]["source_run"])
    ev = read_csv(src / "events.csv")
    for r in ev:
        for k in ("pos", "t0", "t1", "t1pos", "bin", "kind", "side"):
            r[k] = int(float(r[k]))
        for k in ("net_ret", "u"):
            r[k] = float(r[k])
    return folder, j, src, ev


def sorted_event_order(ev):
    """Positions of the events sorted by (t0, symbol), stable -- the pooled time order; plus the unique ns index (t0 seconds * 1e9 + rank within equal t0)."""
    order = sorted(range(len(ev)), key=lambda i: (ev[i]["t0"], ev[i]["symbol"]))
    t0ns, prev, k = [], None, 0
    for i in order:
        t = ev[i]["t0"]
        k = k + 1 if t == prev else 0
        prev = t
        t0ns.append(t * 10**9 + k)
    return order, t0ns


def check_integrity(R, folder, j, src, ev):
    m = j["meta"]
    R.check("no section failed", not m["errors"], f"errors: {m['errors']}")
    R.check("not a --quick run (full sizes)", not m["quick"], "quick=True")
    ssum = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    R.check("source run is a dollar-bar run of the same venue", ssum["meta"].get("timeframe") == "dollar" and ssum["meta"]["venue"] == m["venue"], f"{ssum['meta'].get('timeframe')}, {ssum['meta']['venue']} vs {m['venue']}")
    bar_dir = Path(m["bar_dir"])
    R.check("bar folder is the one the source run recorded", Path(ssum["meta"]["candle_dir"]).resolve() == bar_dir.resolve() or str(ssum["meta"]["candle_dir"]).endswith(bar_dir.name),
            f"{ssum['meta']['candle_dir']} vs {bar_dir}")
    R.check("bar manifest md5 matches", md5_file(bar_dir / "manifest.json") == m["bars_manifest_md5"], "manifest changed since the run")
    R.check("event count equals events.csv", int(m["events"]) == len(ev), f"{m['events']} vs {len(ev)}")
    for n in SCRIPTS:
        R.check(f"script md5 {n} unchanged since the run", md5_file(HERE / n) == m["script_md5"].get(n), "file differs from the recorded md5")
    R.check("every section present in the results", all(k in j["results"] for k in ("ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "ch9", "ch10", "ch13", "ch14", "ch15", "ch16", "ch17", "ch18", "ch19", "ch20", "ch21")),
            f"present: {sorted(j['results'])}")


def check_ch3(R, j, ev):
    r = j["results"]["ch3"]
    n = len(ev)
    for key, col in (("meta_label_bin", "bin"), ("barrier_outcome_kind", "kind")):
        cnt = {}
        for e in ev:
            cnt[e[col]] = cnt.get(e[col], 0) + 1
        got = r[key]["classes"]
        R.check(f"ch3 {key}: rows", int(r[key]["rows_before"]) == n, f"{r[key]['rows_before']} vs {n}")
        R.check(f"ch3 {key}: class set", sorted(int(k) for k in got) == sorted(cnt), f"{sorted(got)} vs {sorted(cnt)}")
        for k, v in cnt.items():
            R.close(f"ch3 {key} share of class {k}", got[str(k)], v / n)


def check_ch4(R, j, ev, src, cfg):
    r = j["results"]["ch4"]
    syms_df = read_csv(src / "symbols.csv")
    counts = {}
    for e in ev:
        counts[e["symbol"]] = counts.get(e["symbol"], 0) + 1
    order = sorted(counts, key=lambda s: (-counts[s], s))
    for row in r["uniqueness_and_bootstrap"]:
        s = row["symbol"]
        R.check(f"ch4 symbol {s} is among the most active in the order of the venue", s == order[r["uniqueness_and_bootstrap"].index(row)], f"expected {order[r['uniqueness_and_bootstrap'].index(row)]}")
        es = sorted((e for e in ev if e["symbol"] == s), key=lambda e: e["pos"])[:cfg["ch4_events"]]
        lo, hi = min(e["pos"] for e in es), max(e["t1pos"] for e in es)
        c = [0] * (hi - lo + 2)
        for e in es:                                                   # difference array: number of events covering each bar
            c[e["pos"] - lo] += 1
            c[e["t1pos"] - lo + 1] -= 1
        run, cover = 0, []
        for x in c[:hi - lo + 1]:
            run += x
            cover.append(run)
        us = []
        for e in es:
            seg = cover[e["pos"] - lo:e["t1pos"] - lo + 1]
            us.append(sum(1.0 / x for x in seg) / len(seg))
        R.check(f"ch4 {s}: events used", int(row["events"]) == len(es), f"{row['events']} vs {len(es)}")
        R.check(f"ch4 {s}: bars spanned", int(row["bars_spanned"]) == hi - lo + 1, f"{row['bars_spanned']} vs {hi - lo + 1}")
        R.close(f"ch4 {s}: mean average uniqueness", row["mean_avg_uniqueness"], sum(us) / len(us), rel=1e-9)
        R.check(f"ch4 {s}: agreement with the project below 1e-9", row["max_abs_diff_vs_project"] < 1e-9, f"{row['max_abs_diff_vs_project']}")
    bars_by = {d["symbol"]: float(d["bars"]) for d in syms_df}
    dens = median([counts[s] / bars_by[s] for s in counts])
    hold = median([e["t1pos"] - e["pos"] for e in ev])
    mc = r["monte_carlo"]
    R.close("ch4 MC: events per bar (median over symbols)", mc["events_per_bar_median"], dens)
    R.close("ch4 MC: median holding (bars)", mc["median_holding_bars"], hold)
    num_obs = int(min(max(round(dens * cfg["ch4_mc_bars"]), 5), cfg["ch4_mc_obs_cap"]))
    R.check("ch4 MC: numObs", mc["numObs"] == num_obs, f"{mc['numObs']} vs {num_obs}")
    R.check("ch4 MC: maxH", mc["maxH"] == int(max(3, round(2 * hold))), f"{mc['maxH']} vs {int(max(3, round(2 * hold)))}")
    R.check("ch4 MC: sequential bootstrap not below standard on average", mc["seqU_mean"] >= mc["stdU_mean"] - 1e-12, f"{mc['seqU_mean']} vs {mc['stdU_mean']}")


def check_ch7(R, j, ev, cfg):
    r = j["results"]["ch7"]
    n = len(ev)
    order, t0ns = sorted_event_order(ev)
    t1ns = [ev[i]["t1"] * 10**9 for i in order]
    R.check("ch7 events", r["events"] == n and r["unique_index"], f"{r['events']} vs {n}, unique={r['unique_index']}")
    ns = cfg["cv"]
    mbrg = int(n * 0.01)
    base, rem = divmod(n, ns)
    sizes = [base + (1 if i < rem else 0) for i in range(ns)]
    i0, total_bad = 0, 0
    import bisect
    for k, sz in enumerate(sizes):
        i, jj = i0, i0 + sz
        i0 = jj
        t0 = t0ns[i]
        tmax1 = max(t1ns[i:jj])
        max_idx = bisect.bisect_left(t0ns, tmax1)
        left = [p for p in range(n) if t1ns[p] <= t0]
        right = list(range(max_idx + mbrg, n)) if max_idx < n else []
        right = [p for p in right if p >= 0]
        train = left + right
        tmin = t0ns[i]
        overlap = sum(1 for p in train if t0ns[p] <= tmax1 and t1ns[p] >= tmin)
        f = r["folds"][k]
        total_bad += overlap
        R.check(f"ch7 fold {k}: test size", f["test"] == sz, f"{f['test']} vs {sz}")
        R.check(f"ch7 fold {k}: train left / right / total", f["train_left"] == len(left) and f["train_right"] == len(right) and f["train"] == len(train), f"{f['train_left']}/{f['train_right']}/{f['train']} vs {len(left)}/{len(right)}/{len(train)}")
        R.check(f"ch7 fold {k}: embargo events", f["embargo_events"] == mbrg, f"{f['embargo_events']} vs {mbrg}")
        R.check(f"ch7 fold {k}: removed vs plain k-fold", f["removed_vs_plain_kfold"] == (n - sz) - len(train), f"{f['removed_vs_plain_kfold']} vs {(n - sz) - len(train)}")
        R.check(f"ch7 fold {k}: training labels overlapping the test span", f["overlapping_train_labels"] == overlap, f"{f['overlapping_train_labels']} vs {overlap}")
        lp = sum(1 for p in range(i) if t1ns[p] >= tmin)
        R.check(f"ch7 fold {k}: left labels that run into the test block", f["left_labels_purged"] == lp, f"{f['left_labels_purged']} vs {lp}")
    R.check("ch7 total overlapping labels", r["total_overlapping_train_labels"] == total_bad, f"{r['total_overlapping_train_labels']} vs {total_bad}")


def check_ch8(R, j, ev, cfg):
    r = j["results"]["ch8"]["sample"]
    order, _ = sorted_event_order(ev)
    seq = [ev[i] for i in order]
    nS = cfg["model_sample"]
    if len(seq) > nS:
        k = len(seq) // nS
        seq = seq[::k][:nS]
    share = sum(e["bin"] for e in seq) / len(seq)
    R.check("ch8 sample size", r["events"] == len(seq) and r["of"] == len(ev), f"{r['events']} of {r['of']} vs {len(seq)} of {len(ev)}")
    R.close("ch8 label share", r["label_share_1"], share)
    R.close("ch8 majority-class share", r["majority_share"], max(share, 1 - share))
    for m in ("MDI", "MDA", "SFI"):
        R.check(f"ch8 {m}: accuracies are probabilities", 0 <= j["results"]["ch8"][m]["oob_accuracy"] <= 1 and 0 <= j["results"]["ch8"][m]["oos_accuracy"] <= 1)
        imp = j["results"]["ch8"][m]["importance_mean"]
        R.check(f"ch8 {m}: one importance per declared feature", len(imp) == 14, f"{len(imp)} features")
    R.check("ch8 MDI importances sum to 1", abs(sum(j["results"]["ch8"]["MDI"]["importance_mean"].values()) - 1) < 1e-9, f"{sum(j['results']['ch8']['MDI']['importance_mean'].values())}")


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bin_sr(sl, pt, freq, p):
    return ((pt - sl) * p + sl) / ((pt - sl) * math.sqrt(p * (1 - p))) * math.sqrt(freq)


def bin_hr(sl, pt, freq, tsr):
    """Smallest precision p reaching Sharpe tsr by bisection on Equation 56 (increasing in p): independent of the closed-form root the book prints."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bin_sr(sl, pt, freq, mid) < tsr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def check_ch15(R, j, ev, src, folder):
    r = j["results"]["ch15"]
    s = r["snippet_15_1"]
    p = s["p"]
    R.close("ch15.1 exact value", s["formula"], (2 * p - 1) / (2 * math.sqrt(p * (1 - p))))
    R.check("ch15.1 mean within 5 standard errors of 2p-1", abs(s["mean"] - (2 * p - 1)) < 5 * s["std"] / math.sqrt(s["n"]), f"{s['mean']} vs {2 * p - 1}")
    R.close("ch15.1 ratio = mean / std", s["sharpe"], s["mean"] / s["std"])
    R.check("ch15.2 identity difference below 1e-12", r["snippet_15_2_max_abs_diff"] < 1e-12, f"{r['snippet_15_2_max_abs_diff']}")
    pred = read_csv(src / "cpcv_predictions.csv")
    p0 = sorted(((int(float(d["event"])), float(d["p"])) for d in pred if int(float(d["path"])) == 0))
    step = float(json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))["params"]["step"])
    wins = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))["meta"]["window_epoch"]
    streams = {}
    for variant in ("primary", "meta"):
        rows = []
        for idx, pp in p0:
            if variant == "meta":
                if pp <= 0.5:
                    continue
                z = (pp - 0.5) / math.sqrt(pp * (1 - pp))
                size = 2 * norm_cdf(z) - 1
                size = min(max(round(size / step) * step, 0.0), 1.0)
                if size <= 0:
                    continue
            rows.append(ev[idx])
        streams[variant] = rows
    by_variant = {row["variant"]: row for row in r["strategy_risk"]}
    for variant, rows in streams.items():
        row = by_variant[variant]
        rets = [e["net_ret"] for e in rows]
        years = (max(e["t1"] for e in rows) - min(e["t0"] for e in rows)) / (365.25 * 86400.0)
        pos = [x for x in rets if x > 0]
        neg = [x for x in rets if x <= 0]
        phat = len(pos) / len(rets)
        freq = len(rets) / years
        R.check(f"ch15 {variant}: bets", row["bets"] == len(rets), f"{row['bets']} vs {len(rets)}")
        R.close(f"ch15 {variant}: years", row["years"], years)
        R.close(f"ch15 {variant}: precision", row["precision"], phat)
        R.close(f"ch15 {variant}: average win", row["avg_win"], sum(pos) / len(pos))
        R.close(f"ch15 {variant}: average loss", row["avg_loss"], sum(neg) / len(neg))
        pt, sl = sum(pos) / len(pos), sum(neg) / len(neg)
        R.close(f"ch15 {variant}: Sharpe implied by observed precision (Eq. 56)", row["implied_sr_at_observed_precision"], bin_sr(sl, pt, freq, phat), rel=1e-9)
        for t in (1, 2):
            R.close(f"ch15 {variant}: precision needed for Sharpe {t} (bisection vs the printed root)", row[f"implied_precision_sr{t}"], bin_hr(sl, pt, freq, float(t)), rel=1e-7, ab=1e-9)
            f_needed = (t * (pt - sl)) ** 2 * phat * (1 - phat) / ((pt - sl) * phat + sl) ** 2
            if abs(bin_sr(sl, pt, f_needed, phat) - t) < 1e-8 * max(1.0, t):
                R.close(f"ch15 {variant}: bets per year needed for Sharpe {t}", row[f"implied_freq_sr{t}"], f_needed, rel=1e-9)
            else:
                R.check(f"ch15 {variant}: bets per year needed for Sharpe {t} is none (extraneous root)", row[f"implied_freq_sr{t}"] is None, f"{row[f'implied_freq_sr{t}']}")
            thres = bin_hr(sl, pt, freq, float(t))
            pf = norm_cdf((thres - phat) / (phat * (1 - phat)))               # as printed: standard deviation p(1-p)
            R.close(f"ch15 {variant}: probability of failure for Sharpe {t} (as printed)", row[f"prob_failure_sr{t}"], pf, rel=1e-6, ab=1e-8)
    # windows
    got = {(w["variant"], w["window"]): w for w in r["windows"]}
    for variant, rows in streams.items():
        for wn, (a, b) in wins.items():
            m_ = [e for e in rows if a <= e["t1"] <= b]
            w = got[(variant, wn)]
            R.check(f"ch15 window {variant}/{wn}: bets", w["bets"] == len(m_), f"{w['bets']} vs {len(m_)}")
            if len(m_) < 30:
                R.check(f"ch15 window {variant}/{wn}: n/a below 30 bets", w["precision"] is None and w["mean_net_bp"] is None)
            else:
                R.close(f"ch15 window {variant}/{wn}: precision", w["precision"], sum(1 for e in m_ if e["net_ret"] > 0) / len(m_))
                R.close(f"ch15 window {variant}/{wn}: mean net return (bp)", w["mean_net_bp"], 1e4 * sum(e["net_ret"] for e in m_) / len(m_))


def check_ch17_19(R, j):
    r = j["results"]
    d = r["ch17"]["event_features_recomputed_by_book_code"]
    for f in ("sadf", "ent_plugin", "ent_konto"):
        R.check(f"ch17/18 {f}: events re-computed by the book's code >= 30 and largest difference to the stored feature below 1e-8",
                d[f]["events_checked"] >= 30 and d[f]["max_abs_diff"] < 1e-8, f"{d[f]}")
    lz = r["ch18"]["lempel_ziv_library"]
    R.check("ch18 Lempel-Ziv: min <= mean <= max words", lz["min_words"] <= lz["mean_words"] <= lz["max_words"], f"{lz}")
    R.check("ch19 Corwin-Schultz: values compared and difference below 1e-9", r["ch19"]["values_compared"] > 1000 and r["ch19"]["max_abs_diff_vs_project"] < 1e-9, f"{r['ch19']['values_compared']}, {r['ch19']['max_abs_diff_vs_project']}")
    R.check("ch19 high-low volatility: rank correlation is a correlation", -1 <= r["ch19"]["rank_correlation_hl_sigma_vs_close_to_close_std_median"] <= 1)


def lin_parts(num_atoms, num_threads):
    """Closed form of Snippet 20.5: equally spaced boundaries, rounded up."""
    n = min(num_threads, num_atoms)
    return [int(math.ceil(x)) for x in np.linspace(0, num_atoms, n + 1)]


def check_ch20(R, j):
    r = j["results"]["ch20"]
    R.check("ch20 Cartesian product: both versions equal and 12 long", r["cartesian"]["equal"] and r["cartesian"]["unvectorized"] == 12 and r["cartesian"]["vectorized"] == 12)
    o = r["one_touch"]
    R.check("ch20 one-touch: single-thread = multiprocessing = vectorised", o["single_equals_multi"] and o["single_equals_vectorised"])
    R.check("ch20 one-touch: touched paths within 0..paths", 0 <= o["touched"] <= o["paths"])
    R.check("ch20 one-touch: printed loop's last-touch differs on no more paths than touched", 0 <= o["last_touch_as_printed_differs_on_paths"] <= o["touched"])
    R.check("ch20 linParts(10, 3)", r["partitions"]["linParts_10_atoms_3"] == lin_parts(10, 3), f"{r['partitions']['linParts_10_atoms_3']} vs {lin_parts(10, 3)}")
    for key in ("nestedParts_10_atoms_3", "nestedParts_upper_10_3"):
        x = r["partitions"][key]
        R.check(f"ch20 {key}: starts at 0, ends at 10, strictly increasing, 4 boundaries", x[0] == 0 and x[-1] == 10 and len(x) == 4 and all(b > a for a, b in zip(x, x[1:])), f"{x}")
    R.check("ch20 getPCs: difference to the direct product below 1e-10", r["getPCs"]["max_abs_diff_vs_direct"] < 1e-10, f"{r['getPCs']['max_abs_diff_vs_direct']}")


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def read_bar_closes(bar_dir, symbol):
    rows = {}
    with open(Path(bar_dir) / "bars" / f"{symbol}.csv", newline="") as f:
        for r in csv.DictReader(f):
            c = V.num(r.get("close"))
            if r["date"] and not math.isnan(c) and c > 0:
                k = V.ts_key(r["date"])
                if k not in rows:
                    rows[k] = c
    return rows


def daily_last(rows):
    last = {}
    for k in sorted(rows):
        last[k // 86400] = rows[k]
    return last


def check_ch21(R, j, ev, cfg):
    r = j["results"]["ch21"]
    counts = {}
    for e in ev:
        counts[e["symbol"]] = counts.get(e["symbol"], 0) + 1
    order = sorted(counts, key=lambda s: (-counts[s], s))
    assets = order[:cfg["ch21_assets"]]
    R.check("ch21 assets are the most active symbols", r["assets"] == assets, f"{r['assets']} vs {assets}")
    bar_dir = j["meta"]["bar_dir"]
    series = {s: daily_last(read_bar_closes(bar_dir, s)) for s in assets}
    days = sorted(set().union(*[set(v) for v in series.values()]))
    ret = []
    for a, b in zip(days, days[1:]):
        row = []
        for s in assets:
            if a in series[s] and b in series[s]:
                row.append(math.log(series[s][b]) - math.log(series[s][a]))
            else:
                row.append(None)
        if all(x is not None for x in row):
            ret.append(row)
    nd, H, n = cfg["ch21_days"], cfg["ch21_horizon"], len(assets)
    R.check("ch21 enough common days", len(ret) >= nd, f"{len(ret)} < {nd}")
    ret = ret[-nd:]
    cuts = [int(x) for x in np.linspace(0, len(ret), H + 1)]
    blocks = [ret[cuts[i]:cuts[i + 1]] for i in range(H)]
    R.check("ch21 block lengths", [len(b) for b in blocks] == r["block_days"], f"{r['block_days']} vs {[len(b) for b in blocks]}")
    np.random.seed(cfg["seed"])
    means, covs, cs = [], [], []
    for b in blocks:
        m = [sum(row[a] for row in b) / len(b) for a in range(n)]
        cov = [[sum((row[a] - m[a]) * (row[c] - m[c]) for row in b) / (len(b) - 1) for c in range(n)] for a in range(n)]
        cc = np.random.uniform(size=n) * np.array([cov[a][a] for a in range(n)]) ** 0.5
        means.append(m)
        covs.append(cov)
        cs.append([float(x) for x in cc])
    for h in range(H):
        for a in range(n):
            R.close(f"ch21 block {h} asset {a}: mean daily return", r["mean_daily"][h][a], means[h][a], rel=1e-9, ab=1e-15)
            R.close(f"ch21 block {h} asset {a}: cost coefficient (book's uniform draw x volatility)", r["cost_coefficients"][h][a], cs[h][a], rel=1e-9, ab=1e-15)

    def weights_vectors(k, n_):
        out = []
        for v in product(range(-k, k + 1), repeat=n_):
            if sum(abs(x) for x in v) == k:
                out.append(tuple(x / k for x in v))
        return out
    wv = weights_vectors(n, n)
    R.check("ch21 distinct weight vectors (L1 sphere of radius k in Z^n)", r["weight_vectors_distinct"] == len(wv), f"{r['weight_vectors_distinct']} vs {len(wv)}")
    R.check("ch21 distinct trajectories", r["trajectories_distinct"] == len(wv) ** H, f"{r['trajectories_distinct']} vs {len(wv) ** H}")

    def sr_of(traj):
        mean, var, prev = 0.0, 0.0, (0.0,) * n
        for h in range(H):
            w = traj[h]
            tc = sum(cs[h][a] * math.sqrt(abs(w[a] - prev[a])) for a in range(n))
            mean += sum(w[a] * means[h][a] for a in range(n)) - tc
            var += sum(w[a] * covs[h][a][c] * w[c] for a in range(n) for c in range(n))
            prev = w
        return mean / math.sqrt(var)
    best, best_t = None, None
    for traj in product(wv, repeat=H):
        s = sr_of(traj)
        if best is None or s > best:
            best, best_t = s, traj
    R.close("ch21 dynamic solution: best Sharpe over all distinct trajectories", r["dynamic_sr"], best, rel=1e-9, ab=1e-12)
    dyn = r["dynamic_weights"]
    R.check("ch21 dynamic solution: weights (rounded to 4 places)", all(abs(dyn[a][h] - best_t[h][a]) < 1.1e-4 for a in range(n) for h in range(H)), f"{dyn} vs {best_t}")
    stat = []
    for h in range(H):
        S = np.array(covs[h])
        a_ = np.array(means[h]).reshape(-1, 1)
        w = np.linalg.solve(S, a_)
        w = w / (a_.T @ w).item()
        w = w / np.abs(w).sum()
        stat.append(tuple(float(x) for x in w.ravel()))
    R.close("ch21 static solution Sharpe (closed form, re-solved each block, with trading costs)", r["static_sr"], sr_of(tuple(stat)), rel=1e-8, ab=1e-10)
    R.check("ch21 every weight column has total absolute weight 1", all(abs(sum(abs(x) for x in w) - 1) < 1e-9 for w in best_t))


def example_4_6_exact():
    """Exact expected average uniqueness of the book's three-label example (t0 = 0, 2, 4; t1 = 2, 3, 5; bars 0..5) for a bootstrap sample of three labels: standard (uniform draws with
    replacement) and sequential (Snippet 4.5's probabilities proportional to the uniqueness each candidate would have), by enumeration of every draw sequence."""
    spans = [(0, 2), (2, 3), (4, 5)]
    nb = 6

    def avg_u(sample):
        c = [0] * nb
        for i in sample:
            for t in range(spans[i][0], spans[i][1] + 1):
                c[t] += 1
        us = []
        for i in sample:
            seg = range(spans[i][0], spans[i][1] + 1)
            us.append(sum(1.0 / c[t] for t in seg) / len(seg))
        return sum(us) / len(us)
    std = sum(avg_u((a, b, c)) for a in range(3) for b in range(3) for c in range(3)) / 27.0

    def seq(phi, prob):
        if len(phi) == 3:
            return prob * avg_u(phi)
        cand = []
        for i in range(3):
            c = [0] * nb
            for k in list(phi) + [i]:
                for t in range(spans[k][0], spans[k][1] + 1):
                    c[t] += 1
            seg = range(spans[i][0], spans[i][1] + 1)
            cand.append(sum(1.0 / c[t] for t in seg) / len(seg))
        tot = sum(cand)
        return sum(seq(list(phi) + [i], prob * cand[i] / tot) for i in range(3))
    return std, seq([], 1.0)


def check_examples(R, j):
    r = j["results"]
    c = r["ch3"]["cusum_snippet_2_4"]
    R.check("ch3 Snippet 2.4: printed CUSUM = the project's filter on real bars for every symbol", bool(c) and all(v["identical"] and v["events_book"] == v["events_project"] and v["events_book"] > 0 for v in c.values()), f"{c}")
    e = r["ch4"]["book_example_4_6"]
    std, seq = example_4_6_exact()
    tol = 5 * 0.25 / math.sqrt(e["reps"])
    R.check("ch4 Snippet 4.6: standard-bootstrap mean uniqueness near the exact value", abs(e["standard_bootstrap_uniqueness_mean"] - std) < tol, f"{e['standard_bootstrap_uniqueness_mean']} vs exact {std} (tol {tol})")
    R.check("ch4 Snippet 4.6: sequential-bootstrap mean uniqueness near the exact value", abs(e["sequential_bootstrap_uniqueness_mean"] - seq) < tol, f"{e['sequential_bootstrap_uniqueness_mean']} vs exact {seq} (tol {tol})")
    R.check("ch4 Snippet 4.6: exact sequential value above the exact standard value", seq > std, f"{seq} vs {std}")
    t = r["ch16"]["table_16_1"]
    R.check("ch16 Snippet 16.4 reproduces Table 16.1 (printed to two decimals) for HRP, IVP and the CLA stand-in", all(t[k] <= 0.0051 for k in ("HRP", "IVP", "CLA")), f"{t}")
    m = r["ch13"]["main_13_1"]
    keys = {f"{f}|{h}" for f in (10, 5, 0, -5, -10) for h in (5, 10, 25, 50, 100)}
    R.check("ch13 Snippet 13.1: the 25 printed (forecast, half-life) combinations", set(m["best_node"]) == keys, f"{sorted(m['best_node'])}")
    R.check("ch13 Snippet 13.1: best nodes lie on the printed mesh (0..10 in steps of 0.5) and are at least the mesh median",
            all(0 <= d["pt"] <= 10 and 0 <= d["sl"] <= 10 and abs(d["pt"] * 2 - round(d["pt"] * 2)) < 1e-9 and abs(d["sl"] * 2 - round(d["sl"] * 2)) < 1e-9 and d["sharpe"] >= d["median_sharpe_over_mesh"] for d in m["best_node"].values()))
    w = r["ch8"]["snippet_8_6_printed_example"]
    R.check("ch8 Snippet 8.6 printed example is a tau in [-1, 1]", -1 <= w <= 1, f"{w}")


def day_str(d):
    import datetime
    return (datetime.datetime(1970, 1, 1) + datetime.timedelta(days=int(d))).strftime("%Y-%m-%d")


def subset_daily(j):
    """{symbol: {utc day number: last bar close of that day}} for the run's subset symbols, from the raw bar files."""
    return {s: daily_last(read_bar_closes(j["meta"]["bar_dir"], s)) for s in j["meta"]["subset"]}


def hole_filtered_days(series, min_share=0.8):
    days = sorted(set().union(*[set(v) for v in series.values()]))
    nsym = len(series)
    return [d for d in days if sum(1 for v in series.values() if d in v) / nsym >= min_share]


def gap_returns(series, days, syms, max_gap, kind):
    """{day: {symbol: return}} between consecutive rows of `days`, only when the two rows are at most max_gap days apart; a missing close gives None."""
    out = {}
    for a, b in zip(days, days[1:]):
        if b - a > max_gap:
            continue
        row = {}
        for s in syms:
            if a in series[s] and b in series[s]:
                row[s] = (math.log(series[s][b]) - math.log(series[s][a])) if kind == "log" else series[s][b] / series[s][a] - 1.0
            else:
                row[s] = None
        out[b] = row
    return out


def check_ch2(R, j, cfg):
    r = j["results"]["ch2"]
    series = subset_daily(j)
    syms = sorted(series, key=lambda s: j["meta"]["subset"].index(s))
    max_gap = 1 if j["meta"]["venue"] == "kraken" else 5
    all_days = sorted(set().union(*[set(v) for v in series.values()]))
    days = hole_filtered_days(series)
    rets = gap_returns(series, days, syms, max_gap, "log")
    rows = sorted(rets)[-cfg["ch2_days"]:]
    keep = [s for s in syms if sum(1 for d in rows if rets[d][s] is not None) / len(rows) >= 0.95]
    full = [d for d in rows if all(rets[d][s] is not None for s in keep)]
    n = len(keep)
    R.check("ch2 symbols used", r["symbols_used"] == n, f"{r['symbols_used']} vs {n}")
    R.check("ch2 common days", r["days"] == len(full), f"{r['days']} vs {len(full)}")
    R.check("ch2 window", r["window"] == [day_str(full[0]), day_str(full[-1])], f"{r['window']} vs {[day_str(full[0]), day_str(full[-1])]}")
    R.check("ch2 data-hole days inside the window span", r["days_dropped_as_data_holes"] == sum(1 for d in all_days if d not in set(days) and d >= rows[0]), f"{r['days_dropped_as_data_holes']}")
    X = np.array([[rets[d][s] for s in keep] for d in full])
    cov = np.cov(X, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    R.close("ch2 largest eigenvalue share", r["eigen"]["first_share"], float(ev[0] / ev.sum()), rel=1e-8)
    R.close("ch2 three largest eigenvalues share", r["eigen"]["top3_share"], float(ev[:3].sum() / ev.sum()), rel=1e-8)
    R.close("ch2 smallest / largest eigenvalue", r["eigen"]["smallest_over_largest"], float(ev[-1] / ev[0]), rel=1e-6)
    R.check("ch2 risk-allocation variance equals the target 1", abs(r["last_pc"]["variance"] - 1) < 1e-6 and abs(r["even_pcs"]["variance"] - 1) < 1e-6, f"{r['last_pc']['variance']}, {r['even_pcs']['variance']}")


def check_ch16(R, j, cfg):
    r = j["results"]["ch16"]
    series = subset_daily(j)
    syms = sorted(series, key=lambda s: j["meta"]["subset"].index(s))
    max_gap = 1 if j["meta"]["venue"] == "kraken" else 5
    all_days = sorted(set().union(*[set(v) for v in series.values()]))
    end = all_days[-1]
    in_span = [d for d in all_days if d >= end - cfg["ch16_days"]]
    nsym = len(syms)
    days = [d for d in in_span if sum(1 for s in syms if d in series[s]) / nsym >= 0.8]
    filled = {}
    for s in syms:                                              # forward fill of at most 3 consecutive missing rows
        col, last, run = [], None, 0
        for d in days:
            if d in series[s]:
                last, run = series[s][d], 0
                col.append(last)
            else:
                run += 1
                col.append(last if (last is not None and run <= 3) else None)
        filled[s] = col
    keep = [s for s in syms if sum(1 for x in filled[s] if x is not None) / len(days) >= 0.98]
    rows = [i for i in range(len(days)) if all(filled[s][i] is not None for s in keep)]
    ds = [days[i] for i in rows]
    ser2 = {s: {days[i]: filled[s][i] for i in rows} for s in keep}
    rets = gap_returns(ser2, ds, keep, max_gap, "pct")
    rd = [d for d in sorted(rets) if all(rets[d][s] is not None for s in keep)]
    X = np.array([[rets[d][s] for s in keep] for d in rd])
    cov = np.cov(X, rowvar=False)
    R.check("ch16 symbols", r["data"]["symbols"] == len(keep), f"{r['data']['symbols']} vs {len(keep)}")
    R.check("ch16 days and window", r["data"]["days"] == len(rd) and r["data"]["window"] == [day_str(rd[0]), day_str(rd[-1])], f"{r['data']} vs {len(rd)} {day_str(rd[0])} {day_str(rd[-1])}")
    ivp = (1.0 / np.diag(cov)) / (1.0 / np.diag(cov)).sum()
    R.close("ch16 inverse-variance portfolio: effective number of positions", r["full_sample"]["effective_n"]["ivp"], float(1.0 / (ivp ** 2).sum()), rel=1e-8)
    R.close("ch16 inverse-variance portfolio: largest weight", r["full_sample"]["max_weight"]["ivp"], float(ivp.max()), rel=1e-8)
    for k in ("hrp", "ivp", "cla"):
        R.check(f"ch16 {k}: effective N between 1 and the number of symbols", 1 - 1e-9 <= r["full_sample"]["effective_n"][k] <= len(keep) + 1e-9, f"{r['full_sample']['effective_n'][k]}")
        R.check(f"ch16 {k}: largest weight in (0, 1]", 0 < r["full_sample"]["max_weight"][k] <= 1 + 1e-12)
    R.check("ch16 HRP agrees with the project's hrp_alloc.py", r["full_sample"]["hrp_max_abs_diff_vs_project"] < 1e-9, f"{r['full_sample']['hrp_max_abs_diff_vs_project']}")


def check_text(R, folder, j):
    t = (folder / "SUMMARY.md").read_text(encoding="utf-8")
    r = j["results"]
    R.check("SUMMARY.md names the venue's run and events", f"{j['meta']['events']:,} events" in t, "event count not found")
    for k, row in enumerate(r["ch7"]["folds"]):
        R.check(f"SUMMARY ch7 fold {k} row", f"| {row['fold']} | {row['test']:,} | {row['train']:,} | {row['removed_vs_plain_kfold']:,} | {row['train_left']:,} / {row['train_right']:,} | {row['left_labels_purged']:,} |" in t)
    for row in r["ch4"]["uniqueness_and_bootstrap"]:
        R.check(f"SUMMARY ch4 {row['symbol']} mean uniqueness {row['mean_avg_uniqueness']:.4f}", f"| {row['symbol']} | {row['events']} | {row['bars_spanned']:,} | {row['mean_avg_uniqueness']:.4f} |" in t)
    R.check("SUMMARY ch8 majority share", f"Majority-class share {r['ch8']['sample']['majority_share']:.4f}" in t)
    R.check("SUMMARY ch21 Sharpe values", f"Sharpe {r['ch21']['static_sr']:.3f}" in t and f"Sharpe {r['ch21']['dynamic_sr']:.3f}" in t)
    for row in r["ch15"]["strategy_risk"]:
        R.check(f"SUMMARY ch15 {row['variant']} row", f"| {row['variant']} | {row['bets']:,} | {row['bets_per_year']:,.0f} | {row['precision']:.4f} | {row['avg_win']:.5f} | {row['avg_loss']:.5f} |" in t)
    for w in r["ch15"]["windows"]:
        pr = "n/a" if w["precision"] is None else format(w["precision"], ".4f")
        bp = "n/a" if w["mean_net_bp"] is None else format(w["mean_net_bp"], ".1f")
        R.check(f"SUMMARY ch15 window {w['variant']}/{w['window']}", f"| {w['variant']} | {w['window']} | {w['bets']:,} | {pr} | {bp} |" in t)
    R.check("SUMMARY states the probability-of-failure caution", "Caution on the 'probability of failure' column" in t)
    R.check("SUMMARY has no failed section", "This section failed" not in t)


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    folder, j, src, ev = load(argv[1])
    cfg = j["meta"]["config"]
    R = Report()
    check_integrity(R, folder, j, src, ev)
    check_ch3(R, j, ev)
    check_ch4(R, j, ev, src, cfg)
    check_ch7(R, j, ev, cfg)
    check_ch8(R, j, ev, cfg)
    check_ch15(R, j, ev, src, folder)
    check_ch17_19(R, j)
    check_ch20(R, j)
    check_ch21(R, j, ev, cfg)
    check_ch2(R, j, cfg)
    check_ch16(R, j, cfg)
    check_examples(R, j)
    check_text(R, folder, j)
    return R.done()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
