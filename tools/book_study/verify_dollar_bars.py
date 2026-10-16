#!/usr/bin/env python
"""Independent check of the dollar bars written by build_dollar_bars.py.  Standard library only; it does not import the builder.

From the SOURCE files (Kraken: official 1-minute files + trade-derived 1-minute files; Alpaca: the hourly files) it rebuilds, with plain loops, the rows of a symbol, the segments, every bar
(start, end, open, high, low, close, volume, value, rows, threshold) and the hourly reference series, following the definition printed at the top of build_dollar_bars.py, and compares them with
bars/<SYM>.csv and ref_1h/<SYM>.csv.  It also checks the manifest entry (bar count, source file md5, file md5) and re-derives about every 500th threshold by direct summation over the window.

    python tools/book_study/verify_dollar_bars.py kraken --dump <Kraken OHLCVT folder> --fine runtime/kraken_fine_export --bars runtime/dollar_bars/kraken --symbols BTC ETH ...
    python tools/book_study/verify_dollar_bars.py alpaca --hourly runtime/alpaca_hourly --bars runtime/dollar_bars/alpaca --symbols AAPL ...
Exit code 0 = all pass.
"""
from __future__ import annotations

import calendar
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

WINDOW_S = 30 * 86400
MIN_HIST_S = 7 * 86400
BREAK_S = {"kraken": 48 * 3600, "alpaca": 120 * 3600}
ROW_S = {"kraken": 60, "alpaca": 3600}
CUT_S = calendar.timegm((2025, 4, 1, 0, 0, 0))
DUMP_PAIR = {"BTC": "XBTUSD", "DOGE": "XDGUSD"}
NAN = math.nan


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v


def parse_ts(s: str) -> int:
    d, t = s.split(" ")
    y, mo, da = (int(x) for x in d.split("-"))
    h, mi, se = (int(x) for x in t.split(":"))
    return calendar.timegm((y, mo, da, h, mi, se))


def fmt_ts(t: int) -> str:
    g = time.gmtime(t)
    return "%04d-%02d-%02d %02d:%02d:%02d" % (g.tm_year, g.tm_mon, g.tm_mday, g.tm_hour, g.tm_min, g.tm_sec)


def read_rows_noheader(path):
    """Kraken official file: ts,open,high,low,close,volume,trades (no header)."""
    out = []
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return out
    with open(p, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 6:
                continue
            out.append((int(r[0]), num(r[1]), num(r[2]), num(r[3]), num(r[4]), num(r[5])))
    return out


def read_rows_header(path, ts_col="ts"):
    out = []
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return out
    with open(p, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            ts = r[ts_col]
            t = int(ts) if ts_col == "ts" else parse_ts(ts)
            out.append((t, num(r["open"]), num(r["high"]), num(r["low"]), num(r["close"]), num(r["volume"])))
    return out


def clean(rows):
    """Close missing or <= 0 dropped; duplicate times: the first in the given order is kept; sorted by time."""
    seen, keep = set(), []
    for r in rows:
        c = r[4]
        if c != c or not c > 0:
            continue
        if r[0] in seen:
            continue
        seen.add(r[0])
        keep.append(r)
    keep.sort(key=lambda r: r[0])
    return keep


def kraken_rows(dump_dir, fine_dir, sym):
    pair = DUMP_PAIR.get(sym, sym + "USD")
    d = read_rows_noheader(Path(dump_dir) / f"{pair}_1.csv")
    f = read_rows_header(Path(fine_dir) / f"{sym}_trades_ft_1m.csv")
    r = read_rows_header(Path(fine_dir) / f"{sym}_trades_refresh_1m.csv")
    r0 = min(x[0] for x in r) if r else None
    parts = [x for x in d if x[0] < CUT_S]
    parts += [x for x in f if x[0] >= CUT_S and (r0 is None or x[0] < r0)]
    parts += [x for x in r if x[0] >= CUT_S]
    return clean(parts)


def alpaca_rows(hourly_dir, sym):
    return clean(read_rows_header(Path(hourly_dir) / f"{sym}_1h.csv", ts_col="date"))


def row_value(r) -> float:
    v = (r[2] + r[3] + r[4]) / 3.0 * r[5]
    return v if (v == v and not math.isinf(v) and v > 0) else 0.0


def segments(rows, dur, brk):
    out, a = [], 0
    for i in range(1, len(rows)):
        if rows[i][0] - (rows[i - 1][0] + dur) > brk:
            out.append((a, i))
            a = i
    if rows:
        out.append((a, len(rows)))
    return out


def bars_of(rows, dur, brk, brute_every=500):
    """Returns (bars, brute_failures, brute_checked).  A bar: dict with start, end, open, high, low, close, volume, value, n, thr."""
    bars, brute_bad, brute_n = [], [], 0
    for (a0, b0) in segments(rows, dur, brk):
        seg = rows[a0:b0]
        n = len(seg)
        s = [r[0] for r in seg]
        v = [row_value(r) for r in seg]
        cs = [0.0]
        for x in v:
            cs.append(cs[-1] + x)
        Hs, Ls, Vs = [r[2] for r in seg], [r[3] for r in seg], [r[5] for r in seg]
        clean_hl = not any(x != x for x in Hs) and not any(x != x for x in Ls) and not any(x != x for x in Vs)
        a = s[0]
        j = 0
        while j < n and s[j] < a + MIN_HIST_S:
            j += 1
        lo = hi = 0
        while j < n:
            t = s[j]
            w0 = max(a, t - WINDOW_S)
            while lo < n and s[lo] < w0:
                lo += 1
            hi_ = hi
            while hi_ < n and s[hi_] + dur <= t:
                hi_ += 1
            hi = hi_
            hours = (t - w0) / 3600.0
            thr = (cs[hi] - cs[lo]) / hours if (hours > 0 and hi > lo) else 0.0
            if not (thr == thr) or math.isinf(thr) or thr <= 0.0:
                j += 1
                continue
            target = cs[j] + thr
            k = j
            while k < n and cs[k + 1] < target:
                k += 1
            if k >= n:
                break
            if clean_hl:                                    # no missing high / low / volume in this segment: the built-in max / min / sum give the same numbers as the row loop
                hmax, lmin = max(Hs[j:k + 1]), min(Ls[j:k + 1])
                hnan = lnan = False
                vol = sum(Vs[j:k + 1])
            else:
                vol = 0.0
                hmax, lmin = -math.inf, math.inf
                hnan = lnan = False
                for i in range(j, k + 1):
                    r = seg[i]
                    x = r[5]
                    vol += 0.0 if x != x else x
                    if r[2] != r[2]:
                        hnan = True
                    elif r[2] > hmax:
                        hmax = r[2]
                    if r[3] != r[3]:
                        lnan = True
                    elif r[3] < lmin:
                        lmin = r[3]
            bars.append({"start": s[j], "end": s[k] + dur, "open": seg[j][1], "high": NAN if hnan else hmax, "low": NAN if lnan else lmin, "close": seg[k][4],
                         "volume": vol, "value": cs[k + 1] - cs[j], "n": k - j + 1, "thr": thr})
            if len(bars) % brute_every == 1:
                brute_n += 1
                tot = 0.0
                for r in seg:                               # direct definition: rows that start in [w0, t) and end by t
                    if r[0] >= w0 and r[0] + dur <= t:
                        tot += row_value(r)
                bthr = tot / hours
                if not close(bthr, thr, 1e-9):
                    brute_bad.append((fmt_ts(t), thr, bthr))
            j = k + 1
    return bars, brute_bad, brute_n


def hourly_of(rows):
    out, cur = [], None
    for r in rows:
        h = r[0] // 3600 * 3600
        if cur is None or cur["ts"] != h:
            if cur is not None:
                out.append(cur)
            cur = {"ts": h, "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": 0.0 if r[5] != r[5] else r[5]}
            cur["_seen_v"] = r[5] == r[5]
            continue
        if r[2] == r[2] and (cur["high"] != cur["high"] or r[2] > cur["high"]):
            cur["high"] = r[2]
        if r[3] == r[3] and (cur["low"] != cur["low"] or r[3] < cur["low"]):
            cur["low"] = r[3]
        cur["close"] = r[4]
        if r[5] == r[5]:
            cur["volume"] += r[5]
            cur["_seen_v"] = True
    if cur is not None:
        out.append(cur)
    for c in out:
        if not c.pop("_seen_v"):
            c["volume"] = 0.0                                       # sum of nothing
    return out


def close(a, b, rel=1e-9, ab=1e-12) -> bool:
    if a != a and b != b:
        return True
    if a != a or b != b:
        return False
    return abs(a - b) <= max(ab, rel * max(abs(a), abs(b)))


def read_csv_bars(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x) -> float:
    return NAN if x == "" else float(x)


def verify_symbol(venue, sym, rows, bars_dir, man_entry, report):
    """report(name, ok, detail) ; returns None"""
    dur, brk = ROW_S[venue], BREAK_S[venue]
    mine, brute_bad, brute_n = bars_of(rows, dur, brk)
    theirs = read_csv_bars(Path(bars_dir) / "bars" / f"{sym}.csv")
    report(f"{sym}: number of bars (rebuilt {len(mine):,}, file {len(theirs):,}, manifest {man_entry.get('bars')})", len(mine) == len(theirs) == int(man_entry["bars"]), "")
    bad = []
    for i, (m, t) in enumerate(zip(mine, theirs)):
        if (fmt_ts(m["end"]) != t["date"] or fmt_ts(m["start"]) != t["start"] or int(t["n_rows"]) != m["n"]
                or not all(close(m[k], fnum(t[k]), 1e-11) for k in ("open", "high", "low", "close"))
                or not all(close(m[k], fnum(t[kk]), 1e-8) for k, kk in (("volume", "volume"), ("value", "value"), ("thr", "threshold")))):
            bad.append(i)
            if len(bad) >= 5:
                break
    report(f"{sym}: every bar (start, end, open, high, low, close, volume, value, rows, threshold) equals the rebuilt bar", not bad,
           f"first differing bars {bad}: rebuilt {mine[bad[0]] if bad else ''} file {theirs[bad[0]] if bad else ''}")
    report(f"{sym}: {brute_n} thresholds re-derived by direct summation over their window", not brute_bad and brute_n > 0, str(brute_bad[:3]))
    ordered = all(mine[i]["start"] >= mine[i - 1]["end"] for i in range(1, len(mine)))
    report(f"{sym}: bars are in order, never overlap", ordered, "")
    # no bar spans a break: all rows inside a bar are consecutive within a segment, tested via the row index of its start and end
    idx = {r[0]: i for i, r in enumerate(rows)}
    span_ok = True
    for m in mine[:: max(1, len(mine) // 2000)]:
        i0, i1 = idx[m["start"]], idx[m["end"] - dur]
        for i in range(i0 + 1, i1 + 1):
            if rows[i][0] - (rows[i - 1][0] + dur) > brk:
                span_ok = False
                break
    report(f"{sym}: no sampled bar spans a data break", span_ok, "")
    h = hourly_of(rows)
    ref = read_csv_bars(Path(bars_dir) / "ref_1h" / f"{sym}.csv")
    ok_ref = len(h) == len(ref) and all(fmt_ts(a["ts"]) == b["date"] and all(close(a[k], fnum(b[k]), 1e-11) for k in ("open", "high", "low", "close")) and close(a["volume"], fnum(b["volume"]), 1e-8)
                                       for a, b in zip(h, ref))
    report(f"{sym}: hourly reference series ({len(h):,} hours) equals the rebuilt one", ok_ref, f"rebuilt {len(h)} file {len(ref)}")
    report(f"{sym}: bars file md5 equals the manifest", md5(Path(bars_dir) / "bars" / f"{sym}.csv") == man_entry["bars_md5"], "")
    report(f"{sym}: reference file md5 equals the manifest", md5(Path(bars_dir) / "ref_1h" / f"{sym}.csv") == man_entry["ref_md5"], "")


def main(argv):
    if not argv or argv[0] not in ("kraken", "alpaca"):
        raise SystemExit(__doc__)
    venue = argv[0]
    opts, i = {}, 1
    while i < len(argv):
        k = argv[i]
        if k == "--symbols":
            j = i + 1
            opts["symbols"] = []
            while j < len(argv) and not argv[j].startswith("--"):
                opts["symbols"].append(argv[j])
                j += 1
            i = j
        else:
            opts[k.lstrip("-")] = argv[i + 1]
            i += 2
    bars_dir = Path(opts["bars"])
    man = json.loads((bars_dir / "manifest.json").read_text())
    results = []

    def report(name, ok, detail):
        results.append(bool(ok))
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok or not detail else "\n      " + str(detail)), flush=True)

    report(f"manifest is for the {venue} venue with window 30 days, history 7 days, break {BREAK_S[venue] // 3600} h, row {ROW_S[venue]} s",
           man["venue"] == venue and man["window_days"] == 30 and man["min_history_days"] == 7 and man["break_hours"] == BREAK_S[venue] // 3600 and man["row_seconds"] == ROW_S[venue], "")
    for sym in opts["symbols"]:
        t0 = time.time()
        e = man["symbols"][sym]
        if venue == "kraken":
            dump, fine = Path(opts["dump"]), Path(opts["fine"])
            pair = DUMP_PAIR.get(sym, sym + "USD")
            src = {"dump": dump / f"{pair}_1.csv", "ft": fine / f"{sym}_trades_ft_1m.csv", "refresh": fine / f"{sym}_trades_refresh_1m.csv"}
            sm = e["source_md5"]
            same = all((md5(p) if p.exists() and p.stat().st_size > 0 else None) == sm[k] for k, p in src.items())
            report(f"{sym}: source files have the md5 recorded by the builder", same, str(sm))
            rows = kraken_rows(dump, fine, sym)
        else:
            p = Path(opts["hourly"]) / f"{sym}_1h.csv"
            report(f"{sym}: source file has the md5 recorded by the builder", md5(p) == e["source_md5"], "")
            rows = alpaca_rows(opts["hourly"], sym)
        report(f"{sym}: rows rebuilt ({len(rows):,}) equal the manifest", len(rows) == int(e["rows"]), f"{len(rows)} vs {e['rows']}")
        verify_symbol(venue, sym, rows, bars_dir, e, report)
        print(f"      ({sym} took {time.time() - t0:.0f}s)", flush=True)
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
