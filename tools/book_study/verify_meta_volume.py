#!/usr/bin/env python
"""Independent checker for a run_meta_volume.py output folder (stdlib only).

It is verify_meta_venue.py with three additions, none of which imports volume_features.py, meta_events_vol.py or numpy:
  * the four volume features (kyle_lam, amihud_lam, vpin, dvol_z) are recomputed for EVERY event from the raw candle file with plain loops,
    straight from the definitions in the docstring of volume_features.py, and compared with events.csv (every column, every event);
  * the events dropped for a non-finite volume feature are counted independently and must equal the run's count;
  * volume_extra.json (written by run_meta_volume.py) must list the declared features and windows, and the md5 of the three volume scripts next to this checker
    must equal the md5 recorded in the run.
Everything else (events, weights, CPCV, paths, DSR, walk-forward, windows, SUMMARY cells) is checked by verify_meta_venue.py as for the plain venue run, with
the 18-feature list.  Exit code 0 = all checks pass.

    python tools/book_study/verify_meta_volume.py runtime/book_meta_kraken_vol/<timestamp>        # prior trial count 104 is expected; override with --prior-trials N
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import verify_book_study as V
import verify_meta_venue as VM

HERE = Path(__file__).resolve().parent
NAMES = ["kyle_lam", "amihud_lam", "vpin", "dvol_z"]
W_REG, W_FLOW, W_Z = 168, 24, 720
EXPECTED_SEED = 20261017
EXPECTED_PRIOR_VOLUME = 104
NAN = math.nan


def _read_volume_bars(candle_dir, symbol):
    """(sorted epoch keys, closes, volumes) with the loader's rules: close present and > 0; the last row wins for a repeated timestamp."""
    rows = {}
    for r in V.read(Path(candle_dir) / f"{symbol}_1h.csv"):
        c = V.num(r.get("close"))
        t = r.get("date")
        if t is not None and not math.isnan(c):
            rows[V.ts_key(t)] = (c, V.num(r.get("volume")))
    keys = sorted(k for k in rows if rows[k][0] > 0)
    return keys, [rows[k][0] for k in keys], [rows[k][1] for k in keys]


def prep(candle_dir, symbol, run_keys):
    keys, c, v = _read_volume_bars(candle_dir, symbol)
    n = len(run_keys)
    if keys[:n] != [float(k) for k in run_keys]:
        raise ValueError(f"{symbol}: the volume file's bars do not equal the bars the checker uses")
    c, v = c[:n], v[:n]
    r = [NAN] + [math.log(c[i] / c[i - 1]) for i in range(1, n)]
    dv = [c[i] * v[i] for i in range(n)]
    b, cur = [0.0] * n, 0.0
    for i in range(1, n):
        d = c[i] - c[i - 1]
        if d > 0:
            cur = 1.0
        elif d < 0:
            cur = -1.0
        b[i] = cur
    sig_cache, m_cache = {}, {}

    def slope(x, y):
        m = len(x)
        xm, ym = math.fsum(x) / m, math.fsum(y) / m
        sxx = math.fsum([(a - xm) ** 2 for a in x])
        if not sxx > 0:
            return NAN
        return math.fsum([(a - xm) * (q - ym) for a, q in zip(x, y)]) / sxx

    def sigma(i):                      # population std of the W_REG returns ending at bar i
        if i not in sig_cache:
            if i - W_REG + 1 < 1:
                sig_cache[i] = NAN
            else:
                rs = r[i - W_REG + 1:i + 1]
                mu = math.fsum(rs) / W_REG
                sig_cache[i] = math.sqrt(math.fsum([(q - mu) ** 2 for q in rs]) / W_REG)
        return sig_cache[i]

    def m_at(i):                       # log(1 + mean dollar volume of the W_FLOW bars ending at i)
        if i not in m_cache:
            m_cache[i] = math.log1p(math.fsum(dv[i - W_FLOW + 1:i + 1]) / W_FLOW) if i - W_FLOW + 1 >= 0 else NAN
        return m_cache[i]

    def f(p):
        out = {k: NAN for k in NAMES}
        if p >= W_REG:
            lo = p - W_REG + 1
            idx = range(lo, p + 1)
            mean_dv = math.fsum([dv[i] for i in idx]) / W_REG
            out["kyle_lam"] = slope([b[i] * dv[i] for i in idx], [r[i] for i in idx]) * mean_dv
            out["amihud_lam"] = slope([dv[i] for i in idx], [abs(r[i]) for i in idx]) * mean_dv
        if p >= W_REG + W_FLOW - 1:
            num = den = 0.0
            ok = True
            for i in range(p - W_FLOW + 1, p + 1):
                sd = sigma(i)
                if not (sd > 0):
                    ok = False
                    break
                num += v[i] * abs(math.erf(r[i] / (sd * math.sqrt(2.0))))
                den += v[i]
            if ok and den > 0:
                out["vpin"] = num / den
        if p >= W_FLOW - 1 + W_Z - 1:
            ms = [m_at(i) for i in range(p - W_Z + 1, p + 1)]
            mu = math.fsum(ms) / W_Z
            sd = math.sqrt(math.fsum([(q - mu) ** 2 for q in ms]) / W_Z)
            if sd > 0:
                out["dvol_z"] = (ms[-1] - mu) / sd
        return {k: (x if math.isfinite(x) else NAN) for k, x in out.items()}

    return f


def extra_checks(d, meta, check):
    p = Path(d) / "volume_extra.json"
    ok_file = p.exists()
    extra = json.loads(p.read_text(encoding="utf-8")) if ok_file else {}
    check("volume_extra.json lists the four declared volume features, windows 168 / 24 / 720 and this run's seed and prior trial count",
          ok_file and extra.get("volume_features") == NAMES and extra.get("windows") == {"W_REG": W_REG, "W_FLOW": W_FLOW, "W_Z": W_Z}
          and extra.get("seed") == EXPECTED_SEED and extra.get("prior_trials") == VM.EXPECTED_PRIOR, str(extra.get("windows")))
    changed = []
    for name, md5 in (extra.get("script_md5") or {}).items():
        q = HERE / name
        if not q.exists() or hashlib.md5(q.read_bytes()).hexdigest() != md5:
            changed.append(name)
    check("the volume scripts next to this checker (volume_features.py, meta_events_vol.py, run_meta_volume.py) are the ones that produced the run (md5 recorded in the run)",
          ok_file and sorted((extra.get("script_md5") or {})) == ["meta_events_vol.py", "run_meta_volume.py", "volume_features.py"] and not changed, f"changed or missing: {changed}")


VM.EXTRA_FEATURES = list(NAMES)
VM.EXTRA_TOL = {"kyle_lam": (1e-6, 1e-12), "amihud_lam": (1e-6, 1e-12), "vpin": (1e-7, 1e-10), "dvol_z": (1e-7, 1e-9)}
VM.EXTRA_PREP = prep
VM.EXTRA_DROP_KEY = "events_dropped_missing_volume_feature"
VM.EXTRA_CHECKS = extra_checks


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    VM.EXPECTED_PRIOR = EXPECTED_PRIOR_VOLUME
    if "--prior-trials" in argv:
        i_ = argv.index("--prior-trials")
        VM.EXPECTED_PRIOR = int(argv[i_ + 1])
        del argv[i_:i_ + 2]
    try:
        raise SystemExit(VM.main(argv[0]))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
