"""Dollar bars (the book's "volume clock", Chapter 2: a bar closes when a fixed amount of VALUE has traded, not when a fixed amount of TIME has passed), for Kraken and Alpaca.

WHERE THIS DIFFERS FROM THE BOOK (declared before any result was seen):
  * The book builds bars from ticks.  Kraken: the finest data are Kraken's official 1-minute files (to 2025-03-31) and 1-minute rows built by tools/book_study/export_kraken_post_dump.py from the trade
    files (from 2025-04-01), so a Kraken row is one minute.  Alpaca: the finest data downloaded are 1-hour bars (tools/book_study/fetch_alpaca_hourly.py), so an Alpaca row is one hour; a bar there is a whole
    number of hours and the clock is only coarsely changed in busy hours.  A row is never split.
  * The threshold is the trailing average traded value per clock hour (below), the project's own choice; the book leaves the size of a bar to the user.

DEFINITION (written from the idea, not copied):
  row value        v = ((high + low + close) / 3) * volume             (missing or negative -> 0)
  segment          a maximal run of rows in which no two neighbours are more than BREAK apart (Kraken 48 h, Alpaca 120 h).  A bar never spans a break, a bar that is still open at a break is dropped,
                   and the history used for the threshold restarts after the break.  Nothing is filled in.
  threshold T      for a bar that starts at time t (the start of its first row), inside a segment that began at time a:
                       window = [max(a, t - 30 days), t),   T = (sum of v over the rows that start in the window and end by t) / (length of the window in hours)
                   i.e. the average traded value per clock hour over the last 30 days (shorter at the start of a segment).  A bar can only start when t - a >= 7 days; rows before that only feed the history.
  bar              starts at the first row at or after the previous bar's end and takes rows until their cumulative v reaches T (the row that reaches it is included).  If T is not > 0 no bar starts at that row.
                   A final bar that never reaches T is dropped.
                   start = start of the first row, end = START + DURATION of the last row (when the close happened), open = first row's open, high = max, low = min, close = last row's close,
                   volume = sum, value = sum of v, n_rows.
  One bar therefore holds about one average clock-hour of traded value: busy periods give several bars an hour, quiet ones one bar in several hours.
  The pipeline stamps a bar with its END, so that anything at or before the stamp is known at the stamp.
Also written, per symbol, an hourly reference series (hour start, OHLCV) from the same rows, used by the pipeline for the market state and the Corwin-Schultz spread (both are defined on equal time intervals).
Nothing in the source folders is changed.  The output folder is built symbol by symbol and can be continued: a symbol counts as done when its meta/<SYM>.json exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_S = 30 * 86400
MIN_HIST_S = 7 * 86400
BREAK_S = {"kraken": 48 * 3600, "alpaca": 120 * 3600}
ROW_S = {"kraken": 60, "alpaca": 3600}
ROW_COLS = ["ts", "open", "high", "low", "close", "volume"]
KRAKEN_SYMBOLS = ["ADA", "ALGO", "ANKR", "APT", "ARB", "ATOM", "AVAX", "BAL", "BNB", "BTC", "DOGE", "DOT", "ETH", "FIL", "ICP", "ICX", "KSM", "LINK", "LRC", "NEAR", "OP", "PAXG", "SOL", "TRX", "XLM", "XMR", "XRP"]
DUMP_PAIR = {"BTC": "XBTUSD", "DOGE": "XDGUSD"}
KRAKEN_CUT_S = int(pd.Timestamp("2025-04-01", tz="UTC").timestamp())


def row_value(rows: pd.DataFrame) -> np.ndarray:
    v = (rows["high"].to_numpy(float) + rows["low"].to_numpy(float) + rows["close"].to_numpy(float)) / 3.0 * rows["volume"].to_numpy(float)
    return np.where(np.isfinite(v) & (v > 0), v, 0.0)


def split_segments(start: np.ndarray, dur: int, break_s: int):
    """Index pairs [a, b) of the maximal runs whose neighbours are at most break_s apart (gap = next start - this end)."""
    if len(start) == 0:
        return []
    gap = start[1:] - (start[:-1] + dur)
    cut = np.flatnonzero(gap > break_s) + 1
    edges = np.concatenate([[0], cut, [len(start)]])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def bars_one_segment(rows: pd.DataFrame, dur: int, window_s: int = WINDOW_S, min_hist_s: int = MIN_HIST_S):
    """rows: one segment (sorted, unique ts).  Returns (J, K, thr): bar i takes rows J[i]..K[i] inclusive; thr[i] is its threshold."""
    s = rows["ts"].to_numpy(np.int64)
    e = s + dur
    v = row_value(rows)
    n = len(s)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    a = int(s[0])
    j = int(np.searchsorted(s, a + min_hist_s, side="left"))
    J, K, T = [], [], []
    while j < n:
        t = int(s[j])
        w0 = max(a, t - window_s)
        hi = int(np.searchsorted(e, t, side="right"))             # rows that end by t
        lo = int(np.searchsorted(s, w0, side="left"))             # rows that start in the window
        hours = (t - w0) / 3600.0
        thr = (cs[hi] - cs[lo]) / hours if hours > 0 and hi > lo else 0.0
        if not np.isfinite(thr) or thr <= 0.0:
            j += 1
            continue
        k = int(np.searchsorted(cs, cs[j] + thr, side="left")) - 1      # smallest k >= j with cs[k+1] - cs[j] >= thr
        if k < j:
            k = j
        if k >= n:
            break
        J.append(j); K.append(k); T.append(thr)
        j = k + 1
    return np.array(J, np.int64), np.array(K, np.int64), np.array(T, float)


def dollar_bars(rows: pd.DataFrame, dur: int, break_s: int, window_s: int = WINDOW_S, min_hist_s: int = MIN_HIST_S) -> pd.DataFrame:
    """rows: ts (epoch s of the row start), open, high, low, close, volume, sorted by ts with unique ts.  Returns the bars: start, end (epoch s), open, high, low, close, volume, value, n_rows, threshold."""
    rows = rows.reset_index(drop=True)
    out = []
    for a, b in split_segments(rows["ts"].to_numpy(np.int64), dur, break_s):
        seg = rows.iloc[a:b].reset_index(drop=True)
        J, K, T = bars_one_segment(seg, dur, window_s, min_hist_s)
        if not len(J):
            continue
        o, h, l, c, vol = (seg[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        v = row_value(seg)
        cs = np.concatenate([[0.0], np.cumsum(v)])
        cvol = np.concatenate([[0.0], np.cumsum(np.nan_to_num(vol, nan=0.0))])
        idx = np.empty(2 * len(J), np.int64)
        idx[0::2], idx[1::2] = J, K + 1
        n = len(seg)
        hi = np.append(h, -np.inf)                                # sentinel so that index n is valid for reduceat
        lo = np.append(l, np.inf)
        hmax = np.maximum.reduceat(hi, idx)[0::2]
        lmin = np.minimum.reduceat(lo, idx)[0::2]
        s = seg["ts"].to_numpy(np.int64)
        out.append(pd.DataFrame({"start": s[J], "end": s[K] + dur, "open": o[J], "high": hmax, "low": lmin, "close": c[K], "volume": cvol[K + 1] - cvol[J],
                                 "value": cs[K + 1] - cs[J], "n_rows": K - J + 1, "threshold": T}))
    if not out:
        return pd.DataFrame(columns=["start", "end", "open", "high", "low", "close", "volume", "value", "n_rows", "threshold"])
    return pd.concat(out, ignore_index=True)


def to_hourly(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows of any length up to one hour -> one row per clock hour that has a row: ts = hour start (epoch s), open = first, high = max, low = min, close = last, volume = sum."""
    if not len(rows):
        return pd.DataFrame(columns=ROW_COLS)
    h = (rows["ts"].to_numpy(np.int64) // 3600) * 3600
    g = rows.groupby(h, sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).reset_index()
    out = out.rename(columns={out.columns[0]: "ts"})
    return out[ROW_COLS]


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df[ROW_COLS].copy()
    for c in ROW_COLS[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["close"])
    d = d[d["close"] > 0].drop_duplicates("ts", keep="first").sort_values("ts", kind="stable")
    return d.reset_index(drop=True)


def assemble_kraken(dump: pd.DataFrame, ft: pd.DataFrame, refresh: pd.DataFrame) -> pd.DataFrame:
    """Official 1-minute rows (before 2025-04-01), then trade-derived 1-minute rows from freqtrade's trade file up to where the refreshed trade file starts, then the refreshed trade file."""
    parts = [dump[dump["ts"] < KRAKEN_CUT_S]]
    r0 = int(refresh["ts"].min()) if len(refresh) else None
    f = ft[ft["ts"] >= KRAKEN_CUT_S]
    if r0 is not None:
        f = f[f["ts"] < r0]
    parts += [f, refresh[refresh["ts"] >= KRAKEN_CUT_S]]
    return clean_rows(pd.concat([p[ROW_COLS] for p in parts], ignore_index=True))


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def stamp(ts) -> str:
    return str(pd.Timestamp(int(ts), unit="s"))


def write_symbol(out: Path, sym: str, rows: pd.DataFrame, venue: str, extra: dict) -> dict:
    dur, brk = ROW_S[venue], BREAK_S[venue]
    bars = dollar_bars(rows, dur, brk)
    ref = to_hourly(rows)
    segs = split_segments(rows["ts"].to_numpy(np.int64), dur, brk)
    b = bars.copy()
    b.insert(0, "date", pd.to_datetime(b["end"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S"))
    b["start"] = pd.to_datetime(b["start"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    b = b.drop(columns=["end"])
    r = ref.copy()
    r.insert(0, "date", pd.to_datetime(r["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S"))
    r = r.drop(columns=["ts"])
    (out / "bars").mkdir(parents=True, exist_ok=True)
    (out / "ref_1h").mkdir(parents=True, exist_ok=True)
    pb, pr = out / "bars" / f"{sym}.csv", out / "ref_1h" / f"{sym}.csv"
    b.to_csv(pb, index=False, float_format="%.12g")
    r.to_csv(pr, index=False, float_format="%.12g")
    gaps = [(int(rows["ts"].iloc[segs[i + 1][0]] - (rows["ts"].iloc[segs[i][1] - 1] + dur))) for i in range(len(segs) - 1)]
    per_day = (len(bars) / ((bars["end"].iloc[-1] - bars["start"].iloc[0]) / 86400.0)) if len(bars) > 1 else None
    res = {"rows": int(len(rows)), "first_row": stamp(rows["ts"].iloc[0]) if len(rows) else None, "last_row_end": stamp(rows["ts"].iloc[-1] + dur) if len(rows) else None,
            "segments": int(len(segs)), "breaks": int(len(segs) - 1), "longest_break_hours": (max(gaps) / 3600.0 if gaps else 0.0),
            "bars": int(len(bars)), "first_bar_end": str(b["date"].iloc[0]) if len(b) else None, "last_bar_end": str(b["date"].iloc[-1]) if len(b) else None,
            "bars_per_day_overall": per_day, "median_rows_per_bar": float(bars["n_rows"].median()) if len(bars) else None,
            "share_bars_with_one_row": float((bars["n_rows"] == 1).mean()) if len(bars) else None,
            "bars_md5": md5(pb), "ref_md5": md5(pr), **extra}
    save_meta(out, sym, res)
    return res


def save_meta(out: Path, sym: str, meta: dict):
    """Written last for a symbol: its presence means the symbol's two CSVs are complete (a run that is stopped half-way leaves no meta file for the symbol it was writing)."""
    (out / "meta").mkdir(parents=True, exist_ok=True)
    tmp = out / "meta" / f".{sym}.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    tmp.replace(out / "meta" / f"{sym}.json")


def done(out: Path, sym: str) -> bool:
    return (out / "meta" / f"{sym}.json").exists()


def read_fine(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=ROW_COLS)
    d = pd.read_csv(p, usecols=ROW_COLS)
    return d


def read_dump(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=ROW_COLS)
    return pd.read_csv(p, header=None, names=ROW_COLS + ["trades"], usecols=range(6), dtype={"ts": "int64"})


def run_kraken(dump_dir, fine_dir, out, symbols=None, log=print, deadline=None):
    out = Path(out)
    dump_dir, fine_dir = Path(dump_dir), Path(fine_dir)
    man = {"symbols": {}}
    for sym in (symbols or KRAKEN_SYMBOLS):
        if done(out, sym):
            continue
        if deadline is not None and time.time() > deadline:
            return man, False
        pair = DUMP_PAIR.get(sym, sym + "USD")
        paths = {"dump": dump_dir / f"{pair}_1.csv", "ft": fine_dir / f"{sym}_trades_ft_1m.csv", "refresh": fine_dir / f"{sym}_trades_refresh_1m.csv"}
        d, f, r = read_dump(paths["dump"]), read_fine(paths["ft"]), read_fine(paths["refresh"])
        rows = assemble_kraken(d, f, r)
        src = {"source_rows": {"dump_before_cut": int((d["ts"] < KRAKEN_CUT_S).sum()) if len(d) else 0, "ft_used": int(((f["ts"] >= KRAKEN_CUT_S) & ((f["ts"] < r["ts"].min()) if len(r) else True)).sum()) if len(f) else 0,
                                "refresh_used": int((r["ts"] >= KRAKEN_CUT_S).sum()) if len(r) else 0},
               "source_md5": {k: (md5(p) if p.exists() and p.stat().st_size > 0 else None) for k, p in paths.items()}}
        if not len(rows):
            save_meta(out, sym, {"rows": 0, "bars": 0, **src})
            log(f"{sym}: no rows")
            continue
        m = write_symbol(out, sym, rows, "kraken", src)
        log(f"{sym}: rows={m['rows']:,} segments={m['segments']} bars={m['bars']:,} first_bar_end={m['first_bar_end']} last={m['last_bar_end']} bars/day={m['bars_per_day_overall']}")
    return man, True


def run_alpaca(hourly_dir, out, symbols=None, log=print, deadline=None):
    out = Path(out)
    hourly_dir = Path(hourly_dir)
    src_man_p = hourly_dir / "manifest.json"
    src_man = json.loads(src_man_p.read_text()) if src_man_p.exists() else {}
    man = {"symbols": {}}
    files = sorted(p for p in hourly_dir.glob("*_1h.csv") if not p.name.startswith("._"))
    for p in files:
        sym = p.name[:-len("_1h.csv")]
        if symbols and sym not in symbols:
            continue
        if done(out, sym):
            continue
        if deadline is not None and time.time() > deadline:
            return man, False
        sm = md5(p)
        if p.name in src_man and src_man[p.name] != sm:
            raise SystemExit(f"{p}: md5 differs from the source manifest; stopping")
        if p.stat().st_size < 50:
            save_meta(out, sym, {"rows": 0, "bars": 0, "source_md5": sm})
            continue
        d = pd.read_csv(p, keep_default_na=False, na_values=[""], usecols=["date", "open", "high", "low", "close", "volume"])
        d["ts"] = ((pd.to_datetime(d["date"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).astype("int64")
        rows = clean_rows(d)
        if not len(rows):
            save_meta(out, sym, {"rows": 0, "bars": 0, "source_md5": sm})
            continue
        m = write_symbol(out, sym, rows, "alpaca", {"source_md5": sm})
        log(f"{sym}: rows={m['rows']:,} segments={m['segments']} bars={m['bars']:,} bars/day={m['bars_per_day_overall']}")
    return man, True


def finalize(out: Path, venue: str) -> dict:
    metas = {p.stem: json.loads(p.read_text()) for p in sorted((out / "meta").glob("*.json"))}
    man = {"venue": venue, "window_days": WINDOW_S // 86400, "min_history_days": MIN_HIST_S // 86400, "break_hours": BREAK_S[venue] // 3600, "row_seconds": ROW_S[venue], "symbols": metas}
    (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    (out / "README.txt").write_text(__doc__)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="venue", required=True)
    k = sub.add_parser("kraken")
    k.add_argument("--dump", required=True, help="Kraken OHLCVT folder (<PAIR>_1.csv)")
    k.add_argument("--fine", required=True, help="runtime/kraken_fine_export")
    al = sub.add_parser("alpaca")
    al.add_argument("--hourly", required=True, help="runtime/alpaca_hourly")
    for s_ in (k, al):
        s_.add_argument("--out", required=True, help="output folder; created on the first call, continued on later calls (symbols with a meta file are skipped)")
        s_.add_argument("--symbols", nargs="+", default=None)
        s_.add_argument("--max-seconds", type=float, default=None, help="stop cleanly after this many seconds (exit code 3 = not finished; run the same command again)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.max_seconds if a.max_seconds else None
    lg = lambda m: print(m, flush=True)       # noqa: E731
    _, finished = run_kraken(a.dump, a.fine, out, a.symbols, lg, deadline) if a.venue == "kraken" else run_alpaca(a.hourly, out, a.symbols, lg, deadline)
    if not finished:
        print("NOT FINISHED: run the same command again", flush=True)
        return 3
    finalize(out, a.venue)
    print("finished; manifest.json written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
