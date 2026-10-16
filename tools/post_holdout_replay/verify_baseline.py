#!/usr/bin/env python
"""Independent checker for a plain-rule baseline output folder (stdlib only).

Recomputes what it can with its own code (it imports neither baseline_rules nor replay_stats):
trade bookkeeping, pooled statistics, window bounds, and - the part that matters most - it re-derives the
moving average for a sample of trades straight from the raw candle files and checks that each trade
entered on the correct side of the average at the signal candle's close, at the expected price.
Exit code 0 = all pass.

    python tools/post_holdout_replay/verify_baseline.py runtime/post_holdout_baseline/<timestamp>
"""
from __future__ import annotations

import bisect
import calendar
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def pf(pnls: list[float]) -> float:
    gain = sum(p for p in pnls if p > 0)
    loss = -sum(p for p in pnls if p < 0)
    if loss > 0:
        return gain / loss
    return math.inf if gain > 0 else math.nan


def same(a: float, b: float, tol: float = 1e-6) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def parse_ts(s) -> float:
    s = str(s).strip()
    try:
        v = float(s)
        return v / 1000.0 if abs(v) >= 1e11 else v
    except ValueError:
        pass
    s = s.replace("T", " ")[:19]
    return float(calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M:%S")))   # UTC, independent of the local timezone


def ts_key(s) -> float:
    return parse_ts(s)


def load_raw_close(gmx_dir: str, tv_dir: str, symbol: str, timeframe: str = "1h"):
    """(sorted epoch list, closes list): GMX candles plus only TradingView candles older than the first GMX candle."""
    gmx, tv = {}, {}
    gpath = Path(gmx_dir) / f"gmx_arbitrum_{symbol}_{timeframe}.csv"
    if gpath.exists():
        for r in read(gpath):
            c = num(r.get("close", r.get("Close")))
            t = r.get("open_time", r.get("Date"))
            if t is not None and not math.isnan(c):
                gmx[ts_key(t)] = c
    tpath = Path(tv_dir) / f"tradingview_{symbol}_{timeframe}.csv"
    if tpath.exists():
        for r in read(tpath):
            c = num(r.get("Close", r.get("close")))
            t = r.get("Date", r.get("time"))
            if t is not None and not math.isnan(c):
                tv[ts_key(t)] = c
    first_gmx = min(gmx) if gmx else math.inf
    merged = {t: c for t, c in tv.items() if t < first_gmx}
    merged.update(gmx)
    keys = sorted(merged)
    return keys, [merged[k] for k in keys]


def average_at(keys, closes, cum, t: float, rule: str, min_bars_d: int, hours: int):
    """Simple average of closes ending at (and including) time t, or None if the rule's minimum is not met."""
    hi = bisect.bisect_right(keys, t)
    if hi == 0 or keys[hi - 1] != t:
        return None
    if rule == "ma200d":
        lo = bisect.bisect_right(keys, t - 200 * 86400)   # window (t - 200D, t]
        n = hi - lo
        return (cum[hi] - cum[lo]) / n if n >= min_bars_d else None
    if rule == "ma200h":
        if hi < hours:
            return None
        return (cum[hi] - cum[hi - hours]) / hours
    return None


def main(folder: str) -> int:
    d = Path(folder)
    checks: list[tuple[str, bool, str]] = []
    sym = read(d / "baseline_per_symbol.csv")
    trades = read(d / "baseline_trades.csv") if nonempty(d / "baseline_trades.csv") else []
    blob = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    meta, windows, numbers = blob["meta"], blob["windows"], blob["numbers"]
    integrity = json.loads((d / "integrity.json").read_text(encoding="utf-8"))
    june0, june1 = ts_key(meta["june_2026"][0]), ts_key(meta["june_2026"][1])

    key = lambda r: (r["segment"], r["rule"], r["direction"].lower(), r["symbol"])
    # 1. trade counts per scored row
    counts = defaultdict(int)
    for t in trades:
        counts[key(t)] += 1
    bad = [(key(r), r["trades"], counts[key(r)]) for r in sym if int(float(r["trades"])) != counts[key(r)]]
    scored = {key(r) for r in sym}
    orphans = sorted({key(t) for t in trades if key(t) not in scored})
    checks.append(("per-row trade counts equal rows in baseline_trades.csv", not bad, str(bad[:4])))
    checks.append(("every trade belongs to a scored row", not orphans, str(orphans[:4])))

    # 2. window bounds and ordering
    hend = {(r["direction"], r["symbol"]): ts_key(r["holdout_end"]) for r in sym}
    late = [(key(t), t["entry_time"]) for t in trades if t["segment"] == "post_holdout"
            and (t["direction"].lower(), t["symbol"]) in hend
            and ts_key(t["entry_time"]) <= hend[(t["direction"].lower(), t["symbol"])]]
    checks.append(("post_holdout trades all enter strictly after that symbol's holdout end", not late, str(late[:3])))
    outside = [(key(t), t["entry_time"]) for t in trades if t["segment"] == "june_2026"
               and not (june0 <= ts_key(t["entry_time"]) < june1)]
    checks.append(("june_2026 trades all enter inside 2026-06-01..2026-06-30", not outside, str(outside[:3])))
    order = [(key(t), t["entry_time"]) for t in trades if ts_key(t["exit_time"]) < ts_key(t["entry_time"])]
    checks.append(("no exit before its entry", not order, str(order[:3])))

    # 3. pooled statistics recomputed
    for segment, rules in numbers.items():
        for rule, dr in rules.items():
            for direction, groups in dr.items():
                pnls = [num(t["net_pnl"]) for t in trades
                        if t["segment"] == segment and t["rule"] == rule and t["direction"].lower() == direction]
                s = groups["all"]
                ok = len(pnls) == s["trades"] and same(pf(pnls), num(str(s["profit_factor"]))) and same(sum(pnls), float(s["net_pnl"]))
                checks.append((f"{segment}/{rule}/{direction}: pooled trades / PF / net P&L recomputed match", ok,
                               f"recomputed {len(pnls)} {pf(pnls)} {sum(pnls):.2f} vs {s['trades']} {s['profit_factor']} {s['net_pnl']:.2f}"))
    # window groups recomputed
    for segment, w in windows.items():
        w0, w1 = ts_key(w["start"]), ts_key(w["end"])
        gname = f"entered_during_{w['name']}"
        for rule, dr in numbers.get(segment, {}).items():
            for direction, groups in dr.items():
                if gname not in groups:
                    continue
                pnls = [num(t["net_pnl"]) for t in trades if t["segment"] == segment and t["rule"] == rule
                        and t["direction"].lower() == direction and w0 <= ts_key(t["entry_time"]) <= w1]
                s = groups[gname]
                ok = len(pnls) == s["trades"] and same(pf(pnls), num(str(s["profit_factor"]))) and same(sum(pnls), float(s["net_pnl"]))
                checks.append((f"{segment}/{rule}/{direction}: {gname} recomputed match", ok,
                               f"{len(pnls)} {pf(pnls)} vs {s['trades']} {s['profit_factor']}"))

    # 4. signal-count consistency
    sig_bad = []
    by = defaultdict(dict)
    for r in sym:
        by[(r["segment"], r["rule"], r["symbol"])][r["direction"]] = r
        if r["rule"] == "always" and int(float(r["signal_bars"])) != int(float(r["candles"])):
            sig_bad.append(("always", key(r)))
        if int(float(r["signal_bars"])) > int(float(r["valid_average_bars"])):
            sig_bad.append(("signals>valid", key(r)))
    for (seg, rule, sym_name), pair in by.items():
        if rule != "always" and "long" in pair and "short" in pair:
            both = int(float(pair["long"]["signal_bars"])) + int(float(pair["short"]["signal_bars"]))
            valid = int(float(pair["long"]["valid_average_bars"]))
            if both > valid or valid - both > max(2, 0.01 * valid):
                sig_bad.append(("long+short vs valid", seg, rule, sym_name, both, valid))
    checks.append(("signal counts consistent (always = every candle; long+short = candles with an average)", not sig_bad, str(sig_bad[:3])))

    # 5. spot-check the rule against the raw candle files
    min_bars_d, hours = int(meta["ma_days_min_bars"]), int(meta["ma_hours"])
    slip = float(meta["slippage_plus_impact_bps"]) / 10000.0
    sample = defaultdict(list)
    for t in trades:
        sample[(t["rule"], t["direction"].lower())].append(t)
    rule_bad, price_bad, checked, unreadable = [], [], 0, []
    cache = {}
    for (rule, direction), ts_list in sorted(sample.items()):
        step = max(1, len(ts_list) // 150)
        for t in ts_list[::step][:150]:
            symbol = t["symbol"]
            if symbol not in cache:
                keys, closes = load_raw_close(meta["gmx_dir"], meta["tradingview_dir"], symbol, meta["timeframe"])
                cum = [0.0]
                for c in closes:
                    cum.append(cum[-1] + c)
                cache[symbol] = (keys, closes, cum)
            keys, closes, cum = cache[symbol]
            if not keys:
                unreadable.append(symbol)
                continue
            te = ts_key(t["entry_time"])
            i = bisect.bisect_left(keys, te)
            if i >= len(keys) or keys[i] != te:
                unreadable.append((symbol, t["entry_time"]))
                continue
            close = closes[i]
            expected = close * (1 + slip) if direction == "long" else close * (1 - slip)
            if abs(num(t["entry_price"]) / expected - 1.0) > 1e-6:
                price_bad.append((rule, direction, symbol, t["entry_time"], t["entry_price"], expected))
            checked += 1
            if rule == "always":
                continue
            avg = average_at(keys, closes, cum, te, rule, min_bars_d, hours)
            if avg is None:
                rule_bad.append((rule, direction, symbol, t["entry_time"], "average not computable"))
                continue
            rel = close / avg - 1.0
            wrong_side = (rel < -0.005) if direction == "long" else (rel > 0.005)
            if wrong_side:
                rule_bad.append((rule, direction, symbol, t["entry_time"], f"close/avg-1={rel:.4f}"))
    checks.append((f"rule check against raw candles: {checked} sampled trades entered on the correct side of the average",
                   checked > 0 and not rule_bad, str(rule_bad[:3]) if rule_bad else "no trades sampled"))
    checks.append(("entry price = signal candle close +/- slippage/impact (sampled trades)", checked > 0 and not price_bad, str(price_bad[:3])))
    checks.append(("every sampled trade's candle found in the raw files", not unreadable, str(unreadable[:3])))

    # 6. comparison with the model replay
    rdir = meta.get("replay_dir")
    if rdir and (Path(rdir) / "replay_per_symbol.csv").exists():
        rep = read(Path(rdir) / "replay_per_symbol.csv")
        rep_idx = {(r["direction"], r["symbol"]): r for r in rep}
        ph = {}
        for r in sym:
            if r["segment"] == "post_holdout":
                ph.setdefault((r["direction"], r["symbol"]), r)
        missing = sorted(set(rep_idx) ^ set(ph))
        checks.append(("same symbol-directions as the model replay", not missing, f"{len(missing)} differ: {missing[:4]}"))
        diff = [(k, ph[k]["window_start"], rep_idx[k]["replay_start"], ph[k]["window_end"], rep_idx[k]["replay_end"], ph[k]["candles"], rep_idx[k]["replay_candles"])
                for k in ph if k in rep_idx and (ph[k]["window_start"] != rep_idx[k]["replay_start"] or ph[k]["window_end"] != rep_idx[k]["replay_end"]
                                                 or int(float(ph[k]["candles"])) != int(float(rep_idx[k]["replay_candles"])))]
        checks.append(("post-holdout window (start, end, candles) identical to the model replay for every symbol-direction", not diff, str(diff[:2])))
        # the baseline's equal-weight index must be the same series the model replay used
        idx_b, idx_r = d / "equal_weight_index_daily.csv", Path(rdir) / "equal_weight_index_daily.csv"
        if "post_holdout" in windows and idx_b.exists() and idx_r.exists():
            def series(p):
                out = {}
                for r in read(p):
                    k = r["Date"] if "Date" in r else list(r.values())[0]
                    v = num(r["equal_weight_index"])
                    if not math.isnan(v):
                        out[str(k)[:10]] = v
                return out
            sb, sr = series(idx_b), series(idx_r)
            dates = sorted(set(sb) & set(sr))
            rel = {k: abs(sb[k] / sr[k] - 1.0) for k in dates if sr[k]}
            last = dates[-1] if dates else None
            worst_before = max((v for k, v in rel.items() if k != last), default=math.inf)
            worst_last = rel.get(last, math.inf)
            # The data files are refreshed hourly and the newest candle can be revised between two runs, so the
            # final date gets a small tolerance; every earlier date must be identical.
            checks.append(("equal-weight index (sell-off window source) identical to the model replay's (final date within 1e-4)",
                           set(sb) == set(sr) and worst_before < 1e-9 and worst_last < 1e-4,
                           f"dates only in one: {len(set(sb) ^ set(sr))}; max relative difference before last date {worst_before:.3g}, last date {worst_last:.3g}"))
        comp_path = d / "comparison_with_model.csv"
        if nonempty(comp_path):
            comp = read(comp_path)
            mism = [(c["direction"], c["symbol"]) for c in comp
                    if (c["direction"], c["symbol"]) in rep_idx
                    and not same(num(c["model_pf"]), num(rep_idx[(c["direction"], c["symbol"])]["profit_factor"]))]
            checks.append(("comparison_with_model.csv carries the replay's model PF unchanged", not mism, str(mism[:3])))
    else:
        checks.append(("model replay folder available for comparison (informational)", True, "no replay folder recorded; comparison skipped"))

    # 7. run health, production untouched
    skipped = read(d / "skipped.csv") if nonempty(d / "skipped.csv") else []
    errs = [(r["direction"], r["symbol"], r["reason"][:60]) for r in skipped if r["reason"].startswith("error")]
    checks.append(("at least one row was scored", len(sym) > 0, f"{len(sym)} rows"))
    checks.append(("no symbol ended in an internal error", not errs, str(errs[:3])))
    checks.append(("models/ calibration + file counts unchanged during the run", bool(integrity["unchanged"]), json.dumps(integrity["after"])[:200]))

    width = max(len(c[0]) for c in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {'' if ok else detail}")
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
