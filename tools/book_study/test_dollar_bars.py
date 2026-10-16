import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import build_dollar_bars as B

DAY = 86400


def make_rows(n, seed=0, dur=60, gap_prob=0.3, vol_scale=100.0, t0=1_600_000_000):
    """Irregular rows: each row follows the previous one after a random number of dur-steps (some minutes have no row)."""
    rng = np.random.default_rng(seed)
    steps = 1 + (rng.random(n) < gap_prob) * rng.integers(1, 30, n)
    ts = t0 + np.cumsum(steps) * dur
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.random(n) * 0.001)
    l = np.minimum(o, c) * (1 - rng.random(n) * 0.001)
    vol = rng.exponential(vol_scale, n)
    return pd.DataFrame({"ts": ts.astype(np.int64), "open": o, "high": h, "low": l, "close": c, "volume": vol})


def literal(rows, dur, break_s, window_s, min_hist_s):
    """Plain-loop reference of the definition in the module docstring."""
    ts = rows["ts"].tolist()
    n = len(ts)
    v = []
    for i in range(n):
        x = (rows["high"].iloc[i] + rows["low"].iloc[i] + rows["close"].iloc[i]) / 3.0 * rows["volume"].iloc[i]
        v.append(x if np.isfinite(x) and x > 0 else 0.0)
    seg_start = [0] * n
    for i in range(1, n):
        seg_start[i] = i if ts[i] - (ts[i - 1] + dur) > break_s else seg_start[i - 1]
    seg_end = [n] * n
    for i in range(n - 2, -1, -1):
        seg_end[i] = i + 1 if ts[i + 1] - (ts[i] + dur) > break_s else seg_end[i + 1]
    out = []
    j = 0
    while j < n:
        a_idx, b_idx = seg_start[j], seg_end[j]
        a = ts[a_idx]
        t = ts[j]
        if t - a < min_hist_s:
            j += 1
            continue
        w0 = max(a, t - window_s)
        tot = sum(v[i] for i in range(a_idx, j) if ts[i] >= w0 and ts[i] + dur <= t)
        thr = tot / ((t - w0) / 3600.0)
        if not thr > 0:
            j += 1
            continue
        acc, k = 0.0, j
        while k < b_idx:
            acc += v[k]
            if acc >= thr:
                break
            k += 1
        if k >= b_idx:                                 # the bar would reach the end of the segment without reaching T: dropped, and nothing after it in this segment can start a bar
            j = b_idx
            continue
        seg = rows.iloc[j:k + 1]
        out.append((ts[j], ts[k] + dur, rows["open"].iloc[j], seg["high"].max(), seg["low"].min(), rows["close"].iloc[k], seg["volume"].sum(), sum(v[j:k + 1]), k - j + 1, thr))
        j = k + 1
    return pd.DataFrame(out, columns=["start", "end", "open", "high", "low", "close", "volume", "value", "n_rows", "threshold"])


class T(unittest.TestCase):
    def test_matches_literal_loop(self):
        for seed in range(4):
            rows = make_rows(3000, seed, gap_prob=0.2)
            a = B.dollar_bars(rows, 60, 48 * 3600, window_s=5 * 3600, min_hist_s=2 * 3600)
            b = literal(rows, 60, 48 * 3600, 5 * 3600, 2 * 3600)
            self.assertGreater(len(a), 20)
            pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_matches_literal_with_breaks(self):
        rows = make_rows(2500, 7, gap_prob=0.05)
        # cut two holes of 10 hours (break limit here is 6 hours) at rows 900 and 1700
        rows.loc[900:, "ts"] += 10 * 3600
        rows.loc[1700:, "ts"] += 10 * 3600
        a = B.dollar_bars(rows, 60, 6 * 3600, window_s=5 * 3600, min_hist_s=2 * 3600)
        b = literal(rows, 60, 6 * 3600, 5 * 3600, 2 * 3600)
        self.assertEqual(len(B.split_segments(rows["ts"].to_numpy(), 60, 6 * 3600)), 3)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        # no bar spans a break and none starts within min_hist of a segment start
        seg_starts = [rows["ts"].iloc[x] for x, _ in B.split_segments(rows["ts"].to_numpy(), 60, 6 * 3600)]
        for s0, s1 in zip(seg_starts, seg_starts[1:] + [10 ** 12]):
            inside = a[(a["start"] >= s0) & (a["start"] < s1)]
            self.assertTrue((inside["end"] <= s1).all() or s1 == 10 ** 12)
            if len(inside):
                self.assertGreaterEqual(int(inside["start"].min()) - s0, 2 * 3600)

    def test_hand_case_constant_flow(self):
        # one row per minute, price 10, volume 1 -> v = 10 per row.  Window 2 h, min history 1 h, no breaks.
        n = 600
        rows = pd.DataFrame({"ts": 1_600_000_000 + np.arange(n) * 60, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1.0})
        a = B.dollar_bars(rows, 60, 3600, window_s=2 * 3600, min_hist_s=3600)
        # at t = a0 + 1 h the window holds 60 rows of v=10 over 1 hour -> T = 600 per hour -> a bar is 60 rows = 1 hour
        self.assertTrue((a["n_rows"] == 60).all())
        self.assertTrue(np.allclose(a["threshold"], 600.0))
        self.assertEqual(int(a["start"].iloc[0]), 1_600_000_000 + 3600)
        self.assertEqual(int(a["end"].iloc[0]), 1_600_000_000 + 3600 + 3600)               # end = start of the last row + its 60 s
        self.assertTrue((a["start"].to_numpy()[1:] == a["end"].to_numpy()[:-1]).all())

    def test_quiet_time_makes_longer_bars_busy_time_shorter(self):
        n = 3000
        rows = pd.DataFrame({"ts": 1_600_000_000 + np.arange(n) * 60, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1.0})
        rows.loc[1500:2000, "volume"] = 10.0                                                  # busy
        a = B.dollar_bars(rows, 60, 3600, window_s=24 * 3600, min_hist_s=3600)
        dur = (a["end"] - a["start"]).to_numpy()
        busy = a[(a["start"] >= rows["ts"].iloc[1600]) & (a["end"] <= rows["ts"].iloc[1900])]
        self.assertGreater(len(busy), 3)
        self.assertLess(busy["n_rows"].mean(), a[a["end"] < rows["ts"].iloc[1400]]["n_rows"].mean())

    def test_causal(self):
        rows = make_rows(2500, 2)
        a = B.dollar_bars(rows, 60, 48 * 3600, window_s=5 * 3600, min_hist_s=2 * 3600)
        cut = 1500
        r2 = rows.copy()
        r2.loc[cut:, ["open", "high", "low", "close", "volume"]] *= 3.0
        b = B.dollar_bars(r2, 60, 48 * 3600, window_s=5 * 3600, min_hist_s=2 * 3600)
        t_cut = rows["ts"].iloc[cut]
        ok = a[a["end"] <= t_cut].reset_index(drop=True)
        self.assertGreater(len(ok), 10)
        pd.testing.assert_frame_equal(ok, b.iloc[:len(ok)].reset_index(drop=True), check_dtype=False)

    def test_no_volume_no_bars_and_bad_values_count_as_zero(self):
        rows = make_rows(800, 3)
        z = rows.copy(); z["volume"] = 0.0
        self.assertEqual(len(B.dollar_bars(z, 60, 48 * 3600, 5 * 3600, 2 * 3600)), 0)
        r = rows.copy()
        r.loc[100:110, "volume"] = np.nan
        r.loc[120, "volume"] = -5.0
        a = B.dollar_bars(r, 60, 48 * 3600, 5 * 3600, 2 * 3600)
        b = literal(r, 60, 48 * 3600, 5 * 3600, 2 * 3600)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        self.assertTrue(np.isfinite(a[["open", "high", "low", "close", "volume", "value"]].to_numpy()).all())

    def test_ohlc_consistency_and_volume_conservation(self):
        rows = make_rows(3000, 5)
        a = B.dollar_bars(rows, 60, 48 * 3600, 5 * 3600, 2 * 3600)
        self.assertTrue((a["high"] >= a[["open", "close"]].max(axis=1) - 1e-9).all())
        self.assertTrue((a["low"] <= a[["open", "close"]].min(axis=1) + 1e-9).all())
        self.assertTrue((a["value"] >= a["threshold"] - 1e-9).all())
        # bars do not overlap in time
        self.assertTrue((a["start"].to_numpy()[1:] >= a["end"].to_numpy()[:-1]).all())
        # volume of bars equals volume of the rows they cover
        tot = 0.0
        for _, r in a.iterrows():
            m = (rows["ts"] >= r["start"]) & (rows["ts"] + 60 <= r["end"])
            tot += rows.loc[m, "volume"].sum()
        self.assertAlmostEqual(tot, a["volume"].sum(), places=6)

    def test_to_hourly_by_hand(self):
        t = 3600 * 1000
        rows = pd.DataFrame({"ts": [t, t + 60, t + 3599 - 59, t + 3600], "open": [1.0, 2, 3, 9], "high": [5.0, 4, 8, 9], "low": [0.5, 2, 1, 9], "close": [2.0, 3, 4, 9], "volume": [1.0, 1, 2, 5]})
        h = B.to_hourly(rows)
        self.assertEqual(list(h["ts"]), [t, t + 3600])
        r = h.iloc[0]
        self.assertEqual((r.open, r.high, r.low, r.close, r.volume), (1.0, 8.0, 0.5, 4.0, 4.0))

    def test_assemble_kraken_junctions(self):
        cut = B.KRAKEN_CUT_S
        mk = lambda ts, c: pd.DataFrame({"ts": ts, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
        dump = mk([cut - 120, cut - 60, cut + 60], [1.0, 2.0, 99.0])          # the row at/after the cut is not taken from the dump
        ft = mk([cut - 60, cut, cut + 60, cut + 120], [98.0, 3.0, 4.0, 5.0])  # the row before the cut is not taken from the trade file
        refresh = mk([cut + 120, cut + 180], [6.0, 7.0])                       # refresh takes over from its first row; ft rows at/after it are dropped
        r = B.assemble_kraken(dump, ft, refresh)
        self.assertEqual(list(r["ts"]), [cut - 120, cut - 60, cut, cut + 60, cut + 120, cut + 180])
        self.assertEqual(list(r["close"]), [1.0, 2.0, 3.0, 4.0, 6.0, 7.0])
        r2 = B.assemble_kraken(dump, ft, pd.DataFrame(columns=B.ROW_COLS))     # no refresh file: all ft rows are used
        self.assertEqual(list(r2["close"]), [1.0, 2.0, 3.0, 4.0, 5.0])
        r3 = B.assemble_kraken(pd.DataFrame(columns=B.ROW_COLS), ft, pd.DataFrame(columns=B.ROW_COLS))    # no dump (BNB)
        self.assertEqual(list(r3["close"]), [3.0, 4.0, 5.0])

    def test_end_to_end_alpaca_and_kraken(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # alpaca: 60 days of hourly rows, 14 per day (a trading day), weekends missing
            hrs = []
            day0 = pd.Timestamp("2021-01-04")
            for d in range(60):
                day = day0 + pd.Timedelta(days=d)
                if day.dayofweek >= 5:
                    continue
                for h in range(14, 20):
                    hrs.append(day + pd.Timedelta(hours=h))
            n = len(hrs)
            df = pd.DataFrame({"date": [x.strftime("%Y-%m-%d %H:%M:%S") for x in hrs], "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 1000.0, "trade_count": 5, "vwap": 10.0})
            src = tmp / "ah"
            src.mkdir()
            df.to_csv(src / "AAA_1h.csv", index=False)
            (src / "BBB_1h.csv").write_text("date,open,high,low,close,volume,trade_count,vwap\n")
            (src / "._AAA_1h.csv").write_bytes(b"\x00\xb0\xff")
            (src / "manifest.json").write_text(json.dumps({"AAA_1h.csv": B.md5(src / "AAA_1h.csv"), "BBB_1h.csv": B.md5(src / "BBB_1h.csv")}))
            out = tmp / "out"
            out.mkdir()
            _, fin = B.run_alpaca(src, out, log=lambda *_: None)
            self.assertTrue(fin)
            man = B.finalize(out, "alpaca")
            self.assertEqual(man["symbols"]["BBB"]["bars"], 0)
            self.assertGreater(man["symbols"]["AAA"]["bars"], 10)
            self.assertEqual(man["symbols"]["AAA"]["breaks"], 0)                  # weekends (<= 3 days) are not breaks for equities
            b = pd.read_csv(out / "bars" / "AAA.csv")
            self.assertTrue(set(["date", "start", "open", "high", "low", "close", "volume", "value", "n_rows", "threshold"]) <= set(b.columns))
            self.assertTrue((pd.to_datetime(b["date"]) > pd.to_datetime(b["start"])).all())
            self.assertEqual(len(pd.read_csv(out / "ref_1h" / "AAA.csv")), n)
            # a changed source file is refused
            (src / "AAA_1h.csv").write_text(df.head(10).to_csv(index=False))
            with self.assertRaises(SystemExit):
                B.run_alpaca(src, tmp / "out2", log=lambda *_: None)

    def test_resume_and_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "ah"
            src.mkdir()
            hrs = [pd.Timestamp("2021-01-04") + pd.Timedelta(hours=h) for h in range(24 * 40)]
            df = pd.DataFrame({"date": [x.strftime("%Y-%m-%d %H:%M:%S") for x in hrs], "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 1000.0})
            for s_ in ("AAA", "BBB", "CCC"):
                df.to_csv(src / f"{s_}_1h.csv", index=False)
            out = tmp / "o"
            out.mkdir()
            _, fin = B.run_alpaca(src, out, log=lambda *_: None, deadline=0.0)          # deadline already passed: nothing is processed
            self.assertFalse(fin)
            self.assertEqual(len(list((out / "meta").glob("*.json"))) if (out / "meta").exists() else 0, 0)
            _, fin = B.run_alpaca(src, out, symbols=["AAA"], log=lambda *_: None)
            self.assertTrue(fin)
            first = (out / "meta" / "AAA.json").read_text()
            logged = []
            _, fin = B.run_alpaca(src, out, log=logged.append)                           # AAA is skipped, BBB and CCC are done
            self.assertTrue(fin)
            self.assertEqual(sorted(x.split(":")[0] for x in logged), ["BBB", "CCC"])
            self.assertEqual((out / "meta" / "AAA.json").read_text(), first)
            man = B.finalize(out, "alpaca")
            self.assertEqual(sorted(man["symbols"]), ["AAA", "BBB", "CCC"])


if __name__ == "__main__":
    unittest.main()
