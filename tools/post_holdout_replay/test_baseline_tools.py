"""Tests for the plain-rule baseline helpers and checker. Run: python -m unittest tools/post_holdout_replay/test_baseline_tools.py -v"""
from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baseline_rules as B
import verify_baseline as V

SLIP = 0.0007


def hourly(n=6000, start="2026-01-01", seed=1, drift=0.0003):
    idx = pd.date_range(start, periods=n, freq="h")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.003, n)))
    return pd.Series(close, index=idx)


class Rules(unittest.TestCase):
    def test_ma200d_needs_80_percent_of_window(self):
        c = hourly(5000)
        avg = B.average_for_rule(c, "ma200d")
        first_valid = avg.first_valid_index()
        self.assertEqual(c.index.get_loc(first_valid), B.MA_DAYS_MIN_BARS - 1)

    def test_ma200d_is_time_based_across_gaps(self):
        c = hourly(6000)
        gapped = c.drop(c.index[1000:1500])           # 500 missing hours
        avg = B.average_for_rule(gapped, "ma200d")
        t = gapped.index[-1]
        window = gapped[(gapped.index > t - pd.Timedelta(days=200)) & (gapped.index <= t)]
        self.assertAlmostEqual(avg.iloc[-1], window.mean(), places=9)
        self.assertGreaterEqual(len(window), B.MA_DAYS_MIN_BARS)

    def test_ma200h_matches_plain_rolling(self):
        c = hourly(500)
        avg = B.average_for_rule(c, "ma200h")
        self.assertTrue(np.isnan(avg.iloc[198]))
        self.assertAlmostEqual(avg.iloc[199], c.iloc[:200].mean(), places=9)

    def test_no_lookahead(self):
        c = hourly(5200)
        k = 4900
        base = B.average_for_rule(c, "ma200d").iloc[k]
        c2 = c.copy()
        c2.iloc[k + 1:] *= 5.0                        # change only the FUTURE
        self.assertEqual(base, B.average_for_rule(c2, "ma200d").iloc[k])

    def test_signal_sides_and_equality(self):
        close = np.array([10.0, 11.0, 9.0, 10.0, 12.0])
        avg = np.array([10.0, 10.0, 10.0, np.nan, 10.0])
        self.assertEqual(B.signal_array(close, avg, "long", "ma200d").tolist(), [False, True, False, False, True])
        self.assertEqual(B.signal_array(close, avg, "short", "ma200d").tolist(), [False, False, True, False, False])
        self.assertTrue(B.signal_array(close, avg, "long", "always").all())
        self.assertTrue(B.signal_array(close, avg, "short", "always").all())

    def test_long_and_short_never_overlap(self):
        c = hourly(5200)
        avg = B.average_for_rule(c, "ma200d").to_numpy()
        lo = B.signal_array(c.to_numpy(), avg, "long", "ma200d")
        sh = B.signal_array(c.to_numpy(), avg, "short", "ma200d")
        self.assertFalse((lo & sh).any())
        self.assertEqual(int(lo.sum() + sh.sum()), B.valid_average_bars(avg, "ma200d"))

    def test_window_slice(self):
        ts = pd.date_range("2026-05-30", periods=24 * 5, freq="h")
        lo, hi = B.window_slice(ts, *B.JUNE_2026, min_len=2)
        self.assertEqual(pd.Timestamp(ts[lo]), pd.Timestamp("2026-06-01"))
        self.assertTrue(pd.Timestamp(ts[hi - 1]) < pd.Timestamp("2026-07-01"))
        self.assertIsNone(B.window_slice(ts, *B.JUNE_2026, min_len=500))
        self.assertIsNone(B.window_slice(ts, pd.Timestamp("2027-01-01"), None))

    def test_entries_and_exposure(self):
        tr = pd.DataFrame({"entry_time": ["2026-06-01 00:00:00", "2026-06-30 23:00:00", "2026-07-01 00:00:00"],
                           "exit_time": ["2026-06-01 10:00:00", "2026-07-01 03:00:00", "2026-07-01 05:00:00"]})
        self.assertEqual(len(B.entries_in_half_open(tr, *B.JUNE_2026)), 2)
        self.assertEqual(len(B.entries_in(tr, pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02"))), 1)
        self.assertAlmostEqual(B.bars_in_market_pct(tr.iloc[:1], 100), 10.0)
        self.assertEqual(B.bars_in_market_pct(tr.iloc[:0], 100), 0.0)


# ------------------------------------------------------------------ checker fixtures
def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pf_of(p):
    g, l = sum(x for x in p if x > 0), -sum(x for x in p if x < 0)
    return g / l if l else (float("inf") if g else float("nan"))


def build_fixture(root: Path):
    """One symbol, 6000 rising hourly candles (GMX file holds the later 4000, a TradingView file the earlier 2000)."""
    gmx_dir, tv_dir, out = root / "gmx", root / "tv", root / "out"
    for p in (gmx_dir, tv_dir, out):
        p.mkdir()
    c = hourly(6000)
    gm = c.iloc[2000:]
    write_csv(gmx_dir / "gmx_arbitrum_AAA_1h.csv",
              [{"open_time": str(t), "open": v, "high": v, "low": v, "close": v, "volume": 0} for t, v in gm.items()],
              ["open_time", "open", "high", "low", "close", "volume"])
    tv = c.iloc[:2000]
    write_csv(tv_dir / "tradingview_AAA_1h.csv",
              [{"Date": str(t), "Open": v, "High": v, "Low": v, "Close": v, "Volume": 0} for t, v in tv.items()],
              ["Date", "Open", "High", "Low", "Close", "Volume"])
    avg = B.average_for_rule(c, "ma200d")
    above = [t for t in c.index if t > pd.Timestamp("2026-07-01") and not np.isnan(avg[t]) and c[t] > avg[t]]
    entries = above[100:104]
    hend = pd.Timestamp("2026-07-05")
    trades = []
    for i, t in enumerate(entries):
        pnl = [30.0, -10.0, 25.0, -5.0][i]
        trades.append({"segment": "post_holdout", "rule": "ma200d", "symbol": "AAA", "direction": "LONG", "entry_time": str(t),
                       "exit_time": str(t + pd.Timedelta(hours=5)), "entry_price": c[t] * (1 + SLIP), "net_pnl": pnl, "return_pct": pnl / 1000,
                       "methodology": "current"})
    tfields = list(trades[0])
    write_csv(out / "baseline_trades.csv", trades, tfields)
    pnls = [t["net_pnl"] for t in trades]
    row = {"segment": "post_holdout", "rule": "ma200d", "direction": "long", "symbol": "AAA", "methodology": "current",
           "holdout_end": str(hend), "window_start": str(hend + pd.Timedelta(hours=1)), "window_end": str(c.index[-2]),
           "candles": 1000, "valid_average_bars": 900, "signal_bars": 500, "trades": 4, "win_rate_pct": 50.0, "return_pct": 1.0,
           "profit_factor": pf_of(pnls), "max_drawdown_pct": -1.0, "price_return_pct": 2.0, "bars_in_market_pct": 5.0}
    rowfields = list(row)
    write_csv(out / "baseline_per_symbol.csv", [row], rowfields)
    (out / "skipped.csv").write_text("direction,symbol,reason\n")
    (out / "integrity.json").write_text(json.dumps({"before": {}, "after": {}, "unchanged": True}))
    allg = {"trades": 4, "wins": 2, "win_rate_pct": 50.0, "profit_factor": pf_of(pnls), "net_pnl": sum(pnls), "mean_return_pct": 0.0}
    blob = {"meta": {"gmx_dir": str(gmx_dir), "tradingview_dir": str(tv_dir), "timeframe": "1h", "slippage_plus_impact_bps": 7.0,
                     "ma_days_min_bars": B.MA_DAYS_MIN_BARS, "ma_hours": 200, "june_2026": [str(B.JUNE_2026[0]), str(B.JUNE_2026[1])],
                     "replay_dir": None},
            "windows": {}, "numbers": {"post_holdout": {"ma200d": {"long": {"all": allg}}}}, "model_reference": {}}
    (out / "summary_numbers.json").write_text(json.dumps(blob))
    return out, tfields, rowfields, trades, row


def run_checker(out: Path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = V.main(str(out))
    return rc, buf.getvalue()


class Checker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out, self.tfields, self.rowfields, self.trades, self.row = build_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def rewrite_trades(self, mutate):
        trades = [dict(t) for t in self.trades]
        mutate(trades)
        write_csv(self.out / "baseline_trades.csv", trades, self.tfields)

    def test_clean_fixture_passes(self):
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 0, text)
        self.assertIn("correct side of the average", text)

    def test_wrong_side_trade_fails(self):
        # crash the raw price for 20 candles so a LONG entered there is clearly below its average
        path = self.root / "gmx" / "gmx_arbitrum_AAA_1h.csv"
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        c = hourly(6000)
        t_bad = c.index[5010]
        for r in rows:
            if c.index[5000] <= pd.Timestamp(r["open_time"]) < c.index[5020]:
                r["close"] = float(r["close"]) * 0.3
        write_csv(path, rows, ["open_time", "open", "high", "low", "close", "volume"])
        crashed = c[t_bad] * 0.3
        def m(tr):
            tr[0]["entry_time"], tr[0]["exit_time"] = str(t_bad), str(t_bad + pd.Timedelta(hours=5))
            tr[0]["entry_price"] = crashed * (1 + SLIP)
        self.rewrite_trades(m)
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  rule check", text)

    def test_wrong_entry_price_fails(self):
        self.rewrite_trades(lambda tr: tr[1].update(entry_price=float(tr[1]["entry_price"]) * 1.01))
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  entry price", text)

    def test_trade_inside_holdout_fails(self):
        c = hourly(6000)
        t = pd.Timestamp("2026-07-03 00:00:00")
        def m(tr):
            tr[0]["entry_time"], tr[0]["exit_time"] = str(t), str(t + pd.Timedelta(hours=5))
            tr[0]["entry_price"] = c[t] * (1 + SLIP)
        self.rewrite_trades(m)
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  post_holdout trades all enter strictly after", text)

    def test_summary_mismatch_fails(self):
        p = self.out / "summary_numbers.json"
        blob = json.loads(p.read_text())
        blob["numbers"]["post_holdout"]["ma200d"]["long"]["all"]["net_pnl"] += 50
        p.write_text(json.dumps(blob))
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  post_holdout/ma200d/long: pooled", text)

    def test_exit_before_entry_fails(self):
        self.rewrite_trades(lambda tr: tr[2].update(exit_time=str(pd.Timestamp(tr[2]["entry_time"]) - pd.Timedelta(hours=1))))
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  no exit before its entry", text)

    def test_orphan_trade_fails_without_crashing(self):
        self.rewrite_trades(lambda tr: tr[0].update(symbol="ZZZ"))
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  every trade belongs to a scored row", text)

    def test_internal_error_fails(self):
        (self.out / "skipped.csv").write_text("direction,symbol,reason\nlong,AAA,error: boom\n")
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  no symbol ended in an internal error", text)

    def test_always_rule_needs_every_candle_signalled(self):
        rows = [dict(self.row, rule="always", signal_bars=10)]
        write_csv(self.out / "baseline_per_symbol.csv", [self.row] + rows, self.rowfields)
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  signal counts consistent", text)

    def test_changed_models_fails(self):
        (self.out / "integrity.json").write_text(json.dumps({"before": {"a": 1}, "after": {"a": 2}, "unchanged": False}))
        rc, text = run_checker(self.out)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  models/", text)


if __name__ == "__main__":
    unittest.main()
