import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import export_kraken_post_dump as X


def ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


class T(unittest.TestCase):
    def test_trades_to_minutes_by_hand(self):
        tr = pd.DataFrame({"timestamp": [ms("2025-04-01 00:00:05"), ms("2025-04-01 00:00:59"), ms("2025-04-01 00:01:00"), ms("2025-04-01 00:03:30"), ms("2025-04-01 00:00:05")],
                           "id": ["a", "b", "c", "d", "a"], "price": [10.0, 12.0, 11.0, 9.0, 10.0], "amount": [1.0, 2.0, 3.0, 4.0, 1.0]})
        m = X.trades_to_minutes(tr)
        self.assertEqual(list(m["ts"]), [ms("2025-04-01 00:00:00") // 1000, ms("2025-04-01 00:01:00") // 1000, ms("2025-04-01 00:03:00") // 1000])   # no row for the empty minutes
        r0 = m.iloc[0]
        self.assertEqual((r0.open, r0.high, r0.low, r0.close, r0.volume, r0["count"]), (10.0, 12.0, 10.0, 12.0, 3.0, 2))   # the duplicate id "a" is dropped
        self.assertEqual(m.iloc[2].volume, 4.0)

    def test_order_independent_and_bad_rows_removed(self):
        rng = np.random.default_rng(0)
        n = 500
        ts = np.sort(rng.integers(ms("2025-04-01"), ms("2025-04-01") + 3 * 3600 * 1000, n))
        tr = pd.DataFrame({"timestamp": ts, "id": [str(i) for i in range(n)], "price": 100 + rng.normal(0, 1, n), "amount": rng.exponential(1.0, n)})
        a = X.trades_to_minutes(tr)
        b = X.trades_to_minutes(tr.sample(frac=1.0, random_state=1))
        pd.testing.assert_frame_equal(a, b)
        bad = tr.copy()
        bad.loc[3, "price"] = -1.0
        bad.loc[4, "amount"] = np.nan
        c = X.trades_to_minutes(bad)
        self.assertEqual(int(c["count"].sum()), n - 2)
        self.assertAlmostEqual(float(a["volume"].sum()), float(tr["amount"].sum()), places=9)

    def test_candles_to_rows_cut_and_timezone(self):
        d = pd.date_range("2025-03-31 23:50", periods=6, freq="5min", tz="UTC")
        df = pd.DataFrame({"date": d, "open": 1.0, "high": 2.0, "low": 0.5, "close": [1, 1, 1, 1, 0, 1.0], "volume": 3.0})
        r = X.candles_to_rows(df)
        self.assertEqual(r["ts"].iloc[0], X.CUT_S)               # 2025-04-01 00:00 is the first row kept
        self.assertEqual(len(r), 3)                                # rows at 23:50 and 23:55 (before the cut) and the row with close 0 are gone: 00:00, 00:05, 00:15 remain
        self.assertTrue(r["ts"].is_monotonic_increasing)

    def test_main_end_to_end_with_stubbed_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            kd, rf, out = Path(tmp) / "kd", Path(tmp) / "rf", Path(tmp) / "out"
            kd.mkdir(); rf.mkdir()
            for p in (kd / "ADA_USD-5m.feather", kd / "ADA_USD-trades.feather", rf / "ADA_USD-trades.feather"):
                p.write_bytes(b"x")
            cand = pd.DataFrame({"date": pd.date_range("2025-03-31 23:55", periods=4, freq="5min", tz="UTC"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
            trd = pd.DataFrame({"timestamp": [ms("2025-03-31 23:59:59"), ms("2025-04-01 00:00:10"), ms("2025-04-01 00:00:20")], "id": ["1", "2", "3"], "price": [5.0, 6.0, 7.0], "amount": [1.0, 1.0, 1.0]})

            def fake(path):
                return cand.copy() if str(path).endswith("5m.feather") else trd.copy()
            with mock.patch.object(pd, "read_feather", side_effect=fake):
                X.main(["--kraken-data", str(kd), "--refresh", str(rf), "--out", str(out)])
            cov = json.loads((out / "coverage.json").read_text())
            self.assertEqual(cov["ADA"]["5m"]["rows"], 3)
            self.assertEqual(cov["ADA"]["trades_ft_1m"]["rows"], 1)                     # the trade before the cut is not exported
            self.assertEqual(cov["BTC"]["5m"], None)                                      # symbol without files is recorded as missing, nothing invented
            t = pd.read_csv(out / "ADA_trades_ft_1m.csv")
            self.assertEqual((t.iloc[0].open, t.iloc[0].close, t.iloc[0].volume, t.iloc[0]["count"]), (6.0, 7.0, 2.0, 2))
            man = json.loads((out / "manifest.json").read_text())
            self.assertIn("ADA_5m.csv", man)
            with self.assertRaises(SystemExit):                                           # refuses to overwrite
                X.main(["--kraken-data", str(kd), "--refresh", str(rf), "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
