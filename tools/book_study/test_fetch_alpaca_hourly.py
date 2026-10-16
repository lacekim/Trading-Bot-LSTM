import json, tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
import fetch_alpaca_hourly as F


def sdk_like(symbols, n=5):
    rows = []
    for s in symbols:
        for i in range(n):
            rows.append((s, pd.Timestamp("2020-01-02 14:00", tz="UTC") + pd.Timedelta(hours=i), 10 + i, 11 + i, 9 + i, 10.5 + i, 100.0 * (i + 1), 7, 10.2 + i))
    df = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
    return df.set_index(["symbol", "timestamp"])


class T(unittest.TestCase):
    def test_bars_to_frames(self):
        df = sdk_like(["AAA", "NA"])
        df.loc[("AAA", pd.Timestamp("2020-01-02 15:00", tz="UTC")), "close"] = 0.0       # a non-positive close is dropped
        fr = F.bars_to_frames(df)
        self.assertEqual(set(fr), {"AAA", "NA"})
        self.assertEqual(len(fr["AAA"]), 4)
        self.assertEqual(fr["NA"]["date"].iloc[0], "2020-01-02 14:00:00")
        self.assertEqual(list(fr["NA"].columns), ["date", "open", "high", "low", "close", "volume", "trade_count", "vwap"])

    def test_read_universe_keeps_NA_ticker(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "u.csv"
            p.write_text("month_end,rank,symbol\n2017-01-31,1,SPY\n2017-01-31,2,NA\n2017-02-28,1,SPY\n")
            self.assertEqual(F.read_universe(p), ["NA", "SPY"])

    def test_run_resume_empty_and_failed(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "o"
            calls = []
            def fake(client, batch, start, end, feed):
                calls.append(list(batch))
                if "BAD" in batch:
                    raise RuntimeError("boom")
                return sdk_like([s for s in batch if s != "EMPTY"])
            failed = F.run(["AAA", "BAD", "EMPTY", "ZZZ"], out, None, None, "sip", 2, None, fetch=fake)
            self.assertEqual(sorted(calls), [["AAA", "BAD"], ["EMPTY", "ZZZ"]])
            self.assertEqual(failed, ["AAA", "BAD"])                                    # the whole batch containing the failing request is reported
            self.assertFalse((out / "AAA_1h.csv").exists())
            self.assertEqual((out / "EMPTY_1h.csv").read_text().strip(), "date,open,high,low,close,volume,trade_count,vwap")   # asked, nothing returned
            self.assertEqual(len(pd.read_csv(out / "ZZZ_1h.csv")), 5)
            calls.clear()
            failed = F.run(["AAA", "BAD", "EMPTY", "ZZZ"], out, None, None, "sip", 2, None, fetch=fake)
            self.assertEqual(sorted(calls), [["AAA", "BAD"]])                           # completed symbols are not fetched again
            self.assertEqual((out / "failed.txt").read_text().split(), ["AAA", "BAD"])
            man = json.loads((out / "manifest.json").read_text())
            self.assertEqual(set(man), {"EMPTY_1h.csv", "ZZZ_1h.csv"})

    def test_appledouble_stub_is_ignored(self):
        """Regression: on the external drive macOS leaves '._SYM_1h.csv' stubs (binary, not UTF-8); the coverage step crashed on one."""
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "o"
            out.mkdir()
            (out / "._AAA_1h.csv").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        \xb0\xb0\xff\xfe")
            def fake(client, batch, start, end, feed):
                return sdk_like(batch)
            failed = F.run(["AAA", "ZZZ"], out, None, None, "sip", 2, None, fetch=fake)
            self.assertEqual(failed, [])
            man = json.loads((out / "manifest.json").read_text())
            self.assertEqual(set(man), {"AAA_1h.csv", "ZZZ_1h.csv"})
            self.assertNotIn("._AAA_1h.csv", (out / "coverage.txt").read_text())


if __name__ == "__main__":
    unittest.main()
