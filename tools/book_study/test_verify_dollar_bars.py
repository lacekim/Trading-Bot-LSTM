import csv, tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
import build_dollar_bars as B
import verify_dollar_bars as V


def make_rows(seed, n=24 * 60 * 40, dur=60, gap_at=None, gap_len=0):
    rng = np.random.default_rng(seed)
    ts = 1_600_000_000 // 3600 * 3600 + dur * np.arange(n, dtype=np.int64)
    if gap_at:
        ts[gap_at:] += gap_len
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.random(n) * 0.001)
    l = np.minimum(o, c) * (1 - rng.random(n) * 0.001)
    vol = rng.exponential(5.0, n) * (1 + 4 * (rng.random(n) < 0.02))
    vol[rng.random(n) < 0.05] = 0.0
    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": vol})


class T(unittest.TestCase):
    def test_equal_to_the_builder_incl_a_break(self):
        for seed, dur, brk in ((1, 60, 48 * 3600), (2, 3600, 120 * 3600)):
            n = 24 * 60 * 40 if dur == 60 else 24 * 120
            df = make_rows(seed, n=n, dur=dur, gap_at=n // 2, gap_len=dur * 0 + (60 * 3600 * 2 if dur == 60 else 7 * 86400))
            want = B.dollar_bars(df, dur, brk)
            rows = [tuple(x) for x in df[["ts", "open", "high", "low", "close", "volume"]].itertuples(index=False, name=None)]
            rows = [(int(r[0]),) + tuple(float(y) for y in r[1:]) for r in rows]
            got, bad, nchk = V.bars_of(rows, dur, brk, brute_every=7)
            self.assertGreater(len(want), 20)
            self.assertEqual(len(got), len(want))
            self.assertEqual(bad, [])
            self.assertGreater(nchk, 3)
            for g, (_, w) in zip(got, want.iterrows()):
                self.assertEqual((g["start"], g["end"], g["n"]), (int(w["start"]), int(w["end"]), int(w["n_rows"])))
                for a, b in (("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("volume", "volume"), ("value", "value"), ("thr", "threshold")):
                    self.assertTrue(V.close(g[a], float(w[b]), 1e-9), (a, g[a], w[b]))

    def test_catches_a_planted_error_in_a_bar_file(self):
        df = make_rows(3, n=24 * 60 * 20)
        rows = [(int(r[0]),) + tuple(float(y) for y in r[1:]) for r in df[["ts", "open", "high", "low", "close", "volume"]].itertuples(index=False, name=None)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            res = B.write_symbol(out, "X", df, "kraken", {})
            def run():
                msgs = []
                V.verify_symbol("kraken", "X", rows, out, res, lambda n, ok, det: msgs.append((n, ok)))
                return msgs
            self.assertTrue(all(ok for _, ok in run()))
            p = out / "bars" / "X.csv"
            txt = p.read_text().splitlines()
            f = txt[40].split(",")
            f[5] = str(float(f[5]) * 1.0001)           # close changed in the 40th bar
            txt[40] = ",".join(f)
            p.write_text("\n".join(txt) + "\n")
            m = run()
            self.assertTrue(any(not ok for n, ok in m if "every bar" in n))
            self.assertTrue(any(not ok for n, ok in m if "md5" in n))


if __name__ == "__main__":
    unittest.main()
