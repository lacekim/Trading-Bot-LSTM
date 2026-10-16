import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import bar_loader as BL
import book_port_e as E
import book_vol_printed as V

ROOT = Path(__file__).resolve().parents[2] / "runtime" / "dollar_bars"


class T(unittest.TestCase):
    def test_equals_the_verbatim_port(self):
        for venue, syms in (("kraken", ["BTC", "ETH"]), ("alpaca", None)):
            d = ROOT / venue
            if not d.exists():
                self.skipTest(f"{d} missing")
            man = json.loads((d / "manifest.json").read_text())
            names = syms or BL.list_symbols(d)[:3]
            for s in names:
                b = BL.load_one(d, s, int(man["break_hours"]) * 3600)
                a = E.getDailyVol(pd.Series(b.close, index=b.idx)).reindex(b.idx).to_numpy()
                v = V.daily_vol_printed(b.t, b.close)
                np.testing.assert_allclose(v, a, rtol=0, atol=1e-12, equal_nan=True, err_msg=f"{venue} {s}")

    def test_small_hand_case(self):
        idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 12:00", "2020-01-02 00:00", "2020-01-02 12:00", "2020-01-03 06:00"])
        c = np.array([100., 101., 103., 102., 105.])
        t = (idx - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        v = V.daily_vol_printed(t.to_numpy(np.int64), c)
        a = E.getDailyVol(pd.Series(c, index=idx)).reindex(idx).to_numpy()
        np.testing.assert_allclose(v, a, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
