import unittest

import numpy as np
import pandas as pd

import book_study_core as C
import market_features as MKF
import meta_eval as ME
import meta_events as E
import micro_features as MF
from test_three_algo import synthetic_events


def hourly_index(seed=1, days=260, gap_day=None):
    rng = np.random.default_rng(seed)
    hours = np.arange(days * 24)
    if gap_day is not None:
        hours = hours[(hours // 24) != gap_day]
    t = pd.to_datetime(hours * 3600 + 1_600_000_000 - (1_600_000_000 % 86400), unit="s")
    return pd.Series(np.exp(np.cumsum(rng.standard_normal(len(hours)) * 0.004)), index=t)


class TestDaily(unittest.TestCase):
    def test_daily_features_use_the_last_hour_of_each_utc_day_and_skip_empty_days(self):
        idx = hourly_index(gap_day=150)
        ser = MKF.index_daily_features(idx)
        # literal: last level of each UTC day that has any hour
        days = {}
        for ts, v in idx.items():
            days[int(ts.timestamp()) // 86400] = v
        keys = sorted(days)
        x = np.log(np.array([days[k] for k in keys]))
        sad = MF.sadf_windows(x, E.SADF_WINDOW, E.SADF_MINSL, E.SADF_LAGS)
        plug, kon = MF.entropy_windows(x, E.ENT_WINDOW, E.ENT_WORD)
        self.assertEqual(list(ser["mkt_sadf"].index), keys)
        np.testing.assert_allclose(ser["mkt_sadf"].to_numpy(), sad, rtol=0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(ser["mkt_ent_plugin"].to_numpy(), plug, rtol=0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(ser["mkt_ent_konto"].to_numpy(), kon, rtol=0, atol=1e-12, equal_nan=True)
        self.assertNotIn(150, [k - keys[0] for k in keys])               # the empty day is absent, not zero

    def test_nan_before_a_full_window(self):
        ser = MKF.index_daily_features(hourly_index(days=150))
        self.assertTrue(np.isnan(ser["mkt_sadf"].to_numpy()[:E.SADF_WINDOW - 1]).all())
        self.assertTrue(np.isfinite(ser["mkt_sadf"].to_numpy()[E.SADF_WINDOW:]).any())


class TestAttach(unittest.TestCase):
    def test_event_takes_the_last_completed_day_and_cannot_see_its_own_day(self):
        idx = hourly_index(days=260)
        ser = MKF.index_daily_features(idx)
        day0 = int(idx.index[0].timestamp()) // 86400
        d = 200
        t0 = (day0 + d) * 86400 + 5 * 3600
        df = pd.DataFrame({"t0": [t0, t0 + 3600 * 12]})
        out = MKF.attach(df, idx)
        for nm in MKF.MARKET_FEATURES:
            self.assertEqual(out[nm][0], ser[nm].loc[day0 + d - 1])
            self.assertEqual(out[nm][1], out[nm][0])
        # change the index only from the event's own day on: features must not move
        idx2 = idx.copy()
        idx2[idx2.index >= pd.Timestamp((day0 + d) * 86400, unit="s")] *= 1.37
        out2 = MKF.attach(df, idx2)
        for nm in MKF.MARKET_FEATURES:
            self.assertEqual(out2[nm][0], out[nm][0])
        # change an earlier day: the feature moves
        idx3 = idx.copy()
        idx3[(idx3.index >= pd.Timestamp((day0 + d - 20) * 86400, unit="s")) & (idx3.index < pd.Timestamp((day0 + d - 10) * 86400, unit="s"))] *= 1.2
        self.assertNotEqual(MKF.attach(df, idx3)["mkt_sadf"][0], out["mkt_sadf"][0])

    def test_other_columns_untouched(self):
        idx = hourly_index()
        df = pd.DataFrame({"t0": [1_600_000_000 + 220 * 86400], "a": [1.5], "symbol": ["X"]})
        out = MKF.attach(df, idx)
        self.assertEqual(list(out.columns), ["t0", "a", "symbol"] + MKF.MARKET_FEATURES)
        self.assertEqual(out["a"][0], 1.5)


class TestFeatureList(unittest.TestCase):
    def setUp(self):
        self._saved = list(E.FEATURES)

    def tearDown(self):
        E.FEATURES = self._saved

    def test_use_market_features_is_idempotent_and_reaches_prepare(self):
        closes, df0 = synthetic_events()
        t0 = df0["t0"].to_numpy(np.int64)
        _, idx, _ = C.market_state(closes, t0)
        df = MKF.attach(df0, idx)
        df = df[np.isfinite(df[MKF.MARKET_FEATURES].to_numpy()).all(axis=1)].reset_index(drop=True)
        self.assertGreater(len(df), 100)
        MKF.use_market_features()
        MKF.use_market_features()
        self.assertEqual(len(E.FEATURES), 17)
        self.assertEqual(E.FEATURES[:14], self._saved)
        X = ME.prepare(df)[0]
        self.assertEqual(X.shape, (len(df), 17))
        np.testing.assert_array_equal(X[:, 14:], df[MKF.MARKET_FEATURES].to_numpy())
        preds, smet, _ = ME.run_cpcv(df, 4, 2, 0.01, dict(n_estimators=5, bootstrap="sequential", max_draws=1500), 1, log=lambda m: None)
        self.assertTrue(((preds["p"] >= 0) & (preds["p"] <= 1)).all() if "p" in preds.columns else True)
        self.assertEqual(len(smet), 6)


if __name__ == "__main__":
    unittest.main()
