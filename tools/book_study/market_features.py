"""Chapter 17-18 features of the MARKET (the equal-weight index of the 200-day state), added to the meta-labeling pipeline's events.

The book computes SADF (Ch.17, Snippets 17.1-17.4) and the entropy features (Ch.18, Snippets 18.1, 18.3, 18.4) on a price series; the pipeline already
does it per symbol.  Here the same functions, with exactly the same declared settings (meta_events.SADF_WINDOW / SADF_MINSL / SADF_LAGS = 100 / 20 / 1 and
ENT_WINDOW / ENT_WORD = 100 / 2), are applied to the index: the daily log of the index level (its last hourly value of each UTC day), and each event takes
the value of the last COMPLETED UTC day before its entry, as the per-symbol features do.  No threshold, no new parameter.

The index is the one of book_study_core.market_state: the level moves by the mean hourly log return of the symbols that have a return at that hour, and hours with
fewer than 10 such symbols are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import meta_events as E
import micro_features as MF

MARKET_FEATURES = ["mkt_sadf", "mkt_ent_plugin", "mkt_ent_konto"]


def index_daily_features(idx: pd.Series) -> dict:
    """{feature name: Series indexed by UTC day number} from the hourly index level (positive floats, DatetimeIndex)."""
    d = np.log(idx.resample("1D").last().dropna())
    x = d.to_numpy()
    day_no = d.index.to_numpy().astype("datetime64[D]").astype(np.int64)
    plug, kon = MF.entropy_windows(x, E.ENT_WINDOW, E.ENT_WORD)
    return {"mkt_sadf": pd.Series(MF.sadf_windows(x, E.SADF_WINDOW, E.SADF_MINSL, E.SADF_LAGS), index=day_no),
            "mkt_ent_plugin": pd.Series(plug, index=day_no), "mkt_ent_konto": pd.Series(kon, index=day_no)}


def attach(df: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    """Copy of df (needs t0 in epoch seconds) with the three market columns: each event takes the last completed UTC day before t0."""
    ser = index_daily_features(idx)
    day_no = pd.DatetimeIndex(pd.to_datetime(df["t0"].to_numpy(np.int64), unit="s")).to_numpy().astype("datetime64[D]").astype(np.int64)
    prev = day_no - 1
    out = df.copy()
    for nm in MARKET_FEATURES:
        out[nm] = ser[nm].reindex(prev).to_numpy()
    return out


def use_market_features():
    """Make the pipeline's feature list the 14 per-event features plus the three market features (in-process; no file is changed)."""
    if E.FEATURES[-3:] != MARKET_FEATURES:
        E.FEATURES = list(E.FEATURES) + MARKET_FEATURES
    return E.FEATURES
