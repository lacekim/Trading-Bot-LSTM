"""Meta-labeling events with the four volume features added, without editing meta_events.py.

build_meta_events_vol() calls meta_events.build_meta_events() unchanged (with the original 14-feature list), adds the volume columns at each
event bar, drops events whose volume features are not finite, and recomputes the uniqueness / attribution weights on the events that remain
(Snippets 4.1, 4.2, 4.10, exactly as meta_events does).  If no event is dropped the first 14 features, weights and rows are those of the
plain pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import meta_events as E
import volume_features as VF

BASE_FEATURES = list(E.FEATURES)
_ORIG_BUILD = E.build_meta_events         # captured at import, so a caller may replace E.build_meta_events with a wrapper around this module
FEATURES_18 = BASE_FEATURES + VF.VOL_FEATURES


def build_meta_events_vol(closes: dict, ohlc: dict, vols: dict, h: int = 100, ptsl=(2.0, 2.0), cost_frac: float = 0.0024, log=print):
    for s, c in closes.items():
        if s not in vols or not vols[s].index.equals(c[c > 0].index):
            raise SystemExit(f"{s}: volume column does not sit on the bars of the close series used by the loader; stopping")
    saved = E.FEATURES
    E.FEATURES = BASE_FEATURES
    try:
        df, info = _ORIG_BUILD(closes, ohlc, h=h, ptsl=ptsl, cost_frac=cost_frac, log=log)
    finally:
        E.FEATURES = saved
    cols = {k: np.full(len(df), np.nan) for k in VF.VOL_FEATURES}
    for sym, grp in df.groupby("symbol", sort=False):
        c = closes[sym].astype(float)
        c = c[c > 0]
        feats = VF.volume_features(c.to_numpy(), vols[sym].to_numpy())
        ix = grp.index.to_numpy()
        p = grp["pos"].to_numpy(np.int64)
        for k in VF.VOL_FEATURES:
            cols[k][ix] = feats[k][p]
    for k in VF.VOL_FEATURES:
        df[k] = cols[k]
    keep = np.isfinite(df[VF.VOL_FEATURES].to_numpy(dtype=np.float64)).all(axis=1)
    n_drop = int((~keep).sum())
    if n_drop:
        df = df[keep].reset_index(drop=True)
        u_arr = np.full(len(df), np.nan)
        w_arr = np.full(len(df), np.nan)
        for sym, grp in df.groupby("symbol", sort=False):
            c = closes[sym].astype(float)
            c = c[c > 0]
            ix = grp.index.to_numpy()
            u_arr[ix], w_arr[ix] = E.attribution_weights(grp["pos"].to_numpy(np.int64), grp["t1pos"].to_numpy(np.int64), np.log(c.to_numpy()))
        df["u"], df["w_raw"] = u_arr, w_arr
        df["w"] = df["w_raw"] * len(df) / df["w_raw"].sum()
    info = dict(info)
    info["events_dropped_missing_volume_feature"] = n_drop
    info["events_kept"] = int(len(df))
    info["kind_share"] = {str(k): float((df["kind"] == k).mean()) for k in (1, -1, 0)}
    info["bin_share"] = float(df["bin"].mean())
    log(f"volume features: {n_drop:,} events dropped for a non-finite volume feature; kept {len(df):,}")
    return df, info
