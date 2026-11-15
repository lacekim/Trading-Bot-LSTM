"""Sentiment state per the rule committed in CHANGES.md (d358369). Past data only."""
import numpy as np, pandas as pd
WIN, P_IN, P_OUT = 180, 0.90, 0.30

def sentiment_4h(path):
    s = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["sentiment_weighted_total"]
    s.index = s.index.tz_convert(None)
    return s.resample("4h").mean()          # same 4h bins as the engine's candle resample

def state(sent4h, index):
    x = sent4h.reindex(index).shift(1)      # decision at bar t uses bar t-1 only
    v = x.to_numpy(float); n = len(v); pct = np.full(n, np.nan)
    for t in range(n):
        w = v[max(0, t - WIN + 1): t + 1]
        if np.isnan(v[t]) or np.isnan(w).sum() > 0 or len(w) < WIN:
            continue
        pct[t] = (w <= v[t]).mean()
    st = np.zeros(n, bool); on = False
    for t in range(n):
        if np.isnan(pct[t]):
            on = False
        elif pct[t] >= P_IN:
            on = True
        elif pct[t] <= P_OUT:
            on = False
        st[t] = on
    return pd.Series(st, index=index), pd.Series(pct, index=index)
