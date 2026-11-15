"""Crowd-mood regime gate per the rule committed in 261759c. Past data only."""
import numpy as np, pandas as pd
MOOD_ASSETS = ("BTC", "ETH", "SOL"); ROLL, WIN, P = 42, 360, 0.90

def mood_4h(root):
    num = den = None
    for a in MOOD_ASSETS:
        d = pd.read_csv(f"{root}/runtime/santiment/{a}_social_1h.csv", parse_dates=["datetime"]).set_index("datetime")
        d.index = d.index.tz_convert(None)
        n = d.sentiment_weighted_total * d.social_volume_total; v = d.social_volume_total
        num = n if num is None else num.add(n, fill_value=0); den = v if den is None else den.add(v, fill_value=0)
    hourly = (num / den.replace(0, np.nan))
    return hourly.resample("4h").mean()

def euphoric(m4h):
    """Series on the 4h grid: True when the 7-day mood (through the previous bar) is at/above its trailing 60-day 90th pct."""
    slow = m4h.rolling(ROLL, min_periods=ROLL).mean().shift(1)
    v = slow.to_numpy(float); out = np.zeros(len(v), bool)
    for t in range(len(v)):
        w = v[max(0, t - WIN + 1): t + 1]
        if len(w) < WIN or np.isnan(w).any():
            continue
        out[t] = (w <= v[t]).mean() >= P
    return pd.Series(out, index=slow.index), slow
