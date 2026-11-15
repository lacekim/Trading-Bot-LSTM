"""Option 3 rule (committed in 5d266fc): Sentdex thresholds on FinBERT net, previous UTC day only, hold on days without news."""
import pandas as pd
ENTRY, EXIT = 0.8, -0.4

def daily_state(daily_sent_csv, ticker):
    d = pd.read_csv(daily_sent_csv)
    d = d[d.ticker == ticker].copy()
    if d.empty:
        return pd.Series(dtype=bool)
    d["date"] = pd.to_datetime(d.date); d = d.sort_values("date")
    net = (d.sent_positive - d.sent_negative).values
    on = False; out = {}
    for day, n in zip(d.date, net):
        if n >= ENTRY: on = True
        elif n <= EXIT: on = False
        out[day + pd.Timedelta(days=1)] = on      # effective from the NEXT UTC day
    s = pd.Series(out)
    full = pd.date_range(s.index.min(), s.index.max() + pd.Timedelta(days=4000), freq="D")
    return s.reindex(full).ffill().fillna(False).astype(bool)

def on_bars(daily, index):
    """State for each bar: the state effective on the bar's UTC calendar day."""
    if daily.empty: return pd.Series(False, index=index)
    return pd.Series(daily.reindex(index.normalize()).fillna(False).astype(bool).values, index=index)
