"""Option 2 rule (committed in 6f1bd01): long while the latest Listing 7.1 prediction is > 0; a prediction from day D acts from day D+1; off before the symbol's first test prediction."""
import pandas as pd
def daily_state(preds_csv, ticker):
    p = pd.read_csv(preds_csv); p = p[p.ticker == ticker]
    if p.empty: return pd.Series(dtype=bool)
    p = p.assign(date=pd.to_datetime(p.date)).sort_values("date")
    s = pd.Series((p.y_pred > 0).values, index=p.date + pd.Timedelta(days=1))
    s = s[~s.index.duplicated(keep="last")]
    full = pd.date_range(s.index.min(), s.index.max() + pd.Timedelta(days=4000), freq="D")
    return s.reindex(full).ffill().astype(bool)
