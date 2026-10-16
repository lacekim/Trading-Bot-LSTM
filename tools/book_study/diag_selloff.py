"""Descriptive look at the sell-off window from a finished fast-drop run folder (reads events_fastdrop.csv only; nothing fitted)."""
import json, sys, pandas as pd, numpy as np
d = pd.read_csv(sys.argv[1] + "/events_fastdrop.csv")
lo, hi = [int(x) for x in json.load(open(sys.argv[1] + "/summary_numbers.json"))["meta"]["window_epoch"]["sell_off_post_holdout"]]
w = d[(d.t0 >= lo) & (d.t0 < hi)].copy()
w["day"] = pd.to_datetime(w.t0, unit="s").dt.strftime("%Y-%m-%d")
w["ml"] = (w.mom_side == 1) & (w.mkt_state == 1)
print("events in window", len(w), "| mom-long-up candidates", int(w.ml.sum()))
g = w.groupby("day")
t = pd.DataFrame({"events": g.size(), "idx_ret24h_%": g.idxret24.mean() * 100, "idx_ret72h_%": g.idxret72.mean() * 100, "idx_ret168h_%": g.idxret168.mean() * 100,
                  "mean_ret_long_%_all": g.ret_long.mean() * 100})
m = w[w.ml].groupby("day")
t["momlong_events"] = m.size()
t["momlong_ret_%"] = m.ret_long.mean() * 100
t["momlong_share_losing"] = m.ret_long.apply(lambda x: (x < 0).mean())
print(t.round(2).to_string())
a = w[w.ml]
print("\nmom-long-up trades in window: n", len(a), "| mean ret_long %.2f%% | share ret_long<0 %.3f | symbols %d" % (a.ret_long.mean() * 100, (a.ret_long < 0).mean(), a.symbol.nunique()))
print("Whole-window equal-event mean of all events ret_long: %.2f%%" % (w.ret_long.mean() * 100))
print("by symbol count of mom-long trades, top 5 share: %.3f" % (a.symbol.value_counts().head(5).sum() / len(a)))
