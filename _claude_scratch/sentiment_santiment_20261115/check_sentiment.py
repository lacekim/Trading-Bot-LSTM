"""Independent checks: (1) recompute headline numbers from saved trade/equity files; (2) no look-ahead: cutting the
sentiment data at a date never changes an earlier state; (3) diagnostic only (not a rule change): the same trades with
the 12 bp per side added back, to separate the signal from the cost of trading it."""
import json, os, numpy as np, pandas as pd
import sentiment_state as S
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ok=True
for mode in ("gc","sent"):
    res=json.load(open(f"result_{mode}.json"))["rows"]
    for v in ("LIVE","VALIDATED"):
        T=pd.read_csv(f"{mode}_{v}_trades.csv"); eq=pd.read_csv(f"{mode}_{v}_equity.csv",index_col=0,parse_dates=True).iloc[:,0]
        tot=round((eq.iloc[-1]/100000-1)*100,1); dd=round(((eq/eq.cummax())-1).min()*100,1); n=len(T)
        m=(tot==res[v]["total_pct"] and dd==res[v]["max_dd_pct"] and n==res[v]["trades"]); ok&=m
        line=f"{mode:4s} {v:9s} recomputed total={tot}% maxDD={dd}% trades={n} match={m}"
        if mode=="sent":
            # gross: undo the 12 bp on entry and on exit, P&L on the same notional (approximation, stated)
            notional=T.pnl/T.ret
            gross_ret=(1+T.ret)/(1-0.0012)*(1+0.0012)-1
            line+=f" | costs paid ~{round(float((notional*(gross_ret-T.ret)).sum()/1000),1)}% of starting capital; gross P&L ~{round(float((notional*gross_ret).sum()/1000),1)}%; mean hold {round(float((pd.to_datetime(T.exit)-pd.to_datetime(T.entry)).dt.total_seconds().mean()/3600),1)} h"
        print(line)
# no look-ahead
idx=pd.date_range("2025-11-15","2026-10-12 20:00",freq="4h")
for a in ("ETH","SOL","XMR"):
    full=S.sentiment_4h(f"{R}/runtime/santiment/{a}_social_1h.csv")
    st_full,_=S.state(full,idx)
    for cut in ("2026-03-01","2026-07-15"):
        st_cut,_=S.state(full[full.index<cut],idx)
        same=(st_cut[idx<cut]==st_full[idx<cut]).all(); ok&=bool(same)
        print(f"no look-ahead {a} cut {cut}: earlier states unchanged = {bool(same)}")
print("ALL CHECKS PASSED" if ok else "CHECK FAILED")
