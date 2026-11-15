"""Independent check of option 2 from the saved files only, plus: every entry follows a positive prediction dated before the entry day."""
import json, os, pandas as pd, numpy as np
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
res=json.load(open("option2_result.json")); ok=True
for k,r in res["kraken"].items():
    mode,v,name=k.split("|")
    T=pd.read_csv(f"k2_{mode}_{v}_{name}_trades.csv"); eq=pd.read_csv(f"k2_{mode}_{v}_{name}_equity.csv",index_col=0,parse_dates=True).iloc[:,0]
    tot=round((eq.iloc[-1]/100000-1)*100,1); dd=round(((eq/eq.cummax())-1).min()*100,1); m=(tot==r["total_pct"] and dd==r["max_dd_pct"] and len(T)==r["trades"]); ok&=m
    print(f"kraken {k:48s} recomputed {tot}% / {dd}% / {len(T)} trades  match={m}")
def port(df):
    ev=[]
    for i,x in df.iterrows(): ev+=[(x.entry_time,0,i),(x.exit_time,1,i)]
    ev.sort(); cap=100000.0; peak=cap; dd=0.0; n={}
    for ts,k,i in ev:
        if k==0: n[i]=cap*0.02
        else: cap+=n[i]*df.net_ret_pct[i]/100; peak=max(peak,cap); dd=min(dd,(cap/peak-1)*100)
    return round((cap/100000-1)*100,2), round(dd,2)
for mode in ("gc","sent"):
    c=pd.read_csv(f"a2_{mode}_closed.csv",parse_dates=["entry_time","exit_time"]); o=pd.read_csv(f"a2_{mode}_open.csv",parse_dates=["entry_time","exit_time"])
    both=pd.concat([c[c.entry_time>="2025-12-10"],o[o.entry_time>="2025-12-10"]],ignore_index=True); tot,dd=port(both); e=res["alpaca"][f"{mode}|entered_from_2025-12-10"]
    m=abs(tot-e["return_pct"])<0.02 and abs(dd-e["max_dd_pct"])<0.02; ok&=m
    print(f"alpaca {mode:4s} recomputed {tot}% / {dd}%  engine {e['return_pct']}% / {e['max_dd_pct']}%  match={m}")
bad=0; n=0
for venue,files in (("kraken",["k2_sent_VALIDATED_window_2025-08-02_to_2026-10-12_trades.csv"]),("alpaca",["a2_sent_closed.csv"])):
    p=pd.read_csv(f"{R}/runtime/llm_news/{venue}/listing_7_1_preds.csv",parse_dates=["date"])
    for f in files:
        T=pd.read_csv(f)
        ecol,acol=("entry","asset") if venue=="kraken" else ("entry_time","symbol")
        T[ecol]=pd.to_datetime(T[ecol])
        for x in T.itertuples():
            day=getattr(x,ecol).normalize(); sym=getattr(x,acol)
            q=p[(p.ticker==sym)&(p.date<day)].sort_values("date").tail(1)
            n+=1
            if q.empty or not (q.y_pred.iloc[0]>0): bad+=1
print(f"entries whose latest prediction dated before the entry day was not positive: {bad} of {n} (must be 0)")
ok&=(bad==0)
print("ALL CHECKS PASSED" if ok else "CHECK FAILED")
