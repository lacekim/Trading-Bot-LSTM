"""Independent check of option 3 from the saved files only, plus a no-look-ahead test of the sentiment state."""
import json, os, pandas as pd, numpy as np
import sentdex_state as S
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
res=json.load(open("option3_result.json")); ok=True
for k,r in res["kraken"].items():
    mode,v,name=k.split("|")
    T=pd.read_csv(f"k_{mode}_{v}_{name}_trades.csv"); eq=pd.read_csv(f"k_{mode}_{v}_{name}_equity.csv",index_col=0,parse_dates=True).iloc[:,0]
    tot=round((eq.iloc[-1]/100000-1)*100,1); dd=round(((eq/eq.cummax())-1).min()*100,1); m=(tot==r["total_pct"] and dd==r["max_dd_pct"] and len(T)==r["trades"]); ok&=m
    print(f"kraken {k:36s} recomputed {tot}% / {dd}% / {len(T)} trades  match={m}")
def port(df):
    ev=[]
    for i,x in df.iterrows(): ev+=[(x.entry_time,0,i),(x.exit_time,1,i)]
    ev.sort(); cap=100000.0; peak=cap; dd=0.0; n={}
    for ts,k,i in ev:
        if k==0: n[i]=cap*0.02
        else: cap+=n[i]*df.net_ret_pct[i]/100; peak=max(peak,cap); dd=min(dd,(cap/peak-1)*100)
    return round((cap/100000-1)*100,2), round(dd,2)
for mode in ("gc","sent"):
    c=pd.read_csv(f"a_{mode}_closed.csv",parse_dates=["entry_time","exit_time"]); o=pd.read_csv(f"a_{mode}_open.csv",parse_dates=["entry_time","exit_time"])
    for name,s in {"entered_from_2024-01-01":"2024-01-01","entered_from_2025-10-10":"2025-10-10"}.items():
        both=pd.concat([c[c.entry_time>=s],o[o.entry_time>=s]],ignore_index=True); tot,dd=port(both); e=res["alpaca"][f"{mode}|{name}"]
        m=abs(tot-e["return_pct"])<0.02 and abs(dd-e["max_dd_pct"])<0.02; ok&=m
        print(f"alpaca {mode:4s} {name:26s} recomputed {tot}% / {dd}%  engine {e['return_pct']}% / {e['max_dd_pct']}%  match={m}")
# no look-ahead: the state on day D must not change when sentiment from day D onward is removed
src=R+"/runtime/llm_news/kraken/daily_sent.csv"; full=pd.read_csv(src)
for t,cut in (("BTC","2025-03-01"),("SOL","2024-08-15")):
    a=S.daily_state(src,t); tmp="/tmp/_ds.csv"; full[pd.to_datetime(full.date)<cut].to_csv(tmp,index=False); b=S.daily_state(tmp,t)
    days=b.index[b.index<=pd.Timestamp(cut)]; same=(a.reindex(days)==b.reindex(days)).all(); ok&=bool(same)
    print(f"no look-ahead {t}: state through {cut} unchanged when sentiment from {cut} on is removed = {bool(same)}")
# every option-3 Kraken entry: the previous UTC day's net must be >= 0.8 or a later on-state from an earlier day
d=pd.read_csv(src); d["net"]=d.sent_positive-d.sent_negative
T=pd.read_csv("k_sent_VALIDATED_from_2024-01-01_trades.csv",parse_dates=["entry"])
bad=0
for x in T.itertuples():
    prev=d[(d.ticker==x.asset)&(pd.to_datetime(d.date)<x.entry.normalize())].sort_values("date")
    last=prev[(prev.net>=0.8)|(prev.net<=-0.4)].tail(1)
    if last.empty or last.net.iloc[0]<0.8: bad+=1
print(f"VALIDATED entries whose last threshold reading before the entry day was not an entry reading: {bad} of {len(T)} (must be 0)")
ok&=(bad==0)
print("ALL CHECKS PASSED" if ok else "CHECK FAILED")
