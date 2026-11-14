"""Independent check of the Alpaca HMM-filter run, from the saved trade files only (no engine code)."""
import json, pandas as pd, numpy as np
def port(df):
    ev=[]
    for i,r in df.iterrows(): ev+=[(r.entry_time,0,i),(r.exit_time,1,i)]
    ev.sort(); cap=100000.0; peak=cap; dd=0.0; n={}
    for ts,k,i in ev:
        if k==0: n[i]=cap*0.02
        else: cap+=n[i]*df.net_ret_pct[i]/100; peak=max(peak,cap); dd=min(dd,(cap/peak-1)*100)
    return round((cap/100000-1)*100,2), round(dd,2)
ok=True
for mode in ("off","hmm"):
    c=pd.read_csv(f"alpaca_{mode}_closed_trades.csv",parse_dates=["entry_time","exit_time"]); o=pd.read_csv(f"alpaca_{mode}_open_positions.csv",parse_dates=["entry_time","exit_time"])
    eng=json.load(open(f"alpaca_regime_{mode}.json"))["rows"]
    for name,s in {"entered_from_2024-01-01":"2024-01-01","entered_from_2025-10-10":"2025-10-10"}.items():
        cc=c[c.entry_time>=s]; oo=o[o.entry_time>=s]; both=pd.concat([cc,oo],ignore_index=True)
        tot,dd=port(both); e=eng[name]
        m=(len(cc)==e["closed_trades"] and len(oo)==e["open_positions"] and abs(tot-e["with_open_marked_return_pct"])<0.02)
        ok&=m; print(f"{mode:4s} {name:26s} closed={len(cc)} open={len(oo)} return={tot}% maxDD={dd}% | engine {e['with_open_marked_return_pct']}% / {e['with_open_marked_max_dd_pct']}% | match={m}")
a=pd.read_csv("alpaca_off_closed_trades.csv",parse_dates=["entry_time"]); b=pd.read_csv("alpaca_hmm_closed_trades.csv",parse_dates=["entry_time"])
ka=set(zip(a.symbol,a.entry_time)); kb=set(zip(b.symbol,b.entry_time))
print("closed trades: off",len(ka),"hmm",len(kb),"| in hmm but not in off:",len(kb-ka),"(must be 0: the filter can only remove entries)")
ok&=(len(kb-ka)==0)
print("ALL MATCH" if ok else "MISMATCH")
