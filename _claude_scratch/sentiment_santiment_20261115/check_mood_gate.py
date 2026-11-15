"""Checks for test 2: (1) headline numbers recomputed from saved files; (2) no look-ahead in the euphoria series;
(3) which trades the gate changed, one by one."""
import json, os, pandas as pd
import mood_gate as G
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
ok=True
for mode in ("off","gate"):
    res=json.load(open(f"mood_result_{mode}.json"))["rows"]
    for v in ("LIVE","VALIDATED"):
        T=pd.read_csv(f"mood_{mode}_{v}_trades.csv"); eq=pd.read_csv(f"mood_{mode}_{v}_equity.csv",index_col=0,parse_dates=True).iloc[:,0]
        tot=round((eq.iloc[-1]/100000-1)*100,1); dd=round(((eq/eq.cummax())-1).min()*100,1)
        m=(tot==res[v]["total_pct"] and dd==res[v]["max_dd_pct"] and len(T)==res[v]["trades"]); ok&=m
        print(f"{mode:4s} {v:9s} recomputed total={tot}% maxDD={dd}% trades={len(T)} match={m}")
m4=G.mood_4h(R); full,_=G.euphoric(m4)
for cut in ("2026-04-15","2026-07-01"):
    part,_=G.euphoric(m4[m4.index<cut]); same=(part==full[full.index<cut]).all(); ok&=bool(same)
    print(f"no look-ahead, mood data cut at {cut}: earlier euphoria flags unchanged = {bool(same)}")
for v in ("LIVE","VALIDATED"):
    a=pd.read_csv(f"mood_off_{v}_trades.csv"); b=pd.read_csv(f"mood_gate_{v}_trades.csv")
    ka=set(zip(a.asset,a.entry)); kb=set(zip(b.asset,b.entry))
    gone=a[[k in ka-kb for k in zip(a.asset,a.entry)]]; new=b[[k in kb-ka for k in zip(b.asset,b.entry)]]
    print(f"\n{v}: trades only WITHOUT the gate (blocked or moved): {len(gone)}")
    for r in gone.itertuples(): print(f"   {r.asset:5s} {r.entry} -> {r.exit}  {r.ret*100:+.1f}%  pnl {r.pnl:+.0f}  exit={r.why}")
    print(f"{v}: trades only WITH the gate (entered later instead): {len(new)}")
    for r in new.itertuples(): print(f"   {r.asset:5s} {r.entry} -> {r.exit}  {r.ret*100:+.1f}%  pnl {r.pnl:+.0f}  exit={r.why}")
print("\nALL CHECKS PASSED" if ok else "\nCHECK FAILED")
