"""Independent check (plain numpy, no code shared with the engine) of the 100/400 crossover on the Alpaca daily closes."""
import json, os, numpy as np, pandas as pd
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
uni=[r["symbol"] for r in json.load(open(R+"/runtime/cross_asset_golden_cross_results.json"))["results"] if r["robust"] and r["symbol"] not in {"BTC-USD","ETH-USD","SOL-USD","PAXG-USD"}]
m={"GC=F":"GLD","PA=F":"PPLT","BRK-B":"BRK.B"}
d=pd.read_csv(R+"/runtime/venue_candles/alpaca_1d/alpaca_daily.csv.gz",parse_dates=["date"]); have=set(d.symbol.unique())
C=0.0012; S,L=100,400; closed=[]; opens=[]
for u in uni:
    s=m.get(u,u)
    if s not in have: continue
    x=d[d.symbol==s].sort_values("date"); c=x.close.to_numpy(float); t=x.date.to_numpy(); cs=np.concatenate([[0],np.cumsum(c)])
    ms=np.full(len(c),np.nan); ml=np.full(len(c),np.nan); ms[S-1:]=(cs[S:]-cs[:-S])/S; ml[L-1:]=(cs[L:]-cs[:-L])/L
    pos=None
    for i in range(L-1,len(c)):
        up=ms[i]>ml[i]; prev=(ms[i-1]>ml[i-1]) if i>L-1 else False
        if pos is None and up and not prev: pos=i
        elif pos is not None and (not up) and prev: closed.append((t[pos],(c[i]*(1-C)/(c[pos]*(1+C))-1)*100)); pos=None
    if pos is not None: opens.append((t[pos],(c[-1]*(1-C)/(c[pos]*(1+C))-1)*100))
eng=json.load(open("alpaca_book_in_bar_engine.json")); ok=True
for name,s in {"entered_from_2024-01-01":"2024-01-01","entered_from_2025-10-10":"2025-10-10","whole_file":None}.items():
    nc=sum(1 for e,_ in closed if s is None or e>=np.datetime64(s)); no=sum(1 for e,_ in opens if s is None or e>=np.datetime64(s))
    match=(nc==eng[name]["closed_trades"] and no==eng[name]["open_positions"]); ok&=match
    print(f"{name:26s} independent closed={nc} open={no} | engine closed={eng[name]['closed_trades']} open={eng[name]['open_positions']} | match={match}")
print("ALL MATCH" if ok else "MISMATCH")
