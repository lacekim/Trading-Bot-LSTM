import json, glob, numpy as np, pandas as pd, warnings; warnings.simplefilter("ignore")
R="/sessions/rcw-01evuvgej6pp73aaaw9sdbhm/mnt/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera"
print("== KRAKEN: recompute headline numbers from saved trades/equity, separate code")
kb=json.load(open("kraken_bar.json")); bad=0
for k,r in kb.items():
    v,name=k.split("|")
    T=pd.read_csv(f"kraken_{v}_{name}_trades.csv"); eq=pd.read_csv(f"kraken_{v}_{name}_equity.csv",index_col=0,parse_dates=True).iloc[:,0]
    tot=round((eq.iloc[-1]/100000-1)*100,1); dd=round(((eq/eq.cummax())-1).min()*100,1)
    pf=round(T.pnl[T.pnl>0].sum()/-T.pnl[T.pnl<0].sum(),2); n=len(T)
    # capital identity: final equity = 100k + realised pnl + open mark-to-market (>= realised only if open pos)
    ok=(tot==r["total_pct"] and dd==r["max_dd_pct"] and pf==r["pf"] and n==r["trades"])
    bad+=not ok
    print(f"{k:55s} n={n} tot={tot} dd={dd} pf={pf} match={ok} realised_only_total={round(T.pnl.sum()/1000,1)}%")
print("kraken mismatches:",bad)
print("== ALPACA: independent golden-cross on raw closes (numpy), plus open positions marked at last close")
d=pd.read_csv(R+"/runtime/venue_candles/alpaca_1d/alpaca_daily.csv.gz",parse_dates=["date"])
mine=pd.read_csv("alpaca_daily_gc_trades.csv",parse_dates=["entry_time","exit_time"])
syms=sorted(mine.symbol.unique().tolist()+[s for s in pd.read_csv("alpaca_daily_gc_per_symbol.csv").symbol if s not in set(mine.symbol)])
syms=sorted(set(syms)); C=0.0012; closed=[]; opens=[]
for s in syms:
    x=d[d.symbol==s].sort_values("date"); c=x.close.to_numpy(float); t=x.date.to_numpy()
    cs=np.concatenate([[0],np.cumsum(c)])
    ma=lambda n:(cs[n:]-cs[:-n])/n
    m50=np.full(len(c),np.nan); m200=np.full(len(c),np.nan); m50[49:]=ma(50); m200[199:]=ma(200)
    pos=None
    for i in range(200,len(c)):
        up=m50[i]>m200[i]; prev=(m50[i-1]>m200[i-1]) if not np.isnan(m200[i-1]) else False
        if pos is None and up and not prev: pos=i
        elif pos is not None and (not up) and prev:
            closed.append((s,t[pos],t[i],(c[i]*(1-C)/(c[pos]*(1+C))-1)*100)); pos=None
    if pos is not None: opens.append((s,t[pos],t[-1],(c[-1]*(1-C)/(c[pos]*(1+C))-1)*100))
Cl=pd.DataFrame(closed,columns=["symbol","entry","exit","ret"]); Op=pd.DataFrame(opens,columns=["symbol","entry","exit","ret"])
print("closed trades: engine",len(mine),"independent",len(Cl),"| max abs return diff",float(np.abs(np.sort(mine.net_ret_pct.values)-np.sort(Cl.ret.values)).max()) if len(mine)==len(Cl) else "n/a")
print("open positions at 2026-10-02:",len(Op),"| mean open return %",round(Op.ret.mean(),2),"| winners",int((Op.ret>0).sum()))
def port(df):
    ev=[]
    for r in df.itertuples(): ev+= [(r.entry,0,r.Index),(r.exit,1,r.Index)]
    ev.sort(); cap=100000.; peak=cap; dd=0; notional={}
    for ts,k,i in ev:
        if k==0: notional[i]=cap*0.02
        else:
            cap+=notional[i]*df.ret[i]/100; peak=max(peak,cap); dd=min(dd,(cap/peak-1)*100)
    return round((cap/100000-1)*100,2),round(dd,2)
res={}
for name,s in {"whole_file":None,"entered_from_2024-01-01":"2024-01-01","entered_from_2025-10-10":"2025-10-10"}.items():
    a=Cl if s is None else Cl[Cl.entry>=np.datetime64(s)]; b=Op if s is None else Op[Op.entry>=np.datetime64(s)]
    both=pd.concat([a,b],ignore_index=True)
    w,l=both.ret[both.ret>0].sum(),-both.ret[both.ret<0].sum()
    res[name]=dict(closed=len(a),closed_only=port(a.reset_index(drop=True)),open_marked=len(b),with_open_marked=port(both),pf_with_open=round(w/l,2) if l else None,win_pct=round((both.ret>0).mean()*100,1))
    print(name,res[name])
json.dump(res,open("alpaca_bar_checked.json","w"),indent=1,default=str)
