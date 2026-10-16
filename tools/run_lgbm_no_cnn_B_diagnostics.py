import numpy as np, pandas as pd, json
P = pd.read_pickle("/home/claude/picksB.pkl")
def st(r):
    r=np.asarray(r)
    if len(r)==0: return dict(trades=0,mean_bp=0.0,pf=0.0,sum_pct=0.0)
    g,l=r[r>0].sum(),-r[r<0].sum()
    return dict(trades=len(r),mean_bp=round(r.mean()*1e4,1),pf=round(g/l,2) if l>0 else float('inf'),sum_pct=round(r.sum()*100,1))
out={"concentration":[], "regime":[]}
print("CONCENTRATION")
for k,f in P.groupby("fold"):
    by=f.groupby("symbol").net.agg(["sum","count"]).sort_values("sum",ascending=False)
    tot=f.net.sum()
    top3=list(by.index[:3])
    ex=f[~f.symbol.isin(top3)].net
    # time concentration: best single week share
    wk=f.groupby(f.timestamp.dt.to_period("W")).net.sum().sort_values(ascending=False)
    wk_share = wk.iloc[0]/tot if tot>0 else float('nan')
    d=dict(fold=k,symbols=int(f.symbol.nunique()),total=st(f.net),top3=top3,
           top3_trades=[int(by.loc[s,'count']) for s in top3],
           top3_share=round(by["sum"].iloc[:3].sum()/tot,2) if tot>0 else None,
           excl_top3=st(ex), best_week=str(wk.index[0]), best_week_share=round(float(wk_share),2) if tot>0 else None)
    out["concentration"].append(d)
    print(f"fold {k}: syms={d['symbols']} total {d['total']['mean_bp']}bp PF={d['total']['pf']} | top3 {top3} trades={d['top3_trades']} share={d['top3_share']} | "
          f"WITHOUT top3: n={d['excl_top3']['trades']} {d['excl_top3']['mean_bp']}bp PF={d['excl_top3']['pf']} | best week {d['best_week']} share={d['best_week_share']}")
print("\nREGIME GATE (primary: BTC above its 200-day MA; secondary: >50% of coins above their 200h MA)")
for k,f in P.groupby("fold"):
    on = f[f.btc_px_vs_ma200d>0].net; off=f[f.btc_px_vs_ma200d<=0].net
    br = f[f.mkt_above_ma200>0.5].net
    d=dict(fold=k,all=st(f.net),btc_bull=st(on),btc_bear=st(off),breadth_gate=st(br))
    out["regime"].append(d)
    print(f"fold {k}: all n={d['all']['trades']} {d['all']['mean_bp']}bp PF={d['all']['pf']} | BTC>200d n={d['btc_bull']['trades']} {d['btc_bull']['mean_bp']}bp PF={d['btc_bull']['pf']} | "
          f"BTC<200d n={d['btc_bear']['trades']} {d['btc_bear']['mean_bp']}bp PF={d['btc_bear']['pf']} | breadth>50% n={d['breadth_gate']['trades']} {d['breadth_gate']['mean_bp']}bp PF={d['breadth_gate']['pf']}")
g=[r['btc_bull'] for r in out['regime']]
print("\nPRIMARY GATE PASS (all folds sum>0, PF>1.3, n>=50):", all(x['sum_pct']>0 and x['pf']>1.3 and x['trades']>=50 for x in g))
json.dump(out,open("/home/claude/B_diagnostics.json","w"),indent=1,default=float)
