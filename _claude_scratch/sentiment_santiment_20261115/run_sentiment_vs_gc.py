"""Usage: python3 run_sentiment_vs_gc.py gc|sent   -- the user's run() with only the signal column replaced for 'sent'."""
import json, sys, os, numpy as np, pandas as pd
import sentiment_state as S
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
mode=sys.argv[1]; INFO={}
def _sig(a, d):
    if mode=="gc": return
    st,pct=S.state(S.sentiment_4h(f"{R}/runtime/santiment/{a}_social_1h.csv"), d.index)
    d["gc"]=st.values; d["cross_up"]=d.gc & ~d.gc.shift(1, fill_value=False)
    w=d.index>=START
    INFO[a]=dict(share_on=round(float(d.gc[w].mean()),3), entries=int(d.cross_up[w].sum()))
src=open(R+"/tools/run_golden_cross_variants.py").read()
src=src.replace('sys.path.insert(0, "/home/claude")\n','').replace('from featherlite import read_feather\n','')
src=src.replace('BASE = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/"','BASE = R+"/runtime/venue_candles_v2/kraken_1h/"')
src=src.replace('h = read_feather(f"{BASE}{a}_USD-1h.feather").set_index("date")','h = pd.read_csv(f"{BASE}{a}_1h.csv", parse_dates=["date"]).set_index("date")')
k='    data[a] = d.dropna(subset=["ma200"])'; assert src.count(k)==1
src=src.replace(k,'    _sig(a, d)\n'+k)
src=src.replace('    yrs = (eq.index[-1] - eq.index[0]).days / 365.25','    globals()["_LAST"]=(T.copy(),eq.copy())\n    yrs = (eq.index[-1] - eq.index[0]).days / 365.25')
src=src[:src.index('VARIANTS = [')]
START=pd.Timestamp("2025-12-16"); END=pd.Timestamp("2026-10-13")
exec(src)
out={"mode":mode,"window":[str(START),str(END)],"signal_info":INFO,"rows":{}}
for v in ("LIVE","VALIDATED"):
    r=run(v,START,END); T,eq=_LAST
    T.to_csv(f"{mode}_{v}_trades.csv",index=False); eq.to_csv(f"{mode}_{v}_equity.csv")
    r["ret_over_dd"]=round(r["total_pct"]/abs(r["max_dd_pct"]),2) if r["max_dd_pct"]<0 else None
    r["per_asset_pnl"]={k:round(float(x),0) for k,x in T.groupby("asset").pnl.sum().items()} if len(T) else {}
    out["rows"][v]=r
    print(f"{mode:4s} {v:9s} trades={r['trades']:3d} total={r['total_pct']:6.1f}% maxDD={r['max_dd_pct']:6.1f}% ret/DD={r['ret_over_dd']} PF={r['pf']} win={r['win_pct']}% | pnl by asset {r['per_asset_pnl']}")
if INFO: print("  sentiment state: share of bars ON and entries since start:",INFO)
json.dump(out,open(f"result_{mode}.json","w"),indent=1,default=str)
