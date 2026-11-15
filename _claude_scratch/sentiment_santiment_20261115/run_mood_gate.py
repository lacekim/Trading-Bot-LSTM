"""Usage: python3 run_mood_gate.py off|gate  -- the user's run(); one changed line: 'if want:' -> 'if want and bool(r.reg_ok):'"""
import json, sys, os, numpy as np, pandas as pd
import mood_gate as G
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
mode=sys.argv[1]; START=pd.Timestamp("2025-12-16"); END=pd.Timestamp("2026-10-13")
EU,_=G.euphoric(G.mood_4h(R))
def _reg(d):
    if mode=="off": return True
    return (~EU.reindex(d.index).fillna(False).astype(bool)).values
src=open(R+"/tools/run_golden_cross_variants.py").read()
src=src.replace('sys.path.insert(0, "/home/claude")\n','').replace('from featherlite import read_feather\n','')
src=src.replace('BASE = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/"','BASE = R+"/runtime/venue_candles_v2/kraken_1h/"')
src=src.replace('h = read_feather(f"{BASE}{a}_USD-1h.feather").set_index("date")','h = pd.read_csv(f"{BASE}{a}_1h.csv", parse_dates=["date"]).set_index("date")')
k='    data[a] = d.dropna(subset=["ma200"])'; assert src.count(k)==1
src=src.replace(k,'    d["reg_ok"] = _reg(d)\n'+k)
k='            if want:'; assert src.count(k)==1
src=src.replace(k,'            if want and bool(r.reg_ok):')
src=src.replace('    yrs = (eq.index[-1] - eq.index[0]).days / 365.25','    globals()["_LAST"]=(T.copy(),eq.copy())\n    yrs = (eq.index[-1] - eq.index[0]).days / 365.25')
src=src[:src.index('VARIANTS = [')]
exec(src)
w=EU[(EU.index>=START)&(EU.index<END)]
out={"mode":mode,"window":[str(START),str(END)],"euphoric_share_of_bars":round(float(w.mean()),3),"first_gate_bar":str(EU[EU].index.min()) if EU.any() else None,"rows":{}}
for v in ("LIVE","VALIDATED"):
    r=run(v,START,END); T,eq=_LAST
    T.to_csv(f"mood_{mode}_{v}_trades.csv",index=False); eq.to_csv(f"mood_{mode}_{v}_equity.csv")
    r["ret_over_dd"]=round(r["total_pct"]/abs(r["max_dd_pct"]),2) if r["max_dd_pct"]<0 else None
    out["rows"][v]=r
    print(f"{mode:4s} {v:9s} trades={r['trades']:3d} total={r['total_pct']:6.1f}% maxDD={r['max_dd_pct']:6.1f}% ret/DD={r['ret_over_dd']} PF={r['pf']} win={r['win_pct']}%")
print("  euphoric share of bars in window:",out["euphoric_share_of_bars"],"| first euphoric bar:",out["first_gate_bar"])
json.dump(out,open(f"mood_result_{mode}.json","w"),indent=1,default=str)
