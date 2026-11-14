"""SAT ch.14 crossover windows (100/400) inside the user's golden cross engine (tools/run_golden_cross_variants.py run()).
The ONLY difference from _claude_scratch/gc_bar_20261114/kraken_bar.py is the two window lengths: 50 -> 100 and 200 -> 400.
Same 4h Kraken bars, 6 assets, 12 bp per side, 15% allocation, -8% stop, same windows. Column names ma50/ma200 are kept
so run() is untouched; they hold the 100- and 400-bar averages here."""
import json, sys, os, numpy as np, pandas as pd
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
src=open(R+"/tools/run_golden_cross_variants.py").read()
src=src.replace('sys.path.insert(0, "/home/claude")\n','').replace('from featherlite import read_feather\n','')
src=src.replace('BASE = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/"','BASE = R+"/runtime/venue_candles_v2/kraken_1h/"')
src=src.replace('h = read_feather(f"{BASE}{a}_USD-1h.feather").set_index("date")','h = pd.read_csv(f"{BASE}{a}_1h.csv", parse_dates=["date"]).set_index("date")')
a='d["ma50"] = d.close.rolling(50).mean(); d["ma200"] = d.close.rolling(200).mean()'
assert a in src
if len(sys.argv)>1 and sys.argv[1]=="book": src=src.replace(a,'d["ma50"] = d.close.rolling(100).mean(); d["ma200"] = d.close.rolling(400).mean()')
tag=sys.argv[1] if len(sys.argv)>1 else "gc"
src=src.replace('    yrs = (eq.index[-1] - eq.index[0]).days / 365.25','    globals()["_LAST"]=(T.copy(),eq.copy())\n    yrs = (eq.index[-1] - eq.index[0]).days / 365.25')
src=src[:src.index('VARIANTS = [')]
exec(src)
out={}
W={"from_2024-01-01":(pd.Timestamp("2024-01-01"),None),"last_365d":(pd.Timestamp("2025-10-13"),pd.Timestamp("2026-10-13")),"whole_history":(None,None)}
for name,(s,e) in W.items():
    for v in ("LIVE","VALIDATED"):
        r=run(v,s,e); T,eq=_LAST
        r["independent"]=dict(total_pct=round((eq.iloc[-1]/100000-1)*100,1),max_dd_pct=round(((eq/eq.cummax())-1).min()*100,1),trades=len(T),pf=round(T.pnl[T.pnl>0].sum()/-T.pnl[T.pnl<0].sum(),2))
        out[f"{v}|{name}"]=r
        print(f"{tag:5s} {name:16s} {v:9s} trades={r['trades']:4d} total={r['total_pct']:9.1f}% maxDD={r['max_dd_pct']:6.1f}% PF={r['pf']} PFex3={r['pf_ex_top3']} win={r['win_pct']}%")
json.dump(out,open(f"kraken_{tag}_in_bar_engine.json","w"),indent=1,default=str)
