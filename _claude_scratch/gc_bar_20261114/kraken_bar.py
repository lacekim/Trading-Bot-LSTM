import json, sys, numpy as np, pandas as pd
R="/sessions/rcw-01evuvgej6pp73aaaw9sdbhm/mnt/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera"
src=open("gc_orig.py").read()
src=src.replace('sys.path.insert(0, "/home/claude")\n','').replace('from featherlite import read_feather\n','')
src=src.replace('BASE = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/"','BASE = R+"/runtime/venue_candles_v2/kraken_1h/"')
src=src.replace('h = read_feather(f"{BASE}{a}_USD-1h.feather").set_index("date")','h = pd.read_csv(f"{BASE}{a}_1h.csv", parse_dates=["date"]).set_index("date")')
# keep trades and equity for the independent check (no logic change)
src=src.replace('    yrs = (eq.index[-1] - eq.index[0]).days / 365.25','    globals()["_LAST"]=(T.copy(),eq.copy())\n    yrs = (eq.index[-1] - eq.index[0]).days / 365.25')
src=src[:src.index('VARIANTS = [')]
exec(src)
print("data", idx[0], "->", idx[-1])
out={}
W={"whole_history":(None,None),"from_2024-01-01":(pd.Timestamp("2024-01-01"),None),"last_365d_2025-10-13_to_2026-10-12":(pd.Timestamp("2025-10-13"),pd.Timestamp("2026-10-13"))}
for name,(s,e) in W.items():
    for v in ("LIVE","VALIDATED"):
        r=run(v,s,e); T,eq=_LAST
        T.to_csv(f"kraken_{v}_{name}_trades.csv",index=False); eq.to_csv(f"kraken_{v}_{name}_equity.csv")
        out[f"{v}|{name}"]=r
        print(f"{name:38s} {v:9s} trades={r['trades']:4d} total={r['total_pct']:9.1f}% CAGR={r['cagr_pct']:6.2f}% maxDD={r['max_dd_pct']:6.1f}% PF={r['pf']} PFex3={r['pf_ex_top3']} win={r['win_pct']}% eq {eq.index[0].date()}..{eq.index[-1].date()}")
json.dump(out,open("kraken_bar.json","w"),indent=1,default=str)
