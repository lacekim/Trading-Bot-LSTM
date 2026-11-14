import json, copy, pandas as pd
from pathlib import Path
R="/sessions/rcw-01evuvgej6pp73aaaw9sdbhm/mnt/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera"
src=open(R+"/tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py").read()
src=src.replace("from dotenv import load_dotenv\n","").replace('load_dotenv(PROJECT_ROOT / ".env")\n',"")
src=src.replace("PROJECT_ROOT = Path(__file__).resolve().parents[1]","PROJECT_ROOT = Path(R)")
src=src[:src.index("def fetch_4h_close")]
exec(src)
uni=_load_robust_daily_universe()
print("universe:",len(uni),"instruments; costs",ENTRY_COST,EXIT_COST,"alloc %",PER_TRADE_ALLOCATION_PCT)
d=pd.read_csv(R+"/runtime/venue_candles/alpaca_1d/alpaca_daily.csv.gz",parse_dates=["date"])
have=set(d.symbol.unique())
allt=[];rows=[];missing=[]
for u in uni:
    s=_alpaca_symbol(u["symbol"])
    if s not in have:
        missing.append((u["symbol"],s)); continue
    close=d[d.symbol==s].set_index("date")["close"].sort_index().dropna()
    tr=backtest_one(close)
    for t in tr: t["symbol"]=s
    rb=robustness(tr) or {}
    rows.append(dict(symbol=s,first=str(close.index[0].date()),last=str(close.index[-1].date()),**rb))
    allt+=tr
print("missing from local Alpaca file:",missing)
T=pd.DataFrame(allt); T.to_csv("alpaca_daily_gc_trades.csv",index=False)
pd.DataFrame(rows).to_csv("alpaca_daily_gc_per_symbol.csv",index=False)
out={"instruments_used":len(rows),"missing":missing,"data_last_day":str(d.date.max().date())}
def stats(tr):
    p=simulate_portfolio(copy.deepcopy(tr))
    r=pd.Series([t["net_ret_pct"] for t in tr])
    w,l=r[r>0].sum(),abs(r[r<0].sum())
    return dict(n_trades=p["n_trades"],total_return_pct=round(p["total_return_pct"],2),max_drawdown_pct=round(p["max_drawdown_pct"],2),pf=round(w/l,2) if l else None,win_pct=round((r>0).mean()*100,1) if len(r) else None)
for name,s in {"whole_file_2016_to_2026-10-02":None,"entered_from_2024-01-01":"2024-01-01","entered_from_2025-10-10":"2025-10-10"}.items():
    tr=[t for t in allt if s is None or t["entry_time"]>=pd.Timestamp(s)]
    out[name]=stats(tr); print(f"{name:32s}",out[name])
robust=sum(1 for r in rows if r.get("robust")); print("robust per their rule:",robust,"of",len(rows))
out["robust_count"]=robust
json.dump(out,open("alpaca_bar.json","w"),indent=1,default=str)
