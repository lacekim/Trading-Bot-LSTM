"""The book's HMM regime filter on the user's golden cross, Alpaca. Usage: python3 regime_filter_alpaca.py off|hmm
Engine: compute_trend_state / backtest_one / simulate_portfolio from tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py.
Two changed lines in backtest_one: the state table gets a reg_ok column, and the entry line
'if pos is None and row["cross_up"]:' becomes 'if pos is None and row["cross_up"] and row["reg_ok"]:'.
Exits are untouched. A blocked entry waits for the next cross (the book's behaviour). 'off' must reproduce the bar.
HMM: one per instrument, fitted on that instrument's daily returns BEFORE 2024-01-01 only.
Open positions are marked at the last close (an addition; the engine counts closed trades only). Both are reported."""
import json, sys, os, copy, warnings, numpy as np, pandas as pd
from pathlib import Path
import hmm_np as H
warnings.simplefilter("ignore")
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
mode=sys.argv[1]; CUT=pd.Timestamp("2024-01-01"); REG=None; OPEN=[]
src=open(R+"/tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py").read()
src=src.replace("from dotenv import load_dotenv\n","").replace('load_dotenv(PROJECT_ROOT / ".env")\n',"")
src=src.replace("PROJECT_ROOT = Path(__file__).resolve().parents[1]","PROJECT_ROOT = Path(R)")
src=src[:src.index("def fetch_4h_close")]
k="    state = compute_trend_state(close)\n"; assert src.count(k)==1
src=src.replace(k,k+'    state["reg_ok"] = True if REG is None else REG.reindex(state.index).fillna(False).astype(bool).values\n')
k='        if pos is None and row["cross_up"]:'; assert src.count(k)==1
src=src.replace(k,'        if pos is None and row["cross_up"] and row["reg_ok"]:')
# record the position still open at the end of the data (not a rule change: the engine simply drops it)
k="    return trades\n"; assert src.count(k)==1
src=src.replace(k,'    if pos is not None:\n        OPEN.append({"entry_time": pos["entry_time"], "exit_time": state.index[-1], "net_ret_pct": (state["close"].iloc[-1] * (1 - EXIT_COST) / (pos["entry_price"] * (1 + ENTRY_COST)) - 1) * 100})\n'+k)
exec(src)
d=pd.read_csv(R+"/runtime/venue_candles/alpaca_1d/alpaca_daily.csv.gz",parse_dates=["date"]); have=set(d.symbol.unique())
closed=[]; models={}; shares=[]
for u in _load_robust_daily_universe():
    s=_alpaca_symbol(u["symbol"])
    if s not in have: continue
    close=d[d.symbol==s].set_index("date")["close"].sort_index().dropna()
    if mode=="hmm":
        ret=close.pct_change().dropna(); m=H.fit(ret[ret.index<CUT].values); calm=H.calm_state(m)
        REG=pd.Series(H.last_state_online(ret.values,m)==calm,index=ret.index)
        models[s]=dict(train_days=int((ret.index<CUT).sum()),iters=m["iters"],sd_calm=float(np.sqrt(m["var"][calm])),sd_wild=float(np.sqrt(m["var"][1-calm])),calm_share_from_2024=round(float(REG[REG.index>=CUT].mean()),4))
        shares.append(models[s]["calm_share_from_2024"])
    tr=backtest_one(close)
    for t in tr: t["symbol"]=s
    closed+=tr
pd.DataFrame(closed).to_csv(f"alpaca_{mode}_closed_trades.csv",index=False); pd.DataFrame(OPEN).to_csv(f"alpaca_{mode}_open_positions.csv",index=False)
out={"mode":mode,"instruments":len(set(t["symbol"] for t in closed)),"models":models,"rows":{}}
for name,s in {"entered_from_2024-01-01":"2024-01-01","entered_from_2025-10-10":"2025-10-10"}.items():
    c=[t for t in closed if t["entry_time"]>=pd.Timestamp(s)]; o=[t for t in OPEN if t["entry_time"]>=pd.Timestamp(s)]
    pc=simulate_portfolio(copy.deepcopy(c)); pb=simulate_portfolio(copy.deepcopy(c+o))
    r=pd.Series([t["net_ret_pct"] for t in c+o]); w,l=r[r>0].sum(),abs(r[r<0].sum())
    out["rows"][name]=dict(closed_trades=len(c),closed_only_return_pct=round(float(pc["total_return_pct"]),2),open_positions=len(o),with_open_marked_return_pct=round(float(pb["total_return_pct"]),2),with_open_marked_max_dd_pct=round(float(pb["max_drawdown_pct"]),2),pf_with_open=round(float(w/l),2) if l else None,win_pct=round(float((r>0).mean()*100),1) if len(r) else None,mean_trade_pct=round(float(r.mean()),2) if len(r) else None)
    print(f"{mode:4s} {name:26s}",out["rows"][name])
if shares: print("  calm share of days from 2024 across instruments: min %.2f median %.2f max %.2f"%(min(shares),float(np.median(shares)),max(shares)))
json.dump(out,open(f"alpaca_regime_{mode}.json","w"),indent=1,default=str)
