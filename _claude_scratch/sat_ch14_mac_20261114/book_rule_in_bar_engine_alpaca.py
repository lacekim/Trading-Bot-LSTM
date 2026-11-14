"""SAT ch.14 crossover windows (100/400) inside the user's Alpaca golden cross functions
(tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py: compute_trend_state, backtest_one, simulate_portfolio).
The ONLY difference between the two runs is rolling(50)/rolling(200) -> rolling(100)/rolling(400) in compute_trend_state.
Daily closes from runtime/venue_candles/alpaca_1d, robust-on-daily universe, 12 bp per side, 2% per trade.
Open positions are marked at the last close (an addition: the engine counts closed trades only); both are reported."""
import json, sys, os, copy, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.simplefilter("ignore")
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
src=open(R+"/tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py").read()
src=src.replace("from dotenv import load_dotenv\n","").replace('load_dotenv(PROJECT_ROOT / ".env")\n',"")
src=src.replace("PROJECT_ROOT = Path(__file__).resolve().parents[1]","PROJECT_ROOT = Path(R)")
src=src[:src.index("def fetch_4h_close")]
tag=sys.argv[1]
if tag=="book":
    assert "close.rolling(50).mean()" in src and "close.rolling(200).mean()" in src
    src=src.replace("close.rolling(50).mean()","close.rolling(100).mean()").replace("close.rolling(200).mean()","close.rolling(400).mean()")
exec(src)
d=pd.read_csv(R+"/runtime/venue_candles/alpaca_1d/alpaca_daily.csv.gz",parse_dates=["date"]); have=set(d.symbol.unique())
closed=[]; opens=[]
for u in _load_robust_daily_universe():
    s=_alpaca_symbol(u["symbol"])
    if s not in have: continue
    close=d[d.symbol==s].set_index("date")["close"].sort_index().dropna()
    tr=backtest_one(close); closed+=tr
    st=compute_trend_state(close).dropna(subset=["ma200"])
    # open position at the end = last cross_up after the last cross_down
    ups=st.index[st.cross_up]; downs=st.index[st.cross_down]
    if len(ups) and (not len(downs) or ups[-1]>downs[-1]):
        e=st.close[ups[-1]]*(1+ENTRY_COST); x=close.iloc[-1]*(1-EXIT_COST)
        opens.append({"entry_time":ups[-1],"exit_time":close.index[-1],"net_ret_pct":(x/e-1)*100,"open":True})
out={}
for name,s in {"entered_from_2024-01-01":"2024-01-01","entered_from_2025-10-10":"2025-10-10","whole_file":None}.items():
    c=[t for t in closed if s is None or t["entry_time"]>=pd.Timestamp(s)]; o=[t for t in opens if s is None or t["entry_time"]>=pd.Timestamp(s)]
    pc=simulate_portfolio(copy.deepcopy(c)); pb=simulate_portfolio(copy.deepcopy(c+o))
    r=pd.Series([t["net_ret_pct"] for t in c+o]); w,l=r[r>0].sum(),abs(r[r<0].sum())
    out[name]=dict(closed_trades=len(c),closed_only_return_pct=round(pc["total_return_pct"],2),open_positions=len(o),with_open_marked_return_pct=round(pb["total_return_pct"],2),with_open_marked_max_dd_pct=round(pb["max_drawdown_pct"],2),pf_with_open=round(float(w/l),2) if l else None,win_pct=round(float((r>0).mean()*100),1) if len(r) else None)
    print(f"{tag:5s} {name:26s}",out[name])
json.dump(out,open(f"alpaca_{tag}_in_bar_engine.json","w"),indent=1,default=str)
