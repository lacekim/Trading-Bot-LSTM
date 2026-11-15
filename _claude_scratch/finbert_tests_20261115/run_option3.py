"""Usage: python3 run_option3.py   -- golden cross and option 3 on both venues, user's engines, only the signal replaced."""
import json, os, copy, warnings, numpy as np, pandas as pd
from pathlib import Path
import sentdex_state as S
warnings.simplefilter("ignore")
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
OUT={"kraken":{},"alpaca":{}}

# ---------- Kraken: run() from tools/run_golden_cross_variants.py ----------
def kraken(mode):
    MODE=mode
    def _sig(a, d):
        if MODE=="gc": return
        st=S.on_bars(S.daily_state(R+"/runtime/llm_news/kraken/daily_sent.csv", a), d.index)
        d["gc"]=st.values; d["cross_up"]=d.gc & ~d.gc.shift(1, fill_value=False)
    src=open(R+"/tools/run_golden_cross_variants.py").read()
    src=src.replace('sys.path.insert(0, "/home/claude")\n','').replace('from featherlite import read_feather\n','')
    src=src.replace('BASE = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/"','BASE = R+"/runtime/venue_candles_v2/kraken_1h/"')
    src=src.replace('h = read_feather(f"{BASE}{a}_USD-1h.feather").set_index("date")','h = pd.read_csv(f"{BASE}{a}_1h.csv", parse_dates=["date"]).set_index("date")')
    k='    data[a] = d.dropna(subset=["ma200"])'; assert src.count(k)==1
    src=src.replace(k,'    _sig(a, d)\n'+k)
    src=src.replace('    yrs = (eq.index[-1] - eq.index[0]).days / 365.25','    globals()["_LAST"]=(T.copy(),eq.copy())\n    yrs = (eq.index[-1] - eq.index[0]).days / 365.25')
    src=src[:src.index('VARIANTS = [')]
    g={"R":R,"_sig":_sig,"__name__":"k"}; exec(src,g)
    for name,(s,e) in {"from_2024-01-01":("2024-01-01",None),"last_365d":("2025-10-13","2026-10-13")}.items():
        for v in ("LIVE","VALIDATED"):
            r=g["run"](v,pd.Timestamp(s),pd.Timestamp(e) if e else None); T,eq=g["_LAST"]
            T.to_csv(f"k_{mode}_{v}_{name}_trades.csv",index=False); eq.to_csv(f"k_{mode}_{v}_{name}_equity.csv")
            r["ret_over_dd"]=round(r["total_pct"]/abs(r["max_dd_pct"]),2) if r["max_dd_pct"]<0 else None
            OUT["kraken"][f"{mode}|{v}|{name}"]=r
            print(f"kraken {mode:4s} {v:9s} {name:16s} trades={r['trades']:4d} total={r['total_pct']:7.1f}% maxDD={r['max_dd_pct']:6.1f}% ret/DD={r['ret_over_dd']} PF={r['pf']}")

# ---------- Alpaca: functions from tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py ----------
def alpaca(mode):
    src=open(R+"/tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py").read()
    src=src.replace("from dotenv import load_dotenv\n","").replace('load_dotenv(PROJECT_ROOT / ".env")\n',"")
    src=src.replace("PROJECT_ROOT = Path(__file__).resolve().parents[1]","PROJECT_ROOT = Path(R)")
    src=src[:src.index("def fetch_4h_close")]
    k="    state = compute_trend_state(close)\n"; assert src.count(k)==1
    src=src.replace(k,k+"    state = _SIG(state)\n")
    k="    return trades\n"; assert src.count(k)==1
    src=src.replace(k,'    if pos is not None:\n        OPEN.append({"entry_time": pos["entry_time"], "exit_time": state.index[-1], "net_ret_pct": (state["close"].iloc[-1] * (1 - EXIT_COST) / (pos["entry_price"] * (1 + ENTRY_COST)) - 1) * 100})\n'+k)
    OPEN=[]; cur={"sym":None}
    def _SIG(state):
        if mode=="gc": return state
        on=S.on_bars(S.daily_state(R+"/runtime/llm_news/alpaca/daily_sent.csv", cur["sym"]), state.index)
        prev=on.shift(1, fill_value=False)
        state=state.copy(); state["cross_up"]=(on & ~prev).values; state["cross_down"]=(~on & prev).values
        return state
    g={"R":R,"Path":Path,"_SIG":_SIG,"OPEN":OPEN,"__name__":"a"}; exec(src,g)
    d=pd.read_csv(R+"/runtime/venue_candles/alpaca_1d/alpaca_daily.csv.gz",parse_dates=["date"]); have=set(d.symbol.unique())
    closed=[]; missing_sent=[]
    sent_tickers=set(pd.read_csv(R+"/runtime/llm_news/alpaca/daily_sent.csv",usecols=["ticker"]).ticker.unique())
    for u in g["_load_robust_daily_universe"]():
        s=g["_alpaca_symbol"](u["symbol"])
        if s not in have: continue
        if s not in sent_tickers: missing_sent.append(s)
        cur["sym"]=s
        close=d[d.symbol==s].set_index("date")["close"].sort_index().dropna()
        tr=g["backtest_one"](close)
        for t in tr: t["symbol"]=s
        closed+=tr
    for o in OPEN: pass
    pd.DataFrame(closed).to_csv(f"a_{mode}_closed.csv",index=False); pd.DataFrame(OPEN).to_csv(f"a_{mode}_open.csv",index=False)
    for name,s in {"entered_from_2024-01-01":"2024-01-01","entered_from_2025-10-10":"2025-10-10"}.items():
        c=[t for t in closed if t["entry_time"]>=pd.Timestamp(s)]; o=[t for t in OPEN if t["entry_time"]>=pd.Timestamp(s)]
        pb=g["simulate_portfolio"](copy.deepcopy(c+o)); pc=g["simulate_portfolio"](copy.deepcopy(c))
        r=dict(closed=len(c),open=len(o),return_pct=round(float(pb["total_return_pct"]),2),max_dd_pct=round(float(pb["max_drawdown_pct"]),2),closed_only_return_pct=round(float(pc["total_return_pct"]),2))
        r["ret_over_dd"]=round(r["return_pct"]/abs(r["max_dd_pct"]),2) if r["max_dd_pct"]<0 else None
        OUT["alpaca"][f"{mode}|{name}"]=r
        print(f"alpaca {mode:4s} {name:26s} {r}")
    if mode=="sent": OUT["alpaca"]["instruments_without_sentiment"]=missing_sent; print("alpaca instruments without any sentiment:",missing_sent)

for m in ("gc","sent"): kraken(m)
for m in ("gc","sent"): alpaca(m)
json.dump(OUT,open("option3_result.json","w"),indent=1,default=str)
