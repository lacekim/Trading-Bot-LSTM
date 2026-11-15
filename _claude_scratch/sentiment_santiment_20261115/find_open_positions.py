"""Which positions are still open at the end of the window, with and without the gate (VALIDATED and LIVE).
Re-runs the same engine with a hook that records the open positions after the last bar; no rule changes."""
import os, sys, pandas as pd
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
import mood_gate as G
EU,_=G.euphoric(G.mood_4h(R)); START=pd.Timestamp("2025-12-16"); END=pd.Timestamp("2026-10-13")
for mode in ("off","gate"):
    def _reg(d): return True if mode=="off" else (~EU.reindex(d.index).fillna(False).astype(bool)).values
    src=open(R+"/tools/run_golden_cross_variants.py").read()
    src=src.replace('sys.path.insert(0, "/home/claude")\n','').replace('from featherlite import read_feather\n','')
    src=src.replace('BASE = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/"','BASE = R+"/runtime/venue_candles_v2/kraken_1h/"')
    src=src.replace('h = read_feather(f"{BASE}{a}_USD-1h.feather").set_index("date")','h = pd.read_csv(f"{BASE}{a}_1h.csv", parse_dates=["date"]).set_index("date")')
    src=src.replace('    data[a] = d.dropna(subset=["ma200"])','    d["reg_ok"] = _reg(d)\n    data[a] = d.dropna(subset=["ma200"])')
    src=src.replace('            if want:','            if want and bool(r.reg_ok):')
    src=src.replace('    eq = pd.Series(dict(equity))','    globals()["_OPEN"]={a:p for a,p in pos.items() if p}\n    eq = pd.Series(dict(equity))')
    src=src[:src.index('VARIANTS = [')]
    exec(src)
    for v in ("LIVE","VALIDATED"):
        run(v,START,END)
        for a,p in _OPEN.items():
            last=data[a].close[data[a].index<END].iloc[-1]
            print(f"{mode:4s} {v:9s} open {a:5s} entered {p['t']} notional {p['notional']:.0f} unrealised {(last/p['entry']-1)*100:+.1f}% (= {p['notional']*(last/p['entry']-1):+.0f})")
    # the euphoria flag at every golden-cross fresh cross in the window
    if mode=="off":
        print("fresh golden crosses in the window and the euphoria flag at that bar:")
        for a,d in data.items():
            for ts in d.index[d.cross_up & (d.index>=START) & (d.index<END)]:
                print(f"   {a:5s} {ts}  euphoric={bool(EU.get(ts,False))}")
