"""The book's HMM regime filter on the user's golden cross, Kraken. Usage: python3 regime_filter_kraken.py off|hmm
Engine: run() from tools/run_golden_cross_variants.py, unchanged except ONE line: 'if want:' -> 'if want and bool(r.reg_ok):'
(a new entry needs the calm regime; exits are untouched). 'off' sets reg_ok True everywhere and must reproduce the bar.
HMM: one per asset, fitted on that asset's 4h returns BEFORE 2024-01-01 only; regime at bar t uses returns up to bar t.
Same 4h Kraken bars (runtime/venue_candles_v2/kraken_1h), 6 assets, 12 bp per side, 15% allocation, -8% stop."""
import json, sys, os, numpy as np, pandas as pd
import hmm_np as H
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
mode=sys.argv[1]; CUT=pd.Timestamp("2024-01-01"); MODELS={}; SHARE={}
def _reg(a, d):
    if mode=="off": return True
    ret=d.close.pct_change().dropna()
    m=H.fit(ret[ret.index<CUT].values); calm=H.calm_state(m)
    st=pd.Series(H.last_state_online(ret.values,m)==calm,index=ret.index)
    MODELS[a]=dict(train_bars=int((ret.index<CUT).sum()),iters=m["iters"],loglik=m["loglik"],sd_calm=float(np.sqrt(m["var"][calm])),sd_wild=float(np.sqrt(m["var"][1-calm])),
                   stay_calm=m["A"][calm][calm],stay_wild=m["A"][1-calm][1-calm])
    oos=st[st.index>=CUT]; SHARE[a]=round(float(oos.mean()),4)
    return st.reindex(d.index).fillna(False)
src=open(R+"/tools/run_golden_cross_variants.py").read()
src=src.replace('sys.path.insert(0, "/home/claude")\n','').replace('from featherlite import read_feather\n','')
src=src.replace('BASE = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/"','BASE = R+"/runtime/venue_candles_v2/kraken_1h/"')
src=src.replace('h = read_feather(f"{BASE}{a}_USD-1h.feather").set_index("date")','h = pd.read_csv(f"{BASE}{a}_1h.csv", parse_dates=["date"]).set_index("date")')
k='    data[a] = d.dropna(subset=["ma200"])'; assert src.count(k)==1
src=src.replace(k,'    d["reg_ok"] = _reg(a, d)\n'+k)
k='            if want:'; assert src.count(k)==1
src=src.replace(k,'            if want and bool(r.reg_ok):')
src=src.replace('    yrs = (eq.index[-1] - eq.index[0]).days / 365.25','    globals()["_LAST"]=(T.copy(),eq.copy())\n    yrs = (eq.index[-1] - eq.index[0]).days / 365.25')
src=src[:src.index('VARIANTS = [')]
exec(src)
out={"mode":mode,"models":MODELS,"share_of_bars_calm_from_2024":SHARE,"rows":{}}
W={"from_2024-01-01":(CUT,None),"last_365d":(pd.Timestamp("2025-10-13"),pd.Timestamp("2026-10-13"))}
for name,(s,e) in W.items():
    for v in ("LIVE","VALIDATED"):
        r=run(v,s,e); T,eq=_LAST
        T.to_csv(f"kraken_{mode}_{v}_{name}_trades.csv",index=False); eq.to_csv(f"kraken_{mode}_{v}_{name}_equity.csv")
        ret=eq.pct_change().dropna(); r["sharpe_4h_annualised"]=round(float(np.sqrt(6*365)*ret.mean()/ret.std()),2)
        out["rows"][f"{v}|{name}"]=r
        print(f"{mode:4s} {name:16s} {v:9s} trades={r['trades']:4d} total={r['total_pct']:7.1f}% maxDD={r['max_dd_pct']:6.1f}% ret/DD={round(r['total_pct']/abs(r['max_dd_pct']),2)} PF={r['pf']} PFex3={r['pf_ex_top3']} win={r['win_pct']}% sharpe={r['sharpe_4h_annualised']}")
if mode=="hmm":
    for a,m in MODELS.items(): print(f"  {a}: train bars {m['train_bars']}, iters {m['iters']}, sd calm {m['sd_calm']:.4f} wild {m['sd_wild']:.4f}, stay calm {m['stay_calm']:.3f} wild {m['stay_wild']:.3f}, calm share from 2024 {SHARE[a]}")
json.dump(out,open(f"kraken_regime_{mode}.json","w"),indent=1,default=str)
