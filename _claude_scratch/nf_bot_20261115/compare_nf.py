"""Read book_bot_v2 (12 features) and the no-ffd copy (11) the same way (compare_option1.py method)."""
import json, glob, os, pandas as pd
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
runs={v:{"v2":R+f"/tools/book_study/book_bot_v2/{v}","nf":sorted(glob.glob(R+f"/runtime/book_bot_nf/{v}_*"))[-1]} for v in ("kraken","alpaca")}
def spans(d):
    e=pd.read_csv(d+"/equity_curves_walk_forward.csv"); out={}
    for sp,g in e.groupby("span"):
        eq=(1+g.ret).cumprod(); out[sp]=dict(gain_pct=round(float(eq.iloc[-1]-1)*100,2),max_dd_pct=round(float((eq/eq.cummax()-1).min())*100,2))
    return out
res={}
for v,pair in runs.items():
    res[v]={}
    for k,d in pair.items():
        m=json.load(open(d+"/meta.json")); s=spans(d)
        res[v][k]=dict(run=os.path.relpath(d,R),n_features=len(m["features"]),ffd_in_features="ffd" in m["features"],events=m["events"],wf_auc=round(m["walk_forward"]["auc"],4),cpcv_auc=round(m["cpcv"]["auc_mean"],4),otr=[m["otr"]["pt"],m["otr"]["sl"]],ch14_last365=round(m["ch14"]["gain_pct"],2),spans=s,seconds=m.get("seconds"))
        print(v,k,{x:res[v][k][x] for x in ("n_features","ffd_in_features","events","wf_auc","cpcv_auc","otr","ch14_last365","seconds")},s)
json.dump(res,open("nf_result.json","w"),indent=1,default=str)
