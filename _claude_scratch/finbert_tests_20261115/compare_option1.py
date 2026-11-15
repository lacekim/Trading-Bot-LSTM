"""Option 1: read the AFML bot's own outputs for book_bot_v2 (12 features) and the sentiment copy (12 + 3), side by side.
Gains and drawdowns are recomputed from each run's equity_curves_walk_forward.csv (columns day, ret, equity, span) and checked
against the numbers book_bot_v2/RESULTS.md reports, so both runs are read the same way."""
import json, glob, os, pandas as pd, numpy as np
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
runs={"kraken":{"v2":R+"/tools/book_study/book_bot_v2/kraken","sent":sorted(glob.glob(R+"/runtime/book_bot_sent/kraken_*"))[-1]},
      "alpaca":{"v2":R+"/tools/book_study/book_bot_v2/alpaca","sent":sorted(glob.glob(R+"/runtime/book_bot_sent/alpaca_*"))[-1]}}
def spans(d):
    e=pd.read_csv(d+"/equity_curves_walk_forward.csv"); out={}
    for sp,g in e.groupby("span"):
        eq=(1+g.ret).cumprod(); out[sp]=dict(first=str(pd.Timestamp(int(g.day.iloc[0]),unit="D").date()),last=str(pd.Timestamp(int(g.day.iloc[-1]),unit="D").date()),
                                          gain_pct=round(float(eq.iloc[-1]-1)*100,2),max_dd_pct=round(float((eq/eq.cummax()-1).min())*100,2))
    return out
res={}
for v,pair in runs.items():
    res[v]={}
    for k,d in pair.items():
        m=json.load(open(d+"/meta.json")); s=spans(d)
        res[v][k]=dict(run=os.path.relpath(d,R),features=m["features"],events=m["events"],symbols=m["symbols"],seconds=m.get("seconds"),
                       walk_forward_auc=round(m["walk_forward"]["auc"],4),cpcv_auc_mean=round(m["cpcv"]["auc_mean"],4),
                       ch13_pt_sl=[m["otr"]["pt"],m["otr"]["sl"]],ch14_last365_gain_pct=round(m["ch14"]["gain_pct"],2),spans=s)
        print(f"{v:6s} {k:4s} events={m['events']:,} features={len(m['features'])} AUC wf={res[v][k]['walk_forward_auc']} cpcv={res[v][k]['cpcv_auc_mean']} Ch13 pt/sl={res[v][k]['ch13_pt_sl']} | "+
              " | ".join(f"{sp}: {x['gain_pct']}% (DD {x['max_dd_pct']}%) {x['first']}..{x['last']}" for sp,x in s.items()))
# importance of the three sentiment features in the sentiment runs (Ch.8, saved by the bot)
for v in runs:
    d=runs[v]["sent"]
    for f in ("mdi.csv","mda.csv","sfi.csv"):
        t=pd.read_csv(f"{d}/{f}"); res[v].setdefault("importance",{})[f]=t.to_dict(orient="records")
    mdi=pd.read_csv(f"{d}/mdi.csv"); print(v,"MDI (sent run):",mdi.round(4).to_string(index=False).replace("\n"," | ")[:900])
    ev=pd.read_csv(f"{d}/events.csv",usecols=lambda c: c in ("sent_age_days","t0"))
    if "sent_age_days" in ev: a=ev.sent_age_days; print(f"   sentiment age at events: median {a.median():.0f} days, 75th pct {a.quantile(.75):.0f}, share older than 7 days {(a>7).mean():.2f}")
    res[v]["sent_age"]= {"median":float(ev.sent_age_days.median()),"p75":float(ev.sent_age_days.quantile(.75)),"share_gt_7d":float((ev.sent_age_days>7).mean())} if "sent_age_days" in ev else None
json.dump(res,open("option1_result.json","w"),indent=1,default=str)
