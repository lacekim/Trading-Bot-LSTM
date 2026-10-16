"""Golden cross (50/200 MA, 4h) entry/exit variants on the dry run's 6-asset universe.
Same costs (12bp/side), 15% allocation per asset, cash-constrained, -8% intrabar hard stop.
Data: local Kraken 1h feathers resampled to 4h (through 2026-01-05; Kraken API blocked here).
Variants are fixed up front (logic-driven, not a parameter search):
  LIVE      : what the running bot does -- enter whenever 50>200 and flat (re-enters right after a stop)
  VALIDATED : what was backtested -- enter only on a fresh cross; after a stop wait for the next cross
  FAST_EXIT : VALIDATED + also exit when 4h close < 200MA; re-enter when close back > 200MA while 50>200
  SLOPE     : VALIDATED + only take a cross if the 200MA is rising (vs 30 bars / 5 days ago)
  TRAIL     : VALIDATED + chandelier trail: exit if close < highest high since entry - 7x ATR14(4h) (~3x daily ATR)
  RSI_ENTRY     : golden cross active + RSI(14) dips below 40 and turns back up (buy the pullback, no cross entry)
  STOCH_ENTRY   : golden cross active + StochRSI(14,14,3,3) %K crosses back up through 20
  RSI_REENTRY   : VALIDATED, plus after a -8% stop re-enter on the RSI pullback turn if golden cross still active
  STOCH_REENTRY : same re-entry idea using the StochRSI turn
Pass rule (set before running): better return/max-drawdown than VALIDATED in BOTH halves of history."""
import sys, json
sys.path.insert(0, "/home/claude")
import numpy as np, pandas as pd
from featherlite import read_feather

BASE = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/"
ASSETS = ["BTC", "ETH", "XMR", "PAXG", "SOL", "TRX"]
COST, ALLOC, STOP, CAP0 = 0.0012, 0.15, -0.08, 100_000.0

data = {}
for a in ASSETS:
    h = read_feather(f"{BASE}{a}_USD-1h.feather").set_index("date")
    d = h.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    d["ma50"] = d.close.rolling(50).mean(); d["ma200"] = d.close.rolling(200).mean()
    d["gc"] = d.ma50 > d.ma200
    d["cross_up"] = d.gc & ~d.gc.shift(1, fill_value=False)
    d["ma200_rising"] = d.ma200 > d.ma200.shift(30)
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    # RSI(14) Wilder, StochRSI(14,14,3,3) -- standard settings, fixed before testing
    delta = d.close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn)
    lo, hi = d.rsi.rolling(14).min(), d.rsi.rolling(14).max()
    d["stk"] = ((d.rsi - lo) / (hi - lo) * 100).rolling(3).mean()
    d["rsi_turn"] = (d.rsi > 40) & (d.rsi.shift(1) <= 40)          # RSI pullback below 40 turning back up
    d["stoch_turn"] = (d.stk > 20) & (d.stk.shift(1) <= 20)        # StochRSI %K back up through 20
    data[a] = d.dropna(subset=["ma200"])

idx = sorted(set().union(*[d.index for d in data.values()]))


def run(variant, start=None, end=None):
    cap = CAP0; pos = {a: None for a in ASSETS}; armed = {a: False for a in ASSETS}
    trades, equity = [], []
    for ts in idx:
        if (start and ts < start) or (end and ts >= end):
            continue
        deployed = sum(p["notional"] for p in pos.values() if p)
        free = cap - deployed
        mtm = 0.0
        for a in ASSETS:
            d = data[a]
            if ts not in d.index:
                if pos[a]: mtm += pos[a]["notional"] * (pos[a]["last"] / pos[a]["entry"] - 1)
                continue
            r = d.loc[ts]; p = pos[a]
            if p:
                p["hh"] = max(p["hh"], r.high)
                exit_px, why = None, None
                stop_px = p["entry"] * (1 + STOP)
                if r.low <= stop_px:
                    exit_px, why = stop_px, "stop"
                elif not r.gc:
                    exit_px, why = r.close, "death_cross"
                elif variant == "FAST_EXIT" and r.close < r.ma200:
                    exit_px, why = r.close, "close<ma200"
                elif variant == "TRAIL" and r.close < p["hh"] - 7 * r.atr:
                    exit_px, why = r.close, "trail"
                if exit_px is not None:
                    ret = exit_px * (1 - COST) / p["entry"] - 1
                    cap += p["notional"] * ret; free += p["notional"]
                    trades.append(dict(asset=a, entry=p["t"], exit=ts, ret=ret, pnl=p["notional"] * ret, why=why))
                    pos[a] = None
                    # re-entry arming rules
                    if variant == "LIVE": armed[a] = True
                    elif variant in ("RSI_REENTRY", "STOCH_REENTRY") and why == "stop": armed[a] = "after_stop"
                    elif variant == "FAST_EXIT" and why == "close<ma200": armed[a] = "reclaim"
                    else: armed[a] = False
                else:
                    p["last"] = r.close
                    mtm += p["notional"] * (r.close / p["entry"] - 1)
                    continue
            # flat: entry logic
            want = False
            if variant == "LIVE":
                want = bool(r.gc)
            elif variant in ("RSI_ENTRY", "STOCH_ENTRY"):
                trig = r.rsi_turn if variant == "RSI_ENTRY" else r.stoch_turn
                want = bool(r.gc and trig)
            elif variant in ("RSI_REENTRY", "STOCH_REENTRY"):
                trig = r.rsi_turn if variant == "RSI_REENTRY" else r.stoch_turn
                want = bool(r.cross_up) or bool(armed[a] == "after_stop" and r.gc and trig)
            else:
                if r.cross_up:
                    want = True if variant != "SLOPE" else bool(r.ma200_rising)
                elif variant == "FAST_EXIT" and armed[a] == "reclaim" and r.gc and r.close > r.ma200:
                    want = True
            if not r.gc: armed[a] = False
            if want:
                notional = cap * ALLOC
                if notional <= free and notional > 100:
                    pos[a] = dict(entry=r.close * (1 + COST), notional=notional, t=ts, hh=r.high, last=r.close)
                    free -= notional; armed[a] = False
        equity.append((ts, cap + mtm))
    eq = pd.Series(dict(equity))
    T = pd.DataFrame(trades)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    tot = eq.iloc[-1] / CAP0 - 1
    cagr = (eq.iloc[-1] / CAP0) ** (1 / yrs) - 1
    g, l = T.pnl[T.pnl > 0].sum(), -T.pnl[T.pnl < 0].sum()
    ex3 = T.sort_values("pnl").iloc[:-3] if len(T) > 3 else T
    g3, l3 = ex3.pnl[ex3.pnl > 0].sum(), -ex3.pnl[ex3.pnl < 0].sum()
    return dict(variant=variant, trades=len(T), win_pct=round((T.pnl > 0).mean() * 100, 1),
                total_pct=round(tot * 100, 1), cagr_pct=round(cagr * 100, 2), max_dd_pct=round(dd * 100, 1),
                cagr_over_dd=round(cagr / abs(dd), 2) if dd < 0 else None, pf=round(g / l, 2),
                pf_ex_top3=round(g3 / l3, 2), stops=int((T.why == "stop").sum()),
                exits=T.why.value_counts().to_dict())


VARIANTS = ["LIVE", "VALIDATED", "RSI_ENTRY", "STOCH_ENTRY", "RSI_REENTRY", "STOCH_REENTRY"]
mid = idx[0] + (idx[-1] - idx[0]) / 2
res = {}
for period, (s, e) in {"full": (None, None), "first_half": (None, mid), "second_half": (mid, None)}.items():
    print(f"\n=== {period} {'' if not s else s.date()}..{'' if not e else e.date()}")
    res[period] = []
    for v in VARIANTS:
        r = run(v, s, e); res[period].append(r)
        print(f"{v:10s} trades={r['trades']:4d} win={r['win_pct']:5.1f}% total={r['total_pct']:7.1f}% CAGR={r['cagr_pct']:6.2f}% "
              f"maxDD={r['max_dd_pct']:6.1f}% CAGR/DD={r['cagr_over_dd']} PF={r['pf']} PFex3={r['pf_ex_top3']} stops={r['stops']}")
print(f"\nsplit at {mid.date()}; data {idx[0].date()}..{idx[-1].date()}")
for v in VARIANTS[2:]:
    ok = all(next(x for x in res[h] if x["variant"] == v)["cagr_over_dd"] > next(x for x in res[h] if x["variant"] == "VALIDATED")["cagr_over_dd"]
             for h in ("first_half", "second_half"))
    print(f"{v}: beats VALIDATED on CAGR/DD in both halves -> {ok}")
json.dump(res, open("/home/claude/gc_rsi_results.json", "w"), indent=1, default=str)
