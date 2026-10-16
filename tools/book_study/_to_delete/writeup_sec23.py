"""Section 2.3 (Equations 1-15): imbalance and run bars, built from the book's equations on our finest rows.
Rows: Kraken ETH and Alpaca AAPL hourly reference rows (runtime/dollar_bars/*/ref_1h). b_t = Equation 1 on row closes; v_t = row volume (volume bars) or close*volume (dollar bars).
Where the book is silent: E0[T] start (24 rows), EWMA over bars with N=100 (alpha=2/101), warm-up on the first 24 rows, first b_t = +1 (assumptions, stated in the entries)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import writeup_core as W

REF = {"Kraken": W.ROOT / "runtime/dollar_bars/kraken/ref_1h/ETH.csv", "Alpaca": W.ROOT / "runtime/dollar_bars/alpaca/ref_1h/AAPL.csv"} if hasattr(W, "ROOT") else None
if REF is None:
    root = Path(__file__).resolve().parents[2]
    REF = {"Kraken": root / "runtime/dollar_bars/kraken/ref_1h/ETH.csv", "Alpaca": root / "runtime/dollar_bars/alpaca/ref_1h/AAPL.csv"}
ALPHA, T0, WARM = 2 / 101, 24.0, 24
KINDS = {"TIB": ("tick", "imbalance", None), "VIB": ("volume", "imbalance", "v"), "DIB": ("dollar", "imbalance", "d"),
         "TRB": ("tick", "run", None), "VRB": ("volume", "run", "v"), "DRB": ("dollar", "run", "d")}


def tick_rule(close):
    d = np.diff(close, prepend=close[0])
    b = np.empty(len(close)); prev = 1.0                       # first b_t: +1 (the book is silent)
    for i, x in enumerate(d):
        if x != 0: prev = 1.0 if x > 0 else -1.0
        b[i] = prev                                            # Equation 1
    return b


def make_bars(b, v, mode, kind):
    """Equations 2-3 / 4-9 (imbalance) and 10-12 / 13-15 (runs). Returns per-bar records."""
    n = len(b)
    w = np.ones(n) if v is None else v
    # warm-up estimates from the first WARM rows
    def est(sl):
        bb, ww = b[sl], w[sl]
        pb = (bb > 0).mean()
        vp = ww[bb > 0].mean() if (bb > 0).any() else 0.0
        vm = ww[bb < 0].mean() if (bb < 0).any() else 0.0
        return pb, vp, vm, (bb * ww).mean()
    pb, vp, vm, imb = est(slice(0, WARM))
    ET = T0
    out = []; s = WARM
    while s < n:
        # threshold from current expectations (Equations 3, 9, 12, 15)
        if mode == "imbalance":
            thr = ET * abs(2 * pb * vp - (pb * vp + (1 - pb) * vm)) if v is not None else ET * abs(2 * pb - 1)
        else:
            thr = ET * (max(pb, 1 - pb) if v is None else max(pb * vp, (1 - pb) * vm))
        up = dn = th = 0.0; e = None
        for i in range(s, n):
            x = b[i] * w[i]
            th += x
            if x > 0: up += x
            else: dn -= x
            theta = abs(th) if mode == "imbalance" else max(up, dn)
            if theta >= thr and i >= s:
                e = i; break
        if e is None: break
        sl = slice(s, e + 1)
        bb, ww = b[sl], w[sl]
        out.append(dict(start=s, end=e, T=e - s + 1, thr=thr, theta=theta, ET=ET, pb=pb, vp=vp, vm=vm))
        ET = (1 - ALPHA) * ET + ALPHA * (e - s + 1)
        p2 = (bb > 0).mean(); v2p = ww[bb > 0].mean() if (bb > 0).any() else vp; v2m = ww[bb < 0].mean() if (bb < 0).any() else vm
        pb, vp, vm = (1 - ALPHA) * pb + ALPHA * p2, (1 - ALPHA) * vp + ALPHA * v2p, (1 - ALPHA) * vm + ALPHA * v2m
        s = e + 1
    return pd.DataFrame(out)


def run():
    res = {}
    for venue, f in REF.items():
        d = pd.read_csv(f, parse_dates=["date"])
        close = d["close"].to_numpy(float); vol = d["volume"].to_numpy(float).clip(min=0)
        b = tick_rule(close)
        rows = len(d)
        for k, (unit, mode, vv) in KINDS.items():
            v = None if vv is None else (vol if vv == "v" else vol * close)
            bars = make_bars(b, v, mode, k)
            # independent check: replay each bar from its stored threshold: theta must reach thr on the last row and not before
            ok = 0
            w = np.ones(rows) if v is None else v
            for r in bars.itertuples():
                x = (b * w)[r.start:r.end + 1]
                cs = np.cumsum(x)
                if mode == "imbalance":
                    th = np.abs(cs)
                else:
                    th = np.maximum(np.cumsum(np.where(x > 0, x, 0)), np.cumsum(np.where(x < 0, -x, 0)))
                ok += int(th[-1] >= r.thr and (len(th) == 1 or (th[:-1] < r.thr).all()))
            # independent recomputation of E0[T] from the stored bar lengths
            et = pd.Series(bars["T"]).shift(1).fillna(T0)
            ew = [T0]
            for t in bars["T"].iloc[:-1]: ew.append((1 - ALPHA) * ew[-1] + ALPHA * t)
            et_ok = bool(np.allclose(ew, bars["ET"].to_numpy()))
            days = (d["date"].iloc[-1] - d["date"].iloc[WARM]).days or 1
            res[(venue, k)] = dict(rows=rows - WARM, bars=len(bars), rows_per_bar=float(bars["T"].mean()), median=float(bars["T"].median()), one_row=float((bars["T"] == 1).mean()),
                                   max_T=int(bars["T"].max()), last_ET=float(bars["ET"].iloc[-1]), per_day=len(bars) / days, check=f"{ok}/{len(bars)}", et_ok=et_ok,
                                   buys=float((b > 0).mean()))
    return res


if __name__ == "__main__":
    res = run()
    ser = {f"{a}|{k}": v for (a, k), v in res.items()}
    json.dump(ser, open(W.OUT / "sec23.json", "w"), indent=1)
    for k, v in ser.items(): print(k, v)
