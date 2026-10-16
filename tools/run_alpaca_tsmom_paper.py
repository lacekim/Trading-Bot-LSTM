"""
Rebuild of the BASELINE time-series momentum strategy from Deng (Nov 2025), "Baseline Time-Series
Momentum and Grid-Sized Enhancement", on Alpaca daily ETF data. OFFLINE research script: it writes
only to runtime/alpaca_tsmom_etfs/ and runtime/alpaca_tsmom_paper_results.json. It touches no live-bot
file and never writes into runtime/alpaca_daily/ (the other script globs *.parquet there).

THIS IS NOT A REPLICATION OF THE PAPER. The paper uses 58 futures from 1969; Alpaca has no futures and
only ~2016+ history. This tests the paper's BASELINE RULE on a 25-ETF proxy universe over what is
essentially the paper's own out-of-sample era. It cannot test the paper's in-sample (1969-2014) claim,
and it does not test the GRID overlay.

------------------------- PREDECLARED SPECIFICATION (fixed before any result) -------------------------
UNIVERSE (25 ETFs, fixed list; missing symbols are reported, not substituted):
  equity    : SPY QQQ IWM EFA EEM EWJ EWZ FXI
  bond      : IEF TLT TIP LQD HYG
  commodity : GLD SLV USO UNG DBA DBB
  fx        : UUP FXE FXY FXB FXA FXC
PRICES: Alpaca daily closes, adjustment=all (split+dividend adjusted, i.e. total-return-like). Missing
  days are forward-filled (0 return). Commodity/FX ETFs embed their own roll/fee drag; not corrected.
VOL: sigma_t = sqrt(252 * EWMA(r^2)), daily returns, halflife 252 days, min 126 days (zero-mean).
SIGNAL (at close of month-end day j): S_i = sum over the last 252 trading days of r_t / sigma_{t-1}
  (needs >= 240 valid terms). Position direction = sign(S_i): +1 long, -1 short (shorts allowed).
SIZING: w_i = sign_i * (40% / sigma_i(j)) / N_active, N_active = assets with a valid signal and sigma.
  Portfolio gross exposure capped at 6.0 (all weights scaled down proportionally if above).
  No rebalancing between month-ends (weights drift).
EXECUTION: signal at month-end close j, trade at the close of the next trading day j+1.
EVALUATION START: first month-end with >= 8 active assets (that is the first trade date).
COSTS: headline 5 bp per unit of notional traded (turnover measured in units of equity).
  Sensitivities (information only): 1.5 bp (the paper's), 10, 20 bp.
FINANCING: headline NONE (paper-style, futures-like). Sensitivity B (information only): flat 3%/yr on
  gross exposure above 1.0x.
COMPARATORS (same start date, same 5 bp costs, monthly rebalanced): SPY buy&hold; 60/40 SPY/IEF;
  equal-weight long-only of the active ETFs; and CONTROL = same sizing/leverage but always long.
VERDICT (headline = 5 bp, no financing). PASS only if ALL hold:
  (i)   net Sharpe (rf=0) >= 0.33 (the paper's own out-of-sample baseline Sharpe);
  (ii)  net CAGR > 0 at 20 bp per unit turnover;
  (iii) rolling 756-day CAGR > 0 in >= 60% of windows;
  (iv)  after removing the single best-contributing asset class, net Sharpe is still > 0;
  (v)   net Sharpe beats the always-long CONTROL (so the signal adds value over its own sizing).
A June 2026 window (2026-05-15 -> 2026-07-01) is always reported on its own for every series.
No parameter is changed after seeing results. Sample is ~9 years: Sharpe standard error is roughly
0.33, so a PASS or FAIL here is weak evidence either way.
-------------------------------------------------------------------------------------------------------

Run (project root, `javier` env; needs ALPACA_API_KEY / ALPACA_SECRET_KEY in .env):
    python tools/run_alpaca_tsmom_paper.py
    python tools/run_alpaca_tsmom_paper.py --feed iex
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_alpaca_momentum_test as base   # reuse credentials loader, metrics, fmt (no side effects on import)

PROJECT_ROOT = base.PROJECT_ROOT
CACHE_DIR = PROJECT_ROOT / "runtime" / "alpaca_tsmom_etfs"
OUT_PATH = PROJECT_ROOT / "runtime" / "alpaca_tsmom_paper_results.json"

CLASSES = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "EWZ", "FXI"],
    "bond": ["IEF", "TLT", "TIP", "LQD", "HYG"],
    "commodity": ["GLD", "SLV", "USO", "UNG", "DBA", "DBB"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA", "FXC"],
}
ALL_SYMS = [s for v in CLASSES.values() for s in v]
CLASS_OF = {s: c for c, v in CLASSES.items() for s in v}

LOOKBACK = 252
SIG_MIN_TERMS = 240
VOL_HALFLIFE = 252
VOL_MIN = 126
TARGET_VOL = 0.40
GROSS_CAP = 6.0
MIN_ACTIVE = 8
COST_HEADLINE = 5.0
COSTS_SENS = [1.5, 5.0, 10.0, 20.0]
FIN_SENS = 0.03
ROLL_WIN = 756
PAPER_OOS_SHARPE = 0.33
JUNE_START, JUNE_END = base.JUNE_START, base.JUNE_END


# ======================= data =======================

def fetch(feed_arg: str) -> None:
    out = CACHE_DIR / "etfs.parquet"
    if out.exists():
        return
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    k, s = base._creds()
    client = StockHistoricalDataClient(k, s)
    ff = base.CACHE_DIR / "feed.txt"
    feed = ff.read_text().strip() if ff.exists() else (feed_arg if feed_arg != "auto" else "sip")
    end = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    df = None
    for attempt in range(4):
        try:
            req = StockBarsRequest(symbol_or_symbols=ALL_SYMS, timeframe=TimeFrame.Day, start=base.START_DATE,
                                   end=end, adjustment=Adjustment.ALL,
                                   feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
            df = client.get_stock_bars(req).df
            break
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if feed == "sip" and feed_arg == "auto" and ("subscription" in msg or "forbidden" in msg or "403" in msg):
                print(f"  SIP not permitted ({exc}); switching to IEX")
                feed = "iex"
                continue
            print(f"  attempt {attempt+1} failed: {exc}")
            time.sleep(3.0 * (attempt + 1))
    if df is None or df.empty:
        raise SystemExit("no ETF bars returned from Alpaca")
    df = df.reset_index()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    slim = pd.DataFrame({"symbol": df["symbol"].astype(str), "date": ts, "close": df["close"].astype(float)})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(out)
    (CACHE_DIR / "feed.txt").write_text(feed)
    print(f"  fetched {slim['symbol'].nunique()} ETFs, {len(slim)} bars, feed={feed} -> {out.name}")


def load_close() -> tuple[pd.DataFrame, str]:
    df = pd.read_parquet(CACHE_DIR / "etfs.parquet").dropna(subset=["close"]).drop_duplicates(["symbol", "date"], keep="last")
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    feed = (CACHE_DIR / "feed.txt").read_text().strip() if (CACHE_DIR / "feed.txt").exists() else "unknown"
    return close, feed


# ======================= engine =======================

def month_end_positions(index: pd.DatetimeIndex) -> list[int]:
    s = pd.Series(np.arange(len(index)), index=index)
    return [int(v) for v in s.groupby([index.year, index.month]).max().values]


class TS:
    def __init__(self, close: pd.DataFrame):
        self.syms = [s for s in ALL_SYMS if s in close.columns]
        self.missing = [s for s in ALL_SYMS if s not in close.columns]
        c = close[self.syms].sort_index().ffill()
        self.close = c
        self.dates = c.index
        self.r = c.pct_change(fill_method=None)
        var = (self.r ** 2).ewm(halflife=VOL_HALFLIFE, min_periods=VOL_MIN).mean()
        self.sigma = np.sqrt(var * 252.0)
        z = self.r / self.sigma.shift(1)
        self.S = z.rolling(LOOKBACK, min_periods=SIG_MIN_TERMS).sum()
        self.R = self.r.fillna(0.0).values
        self.Sv = self.S.values
        self.sg = self.sigma.values
        self.me = month_end_positions(self.dates)

    def active_mask(self, j: int) -> np.ndarray:
        return ~np.isnan(self.Sv[j]) & ~np.isnan(self.sg[j]) & (self.sg[j] > 0)

    def targets(self, kind: str) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        m = len(self.syms)
        n = len(self.dates)
        for j in self.me:
            e = j + 1
            if e >= n:
                continue
            act = self.active_mask(j)
            na = int(act.sum())
            if na < MIN_ACTIVE:
                continue
            w = np.zeros(m)
            if kind in ("tsmom", "control"):
                sgn = np.sign(self.Sv[j]) if kind == "tsmom" else np.ones(m)
                w[act] = sgn[act] * (TARGET_VOL / self.sg[j][act]) / na
                g = np.abs(w).sum()
                if g > GROSS_CAP:
                    w *= GROSS_CAP / g
            elif kind == "ew":
                w[act] = 1.0 / na
            elif kind == "spy":
                w[self.syms.index("SPY")] = 1.0
            elif kind == "6040":
                w[self.syms.index("SPY")] = 0.6
                w[self.syms.index("IEF")] = 0.4
            else:
                raise ValueError(kind)
            out[e] = w
        return out

    def simulate(self, targets: dict[int, np.ndarray], cost_bp: float, fin_rate: float = 0.0,
                 start_e: int | None = None):
        n, m = self.R.shape
        w = np.zeros(m)
        net = np.full(n, np.nan)
        contrib = np.zeros((n, m))
        grossh = np.full(n, np.nan)
        neth = np.full(n, np.nan)
        turns = []
        first = start_e if start_e is not None else min(targets)
        for t in range(1, n):
            c = w * self.R[t]
            pr = c.sum()
            gross = np.abs(w).sum()
            rg = pr - fin_rate / 252.0 * max(gross - 1.0, 0.0)
            w = w * (1.0 + self.R[t]) / (1.0 + rg)
            cost = 0.0
            if t in targets and t >= first:
                wt = targets[t]
                turn = float(np.abs(wt - w).sum())
                cost = cost_bp / 1e4 * turn
                w = wt.copy()
                turns.append(turn)
            if t >= first:
                net[t] = (1.0 + rg) * (1.0 - cost) - 1.0
                contrib[t] = c
                grossh[t] = np.abs(w).sum()
                neth[t] = w.sum()
        s = pd.Series(net, index=self.dates)
        return s.iloc[first:], contrib[first:], grossh[first:], neth[first:], turns


def window_ret(r: pd.Series, lo, hi) -> float:
    x = r[(r.index >= lo) & (r.index <= hi)].dropna()
    return float((1 + x).prod() - 1) if len(x) else float("nan")


def rolling_cagr(r: pd.Series, win: int = ROLL_WIN) -> pd.Series:
    lr = np.log1p(r.dropna())
    return np.exp(lr.rolling(win).sum() * 252.0 / win) - 1.0


def sharpe_se(sh: float, years: float) -> float:
    return float(np.sqrt((1 + 0.5 * sh ** 2) / max(years, 1e-9)))


def run_study(close: pd.DataFrame, feed: str = "n/a", write: bool = True) -> dict:
    ts = TS(close)
    print(f"\ndata: {len(ts.syms)}/{len(ALL_SYMS)} ETFs, {ts.dates[0].date()} -> {ts.dates[-1].date()}, feed={feed}")
    if ts.missing:
        print(f"MISSING symbols (not substituted): {ts.missing}")
    first_dates = {s: str(ts.close[s].first_valid_index().date()) for s in ts.syms}
    late = {s: d for s, d in first_dates.items() if pd.Timestamp(d) > ts.dates[0] + pd.Timedelta(days=10)}
    if late:
        print(f"ETFs with data starting after the panel start: {late}")
    big = [(s, str(d.date()), round(float(v), 3)) for s in ts.syms for d, v in ts.r[s][ts.r[s].abs() > 0.25].items()]
    print(f"data hygiene: {len(big)} ETF-days with |daily return| > 25% " + (f"-> {big[:8]}" if big else ""))

    t_ts = ts.targets("tsmom")
    if not t_ts:
        raise SystemExit("no valid rebalance dates (insufficient history)")
    e0 = min(t_ts)
    act_counts = [int(ts.active_mask(j).sum()) for j in ts.me if j + 1 in t_ts]
    print(f"first trade date {ts.dates[e0].date()}; active ETFs per rebalance min/median/max = "
          f"{min(act_counts)}/{int(np.median(act_counts))}/{max(act_counts)}")
    print("REMINDER: ETF proxy universe, ~9y sample, not the paper's futures/1969 data; weak evidence either way.")

    series: dict[str, pd.Series] = {}
    r_ts, contrib, gross, net_exp, turns = ts.simulate(t_ts, COST_HEADLINE)
    series["TSMOM paper baseline (5bp)"] = r_ts
    series["CONTROL same sizing, always long"] = ts.simulate(ts.targets("control"), COST_HEADLINE, start_e=e0)[0]
    series["EW long-only, active ETFs"] = ts.simulate(ts.targets("ew"), COST_HEADLINE, start_e=e0)[0]
    series["SPY buy&hold"] = ts.simulate({e0: ts.targets("spy")[e0]}, COST_HEADLINE, start_e=e0)[0]
    series["60/40 (SPY/IEF)"] = ts.simulate(ts.targets("6040"), COST_HEADLINE, start_e=e0)[0]
    names = list(series)
    key = names[0]

    print(f"\n=== OVERALL (signal->next-day-close execution, {COST_HEADLINE:g} bp per unit turnover, no financing) ===")
    allm = {}
    for nme in names:
        allm[nme] = base.metrics(series[nme])
        print(f"  {nme:34s} {base.fmt(allm[nme])}")
    yrs = len(r_ts) / 252.0
    sh = allm[key]["sharpe"]
    print(f"  TSMOM: avg gross exposure {np.nanmean(gross):.2f}x, avg net exposure {np.nanmean(net_exp):+.2f}x, "
          f"avg turnover/rebalance {np.mean(turns):.2f}x equity ({np.mean(turns)*12:.1f} turns/yr), "
          f"approx Sharpe SE {sharpe_se(sh, yrs):.2f} over {yrs:.1f}y")
    spy_c = series["SPY buy&hold"].reindex(r_ts.index)
    print(f"  TSMOM daily return correlation with SPY: {float(r_ts.corr(spy_c)):.2f}")

    mid = r_ts.index[len(r_ts) // 2]
    print(f"\n=== FIRST vs SECOND HALF (information only; split at {mid.date()}) ===")
    for lab, sl in (("first half", lambda x: x[x.index < mid]), ("second half", lambda x: x[x.index >= mid])):
        print(f"  --- {lab} ---")
        for nme in names:
            print(f"    {nme:34s} {base.fmt(base.metrics(sl(series[nme])))}")

    print("\n=== CALENDAR YEARS (total return %) ===")
    yr = pd.DataFrame({n: (1 + series[n].dropna()).groupby(series[n].dropna().index.year).prod() - 1 for n in names}) * 100
    print("  year  " + "".join(f"{n[:16]:>18s}" for n in names))
    for y, row in yr.iterrows():
        print(f"  {y}  " + "".join(f"{v:18.1f}" for v in row.values))

    print(f"\n=== JUNE 2026 WINDOW ({JUNE_START.date()} -> {JUNE_END.date()}), on its own ===")
    june = {}
    for nme in names:
        june[nme] = window_ret(series[nme], JUNE_START, JUNE_END)
        print(f"  {nme:34s} {june[nme]*100:7.2f}%")

    print("\n=== ATTRIBUTION by asset class (TSMOM, gross of costs; sum of daily contributions, % of starting equity) ===")
    cls_idx = {c: [i for i, s in enumerate(ts.syms) if CLASS_OF[s] == c] for c in CLASSES}
    cls_tot = {c: float(contrib[:, ix].sum()) * 100 for c, ix in cls_idx.items()}
    tot = sum(cls_tot.values())
    for c, v in cls_tot.items():
        print(f"  {c:10s} {v:8.1f}  ({(v/tot*100 if tot else float('nan')):6.1f}% of gross total {tot:.1f})")
    per_sym = pd.Series(contrib.sum(0) * 100, index=ts.syms).sort_values(ascending=False)
    print("  top 5 ETFs:    " + ", ".join(f"{s} {v:.1f}" for s, v in per_sym.head(5).items()))
    print("  bottom 5 ETFs: " + ", ".join(f"{s} {v:.1f}" for s, v in per_sym.tail(5).items()))
    best_cls = max(cls_tot, key=cls_tot.get)
    r_ex = r_ts - pd.Series(contrib[:, cls_idx[best_cls]].sum(1), index=r_ts.index)
    sh_ex = base.metrics(r_ex)["sharpe"]
    print(f"  best class = {best_cls}; net Sharpe with that class's contribution removed: {sh_ex:.2f}")

    print("\n=== ROLLING 756-DAY (~3y) CAGR (heavily overlapping windows) ===")
    rc = rolling_cagr(r_ts).dropna()
    pos_share = float((rc > 0).mean()) if len(rc) else float("nan")
    print(f"  windows={len(rc)}  share with CAGR > 0: {pos_share*100:.1f}%  median {rc.median()*100:.1f}%  "
          f"min {rc.min()*100:.1f}%  max {rc.max()*100:.1f}%")

    print("\n=== SENSITIVITY (information only): cost, financing ===")
    sens = {}
    for cbp in COSTS_SENS:
        rr = ts.simulate(t_ts, cbp)[0]
        sens[f"cost_{cbp:g}"] = base.metrics(rr)
        print(f"  TSMOM @ {cbp:5.1f} bp/unit            {base.fmt(sens[f'cost_{cbp:g}'])}")
    rr = ts.simulate(t_ts, COST_HEADLINE, fin_rate=FIN_SENS)[0]
    sens["financing_3pct"] = base.metrics(rr)
    print(f"  TSMOM @ 5 bp + 3%/yr fin on >1x   {base.fmt(sens['financing_3pct'])}")

    cagr20 = sens["cost_20"]["cagr"]
    ctrl_sh = allm["CONTROL same sizing, always long"]["sharpe"]
    v = {
        "i_sharpe_ge_paper_oos": bool(sh >= PAPER_OOS_SHARPE),
        "ii_cagr_pos_at_20bp": bool(cagr20 > 0),
        "iii_rolling3y_pos_ge_60pct": bool(pos_share >= 0.60),
        "iv_pos_sharpe_ex_best_class": bool(sh_ex > 0),
        "v_beats_always_long_control": bool(sh > ctrl_sh),
    }
    print("\n=== VERDICT (predeclared; headline 5 bp, no financing) ===")
    print(f"  (i)   net Sharpe {sh:.2f} vs {PAPER_OOS_SHARPE} (paper OOS)  -> {v['i_sharpe_ge_paper_oos']}")
    print(f"  (ii)  CAGR at 20 bp {cagr20*100:.2f}%  -> {v['ii_cagr_pos_at_20bp']}")
    print(f"  (iii) rolling-3y positive share {pos_share*100:.1f}% (>=60% required)  -> {v['iii_rolling3y_pos_ge_60pct']}")
    print(f"  (iv)  Sharpe ex-{best_cls} {sh_ex:.2f}  -> {v['iv_pos_sharpe_ex_best_class']}")
    print(f"  (v)   Sharpe {sh:.2f} vs always-long control {ctrl_sh:.2f}  -> {v['v_beats_always_long_control']}")
    overall = all(v.values())
    print(f"\n  OVERALL VERDICT: {'PASS' if overall else 'FAIL'}  -- ETF proxy universe, ~{yrs:.0f}y sample, Sharpe SE ~{sharpe_se(sh, yrs):.2f}. "
          f"Not a test of the paper's futures/1969 results or of its GRID overlay.")

    res = dict(feed=feed, first_trade=str(ts.dates[e0].date()), n_etfs=len(ts.syms), missing=ts.missing,
               overall=allm, june_2026=june, class_contribution_pct=cls_tot, best_class=best_cls,
               sharpe_ex_best_class=sh_ex, rolling3y_positive_share=pos_share, sensitivity=sens,
               verdict=v, overall_pass=overall, avg_gross=float(np.nanmean(gross)),
               avg_net=float(np.nanmean(net_exp)), avg_turnover=float(np.mean(turns)))
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(res, indent=2, default=str))
        print(f"\nwrote {OUT_PATH}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", default="auto", choices=["auto", "sip", "iex"])
    args = ap.parse_args()
    print("=== ETF bars ===")
    fetch(args.feed)
    close, feed = load_close()
    run_study(close, feed)


if __name__ == "__main__":
    main()
