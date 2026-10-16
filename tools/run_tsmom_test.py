#!/usr/bin/env python3
"""Diversified, volatility-scaled time-series momentum (TSMOM) -- a fully
predeclared test on long-history daily ETF data. Separate from, and not a
rescue of, the hourly crypto LightGBM system; nothing here touches the live
bot or any of its files.

PREDECLARED SPEC (stated in chat before this was written; no parameter in
this file was chosen or adjusted after seeing any result):

  Universe (fixed, not tuned): SPY, EFA, EEM (equities); IEF, TLT (bonds);
    GLD, DBC (commodities); UUP (dollar proxy). Adjusted ("total return")
    daily closes from yfinance.
  Signal: trailing 252-trading-day (~12 month) return, measured at the last
    trading day of each month. Long if > 0, otherwise OUT (cash earning 0%).
    No shorting.
  Weights: inverse volatility (60-day daily-return std) across the assets
    that are long, normalised to sum to 1.
  Risk scaling: scale so the portfolio's estimated annual volatility (60-day
    trailing covariance of the selected assets) is 10%, with total leverage
    capped at 1.5x. Unleveraged remainder is cash at 0%.
  Rebalance: monthly, signal at month-end close, trades take effect from the
    NEXT trading day's returns (no same-bar fill). Weights then drift with
    prices until the next rebalance.
  Costs: 5bp per side on traded notional at each rebalance (turnover-based).
  An asset becomes eligible only once it has 252 days of return history AND
    60 days of volatility history. The evaluation window starts at the first
    month-end with >= 6 eligible assets.
  Comparators (same costs, same monthly rebalance, same evaluation dates):
    (a) equal-weight basket of the eligible assets, (b) 60% SPY / 40% IEF.
    Per-asset buy-and-hold is shown for information only.
  Sub-periods: pre-2012 (evaluation start -> 2011-12-31) and post-2012
    (2012-01-01 -> end of data), each reported separately, plus every
    calendar year, plus the June 2026 window (2026-05-15 -> 2026-07-01)
    reported on its own and never folded into any aggregate.
  VERDICT BAR (predeclared): TSMOM passes only if its Calmar ratio
    (annualised return / |max drawdown|) beats BOTH comparators' Calmar in
    BOTH sub-periods. Sharpe is reported but is not the bar.

KNOWN SIMPLIFICATIONS (disclosed, not hidden): cash earns 0% (understates
returns in high-rate years), borrowing cost on leverage above 1x is not
modelled (overstates returns when leveraged), no taxes, no per-asset or
per-class weight caps beyond the 1.5x leverage cap, and the "pre-2012"
window is only ~6 years because several of these ETFs launched in the
2000s -- a short sample for a strategy whose published evidence spans
decades. A pass here would not be strong evidence; a fail would be real.

USAGE (in the `javier` env; needs network for yfinance):
    cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
    python tools/run_tsmom_test.py
Writes runtime/tsmom_prices.csv (inputs, for reproducibility) and
runtime/tsmom_results.json.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "tools") \
    else Path(__file__).resolve().parent
PRICES_PATH = PROJECT_ROOT / "runtime" / "tsmom_prices.csv"
OUT_PATH = PROJECT_ROOT / "runtime" / "tsmom_results.json"

UNIVERSE = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "UUP"]
LOOKBACK = 252
VOL_WINDOW = 60
TARGET_VOL = 0.10
MAX_LEVERAGE = 1.5
COST_PER_SIDE = 5.0 / 10000.0
MIN_ELIGIBLE = 6
SPLIT_DATE = pd.Timestamp("2012-01-01")
JUNE_START = pd.Timestamp("2026-05-15")
JUNE_END = pd.Timestamp("2026-07-01")
DOWNLOAD_START = "1999-01-01"


# ---------------- data ----------------

def download_prices() -> pd.DataFrame:
    import yfinance as yf
    cols = {}
    for tkr in UNIVERSE:
        last_err = None
        for attempt in range(3):
            try:
                df = yf.download(tkr, start=DOWNLOAD_START, interval="1d", auto_adjust=True,
                                 progress=False, threads=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                s = df["Close"].dropna()
                if len(s) < 500:
                    raise ValueError(f"only {len(s)} rows")
                cols[tkr] = s
                print(f"  {tkr}: {len(s)} rows, {s.index.min().date()} -> {s.index.max().date()}")
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(2.0)
        else:
            raise SystemExit(f"download failed for {tkr}: {last_err} -- aborting rather than silently dropping an asset")
    prices = pd.DataFrame(cols).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICES_PATH)
    return prices


# ---------------- simulation ----------------

def month_end_dates(index: pd.DatetimeIndex) -> list:
    s = pd.Series(index, index=index)
    return [pd.Timestamp(x) for x in s.groupby([index.year, index.month]).max().values]


def simulate(prices: pd.DataFrame, weight_fn, rebal_dates: list, start_date) -> pd.Series:
    """Generic monthly-rebalanced simulator.

    weight_fn(date) -> pd.Series of target position weights (fraction of
    equity, may sum to >1 when levered; the remainder is cash at 0%).
    Weights are decided with data through `date` and applied to returns from
    the NEXT row onward. Between rebalances positions drift with prices.
    Cost = COST_PER_SIDE * sum|traded notional| (fraction of equity) charged
    at the rebalance. Returns the daily equity-return series."""
    rets = prices.pct_change()
    idx = prices.index
    pos_of = {d: i for i, d in enumerate(idx)}
    rebal = [d for d in rebal_dates if d >= start_date]
    daily = pd.Series(0.0, index=idx)
    equity = 1.0
    cur_pos = pd.Series(0.0, index=prices.columns)   # dollar positions, equity units
    first_i = pos_of[rebal[0]]
    for k, d in enumerate(rebal):
        i = pos_of[d]
        target_w = weight_fn(d).reindex(prices.columns).fillna(0.0)
        # drift old positions are already carried in cur_pos (dollars); convert to weights
        old_w = cur_pos / equity if equity != 0 else cur_pos * 0.0
        turnover = float((target_w - old_w).abs().sum())
        cost = turnover * COST_PER_SIDE
        equity *= (1.0 - cost)
        daily.iloc[i] = (1.0 + daily.iloc[i]) * (1.0 - cost) - 1.0 if i > first_i else -cost
        pos = target_w * equity
        j_end = pos_of[rebal[k + 1]] if k + 1 < len(rebal) else len(idx) - 1
        cash = equity - pos.sum()
        for j in range(i + 1, j_end + 1):
            r = rets.iloc[j].fillna(0.0)
            pos = pos * (1.0 + r)
            new_equity = cash + pos.sum()
            daily.iloc[j] = new_equity / equity - 1.0 if equity != 0 else 0.0
            equity = new_equity
        cur_pos = pos
    return daily.iloc[first_i:]


def build_weight_fns(prices: pd.DataFrame):
    rets = prices.pct_change()
    ret_lb = prices / prices.shift(LOOKBACK) - 1.0
    vol = rets.rolling(VOL_WINDOW).std()
    eligible = ret_lb.notna() & vol.notna()

    def tsmom(d):
        elig = eligible.loc[d]
        longs = [a for a in prices.columns if elig[a] and ret_lb.loc[d, a] > 0]
        if not longs:
            return pd.Series(0.0, index=prices.columns)
        inv = 1.0 / vol.loc[d, longs]
        w = inv / inv.sum()
        window = rets.loc[:d].iloc[-VOL_WINDOW:][longs]
        cov = window.cov().values * 252.0
        pvol = float(np.sqrt(w.values @ cov @ w.values))
        lev = MAX_LEVERAGE if pvol <= 0 else min(MAX_LEVERAGE, TARGET_VOL / pvol)
        out = pd.Series(0.0, index=prices.columns)
        out[longs] = w * lev
        return out

    def equal_weight(d):
        elig = [a for a in prices.columns if eligible.loc[d, a]]
        out = pd.Series(0.0, index=prices.columns)
        if elig:
            out[elig] = 1.0 / len(elig)
        return out

    def sixty_forty(d):
        out = pd.Series(0.0, index=prices.columns)
        out["SPY"] = 0.6
        out["IEF"] = 0.4
        return out

    return tsmom, equal_weight, sixty_forty, eligible


# ---------------- metrics ----------------

def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return dict(days=len(r), cagr=float("nan"), vol=float("nan"), sharpe=float("nan"),
                    max_dd=float("nan"), calmar=float("nan"))
    eq = (1 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    calmar = float(cagr / abs(dd)) if dd < 0 else float("nan")
    return dict(days=int(len(r)), cagr=cagr, vol=vol, sharpe=sharpe, max_dd=dd, calmar=calmar)


def fmt(m: dict) -> str:
    return (f"days={m['days']:5d}  cagr={m['cagr']*100:6.2f}%  vol={m['vol']*100:5.2f}%  "
            f"sharpe={m['sharpe']:5.2f}  maxDD={m['max_dd']*100:6.1f}%  calmar={m['calmar']:5.2f}")


def window_return(r: pd.Series, lo, hi):
    sub = r[(r.index >= lo) & (r.index <= hi)]
    if len(sub) == 0:
        return dict(days=0, return_pct=None)
    return dict(days=int(len(sub)), return_pct=float(((1 + sub).prod() - 1) * 100))


def main():
    print("=== downloading prices ===")
    prices = download_prices()
    fp = hashlib.md5(prices.round(6).to_csv().encode()).hexdigest()
    print(f"price data fingerprint (md5, 6dp): {fp}")
    run_analysis(prices, fp)


def run_analysis(prices: pd.DataFrame, fingerprint: str = "n/a", write: bool = True) -> dict:
    tsmom_fn, ew_fn, sf_fn, eligible = build_weight_fns(prices)
    rebal_dates = month_end_dates(prices.index)
    n_elig = eligible.sum(axis=1)
    starts = [d for d in rebal_dates if n_elig.loc[d] >= MIN_ELIGIBLE]
    if not starts:
        raise SystemExit("never reached MIN_ELIGIBLE assets -- check the data")
    start_date = starts[0]
    print(f"\nevaluation start (first month-end with >= {MIN_ELIGIBLE} eligible assets): {start_date.date()}")
    print(f"data end: {prices.index.max().date()}")

    strategies = {
        "TSMOM": simulate(prices, tsmom_fn, rebal_dates, start_date),
        "EqualWeight": simulate(prices, ew_fn, rebal_dates, start_date),
        "60/40": simulate(prices, sf_fn, rebal_dates, start_date),
    }

    # annual turnover of TSMOM (informational)
    pos_turn = []
    prev = pd.Series(0.0, index=prices.columns)
    for d in [x for x in rebal_dates if x >= start_date]:
        w = tsmom_fn(d)
        pos_turn.append(float((w - prev).abs().sum()))
        prev = w
    print(f"TSMOM average one-way-sum turnover per monthly rebalance (ignores drift): {np.mean(pos_turn):.2f}x equity")

    out = dict(fingerprint=fingerprint, eval_start=str(start_date.date()), data_end=str(prices.index.max().date()),
               overall={}, pre2012={}, post2012={}, calendar_years={}, june_2026={}, verdict={})

    print("\n=== OVERALL (evaluation window) ===")
    for name, r in strategies.items():
        m = metrics(r); out["overall"][name] = m
        print(f"  {name:12s} {fmt(m)}")

    pre = {k: v[v.index < SPLIT_DATE] for k, v in strategies.items()}
    post = {k: v[v.index >= SPLIT_DATE] for k, v in strategies.items()}
    print(f"\n=== PRE-2012 ({start_date.date()} -> 2011-12-31) ===")
    for name, r in pre.items():
        m = metrics(r); out["pre2012"][name] = m
        print(f"  {name:12s} {fmt(m)}")
    print("\n=== POST-2012 (2012-01-01 -> end) ===")
    for name, r in post.items():
        m = metrics(r); out["post2012"][name] = m
        print(f"  {name:12s} {fmt(m)}")

    print("\n=== CALENDAR YEARS (total return %) ===")
    years = sorted(set(strategies["TSMOM"].index.year))
    print(f"  {'year':6s}" + "".join(f"{n:>13s}" for n in strategies))
    for y in years:
        row = {}
        for n, r in strategies.items():
            sub = r[r.index.year == y]
            row[n] = float(((1 + sub).prod() - 1) * 100) if len(sub) else float("nan")
        out["calendar_years"][int(y)] = row
        print(f"  {y:<6d}" + "".join(f"{row[n]:13.1f}" for n in strategies))

    print("\n=== JUNE 2026 WINDOW (2026-05-15 -> 2026-07-01), reported on its own ===")
    for name, r in strategies.items():
        w = window_return(r, JUNE_START, JUNE_END); out["june_2026"][name] = w
        if w["return_pct"] is None:
            print(f"  {name:12s} no data in window (data ends {prices.index.max().date()})")
        else:
            print(f"  {name:12s} days={w['days']} return={w['return_pct']:.2f}%")
    for tkr in prices.columns:
        w = window_return(prices[tkr].pct_change().dropna(), JUNE_START, JUNE_END)
        out["june_2026"][f"B&H {tkr}"] = w
        if w["return_pct"] is not None:
            print(f"  B&H {tkr:8s} days={w['days']} return={w['return_pct']:.2f}%")

    print("\n=== per-asset buy-and-hold over evaluation window (information only) ===")
    for tkr in prices.columns:
        r = prices[tkr].pct_change()
        r = r[r.index > start_date].dropna()
        m = metrics(r)
        print(f"  {tkr:5s} {fmt(m)}")

    print("\n=== VERDICT (predeclared bar: TSMOM Calmar beats BOTH comparators' Calmar in BOTH sub-periods) ===")
    verdict_parts = {}
    for label, period in [("pre2012", out["pre2012"]), ("post2012", out["post2012"])]:
        t = period["TSMOM"]["calmar"]
        ew = period["EqualWeight"]["calmar"]; sf = period["60/40"]["calmar"]
        ok = bool(np.isfinite(t) and np.isfinite(ew) and np.isfinite(sf) and t > ew and t > sf)
        verdict_parts[label] = ok
        print(f"  {label}: TSMOM calmar={t:.2f} vs EW={ew:.2f}, 60/40={sf:.2f} -> {'beats both' if ok else 'does NOT beat both'}")
    passed = all(verdict_parts.values())
    out["verdict"] = dict(per_period=verdict_parts, passed=passed)
    print(f"\n  OVERALL VERDICT: {'PASS' if passed else 'FAIL'} on the predeclared bar. "
          f"(pre-2012 sample is short; a pass would be weak evidence, a fail is informative.)")

    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {OUT_PATH} and {PRICES_PATH}")
    return out


if __name__ == "__main__":
    main()
