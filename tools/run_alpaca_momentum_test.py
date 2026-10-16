#!/usr/bin/env python3
"""Long-only cross-sectional MOMENTUM test on the data you actually trade
with: Alpaca daily bars for U.S. stocks. Fetches (and caches) the bars, then
tests the proposed strategy MINUS the quality score, which Alpaca cannot
supply (no fundamentals). Nothing here touches the live bot.

READ THIS FIRST -- WHAT THIS CAN AND CANNOT SHOW
  * SURVIVORSHIP BIAS (unfixable with this source): the stock list is
    Alpaca's CURRENT active, tradable assets. Stocks that were delisted,
    went bankrupt or were acquired are (almost certainly) missing from the
    history, which flatters any strategy and momentum in particular. Every
    result below is an optimistic upper bound on that count.
  * SHORT HISTORY: Alpaca daily bars start around 2016, so there is NO
    pre-2012 period and the sample is one mostly-bull regime. The
    "rolling 5-year" windows overlap heavily (only ~2 independent 5-year
    periods exist). A PASS would be weak evidence. A FAIL would be real.
  * NO QUALITY SCORE: the 70/30 momentum+quality blend cannot be built.
    This tests the 100%-momentum leg only.
  * NO MARKET CAP / SECTORS / POINT-IN-TIME CONSTITUENTS: the $2B screen is
    replaced by a dollar-volume RANK screen (below); the 20% sector cap
    cannot be enforced (the 5%-per-stock cap never binds at 1/30 weights).
  * STOCK-LIST FILTER IS A NAME HEURISTIC (drops ETFs, trusts, funds, SPACs,
    warrants, preferreds, ADRs/depositary receipts by name keywords); some
    legitimate stocks (e.g., trust-named REITs) get dropped and a few
    non-stocks may slip through.
  * PRICES ARE DIVIDEND-ADJUSTED (adjustment=all), so the $5 price screen
    uses adjusted history (slightly stricter in earlier years).
  * VOLUME depends on the Alpaca feed available to your plan (SIP, or
    IEX-only which understates volume ~30x). The screen therefore uses
    dollar-volume RANK, not an absolute $ threshold, so it works on either.
  * A halted or missing bar is carried as a 0% day for held names, and bad
    vendor prints are not cleaned (the script reports how many |daily
    return| > 50% symbol-days exist).

PREDECLARED (before any result; nothing is tuned afterwards)
  Universe at each month-end: names with price > $5, volume > 0 that day, and
    history for every lookback below, ranked by trailing-60-day MEDIAN
    dollar volume (needs >= 40 valid days); keep the top N = 1000.
  Score (base): 0.5 * (12-1 month return) + 0.5 * (6-1 month return), where
    "-1" skips the most recent 21 trading days; 12m = 252d, 9m = 189d,
    6m = 126d lookbacks. No quality term (unavailable).
  Portfolio: top 30 by score, equal-weight 1/30 each (unfilled slots stay
    cash at 0%), monthly rebalance, buffer: an existing holding is kept
    until its rank exceeds 60, then replaced by the best-ranked non-holdings.
    No leverage.
  Timing: signal on the last trading day's close of the month; the trade
    executes at the NEXT trading day's close (no same-bar fill).
  Costs: 10 bp per side on traded notional (sensitivity at 5 and 20 bp).
  Comparators (same costs, same lag, same monthly rebalance): SPY buy-and-
    hold; 60/40 (60% SPY / 40% IEF); equal-weight of the same screened
    universe; plain 12-1 momentum (top 30 by 12-1 only, no buffer).
  Evaluation start: first month-end with >= 200 eligible names; all series
    compared from the same day.
  VERDICT (all must hold, for the BASE variant at 10 bp):
     (i)   Calmar > SPY's AND > 60/40's
     (ii)  CAGR > SPY's
     (iii) beats SPY AND 60/40 on > 50% of rolling 5-year (1260-day) windows
     (iv)  ROBUSTNESS: at least 3 of these 4 lookback variants satisfy (i)
           and (ii) versus SPY: base blend (12/6), 12-1 only, 9-1 only,
           blend (9/6)
     (v)   CONCENTRATION: the 10 largest stock contributors account for
           < 50% of total strategy profit
  Also reported (no verdict): universe sizes N=500 and 2000, costs 5/20 bp,
  calendar years, first/second half of the sample, turnover, cash fraction,
  and the June 2026 window (2026-05-15 -> 2026-07-01) on its own.

USAGE (javier env: alpaca-py, pandas, pyarrow, python-dotenv; .env has
ALPACA_API_KEY / ALPACA_SECRET_KEY, as for your other Alpaca scripts):
    cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
    python tools/run_alpaca_momentum_test.py            # fetch (cached) + test
    python tools/run_alpaca_momentum_test.py --feed iex # force IEX
The first run downloads several thousand symbols in batches and can take a
while; batches are cached in runtime/alpaca_daily/ so a re-run only tests.
Writes runtime/alpaca_momentum_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "tools") \
    else Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "runtime" / "alpaca_daily"
OUT_PATH = PROJECT_ROOT / "runtime" / "alpaca_momentum_results.json"

START_DATE = datetime(2015, 1, 1, tzinfo=timezone.utc)
BATCH_SIZE = 150
BENCH = ["SPY", "IEF"]

N_HOLD = 30
BUFFER_RANK = 60
N_UNIVERSE = 1000
MIN_ELIGIBLE_START = 200
SKIP = 21
LB12, LB9, LB6 = 252, 189, 126
DV_WIN = 60
DV_MIN = 40
COST_BPS = 10.0
ROLL_WIN = 1260
JUNE_START = pd.Timestamp("2026-05-15")
JUNE_END = pd.Timestamp("2026-07-01")
EXCLUDE_NAME = re.compile(
    r"\b(ETF|ETN|FUND|TRUST|INDEX|NOTE|NOTES|WARRANT|WARRANTS|RIGHT|RIGHTS|UNIT|UNITS|PREFERRED|"
    r"ACQUISITION|SPAC|DEPOSITARY|ADR|ADS)\b|ISHARES|SPDR|VANGUARD|PROSHARES|INVESCO|ETFS",
    re.IGNORECASE)
OK_EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}


# ======================= data: universe + fetch =======================

def _creds():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not k or not s:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set (check .env)")
    return k, s


def get_universe(refresh: bool) -> list[str]:
    path = CACHE_DIR / "universe.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())["symbols"]
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest
    k, s = _creds()
    assets = TradingClient(k, s, paper=True).get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    syms, dropped = [], 0
    for a in assets:
        ex = getattr(a.exchange, "value", a.exchange)
        if not a.tradable or str(ex).upper() not in OK_EXCHANGES:
            continue
        if EXCLUDE_NAME.search(a.name or ""):
            dropped += 1
            continue
        if a.symbol in BENCH:
            continue
        syms.append(a.symbol)
    syms = sorted(set(syms))
    print(f"  asset list: {len(assets)} active US equities -> {len(syms)} kept "
          f"({dropped} dropped by name filter, rest by exchange/tradable)")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"symbols": syms, "n_assets_total": len(assets), "n_name_dropped": dropped,
                                "fetched_at": datetime.now(timezone.utc).isoformat()}))
    return syms


def fetch_bars(symbols: list[str], feed_arg: str, tag: str) -> str:
    """Fetch daily split+dividend-adjusted bars in cached batches. Returns feed used."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    k, s = _creds()
    client = StockHistoricalDataClient(k, s)
    feed_file = CACHE_DIR / "feed.txt"
    feed = feed_file.read_text().strip() if feed_file.exists() else (feed_arg if feed_arg != "auto" else "sip")
    end = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
    batches = [BENCH] + batches
    for bi, batch in enumerate(batches):
        out = CACHE_DIR / (f"bench.parquet" if bi == 0 else f"batch_{bi:04d}.parquet")
        if out.exists():
            continue
        for attempt in range(4):
            try:
                req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=START_DATE, end=end,
                                       adjustment=Adjustment.ALL,
                                       feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
                df = client.get_stock_bars(req).df
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if feed == "sip" and feed_arg == "auto" and ("subscription" in msg or "forbidden" in msg or "403" in msg):
                    print(f"  SIP not permitted on this plan ({exc}); switching to IEX")
                    feed = "iex"
                    continue
                print(f"  batch {bi}: attempt {attempt+1} failed: {exc}")
                time.sleep(3.0 * (attempt + 1))
        else:
            print(f"  batch {bi}: giving up after retries (symbols skipped)")
            continue
        if df is None or df.empty:
            print(f"  batch {bi}: empty")
            pd.DataFrame(columns=["symbol", "date", "close", "volume"]).to_parquet(out)
            continue
        df = df.reset_index()
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        slim = pd.DataFrame({"symbol": df["symbol"].astype(str), "date": ts, "close": df["close"].astype(float),
                             "volume": df["volume"].astype(float)})
        slim.to_parquet(out)
        print(f"  batch {bi}/{len(batches)-1}: {slim['symbol'].nunique()} symbols, {len(slim)} bars")
        time.sleep(0.5)
    feed_file.write_text(feed)
    return feed


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    # External/network drives create "._name" AppleDouble metadata stubs next to real files;
    # they match *.parquet but are not parquet. Skip them (and say so), but fail loudly on any
    # REAL cache file that cannot be read, rather than silently dropping data.
    all_files = sorted(CACHE_DIR.glob("*.parquet"))
    stubs = [f for f in all_files if f.name.startswith("._")]
    files = [f for f in all_files if not f.name.startswith("._")]
    if stubs:
        print(f"  ignoring {len(stubs)} '._' metadata stub files in the cache folder")
    if not files:
        raise SystemExit("no cached bars found")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"cache file {f.name} is unreadable ({exc}). Delete that one file and re-run to re-fetch it.")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["close"]).drop_duplicates(["symbol", "date"], keep="last")
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    vol = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    feed = (CACHE_DIR / "feed.txt").read_text().strip() if (CACHE_DIR / "feed.txt").exists() else "unknown"
    return close, vol, feed


# ======================= engine =======================

def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 60:
        return dict(days=len(r), cagr=float("nan"), vol=float("nan"), sharpe=float("nan"),
                    max_dd=float("nan"), calmar=float("nan"))
    eq = (1 + r).cumprod()
    cagr = float(eq.iloc[-1] ** (252.0 / len(r)) - 1)
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    return dict(days=int(len(r)), cagr=cagr, vol=vol, sharpe=sharpe, max_dd=dd,
                calmar=float(cagr / abs(dd)) if dd < 0 else float("nan"))


def fmt(m: dict) -> str:
    return (f"days={m['days']:5d} cagr={m['cagr']*100:6.2f}% vol={m['vol']*100:5.2f}% "
            f"sharpe(rf=0)={m['sharpe']:5.2f} maxDD={m['max_dd']*100:6.1f}% calmar={m['calmar']:5.2f}")


class Study:
    def __init__(self, close: pd.DataFrame, vol: pd.DataFrame):
        spy = close["SPY"].dropna()
        cal = spy.index
        close = close.reindex(cal)
        vol = vol.reindex(cal)
        self.cols = list(close.columns)
        self.S = len(self.cols)
        self.dates = cal
        self.T = len(cal)
        self.close_ff = close.ffill(limit=5)
        self.C = self.close_ff.to_numpy()
        self.V = vol.to_numpy()
        self.DV = self.C * self.V
        self.bench_mask = np.array([c in BENCH for c in self.cols])
        self.col_ix = {c: i for i, c in enumerate(self.cols)}
        self.R = np.nan_to_num(self.close_ff.pct_change(fill_method=None).to_numpy(), nan=0.0)
        self._uni_cache = {}
        self._elig_cache = {}
        ix = pd.Series(np.arange(self.T), index=self.dates)
        self.rebal_idx = [int(v) for v in ix.groupby([self.dates.year, self.dates.month]).max().values]
        self.big_moves = int((np.abs(self.R[:, ~self.bench_mask]) > 0.5).sum())

    # ---- eligibility / universe ----
    def eligible(self, j: int) -> np.ndarray:
        if j in self._elig_cache:
            return self._elig_cache[j]
        if j < LB12:
            res = np.array([], dtype=int)
        else:
            C, V = self.C, self.V
            ok = ~np.isnan(C[j]) & (C[j] > 5) & (np.nan_to_num(V[j]) > 0) & ~self.bench_mask
            for lb in (SKIP, LB6, LB9, LB12):
                ok &= ~np.isnan(C[j - lb])
            idx = np.where(ok)[0]
            if len(idx):
                win = self.DV[j - DV_WIN + 1:j + 1][:, idx]
                cnt = np.sum(~np.isnan(win), axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    med = np.nanmedian(win, axis=0)
                good = cnt >= DV_MIN
                idx, med = idx[good], med[good]
                order = np.argsort(-med)
                idx = idx[order]
            res = idx
        self._elig_cache[j] = res
        return res

    def universe(self, j: int, n: int) -> np.ndarray:
        return self.eligible(j)[:n]

    # ---- scores ----
    def ret(self, j: int, lb: int) -> np.ndarray:
        return self.C[j - SKIP] / self.C[j - lb] - 1.0

    def score(self, j: int, kind: str) -> np.ndarray:
        if kind == "blend12_6":
            return 0.5 * self.ret(j, LB12) + 0.5 * self.ret(j, LB6)
        if kind == "mom12":
            return self.ret(j, LB12)
        if kind == "mom9":
            return self.ret(j, LB9)
        if kind == "blend9_6":
            return 0.5 * self.ret(j, LB9) + 0.5 * self.ret(j, LB6)
        raise ValueError(kind)

    # ---- weight functions ----
    def momentum_fn(self, kind: str, n_univ: int = N_UNIVERSE, use_buffer: bool = True):
        state = {"held": []}

        def fn(j: int) -> np.ndarray:
            w = np.zeros(self.S)
            uni = self.universe(j, n_univ)
            if len(uni) == 0:
                state["held"] = []
                return w
            sc = self.score(j, kind)[uni]
            ranked = uni[np.argsort(-sc, kind="stable")]
            if use_buffer:
                rank_of = {int(s): k for k, s in enumerate(ranked)}
                keep = [s for s in state["held"] if s in rank_of and rank_of[s] < BUFFER_RANK]
                keep = sorted(keep, key=lambda s: rank_of[s])[:N_HOLD]
                sel = list(keep)
                kset = set(keep)
                for s in ranked:
                    if len(sel) >= N_HOLD:
                        break
                    if int(s) not in kset:
                        sel.append(int(s))
            else:
                sel = [int(s) for s in ranked[:N_HOLD]]
            state["held"] = sel
            w[sel] = 1.0 / N_HOLD
            return w
        return fn

    def equal_weight_fn(self, n_univ: int = N_UNIVERSE):
        def fn(j: int) -> np.ndarray:
            w = np.zeros(self.S)
            uni = self.universe(j, n_univ)
            if len(uni):
                w[uni] = 1.0 / len(uni)
            return w
        return fn

    def fixed_fn(self, weights: dict):
        w0 = np.zeros(self.S)
        for k, v in weights.items():
            w0[self.col_ix[k]] = v
        return lambda j: w0.copy()

    # ---- simulator: signal at close j, execute at close j+1 ----
    def simulate(self, weight_fn, cost_bps: float = COST_BPS, first: int | None = None, lag: int = 1):
        cost = cost_bps / 10000.0
        R, T, S = self.R, self.T, self.S
        rebal = set(self.rebal_idx)
        E, pos, cash = 1.0, np.zeros(S), 1.0
        daily = np.zeros(T)
        cashfrac = np.full(T, np.nan)
        contrib = np.zeros(S)
        turns, pending = [], None
        for j in range(first, T):
            Eprev = E
            if j > first:
                r = R[j]
                contrib += pos * r / Eprev
                pos = pos * (1.0 + r)
                E = cash + pos.sum()
            if pending is not None and j == pending[0]:
                w = pending[1]
                oldw = pos / E
                t = float(np.abs(w - oldw).sum())
                turns.append(t)
                E *= (1.0 - t * cost)
                pos = w * E
                cash = E - pos.sum()
                pending = None
            daily[j] = E / Eprev - 1.0 if j > first else 0.0
            cashfrac[j] = cash / E
            if j in rebal and j + lag < T:
                pending = (j + lag, weight_fn(j))
        return pd.Series(daily, index=self.dates), pd.Series(cashfrac, index=self.dates), contrib, turns


def rolling_cagr(r: pd.Series, win: int = ROLL_WIN) -> pd.Series:
    lg = np.log1p(r)
    return np.expm1(lg.rolling(win).sum() * (252.0 / win)).dropna()


def window_ret(r: pd.Series, lo, hi):
    sub = r[(r.index >= lo) & (r.index <= hi)]
    return None if len(sub) == 0 else float(((1 + sub).prod() - 1) * 100)


def run_study(close: pd.DataFrame, vol: pd.DataFrame, feed: str = "n/a", write: bool = True) -> dict:
    st = Study(close, vol)
    elig_counts = {j: len(st.eligible(j)) for j in st.rebal_idx if j >= LB12}
    starts = [j for j, n in elig_counts.items() if n >= MIN_ELIGIBLE_START]
    if not starts:
        raise SystemExit(f"never reached {MIN_ELIGIBLE_START} eligible names; max seen {max(elig_counts.values()) if elig_counts else 0}")
    first = starts[0]
    eval_i = first + 2
    print(f"\ndata: {st.S} symbols incl. benchmarks, {st.dates[0].date()} -> {st.dates[-1].date()}, feed={feed}")
    print(f"evaluation start: signal {st.dates[first].date()} (first month-end with >= {MIN_ELIGIBLE_START} eligible "
          f"names: {elig_counts[first]}), compared from {st.dates[eval_i].date()}")
    print(f"data hygiene: {st.big_moves} symbol-days with |daily return| > 50% (not cleaned)")
    sizes = [elig_counts[j] for j in st.rebal_idx if j >= first]
    print(f"eligible names per month-end: min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)}")
    print("REMINDER: survivorship-biased universe, one short mostly-bull sample; a pass is weak evidence.\n")

    def run(fn, bps=COST_BPS):
        return st.simulate(fn, bps, first)

    runs = {}
    runs["MOM base (blend 12/6, buffer)"] = run(st.momentum_fn("blend12_6"))
    runs["MOM 12-1 only (buffer)"] = run(st.momentum_fn("mom12"))
    runs["MOM 9-1 only (buffer)"] = run(st.momentum_fn("mom9"))
    runs["MOM blend 9/6 (buffer)"] = run(st.momentum_fn("blend9_6"))
    runs["PLAIN 12-1 (no buffer)"] = run(st.momentum_fn("mom12", use_buffer=False))
    runs["EW screened universe"] = run(st.equal_weight_fn())
    runs["SPY buy&hold"] = run(st.fixed_fn({"SPY": 1.0}))
    runs["60/40 (SPY/IEF)"] = run(st.fixed_fn({"SPY": 0.6, "IEF": 0.4}))
    series = {k: v[0].iloc[eval_i:] for k, v in runs.items()}

    out = dict(feed=feed, first_signal=str(st.dates[first].date()), eval_start=str(st.dates[eval_i].date()),
               data_end=str(st.dates[-1].date()), n_symbols=st.S, big_moves=st.big_moves,
               overall={}, halves={}, calendar_years={}, june_2026={}, sensitivity={}, verdict={})

    print("=== OVERALL (10 bp/side, signal->next-day-close execution) ===")
    for k, r in series.items():
        m = metrics(r)
        out["overall"][k] = m
        print(f"  {k:32s} {fmt(m)}")
    base = runs["MOM base (blend 12/6, buffer)"]
    print(f"  base: avg one-way turnover per rebalance {np.mean(base[3]):.2f}x equity, "
          f"avg cash fraction {base[1].iloc[eval_i:].mean()*100:.1f}%")

    mid = series["SPY buy&hold"].index[len(series['SPY buy&hold']) // 2]
    print(f"\n=== FIRST vs SECOND HALF of the sample (information only; split at {mid.date()}) ===")
    for lab, sel in [("first half", lambda r: r[r.index < mid]), ("second half", lambda r: r[r.index >= mid])]:
        out["halves"][lab] = {}
        print(f"  --- {lab} ---")
        for k in ["MOM base (blend 12/6, buffer)", "EW screened universe", "SPY buy&hold", "60/40 (SPY/IEF)"]:
            m = metrics(sel(series[k]))
            out["halves"][lab][k] = m
            print(f"    {k:32s} {fmt(m)}")

    print("\n=== CALENDAR YEARS (total return %) ===")
    ks = ["MOM base (blend 12/6, buffer)", "PLAIN 12-1 (no buffer)", "EW screened universe", "SPY buy&hold", "60/40 (SPY/IEF)"]
    print(f"  {'year':6s}" + "".join(f"{k[:14]:>16s}" for k in ks))
    for y in sorted(set(series[ks[0]].index.year)):
        row = {}
        for k in ks:
            sub = series[k][series[k].index.year == y]
            row[k] = float(((1 + sub).prod() - 1) * 100)
        out["calendar_years"][int(y)] = row
        print(f"  {y:<6d}" + "".join(f"{row[k]:16.1f}" for k in ks))

    print(f"\n=== JUNE 2026 WINDOW ({JUNE_START.date()} -> {JUNE_END.date()}), on its own ===")
    for k, r in series.items():
        w = window_ret(r, JUNE_START, JUNE_END)
        out["june_2026"][k] = w
        print(f"  {k:32s} " + ("no data" if w is None else f"{w:.2f}%"))

    print(f"\n=== ROLLING {ROLL_WIN}-DAY (~5y) CAGR: base vs SPY / 60-40 / EW (heavily overlapping windows) ===")
    rb = rolling_cagr(series["MOM base (blend 12/6, buffer)"])
    roll = {}
    for comp in ["SPY buy&hold", "60/40 (SPY/IEF)", "EW screened universe"]:
        rc = rolling_cagr(series[comp])
        both = rb.index.intersection(rc.index)
        if len(both) == 0:
            roll[comp] = (float("nan"), 0)
            print(f"  vs {comp:22s} no complete 5y window (sample too short)")
            continue
        win = float((rb.loc[both] > rc.loc[both]).mean())
        roll[comp] = (win, len(both))
        print(f"  vs {comp:22s} windows={len(both):5d} base beats in {win*100:5.1f}%  "
              f"median CAGR diff {(rb.loc[both]-rc.loc[both]).median()*100:+.2f} pts")
    out["rolling_5y"] = {k: dict(win_frac=v[0], windows=v[1]) for k, v in roll.items()}

    print("\n=== SENSITIVITY (information only, no verdict): costs and universe size ===")
    for bps in (5.0, 10.0, 20.0):
        m = metrics(st.simulate(st.momentum_fn("blend12_6"), bps, first)[0].iloc[eval_i:])
        out["sensitivity"][f"cost_{bps:g}bp"] = m
        print(f"  base @ {bps:4.0f} bp/side        {fmt(m)}")
    for n in (500, 2000):
        m = metrics(st.simulate(st.momentum_fn("blend12_6", n_univ=n), COST_BPS, first)[0].iloc[eval_i:])
        out["sensitivity"][f"universe_{n}"] = m
        print(f"  base, universe top {n:<5d}    {fmt(m)}")

    contrib = base[2]
    names = np.array(st.cols)
    order = np.argsort(-contrib)
    tot = float(contrib.sum())
    top10 = float(contrib[order[:10]].sum())
    share = top10 / tot if tot > 0 else float("nan")
    print(f"\n=== CONCENTRATION (base) ===")
    print("  top 10 contributors (sum of daily P&L contribution, % of starting equity): " +
          ", ".join(f"{names[i]} {contrib[i]*100:.1f}" for i in order[:10]))
    print(f"  top-10 share of total net contribution: {share*100:.0f}%  (total {tot*100:.1f}%)")
    out["concentration"] = dict(top10_share=share, total_contribution=tot,
                                top10=[(str(names[i]), float(contrib[i])) for i in order[:10]])

    print("\n=== VERDICT (predeclared; base variant, 10 bp) ===")
    m = out["overall"]
    spy, sf, bm = m["SPY buy&hold"], m["60/40 (SPY/IEF)"], m["MOM base (blend 12/6, buffer)"]
    c_i = bm["calmar"] > spy["calmar"] and bm["calmar"] > sf["calmar"]
    c_ii = bm["cagr"] > spy["cagr"]
    c_iii = all(np.isfinite(roll[k][0]) and roll[k][0] > 0.5 for k in ["SPY buy&hold", "60/40 (SPY/IEF)"])
    var_names = ["MOM base (blend 12/6, buffer)", "MOM 12-1 only (buffer)", "MOM 9-1 only (buffer)", "MOM blend 9/6 (buffer)"]
    var_ok = [m[k]["calmar"] > spy["calmar"] and m[k]["cagr"] > spy["cagr"] for k in var_names]
    c_iv = sum(var_ok) >= 3
    c_v = bool(np.isfinite(share) and share < 0.5)
    print(f"  (i)   Calmar {bm['calmar']:.2f} vs SPY {spy['calmar']:.2f} / 60-40 {sf['calmar']:.2f}  -> {c_i}")
    print(f"  (ii)  CAGR {bm['cagr']*100:.2f}% vs SPY {spy['cagr']*100:.2f}%  -> {c_ii}")
    print(f"  (iii) rolling-5y win rates vs SPY {roll['SPY buy&hold'][0]*100:.1f}% / 60-40 {roll['60/40 (SPY/IEF)'][0]*100:.1f}%  -> {c_iii}")
    print(f"  (iv)  lookback variants clearing (i)&(ii) vs SPY: {sum(var_ok)}/4 ({dict(zip(var_names, var_ok))})  -> {c_iv}")
    print(f"  (v)   top-10 contributors' share of profit {share*100:.0f}% (< 50% required)  -> {c_v}")
    passed = bool(c_i and c_ii and c_iii and c_iv and c_v)
    out["verdict"] = dict(i=bool(c_i), ii=bool(c_ii), iii=bool(c_iii), iv=bool(c_iv), v=c_v, passed=passed)
    print(f"\n  OVERALL VERDICT: {'PASS' if passed else 'FAIL'}  -- on a survivorship-biased universe, no quality score, "
          f"one short sample. A PASS is not validation of the full spec; a FAIL is informative.")

    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {OUT_PATH}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", default="auto", choices=["auto", "sip", "iex"])
    ap.add_argument("--refresh-universe", action="store_true")
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=== universe + bars ===")
    syms = get_universe(args.refresh_universe)
    feed = fetch_bars(syms, args.feed, "all")
    close, vol, feed_read = load_panels()
    print(f"loaded panel: {close.shape[1]} symbols, {close.index.min().date()} -> {close.index.max().date()}")
    for b in BENCH:
        if b not in close.columns:
            raise SystemExit(f"benchmark {b} missing from fetched data")
    run_study(close, vol, feed_read or feed)


if __name__ == "__main__":
    main()
