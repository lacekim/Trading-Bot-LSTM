"""Standalone cross-asset regime tracker.

Not wired into the crypto bot -- a separate, informational view answering
"what's currently trending up/down across different asset classes and
instruments right now." Different asset classes move on different clocks
(rate cycles, seasonal commodity supply/demand, crypto's own cycle, sector
rotation), so a single snapshot across all of them shows where each one
currently sits, using the same trend framework already verified for crypto
in this project (50/200-day MA state), applied broadly instead of to one
market.

Data: Yahoo Finance via yfinance, using a curl_cffi session with browser
impersonation -- Yahoo blocks yfinance's default request signature outright
(confirmed: immediate YFRateLimitError on every ticker without it).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests

UNIVERSE: dict[str, list[str]] = {
    "Equities": [
        "SPY", "QQQ", "DIA", "IWM", "VTI",
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "ADBE", "AMD",
        "JPM", "BAC", "WFC", "GS", "MS", "BRK-B", "V", "MA", "AXP",
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "BKNG",
        "WMT", "PG", "KO", "PEP", "COST", "PM",
        "CAT", "BA", "HON", "UPS", "GE", "LMT", "RTX",
        "XOM", "CVX", "COP", "SLB", "EOG",
        "NEE", "DUK", "SO",
        "LIN", "SHW", "FCX", "NEM",
        "PLD", "AMT", "SPG",
        "NFLX", "DIS", "CMCSA", "T", "VZ",
    ],
    "Commodities": [
        "GC=F", "SI=F", "PL=F", "PA=F", "HG=F",
        "CL=F", "BZ=F", "NG=F",
        "ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F", "CT=F", "CC=F",
        "LE=F", "HE=F",
    ],
    "Bonds/Rates": ["SHY", "IEF", "TLT", "^TNX", "^TYX"],
    "Forex": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
        "USDCAD=X", "AUDUSD=X", "NZDUSD=X", "DX-Y.NYB",
    ],
    "Crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
        "AVAX-USD", "LINK-USD", "DOT-USD", "LTC-USD", "BCH-USD", "ATOM-USD",
        "XMR-USD", "TRX-USD", "UNI-USD", "NEAR-USD",
    ],
}

BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 2.0
# Yahoo will silently return a real-looking series for a dead/mismatched
# ticker instead of erroring -- confirmed on UNI-USD: real data stopped on
# 2025-04-17 and every row after that was a flat forward-filled stale value
# (0.000163, nowhere near Uniswap's real price -- almost certainly a ticker
# collision with some unrelated low-cap token). Any symbol whose last real
# observation is older than this gets rejected rather than silently reported.
MAX_STALE_DAYS = 5


@dataclass
class InstrumentState:
    symbol: str
    asset_class: str
    price: float
    ma50: float
    ma200: float
    trend: str
    dist_from_200ma_pct: float
    ret_1m_pct: float
    ret_3m_pct: float
    ret_6m_pct: float
    ret_1y_pct: float
    pct_off_52w_high: float


def _trend_state(price: float, ma50: float, ma200: float) -> str:
    if ma50 > ma200 and price > ma50:
        return "BULL"
    if ma50 < ma200 and price < ma50:
        return "BEAR"
    return "TRANSITION"


def _pct_return(close: pd.Series, days_back: int) -> float | None:
    if len(close) <= days_back:
        return None
    return (close.iloc[-1] / close.iloc[-1 - days_back] - 1) * 100


def _compute_state(symbol: str, asset_class: str, close: pd.Series) -> InstrumentState | None:
    close = close.dropna()
    if len(close) < 210:
        return None
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    price = close.iloc[-1]
    high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    ret_1m = _pct_return(close, 21)
    ret_3m = _pct_return(close, 63)
    ret_6m = _pct_return(close, 126)
    ret_1y = _pct_return(close, 252)
    return InstrumentState(
        symbol=symbol, asset_class=asset_class, price=float(price),
        ma50=float(ma50), ma200=float(ma200),
        trend=_trend_state(price, ma50, ma200),
        dist_from_200ma_pct=float((price / ma200 - 1) * 100),
        ret_1m_pct=ret_1m if ret_1m is not None else float("nan"),
        ret_3m_pct=ret_3m if ret_3m is not None else float("nan"),
        ret_6m_pct=ret_6m if ret_6m is not None else float("nan"),
        ret_1y_pct=ret_1y if ret_1y is not None else float("nan"),
        pct_off_52w_high=float((price / high_52w - 1) * 100),
    )


def fetch_all() -> tuple[list[InstrumentState], list[tuple[str, str, str]]]:
    session = cffi_requests.Session(impersonate="chrome")
    results: list[InstrumentState] = []
    failures: list[tuple[str, str, str]] = []

    for asset_class, symbols in UNIVERSE.items():
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            try:
                data = yf.download(
                    batch, period="3y", interval="1d", session=session,
                    progress=False, group_by="ticker", threads=False,
                )
            except Exception as exc:
                for s in batch:
                    failures.append((s, asset_class, f"batch download error: {exc}"))
                time.sleep(BATCH_DELAY_SECONDS)
                continue

            for sym in batch:
                try:
                    close = data[sym]["Close"] if len(batch) > 1 else data["Close"]
                except (KeyError, TypeError):
                    failures.append((sym, asset_class, "no data returned"))
                    continue
                close = close.dropna()
                if len(close) == 0:
                    failures.append((sym, asset_class, "no data returned"))
                    continue
                last_date = close.index[-1]
                if last_date.tzinfo is None:
                    last_date = last_date.tz_localize("UTC")
                age_days = (datetime.now(timezone.utc) - last_date).days
                if age_days > MAX_STALE_DAYS:
                    failures.append((sym, asset_class, f"stale data -- last real observation {last_date.date()} ({age_days}d ago)"))
                    continue
                state = _compute_state(sym, asset_class, close)
                if state is None:
                    failures.append((sym, asset_class, "insufficient history (<210 daily bars)"))
                else:
                    results.append(state)
            time.sleep(BATCH_DELAY_SECONDS)
    return results, failures


def print_report(results: list[InstrumentState], failures: list[tuple[str, str, str]]) -> None:
    by_class: dict[str, list[InstrumentState]] = {}
    for r in results:
        by_class.setdefault(r.asset_class, []).append(r)

    for asset_class in UNIVERSE:
        rows = by_class.get(asset_class, [])
        if not rows:
            continue
        rows.sort(key=lambda r: (r.trend != "BULL", -r.ret_3m_pct if r.ret_3m_pct == r.ret_3m_pct else 0))
        print(f"\n=== {asset_class} ===")
        print(f"{'Symbol':<10} {'Trend':<11} {'Price':>12} {'vs200MA':>9} {'1M':>7} {'3M':>7} {'6M':>7} {'1Y':>7} {'Off52wHi':>9}")
        for r in rows:
            print(f"{r.symbol:<10} {r.trend:<11} {r.price:>12,.2f} {r.dist_from_200ma_pct:>+8.1f}% "
                  f"{r.ret_1m_pct:>+6.1f}% {r.ret_3m_pct:>+6.1f}% {r.ret_6m_pct:>+6.1f}% {r.ret_1y_pct:>+6.1f}% "
                  f"{r.pct_off_52w_high:>+8.1f}%")

    bull_ct = sum(1 for r in results if r.trend == "BULL")
    bear_ct = sum(1 for r in results if r.trend == "BEAR")
    trans_ct = sum(1 for r in results if r.trend == "TRANSITION")
    print(f"\n=== Summary ===")
    print(f"tracked: {len(results)}  |  BULL: {bull_ct}  BEAR: {bear_ct}  TRANSITION: {trans_ct}")
    if failures:
        print(f"\n{len(failures)} symbols failed / had insufficient data:")
        for sym, ac, reason in failures:
            print(f"  {sym} ({ac}): {reason}")


def save_report(results: list[InstrumentState]) -> None:
    from pathlib import Path
    out_path = Path(__file__).resolve().parents[1] / "runtime" / "cross_asset_snapshot.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")


def main() -> None:
    print(f"fetching {sum(len(v) for v in UNIVERSE.values())} instruments across "
          f"{len(UNIVERSE)} asset classes...")
    results, failures = fetch_all()
    print_report(results, failures)
    save_report(results)


if __name__ == "__main__":
    sys.exit(main())
