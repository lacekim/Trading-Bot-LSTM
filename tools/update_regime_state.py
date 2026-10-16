"""Fetches HYG/IEF/UUP, computes today's intermarket regime using the exact
same rule as run_intermarket_regime_filter_test.py's build_regime_series
(credit risk-on via HYG/IEF ratio > 200MA, dollar not a headwind via UUP <=
200MA, FOMC +/-1 day event blackout), and writes a small cached state file
that GoldenCrossRegimeFilteredStrategy reads before allowing any entry.

Written as a periodically-run updater (not an inline strategy fetch) for the
same reason Config.MARKET_CAP_RISK_ENABLED works this way elsewhere in this
project: a live strategy shouldn't make ad hoc network calls per candle, and
a cached-with-timestamp file lets the strategy fail safe (deny entry) if the
cache goes stale, the same real bug class found and fixed for the market-cap
snapshot earlier this project (69.9h stale, past its 48h gate, silently
rejecting everything). This one is explicit about its own freshness instead
of assuming it's being run on schedule.

Run daily (once market data updates) via a cron/loop, same pattern as
run_cross_asset_paper_trader_loop.py.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yfinance as yf

from tools.run_intermarket_regime_filter_test import FOMC_DATES

STATE_PATH = Path(__file__).resolve().parents[1] / "runtime" / "regime_state.json"


def fetch_close(ticker: str, lookback_days: int = 400) -> pd.Series:
    df = yf.download(ticker, period=f"{lookback_days}d", progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def compute_today_regime() -> dict:
    hyg = fetch_close("HYG")
    ief = fetch_close("IEF")
    uup = fetch_close("UUP")

    ratio = (hyg / ief).dropna()
    ratio_ma200 = ratio.rolling(200).mean()
    credit_risk_on = bool(ratio.iloc[-1] > ratio_ma200.iloc[-1]) if len(ratio) >= 200 else False

    uup_ma200 = uup.rolling(200).mean()
    dollar_not_headwind = bool(uup.iloc[-1] <= uup_ma200.iloc[-1]) if len(uup) >= 200 else False

    today = pd.Timestamp.now(tz="UTC").normalize()
    in_event_window = any(abs((today - d).days) <= 1 for d in FOMC_DATES)

    risk_on = credit_risk_on and dollar_not_headwind and not in_event_window

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "as_of_data_date": str(ratio.index[-1].date()) if len(ratio) else None,
        "credit_risk_on": credit_risk_on,
        "dollar_not_headwind": dollar_not_headwind,
        "in_fomc_event_window": in_event_window,
        "risk_on": risk_on,
        "hyg_ief_ratio": float(ratio.iloc[-1]) if len(ratio) else None,
        "hyg_ief_ratio_ma200": float(ratio_ma200.iloc[-1]) if len(ratio) >= 200 else None,
        "uup_close": float(uup.iloc[-1]) if len(uup) else None,
        "uup_ma200": float(uup_ma200.iloc[-1]) if len(uup) >= 200 else None,
    }


if __name__ == "__main__":
    state = compute_today_regime()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    print(f"\nwrote {STATE_PATH}")
