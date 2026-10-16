"""Automatic asset qualification for the golden-cross (50MA/200MA) system.

Replaces a hand-picked, static asset list with a real, periodic re-scan: for
every candidate in the available universe, run the same golden-cross
backtest and the same strict robustness check used throughout this session
(still profitable after excluding the 3 biggest winning trades) -- only
assets that currently pass are traded. Assets get added or dropped
automatically as their own data changes, not because someone hand-picked
them once.

This is a real backtest sweep (not a cheap check), so it's meant to run on
a slow schedule (see QUALIFICATION_INTERVAL_DAYS), not on every signal
check -- results are cached to disk with a timestamp and reused until the
next scheduled re-scan.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_ma200_asset_qualifier")

# Every symbol with real local Kraken history available -- the full
# candidate universe, not a pre-selected shortlist.
CANDIDATE_UNIVERSE = ['ADA','ALGO','ANKR','APT','ARB','ATOM','AVAX','BAL','BNB','BTC','DOGE','DOT',
                       'ETH','FIL','ICP','ICX','KSM','LINK','LRC','NEAR','OP','PAXG','SOL','TRX',
                       'XLM','XMR','XRP']
PAIR_OVERRIDE = {'BTC': 'XBTUSD', 'DOGE': 'XDGUSD'}
DATA_DIR = Path('/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken')

ENTRY_COST = 12 / 10000
EXIT_COST = 12 / 10000
MIN_TRADES = 8
QUALIFICATION_PATH = Path(__file__).resolve().parents[2] / "runtime" / "ma200_qualified_assets.json"
QUALIFICATION_INTERVAL_DAYS = 7  # real backtest sweep -- not run on every signal check


def _fetch_fresh(pair: str, since_ts: int) -> pd.DataFrame:
    all_rows = []
    since = since_ts
    for _ in range(6):
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240&since={since}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read())
        except Exception:
            break
        if not data.get("result"):
            break
        keys = [k for k in data["result"].keys() if k != "last"]
        if not keys:
            break
        rows = data["result"][keys[0]]
        if not rows:
            break
        all_rows.extend(rows)
        new_since = int(data["result"]["last"])
        if new_since <= since:
            break
        since = new_since
        time.sleep(0.4)
    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz="UTC"))
    fresh = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"]).drop_duplicates("ts")
    fresh["dt"] = pd.to_datetime(fresh["ts"].astype(float), unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        fresh[c] = fresh[c].astype(float)
    return fresh.set_index("dt")[["open", "high", "low", "close", "volume"]]


def _load_full_history(symbol: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{symbol}_USD-1h.feather"
    if not path.exists():
        return None
    h1 = pd.read_feather(path)
    h1["date"] = pd.to_datetime(h1["date"], utc=True)
    h1 = h1.set_index("date")
    hist4h = h1.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    pair = PAIR_OVERRIDE.get(symbol, f"{symbol}USD")
    fresh = _fetch_fresh(pair, int(hist4h.index.max().timestamp()))
    if len(fresh) > 0:
        fresh = fresh[fresh.index > hist4h.index.max()]
    df = pd.concat([hist4h, fresh]).sort_index()
    return df[~df.index.duplicated()]


def _evaluate_golden_cross(symbol: str) -> dict:
    df = _load_full_history(symbol)
    if df is None or len(df) < 210:
        return {"symbol": symbol, "qualified": False, "reason": "insufficient data"}

    close = df["close"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    state = pd.DataFrame({"close": close, "ma50": ma50, "ma200": ma200}).dropna()
    state["golden"] = state["ma50"] > state["ma200"]
    close_arr = state["close"].to_numpy()
    golden = state["golden"].to_numpy()

    trades = []
    in_pos = False
    for i in range(1, len(state)):
        if not in_pos and golden[i] and not golden[i - 1]:
            in_pos = True
            entry_price = close_arr[i]
        elif in_pos and not golden[i] and golden[i - 1]:
            net_ret = (close_arr[i] * (1 - EXIT_COST)) / (entry_price * (1 + ENTRY_COST)) - 1
            trades.append(net_ret * 100)
            in_pos = False

    t = pd.Series(trades)
    if len(t) < MIN_TRADES:
        return {"symbol": symbol, "qualified": False, "reason": f"only {len(t)} trades"}

    sorted_t = t.sort_values(ascending=False).reset_index(drop=True)
    excl3 = sorted_t.iloc[3:] if len(t) > 3 else sorted_t.iloc[0:0]
    wins_excl3 = excl3[excl3 > 0].sum()
    losses_excl3 = abs(excl3[excl3 < 0].sum())
    pf_excl3 = wins_excl3 / losses_excl3 if losses_excl3 > 0 else float("inf")
    qualified = pf_excl3 >= 1.0 and excl3.sum() > 0

    return {
        "symbol": symbol, "qualified": bool(qualified), "n_trades": int(len(t)),
        "win_rate_pct": float((t > 0).mean() * 100), "pf_excl_top3": float(pf_excl3),
        "sum_excl_top3_pct": float(excl3.sum()),
    }


def run_qualification_sweep() -> dict:
    """Real backtest across the full candidate universe. Slow (a real sweep,
    not a cheap check) -- call on a schedule, not per signal check."""
    logger.info(f"running qualification sweep over {len(CANDIDATE_UNIVERSE)} candidates...")
    results = {}
    for symbol in CANDIDATE_UNIVERSE:
        try:
            results[symbol] = _evaluate_golden_cross(symbol)
        except Exception as exc:
            logger.error(f"{symbol}: qualification failed: {exc}")
            results[symbol] = {"symbol": symbol, "qualified": False, "reason": f"error: {exc}"}

    qualified_symbols = sorted(s for s, r in results.items() if r.get("qualified"))
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "qualified_symbols": qualified_symbols,
        "details": results,
    }
    QUALIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALIFICATION_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"qualification sweep complete: {qualified_symbols}")
    return output


def get_qualified_assets(force_refresh: bool = False) -> list[str]:
    """Returns the current auto-qualified symbol list, re-running the sweep
    if the cached result is missing or older than QUALIFICATION_INTERVAL_DAYS."""
    if not force_refresh and QUALIFICATION_PATH.exists():
        cached = json.loads(QUALIFICATION_PATH.read_text(encoding="utf-8"))
        run_at = datetime.fromisoformat(cached["run_at"])
        age_days = (datetime.now(timezone.utc) - run_at).total_seconds() / 86400
        if age_days < QUALIFICATION_INTERVAL_DAYS:
            return cached["qualified_symbols"]
    return run_qualification_sweep()["qualified_symbols"]
