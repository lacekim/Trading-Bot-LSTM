"""Live (dry-run only) paper trader for the cross-asset golden-cross system --
the exact same 50/200-day MA rule already backtested in
run_cross_asset_golden_cross_backtest.py, run forward on real current data.

Deliberately restricted to the 61 instruments that actually passed the
robustness bar (PF >= 1.0 excluding the 3 biggest winning trades), read from
the real, already-computed cross_asset_golden_cross_results.json -- NOT all
116. Trading the other 55 (all 8 forex pairs, most commodities/bonds) would
directly contradict what this session already proved: they showed no real
edge under this rule. This is a real decision, not a silent default -- see
the tool's own README section below if you want to widen the universe later.

Dry-run / paper only. No real orders are ever placed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests

from cross_asset_tracker import UNIVERSE

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
RESULTS_PATH = RUNTIME_DIR / "cross_asset_golden_cross_results.json"
STATE_PATH = RUNTIME_DIR / "cross_asset_paper_trades.json"

STARTING_CAPITAL = 100_000.0
PER_ASSET_ALLOCATION_PCT = 2.0  # matches the backtest's own per-trade sizing
ENTRY_COST = 12 / 10000
EXIT_COST = 12 / 10000
MAX_STALE_DAYS = 5


def _load_robust_universe() -> dict[str, str]:
    """symbol -> asset_class, restricted to robust==True, real data only."""
    if not RESULTS_PATH.exists():
        raise SystemExit(f"no backtest results at {RESULTS_PATH} -- run "
                          f"tools/run_cross_asset_golden_cross_backtest.py first")
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {r["symbol"]: r["asset_class"] for r in data["results"] if r["robust"]}


def _fetch_close_series(symbol: str, session) -> pd.Series | None:
    try:
        df = yf.download(symbol, period="2y", interval="1d", session=session,
                          progress=False, auto_adjust=True)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 210:
        return None
    last_date = close.index[-1]
    if last_date.tzinfo is None:
        last_date = last_date.tz_localize("UTC")
    age_days = (datetime.now(timezone.utc) - last_date).days
    if age_days > MAX_STALE_DAYS:
        return None
    return close


def _compute_signal(close: pd.Series) -> dict:
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    above200 = ma50 > ma200
    prev_above200 = above200.shift(1).fillna(False).astype(bool)
    cross_up = bool((above200 & ~prev_above200).iloc[-1])
    cross_down = bool((~above200 & prev_above200).iloc[-1])
    return {
        "signal": "LONG" if bool(above200.iloc[-1]) else "FLAT",
        "close": float(close.iloc[-1]),
        "just_crossed_up": cross_up,
        "just_crossed_down": cross_down,
        "as_of": str(close.index[-1].date()),
    }


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"capital": STARTING_CAPITAL, "positions": {}, "closed_trades": []}


def save_state(state: dict) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    universe = _load_robust_universe()
    print(f"trading universe (robust only): {len(universe)} instruments")
    state = load_state()
    session = cffi_requests.Session(impersonate="chrome")

    current_prices: dict[str, float] = {}
    for symbol, asset_class in universe.items():
        close = _fetch_close_series(symbol, session)
        if close is None:
            print(f"{symbol}: no usable data this run, skipping")
            continue
        sig = _compute_signal(close)
        current_prices[symbol] = sig["close"]
        pos = state["positions"].get(symbol)

        if sig["signal"] == "LONG" and pos is None:
            allocated = state["capital"] * PER_ASSET_ALLOCATION_PCT / 100
            fill_price = sig["close"] * (1 + ENTRY_COST)
            state["positions"][symbol] = {
                "asset_class": asset_class, "entry_price": fill_price,
                "entry_time": sig["as_of"], "allocated_amount": allocated,
            }
            state["capital"] -= allocated
            print(f"🟢 OPEN {symbol} ({asset_class}) @ {fill_price:.4f}  allocated ${allocated:,.2f}")

        elif sig["signal"] == "FLAT" and pos is not None:
            fill_price = sig["close"] * (1 - EXIT_COST)
            net_ret_pct = (fill_price / pos["entry_price"] - 1) * 100
            allocated = pos["allocated_amount"]
            realized_pnl = allocated * (net_ret_pct / 100)
            state["capital"] += allocated + realized_pnl
            state["closed_trades"].append({
                "symbol": symbol, "asset_class": asset_class,
                "entry_time": pos["entry_time"], "exit_time": sig["as_of"],
                "entry_price": pos["entry_price"], "exit_price": fill_price,
                "net_ret_pct": net_ret_pct, "allocated_amount": allocated,
                "realized_pnl": realized_pnl,
            })
            del state["positions"][symbol]
            print(f"🔴 CLOSE {symbol} ({asset_class}) @ {fill_price:.4f}  "
                  f"net {net_ret_pct:+.2f}%  P&L ${realized_pnl:,.2f}")
        else:
            status = "holding" if pos is not None else "flat"
            print(f"{symbol}: {sig['signal']}, {status}")

    unrealized = 0.0
    for symbol, pos in state["positions"].items():
        if symbol in current_prices:
            move_pct = (current_prices[symbol] / pos["entry_price"] - 1) * 100
            unrealized += pos["allocated_amount"] * (move_pct / 100)
    equity = state["capital"] + sum(p["allocated_amount"] for p in state["positions"].values()) + unrealized

    print(f"\n=== equity summary ===")
    print(f"open positions: {len(state['positions'])}  |  closed trades: {len(state['closed_trades'])}")
    print(f"cash: ${state['capital']:,.2f}  |  unrealized P&L: ${unrealized:,.2f}")
    print(f"current equity: ${equity:,.2f}  (started ${STARTING_CAPITAL:,.2f}, "
          f"{(equity/STARTING_CAPITAL-1)*100:+.2f}%)")

    save_state(state)


if __name__ == "__main__":
    main()
