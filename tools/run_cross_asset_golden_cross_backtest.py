"""Applies the exact same golden-cross (50/200 MA) methodology already
verified on crypto -- entry when 50MA crosses above 200MA, exit on the
death cross, same robustness bar (profit factor >= 1.0 excluding the 3
biggest winning trades) -- to the full 116-instrument cross-asset universe
built for cross_asset_tracker.py: equities, commodities, bonds/rates,
forex, crypto.

Uses DAILY bars here (not crypto's 4h) since that's what's available for
these instruments via Yahoo Finance, and daily 50/200 is in fact the
traditional definition of "golden cross" for stocks/commodities.

Cost assumption: same 12bps round-trip (5bps fee + 7bps slippage) used
throughout the crypto backtests, applied uniformly across every asset class
here so the comparison isolates the strategy's real edge rather than being
confounded by different cost models per asset class. This is a real
simplification -- actual futures/forex/equity trading costs vary by venue
and instrument -- flagged, not hidden.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests

from cross_asset_tracker import UNIVERSE

ENTRY_COST = 12 / 10000
EXIT_COST = 12 / 10000
BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 2.0
MIN_TRADES = 6  # below this, robustness stats are too noisy to call either way
STARTING_CAPITAL = 100_000.0
# 116 instruments can have far more concurrent positions open than crypto's
# 6-asset system (which used 15% per asset) -- 2% keeps total exposure
# sane even if a large fraction end up open at once, same "sized off current
# capital at entry, realized at exit" simplification used throughout.
PER_TRADE_ALLOCATION_PCT = 2.0
# Same check added to cross_asset_tracker.py after UNI-USD turned out to be
# silently stuck at a stale/wrong price for 500+ days -- must be applied
# here too, this script fetches its own series independently.
MAX_STALE_DAYS = 5
OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "cross_asset_golden_cross_results.json"


def compute_trend_state(close: pd.Series) -> pd.DataFrame:
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    above200 = ma50 > ma200
    # dtype bug fixed in ma200_trend.py/bmsb_trend.py this session -- applying
    # the same fix here from the start rather than reintroducing it.
    prev_above200 = above200.shift(1).fillna(False).astype(bool)
    cross_up = above200 & ~prev_above200
    cross_down = (~above200) & prev_above200
    return pd.DataFrame({"close": close, "ma50": ma50, "ma200": ma200,
                          "above200": above200, "cross_up": cross_up, "cross_down": cross_down})


def backtest_one(close: pd.Series) -> list[dict]:
    state = compute_trend_state(close)
    trades = []
    pos = None
    for ts, row in state.iterrows():
        if pd.isna(row["ma200"]):
            continue
        if pos is None and row["cross_up"]:
            pos = {"entry_time": ts, "entry_price": row["close"]}
        elif pos is not None and row["cross_down"]:
            fill_entry = pos["entry_price"] * (1 + ENTRY_COST)
            fill_exit = row["close"] * (1 - EXIT_COST)
            net_ret_pct = (fill_exit / fill_entry - 1) * 100
            trades.append({
                "entry_time": pos["entry_time"], "exit_time": ts, "net_ret_pct": net_ret_pct,
                "entry_price": float(fill_entry), "exit_price": float(fill_exit),
            })
            pos = None
    return trades


def robustness(trades: list[dict]) -> dict | None:
    if len(trades) < MIN_TRADES:
        return None
    rets = pd.Series([t["net_ret_pct"] for t in trades])
    wins, losses = rets[rets > 0], rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
    sorted_rets = rets.sort_values(ascending=False).reset_index(drop=True)
    excl3 = sorted_rets.iloc[3:] if len(rets) > 3 else sorted_rets.iloc[0:0]
    w3, l3 = excl3[excl3 > 0].sum(), abs(excl3[excl3 < 0].sum())
    pf_excl3 = w3 / l3 if l3 > 0 else float("inf")
    return {
        "n_trades": int(len(trades)), "win_rate": float((rets > 0).mean() * 100),
        "pf": float(pf), "pf_excl3": float(pf_excl3), "sum_excl3": float(excl3.sum()),
        "robust": bool(pf_excl3 >= 1.0 and excl3.sum() > 0),
    }


def fetch_all_close_series() -> dict[str, tuple[str, pd.Series]]:
    session = cffi_requests.Session(impersonate="chrome")
    out: dict[str, tuple[str, pd.Series]] = {}
    for asset_class, symbols in UNIVERSE.items():
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            try:
                data = yf.download(batch, period="max", interval="1d", session=session,
                                    progress=False, group_by="ticker", threads=False)
            except Exception as exc:
                print(f"batch failed ({asset_class}): {exc}")
                time.sleep(BATCH_DELAY_SECONDS)
                continue
            for sym in batch:
                try:
                    close = data[sym]["Close"] if len(batch) > 1 else data["Close"]
                except (KeyError, TypeError):
                    continue
                close = close.dropna()
                if len(close) < 210:
                    continue
                last_date = close.index[-1]
                if last_date.tzinfo is None:
                    last_date = last_date.tz_localize("UTC")
                age_days = (datetime.now(timezone.utc) - last_date).days
                if age_days > MAX_STALE_DAYS:
                    print(f"skipping {sym}: stale data, last observation {last_date.date()} ({age_days}d ago)")
                    continue
                out[sym] = (asset_class, close)
            time.sleep(BATCH_DELAY_SECONDS)
    return out


def simulate_portfolio(all_trades: list[dict]) -> dict:
    """Event-driven (not bar-by-bar -- the trade list is already computed)
    portfolio sim: each trade is sized as a % of capital AT ITS OWN ENTRY
    TIME, P&L realized at its own exit time, same "notional fixed at entry,
    capital updated at exit" simplification the crypto system uses (capital
    isn't reserved while a position is open, so aggregate exposure can
    exceed 100% if many positions are concurrently open -- an accepted
    simplification, not a bug)."""
    events = []
    for t in all_trades:
        events.append((t["entry_time"], 0, "entry", t))
        events.append((t["exit_time"], 1, "exit", t))
    events.sort(key=lambda e: (e[0], e[1]))

    capital = STARTING_CAPITAL
    peak = STARTING_CAPITAL
    open_count = 0
    class_running: dict[str, float] = {}
    class_pnl: dict[str, float] = {}

    equity_curve = []
    parallelism = []
    underwater = []
    class_profit_series: dict[str, list] = {}

    if events:
        equity_curve.append({"time": events[0][0].isoformat(), "equity": capital})

    for ts, _, kind, t in events:
        if kind == "entry":
            t["_notional"] = capital * PER_TRADE_ALLOCATION_PCT / 100.0
            open_count += 1
            parallelism.append({"time": ts.isoformat(), "count": open_count})
        else:
            open_count -= 1
            parallelism.append({"time": ts.isoformat(), "count": open_count})

            pnl = t["_notional"] * (t["net_ret_pct"] / 100.0)
            capital += pnl
            class_pnl[t["asset_class"]] = class_pnl.get(t["asset_class"], 0.0) + pnl
            equity_curve.append({"time": ts.isoformat(), "equity": capital})

            peak = max(peak, capital)
            underwater.append({"time": ts.isoformat(), "drawdown_pct": (capital / peak - 1) * 100})

            ac = t["asset_class"]
            class_running[ac] = class_running.get(ac, 0.0) + pnl
            series = class_profit_series.setdefault(ac, [])
            series.append({"time": ts.isoformat(), "pnl": class_running[ac]})

    return {
        "starting_capital": STARTING_CAPITAL,
        "final_capital": capital,
        "total_return_pct": (capital / STARTING_CAPITAL - 1) * 100,
        "max_drawdown_pct": min((u["drawdown_pct"] for u in underwater), default=0.0),
        "equity_curve": equity_curve,
        "class_pnl": class_pnl,
        "parallelism": parallelism,
        "underwater": underwater,
        "class_profit_series": class_profit_series,
    }


def main() -> None:
    print(f"fetching max history for {sum(len(v) for v in UNIVERSE.values())} instruments...")
    series = fetch_all_close_series()
    print(f"got usable history for {len(series)} instruments\n")

    results = []
    all_trades = []
    for sym, (asset_class, close) in series.items():
        trades = backtest_one(close)
        for t in trades:
            all_trades.append({**t, "symbol": sym, "asset_class": asset_class})
        r = robustness(trades)
        if r:
            results.append({"symbol": sym, "asset_class": asset_class, **r})

    portfolio = simulate_portfolio(all_trades)
    print(f"\nportfolio (all {len(all_trades)} trades, {PER_TRADE_ALLOCATION_PCT}% per trade): "
          f"${portfolio['starting_capital']:,.0f} -> ${portfolio['final_capital']:,.0f} "
          f"({portfolio['total_return_pct']:+.1f}%)")

    results.sort(key=lambda r: -r["sum_excl3"])
    print(f"{'Symbol':<12} {'Class':<14} {'N':>4} {'WinRate':>8} {'PF':>7} {'PF_x3':>7} {'Sum_x3':>9}  Robust")
    for r in results:
        print(f"{r['symbol']:<12} {r['asset_class']:<14} {r['n_trades']:>4} {r['win_rate']:>7.1f}% "
              f"{r['pf']:>7.2f} {r['pf_excl3']:>7.2f} {r['sum_excl3']:>8.1f}%  {'YES' if r['robust'] else 'no'}")

    robust = [r for r in results if r["robust"]]
    print(f"\ntested (>= {MIN_TRADES} trades): {len(results)} / {len(series)} instruments with usable history")
    print(f"ROBUST (PF excl top3 >= 1.0 and positive): {len(robust)}")
    print("robust symbols:", [r["symbol"] for r in robust])

    by_class: dict[str, list] = {}
    for r in results:
        by_class.setdefault(r["asset_class"], []).append(r)
    print("\nby asset class:")
    for ac, rows in by_class.items():
        rob = [r for r in rows if r["robust"]]
        print(f"  {ac}: {len(rows)} tested, {len(rob)} robust")

    # json.dumps would emit non-standard "Infinity" tokens for float('inf')
    # (real when a symbol has zero losing trades in-sample) -- browsers'
    # JSON.parse rejects that, so cap for display rather than lose the row.
    json_results = []
    for r in results:
        row = dict(r)
        row["pf"] = row["pf"] if row["pf"] != float("inf") else None
        row["pf_excl3"] = row["pf_excl3"] if row["pf_excl3"] != float("inf") else None
        json_results.append(row)

    # Full per-trade list for the Dashboard-style view (Closed Trades table,
    # Profit Distribution histogram, Daily Profit, Trades Log) -- simulate_portfolio
    # already attached "_notional" (the real stake for that trade) to each dict.
    json_trades = []
    for i, t in enumerate(all_trades):
        notional = t.get("_notional", 0.0)
        json_trades.append({
            "id": i, "symbol": t["symbol"], "asset_class": t["asset_class"],
            "entry_time": t["entry_time"].isoformat(), "exit_time": t["exit_time"].isoformat(),
            "net_ret_pct": t["net_ret_pct"], "stake_amount": notional,
            "profit_abs": notional * (t["net_ret_pct"] / 100.0),
            "open_rate": t["entry_price"], "close_rate": t["exit_price"],
            "amount": notional / t["entry_price"] if t["entry_price"] else 0.0,
        })
    json_trades.sort(key=lambda t: t["exit_time"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_bps_round_trip": (ENTRY_COST + EXIT_COST) * 10000,
        "min_trades": MIN_TRADES,
        "per_trade_allocation_pct": PER_TRADE_ALLOCATION_PCT,
        "results": json_results,
        "portfolio": portfolio,
        "trades": json_trades,
    }), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
