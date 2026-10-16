"""Re-runs the exact same golden-cross (50/200 MA) robustness test
run_cross_asset_golden_cross_backtest.py already did on DAILY Yahoo bars --
same entry/exit rule, same PF-excl-top3 robustness bar, same 12bps
round-trip cost -- but on 4-HOUR Alpaca bars instead, for the current
60-instrument robust-on-daily universe (57 equities + IEF + the two
futures-to-ETF proxies).

Why this exists (2026-09-21): the user asked to switch the LIVE equities
golden-cross signal (tools/run_alpaca_paper_trader.py) from daily Yahoo
bars to 4h Alpaca bars, to match both the crypto side (already 4h Kraken)
and the MA lines on freqUI's own Trade-page chart (also 4h, added earlier
today). Before touching anything live, this checks whether the SAME rule
even holds up on 4h bars at all -- every other change in this project has
had to clear this exact bar first (see Section E/E5's own stoploss
calibration work), so this one doesn't get to skip it just because it's
a timeframe change rather than a strategy change.

Real, load-bearing caveat, not hidden: 50/200 on 4h bars is NOT the same
lookback window as 50/200 on daily bars. With ~2 four-hour bars per
trading session, a 200-bar 4h MA covers roughly 100 trading days (~5
months) of price history, versus a 200-bar DAILY MA covering roughly 200
trading days (~10 months). This is a materially faster, noisier version
of "golden cross," not just a re-sampling of the same signal -- which is
exactly why it needs its own robustness check rather than being assumed
to inherit the daily version's PF 4.48 / -12% hard-stop numbers.

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in the environment (same
.env this project's other Alpaca scripts already use) -- Alpaca's own
historical bars API needs them even for read-only market data, so this
can't run without them, same as run_alpaca_paper_trader.py.

Run this yourself (this session's device shell has been down all day):
    cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
    python tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py
Paste the printed comparison table back so we can see whether the 4h
version actually holds up before anything live gets touched.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DAILY_RESULTS_PATH = PROJECT_ROOT / "runtime" / "cross_asset_golden_cross_results.json"
OUT_PATH = PROJECT_ROOT / "runtime" / "cross_asset_golden_cross_results_4h_alpaca.json"

ENTRY_COST = 12 / 10000
EXIT_COST = 12 / 10000
MIN_TRADES = 6  # same bar as the daily version -- below this, PF is too noisy to call
PER_TRADE_ALLOCATION_PCT = 2.0
STARTING_CAPITAL = 100_000.0
LOOKBACK_YEARS = 6  # as far back as Alpaca's free/IEX feed will actually give us
REQUEST_DELAY_SECONDS = 0.3  # stay well clear of free-tier rate limits across 60 symbols

# Same substitutions run_alpaca_paper_trader.py already uses to go from
# the daily backtest's symbol spelling to a real, tradable Alpaca ticker --
# reused here (not redefined differently) so this test covers the exact
# same instruments the live script would actually trade.
FUTURES_TO_ETF_PROXY = {"GC=F": "GLD", "PA=F": "PPLT"}
EQUITY_SYMBOL_MAP = {"BRK-B": "BRK.B"}

MIGRATED_CRYPTO = {"BTC-USD", "ETH-USD", "SOL-USD", "PAXG-USD"}  # already 4h on Kraken -- out of scope here


def _alpaca_symbol(daily_symbol: str) -> str:
    return (
        FUTURES_TO_ETF_PROXY.get(daily_symbol)
        or EQUITY_SYMBOL_MAP.get(daily_symbol)
        or daily_symbol
    )


def _load_robust_daily_universe() -> list[dict]:
    if not DAILY_RESULTS_PATH.exists():
        raise SystemExit(f"no daily backtest results at {DAILY_RESULTS_PATH}")
    data = json.loads(DAILY_RESULTS_PATH.read_text(encoding="utf-8"))
    return [
        r for r in data["results"]
        if r["robust"] and r["symbol"] not in MIGRATED_CRYPTO
    ]


def compute_trend_state(close: pd.Series) -> pd.DataFrame:
    """Identical to run_cross_asset_golden_cross_backtest.py's own function --
    duplicated rather than imported so this script has no dependency on
    that module's yfinance/cross_asset_tracker imports running at load time."""
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    above200 = ma50 > ma200
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
        return {"n_trades": len(trades), "robust": False, "reason": "too few trades"}
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


def simulate_portfolio(all_trades: list[dict]) -> dict:
    """Identical event-driven sim to run_cross_asset_golden_cross_backtest.py's
    own function (duplicated for the same reason compute_trend_state/
    backtest_one are above) -- each trade sized as % of capital at its own
    entry time, P&L realized at its own exit time."""
    events = []
    for t in all_trades:
        events.append((t["entry_time"], 0, "entry", t))
        events.append((t["exit_time"], 1, "exit", t))
    events.sort(key=lambda e: (e[0], e[1]))

    capital = STARTING_CAPITAL
    peak = STARTING_CAPITAL
    underwater = []
    if events:
        pass

    for ts, _, kind, t in events:
        if kind == "entry":
            t["_notional"] = capital * PER_TRADE_ALLOCATION_PCT / 100.0
        else:
            pnl = t["_notional"] * (t["net_ret_pct"] / 100.0)
            capital += pnl
            peak = max(peak, capital)
            underwater.append((capital / peak - 1) * 100)

    return {
        "starting_capital": STARTING_CAPITAL,
        "final_capital": capital,
        "total_return_pct": (capital / STARTING_CAPITAL - 1) * 100,
        "max_drawdown_pct": min(underwater, default=0.0),
        "n_trades": len(all_trades),
    }


def fetch_4h_close(client, symbol: str) -> pd.Series | None:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    start = datetime.now(timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS)
    try:
        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(4, TimeFrameUnit.Hour),
            start=start,
            limit=10000,
        ))
        df = bars.df
    except Exception as exc:
        print(f"  {symbol}: fetch failed: {exc}")
        return None
    if df is None or df.empty:
        print(f"  {symbol}: no data returned")
        return None
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    close = df["close"].dropna()
    if len(close) < 210:
        print(f"  {symbol}: only {len(close)} 4h bars, need >=210 for ma200 warmup -- skipping")
        return None
    return close


def main() -> None:
    import os

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set (check .env)")

    from alpaca.data.historical import StockHistoricalDataClient

    client = StockHistoricalDataClient(api_key, secret_key)

    daily_universe = _load_robust_daily_universe()
    print(f"daily-robust universe (excl. migrated crypto): {len(daily_universe)} instruments\n")
    daily_by_symbol = {r["symbol"]: r for r in daily_universe}

    rows = []
    all_trades = []
    for r in daily_universe:
        daily_symbol = r["symbol"]
        alpaca_symbol = _alpaca_symbol(daily_symbol)
        close = fetch_4h_close(client, alpaca_symbol)
        time.sleep(REQUEST_DELAY_SECONDS)
        if close is None:
            rows.append({"symbol": daily_symbol, "alpaca_symbol": alpaca_symbol,
                          "asset_class": r["asset_class"], "n_trades": 0, "robust": False,
                          "reason": "no usable 4h history", "daily_robust": True,
                          "daily_pf_excl3": r["pf_excl3"]})
            continue
        trades = backtest_one(close)
        for t in trades:
            all_trades.append({**t, "symbol": daily_symbol, "asset_class": r["asset_class"]})
        rb = robustness(trades) or {"n_trades": len(trades), "robust": False, "reason": "too few trades"}
        rows.append({"symbol": daily_symbol, "alpaca_symbol": alpaca_symbol,
                      "asset_class": r["asset_class"], "daily_robust": True,
                      "daily_pf_excl3": r["pf_excl3"], **rb})

    rows.sort(key=lambda x: -(x.get("pf_excl3") or -999))
    print(f"\n{'Symbol':<10} {'Alpaca':<10} {'N(4h)':>6} {'PF_x3(4h)':>10} {'Robust(4h)':>11}  "
          f"{'PF_x3(daily)':>13}  {'Still robust?'}")
    still_robust = 0
    for row in rows:
        pf4h = row.get("pf_excl3")
        pf4h_str = f"{pf4h:.2f}" if isinstance(pf4h, (int, float)) else "n/a"
        robust_4h = bool(row.get("robust"))
        still_robust += int(robust_4h)
        daily_pf_str = f"{row['daily_pf_excl3']:.2f}" if isinstance(row.get("daily_pf_excl3"), (int, float)) else "inf"
        print(f"{row['symbol']:<10} {row['alpaca_symbol']:<10} {row.get('n_trades', 0):>6} "
              f"{pf4h_str:>10} {'YES' if robust_4h else 'no':>11}  "
              f"{daily_pf_str:>13}  {'YES' if robust_4h else 'NO -- fails on 4h'}")

    n = len(rows)
    print(f"\n{still_robust}/{n} symbols that are robust on DAILY bars are ALSO robust on 4h bars.")

    # Portfolio-level comparison, not just per-symbol pass/fail -- same
    # event-driven sim run_cross_asset_golden_cross_backtest.py itself uses
    # (2% per trade, sized off capital at each trade's own entry time), run
    # here over ONLY this 60-symbol universe's 4h trades so it is directly
    # comparable to the equivalent daily-only number for the SAME universe
    # (that file's own top-level "portfolio" field mixes in the other ~56
    # non-robust/other-asset-class instruments from the full 116-instrument
    # universe, so it is NOT directly comparable to this number).
    portfolio_4h = simulate_portfolio(all_trades) if all_trades else None
    if portfolio_4h:
        print(f"\n4h portfolio (this {n}-symbol universe only, {PER_TRADE_ALLOCATION_PCT}% per trade): "
              f"${portfolio_4h['starting_capital']:,.0f} -> ${portfolio_4h['final_capital']:,.0f} "
              f"({portfolio_4h['total_return_pct']:+.1f}%), max drawdown "
              f"{portfolio_4h['max_drawdown_pct']:.1f}%, {len(all_trades)} trades total")

    if all_trades:
        rets = pd.Series([t["net_ret_pct"] for t in all_trades])
        wins, losses = rets[rets > 0], rets[rets < 0]
        overall_pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
        print(f"pooled 4h PF across all {len(all_trades)} trades from all symbols: {overall_pf:.2f} "
              f"(pooled across symbols, not the same per-symbol robustness bar above)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_years": LOOKBACK_YEARS,
        "min_trades": MIN_TRADES,
        "results": rows,
        "portfolio_4h": portfolio_4h,
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
