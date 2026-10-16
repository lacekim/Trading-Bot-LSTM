"""Real paper-trading execution of the golden-cross system via Alpaca's
paper trading API -- replaces run_cross_asset_paper_trader.py's simulated
JSON-file bookkeeping with actual orders in a real (paper) brokerage
account: real Order objects, real Position objects, real account equity.
Same rigor freqtrade gives the crypto side via Kraken, just for equities/
bonds/gold, which freqtrade itself structurally cannot reach (Kraken lists
crypto only).

Covers the instruments that passed the robustness backtest
(cross_asset_golden_cross_results.json): the 57 robust equities and IEF
trade directly under their own real tickers, and the 2 non-ETF commodity
futures (GC=F, PA=F) are substituted for their nearest liquid ETF proxies
since Alpaca doesn't support raw futures contracts at all. Forex is 0/61
robust already (see the backtest's own caveats) so there's nothing to
substitute there.

Crypto consolidation (2026-09-13): of the 6 assets live on the Kraken/
freqtrade side (BTC/ETH/XMR/PAXG/SOL/TRX), BTC/ETH/SOL/PAXG are confirmed
tradable on Alpaca (verified live via GetAssetsRequest -- see Section B1
of that session's handoff) and migrate here, signaled from the SAME real
Kraken 4h golden-cross data/rule already verified and running live in
freqtrade (trading_bot_v4/signals/ma200_trend.py), not the daily-Yahoo
backtest this file used to run for its equities universe -- that daily
test used a different timeframe/data source and found ETH/SOL not robust
and never tested PAXG at all (not a normal Yahoo ticker); the 4h Kraken
result is the one that's actually been running live and verified, so it
wins (see cross_asset_golden_cross_results.json's robust_source field for
the overridden entries). These four use the crypto system's own proven
15% per-asset sizing, not the broad equities universe's 2%.

XMR and TRX are NOT listed on Alpaca at all (confirmed live, same check)
and never will be tradable here -- XMR almost certainly for privacy-coin/
AML reasons regulated US venues commonly cite for delisting Monero, TRX's
reason unconfirmed but equally final. Both stay on Kraken/freqtrade
permanently; this is a hard platform limit, not a temporary gap, same
category as the GC=F/PA=F futures substitution above (except there's no
substitute possible here -- they just don't move).

Real hard stops (2026-09-13, Section E/E5): every held position closes
early if Alpaca's own unrealized_plpc breaches its universe's own
independently-validated threshold, before the golden-cross signal is even
evaluated -- migrated crypto positions (MIGRATED_CRYPTO_YF_TO_LABEL) at
HARD_STOP_PCT (-8%, see tools/run_stoploss_tail_risk_test.py), everything
else (equities/bonds/commodities) at EQUITIES_HARD_STOP_PCT (-12%, see
tools/run_stoploss_tail_risk_test_equities.py) -- deliberately NOT the
same number: equities' own daily-bar backtest found a different tradeoff
point than crypto's 4h-bar one, so it earned its own threshold rather than
inheriting crypto's.

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in the environment -- free
paper-trading keys from an alpaca.markets account. This cannot run
without them; a script can't generate brokerage credentials.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from tools.run_ma200_cross_strategy import ASSETS as KRAKEN_4H_ASSETS, load_full_history
from trading_bot_v4.signals.ma200_trend import compute_trend_state

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# === PROJECT PATCH: ALPACA_CASH_CHECK START ===
# Added 2026-09-21. Found live in logs/alpaca_auto.log: ETH-USD and SOL-USD
# had failed with "insufficient balance for USD" on EVERY single 4h cycle
# from at least 2026-09-20T07:17Z through the most recent restart, silently
# -- because entry sizing here (notional = equity * allocation_pct / 100)
# is computed off TOTAL EQUITY and this script never checked actual cash
# before submitting; it just tried the order and let Alpaca reject it,
# every cycle, forever, as long as the account stayed this cash-
# constrained. main() now tracks a running cash estimate through the loop
# (decremented after each order this run actually submits) and skips an
# entry BEFORE attempting it once it would exceed that estimate, so a
# doomed order is never even sent -- and notifies once per skip so it is
# visible instead of only ever showing up in a log file someone has to
# grep for.
CASH_SAFETY_BUFFER = 50.0  # leave a small margin rather than sizing to the exact last dollar


def notify(text: str) -> None:
    """Same pattern as tools/run_ma200_paper_trader.py's own notify() --
    this script had no Telegram integration at all before this patch;
    added narrowly for this one new failure path rather than wiring
    notifications into every print() in this file."""
    try:
        from trading_bot_v4.config_v4 import V4Config as Config
        from trading_bot_v4.telegram.notifier import V4TelegramNotifier
    except Exception as exc:
        print(f"[telegram import failed: {exc}] {text}")
        return
    if not getattr(Config, "TELEGRAM_ENABLED", False) or not getattr(Config, "TELEGRAM_BOT_TOKEN", None):
        print(f"[telegram disabled] {text}")
        return
    try:
        V4TelegramNotifier().send_message(text)
    except Exception as exc:
        print(f"[telegram send failed: {exc}] {text}")
# === PROJECT PATCH: ALPACA_CASH_CHECK END ===

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
RESULTS_PATH = RUNTIME_DIR / "cross_asset_golden_cross_results.json"

# === PROJECT PATCH: DECISION_LOGGING START ===
# Added 2026-09-28, prompted by a live SO buy/sell/sell whipsaw on
# 2026-09-22/23 that took real digging to even reconstruct (see
# tools/run_live_vs_backtest_reconciliation_equities.py's PART D and the
# chat history around that date). The root cause: run_alpaca_paper_trader_
# loop.py's own log() only ever recorded loop-level heartbeat lines
# (starting/sleeping/stopped) -- every per-symbol OPEN/STOP/CLOSE/SKIP
# decision below was a bare print(), which only reaches stdout and is
# never persisted anywhere once the loop's parent shell isn't being
# watched live. That made this file's own actual decisions unauditable
# after the fact, which is what made the SO incident hard to pin down.
#
# LOG_PATH is deliberately the EXACT SAME FILE run_alpaca_paper_trader_
# loop.py's own log() already writes to -- same path expression, computed
# independently here since this module can also be run standalone (not
# just via the loop) -- so every decision line below now interleaves
# chronologically with the loop's own starting/sleeping/stopped lines in
# one place, instead of living in two disconnected logs.
#
# _log() is a pure additive replacement for print(): it still prints (so
# running this file directly still shows output exactly as before), and
# it ALSO appends the same line, timestamped, to LOG_PATH. Nothing about
# WHEN a decision fires, WHAT it decides, or the trading logic itself
# changes -- every print() call below is swapped for _log() with its
# exact same message, nothing more.
LOG_PATH = Path(__file__).resolve().parents[1] / "runtime" / "alpaca_paper_trader_loop.log"


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")
# === PROJECT PATCH: DECISION_LOGGING END ===

MAX_STALE_DAYS = 5
PER_ASSET_ALLOCATION_PCT = 2.0  # matches the backtest's own per-trade sizing, for the broad equities universe
CRYPTO_ALLOCATION_PCT = 15.0  # matches the proven, live Kraken golden-cross system's own sizing

# Section E (2026-09-13, see tools/run_stoploss_tail_risk_test.py): validated
# on the same real 6-asset Kraken golden-cross universe this migrated crypto
# signal comes from -- PF 2.31->2.68, max drawdown -33.0%->-17.8%, confirmed
# not a one-trade artifact (E3b: still PF 2.74 excluding each asset's single
# worst loss). Applies ONLY to MIGRATED_CRYPTO_YF_TO_LABEL positions -- the
# broad equities universe below was never part of what was backtested here.
HARD_STOP_PCT = -8.0

# Section E5 (2026-09-13, see tools/run_stoploss_tail_risk_test_equities.py):
# independently validated against this universe's own daily-bar backtest
# (60 robust equities/bonds/commodities instruments, 2% sizing, A2's
# cash-constrained entries) -- NOT copied from crypto's -8%. Unlike crypto,
# every stop distance here costs some PF (this universe's baseline PF 4.48
# has no free lunch), but -12% gave the best tradeoff: max drawdown
# -12.3%->-8.5%, worst single trade -53.9%->-12.1%, for CAGR 9.4%->8.5% and
# PF 4.48->4.08. Tighter distances (-3/-5/-8%) cost far more CAGR (down to
# 3.5%/5.1%/7.0%) for barely better drawdown, so they don't clear the "small
# cost" bar. E3b-verified not a one-trade artifact: PF excluding each
# asset's own single worst loss is 4.31, i.e. higher than with it included.
EQUITIES_HARD_STOP_PCT = -12.0

# BTC/ETH/SOL/PAXG confirmed tradable on Alpaca and migrated from Kraken --
# these use the real 4h Kraken signal (KRAKEN_4H_ASSETS below), not daily
# Yahoo. XMR/TRX are NOT in this set: confirmed not listed on Alpaca at
# all, so they stay on Kraken/freqtrade permanently (see module docstring).
MIGRATED_CRYPTO_YF_TO_LABEL = {
    "BTC-USD": "BTC", "ETH-USD": "ETH", "SOL-USD": "SOL", "PAXG-USD": "PAXG",
}

# GC=F/PA=F are raw futures contracts -- Alpaca doesn't support futures at
# all, so trade the nearest liquid, real ETF proxy instead. Every other
# robust symbol (57 equities + IEF + the migrated crypto set) is already a
# real, directly tradable Alpaca ticker with no substitution needed.
FUTURES_TO_ETF_PROXY = {
    "GC=F": "GLD",   # SPDR Gold Shares
    "PA=F": "PPLT",  # abrdn Physical Palladium -- closest liquid ETF; tracks
                      # physical palladium, not the futures contract, so
                      # it's a real but imperfect substitute, not identical
}

# Alpaca's crypto tickers use "/" not "-" -- BTC-USD (Yahoo/yfinance
# convention used everywhere else in this project) -> BTC/USD (Alpaca).
CRYPTO_SYMBOL_MAP = {"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD", "SOL-USD": "SOL/USD", "PAXG-USD": "PAXG/USD"}

# Yahoo/yfinance and Alpaca disagree on share-class ticker punctuation --
# confirmed live: Alpaca rejected "BRK-B" with "asset not found", it wants
# the dot form.
EQUITY_SYMBOL_MAP = {"BRK-B": "BRK.B"}


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
    return {
        "signal": "LONG" if bool(above200.iloc[-1]) else "FLAT",
        "close": float(close.iloc[-1]),
        "as_of": str(close.index[-1].date()),
    }


def _compute_signal_kraken_4h(yf_symbol: str) -> dict | None:
    """Real signal for the migrated crypto set (BTC/ETH/SOL/PAXG), sourced
    from the SAME Kraken 4h data/rule already verified and running live in
    freqtrade -- not daily Yahoo. label/pair/path come straight from
    run_ma200_cross_strategy.ASSETS so this can never silently drift from
    what the live crypto system actually trades."""
    label = MIGRATED_CRYPTO_YF_TO_LABEL[yf_symbol]
    pair, path = KRAKEN_4H_ASSETS[label]
    try:
        df = load_full_history(pair, path)
    except Exception:
        return None
    state = compute_trend_state(df["close"])
    last = state.iloc[-1]
    if pd.isna(last["ma200"]):
        return None
    return {
        "signal": "LONG" if bool(last["above200"]) else "FLAT",
        "close": float(last["close"]),
        "as_of": str(state.index[-1]),
    }


def main() -> None:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set.\n"
            "Get free paper-trading keys at "
            "https://app.alpaca.markets/paper/dashboard/overview\n"
            "then add to .env:\n"
            "  ALPACA_API_KEY=...\n"
            "  ALPACA_SECRET_KEY=...\n"
        )

    client = TradingClient(api_key, secret_key, paper=True)
    account = client.get_account()
    _log(f"connected to Alpaca paper account: equity=${float(account.equity):,.2f} "
         f"cash=${float(account.cash):,.2f}")
    # === PROJECT PATCH: ALPACA_CASH_CHECK (init) ===
    available_cash = float(account.cash)
    # === END ===

    universe = _load_robust_universe()
    _log(f"trading universe (robust only): {len(universe)} instruments")

    existing_positions = {p.symbol: p for p in client.get_all_positions()}
    session = cffi_requests.Session(impersonate="chrome")

    for yf_symbol, asset_class in universe.items():
        alpaca_symbol = (
            FUTURES_TO_ETF_PROXY.get(yf_symbol)
            or CRYPTO_SYMBOL_MAP.get(yf_symbol)
            or EQUITY_SYMBOL_MAP.get(yf_symbol)
            or yf_symbol
        )
        is_migrated_crypto = yf_symbol in MIGRATED_CRYPTO_YF_TO_LABEL
        if is_migrated_crypto:
            sig = _compute_signal_kraken_4h(yf_symbol)
        else:
            close = _fetch_close_series(yf_symbol, session)
            sig = _compute_signal(close) if close is not None else None
        if sig is None:
            _log(f"{yf_symbol}: no usable data this run, skipping")
            continue
        # Alpaca returns crypto positions keyed without the slash (BTCUSD, not BTC/USD)
        pos_key = alpaca_symbol.replace("/", "")
        held = pos_key in existing_positions
        allocation_pct = CRYPTO_ALLOCATION_PCT if is_migrated_crypto else PER_ASSET_ALLOCATION_PCT

        if held:
            stop_pct = HARD_STOP_PCT if is_migrated_crypto else EQUITIES_HARD_STOP_PCT
            unrealized_pct = float(existing_positions[pos_key].unrealized_plpc) * 100
            if unrealized_pct <= stop_pct:
                try:
                    client.close_position(pos_key)
                    _log(f"STOP {yf_symbol} -> {alpaca_symbol} ({asset_class}) "
                         f"hard stop hit ({unrealized_pct:+.2f}%, threshold {stop_pct:.0f}%)")
                except Exception as e:
                    _log(f"STOP FAILED {yf_symbol} -> {alpaca_symbol}: {e}")
                continue

        if sig["signal"] == "LONG" and not held:
            equity = float(account.equity)
            notional = round(equity * allocation_pct / 100, 2)
            # === PROJECT PATCH: ALPACA_CASH_CHECK (gate) START ===
            if notional > available_cash - CASH_SAFETY_BUFFER:
                msg = (f"SKIP {yf_symbol} -> {alpaca_symbol} ({asset_class}): "
                       f"golden cross signaled LONG but only ${available_cash:,.2f} cash "
                       f"available vs ${notional:,.2f} needed -- not attempting the order")
                _log(msg)
                notify(f"⚠️ {msg} (this will keep recurring every cycle until cash frees up)")
                continue
            # === PROJECT PATCH: ALPACA_CASH_CHECK (gate) END ===
            try:
                order = client.submit_order(MarketOrderRequest(
                    symbol=alpaca_symbol,
                    notional=notional,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC if "/" in alpaca_symbol else TimeInForce.DAY,
                ))
                _log(f"OPEN {yf_symbol} -> {alpaca_symbol} ({asset_class}) "
                     f"notional=${notional:,.2f} order_id={order.id}")
                # === PROJECT PATCH: ALPACA_CASH_CHECK (decrement) ===
                available_cash -= notional
                # === END ===
            except Exception as e:
                _log(f"OPEN FAILED {yf_symbol} -> {alpaca_symbol}: {e}")

        elif sig["signal"] == "FLAT" and held:
            try:
                client.close_position(pos_key)
                _log(f"CLOSE {yf_symbol} -> {alpaca_symbol} ({asset_class})")
            except Exception as e:
                _log(f"CLOSE FAILED {yf_symbol} -> {alpaca_symbol}: {e}")
        else:
            _log(f"{yf_symbol}: {sig['signal']}, {'holding' if held else 'flat'}")


if __name__ == "__main__":
    main()
