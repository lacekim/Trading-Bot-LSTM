"""Combined Telegram status report: freqtrade's Kraken book (both strategies)
plus Alpaca's real positions, in one message.

freqtrade's own /status only ever reports its own SQLite DB (Kraken pairs
only) -- it has no path to Alpaca's account at all (verified: zero Telegram
code anywhere in run_alpaca_paper_trader.py / _loop.py), so there was
previously no single place to see both books together. This doesn't touch
freqtrade's own Telegram integration -- it's a separate, additive report.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import json

from tools.run_ma200_paper_trader import _pair_for, fetch_close_series, STATE_PATH
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.telegram.notifier import V4TelegramNotifier

ROOT = Path(__file__).resolve().parents[1]
FREQTRADE_DBS = {
    "GoldenCrossStrategy (8080)": ROOT / "freqtrade-develop/user_data/tradesv3.sqlite",
    "GoldenCrossRegimeFilteredStrategy (8081)": ROOT / "freqtrade-develop/user_data/tradesv3_regime_filtered.sqlite",
}
MIGRATED_CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "PAXGUSD"}


def _freqtrade_open_trades(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT pair, open_rate, stake_amount FROM trades WHERE is_open = 1"
    ).fetchall()
    conn.close()
    trades = []
    for row in rows:
        label = row["pair"].split("/")[0]
        pnl_pct = None
        try:
            close = fetch_close_series(_pair_for(label))
            current_price = float(close.iloc[-1])
            pnl_pct = (current_price / row["open_rate"] - 1) * 100
        except Exception:
            pass
        trades.append({
            "pair": row["pair"], "open_rate": row["open_rate"],
            "stake_amount": row["stake_amount"], "pnl_pct": pnl_pct,
        })
    return trades


def _ma200_auto_positions() -> dict:
    """--ma200-auto's own tracked positions (runtime/ma200_paper_trades.json)
    -- a third source, separate from freqtrade's SQLite DB and Alpaca's
    API. Two independent streams: the golden-cross stream (the same signal
    freqtrade trades in parallel on Kraken -- the B5 redundancy this doc
    tracks) and BMSB (PAXG only, a fixed 12-week hold not tracked anywhere
    else in the whole system)."""
    if not STATE_PATH.exists():
        return {"golden_cross": [], "bmsb": []}
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    def _with_pnl(label: str, pos: dict) -> dict:
        pnl_pct = None
        current_price = None
        try:
            close = fetch_close_series(_pair_for(label))
            current_price = float(close.iloc[-1])
            pnl_pct = (current_price / pos["entry_price"] - 1) * 100
        except Exception:
            pass
        return {
            "label": label, "entry_price": pos["entry_price"], "pnl_pct": pnl_pct,
            # current_price added 2026-09-21 for the freqUI /status bridge
            # (build_synthetic_open_trades below) -- falls back to
            # entry_price so a synthetic trade row never has a missing
            # current_rate (OpenTradeSchema requires a float there).
            "current_price": current_price if current_price is not None else pos["entry_price"],
        }

    golden_cross = [_with_pnl(label, pos) for label, pos in state.get("positions", {}).items()]
    bmsb = [_with_pnl(label, pos) for label, pos in state.get("bmsb_positions", {}).items()]
    return {"golden_cross": golden_cross, "bmsb": bmsb}


# === PROJECT PATCH: ALPACA_BALANCE_PROFIT_COUNT_BRIDGE (helper) START ===
def _ma200_auto_equity() -> dict | None:
    """Dollar-equity summary for --ma200-auto's simulated portfolio, added
    2026-09-23 (G15) so /balance and /profit can show a real $ total for
    this stream instead of only the per-position % that
    _ma200_auto_positions() above already gives /status.

    Deliberately RE-DERIVES the same formula
    tools/run_ma200_paper_trader.py's own print_equity_summary() uses,
    rather than importing and calling that function directly -- that
    module's main() has real side effects (it places/exits simulated
    positions and sends its own Telegram alerts) and isn't structured as
    an importable pure function. Only STARTING_CAPITAL and
    PER_ASSET_ALLOCATION_PCT (plain constants, no side effects) are
    imported from it, matching the exact same fallback
    print_equity_summary() uses when a position predates the
    allocated_amount field. If this ever drifts from that function's own
    math, trust run_ma200_auto.log's own "=== equity summary ===" block
    as the source of truth and fix this to match it, not the other way
    around.

    Returns None if the state file doesn't exist yet (mirrors
    _ma200_auto_positions()'s own early return)."""
    if not STATE_PATH.exists():
        return None
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    from tools.run_ma200_paper_trader import STARTING_CAPITAL, PER_ASSET_ALLOCATION_PCT

    default_allocation = STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100
    unrealized = 0.0
    allocated_total = 0.0
    for positions in (state.get("positions", {}), state.get("bmsb_positions", {})):
        for label, pos in positions.items():
            allocated = pos.get("allocated_amount", default_allocation)
            allocated_total += allocated
            try:
                close = fetch_close_series(_pair_for(label))
                price = float(close.iloc[-1])
                move_pct = (price / pos["entry_price"] - 1) * 100
                unrealized += allocated * (move_pct / 100)
            except Exception:
                # Same "skip, don't crash the whole summary" behavior as
                # _ma200_auto_positions()'s own _with_pnl() above -- one
                # bad price fetch shouldn't zero out the rest of the total.
                pass

    cash = state.get("capital", 0.0)
    equity = cash + allocated_total + unrealized
    return {
        "cash": cash, "unrealized": unrealized, "equity": equity,
        "starting_capital": STARTING_CAPITAL,
    }
# === PROJECT PATCH: ALPACA_BALANCE_PROFIT_COUNT_BRIDGE (helper) END ===


def _alpaca_positions() -> list[dict]:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        # === PROJECT PATCH: ALPACA_DIAGNOSTIC_LOGGING (2026-09-22) START ===
        # This used to be a silent `return []`, indistinguishable in freqUI
        # from "Alpaca genuinely has zero positions." Added after the
        # equities briefly appeared in the Trade tab sidebar then vanished
        # on a later refresh, with zero exceptions in freqtrade_8080.log --
        # this print (stdout/stderr both land in that log via
        # restart_bots.sh's redirect) is what finally says why, if it
        # happens again.
        print(
            "ALPACA_POSITIONS: ALPACA_API_KEY/ALPACA_SECRET_KEY not visible "
            "in os.environ for this process -- returning [] empty.",
            file=sys.stderr, flush=True,
        )
        # === PROJECT PATCH: ALPACA_DIAGNOSTIC_LOGGING END ===
        return []
    from alpaca.trading.client import TradingClient
    try:
        client = TradingClient(api_key, secret_key, paper=True)
        positions = client.get_all_positions()
    except Exception as exc:
        # === PROJECT PATCH: ALPACA_DIAGNOSTIC_LOGGING (2026-09-22) ===
        print(
            f"ALPACA_POSITIONS: keys were visible but the Alpaca API call "
            f"itself failed: {type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True,
        )
        return []
    return [{
        "symbol": p.symbol, "market_value": float(p.market_value),
        "unrealized_plpc": float(p.unrealized_plpc) * 100,
        # avg_entry_price/current_price/qty added 2026-09-21 for the
        # freqUI /status bridge (build_synthetic_open_trades below) --
        # Alpaca's own Position object already carries these, just wasn't
        # needed by the Telegram-only report before.
        "avg_entry_price": float(p.avg_entry_price),
        "current_price": float(p.current_price),
        "qty": float(p.qty),
        # unrealized_pl (dollar, as opposed to unrealized_plpc's percent)
        # added 2026-09-23 (G15) so build_profit_extra() below can report a
        # real aggregate $ P&L for /profit instead of only per-position %.
        "unrealized_pl": float(p.unrealized_pl),
    } for p in positions]


# === PROJECT PATCH: ALPACA_BALANCE_BRIDGE (helper) START ===
def alpaca_total_equity() -> float | None:
    """Alpaca's own real account equity (cash + every open position's
    market value, i.e. exactly what run_alpaca_paper_trader.py itself
    sizes new orders off) -- added 2026-09-21 so /api/v1/balance
    (see the ALPACA_BALANCE_BRIDGE patch in api_v1.py) can fold it into
    freqUI's Dashboard total, closing the last gap where Alpaca was
    reflected on the Trade tab's row list and chart (ALPACA_STATUS_
    BRIDGE/ALPACA_CANDLE_BRIDGE) and in Telegram's /status (G4's merge)
    but NOT in the Dashboard's own account-total tiles.

    Deliberately NOT built from summing _alpaca_positions()'s
    market_value column: that would silently omit any uninvested cash
    sitting in the account, undercounting real equity. Returns None if
    Alpaca isn't configured or the call fails, so the caller can leave
    the real freqtrade balance result completely untouched rather than
    guess at a number."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        return float(client.get_account().equity)
    except Exception:
        return None
# === PROJECT PATCH: ALPACA_BALANCE_BRIDGE (helper) END ===


def build_synthetic_open_trades(start_id: int = -900001) -> list[dict]:
    """Alpaca's and --ma200-auto's real positions, reshaped into
    freqtrade's own OpenTradeSchema field names -- built 2026-09-21 so
    freqtrade's real REST API (/api/v1/status, see the
    ALPACA_STATUS_BRIDGE patch block in api_v1.py) can append them to its
    own trade list and have them show up as native rows in freqUI's real
    Trade page/live view, no separate UI at all.

    trade_id is always a large negative number (default range starting
    at -900001, counting down) specifically so it can never collide with
    a real freqtrade trade_id (freqtrade's own IDs are positive,
    autoincrementing from its SQLite DB). If anything in freqUI ever
    tries to act on one of these rows (force-exit, cancel, etc.), the
    request hits freqtrade's real trade-lookup code with an ID it will
    never find in its own DB -- a clean "trade not found" error, not any
    real action on a real position. These rows are info-only by
    construction, never by trusting the frontend not to click on them.

    Charting: freqUI's own candlestick chart asks freqtrade's own
    exchange (Kraken) for OHLCV data on whatever pair is selected, which
    has no data for an Alpaca-only symbol like AAPL. See
    fetch_alpaca_or_crypto_candles() below + api_v1.py's
    ALPACA_CANDLE_BRIDGE patch, which feeds real Alpaca market data into
    that same chart request instead, so this isn't a dead end."""
    trades: list[dict] = []
    next_id = start_id

    def _base_trade(*, trade_id: int, pair: str, base: str, exchange: str, strategy: str,
                     amount: float, open_rate: float, current_rate: float) -> dict:
        open_trade_value = amount * open_rate
        current_value = amount * current_rate
        profit_abs = current_value - open_trade_value
        profit_ratio = (current_rate / open_rate - 1) if open_rate else 0.0
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "trade_id": trade_id, "pair": pair, "base_currency": base, "quote_currency": "USD",
            "is_open": True, "is_short": False, "exchange": exchange,
            "amount": amount, "amount_requested": amount,
            "stake_amount": open_trade_value, "max_stake_amount": open_trade_value,
            "strategy": strategy, "enter_tag": None, "timeframe": 240,
            # open_date is approximate (a real entry timestamp isn't
            # available from either source without extra API calls this
            # bridge doesn't make) -- set to "now" rather than left blank,
            # since the field is required and an unparseable value breaks
            # freqUI's date column outright.
            "open_date": now_iso, "open_timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "open_fill_date": now_iso, "open_fill_timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "open_rate": open_rate, "open_rate_requested": open_rate, "open_trade_value": open_trade_value,
            "close_date": None, "close_timestamp": None, "close_rate": None, "close_rate_requested": None,
            "close_profit": None, "close_profit_pct": None, "close_profit_abs": None,
            "profit_ratio": profit_ratio, "profit_pct": profit_ratio * 100, "profit_abs": profit_abs,
            "profit_fiat": None, "realized_profit": 0.0, "realized_profit_ratio": None,
            "exit_reason": None, "exit_order_status": None,
            "stop_loss_abs": None, "stop_loss_ratio": None, "stop_loss_pct": None,
            "stoploss_last_update": None, "stoploss_last_update_timestamp": None,
            "initial_stop_loss_abs": None, "initial_stop_loss_ratio": None, "initial_stop_loss_pct": None,
            "min_rate": None, "max_rate": None,
            "nr_of_successful_entries": 1, "nr_of_successful_exits": 0,
            "has_open_orders": False, "orders": [],
            "leverage": None, "interest_rate": None, "liquidation_price": None,
            "funding_fees": None, "trading_mode": None,
            "amount_precision": None, "price_precision": None, "precision_mode": None,
            "stoploss_current_dist": None, "stoploss_current_dist_pct": None,
            "stoploss_current_dist_ratio": None, "stoploss_entry_dist": None,
            "stoploss_entry_dist_ratio": None,
            "current_rate": current_rate,
            "total_profit_abs": profit_abs, "total_profit_fiat": None, "total_profit_ratio": profit_ratio,
        }

    ma200_auto = _ma200_auto_positions()
    for p in ma200_auto["golden_cross"]:
        trades.append(_base_trade(
            trade_id=next_id, pair=f"{p['label']}/USD", base=p["label"],
            exchange="kraken (--ma200-auto)", strategy="MA200_AUTO_GOLDEN_CROSS",
            amount=1.0, open_rate=p["entry_price"], current_rate=p["current_price"],
        ))
        next_id -= 1
    for p in ma200_auto["bmsb"]:
        trades.append(_base_trade(
            trade_id=next_id, pair=f"{p['label']}/USD", base=p["label"],
            exchange="kraken (--ma200-auto)", strategy="MA200_AUTO_BMSB",
            amount=1.0, open_rate=p["entry_price"], current_rate=p["current_price"],
        ))
        next_id -= 1

    for p in _alpaca_positions():
        symbol = p["symbol"]
        is_migrated_crypto = symbol in MIGRATED_CRYPTO_SYMBOLS
        base = symbol[:-3] if is_migrated_crypto else symbol  # "BTCUSD" -> "BTC"
        pair = f"{base}/USD"
        trades.append(_base_trade(
            trade_id=next_id, pair=pair, base=base, exchange="alpaca",
            strategy="ALPACA_MIGRATED_CRYPTO" if is_migrated_crypto else "ALPACA_EQUITY",
            amount=p["qty"], open_rate=p["avg_entry_price"], current_rate=p["current_price"],
        ))
        next_id -= 1

    return trades


def _parse_freqtrade_timeframe(timeframe: str) -> tuple[int, str]:
    """'4h' -> (4, 'h'), '1d' -> (1, 'd'), '5m' -> (5, 'm')."""
    unit = timeframe[-1]
    amount = int(timeframe[:-1])
    return amount, unit


def fetch_alpaca_or_crypto_candles(base_symbol: str, timeframe: str, limit: int = 500):
    """Real OHLCV candles for a symbol freqtrade's own exchange (Kraken)
    has no market for -- built 2026-09-21 so freqUI's real candlestick
    chart works for these too, not just their entry in the trade list.
    freqtrade only ever analyzes pairs in its strategy's own active
    whitelist; BTC/USD was deliberately dropped from that whitelist
    (FULL_HANDOFF.md Section B5) and genuine equities were never on
    Kraken to begin with, so both need a real external data source here.

    Tries Alpaca's stock bars first (covers equities: AAPL, MSFT, ...),
    then falls back to Alpaca's crypto bars (covers BTC and anything
    else Kraken-native but currently outside the strategy's whitelist).

    Returns (df, resolved_timeframe) -- df is an empty DataFrame (never
    raises) if neither source has data, so the caller (api_v1.py's
    ALPACA_CANDLE_BRIDGE) can fall back to whatever freqtrade's own
    (also-empty) answer was, no worse off than before this bridge
    existed. resolved_timeframe is the timeframe the DATA is actually
    on -- added 2026-09-21 alongside EQUITIES_CHART_DAILY below, once
    that patch made the stock path fetch daily bars regardless of what
    `timeframe` was requested: the caller needs to know this to report
    an honest `timeframe`/`timeframe_ms` in the API response instead of
    just echoing back the (now wrong, for stocks) requested value."""
    import pandas as pd

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    if not api_key or not secret_key:
        return empty, timeframe

    # BUGFIX 2026-09-22 #3: this function's own docstring above assumed
    # the stock-bars lookup would simply find "no such ticker" for
    # BTC/ETH/SOL/PAXG and fall through to the crypto path below --
    # checked live and that assumption is false for at least one of
    # them: Alpaca's stock data API returned real (non-error, non-empty)
    # daily bars for the bare symbol "BTC", priced around $35-38/share --
    # NOT bitcoin, some unrelated equity ticker collision -- which would
    # have silently returned the wrong instrument's data under the
    # BTC/USD chart, with zero error or trace anywhere (found while
    # spot-checking BTC/USD after the G12 NameError fix, using the same
    # fetch_alpaca_or_crypto_candles("BTC", ...) diagnostic pattern as
    # check_vz_candle_bridge.py -- resolved_timeframe came back "1d" and
    # the price scale was an immediate tell it wasn't crypto data).
    # Fix: don't rely on "stock lookup fails" as an implicit crypto
    # detector -- skip the stock path entirely for symbols already known
    # to be migrated crypto (MIGRATED_CRYPTO_SYMBOLS, module-level above).
    is_known_migrated_crypto = f"{base_symbol}USD" in MIGRATED_CRYPTO_SYMBOLS

    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    # MA_WARMUP mirrors GoldenCrossStrategy's own startup_candle_count
    # (210, comfortably above its 200-period SMA) -- built 2026-09-21,
    # same day as the base candle bridge above, once the user pointed out
    # the chart was missing the ma50/ma200 overlay lines every Kraken
    # pair's chart has (GoldenCrossStrategy.plot_config draws exactly
    # those two column names). Fetching only `limit` bars would leave the
    # most recent ~200 candles' ma200 as NaN (not enough lookback to
    # average over), so this fetches `limit + MA_WARMUP` bars, computes
    # both SMAs over the full padded range, then trims back down to the
    # originally requested `limit` rows -- the same pattern freqtrade's
    # own analysis pipeline uses internally via startup_candle_count.
    MA_WARMUP = 210
    fetch_limit = limit + MA_WARMUP

    def _tf_params(tf_str: str):
        """Amount/unit/lookback-start for one Alpaca timeframe string --
        split out so the stock and crypto paths below can each use their
        own timeframe instead of sharing whatever freqUI happened to
        request (see EQUITIES_CHART_DAILY note below)."""
        amount, unit_char = _parse_freqtrade_timeframe(tf_str)
        unit_map = {"m": TimeFrameUnit.Minute, "h": TimeFrameUnit.Hour, "d": TimeFrameUnit.Day}
        unit = unit_map.get(unit_char, TimeFrameUnit.Hour)
        td_map = {"m": pd.Timedelta(minutes=1), "h": pd.Timedelta(hours=1), "d": pd.Timedelta(days=1)}
        # 3x padding: equities markets are closed most of the week relative
        # to crypto/Kraken, so a naive "amount * limit" lookback window comes
        # back short of `limit` real bars for stocks. Alpaca's own `limit`
        # param does the real trimming once fetched.
        lookback = td_map.get(unit_char, pd.Timedelta(hours=1)) * amount * fetch_limit * 3
        start = datetime.now(timezone.utc) - lookback
        return TimeFrame(amount, unit), start

    # === PROJECT PATCH: EQUITIES_CHART_DAILY START ===
    # 2026-09-21: the full 60-symbol backtest (FULL_HANDOFF.md Section G8)
    # confirmed the equities/commodities/bonds golden-cross rule only
    # holds up on DAILY bars (6yr: $100k->$166,686, -2.0% max DD) -- the
    # same rule on 4h Alpaca bars fails broadly ($100k->$122,132, -16.4%
    # max DD, only 6/60 symbols still robust) and was never what actually
    # drives live equity exits (run_alpaca_paper_trader.py has always used
    # daily Yahoo bars for this universe). Until now this chart still
    # requested whatever timeframe freqUI's own strategy asks for (4h,
    # since GoldenCrossStrategy is a 4h Kraken strategy) for EVERY pair,
    # including equities -- so the ma50/ma200 lines drawn for them were a
    # faster/noisier signal than what actually opens/closes those
    # positions. User's explicit instruction: make the chart match the
    # real signal. Stock bars (the path below covering the full
    # equities/commodities/bonds universe, including the GLD/PPLT futures
    # proxies and IEF) now always fetch DAILY regardless of what
    # timeframe freqUI requested.
    #
    # Migrated crypto (BTC/ETH/SOL/PAXG, reached via the crypto-bars
    # fallback path below when the stock lookup finds no such ticker) is
    # deliberately NOT changed here -- their live signal genuinely is 4h
    # Kraken bars (FULL_HANDOFF.md Section B2/B3: the SAME rule tested on
    # daily bars for crypto found only BTC robust, while 4h Kraken bars
    # found BTC/ETH/PAXG/SOL/TRX/XMR all robust -- 4h is their *validated*
    # timeframe, the opposite finding from equities). Forcing daily there
    # would reintroduce this exact mismatch in reverse, so the crypto path
    # keeps using whatever timeframe was actually requested.
    STOCK_CHART_TIMEFRAME = "1d"
    # === PROJECT PATCH: EQUITIES_CHART_DAILY END ===

    def _to_ohlcv_df(bars_df, symbol: str) -> "pd.DataFrame":
        if bars_df is None or bars_df.empty:
            return empty
        df = bars_df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol]
        df = df.rename(columns={"timestamp": "date"})
        if df.empty or "date" not in df.columns:
            return empty
        df["date"] = pd.to_datetime(df["date"], utc=True)
        cols = ["date", "open", "high", "low", "close", "volume"]
        return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    def _with_ma_overlay(df: "pd.DataFrame") -> "pd.DataFrame":
        """Adds the same ma50/ma200 columns GoldenCrossStrategy computes
        for every real Kraken pair (plain SMA -- identical output to
        ta.SMA for this simple-average case), then trims the warmup
        padding back off so the response still has exactly `limit` rows.
        freqUI draws these two column names automatically since they're
        already named in the strategy's own plot_config -- no frontend
        change needed, same reason the base OHLCV columns above needed
        none either."""
        df = df.copy()
        df["ma50"] = df["close"].rolling(window=50, min_periods=50).mean()
        df["ma200"] = df["close"].rolling(window=200, min_periods=200).mean()
        return df.iloc[-limit:].reset_index(drop=True)

    if not is_known_migrated_crypto:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest

            # === PROJECT PATCH: EQUITIES_CHART_DAILY (stock path) ===
            # BUGFIX 2026-09-22: this used to also pass limit=fetch_limit here.
            # Alpaca's bars API returns rows in ascending order starting at
            # `start` and stops once `limit` total rows are collected -- i.e.
            # it caps from the OLDEST end of [start, now], not the newest.
            # `start` is deliberately padded far earlier than fetch_limit
            # actually needs (3x, see the lookback comment in _tf_params
            # above), so capping the request itself at fetch_limit meant the
            # response was the earliest fetch_limit bars after that padded
            # start -- landing around 2-3 years in the past, never reaching
            # today. (Found because VZ/USD's own chart showed a "2023" year
            # marker and a price scale ~30% off the real quote -- the ma50/
            # ma200 "death cross" the user was reading off that chart was
            # real for THAT stale slice of data, just not for today.) Fix:
            # don't cap the request at all -- fetch the full [start, now]
            # range (alpaca-py paginates this automatically; it's at most a
            # few thousand daily rows even at 3x padding, trivial) and let
            # _with_ma_overlay's own `.iloc[-limit:]` below correctly keep
            # the most RECENT `limit` rows instead.
            # BUGFIX 2026-09-22 #2: this line was dropped by mistake while
            # editing the comment block above it -- request below referenced
            # stock_tf/stock_start without this assignment ever running,
            # raising NameError on every call (caught by the except below,
            # which is why it looked like a clean empty result with no
            # traceback visible anywhere).
            # BUGFIX 2026-09-22 #3: this whole stock-path attempt is now
            # skipped entirely (see is_known_migrated_crypto above) for
            # symbols already known to be migrated crypto -- previously it
            # ran unconditionally for every base_symbol, including those.
            stock_tf, stock_start = _tf_params(STOCK_CHART_TIMEFRAME)
            client = StockHistoricalDataClient(api_key, secret_key)
            request = StockBarsRequest(
                symbol_or_symbols=base_symbol, timeframe=stock_tf, start=stock_start,
            )
            df = _to_ohlcv_df(client.get_stock_bars(request).df, base_symbol)
            if not df.empty:
                return _with_ma_overlay(df), STOCK_CHART_TIMEFRAME
            print(
                f"[fetch_alpaca_or_crypto_candles] stock bars EMPTY for {base_symbol} "
                f"(start={stock_start.isoformat()}, no limit cap)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 -- try crypto next, never raise from here
            logger_msg = f"stock bars failed for {base_symbol}: {exc}"
            print(f"[fetch_alpaca_or_crypto_candles] {logger_msg}", flush=True)
    else:
        print(
            f"[fetch_alpaca_or_crypto_candles] {base_symbol} is a known migrated-"
            f"crypto symbol -- skipping the stock-bars lookup entirely (see "
            f"BUGFIX 2026-09-22 #3) and going straight to crypto bars below.",
            flush=True,
        )

    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest

        # Migrated crypto keeps whatever timeframe was actually requested
        # (4h, its own validated live-signal timeframe) -- see
        # EQUITIES_CHART_DAILY note above for why this path is untouched.
        crypto_tf, crypto_start = _tf_params(timeframe)
        crypto_symbol = f"{base_symbol}/USD"
        client = CryptoHistoricalDataClient(api_key, secret_key)
        # BUGFIX 2026-09-22: same fix as the stock path above -- don't cap
        # this request at fetch_limit either, for the same reason (it
        # truncates from the oldest end of the padded [start, now]
        # window). Crypto's 3x padding is smaller in absolute terms (its
        # own requested timeframe, not daily) but the same failure mode
        # applies, so fixing both keeps this consistent rather than
        # leaving a latent copy of the same bug here.
        request = CryptoBarsRequest(
            symbol_or_symbols=crypto_symbol, timeframe=crypto_tf, start=crypto_start,
        )
        df = _to_ohlcv_df(client.get_crypto_bars(request).df, crypto_symbol)
        if not df.empty:
            return _with_ma_overlay(df), timeframe
        print(
            f"[fetch_alpaca_or_crypto_candles] crypto bars EMPTY for {crypto_symbol} "
            f"(start={crypto_start.isoformat()}, no limit cap)",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 -- caller falls back to the empty result
        logger_msg = f"crypto bars failed for {base_symbol}: {exc}"
        print(f"[fetch_alpaca_or_crypto_candles] {logger_msg}", flush=True)

    return empty, timeframe


def build_extra_only() -> str:
    """The non-freqtrade sections of the combined report: --ma200-auto's
    own tracked positions (golden-cross + BMSB) and Alpaca's real
    positions. Split out from build_report() so freqtrade's own /status
    and /status table (freqtrade-develop/freqtrade/rpc/telegram.py, see
    its ALPACA_STATUS_BRIDGE patch block) can import and call just this
    piece directly and append it to their own already-displayed Kraken
    trades, instead of duplicating freqtrade's own SQLite-based trade
    listing. Safe to import directly from telegram.py because freqtrade
    runs in the same "javier" conda env as the rest of this project
    (confirmed 2026-09-20) -- no subprocess bridge needed."""
    lines = []

    ma200_auto = _ma200_auto_positions()
    lines.append(f"*--ma200-auto golden-cross (Kraken)*: {len(ma200_auto['golden_cross'])} open")
    for p in ma200_auto["golden_cross"]:
        pnl = f"{p['pnl_pct']:+.2f}%" if p["pnl_pct"] is not None else "n/a"
        lines.append(f"  {p['label']}: entry ${p['entry_price']:,.4f}, unrl {pnl}")
    lines.append(f"\n*--ma200-auto BMSB (Kraken)*: {len(ma200_auto['bmsb'])} open")
    for p in ma200_auto["bmsb"]:
        pnl = f"{p['pnl_pct']:+.2f}%" if p["pnl_pct"] is not None else "n/a"
        lines.append(f"  {p['label']}: entry ${p['entry_price']:,.4f}, unrl {pnl}")

    positions = _alpaca_positions()
    lines.append(f"\n*Alpaca*: {len(positions)} open")
    crypto = [p for p in positions if p["symbol"] in MIGRATED_CRYPTO_SYMBOLS]
    equities = [p for p in positions if p["symbol"] not in MIGRATED_CRYPTO_SYMBOLS]
    for p in crypto:
        lines.append(f"  {p['symbol']} (migrated crypto): ${p['market_value']:,.2f}, unrl {p['unrealized_plpc']:+.2f}%")
    for p in sorted(equities, key=lambda p: p["symbol"]):
        lines.append(f"  {p['symbol']}: ${p['market_value']:,.2f}, unrl {p['unrealized_plpc']:+.2f}%")
    if not positions:
        lines.append("  (none)")

    return "\n".join(lines)


# === PROJECT PATCH: ALPACA_BALANCE_PROFIT_COUNT_BRIDGE (builders) START ===
# Added 2026-09-23 (G15), same motivation and pattern as G4's
# ALPACA_STATUS_BRIDGE above: freqtrade's native /balance, /profit, and
# /count only ever look at its own SQLite trades (Kraken side, 4 pairs)
# -- they have no path to Alpaca's account or --ma200-auto's simulated
# book at all, same gap /status had before G4. These three builders are
# imported and called from telegram.py's own new
# ALPACA_BALANCE_PROFIT_COUNT_BRIDGE patch block, appended as a second
# message after each command's real, unmodified native output -- never
# replacing or recalculating anything freqtrade itself already reports.
def build_balance_extra() -> str:
    """Appended to /balance: Alpaca's real total account equity (cash +
    every open position's market value -- the same number
    alpaca_total_equity() already provides for the freqUI Dashboard's
    own account-total tile) and --ma200-auto's simulated equity, plus
    the same per-position breakdown /status already shows."""
    lines = []
    total_equity = alpaca_total_equity()
    if total_equity is not None:
        lines.append(f"*Alpaca account equity (real, paper)*: ${total_equity:,.2f}")
    else:
        lines.append("*Alpaca account equity*: _unavailable (no API keys, or the call failed)_")

    ma200_eq = _ma200_auto_equity()
    if ma200_eq is not None:
        ret_pct = (ma200_eq["equity"] / ma200_eq["starting_capital"] - 1) * 100
        lines.append(
            f"*--ma200-auto simulated equity*: ${ma200_eq['equity']:,.2f} "
            f"(started ${ma200_eq['starting_capital']:,.0f}, {ret_pct:+.2f}%)"
        )

    lines.append("")
    lines.append(build_extra_only())
    return "\n".join(lines)


def build_profit_extra() -> str:
    """Appended to plain /profit only (telegram.py's call site gates this
    out for /profit_long and /profit_short -- every position either
    stream holds is long-only, so a direction-filtered report has nothing
    real to add here). Adds Alpaca's real aggregate unrealized $ P&L and
    --ma200-auto's simulated unrealized $ P&L, plus the same per-position
    breakdown /status already shows."""
    lines = []
    positions = _alpaca_positions()
    alpaca_unrl = sum(p["unrealized_pl"] for p in positions)
    lines.append(
        f"*Alpaca unrealized P&L*: ${alpaca_unrl:+,.2f} across {len(positions)} open position"
        f"{'s' if len(positions) != 1 else ''}"
    )

    ma200_eq = _ma200_auto_equity()
    if ma200_eq is not None:
        lines.append(f"*--ma200-auto unrealized P&L*: ${ma200_eq['unrealized']:+,.2f}")

    lines.append("")
    lines.append(build_extra_only())
    return "\n".join(lines)


def build_count_extra() -> str:
    """Appended to /count: open-position counts for Alpaca and
    --ma200-auto -- the two streams freqtrade's native /count table has
    zero visibility into (it only ever counts freqtrade's own SQLite
    trades, i.e. the Kraken side)."""
    positions = _alpaca_positions()
    ma200_auto = _ma200_auto_positions()
    lines = [
        f"*Alpaca*: {len(positions)} open",
        f"*--ma200-auto golden-cross*: {len(ma200_auto['golden_cross'])} open",
        f"*--ma200-auto BMSB*: {len(ma200_auto['bmsb'])} open",
    ]
    return "\n".join(lines)
# === PROJECT PATCH: ALPACA_BALANCE_PROFIT_COUNT_BRIDGE (builders) END ===


def build_report() -> str:
    lines = ["*Combined status -- Kraken + Alpaca*"]

    for label, db_path in FREQTRADE_DBS.items():
        trades = _freqtrade_open_trades(db_path)
        lines.append(f"\n*{label}*: {len(trades)} open")
        for t in trades:
            pnl = f"{t['pnl_pct']:+.2f}%" if t["pnl_pct"] is not None else "n/a"
            lines.append(f"  {t['pair']}: entry ${t['open_rate']:,.2f}, stake ${t['stake_amount']:,.0f}, unrl {pnl}")

    lines.append("\n" + build_extra_only())

    return "\n".join(lines)


def main() -> None:
    # --extra-only: lets a human verify build_extra_only() standalone
    # (what telegram.py's ALPACA_STATUS_BRIDGE actually imports and
    # calls) without sending to Telegram twice.
    extra_only = "--extra-only" in sys.argv[1:]
    report = build_extra_only() if extra_only else build_report()
    print(report)
    if extra_only:
        return
    if not getattr(Config, "TELEGRAM_ENABLED", False) or not getattr(Config, "TELEGRAM_BOT_TOKEN", None):
        print("\n[telegram disabled -- printed only]")
        return
    notifier = V4TelegramNotifier()
    notifier.send_message(report)


if __name__ == "__main__":
    main()
