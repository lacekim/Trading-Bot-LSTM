"""Paper (dry-run) trader for the verified golden cross (50MA/200MA) rule on
BTC, ETH, and XMR.

No real orders are ever placed -- this checks the live signal, tracks
simulated long positions in a JSON state file, and sends Telegram alerts on
entry/exit so the real, honest track record builds up over time before any
real capital is ever committed. Short/bearish crosses (death crosses) are
reported as informational only (no verified short edge -- see
run_ma200_cross_strategy.py and the short-side backtest that failed the
robustness check).

XMR, PAXG, SOL, and TRX are included only under this golden-cross rule --
all four failed the same robustness check under the plain 200MA-only cross.
XMR (privacy/regulatory) and PAXG (tokenized gold) also have structurally
different demand drivers than general crypto risk-on, which is real
diversification value on top of the backtest numbers.

Real -8% hard stop (2026-09-13, Section E): golden-cross positions
(state["positions"]) now also exit if a bar's low breaches
entry_price * 0.92, independent of the death-cross signal -- see
HARD_STOP_PCT above and tools/run_stoploss_tail_risk_test.py for the
backtest that validated it. PAXG's separate BMSB hold (state
["bmsb_positions"]) is deliberately excluded -- that stream was never
part of what was tested.

Runs as a persistent process via trading_bot_v4/signals/ma200_auto.py
(python -m trading_bot_v4.main --ma200-auto), matching the architecture of
the existing --auto scheduler, not an external shell loop.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trading_bot_v4.signals.ma200_trend import current_signal
from trading_bot_v4.signals.ma200_asset_qualifier import get_qualified_assets, PAIR_OVERRIDE
from trading_bot_v4.signals.bmsb_trend import current_bmsb_signal, HOLD_WEEKS
from trading_bot_v4.signals.ma200_asset_qualifier import _load_full_history as load_bmsb_history
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.telegram.notifier import V4TelegramNotifier
from trading_bot_v4.signals.orderflow_capture import capture_cross_event_context

STATE_PATH = Path(__file__).resolve().parents[1] / "runtime" / "ma200_paper_trades.json"
BMSB_ASSETS = ["BTC", "PAXG"]  # only these two passed the de-duplicated robustness check
STARTING_CAPITAL = 100_000.0
PER_ASSET_ALLOCATION_PCT = 15.0  # matches the verified backtest sizing

# Section E (2026-09-13, see tools/run_stoploss_tail_risk_test.py): validated
# across the same 6-asset golden-cross universe, cash-constrained sizing --
# -8% both improved PF (2.31->2.68) and cut max drawdown (-33.0%->-17.8%),
# confirmed not a one-trade artifact via E3b (excluding each asset's single
# worst loss still gives PF 2.74). Applies only to the golden-cross position
# stream below (state["positions"]) -- deliberately NOT to BMSB_ASSETS'
# separate fixed-hold positions, which were never part of what was tested.
HARD_STOP_PCT = -8.0

# Section B5 (2026-09-16): BTC's Alpaca golden-cross entry path is now
# verified end-to-end (real fill 2026-09-15, direction/price/sizing all
# checked against this same Kraken signal -- see FULL_HANDOFF.md Section
# B5). Alpaca is now the sole executor for NEW BTC golden-cross entries;
# this Kraken-side paper trader stops opening new ones for BTC only, to
# retire the redundant duplicate signal-follow. This is a local exclusion
# gating NEW entries only -- it does NOT touch ma200_asset_qualifier.py or
# CANDIDATE_UNIVERSE (shared with backtests, would affect more than this
# one script), and it does NOT touch BTC's existing open golden-cross
# position (still tracked to its natural exit below, hard-stop and
# signal-close both unaffected) or BTC's separate BMSB stream
# (state["bmsb_positions"], independent signal, out of scope entirely).
KRAKEN_ENTRY_EXCLUDED_ASSETS = {"BTC"}


def _pair_for(symbol: str) -> str:
    return PAIR_OVERRIDE.get(symbol, f"{symbol}USD")
FEE_BPS = 5
SLIP_BPS = 7
ROUND_TRIP_COST_PCT = (FEE_BPS + SLIP_BPS) * 2 / 10000 * 100


def fetch_close_series(pair: str) -> pd.Series:
    return fetch_ohlc(pair)["close"]


def fetch_ohlc(pair: str, interval: int = 240) -> pd.DataFrame:
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    key = [k for k in data["result"].keys() if k != "last"][0]
    rows = data["result"][key]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["dt"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df.set_index("dt")[["open", "high", "low", "close"]]


def load_state() -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state.setdefault("bmsb_positions", {})
        state.setdefault("bmsb_last_entry", {})
        state.setdefault("bmsb_closed_trades", [])
        if "capital" not in state:
            # Backfilling capital tracking onto positions that were already
            # open before this feature existed -- allocate the same 15% per
            # position retroactively, exactly matching what was already
            # deployed, so the running balance starts accurate, not at zero.
            n_open = len(state["positions"]) + len(state.get("bmsb_positions", {}))
            allocated = STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100 * n_open
            state["capital"] = STARTING_CAPITAL - allocated
            for pos in state["positions"].values():
                pos.setdefault("allocated_amount", STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100)
            for pos in state.get("bmsb_positions", {}).values():
                pos.setdefault("allocated_amount", STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100)
        return state
    return {
        "positions": {}, "closed_trades": [], "bmsb_positions": {}, "bmsb_last_entry": {},
        "bmsb_closed_trades": [], "capital": STARTING_CAPITAL,
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def notify(text: str) -> None:
    if not getattr(Config, "TELEGRAM_ENABLED", False) or not getattr(Config, "TELEGRAM_BOT_TOKEN", None):
        print(f"[telegram disabled] {text}")
        return
    try:
        notifier = V4TelegramNotifier()
        notifier.send_message(text)
    except Exception as exc:
        print(f"[telegram send failed: {exc}] {text}")


def main() -> None:
    state = load_state()
    now = datetime.now(timezone.utc).isoformat()

    # Dynamic, not hand-picked: re-scans the full candidate universe on its
    # own schedule (see ma200_asset_qualifier.QUALIFICATION_INTERVAL_DAYS)
    # and only opens NEW positions in symbols that currently pass the same
    # robustness bar every asset in this system has been held to. Existing
    # open positions in a symbol that later drops off the list are still
    # tracked to their natural exit -- qualification gates new entries, it
    # doesn't abandon a trade already in progress.
    qualified = set(get_qualified_assets())
    open_symbols = set(state["positions"].keys())
    active_symbols = sorted(qualified | open_symbols)
    if not active_symbols:
        print("no qualified assets and no open positions -- nothing to check")
        return
    print(f"qualified: {sorted(qualified)}  |  active this cycle: {active_symbols}")
    current_prices: dict[str, float] = {}

    for label in active_symbols:
        pair = _pair_for(label)
        # Gates NEW entries only -- existing open positions (pos is not
        # None below) are still evaluated for exit regardless of this flag.
        is_qualified = label in qualified and label not in KRAKEN_ENTRY_EXCLUDED_ASSETS
        ohlc = fetch_ohlc(pair)
        close = ohlc["close"]
        sig = current_signal(close)
        if sig["signal"] == "INSUFFICIENT_DATA":
            print(f"{label}: insufficient data, skipping")
            continue
        current_prices[label] = sig["close"]

        pos = state["positions"].get(label)

        if pos is not None:
            stop_price = pos["entry_price"] * (1 + HARD_STOP_PCT / 100)
            bar_low = float(ohlc["low"].iloc[-1])
            if bar_low <= stop_price:
                entry_price = pos["entry_price"]
                exit_price = stop_price
                net_ret_pct = (exit_price / entry_price - 1) * 100 - ROUND_TRIP_COST_PCT
                allocated = pos.get("allocated_amount", STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100)
                realized_pnl = allocated * (net_ret_pct / 100)
                state["capital"] += allocated + realized_pnl
                state["closed_trades"].append({
                    "asset": label, "entry_time": pos["entry_time"], "exit_time": str(sig["as_of"]),
                    "entry_price": entry_price, "exit_price": exit_price, "net_ret_pct": net_ret_pct,
                    "allocated_amount": allocated, "realized_pnl": realized_pnl,
                    "exit_reason": "hard_stop_8pct",
                })
                del state["positions"][label]
                notify(
                    f"🛑 PAPER LONG stopped out: {label} (-8% hard stop)\n"
                    f"Entry: ${entry_price:,.4f} -> Stop: ${exit_price:,.4f}\n"
                    f"Net return: {net_ret_pct:+.2f}% (after est. costs)  |  Realized P&L: ${realized_pnl:,.2f}\n"
                    f"(dry trade -- no real order placed; see Section E backtest justification)"
                )
                continue

        if sig["signal"] == "LONG" and pos is None and is_qualified:
            allocated = state["capital"] * PER_ASSET_ALLOCATION_PCT / 100
            state["positions"][label] = {
                "entry_price": sig["close"], "entry_time": str(sig["as_of"]), "opened_at": now,
                "allocated_amount": allocated,
            }
            state["capital"] -= allocated
            capture_cross_event_context(label, "golden_cross_up", sig["close"], str(sig["as_of"]))
            notify(
                f"🟢 PAPER LONG opened: {label} (golden cross)\n"
                f"Entry: ${sig['close']:,.4f}  |  Allocated: ${allocated:,.2f}\n"
                f"50MA: ${sig['ma50']:,.4f} / 200MA: ${sig['ma200']:,.4f} ({sig['distance_pct']:+.1f}%)\n"
                f"(dry trade -- no real order placed)"
            )

        elif sig["signal"] == "FLAT" and pos is not None:
            entry_price = pos["entry_price"]
            exit_price = sig["close"]
            net_ret_pct = (exit_price / entry_price - 1) * 100 - ROUND_TRIP_COST_PCT
            allocated = pos.get("allocated_amount", STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100)
            realized_pnl = allocated * (net_ret_pct / 100)
            state["capital"] += allocated + realized_pnl
            state["closed_trades"].append({
                "asset": label, "entry_time": pos["entry_time"], "exit_time": str(sig["as_of"]),
                "entry_price": entry_price, "exit_price": exit_price, "net_ret_pct": net_ret_pct,
                "allocated_amount": allocated, "realized_pnl": realized_pnl,
                "exit_reason": "signal",
            })
            del state["positions"][label]
            notify(
                f"🔴 PAPER LONG closed: {label}\n"
                f"Entry: ${entry_price:,.4f} -> Exit: ${exit_price:,.4f}\n"
                f"Net return: {net_ret_pct:+.2f}% (after est. costs)  |  Realized P&L: ${realized_pnl:,.2f}\n"
                f"(dry trade -- no real order placed)"
            )
            if sig["just_crossed_down"]:
                capture_cross_event_context(label, "death_cross", sig["close"], str(sig["as_of"]))
                notify(f"⚠️ {label} death cross (50MA below 200MA) -- bearish state. No verified short edge; informational only.")

        else:
            status = "holding paper LONG" if pos is not None else "flat, no verified short position taken"
            print(f"{label}: {sig['signal']}, {status}, close=${sig['close']:,.4f} "
                  f"ma50=${sig['ma50']:,.4f} ma200=${sig['ma200']:,.4f} ({sig['distance_pct']:+.1f}%)")

    # Bull Market Support Band -- a separate, independently-verified entry
    # pathway, tracked as its own position stream so it never gets confused
    # with the golden-cross position for the same asset. Only BTC and PAXG:
    # the only two that stayed robust after properly de-duplicating
    # overlapping weekly touches into distinct episodes.
    for label in BMSB_ASSETS:
        df = load_bmsb_history(label)
        if df is None:
            continue
        bmsb_pos = state["bmsb_positions"].get(label)

        if bmsb_pos is None:
            last_entry = state["bmsb_last_entry"].get(label)
            bsig = current_bmsb_signal(df, pd.Timestamp(last_entry) if last_entry else None)
            if bsig.get("new_entry"):
                allocated = state["capital"] * PER_ASSET_ALLOCATION_PCT / 100
                state["bmsb_positions"][label] = {
                    "entry_price": bsig["close"], "entry_time": str(bsig["as_of"]), "opened_at": now,
                    "allocated_amount": allocated,
                }
                state["capital"] -= allocated
                state["bmsb_last_entry"][label] = str(bsig["as_of"])
                notify(
                    f"🟢 PAPER LONG opened: {label} (Bull Market Support Band)\n"
                    f"Entry: ${bsig['close']:,.4f}  |  Allocated: ${allocated:,.2f}\n"
                    f"Fixed {HOLD_WEEKS}-week hold, exactly as backtested\n"
                    f"(dry trade -- no real order placed, separate from any golden-cross position)"
                )
            else:
                print(f"{label} BMSB: no new entry ({bsig.get('reason')})")
        else:
            entry_time = pd.Timestamp(bmsb_pos["entry_time"])
            weeks_held = (pd.Timestamp.now(tz="UTC") - entry_time).days / 7
            if weeks_held >= HOLD_WEEKS:
                exit_price = float(df["close"].iloc[-1])
                entry_price = bmsb_pos["entry_price"]
                net_ret_pct = (exit_price / entry_price - 1) * 100 - ROUND_TRIP_COST_PCT
                allocated = bmsb_pos.get("allocated_amount", STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100)
                realized_pnl = allocated * (net_ret_pct / 100)
                state["capital"] += allocated + realized_pnl
                state["bmsb_closed_trades"].append({
                    "asset": label, "entry_time": bmsb_pos["entry_time"],
                    "exit_time": datetime.now(timezone.utc).isoformat(),
                    "entry_price": entry_price, "exit_price": exit_price, "net_ret_pct": net_ret_pct,
                    "allocated_amount": allocated, "realized_pnl": realized_pnl,
                })
                del state["bmsb_positions"][label]
                notify(
                    f"🔴 PAPER LONG closed: {label} (Bull Market Support Band, {HOLD_WEEKS}w hold complete)\n"
                    f"Entry: ${entry_price:,.4f} -> Exit: ${exit_price:,.4f}\n"
                    f"Net return: {net_ret_pct:+.2f}% (after est. costs)  |  Realized P&L: ${realized_pnl:,.2f}\n"
                    f"(dry trade -- no real order placed)"
                )
            else:
                print(f"{label} BMSB: holding, {weeks_held:.1f}/{HOLD_WEEKS} weeks")

    print_equity_summary(state, current_prices)
    save_state(state)


def print_equity_summary(state: dict, current_prices: dict[str, float]) -> None:
    unrealized = 0.0
    lines = []
    for source, positions in [("golden cross", state["positions"]), ("BMSB", state["bmsb_positions"])]:
        for label, pos in positions.items():
            price = current_prices.get(label)
            if price is None:
                continue
            allocated = pos.get("allocated_amount", STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100)
            move_pct = (price / pos["entry_price"] - 1) * 100
            pnl = allocated * (move_pct / 100)
            unrealized += pnl
            lines.append(f"  {label} ({source}): {move_pct:+.2f}% -> ${pnl:+,.2f}")

    equity = state["capital"] + sum(
        pos.get("allocated_amount", STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100)
        for positions in [state["positions"], state["bmsb_positions"]] for pos in positions.values()
    ) + unrealized
    total_return_pct = (equity / STARTING_CAPITAL - 1) * 100

    print(f"\n=== equity summary ===")
    for line in lines:
        print(line)
    print(f"cash reserve: ${state['capital']:,.2f}")
    print(f"unrealized P&L: ${unrealized:+,.2f}")
    print(f"current equity: ${equity:,.2f}  (started ${STARTING_CAPITAL:,.2f}, {total_return_pct:+.2f}%)")


if __name__ == "__main__":
    main()
