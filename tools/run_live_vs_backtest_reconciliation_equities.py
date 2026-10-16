"""Live-vs-backtest reconciliation for the Alpaca paper-trading deployment
of the locked equities/commodities/bonds golden-cross control validated in
EQUITIES_EXIT_LOGIC_VALIDATION_RECORD.md.

READ-ONLY. This script never places, modifies, or cancels an order, and
never touches tools/run_alpaca_paper_trader.py or its running loop. It only
READS the live paper account (via the same alpaca-py client already used to
manage it -- same connection pattern as tools/check_alpaca_env.py) and
READS history (yfinance, via the exact fetch_ohlcv_batch / build_states
pipeline used throughout the validation sequence) to compare what is
actually happening in production against what the validated, locked
control specifies. Run this locally, where the real ALPACA_API_KEY /
ALPACA_SECRET_KEY and Alpaca network access live:

    python tools/run_live_vs_backtest_reconciliation_equities.py

SCOPE, DISCLOSED:

  - Restricted to the CURRENT equities/commodities/bonds universe (robust==
    True AND asset_class != "Crypto" -- the same 60-symbol universe every
    validation script in this sequence used: 57 equities + GC=F + PA=F +
    IEF, per run_universe_concentration_diagnostics_equities.py's own
    disclosed finding). The migrated-crypto leg (BTC/ETH/SOL/PAXG, Kraken-
    signaled, 15% sizing, -8% stop) is explicitly OUT OF SCOPE -- it was
    never part of this validation sequence and has its own separate signal
    source.

  - The two already-disclosed spec discrepancies this script exists to
    quantify (PARTS A and B below):

      (1) SIZING BASIS (PART A). The live script sizes new entries off
          Alpaca's own `account.equity` -- total mark-to-market equity,
          including unrealized P&L of every open position in BOTH the
          equities and migrated-crypto legs. The validated backtest sizes
          off a realized-capital-only basis (`capital` in
          run_independent_ledger_reconciliation_equities.py, never
          decremented at entry, only ever updated by `capital += pnl` at
          exit -- so it excludes both the notional committed to open
          positions and their unrealized P&L). This script reconstructs a
          real-terms analog of that basis (`capital_analog`) and compares
          hypothetical current-moment sizing under both bases.
          `capital_analog` IS APPROXIMATE, DISCLOSED HERE: the live
          account's cash is a SINGLE shared pool across both the equities/
          commodities/bonds leg and the migrated-crypto leg (Alpaca has no
          sub-account separation), so `capital_analog` computed from total
          account cash is contaminated by crypto-leg cash flows this
          script cannot cleanly separate out. Treat it as a present-
          moment, order-of-magnitude estimate of the sizing-basis drift,
          not a precise reconstruction. This is also PRESENT-TENSE, not a
          per-trade historical reconstruction: this script has no live
          account-equity history, so it cannot know what "2% of equity"
          actually was at any past entry's real moment -- it only answers
          "what would a new entry be sized at right now, under each
          basis."

      (2) STOP MECHANISM (PART B). The live script closes a position when
          Alpaca's own `unrealized_plpc` (checked once per 4h loop cycle,
          see run_alpaca_paper_trader_loop.py) breaches
          EQUITIES_HARD_STOP_PCT. The validated backtest
          (run_gap_through_stop_backtest_equities.py) checks once per
          DAILY BAR whether that day's LOW touched or gapped through the
          stop price computed from the entry fill. Both mechanisms use the
          SAME threshold and the SAME reference price (the position's own
          entry fill) -- what differs is check timing/granularity, not the
          stop level. For every currently open in-scope position, this
          script checks whether the daily-bar model would already have
          flagged an exit that the live 4h-poll mechanism has not (yet)
          caught, which can happen if price gapped through the stop
          intraday and recovered by the next 4h check. The daily-low
          series is restricted to on-or-after that position's own most
          recent BUY fill (found via a 2-year order-history lookup,
          disclosed simplification: does not distinguish an original entry
          from a later add-to-position) -- a position whose entry fill
          falls outside that 2-year window is reported as
          entry_date_unknown and skipped, rather than guessed at over the
          whole fetched history (which would risk false positives from
          before the position even existed).

  - PART C recomputes today's golden-cross signal state (ma50 vs ma200,
    the same `above200` column build_states already produces) for every
    in-scope symbol and flags any mismatch against what Alpaca currently
    holds -- a consistency check, not a trade-by-trade fill audit. A
    mismatch can be entirely benign (e.g. a real cash-constraint skip
    already logged by the live script's own ALPACA_CASH_CHECK patch, or a
    signal that flipped since the live script's last 4h cycle) -- this
    script does not attempt to explain WHY a mismatch exists, only to
    surface that one does.

  - PART D lists closed (filled) orders for in-scope symbols in a trailing
    lookback window, descriptively, alongside that fill date's own
    recomputed golden-cross state -- descriptive context only. This
    script does NOT attempt to algorithmically match a specific live fill
    to a specific backtest-predicted trade: fill timing, partial fills,
    and real cash-constraint skips make that a fundamentally different,
    harder problem than what is validated here, and attempting it would
    overclaim precision this script does not have.

WHAT THIS DOES NOT DO: place, modify, or cancel any order; touch
tools/run_alpaca_paper_trader.py or its running loop; resolve the
survivorship-bias fork (still an explicit, separate, pending decision);
or imply that any of the exit-variant/sizing-variant research already
closed out in EQUITIES_EXIT_LOGIC_VALIDATION_RECORD.md is being revisited.
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
from dotenv import load_dotenv

from tools.run_alpaca_paper_trader import (
    FUTURES_TO_ETF_PROXY,
    EQUITY_SYMBOL_MAP,
    EQUITIES_HARD_STOP_PCT,
    PER_ASSET_ALLOCATION_PCT,
)
from tools.run_sizing_allocation_backtest_equities import build_states
from tools.run_pullback_entry_backtest_equities import fetch_ohlcv_batch

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
RESULTS_PATH = RUNTIME_DIR / "cross_asset_golden_cross_results.json"
OUT_PATH = RUNTIME_DIR / "live_vs_backtest_reconciliation_results.json"

# Predeclared, not tuned to any result seen from a real run.
ORDER_LOOKBACK_DAYS = 90          # PART D descriptive fill-history window
ENTRY_LOOKUP_LOOKBACK_DAYS = 730  # PART B entry-fill-date lookup window (2y, matches this project's
                                  # standard yfinance "2y" fetch period used throughout)

# ---------------------------------------------------------------------------
# Symbol mapping -- reversed from the SAME maps the live script uses
# (imported directly above, never redefined here), so this can never
# silently drift from what run_alpaca_paper_trader.py actually does.
# ---------------------------------------------------------------------------
_INVERSE_FUTURES_PROXY = {etf: fut for fut, etf in FUTURES_TO_ETF_PROXY.items()}
_INVERSE_EQUITY_MAP = {alp: yf for yf, alp in EQUITY_SYMBOL_MAP.items()}


def yf_symbol_for_alpaca(alpaca_symbol: str) -> str:
    """Reverse of the exact substitution run_alpaca_paper_trader.py's
    main() applies going the other way (yf_symbol -> alpaca_symbol). Any
    Alpaca symbol not in either reverse map is assumed to already be its
    own yfinance symbol (true for all 57 plain equities + IEF)."""
    return _INVERSE_FUTURES_PROXY.get(alpaca_symbol) or _INVERSE_EQUITY_MAP.get(alpaca_symbol) or alpaca_symbol


def alpaca_symbol_for_yf(yf_symbol: str) -> str:
    """Forward direction, mirroring the inline expression in
    run_alpaca_paper_trader.py's own main() exactly (FUTURES_TO_ETF_PROXY,
    then EQUITY_SYMBOL_MAP, then unchanged) -- minus the CRYPTO_SYMBOL_MAP
    branch, which never applies to this script's in-scope universe."""
    return FUTURES_TO_ETF_PROXY.get(yf_symbol) or EQUITY_SYMBOL_MAP.get(yf_symbol) or yf_symbol


def _load_current_universe() -> set[str]:
    """The exact CURRENT universe filter used by every validation script in
    this sequence: robust==True AND asset_class != "Crypto" -- 60 symbols
    (57 equities + GC=F + PA=F + IEF, see
    run_universe_concentration_diagnostics_equities.py's own disclosed
    finding). Reimplemented locally rather than imported: this filter is a
    one-line, already-stable, project-wide convention repeated verbatim in
    five other scripts already -- unlike the Alpaca-side maps/constants
    above, there is no live-drift risk here to guard against by importing
    a single shared source."""
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {r["symbol"] for r in data["results"] if bool(r.get("robust")) and r.get("asset_class") != "Crypto"}


# ---------------------------------------------------------------------------
# Alpaca-object -> plain-dict conversion. ALL alpaca-py attribute access is
# isolated to these two functions (plus _find_entry_dates's own order
# fetch) so every function below can be fully tested with plain dicts /
# SimpleNamespace stand-ins, without alpaca-py installed or a real account
# reachable.
# ---------------------------------------------------------------------------
def positions_to_records(positions) -> list[dict]:
    out = []
    for p in positions:
        out.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "cost_basis": float(p.cost_basis),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc) * 100,  # alpaca-py returns a fraction, not a percent
        })
    return out


def orders_to_records(orders) -> list[dict]:
    out = []
    for o in orders:
        if o.filled_at is None or o.filled_avg_price is None:
            continue  # closed-but-unfilled (e.g. canceled) -- not a real fill, out of scope here
        side = o.side.value if hasattr(o.side, "value") else o.side
        filled_at = o.filled_at.isoformat() if hasattr(o.filled_at, "isoformat") else str(o.filled_at)
        out.append({
            "symbol": o.symbol,
            "side": str(side),
            "qty": float(o.filled_qty) if o.filled_qty is not None else None,
            "filled_avg_price": float(o.filled_avg_price),
            "filled_at": filled_at,
        })
    return out


# ---------------------------------------------------------------------------
# PART A -- sizing basis, present-tense.
# ---------------------------------------------------------------------------
def capital_analog(cash: float, in_scope_open_cost_bases: list[float]) -> float:
    """Real-terms analog of the backtest's realized-capital-only `capital`
    ledger (see run_independent_ledger_reconciliation_equities.py): cash
    (already reflects every past realized gain/loss) plus the ORIGINAL COST
    BASIS (not current market value) of every currently open in-scope
    position -- because the backtest's own `capital` variable is never
    decremented at entry and only ever updated by `capital += pnl` at exit,
    so it excludes both the notional committed to open positions and their
    unrealized P&L. Restricted by the caller to in-scope (equities/
    commodities/bonds) positions only -- see module docstring's disclosed
    cross-leg cash-contamination caveat."""
    return cash + sum(in_scope_open_cost_bases)


def hypothetical_sizing_comparison(equity: float, capital_analog_value: float,
                                    allocation_pct: float = PER_ASSET_ALLOCATION_PCT) -> dict:
    """What a brand-new entry signal, if it fired right now, would be sized
    at under each basis -- NOT a retrospective per-trade reconstruction
    (this script has no live account-equity history, so it cannot know
    what "2% of equity" was at any past entry's actual moment)."""
    notional_live_basis = round(equity * allocation_pct / 100, 2)
    notional_capital_analog_basis = round(capital_analog_value * allocation_pct / 100, 2)
    drift = notional_live_basis - notional_capital_analog_basis
    drift_pct_of_capital_basis = (
        (drift / notional_capital_analog_basis * 100) if notional_capital_analog_basis else float("nan")
    )
    return {
        "notional_live_equity_basis": notional_live_basis,
        "notional_capital_analog_basis": notional_capital_analog_basis,
        "drift_dollars": round(drift, 2),
        "drift_pct_of_capital_analog_basis": drift_pct_of_capital_basis,
    }


# ---------------------------------------------------------------------------
# PART B -- stop mechanism divergence, per currently open in-scope position.
# ---------------------------------------------------------------------------
def live_stop_would_trigger(unrealized_plpc_pct: float, stop_pct: float = EQUITIES_HARD_STOP_PCT) -> bool:
    """Exact replica of run_alpaca_paper_trader.py's own live check:
    `unrealized_pct <= stop_pct`."""
    return unrealized_plpc_pct <= stop_pct


def backtest_model_stop_breach(daily_low_since_entry: pd.Series, entry_price: float,
                                stop_pct: float = EQUITIES_HARD_STOP_PCT) -> dict:
    """Replays run_gap_through_stop_backtest_equities.py's own daily-bar
    low-touch/gap model against the REAL fill price and REAL daily lows
    since entry: was the fixed stop distance from entry_price ever touched
    or gapped through on any day's low? Returns the first breach date if
    so. `daily_low_since_entry` must already be sliced by the caller to
    start on-or-after the position's own entry date -- this function does
    not know or assume anything about that slicing, so it isolates purely
    the CHECK-TIMING/CHECK-GRANULARITY difference (once-per-4h last-price
    poll vs. once-per-daily-bar low-touch), never a different stop level
    (entry_price is the same reference both mechanisms use -- live's
    unrealized_plpc is itself relative to avg_entry_price)."""
    stop_price = entry_price * (1 + stop_pct / 100)
    if daily_low_since_entry.empty:
        return {"ever_breached": False, "first_breach_date": None, "stop_price": stop_price}
    breached = daily_low_since_entry <= stop_price
    if not breached.any():
        return {"ever_breached": False, "first_breach_date": None, "stop_price": stop_price}
    first_date = daily_low_since_entry.index[breached.argmax()]
    return {"ever_breached": True, "first_breach_date": str(first_date), "stop_price": stop_price}


def reconcile_stop_mechanism(unrealized_plpc_pct: float, daily_low_since_entry: pd.Series, entry_price: float,
                              stop_pct: float = EQUITIES_HARD_STOP_PCT) -> dict:
    live = live_stop_would_trigger(unrealized_plpc_pct, stop_pct)
    model = backtest_model_stop_breach(daily_low_since_entry, entry_price, stop_pct)
    divergence = bool(model["ever_breached"] and not live)
    return {
        "live_stop_would_trigger_now": live,
        "backtest_model_ever_breached": model["ever_breached"],
        "backtest_model_first_breach_date": model["first_breach_date"],
        "stop_price": model["stop_price"],
        "divergence_model_would_have_exited_but_live_has_not": divergence,
    }


def _find_entry_dates(client, alpaca_symbols: list[str], lookback_days: int = ENTRY_LOOKUP_LOOKBACK_DAYS) -> dict:
    """Latest filled BUY order per symbol, used as this position's entry-
    date proxy (disclosed simplification: does not distinguish an original
    entry from a later add-to-position -- if a symbol was bought more than
    once in this window, the most recent BUY is used). Returns ISO date
    strings, not full objects, keeping this Alpaca-glue function's return
    type as plain as everything else in this module. `client` needs only a
    `get_orders(request)` method, so this is testable with a fake client
    object -- no real alpaca-py connection required."""
    if not alpaca_symbols:
        return {}
    from alpaca.trading.enums import OrderSide, QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    after = datetime.now(timezone.utc) - pd.Timedelta(days=lookback_days)
    req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=alpaca_symbols,
                           side=OrderSide.BUY, after=after, limit=500, nested=False)
    orders = client.get_orders(req)
    records = orders_to_records(orders)
    latest: dict = {}
    for r in records:
        sym = r["symbol"]
        if sym not in latest or r["filled_at"] > latest[sym]:
            latest[sym] = r["filled_at"]
    return latest


# ---------------------------------------------------------------------------
# PART C -- signal-state cross-check.
# ---------------------------------------------------------------------------
def signal_state_classification(above200_now: bool, currently_held: bool) -> str:
    if above200_now and currently_held:
        return "consistent_long_held"
    if not above200_now and not currently_held:
        return "consistent_flat_unheld"
    if above200_now and not currently_held:
        return "signal_long_not_held"
    return "signal_flat_still_held"


# ---------------------------------------------------------------------------
# PART D -- descriptive fill-history context (no automated match/verdict).
# ---------------------------------------------------------------------------
def annotate_fill_with_signal_state(fill: dict, states_for_symbol) -> dict:
    out = dict(fill)
    if states_for_symbol is None or states_for_symbol.empty:
        out["golden_cross_state_on_fill_date"] = None
        return out
    try:
        fill_date = pd.Timestamp(fill["filled_at"]).date()
    except Exception:
        out["golden_cross_state_on_fill_date"] = None
        return out
    match_mask = states_for_symbol.index.date == fill_date
    if not match_mask.any():
        out["golden_cross_state_on_fill_date"] = None
        return out
    row = states_for_symbol.loc[match_mask].iloc[0]
    out["golden_cross_state_on_fill_date"] = {
        "above200": bool(row["above200"]),
        "cross_up": bool(row["cross_up"]),
        "cross_down": bool(row["cross_down"]),
    }
    return out


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set -- see tools/check_alpaca_env.py.\n"
            "This script is READ-ONLY: it never places, modifies, or cancels an order."
        )

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    client = TradingClient(api_key, secret_key, paper=True)
    account = client.get_account()
    equity = float(account.equity)
    cash = float(account.cash)

    current_universe = _load_current_universe()
    all_positions = positions_to_records(client.get_all_positions())
    in_scope_positions = [p for p in all_positions if yf_symbol_for_alpaca(p["symbol"]) in current_universe]
    out_of_scope_count = len(all_positions) - len(in_scope_positions)

    print("=" * 108)
    print("LIVE-VS-BACKTEST RECONCILIATION -- equities/commodities/bonds leg only (READ-ONLY, no orders placed)")
    print("=" * 108)
    print(f"connected: equity=${equity:,.2f}  cash=${cash:,.2f}  "
          f"in-scope open positions={len(in_scope_positions)}  "
          f"(excluded as out-of-scope, e.g. migrated crypto: {out_of_scope_count})")

    # --- PART A: sizing basis --------------------------------------------------
    cap_analog = capital_analog(cash, [p["cost_basis"] for p in in_scope_positions])
    sizing = hypothetical_sizing_comparison(equity, cap_analog)
    print("\nPART A -- sizing basis (present-moment; NOT a per-trade historical reconstruction)")
    print(f"  capital_analog (cash + in-scope open cost bases) = ${cap_analog:,.2f}   "
          f"(APPROXIMATE -- cash pool is shared with the migrated-crypto leg, see module docstring)")
    print(f"  if a new entry signaled right now, notional sized at {PER_ASSET_ALLOCATION_PCT:.0f}%:")
    print(f"    off live account.equity   -> ${sizing['notional_live_equity_basis']:,.2f}")
    print(f"    off capital_analog        -> ${sizing['notional_capital_analog_basis']:,.2f}")
    print(f"    drift                     -> ${sizing['drift_dollars']:+,.2f}  "
          f"({sizing['drift_pct_of_capital_analog_basis']:+.1f}% of the capital_analog-basis size)")

    # --- fetch OHLCV once for every in-scope symbol; used by Parts B, C, D
    yf_symbols = sorted(current_universe)
    price = fetch_ohlcv_batch(yf_symbols)
    states = build_states(price)

    # --- PART B: stop mechanism -------------------------------------------------
    print("\nPART B -- stop mechanism divergence (currently open in-scope positions)")
    held_alpaca_symbols_sorted = sorted({p["symbol"] for p in in_scope_positions})
    entry_dates = _find_entry_dates(client, held_alpaca_symbols_sorted)
    stop_reconciliations = {}
    for p in in_scope_positions:
        alpaca_sym = p["symbol"]
        yf_sym = yf_symbol_for_alpaca(alpaca_sym)
        state = states.get(yf_sym)
        if state is None or state.empty:
            print(f"  {alpaca_sym:<8} no usable OHLCV history fetched -- skipped")
            continue
        entry_date_str = entry_dates.get(alpaca_sym)
        if entry_date_str is None:
            print(f"  {alpaca_sym:<8} entry fill not found in the {ENTRY_LOOKUP_LOOKBACK_DAYS}-day order-history "
                  f"window -- stop-model check skipped (entry_date_unknown)")
            stop_reconciliations[alpaca_sym] = {"entry_date_unknown": True}
            continue
        entry_date = pd.Timestamp(entry_date_str).date()
        low_since_entry = state.loc[state.index.date >= entry_date, "low"]
        recon = reconcile_stop_mechanism(p["unrealized_plpc"], low_since_entry, p["avg_entry_price"])
        stop_reconciliations[alpaca_sym] = recon
        flag = " <-- DIVERGENCE" if recon["divergence_model_would_have_exited_but_live_has_not"] else ""
        breach_str = f" ({recon['backtest_model_first_breach_date']})" if recon["backtest_model_first_breach_date"] else ""
        print(f"  {alpaca_sym:<8} unrealized={p['unrealized_plpc']:+6.2f}%  "
              f"live_would_trigger={str(recon['live_stop_would_trigger_now']):<5}  "
              f"model_ever_breached={str(recon['backtest_model_ever_breached']):<5}{breach_str}{flag}")

    # --- PART C: signal-state cross-check ---------------------------------------
    print("\nPART C -- golden-cross signal-state cross-check (all in-scope symbols)")
    held_alpaca_symbols = set(held_alpaca_symbols_sorted)
    signal_summary = {"consistent_long_held": 0, "consistent_flat_unheld": 0,
                       "signal_long_not_held": 0, "signal_flat_still_held": 0}
    mismatches = []
    for yf_sym in yf_symbols:
        state = states.get(yf_sym)
        if state is None or state.empty:
            continue
        above200_now = bool(state["above200"].iloc[-1])
        alpaca_sym = alpaca_symbol_for_yf(yf_sym)
        currently_held = alpaca_sym in held_alpaca_symbols
        cls = signal_state_classification(above200_now, currently_held)
        signal_summary[cls] += 1
        if cls in ("signal_long_not_held", "signal_flat_still_held"):
            mismatches.append({"yf_symbol": yf_sym, "alpaca_symbol": alpaca_sym, "classification": cls})

    for cls, n in signal_summary.items():
        print(f"  {cls:<28} {n}")
    if mismatches:
        print("  mismatches:")
        for m in mismatches:
            print(f"    {m['yf_symbol']:<10} ({m['alpaca_symbol']}) -- {m['classification']}")
    else:
        print("  no mismatches")

    # --- PART D: descriptive fill-history context -------------------------------
    print(f"\nPART D -- descriptive fill history, last {ORDER_LOOKBACK_DAYS}d, in-scope symbols "
          f"(context only -- not an automated trade-match)")
    in_scope_alpaca_symbols = sorted({alpaca_symbol_for_yf(s) for s in yf_symbols})
    after = datetime.now(timezone.utc) - pd.Timedelta(days=ORDER_LOOKBACK_DAYS)
    orders_req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=in_scope_alpaca_symbols,
                                  after=after, limit=500, nested=False)
    orders = client.get_orders(orders_req)
    fills = orders_to_records(orders)
    annotated_fills = [annotate_fill_with_signal_state(f, states.get(yf_symbol_for_alpaca(f["symbol"])))
                       for f in fills]
    for f in annotated_fills[:30]:
        gc = f.get("golden_cross_state_on_fill_date")
        gc_str = (f"above200={gc['above200']} cross_up={gc['cross_up']} cross_down={gc['cross_down']}"
                  if gc else "n/a")
        print(f"  {f['filled_at']}  {f['symbol']:<8} {f['side']:<4} qty={f['qty']}  "
              f"price={f['filled_avg_price']}  {gc_str}")
    if len(annotated_fills) > 30:
        print(f"  ... and {len(annotated_fills) - 30} more (see output JSON)")
    if not annotated_fills:
        print("  none in this window")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "account": {"equity": equity, "cash": cash},
        "part_a_sizing": {"capital_analog": cap_analog, **sizing},
        "part_b_stop_mechanism": stop_reconciliations,
        "part_c_signal_state": {"summary": signal_summary, "mismatches": mismatches},
        "part_d_fill_history": annotated_fills,
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
