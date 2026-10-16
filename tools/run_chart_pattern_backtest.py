"""Backtests the three rules-based indicators from run_chart_pattern_signals.py
(TD Sequential-simplified Setup, Bollinger squeeze/breakout, Pi Cycle) --
expanding Task #12 from "descriptive, current-state only" into an actual
trailing-365-day backtest, per explicit user request. Scoped via
AskUserQuestion before building (same "predeclare before testing"
discipline as every prior diagnostic in this project):

  - UNIVERSE: "Both, run separately" -- crypto (6 Kraken assets) and
    equities (60-symbol Alpaca universe) each get their own independent
    backtest, never blended into one combined portfolio.
  - PI CYCLE: "Overlay on golden-cross BTC" -- Pi Cycle has no buy signal
    of its own (historically it's only ever used as a "get out near the
    top" warning, never a standalone entry rule), so it is NOT
    backtested as a standalone long/flat system. Instead it's tested as
    an early-exit OVERLAY on top of the real, live, already-validated
    golden-cross BTC signal (trading_bot_v4.signals.ma200_trend) --
    isolating whether adding Pi Cycle's warning actually improves that
    real strategy, rather than inventing an entry rule Pi Cycle was
    never meant to provide.

PREDECLARED RULES (fixed before this was run against real data):

TD SETUP long/flat state: FLAT until a bullish Setup completes (count
reaches TD_SETUP_TARGET) -> LONG; stays LONG until a bearish Setup
completes -> FLAT. Symmetric, state-based, same shape every golden-cross
system in this project already uses.

BOLLINGER SQUEEZE BREAKOUT long/flat state: FLAT until breakout_up fires
-> LONG; stays LONG until breakout_down fires -> FLAT. Same state shape.

PI CYCLE OVERLAY: baseline = the REAL, live golden-cross signal already
running in production (compute_trend_state, same one run_ma200_cross_
strategy.py and the live paper trader use) on BTC 4h bars -- entries
restricted to the trailing window like everything else here. Overlay =
identical entries, but exits EARLY if BTC's DAILY Pi Cycle cross_up fires
before golden-cross's own cross_down. The daily Pi Cycle state is merged
onto the 4h bars shifted by one full day (see _merge_daily_state_onto_
intraday) so a day's own state only becomes visible starting the NEXT
day, once that daily bar has actually closed -- avoiding look-ahead into
a still-forming daily bar. Baseline vs overlay are compared over the SAME
entries and the SAME trailing window.

TEST WINDOW: trailing BACKTEST_WINDOW_DAYS (365) days from "now" (this
run's own UTC timestamp), applied to every backtest in this file
uniformly. Indicators are computed over each asset's FULL available
history first (so lookback-heavy ones -- Bollinger's 126-bar squeeze
percentile in particular -- are valid from day 1 of the window, not
artificially cold-started); only ENTRIES whose signal fires within the
window are counted, so no position can open before the window starts. A
position still open at the very last available bar is marked-to-market
there (open_at_end=True on that trade), not silently dropped as if it
never happened.

COSTS AND SIZING -- reused, not invented: crypto uses this project's own
established combined-portfolio costs/sizing (12bps round-trip, $100k
starting capital, 15% per-asset allocation, imported directly from
tools/run_ma200_cross_strategy.py); equities uses its own established
combined-portfolio costs/sizing (12bps round-trip, $100k starting
capital, 2% per-trade allocation, imported directly from
tools/run_cross_asset_golden_cross_backtest.py). The portfolio simulator
itself (_simulate_portfolio) follows the exact same event-driven pattern
run_cross_asset_golden_cross_backtest.py's own simulate_portfolio uses
(notional sized as a % of capital AT ENTRY, P&L realized at exit; capital
isn't reserved while a position is open, so aggregate notional exposure
CAN exceed 100% of capital if many assets are long at once -- an accepted
simplification already used throughout this project, not a new one),
just parameterized here so one function serves both universes at their
own distinct sizing instead of duplicating the whole thing twice.

DISCLOSED LIMITATIONS:
  - First-pass backtest over a single trailing-365-day window -- no
    walk-forward, no train/test split. Per the standing rule for this
    entire project: this is what the indicators' own rules produced over
    this specific historical window, not a forward guarantee of anything.
  - TD Setup and Bollinger squeeze/breakout are tested INDEPENDENTLY of
    each other and of the existing golden-cross system (2 separate
    standalone systems per universe) -- not combined into one signal.
    Whether combining any of these helps or hurts is a separate,
    later question, not answered here.
  - Network-touching functions (load_full_history, fetch_btc_daily_
    history, fetch_equities_bars_alpaca) could not be dry-run tested
    against live data in this sandbox -- same disclosed limitation as
    every other network function in this project. What WAS dry-run
    tested is every pure function this file adds (_trades_from_events,
    _state_from_td_setup, _state_from_bollinger, _merge_daily_state_
    onto_intraday, _restrict_entries_to_window, _simulate_portfolio),
    against hand-built data with known, hand-computed expected results.

FIVE MORE INDICATORS BACKTESTED (added 2026-09-30) -- MACD, RSI,
Stochastic RSI, Williams Vix Fix ("cm_williams"), and bsvp_xel, using the
exact formulas defined and disclosed in run_chart_pattern_signals.py's own
module docstring (read that first -- especially the bsvp_xel section,
which is an explicitly-confirmed Elder-Ray-based APPROXIMATION, not a
byte-exact port, because the real script's core primitive turned out to
be non-public, copyrighted code). Each was predeclared and confirmed via
AskUserQuestion before being built:

MACD long/flat state: standard trend-following crossover. FLAT until the
MACD line crosses ABOVE the signal line -> LONG; stays LONG until it
crosses back BELOW -> FLAT.

RSI / STOCHASTIC RSI long/flat state: standard oversold-recovery /
overbought-rollover mean-reversion rule, same shape for both (RSI at
30/70, Stochastic RSI's %K at 20/80). FLAT until the indicator crosses UP
through its oversold line (recovery) -> LONG; stays LONG until it rises to
its overbought line and then crosses back DOWN through it (momentum
rolling over) -> FLAT. Confirmed with the user as the standard convention
rather than inventing thresholds.

WILLIAMS VIX FIX ("cm_williams") long/flat state: contrarian
capitulation-buy. FLAT until a "spike" fires (wvf breaches its upper_band
OR its range_high -- a volatility/capitulation event, historically near a
bottom) -> LONG; stays LONG until wvf crosses back DOWN below upper_band
specifically (a disclosed simplification -- always the same band for the
exit regardless of which of the two triggered entry, so the rule is
always well-defined without needing extra state to remember which band
fired) -> FLAT.

BSVP_XEL long/flat state: zero-line crossover of vpo1 (the buy/sell
convergence-divergence oscillator), mirroring MACD's own crossover shape
-- directly supported by the sourced description ("positive readings
indicate bullish dominance, negative readings bearish"). FLAT until vpo1
crosses UP through 0 -> LONG; stays LONG until it crosses back DOWN
through 0 -> FLAT.

Same test-window mechanics, cost/sizing, and portfolio simulator apply to
these five exactly as they do to TD Setup and Bollinger above -- nothing
about the backtest MACHINERY changed, only which indicator supplies the
enter/exit event masks. To avoid re-fetching each universe's full price
history once per indicator (fine at 2 indicators, wasteful at 7), the
crypto and equities drivers below fetch each asset's history ONCE and then
loop every requested indicator over that same fetched data.

Run it locally:
    python tools/run_chart_pattern_backtest.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tools.run_chart_pattern_signals import (
    compute_td_setup,
    compute_bollinger_squeeze_breakout,
    compute_pi_cycle,
    compute_macd,
    compute_rsi,
    compute_stoch_rsi,
    compute_williams_vix_fix,
    compute_bsvp_xel,
    fetch_btc_daily_history,
    fetch_equities_bars_alpaca,
)
from tools.run_ma200_cross_strategy import (
    ASSETS,
    load_full_history,
    ENTRY_COST as CRYPTO_ENTRY_COST,
    EXIT_COST as CRYPTO_EXIT_COST,
    STARTING_CAPITAL as CRYPTO_STARTING_CAPITAL,
    PER_ASSET_ALLOCATION_PCT as CRYPTO_ALLOCATION_PCT,
)
from tools.run_stoploss_tail_risk_test_equities import _load_equities_universe
from tools.run_cross_asset_golden_cross_backtest import (
    ENTRY_COST as EQ_ENTRY_COST,
    EXIT_COST as EQ_EXIT_COST,
    STARTING_CAPITAL as EQ_STARTING_CAPITAL,
    PER_TRADE_ALLOCATION_PCT as EQ_ALLOCATION_PCT,
)
from trading_bot_v4.signals.ma200_trend import compute_trend_state

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
OUT_PATH = RUNTIME_DIR / "chart_pattern_backtest_results.json"

# ---- predeclared parameters -------------------------------------------------
BACKTEST_WINDOW_DAYS = 365
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80
CRYPTO_INDICATORS = ["td_setup", "bollinger", "macd", "rsi", "stoch_rsi", "williams_vix_fix", "bsvp_xel"]
EQUITIES_INDICATORS = ["td_setup", "bollinger", "macd", "rsi", "stoch_rsi", "williams_vix_fix", "bsvp_xel"]
# ------------------------------------------------------------------------------


def _state_from_td_setup(closes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure -- derives LONG-entry / LONG-exit event masks from
    compute_td_setup's own bullish/bearish Setup-complete bar lists. See
    module docstring for the exact, predeclared TD Setup long/flat rule."""
    td = compute_td_setup(closes)
    n = len(closes)
    enter_mask = np.zeros(n, dtype=bool)
    exit_mask = np.zeros(n, dtype=bool)
    for i in td["bullish_setup_complete_bars"]:
        enter_mask[i] = True
    for i in td["bearish_setup_complete_bars"]:
        exit_mask[i] = True
    return enter_mask, exit_mask


def _state_from_bollinger(boll_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pure -- LONG-entry / LONG-exit event masks are literally
    breakout_up / breakout_down, per the predeclared rule."""
    return boll_df["breakout_up"].to_numpy(), boll_df["breakout_down"].to_numpy()


def _state_from_macd(macd_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pure -- standard trend-following crossover: LONG-entry when the
    MACD line crosses above the signal line, LONG-exit on the reverse
    cross. See module docstring for the predeclared rule."""
    above = macd_df["macd_line"] > macd_df["signal_line"]
    prev_above = above.shift(1, fill_value=False)
    enter_mask = (above & ~prev_above).to_numpy()
    exit_mask = ((~above) & prev_above).to_numpy()
    return enter_mask, exit_mask


def _state_from_rsi(rsi: pd.Series, oversold: float = RSI_OVERSOLD, overbought: float = RSI_OVERBOUGHT) -> tuple[np.ndarray, np.ndarray]:
    """Pure -- oversold-recovery entry / overbought-rollover exit. See
    module docstring for the predeclared rule (same shape as
    _state_from_stoch_rsi, different thresholds/series)."""
    above_oversold = rsi > oversold
    enter_mask = (above_oversold & ~above_oversold.shift(1, fill_value=False)).to_numpy()
    overbought_state = rsi >= overbought
    exit_mask = (~overbought_state & overbought_state.shift(1, fill_value=False)).to_numpy()
    return enter_mask, exit_mask


def _state_from_stoch_rsi(stoch_k: pd.Series, oversold: float = STOCH_OVERSOLD, overbought: float = STOCH_OVERBOUGHT) -> tuple[np.ndarray, np.ndarray]:
    """Pure -- same oversold-recovery / overbought-rollover shape as
    _state_from_rsi, applied to Stochastic RSI's %K at the 20/80
    convention instead of RSI's own 30/70."""
    above_oversold = stoch_k > oversold
    enter_mask = (above_oversold & ~above_oversold.shift(1, fill_value=False)).to_numpy()
    overbought_state = stoch_k >= overbought
    exit_mask = (~overbought_state & overbought_state.shift(1, fill_value=False)).to_numpy()
    return enter_mask, exit_mask


def _state_from_williams_vix_fix(wvf_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pure -- contrarian capitulation-buy: LONG-entry on a "spike" (wvf
    breaches upper_band OR range_high), LONG-exit when wvf crosses back
    down below upper_band specifically (a disclosed simplification -- see
    module docstring)."""
    spike = wvf_df["spike"]
    enter_mask = (spike & ~spike.shift(1, fill_value=False)).to_numpy()
    above_upper = wvf_df["wvf"] > wvf_df["upper_band"]
    exit_mask = (~above_upper & above_upper.shift(1, fill_value=False)).to_numpy()
    return enter_mask, exit_mask


def _state_from_bsvp(vpo1: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Pure -- zero-line crossover of bsvp_xel's vpo1 oscillator, same
    shape as _state_from_macd's crossover."""
    above_zero = vpo1 > 0
    enter_mask = (above_zero & ~above_zero.shift(1, fill_value=False)).to_numpy()
    exit_mask = (~above_zero & above_zero.shift(1, fill_value=False)).to_numpy()
    return enter_mask, exit_mask


def _compute_indicator_masks(
    indicator: str, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch -- routes to the right compute_* + _state_from_* pair for
    a given indicator key. Centralizing this here (rather than repeating
    an if/elif ladder in both run_crypto_backtests and
    run_equities_backtests) is what lets both universes share one driver
    loop over CRYPTO_INDICATORS / EQUITIES_INDICATORS."""
    if indicator == "td_setup":
        return _state_from_td_setup(close.to_numpy(dtype=float))
    if indicator == "bollinger":
        return _state_from_bollinger(compute_bollinger_squeeze_breakout(close))
    if indicator == "macd":
        return _state_from_macd(compute_macd(close))
    if indicator == "rsi":
        return _state_from_rsi(compute_rsi(close))
    if indicator == "stoch_rsi":
        return _state_from_stoch_rsi(compute_stoch_rsi(close)["stoch_k"])
    if indicator == "williams_vix_fix":
        return _state_from_williams_vix_fix(compute_williams_vix_fix(close, low))
    if indicator == "bsvp_xel":
        return _state_from_bsvp(compute_bsvp_xel(high, low, close, volume)["vpo1"])
    raise ValueError(f"unknown indicator: {indicator!r}")


def _restrict_entries_to_window(enter_mask: np.ndarray, index: pd.Index, window_start) -> np.ndarray:
    """Pure -- zeroes out any entry signal before window_start, so no
    position can ever be opened before the trailing test window begins.
    Indicators themselves are still computed over the asset's full
    history upstream of this (so lookback-heavy ones are warmed up
    correctly) -- this only restricts which ENTRIES get counted."""
    in_window = np.asarray(index >= window_start)
    return np.asarray(enter_mask, dtype=bool) & in_window


def _trades_from_events(
    close: pd.Series,
    enter_mask: np.ndarray,
    exit_mask: np.ndarray,
    entry_cost: float,
    exit_cost: float,
) -> list[dict]:
    """Pure -- FLAT until enter_mask fires -> LONG (enter at that bar's
    close); stays LONG until exit_mask fires -> FLAT (exit at that bar's
    close). If still open at the series' last bar, marks-to-market there
    (open_at_end=True) rather than dropping the position as if it never
    happened -- same "close out the equity curve at last price"
    convention already used elsewhere in this project."""
    trades: list[dict] = []
    pos = None
    index = close.index
    values = close.to_numpy(dtype=float)
    enter_arr = np.asarray(enter_mask, dtype=bool)
    exit_arr = np.asarray(exit_mask, dtype=bool)
    n = len(values)
    for i in range(n):
        if pos is None and enter_arr[i]:
            pos = {"entry_time": index[i], "entry_price": values[i]}
        elif pos is not None and exit_arr[i]:
            fill_entry = pos["entry_price"] * (1 + entry_cost)
            fill_exit = values[i] * (1 - exit_cost)
            net_ret_pct = (fill_exit / fill_entry - 1) * 100
            trades.append({
                "entry_time": pos["entry_time"], "exit_time": index[i],
                "net_ret_pct": net_ret_pct, "entry_price": float(fill_entry),
                "exit_price": float(fill_exit), "open_at_end": False,
            })
            pos = None
    if pos is not None and n > 0:
        last_i = n - 1
        fill_entry = pos["entry_price"] * (1 + entry_cost)
        fill_exit = values[last_i] * (1 - exit_cost)
        net_ret_pct = (fill_exit / fill_entry - 1) * 100
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": index[last_i],
            "net_ret_pct": net_ret_pct, "entry_price": float(fill_entry),
            "exit_price": float(fill_exit), "open_at_end": True,
        })
    return trades


def _merge_daily_state_onto_intraday(daily_state: pd.Series, intraday_index: pd.Index) -> pd.Series:
    """Pure -- forward-fills a DAILY boolean state (Pi Cycle's cross_up)
    onto an intraday (4h) index, shifted by one full day first so a day's
    own state only becomes visible starting the NEXT day -- once that
    daily bar has actually closed -- avoiding look-ahead into a
    still-forming daily bar."""
    shifted = daily_state.shift(1, fill_value=False)
    return shifted.reindex(intraday_index, method="ffill").fillna(False).astype(bool)


def _simulate_portfolio(all_trades: list[dict], starting_capital: float, allocation_pct: float) -> dict:
    """Pure -- event-driven combined-portfolio sim, same pattern
    run_cross_asset_golden_cross_backtest.py's own simulate_portfolio
    uses (notional sized as allocation_pct of capital AT ENTRY, P&L
    realized at exit; capital isn't reserved while a position is open),
    parameterized here so one function serves both universes at their
    own established sizing instead of duplicating this twice."""
    events = []
    for t in all_trades:
        events.append((t["entry_time"], 0, "entry", t))
        events.append((t["exit_time"], 1, "exit", t))
    events.sort(key=lambda e: (e[0], e[1]))

    capital = starting_capital
    peak = starting_capital
    max_dd = 0.0
    equity_curve = []
    if events:
        equity_curve.append({"time": str(events[0][0]), "equity": capital})

    for ts, _, kind, t in events:
        if kind == "entry":
            t["_notional"] = capital * allocation_pct / 100.0
        else:
            pnl = t["_notional"] * (t["net_ret_pct"] / 100.0)
            capital += pnl
            peak = max(peak, capital)
            max_dd = min(max_dd, (capital / peak - 1) * 100)
            equity_curve.append({"time": str(ts), "equity": capital})

    rets = [t["net_ret_pct"] for t in all_trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    if losses and sum(losses) != 0:
        pf = sum(wins) / abs(sum(losses))
    elif wins:
        pf = float("inf")
    else:
        pf = None

    return {
        "starting_capital": starting_capital,
        "final_capital": capital,
        "total_return_pct": (capital / starting_capital - 1) * 100,
        "max_drawdown_pct": max_dd,
        "n_trades": len(all_trades),
        "win_rate_pct": (len(wins) / len(rets) * 100) if rets else None,
        "profit_factor": pf,
    }


def run_crypto_backtests(indicators: list[str], window_start) -> dict[str, dict]:
    """Network-touching (Kraken fetch via load_full_history) -- not
    dry-run tested in this sandbox, same disclosed limitation as every
    other network function in this project. The pure logic it calls IS
    dry-run tested. Fetches each asset's full history ONCE, then runs
    every requested indicator's backtest against that same fetched data
    -- avoids re-hitting Kraken once per indicator now that there are 7
    of them (was a non-issue at the original 2)."""
    price: dict[str, pd.DataFrame] = {}
    for label, (pair, path) in ASSETS.items():
        price[label] = load_full_history(pair, path)

    results: dict[str, dict] = {}
    for indicator in indicators:
        all_trades: list[dict] = []
        per_symbol_trade_counts: dict[str, int] = {}
        for label, df in price.items():
            close = df["close"]
            enter_mask, exit_mask = _compute_indicator_masks(indicator, close, df["high"], df["low"], df["volume"])
            enter_mask = _restrict_entries_to_window(enter_mask, close.index, window_start)
            trades = _trades_from_events(close, enter_mask, exit_mask, CRYPTO_ENTRY_COST, CRYPTO_EXIT_COST)
            for t in trades:
                t["symbol"] = label
            all_trades.extend(trades)
            per_symbol_trade_counts[label] = len(trades)
        portfolio = _simulate_portfolio(all_trades, CRYPTO_STARTING_CAPITAL, CRYPTO_ALLOCATION_PCT)
        portfolio["per_symbol_trade_counts"] = per_symbol_trade_counts
        results[indicator] = portfolio
    return results


def run_equities_backtests(indicators: list[str], window_start) -> dict[str, dict]:
    """Network-touching (Alpaca fetch via fetch_equities_bars_alpaca) --
    not dry-run tested in this sandbox. The pure logic it calls IS
    dry-run tested. Fetches the equities universe ONCE, then runs every
    requested indicator's backtest against that same fetched data (same
    single-fetch-then-loop reasoning as run_crypto_backtests)."""
    symbols = _load_equities_universe()
    price = fetch_equities_bars_alpaca(symbols)

    results: dict[str, dict] = {}
    for indicator in indicators:
        all_trades: list[dict] = []
        per_symbol_trade_counts: dict[str, int] = {}
        for sym, df in price.items():
            close = df["Close"]
            enter_mask, exit_mask = _compute_indicator_masks(indicator, close, df["High"], df["Low"], df["Volume"])
            enter_mask = _restrict_entries_to_window(enter_mask, close.index, window_start)
            trades = _trades_from_events(close, enter_mask, exit_mask, EQ_ENTRY_COST, EQ_EXIT_COST)
            for t in trades:
                t["symbol"] = sym
            all_trades.extend(trades)
            per_symbol_trade_counts[sym] = len(trades)
        portfolio = _simulate_portfolio(all_trades, EQ_STARTING_CAPITAL, EQ_ALLOCATION_PCT)
        portfolio["per_symbol_trade_counts"] = per_symbol_trade_counts
        results[indicator] = portfolio
    return results


def run_pi_cycle_overlay_backtest(window_start) -> dict:
    """BTC only. Baseline = real, live golden-cross entries/exits.
    Overlay = identical entries, exits early if daily Pi Cycle cross_up
    fires first. See module docstring for the exact rule and the
    look-ahead-avoidance reasoning behind the 1-day shift. Network-
    touching -- not dry-run tested in this sandbox; the pure logic
    (_merge_daily_state_onto_intraday, _trades_from_events,
    _simulate_portfolio) IS."""
    pair, path = ASSETS["BTC"]
    df4h = load_full_history(pair, path)
    close4h = df4h["close"]
    gc = compute_trend_state(close4h)

    daily_close = fetch_btc_daily_history()
    pc = compute_pi_cycle(daily_close)
    pi_cross_up_on_4h = _merge_daily_state_onto_intraday(pc["cross_up"], close4h.index)

    enter_mask = _restrict_entries_to_window(gc["cross_up"].to_numpy(), close4h.index, window_start)
    baseline_exit_mask = gc["cross_down"].to_numpy()
    overlay_exit_mask = (gc["cross_down"] | pi_cross_up_on_4h).to_numpy()

    baseline_trades = _trades_from_events(close4h, enter_mask, baseline_exit_mask, CRYPTO_ENTRY_COST, CRYPTO_EXIT_COST)
    overlay_trades = _trades_from_events(close4h, enter_mask, overlay_exit_mask, CRYPTO_ENTRY_COST, CRYPTO_EXIT_COST)

    baseline_portfolio = _simulate_portfolio(baseline_trades, CRYPTO_STARTING_CAPITAL, CRYPTO_ALLOCATION_PCT)
    overlay_portfolio = _simulate_portfolio(overlay_trades, CRYPTO_STARTING_CAPITAL, CRYPTO_ALLOCATION_PCT)
    return {
        "baseline_golden_cross_only": baseline_portfolio,
        "overlay_with_pi_cycle_early_exit": overlay_portfolio,
    }


def main() -> None:
    as_of = datetime.now(timezone.utc)
    window_start = as_of - timedelta(days=BACKTEST_WINDOW_DAYS)

    print("=" * 100)
    print("CHART-PATTERN INDICATOR BACKTEST -- trailing "
          f"{BACKTEST_WINDOW_DAYS} days, as of {as_of.isoformat()}")
    print("7 indicators tested standalone per universe (TD Setup, Bollinger, MACD, RSI,")
    print("Stochastic RSI, Williams Vix Fix, bsvp_xel). Pi Cycle tested separately as an")
    print("early-exit overlay on the real golden-cross BTC signal (no standalone entry rule).")
    print("First-pass backtest -- no walk-forward, no promise of forward performance.")
    print("=" * 100)

    results = {}

    def _fmt(indicator: str, r: dict) -> str:
        win_rate = f"{r['win_rate_pct']:.1f}%" if r["win_rate_pct"] is not None else "n/a"
        return (f"  {indicator:<18} {r['n_trades']:>4} trades, total_return_pct={r['total_return_pct']:+7.2f}%, "
                f"win_rate={win_rate} max_dd={r['max_drawdown_pct']:.1f}%")

    print("\n-- CRYPTO: all 7 indicators (single fetch, looped) --")
    crypto_results = run_crypto_backtests(CRYPTO_INDICATORS, window_start)
    for indicator, r in crypto_results.items():
        results[f"crypto_{indicator}"] = r
        print(_fmt(indicator, r))

    print("\n-- EQUITIES: all 7 indicators (single fetch, looped) --")
    equities_results = run_equities_backtests(EQUITIES_INDICATORS, window_start)
    for indicator, r in equities_results.items():
        results[f"equities_{indicator}"] = r
        print(_fmt(indicator, r))

    print("\n-- BTC: Pi Cycle overlay vs. golden-cross baseline --")
    results["pi_cycle_overlay"] = run_pi_cycle_overlay_backtest(window_start)
    b = results["pi_cycle_overlay"]["baseline_golden_cross_only"]
    o = results["pi_cycle_overlay"]["overlay_with_pi_cycle_early_exit"]
    print(f"  baseline (golden-cross only): {b['n_trades']} trades, total_return_pct={b['total_return_pct']:+.2f}%")
    print(f"  overlay  (+ Pi Cycle exit):   {o['n_trades']} trades, total_return_pct={o['total_return_pct']:+.2f}%")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "as_of": as_of.isoformat(),
        "backtest_window_days": BACKTEST_WINDOW_DAYS,
        "crypto_entry_cost_bps_each_way": CRYPTO_ENTRY_COST * 10000,
        "crypto_allocation_pct": CRYPTO_ALLOCATION_PCT,
        "equities_entry_cost_bps_each_way": EQ_ENTRY_COST * 10000,
        "equities_allocation_pct": EQ_ALLOCATION_PCT,
        "results": results,
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    print("\nFirst-pass backtest over one trailing-365-day window -- no walk-forward,")
    print("no promise this holds up going forward. Nothing here is wired into any live system.")


if __name__ == "__main__":
    main()
