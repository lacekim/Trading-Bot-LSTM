"""The 200-period moving average family of trend signals, starting from the
user's original rule ("simply look when it crossed the 200") and upgraded
to a golden cross (50MA over 200MA) after both were tested honestly on real
Kraken data, real costs, over the full available history and the golden
cross came back meaningfully better:

  Pure 200MA cross:  BTC PF 1.41, ETH PF 1.62 excluding the 3 biggest
                      winners -- real, but low win rate (13-16%) and fails
                      on XMR (0.79 excl. top 3, not robust).
  Golden cross 50/200: BTC PF 1.85, ETH PF 2.61, XMR PF 1.18, all excluding
                      the 3 biggest winners -- higher win rate (42-49% vs
                      13-16%), more robust on BTC/ETH, and the only version
                      that passes the robustness bar on XMR too.

compute_trend_state / current_signal below implement the golden cross
(current, active rule). There is still no fixed profit target and no
ATR-based exit: the position is held for as long as the 50MA stays above
the 200MA, and closed when the 50MA crosses back below (a "death cross").
"""
from __future__ import annotations

import pandas as pd

FAST_MA_PERIOD = 50
SLOW_MA_PERIOD = 200


def compute_trend_state(close: pd.Series) -> pd.DataFrame:
    """Given a close-price series (4h bars, matching what was backtested),
    return the golden-cross state and cross events. No lookahead: ma50[t],
    ma200[t], and above200[t] only use data through bar t."""
    ma50 = close.rolling(FAST_MA_PERIOD).mean()
    ma200 = close.rolling(SLOW_MA_PERIOD).mean()
    above200 = ma50 > ma200  # "above200" name kept for interface stability; means golden-cross-active
    # .shift(1) on a bool Series introduces a leading NaN, which silently
    # upcasts the whole thing to object dtype -- .fillna(False) removes the
    # NaN but the dtype stays object, so `~` no longer does boolean negation:
    # it calls Python's int.__invert__ on the underlying bool (True -> -2,
    # False -> -1), both of which are truthy. That made cross_up/cross_down
    # stay True on every bar after the first transition instead of firing
    # once -- confirmed on real BTC data (13,857/25,906 bars flagged
    # cross_up, vs. above200 itself only flipping 146 times). The backtest's
    # trade list was unaffected (its entry/exit logic gates on position
    # state, not on this column), but anything reading just_crossed_up/down
    # directly -- live notifications, the order-flow capture hook -- would
    # have fired every bar instead of once. .astype(bool) restores proper
    # bool dtype before the negation.
    prev_above200 = above200.shift(1).fillna(False).astype(bool)
    cross_up = above200 & ~prev_above200
    cross_down = (~above200) & prev_above200
    return pd.DataFrame({
        "close": close, "ma50": ma50, "ma200": ma200, "above200": above200,
        "cross_up": cross_up, "cross_down": cross_down,
    })


def current_signal(close: pd.Series) -> dict:
    """Real-time read: is the 50MA above the 200MA right now (golden cross
    active), and how far apart they are. Requires at least SLOW_MA_PERIOD
    bars of history in `close`."""
    state = compute_trend_state(close)
    last = state.iloc[-1]
    if pd.isna(last["ma200"]):
        return {"signal": "INSUFFICIENT_DATA"}
    distance_pct = (last["ma50"] / last["ma200"] - 1) * 100
    return {
        "signal": "LONG" if last["above200"] else "FLAT",
        "close": float(last["close"]),
        "ma50": float(last["ma50"]),
        "ma200": float(last["ma200"]),
        "distance_pct": float(distance_pct),
        "just_crossed_up": bool(last["cross_up"]),
        "just_crossed_down": bool(last["cross_down"]),
        "as_of": state.index[-1],
    }
