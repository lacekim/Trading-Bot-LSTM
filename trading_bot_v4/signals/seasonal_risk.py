"""Real, verified seasonal risk context for the Nov-Feb window on BTC.

This is informational only -- NOT an automated exit trigger. A hard
"exit during Nov/Dec" rule was tested directly against the live golden-cross
system and found net negative (broke SOL and TRX, meaningfully hurt BTC and
ETH; see the Nov/Dec seasonal exit test). The real pattern underneath that
failed rule is still true and still useful as context, just not as a
mechanical action:

  - In 7 of the last 13 years, BTC's Q1 max drawdown actually fell from a
    peak set in the PRIOR Q4 (Nov/Dec), not from a fresh Q1 high -- the
    correction started in Nov/Dec and continued into Q1 before recovering.
  - Q1 has the largest average intra-quarter drawdown of any quarter
    (-32.1% average, -27.3% median), despite decent median net returns --
    the two aren't contradictory, they're often the same cycle (crash in
    Nov/Dec, recovery within Q1 producing a net-positive quarter anyway).

So: Nov 1 through Feb 28/29 is a real, historically elevated-risk window --
useful to know about, especially if a position is already showing weakness
during it, but not a reason to mechanically exit a position that's still
otherwise healthy. The golden-cross exit rule (death cross) already proved
itself better at judging that than a calendar trigger does.
"""
from __future__ import annotations

from datetime import datetime, date

from trading_bot_v4.signals.garch_volatility import volatility_context

RISK_WINDOW_MONTHS = {11, 12, 1, 2}  # Nov, Dec, Jan, Feb


def in_seasonal_risk_window(when: datetime | date) -> bool:
    return when.month in RISK_WINDOW_MONTHS


def seasonal_context_note(when: datetime | date, position_pnl_pct: float | None) -> str | None:
    """Returns a context string to attach to an alert, or None if there's
    nothing worth surfacing right now. Only fires when it's both the real
    risk window AND the position is already showing weakness -- a healthy,
    still-rising position during Nov/Dec isn't flagged just for the date."""
    if not in_seasonal_risk_window(when):
        return None
    if position_pnl_pct is None or position_pnl_pct >= 0:
        return None
    vol_line = volatility_context(when)
    return (
        "📅 Seasonal risk context: currently in the Nov-Feb window. In 7 of the last 13 years, "
        "BTC's biggest Q1 drawdown actually started as a Nov/Dec decline that continued into Q1 "
        "before recovering. This position is already down -- historically this window is where "
        "corrections that started here either continue or become the recovery leg.\n"
        f"{vol_line}\n"
        "GARCH-confirmed: conditional volatility is significantly higher in this window than the "
        "rest of the year (t=8.57, p<0.0001) -- this is about elevated risk/magnitude, not direction. "
        "Informational only, not an exit signal -- the golden-cross rule itself already tested better "
        "than a calendar-based exit."
    )
