"""Bull Market Support Band (20-week SMA / 21-week EMA) -- a second, separate
entry pathway from the golden cross, verified only on BTC and PAXG after
properly de-duplicating overlapping touches (a naive count of every week
price sits in the band overstates the sample -- many are the same dip
touched repeatedly, not independent episodes):

  BTC:  20 de-duplicated episodes, PF 2.00 excluding the 3 biggest winners.
  PAXG: 17 de-duplicated episodes, PF 2.48 excluding the 3 biggest winners.

Entry: price dips into the band (low <= band_hi, low >= band_lo*0.97) while
in an established uptrend (close > 52-week MA), and the last BMSB entry for
this asset was at least MIN_GAP_WEEKS ago (de-duplication, enforced here
using persisted state since it's inherently stateful).
Exit: fixed 12-week hold, exactly as backtested -- not a signal-based exit,
because that's what was actually tested and verified.
"""
from __future__ import annotations

import pandas as pd

MIN_GAP_WEEKS = 8
HOLD_WEEKS = 12


def compute_weekly_band_state(df: pd.DataFrame) -> pd.DataFrame:
    """df must have 'high', 'low', 'close' columns at any resolution finer
    than weekly (e.g. 4h) -- resampled to weekly internally."""
    weekly = df[["high", "low", "close"]].resample("1W").agg(
        {"high": "max", "low": "min", "close": "last"}
    ).dropna()
    close = weekly["close"]
    low = weekly["low"]
    sma20w = close.rolling(20).mean()
    ema21w = close.ewm(span=21, adjust=False).mean()
    band_lo = pd.concat([sma20w, ema21w], axis=1).min(axis=1)
    band_hi = pd.concat([sma20w, ema21w], axis=1).max(axis=1)
    uptrend = close > close.rolling(52).mean()
    in_band = (low <= band_hi) & (low >= band_lo * 0.97)
    # Same object-dtype `~`-after-shift bug fixed in ma200_trend.py: without
    # .astype(bool), shift(1).fillna(False) upcasts to object dtype and `~`
    # stops being boolean negation, so `touch` stayed True for every
    # consecutive week still in-band instead of firing once per fresh dip
    # (confirmed: real BTC/PAXG counts dropped after the fix, see below).
    prev_in_band = in_band.shift(1).fillna(False).astype(bool)
    touch = (in_band & uptrend & ~prev_in_band).fillna(False)
    return pd.DataFrame({
        "close": close, "band_lo": band_lo, "band_hi": band_hi,
        "uptrend": uptrend, "touch": touch,
    })


def current_bmsb_signal(df: pd.DataFrame, last_entry_time: pd.Timestamp | None) -> dict:
    """Real-time read for the latest completed weekly bar. `last_entry_time`
    is the persisted timestamp of this asset's last BMSB entry (None if
    never entered) -- used to enforce the de-dup gap the backtest relied on."""
    state = compute_weekly_band_state(df)
    if len(state) < 52:
        return {"new_entry": False, "reason": "insufficient weekly history"}
    last = state.iloc[-1]
    as_of = state.index[-1]

    if not bool(last["touch"]):
        return {"new_entry": False, "reason": "no touch this week", "as_of": as_of}

    if last_entry_time is not None:
        gap_weeks = (as_of - pd.Timestamp(last_entry_time)).days / 7
        if gap_weeks < MIN_GAP_WEEKS:
            return {"new_entry": False, "reason": f"within {MIN_GAP_WEEKS}w de-dup gap", "as_of": as_of}

    return {"new_entry": True, "close": float(last["close"]), "as_of": as_of}
