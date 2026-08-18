"""Combine an existing price-based signal frame with an independent
confirming signal (e.g. sentiment), requiring both to agree before a trade
fires.

Motivated by two things found tonight: every price-only signal tested
(binary threshold, triple-barrier, multiple horizons, multiple regimes)
showed no real edge on its own, and the user's own described trading
process never acts on a chart signal without independent confirmation.
Structurally, this has never been tested -- every experiment tonight
evaluated a single signal in isolation. This module makes confluence a
pluggable, testable structure rather than a one-off script.

Both input frames use the same schema `simulate_production_symbol` already
expects: a `timestamp` column and a `model_direction` column valued
"LONG"/"SHORT"/"HOLD". That keeps a confluence-gated signal droppable
straight into the same real-execution backtest engine used for every other
experiment tonight -- no new evaluation path, no new promotion criteria.
"""

from __future__ import annotations

import pandas as pd


def combine_signal_frames(
    price_frame: pd.DataFrame,
    confirming_frame: pd.DataFrame | None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Downgrades price_frame's model_direction to HOLD wherever the
    confirming signal disagrees or is missing for that timestamp.

    confirming_frame=None means "no confirming signal wired up yet" --
    returns price_frame completely unchanged, so callers can tell from the
    return value alone whether confluence was actually applied, rather than
    silently behaving as if an absent signal is a neutral vote.
    """
    if confirming_frame is None:
        return price_frame.copy()

    price_frame = price_frame.copy()
    confirming_frame = confirming_frame[[timestamp_col, "model_direction"]].copy()
    # Merge keys must share tz-awareness or pandas refuses to merge -- both
    # sides get coerced to UTC-aware rather than assuming either side's
    # convention is the "right" one.
    for frame in (price_frame, confirming_frame):
        if frame[timestamp_col].dt.tz is None:
            frame[timestamp_col] = frame[timestamp_col].dt.tz_localize("UTC")
        else:
            frame[timestamp_col] = frame[timestamp_col].dt.tz_convert("UTC")

    merged = price_frame.merge(
        confirming_frame.rename(columns={"model_direction": "_confirming_direction"}),
        on=timestamp_col,
        how="left",
    )
    # A timestamp with no confirming row at all is treated the same as an
    # explicit disagreement -- confluence requires a real, present second
    # opinion, not the absence of one.
    merged["_confirming_direction"] = merged["_confirming_direction"].fillna("HOLD")

    agrees = merged["model_direction"] == merged["_confirming_direction"]
    merged["model_direction"] = merged["model_direction"].where(agrees, "HOLD")
    return merged.drop(columns=["_confirming_direction"])
