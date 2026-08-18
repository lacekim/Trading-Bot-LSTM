"""Point-in-time safety guard for any externally-sourced signal (sentiment,
news, or otherwise).

Motivated by a concrete failure tonight: a quick sentiment pilot sourced
"historical" news via live web search, and the results turned out to be
retrospective analysis pieces that already narrated outcomes ("Bitcoin
turned out to be the winner") rather than real point-in-time reporting --
an unfalsifiable test dressed up as a real one. Any future data loader for
an external signal must be able to prove it isn't doing the same thing
before its output is trusted for backtesting or trading.
"""

from __future__ import annotations

import pandas as pd


def assert_no_lookahead(frame: pd.DataFrame, timestamp_col: str, as_of: pd.Timestamp | str, source_name: str) -> None:
    """Raises ValueError if any row in `frame` is timestamped after `as_of`.

    Call this at the boundary of every external data loader (sentiment,
    news, or any future signal source) before its output is used for
    training, calibration, or live decisioning. A source that cannot pass
    this check for a given `as_of` is not safe to use for that decision,
    full stop -- there is no partial credit.
    """
    if frame.empty:
        return
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None and frame[timestamp_col].dt.tz is not None:
        cutoff = cutoff.tz_localize(frame[timestamp_col].dt.tz)
    violations = frame.loc[frame[timestamp_col] > cutoff]
    if not violations.empty:
        latest = violations[timestamp_col].max()
        raise ValueError(
            f"{source_name}: {len(violations)} row(s) dated after as_of={cutoff} "
            f"(latest violation: {latest}). This source leaks future information "
            f"and cannot be used for a decision made at {cutoff}."
        )
