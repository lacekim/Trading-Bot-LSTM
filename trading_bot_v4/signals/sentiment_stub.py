"""Plug point for a real, point-in-time-verified sentiment signal.

Deliberately unimplemented. There is no affordable, sufficiently-historical
sentiment data source wired up yet -- raw Twitter/Reddit historical access
is cost-prohibitive, GDELT's convenient API only covers a rolling ~1 year,
and the BigQuery/LunarCrush path is still an open decision. A stub that
silently returned neutral or fabricated data would let `combine_signal_frames`
run without anyone noticing confluence was never actually applied. Raising
here instead makes that failure loud rather than silent.

Once a real source is wired up, its loader must call
`trading_bot_v4.utils.point_in_time.assert_no_lookahead` on its own output
before returning it -- see that module's docstring for why.
"""

from __future__ import annotations

import pandas as pd


def load_sentiment_signal_frame(symbol: str, timeframe: str, as_of: pd.Timestamp | str) -> pd.DataFrame:
    raise NotImplementedError(
        f"No point-in-time-verified sentiment source is wired up yet "
        f"(requested {symbol}/{timeframe} as of {as_of}). Implement this against "
        f"a real historical data source and call assert_no_lookahead on its output "
        f"before returning -- do not stub in placeholder or fabricated sentiment data."
    )
