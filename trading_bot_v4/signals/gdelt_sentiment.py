"""Real point-in-time-verified sentiment loader, backed by the verified GDELT
news-tone pull (data/sentiment/gdelt_news_tone_daily_verified.csv).

Only covers the 17 symbols validated tonight (see that file) -- every name
in it was checked against real launch-date history and known common-word
collisions before being trusted. Callers for any other symbol should keep
using signals.sentiment_stub's explicit NotImplementedError rather than
silently getting nothing back.

Point-in-time discipline: GDELT's daily tone for day D is only fully known
once day D has finished (it's an aggregate over that whole day's articles).
Using day D's tone to inform a decision made *during* day D would leak
same-day information forward, so every reading is shifted one full day
before being exposed -- a decision made at any point on day D only ever
sees day D-1's tone. assert_no_lookahead is run on the output as a hard
check, not a formality.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot_v4.utils.point_in_time import assert_no_lookahead

GDELT_TONE_PATH = Path("data/sentiment/gdelt_news_tone_daily_verified.csv")
COVERED_SYMBOLS = {
    "BTC", "ETH", "LTC", "DOGE", "XMR", "ARB", "APE", "PEPE",
    "UNI", "WLD", "TON", "TAO", "ONDO", "EIGEN", "MKR", "FIL", "BNB",
}


def load_gdelt_daily_tone(symbol: str) -> pd.DataFrame:
    symbol = symbol.upper()
    if symbol not in COVERED_SYMBOLS:
        raise NotImplementedError(
            f"{symbol} was not part of tonight's verified GDELT pull ({sorted(COVERED_SYMBOLS)}). "
            f"Do not guess a search phrase and fabricate coverage -- validate a new symbol the same "
            f"way (dry-run cost check, anachronism check against real launch date) before trusting it."
        )
    frame = pd.read_csv(GDELT_TONE_PATH, dtype={"day": str})
    frame = frame[frame["symbol"] == symbol].copy()
    frame["date"] = pd.to_datetime(frame["day"], format="%Y%m%d", utc=True)
    return frame[["date", "avg_tone", "article_count"]].sort_values("date").reset_index(drop=True)


def load_sentiment_signal_frame(symbol: str, timeframe: str, as_of: pd.Timestamp | str) -> pd.DataFrame:
    """Returns a raw (timestamp, sentiment_score, article_count) frame at
    hourly resolution, forward-filled from the prior day's GDELT tone.
    Does not threshold into LONG/SHORT/HOLD -- that decision belongs to
    calibration (sweep threshold/polarity against real backtested
    outcomes), not a hardcoded guess, matching every other signal built
    tonight."""
    daily = load_gdelt_daily_tone(symbol)
    daily["date"] = daily["date"] + pd.Timedelta(days=1)  # shift: day D's tone becomes usable starting day D+1

    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    daily = daily[daily["date"] <= as_of_ts]

    if daily.empty:
        result = pd.DataFrame(columns=["timestamp", "sentiment_score", "article_count"])
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
        return result

    freq = "1h" if timeframe == "1h" else timeframe
    hourly_index = pd.date_range(daily["date"].min(), as_of_ts, freq=freq, tz="UTC")
    hourly = pd.DataFrame({"timestamp": hourly_index})
    hourly["_day"] = hourly["timestamp"].dt.floor("D")
    daily_indexed = daily.set_index(daily["date"].dt.floor("D"))
    hourly["sentiment_score"] = hourly["_day"].map(daily_indexed["avg_tone"])
    hourly["article_count"] = hourly["_day"].map(daily_indexed["article_count"]).fillna(0).astype(int)
    result = hourly.drop(columns=["_day"]).dropna(subset=["sentiment_score"]).reset_index(drop=True)

    assert_no_lookahead(result, "timestamp", as_of_ts, f"gdelt_sentiment/{symbol}")
    return result
