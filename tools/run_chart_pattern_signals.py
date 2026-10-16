"""Chart-pattern / technical-indicator signal report, per the user's own
scoped request: of everything shown on that TradingView screenshot (SMA/EMA
combo, TD Sequential, chart-pattern auto-detector, Volatility Compression
Breakout, Bollinger Bands, Pi Cycle, Volume Profile), only the RULES-BASED
indicators were chosen for this pass -- TD Sequential, Bollinger Band
squeeze/volatility-compression breakout, and Pi Cycle (BTC only). Classic
chart shapes (head-and-shoulders, triangles, etc.) and Volume Profile were
explicitly deferred: both need their own separate design pass (subjective
shape-matching is genuinely hard to define without look-ahead bias or
overfitting; Volume Profile isn't a single trigger and needs its own
signal definition).

READ-ONLY, descriptive only, like every other diagnostic in this project.
NOT wired into any trading decision -- this reports each indicator's
current state per symbol so it can be observed across real data before any
decision about combining it with golden-cross / whale-alert entries or
exits, same "look before wiring in" discipline as the whale monitor.

Runs across the SAME universes every other diagnostic in this sequence
uses -- the 6 Kraken assets (ASSETS, imported from run_ma200_cross_strategy,
never redefined) on their native 4h bars, and the current equities universe
(robust==True, asset_class != "Crypto") on daily bars. Pi Cycle is BTC-only
and needs daily-granularity history specifically (the 111/350-*day* SMAs
are only meaningful on daily bars, not 4h), so it's fetched separately from
Kraken's daily OHLC endpoint rather than reusing the 4h series.

EQUITIES DATA SOURCE (changed 2026-09-30, per explicit request): daily bars
for the equities universe come from Alpaca's own Market Data API
(alpaca.data.historical.StockHistoricalDataClient), using the SAME
ALPACA_API_KEY/ALPACA_SECRET_KEY credentials tools/run_alpaca_paper_trader.py
already uses -- NOT Yahoo Finance, which every other equities diagnostic in
this project (including the live paper trader's own signal computation)
uses. This is a deliberate, disclosed deviation from that project-wide
convention, made because the user explicitly asked for the equities leg to
come from Alpaca specifically rather than Yahoo Finance. Ticker punctuation
differences are handled via the SAME EQUITY_SYMBOL_MAP already established
in run_alpaca_paper_trader.py (e.g. "BRK-B" -> "BRK.B"), imported from
there rather than redefined, so the two scripts can't silently drift apart
on this mapping. DISCLOSED, UNTESTED IN SANDBOX: this sandbox's egress
policy blocks pypi.org (confirmed via direct curl -- 403, "Host not in
allowlist"), so alpaca-py cannot be installed here and the actual network
call (fetch_equities_bars_alpaca) could not be dry-run tested against a
live response, same disclosed limitation as every other network-touching
function in this project (fetch_recent_trades, fetch_btc_daily_history,
the old fetch_ohlcv_batch). What WAS dry-run tested is the pure reshaping
function (_bars_df_to_price_dict) against a hand-built DataFrame matching
alpaca-py's documented .df shape (MultiIndex (symbol, timestamp), lowercase
open/high/low/close/volume/trade_count/vwap columns) -- run this once
locally and report back what actually happens (including any error) before
trusting the equities section of the output.

PRECISE, PREDECLARED DEFINITIONS (fixed before this was run against real
data -- read these before reading the output):

TD SEQUENTIAL -- SIMPLIFIED SETUP ONLY (disclosed simplification):
  This implements DeMark's "TD Setup" phase only -- NOT the full official
  9-13 Countdown with its "qualifier" bar-8/13 low-vs-close comparisons,
  which adds considerable complexity for a marginal precision gain. A
  bullish (Buy) setup count increments each bar where close < close 4 bars
  earlier, consecutively; a bearish (Sell) setup count increments each bar
  where close > close 4 bars earlier, consecutively. Either counter resets
  to 0 the moment its own condition fails, and can start incrementing the
  OTHER counter that same bar if that bar happens to also qualify for it.
  A count reaching 9 is reported as a completed Setup (bullish 9 =
  downtrend-exhaustion / potential-bottom signal; bearish 9 = uptrend-
  exhaustion / potential-top signal) and the counter resets to 0 on the
  next bar regardless of that bar's own qualification, matching the
  official convention of not double-counting past a completed Setup within
  this simplified, Countdown-free version.

BOLLINGER SQUEEZE / VOLATILITY-COMPRESSION BREAKOUT:
  20-bar SMA +/- 2 standard deviations (ddof=1, pandas' default) is the
  band. Bandwidth = (upper - lower) / middle. A bar is "squeezed" when its
  bandwidth is at or below the 10th percentile of the trailing 126 bars'
  own bandwidth history (126 ~ 6 months of daily bars; on 4h crypto bars
  this covers a shorter real-time span, same "not apples-to-apples across
  timeframes" caveat as the whale monitor's fixed trade-count window). A
  breakout is a close outside the band (above upper = bullish, below lower
  = bearish) occurring while squeezed OR within BREAKOUT_GRACE_BARS bars
  after a squeeze ended, so a breakout that fires right as the squeeze
  releases isn't missed by one bar's difference in timing.

PI CYCLE TOP (BTC ONLY, DAILY BARS):
  111-day SMA vs. 2x the 350-day SMA, on BTC daily closes (fetched
  separately at daily granularity for this reason). "Above" state = 111d
  SMA > 2x 350d SMA, historically associated with proximity to a cycle
  top. cross_up = the bar where state flips from below to above.
  Deliberately NOT applied to any asset besides BTC -- it is a BTC-cycle-
  specific heuristic with no established basis for other assets.

DISCLOSED LIMITATIONS:
  - No historical backtest of these signals' predictive value yet -- like
    the whale monitor, this is a first-pass observational report. Whether
    any of these three actually says something useful about forward
    returns for this universe is an open question, not assumed here.
  - The TD Setup simplification (no Countdown-13, no qualifiers) means
    this will disagree with a "true" DeMark TD Sequential count in some
    cases -- disclosed, not hidden.
  - Bollinger squeeze bandwidth-percentile approach is one legitimate
    definition of "squeeze," not the only one (e.g. John Carter's TTM
    Squeeze compares Bollinger Bands to Keltner Channels instead) --
    predeclared and fixed here rather than picked after looking at results.
  - The Alpaca equities fetch (fetch_equities_bars_alpaca) could NOT be
    dry-run tested against a real response in this sandbox -- pypi.org is
    blocked by this sandbox's own egress policy (confirmed via direct curl:
    403, "Host not in allowlist"), so alpaca-py itself couldn't even be
    installed here to inspect. Only the pure reshaping logic downstream of
    the actual API call was tested, against a hand-built mock matching
    alpaca-py's documented response shape. Run this locally and report back
    the actual output (or error) before trusting the equities section.

FIVE MORE INDICATORS (added 2026-09-30, per explicit request, off the
user's own real TradingView chart setup) -- MACD, RSI, Stochastic RSI,
Williams Vix Fix ("cm_williams"), and a volume-pressure oscillator
("bsvp_xel"). Every one of these was predeclared and confirmed via
AskUserQuestion BEFORE any code was written, same discipline as TD Setup /
Bollinger / Pi Cycle above:

MACD -- standard 12/26/9 EMA convention, nothing unusual:
  macd_line = EMA(close, 12) - EMA(close, 26)
  signal_line = EMA(macd_line, 9)
  histogram = macd_line - signal_line

RSI -- standard Wilder-smoothed 14-period RSI:
  gain = max(delta, 0), loss = max(-delta, 0), each smoothed with Wilder's
  EMA (alpha = 1/14, i.e. pandas .ewm(alpha=1/14, adjust=False)) -- NOT a
  plain SMA average, which is a common but technically-wrong shortcut.
  rs = avg_gain / avg_loss; rsi = 100 - 100/(1+rs). Edge cases (all-gain or
  perfectly flat windows, where this divides by zero) are handled
  explicitly: avg_loss==0 and avg_gain>0 -> RSI=100; both zero (flat
  price) -> RSI=50 (neutral), rather than leaving inf/NaN in the output.

STOCHASTIC RSI -- standard 14/14/3/3 convention (RSI period 14, Stoch
lookback 14, %K smoothed 3, %D smoothed 3, SMA smoothing not EMA, matching
the typical charting-platform default):
  stoch_k_raw = (rsi - rolling_min(rsi,14)) / (rolling_max(rsi,14) -
  rolling_min(rsi,14)) * 100 (defined as 50/neutral on a flat 14-bar RSI
  window, avoiding a divide-by-zero); stoch_k = SMA(stoch_k_raw, 3);
  stoch_d = SMA(stoch_k, 3).

WILLIAMS VIX FIX ("cm_williams") -- ChrisMoody's port of Larry Williams'
Vix Fix, formula confirmed via two independently-agreeing sources (a
GitHub Pine mirror and an AFL-conversion writeup):
  wvf = ((highest(close, 22) - low) / highest(close, 22)) * 100
  upper_band = SMA(wvf, 20) + 2.0 * STDEV(wvf, 20, ddof=1)
  range_high = highest(wvf, 50) * 0.85
  "spike" = wvf > upper_band OR wvf > range_high -- a volatility/
  capitulation spike, historically associated with market BOTTOMS (the
  opposite polarity from a Bollinger breakout, which is trend-following).

BSVP_XEL -- NOT a byte-exact port, disclosed and confirmed with the user.
  The real "BUY & SELL VOLUME PRESSURE by @XeL_Arjona" TradingView script
  builds on Vadim Gimelfarb's "Bull And Bear Balance" algorithm (Stocks &
  Commodities, Oct 2003) for its core buying/selling-pressure primitive --
  that primitive turned out to live only in copyrighted, non-public EFS
  code (an eSignal port), which I can't reproduce or safely reverse-
  engineer. Per explicit user confirmation, this SUBSTITUTES the
  well-documented, public-domain Elder Ray Bull Power / Bear Power in its
  place, combined with the confirmed volume-weighting / double-EMA-
  smoothing / convergence-divergence scaffolding (from a ProRealCode port
  of the sibling "...TO PRICE PRESSURE" script):
    ema_close = EMA(close, 13)
    bull_power = max(high - ema_close, 0); bear_power = max(ema_close -
    low, 0) -- clipped at zero (a one-sided "pressure" force, same
    clip(lower=0) pattern this file already uses for RSI's gain/loss;
    Elder Ray's raw Bull/Bear Power can go negative, which the original
    scaffolding's normalization doesn't expect)
    typical_price = (high + low + close) / 3
    bpv = (bull_power / typical_price) * volume; spv = (bear_power /
    typical_price) * volume (mirrors the confirmed BPV=(BP/TP)*volume
    shape)
    bpv_avg, spv_avg = double-EMA-smoothed (span 3, the confirmed
    "signal" period) versions of bpv/spv
    vpo1 = ((bpv_avg - spv_avg) / (bpv_avg + spv_avg)) * 100 (0 when both
    are zero, avoiding a divide-by-zero)
  A second, longer-period convergence term (vpo2, built from a "long"=21
  period per the sourced scaffolding) appears in the original script, but
  its exact construction wasn't clearly specified in the source I could
  access -- rather than reconstruct an unconfirmed second term, bsvp_xel
  here is vpo1 alone.

None of these five have a backtest long/flat rule defined in THIS file --
that's in run_chart_pattern_backtest.py, since RSI/Stoch RSI/Williams Vix
Fix are mean-reversion/exhaustion oscillators without an obvious
standalone trend rule (same design-fork problem Pi Cycle posed), and that
too was resolved via AskUserQuestion before writing any code.

Run it locally:
    python tools/run_chart_pattern_signals.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from tools.run_ma200_cross_strategy import ASSETS, load_full_history
from tools.run_stoploss_tail_risk_test_equities import _load_equities_universe
from tools.run_alpaca_paper_trader import EQUITY_SYMBOL_MAP, FUTURES_TO_ETF_PROXY

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
OUT_PATH = RUNTIME_DIR / "chart_pattern_signals_results.json"

# ---- predeclared parameters -------------------------------------------------
TD_SETUP_TARGET = 9
BOLLINGER_WINDOW = 20
BOLLINGER_NUM_STD = 2.0
SQUEEZE_LOOKBACK_BARS = 126
SQUEEZE_PERCENTILE = 10.0
BREAKOUT_GRACE_BARS = 3
PI_CYCLE_FAST_DAYS = 111
PI_CYCLE_SLOW_DAYS = 350
PI_CYCLE_SLOW_MULTIPLIER = 2.0
BTC_DAILY_PAIR = "XBTUSD"
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
REQUEST_TIMEOUT_SECONDS = 30
ALPACA_EQUITIES_LOOKBACK_DAYS = 730  # ~2y, matches run_alpaca_paper_trader.py's own _fetch_close_series period="2y"
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
STOCH_RSI_PERIOD = 14
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3
WVF_PERIOD = 22
WVF_BB_LENGTH = 20
WVF_BB_MULT = 2.0
WVF_RANGE_LOOKBACK = 50
WVF_RANGE_HIGH_PCT = 0.85
BSVP_EMA_PERIOD = 13  # Elder Ray substitute EMA period
BSVP_SMOOTH_PERIOD = 3  # double-EMA smoothing span, confirmed "signal" period
# ------------------------------------------------------------------------------


def compute_td_setup(closes: np.ndarray, target: int = TD_SETUP_TARGET) -> dict:
    """Simplified TD Setup counters -- see module docstring for the exact,
    disclosed simplification (no Countdown-13, no qualifiers). Returns
    per-bar bullish/bearish running counts and the bar indices where each
    counter reached `target` (a completed Setup)."""
    n = len(closes)
    bullish = np.zeros(n, dtype=int)
    bearish = np.zeros(n, dtype=int)
    bullish_complete: list[int] = []
    bearish_complete: list[int] = []
    for i in range(n):
        if i < 4:
            continue
        prior = closes[i - 4]
        cur = closes[i]
        # bullish leg
        if cur < prior:
            bullish[i] = bullish[i - 1] + 1
        else:
            bullish[i] = 0
        # bearish leg (independent condition, can start same bar bullish resets)
        if cur > prior:
            bearish[i] = bearish[i - 1] + 1
        else:
            bearish[i] = 0

        if bullish[i] >= target:
            bullish_complete.append(i)
            bullish[i] = 0
        if bearish[i] >= target:
            bearish_complete.append(i)
            bearish[i] = 0
    return {
        "bullish_count": bullish,
        "bearish_count": bearish,
        "bullish_setup_complete_bars": bullish_complete,
        "bearish_setup_complete_bars": bearish_complete,
    }


def compute_bollinger_squeeze_breakout(
    close: pd.Series,
    window: int = BOLLINGER_WINDOW,
    num_std: float = BOLLINGER_NUM_STD,
    squeeze_lookback: int = SQUEEZE_LOOKBACK_BARS,
    squeeze_percentile: float = SQUEEZE_PERCENTILE,
    breakout_grace_bars: int = BREAKOUT_GRACE_BARS,
) -> pd.DataFrame:
    """Pure pandas computation -- see module docstring for the exact,
    predeclared squeeze/breakout definitions."""
    middle = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=1)
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle

    threshold = bandwidth.rolling(squeeze_lookback).quantile(squeeze_percentile / 100.0)
    squeeze = bandwidth <= threshold

    last_squeeze_pos = None
    bars_since_arr = np.full(len(close), np.nan)
    squeeze_arr = squeeze.to_numpy()
    for i in range(len(close)):
        if squeeze_arr[i]:
            last_squeeze_pos = i
        if last_squeeze_pos is not None:
            bars_since_arr[i] = i - last_squeeze_pos
    bars_since_squeeze = pd.Series(bars_since_arr, index=close.index)

    eligible = squeeze | (bars_since_squeeze <= breakout_grace_bars)
    breakout_up = eligible & (close > upper)
    breakout_down = eligible & (close < lower)

    return pd.DataFrame({
        "middle": middle, "upper": upper, "lower": lower, "bandwidth": bandwidth,
        "squeeze": squeeze, "breakout_up": breakout_up, "breakout_down": breakout_down,
    })


def compute_pi_cycle(
    close: pd.Series,
    fast_days: int = PI_CYCLE_FAST_DAYS,
    slow_days: int = PI_CYCLE_SLOW_DAYS,
    slow_multiplier: float = PI_CYCLE_SLOW_MULTIPLIER,
) -> pd.DataFrame:
    """Pure pandas computation -- BTC-only, daily-bar Pi Cycle Top."""
    sma_fast = close.rolling(fast_days).mean()
    sma_slow_x = close.rolling(slow_days).mean() * slow_multiplier
    above = sma_fast > sma_slow_x
    cross_up = above & ~above.shift(1, fill_value=False)
    return pd.DataFrame({
        "sma_fast": sma_fast, "sma_slow_x": sma_slow_x,
        "above": above, "cross_up": cross_up,
    })


def compute_macd(
    close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """Pure pandas computation -- standard 12/26/9 EMA MACD, see module
    docstring for the exact, predeclared formula."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram})


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Pure pandas computation -- standard Wilder-smoothed RSI, see module
    docstring for the exact, predeclared formula and edge-case handling."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def compute_stoch_rsi(
    close: pd.Series,
    rsi_period: int = RSI_PERIOD,
    stoch_period: int = STOCH_RSI_PERIOD,
    k_smooth: int = STOCH_K_SMOOTH,
    d_smooth: int = STOCH_D_SMOOTH,
) -> pd.DataFrame:
    """Pure pandas computation -- standard 14/14/3/3 Stochastic RSI (SMA
    smoothing on %K/%D, matching the typical charting-platform default),
    see module docstring for the exact, predeclared formula."""
    rsi = compute_rsi(close, rsi_period)
    lowest = rsi.rolling(stoch_period).min()
    highest = rsi.rolling(stoch_period).max()
    denom = highest - lowest
    stoch_k_raw = ((rsi - lowest) / denom) * 100
    stoch_k_raw = stoch_k_raw.mask(denom == 0, 50.0)
    stoch_k = stoch_k_raw.rolling(k_smooth).mean()
    stoch_d = stoch_k.rolling(d_smooth).mean()
    return pd.DataFrame({"rsi": rsi, "stoch_k_raw": stoch_k_raw, "stoch_k": stoch_k, "stoch_d": stoch_d})


def compute_williams_vix_fix(
    close: pd.Series,
    low: pd.Series,
    period: int = WVF_PERIOD,
    bb_length: int = WVF_BB_LENGTH,
    bb_mult: float = WVF_BB_MULT,
    range_lookback: int = WVF_RANGE_LOOKBACK,
    range_high_pct: float = WVF_RANGE_HIGH_PCT,
) -> pd.DataFrame:
    """Pure pandas computation -- ChrisMoody's Williams Vix Fix
    ("cm_williams"), see module docstring for the exact, sourced formula.
    "spike" flags a volatility/capitulation event (historically a
    contrarian, potential-bottom signal -- opposite polarity from a
    Bollinger breakout)."""
    highest_close = close.rolling(period).max()
    wvf = ((highest_close - low) / highest_close) * 100
    mid = wvf.rolling(bb_length).mean()
    sd = wvf.rolling(bb_length).std(ddof=1) * bb_mult
    upper_band = mid + sd
    range_high = wvf.rolling(range_lookback).max() * range_high_pct
    spike = (wvf > upper_band) | (wvf > range_high)
    return pd.DataFrame({"wvf": wvf, "upper_band": upper_band, "range_high": range_high, "spike": spike})


def compute_bsvp_xel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    ema_period: int = BSVP_EMA_PERIOD,
    smooth_period: int = BSVP_SMOOTH_PERIOD,
) -> pd.DataFrame:
    """Pure pandas computation -- "bsvp_xel", an Elder-Ray-based
    APPROXIMATION of "BUY & SELL VOLUME PRESSURE by @XeL_Arjona", NOT a
    byte-exact port -- see module docstring for exactly why and what was
    substituted, confirmed with the user before this was written."""
    ema_close = close.ewm(span=ema_period, adjust=False).mean()
    bull_power = (high - ema_close).clip(lower=0)
    bear_power = (ema_close - low).clip(lower=0)
    typical_price = (high + low + close) / 3.0
    bpv = (bull_power / typical_price) * volume
    spv = (bear_power / typical_price) * volume
    bpv_avg = bpv.ewm(span=smooth_period, adjust=False).mean().ewm(span=smooth_period, adjust=False).mean()
    spv_avg = spv.ewm(span=smooth_period, adjust=False).mean().ewm(span=smooth_period, adjust=False).mean()
    total_avg = bpv_avg + spv_avg
    vpo1 = ((bpv_avg - spv_avg) / total_avg) * 100
    vpo1 = vpo1.mask(total_avg == 0, 0.0)
    return pd.DataFrame({
        "bull_power": bull_power, "bear_power": bear_power,
        "bpv": bpv, "spv": spv, "bpv_avg": bpv_avg, "spv_avg": spv_avg,
        "vpo1": vpo1,
    })


def fetch_btc_daily_history(pair: str = BTC_DAILY_PAIR) -> pd.Series:
    """Thin, isolated Kraken-API glue for BTC DAILY closes specifically --
    Pi Cycle needs daily granularity, unlike the 4h bars every other
    diagnostic in this project uses for crypto. Kraken's OHLC endpoint
    caps each response at ~720 bars, so this paginates with `since` the
    same way run_ma200_cross_strategy.fetch_fresh does for 4h bars."""
    all_rows: list = []
    since = 0
    for _ in range(10):
        url = f"{KRAKEN_OHLC_URL}?pair={pair}&interval=1440&since={since}"
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
        keys = [k for k in data["result"].keys() if k != "last"]
        if not keys:
            break
        rows = data["result"][keys[0]]
        if not rows:
            break
        all_rows.extend(rows)
        new_since = int(data["result"]["last"])
        if new_since <= since:
            break
        since = new_since
        time.sleep(1)
    if not all_rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"]).drop_duplicates("ts")
    df["dt"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    df["close"] = df["close"].astype(float)
    return df.set_index("dt")["close"].sort_index()


def _bars_df_to_price_dict(bars_df: pd.DataFrame, alpaca_to_original: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Pure reshaping -- no network. Takes alpaca-py's own documented
    BarSet.df shape (MultiIndex (symbol, timestamp), lowercase
    open/high/low/close/volume/trade_count/vwap columns) and returns
    {original_universe_symbol: DataFrame with a "Close" column}, matching
    what _current_state_row expects (df["Close"]) so nothing downstream of
    this needed to change. alpaca_to_original maps the Alpaca-punctuation
    ticker (e.g. "BRK.B") back to this project's own canonical symbol (e.g.
    "BRK-B", the key _load_equities_universe returns) so results stay keyed
    the same way every other diagnostic in this project keys them.

    Also carries High/Low/Volume now (added alongside the 5 new
    indicators -- Williams Vix Fix needs High/Low, bsvp_xel needs
    High/Low/Volume too), not just Close as before."""
    out: dict[str, pd.DataFrame] = {}
    if bars_df is None or bars_df.empty:
        return out
    for alpaca_symbol, group in bars_df.groupby(level=0):
        original_symbol = alpaca_to_original.get(alpaca_symbol, alpaca_symbol)
        g = group.droplevel(0).sort_index()
        out[original_symbol] = pd.DataFrame({
            "Close": g["close"].astype(float),
            "High": g["high"].astype(float),
            "Low": g["low"].astype(float),
            "Volume": g["volume"].astype(float),
        })
    return out


def _parse_invalid_symbol(error_text: str) -> str | None:
    """Pure string parsing -- no network. Alpaca's Data API returns an
    error shaped like {"message": "invalid symbol: GC=F"} when even ONE
    requested symbol in a batch is unrecognized, and confirmed LIVE (first
    real run against this project's actual 60-symbol universe) that this
    fails the ENTIRE batch request -- 0 of 60 symbols came back, not 59 of
    60 -- unlike the old yfinance-based fetch, which tolerated a bad ticker
    per-symbol without failing the whole batch. This extracts the bad
    symbol out of that error message so the caller can drop it and retry,
    restoring that same per-symbol tolerance rather than one bad ticker
    (today's known case: GC=F/PA=F raw futures Alpaca doesn't list at all,
    already handled via FUTURES_TO_ETF_PROXY below -- but ANY symbol could
    go bad this same way later: delisted, renamed, etc.) taking down
    results for every other symbol in the request."""
    match = re.search(r'invalid symbol:\s*([^\s"\'}]+)', error_text)
    return match.group(1) if match else None


def fetch_equities_bars_alpaca(symbols: list[str], lookback_days: int = ALPACA_EQUITIES_LOOKBACK_DAYS) -> dict[str, pd.DataFrame]:
    """Thin, isolated Alpaca Market Data glue -- see module docstring for
    why this replaced Yahoo Finance for the equities leg specifically, and
    for the disclosed "could not be tested against a live response in this
    sandbox" limitation. Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in the
    environment, same requirement tools/run_alpaca_paper_trader.py already
    has and fails the same explicit way if they're missing -- this can't
    generate brokerage credentials any more than that script can.

    Symbol mapping mirrors run_alpaca_paper_trader.py's own precedence
    exactly (FUTURES_TO_ETF_PROXY, then EQUITY_SYMBOL_MAP, both imported
    from there rather than redefined) -- CRYPTO_SYMBOL_MAP is deliberately
    NOT applied here, since _load_equities_universe() already excludes
    asset_class == "Crypto" by construction, so it would only ever be a
    no-op for this universe."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set.\n"
            "Get free paper-trading keys at "
            "https://app.alpaca.markets/paper/dashboard/overview\n"
            "then add to .env:\n"
            "  ALPACA_API_KEY=...\n"
            "  ALPACA_SECRET_KEY=...\n"
        )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    alpaca_to_original = {}
    alpaca_symbols = []
    for sym in symbols:
        mapped = FUTURES_TO_ETF_PROXY.get(sym) or EQUITY_SYMBOL_MAP.get(sym) or sym
        alpaca_to_original[mapped] = sym
        alpaca_symbols.append(mapped)

    client = StockHistoricalDataClient(api_key, secret_key)
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    remaining = list(alpaca_symbols)
    skipped: list[str] = []
    bars_df = pd.DataFrame()
    for _ in range(len(remaining) + 1):
        if not remaining:
            break
        try:
            request = StockBarsRequest(symbol_or_symbols=remaining, timeframe=TimeFrame.Day, start=start)
            bars = client.get_stock_bars(request)
            bars_df = bars.df
            break
        except Exception as exc:
            bad = _parse_invalid_symbol(str(exc))
            if bad and bad in remaining:
                skipped.append(bad)
                remaining.remove(bad)
                continue
            raise
    if skipped:
        print(f"  Alpaca rejected {len(skipped)} symbol(s) as invalid, excluded and retried: {skipped}")

    return _bars_df_to_price_dict(bars_df, alpaca_to_original)


def _current_state_row(
    symbol: str,
    close: pd.Series,
    include_pi_cycle: bool = False,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
) -> dict:
    """high/low/volume are optional: MACD/RSI/Stoch RSI only need close and
    are always reported; Williams Vix Fix needs low (skipped without it);
    bsvp_xel needs high/low/volume (skipped without them) -- e.g. the BTC
    daily Pi Cycle row only ever has a close series, so it gets the
    close-only indicators and not the other two."""
    closes = close.to_numpy(dtype=float)
    td = compute_td_setup(closes)
    boll = compute_bollinger_squeeze_breakout(close)
    row = {
        "symbol": symbol,
        "n_bars": len(close),
        "last_date": str(close.index[-1]) if len(close) else None,
        "td_bullish_count": int(td["bullish_count"][-1]) if len(closes) else None,
        "td_bearish_count": int(td["bearish_count"][-1]) if len(closes) else None,
        "bollinger_squeeze": bool(boll["squeeze"].iloc[-1]) if len(boll) else None,
        "bollinger_breakout_up": bool(boll["breakout_up"].iloc[-1]) if len(boll) else None,
        "bollinger_breakout_down": bool(boll["breakout_down"].iloc[-1]) if len(boll) else None,
        "bollinger_bandwidth": float(boll["bandwidth"].iloc[-1]) if len(boll) and not pd.isna(boll["bandwidth"].iloc[-1]) else None,
    }
    if include_pi_cycle:
        pc = compute_pi_cycle(close)
        row["pi_cycle_above"] = bool(pc["above"].iloc[-1]) if len(pc) else None
        row["pi_cycle_cross_up"] = bool(pc["cross_up"].iloc[-1]) if len(pc) else None

    if len(close):
        macd = compute_macd(close)
        row["macd_line"] = float(macd["macd_line"].iloc[-1]) if not pd.isna(macd["macd_line"].iloc[-1]) else None
        row["macd_signal_line"] = float(macd["signal_line"].iloc[-1]) if not pd.isna(macd["signal_line"].iloc[-1]) else None
        row["macd_histogram"] = float(macd["histogram"].iloc[-1]) if not pd.isna(macd["histogram"].iloc[-1]) else None

        rsi = compute_rsi(close)
        row["rsi"] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

        stoch = compute_stoch_rsi(close)
        row["stoch_rsi_k"] = float(stoch["stoch_k"].iloc[-1]) if not pd.isna(stoch["stoch_k"].iloc[-1]) else None
        row["stoch_rsi_d"] = float(stoch["stoch_d"].iloc[-1]) if not pd.isna(stoch["stoch_d"].iloc[-1]) else None
    else:
        row["macd_line"] = row["macd_signal_line"] = row["macd_histogram"] = None
        row["rsi"] = row["stoch_rsi_k"] = row["stoch_rsi_d"] = None

    if low is not None and len(close):
        wvf = compute_williams_vix_fix(close, low)
        row["williams_vix_fix"] = float(wvf["wvf"].iloc[-1]) if not pd.isna(wvf["wvf"].iloc[-1]) else None
        row["williams_vix_fix_spike"] = bool(wvf["spike"].iloc[-1]) if len(wvf) else None
    else:
        row["williams_vix_fix"] = row["williams_vix_fix_spike"] = None

    if high is not None and low is not None and volume is not None and len(close):
        bsvp = compute_bsvp_xel(high, low, close, volume)
        row["bsvp_xel_vpo1"] = float(bsvp["vpo1"].iloc[-1]) if not pd.isna(bsvp["vpo1"].iloc[-1]) else None
    else:
        row["bsvp_xel_vpo1"] = None

    return row


def main() -> None:
    print("=" * 100)
    print("CHART-PATTERN / INDICATOR SIGNAL REPORT -- rules-based indicators only")
    print("(TD Sequential simplified Setup, Bollinger squeeze/breakout, Pi Cycle for BTC)")
    print("READ-ONLY, descriptive only -- nothing here is wired into any trading decision yet.")
    print("=" * 100)

    results = {"crypto": {}, "equities": {}}

    print("\n-- crypto (4h bars) --")
    for label, (pair, path) in ASSETS.items():
        try:
            df = load_full_history(pair, path)
        except Exception as exc:
            print(f"  {label:<6}: FETCH FAILED -- {exc}")
            results["crypto"][label] = {"error": str(exc)}
            continue
        row = _current_state_row(label, df["close"], high=df["high"], low=df["low"], volume=df["volume"])
        results["crypto"][label] = row
        print(f"  {label:<6} n={row['n_bars']:>5}  TD_bull={row['td_bullish_count']:>2} TD_bear={row['td_bearish_count']:>2}  "
              f"squeeze={str(row['bollinger_squeeze']):<5} brk_up={str(row['bollinger_breakout_up']):<5} brk_dn={row['bollinger_breakout_down']}")
        print(f"          RSI={row['rsi']:.1f}  StochRSI_K={row['stoch_rsi_k']:.1f}  MACD_hist={row['macd_histogram']:.4f}  "
              f"WVF={row['williams_vix_fix']:.2f} spike={row['williams_vix_fix_spike']}  bsvp_xel_vpo1={row['bsvp_xel_vpo1']:.2f}"
              if all(row[k] is not None for k in ("rsi", "stoch_rsi_k", "macd_histogram", "williams_vix_fix", "bsvp_xel_vpo1"))
              else "          (new indicators still warming up -- not enough bars yet)")

    print("\n-- BTC Pi Cycle Top (daily bars, BTC only) --")
    try:
        btc_daily = fetch_btc_daily_history()
        pc_row = _current_state_row("BTC_daily", btc_daily, include_pi_cycle=True)
        results["crypto"]["BTC_pi_cycle_daily"] = pc_row
        print(f"  n_days={pc_row['n_bars']}  pi_cycle_above={pc_row.get('pi_cycle_above')}  "
              f"pi_cycle_cross_up={pc_row.get('pi_cycle_cross_up')}")
    except Exception as exc:
        print(f"  FETCH FAILED -- {exc}")
        results["crypto"]["BTC_pi_cycle_daily"] = {"error": str(exc)}

    print("\n-- equities (daily bars, via Alpaca Market Data API) --")
    symbols = _load_equities_universe()
    try:
        price = fetch_equities_bars_alpaca(symbols)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"  FETCH FAILED -- {exc}")
        price = {}
    print(f"usable OHLCV history for {len(price)} of {len(symbols)} instruments\n")
    for sym, df in price.items():
        row = _current_state_row(sym, df["Close"], high=df["High"], low=df["Low"], volume=df["Volume"])
        results["equities"][sym] = row
        print(f"  {sym:<6} n={row['n_bars']:>5}  TD_bull={row['td_bullish_count']:>2} TD_bear={row['td_bearish_count']:>2}  "
              f"squeeze={str(row['bollinger_squeeze']):<5} brk_up={str(row['bollinger_breakout_up']):<5} brk_dn={row['bollinger_breakout_down']}")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "td_setup_target": TD_SETUP_TARGET,
        "bollinger_window": BOLLINGER_WINDOW,
        "bollinger_num_std": BOLLINGER_NUM_STD,
        "squeeze_lookback_bars": SQUEEZE_LOOKBACK_BARS,
        "squeeze_percentile": SQUEEZE_PERCENTILE,
        "breakout_grace_bars": BREAKOUT_GRACE_BARS,
        "pi_cycle_fast_days": PI_CYCLE_FAST_DAYS,
        "pi_cycle_slow_days": PI_CYCLE_SLOW_DAYS,
        "results": results,
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    print("\nDescriptive only -- no rule or threshold is being proposed here. Nothing in the live")
    print("golden-cross system reads this output yet.")


if __name__ == "__main__":
    main()
