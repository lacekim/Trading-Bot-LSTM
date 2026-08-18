"""Different hypothesis: does sentiment predict volatility expansion,
rather than direction?

Everything tried tonight predicted *which way* price moves next, and
sentiment never moved the needle on that (AUC ~0.50-0.53 whether it's a
hand-coded gate or a learned feature). A genuinely different, well-motivated
question: does a spike in news coverage or sentiment extremity -- bullish
or bearish, doesn't matter which -- predict that price is about to move
*more* than usual? That's direction-agnostic, so it isn't just a rerun of
the same test with a new label.

Target: will the realized high-low range over the next 8 candles exceed
its own symbol-specific rolling median range (binary: volatility expansion
vs not). Features: the same 18 base + 3 relative-strength + 3 sentiment
columns already validated tonight. Model, split, and calibration discipline
unchanged. Runs across all 17 verified-sentiment symbols, not just BTC/ETH
-- also tests whether a smaller, less-arbitraged name shows something the
two most efficiently-traded assets in crypto wouldn't.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from trading_bot_v4.signals.gdelt_sentiment import COVERED_SYMBOLS, load_gdelt_daily_tone
from trading_bot_v4.utils.logger import build_logger
from tools.run_gbm_relative_strength_experiment import (
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
    _btc_return_frame,
    _new_model,
)
from train_model import DataHandler

logger = build_logger("v4_sentiment_volatility_experiment")

SENTIMENT_FEATURE_COLUMNS = ["sentiment_tone", "sentiment_article_count", "sentiment_tone_ma7"]
FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *SENTIMENT_FEATURE_COLUMNS]

FORWARD_WINDOW = 8
MIN_TOTAL_ROWS = 1500
TRAIN_END, CALIBRATION_END = 0.70, 0.85
MIN_SEGMENT_ROWS = 150


def _load_frame(symbol: str) -> pd.DataFrame | None:
    df = load_gmx_ohlc(symbol, "1h")
    if df is None or df.empty:
        return None
    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df = df.copy()
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr
    df = handler.prepare_features(df, prediction_horizon=1)
    if df.empty:
        return None
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})

    # BTC relative-strength features -- reuse the already-validated loader
    # instead of re-deriving it (a from-scratch reimplementation here hit a
    # dtype bug: load_gmx_ohlc's raw output isn't indexed the same way
    # prepare_features' output is, and skipping that step silently turned
    # the "timestamp" column into a row-number float instead of real dates).
    btc_frame = _btc_return_frame("1h")
    df = df.merge(btc_frame, on="timestamp", how="left")
    df[["btc_return_1h", "btc_return_24h"]] = df[["btc_return_1h", "btc_return_24h"]].ffill()
    df["btc_relative_return"] = df["returns"] - df["btc_return_1h"]
    df["btc_relative_return_ma20"] = df["btc_relative_return"].rolling(20).mean()

    # forward realized range, normalized by current ATR -- direction-agnostic
    high_fwd = df["High"].shift(-1).rolling(FORWARD_WINDOW).max().shift(-(FORWARD_WINDOW - 1))
    low_fwd = df["Low"].shift(-1).rolling(FORWARD_WINDOW).min().shift(-(FORWARD_WINDOW - 1))
    forward_range = (high_fwd - low_fwd) / df["ATR"].replace(0, np.nan)
    df["target"] = (forward_range > forward_range.rolling(500, min_periods=100).median()).astype(float)
    df.loc[forward_range.isna(), "target"] = np.nan

    # sentiment features, point-in-time shifted
    daily = load_gdelt_daily_tone(symbol)
    daily["date"] = daily["date"] + pd.Timedelta(days=1)
    daily["tone_ma7"] = daily["avg_tone"].rolling(7, min_periods=1).mean()
    daily = daily.rename(columns={"avg_tone": "sentiment_tone", "article_count": "sentiment_article_count",
                                   "tone_ma7": "sentiment_tone_ma7"})
    timestamp = df["timestamp"] if df["timestamp"].dt.tz is not None else df["timestamp"].dt.tz_localize("UTC")
    df["_day"] = timestamp.dt.floor("D")
    daily_indexed = daily.set_index(daily["date"].dt.floor("D"))
    for col in SENTIMENT_FEATURE_COLUMNS:
        df[col] = df["_day"].map(daily_indexed[col])
    df = df.drop(columns=["_day"])

    df = df.dropna(subset=[*FEATURE_COLUMNS, "target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    n = len(df)
    train_end = int(n * TRAIN_END)
    calibration_end = int(n * CALIBRATION_END)
    boundaries = (train_end, calibration_end - train_end, n - calibration_end)
    if any(size < MIN_SEGMENT_ROWS for size in boundaries):
        return None
    return df.iloc[:train_end], df.iloc[train_end:calibration_end], df.iloc[calibration_end:]


def evaluate(symbol: str) -> None:
    df = _load_frame(symbol)
    if df is None or len(df) < MIN_TOTAL_ROWS:
        logger.info(f"{symbol}: insufficient rows ({0 if df is None else len(df)}), skipping")
        return
    split = _split(df)
    if split is None:
        logger.info(f"{symbol}: insufficient rows per split segment, skipping")
        return
    train, calibration, holdout = split
    if train["target"].nunique() < 2 or holdout["target"].nunique() < 2:
        logger.info(f"{symbol}: target has no variation in train/holdout, skipping")
        return

    model = _new_model()
    model.fit(train[FEATURE_COLUMNS].to_numpy(np.float32), train["target"].to_numpy(int))

    calibration_proba = model.predict_proba(calibration[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    calibration_auc = roc_auc_score(calibration["target"], calibration_proba)

    holdout_proba = model.predict_proba(holdout[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    holdout_auc = roc_auc_score(holdout["target"], holdout_proba)

    logger.info(f"{symbol}: train_rows={len(train):>6} pos_rate={train['target'].mean():.2f} "
                f"calibration_auc={calibration_auc:.3f} holdout_auc={holdout_auc:.3f}")


if __name__ == "__main__":
    for symbol in sorted(COVERED_SYMBOLS):
        try:
            evaluate(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{symbol}: FAILED: {exc}")
