"""Sentiment as a learned feature, not a hand-coded gate.

The confluence test (run_sentiment_confluence_validation.py) imposed a rule
I designed by hand: sentiment must independently agree with price before a
trade fires, with the agree/disagree threshold picked by a coarse sweep.
That's not actually letting the model learn the relationship between
sentiment and outcome -- it's a heuristic bolted on after the fact. A
network's actual job is to learn the mapping from inputs to outputs by
example; if sentiment carries real information, the right way to find that
out is to hand it to the model as a feature alongside price/technical
features and let training figure out how (and how much) to weight it --
not to pre-decide a gating rule myself.

Same target (triple-barrier), same model (per-asset GBM), same calibration/
walk-forward discipline as every other experiment tonight. Only the feature
set changes: FEATURE_COLUMNS plus sentiment_tone, sentiment_article_count,
and a 7-day rolling mean of tone (captures trend, not just level). Covers
only BTC/ETH -- the two symbols with both a real price model and verified,
well-populated GDELT sentiment history.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trading_bot_v4.ml.per_asset_trainer import (
    MIN_CALIBRATION_TRADES,
    MIN_WINDOW_TRADES,
    _ENDPOINT_COLUMNS,
    _select_threshold,
    _simulate_threshold,
    _walk_forward_rows,
)
from trading_bot_v4.signals.gdelt_sentiment import load_gdelt_daily_tone
from trading_bot_v4.utils.logger import build_logger
from tools.run_gbm_relative_strength_experiment import (
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
    _chronological_split,
    _load_symbol_frame as _load_price_frame,
    _new_model,
)

logger = build_logger("v4_sentiment_feature_experiment")

SYMBOLS = ["BTC", "ETH"]
DIRECTIONS = ["long", "short"]
SENTIMENT_FEATURE_COLUMNS = ["sentiment_tone", "sentiment_article_count", "sentiment_tone_ma7"]
FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *SENTIMENT_FEATURE_COLUMNS]


def _load_symbol_frame_with_sentiment(symbol: str, direction: str) -> pd.DataFrame | None:
    df = _load_price_frame(symbol, "1h", direction)
    if df is None:
        return None

    daily = load_gdelt_daily_tone(symbol)
    daily["date"] = daily["date"] + pd.Timedelta(days=1)  # point-in-time: day D's tone usable from day D+1
    daily["tone_ma7"] = daily["avg_tone"].rolling(7, min_periods=1).mean()
    daily = daily.rename(columns={"avg_tone": "sentiment_tone", "article_count": "sentiment_article_count",
                                   "tone_ma7": "sentiment_tone_ma7"})

    timestamp = df["timestamp"] if df["timestamp"].dt.tz is not None else df["timestamp"].dt.tz_localize("UTC")
    df["_day"] = timestamp.dt.floor("D")
    daily_indexed = daily.set_index(daily["date"].dt.floor("D"))
    for col in SENTIMENT_FEATURE_COLUMNS:
        df[col] = df["_day"].map(daily_indexed[col])
    df = df.drop(columns=["_day"]).dropna(subset=SENTIMENT_FEATURE_COLUMNS).reset_index(drop=True)
    return df


def train_and_evaluate(symbol: str, direction: str) -> None:
    df = _load_symbol_frame_with_sentiment(symbol, direction)
    if df is None or len(df) < 1500:
        logger.info(f"{symbol}/{direction}: insufficient sentiment-joined rows ({0 if df is None else len(df)}), skipping")
        return

    split = _chronological_split(df)
    if split is None:
        logger.info(f"{symbol}/{direction}: insufficient rows per split segment, skipping")
        return
    train, calibration, holdout = split

    model = _new_model()
    model.fit(train[FEATURE_COLUMNS].to_numpy(np.float32), train["target"].to_numpy(int))

    calibration_probability = model.predict_proba(calibration[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    calibration_frame = calibration[_ENDPOINT_COLUMNS].reset_index(drop=True)
    threshold, _ = _select_threshold(calibration_frame, calibration_probability, direction, symbol)

    holdout_probability = model.predict_proba(holdout[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    holdout_frame = holdout[_ENDPOINT_COLUMNS].reset_index(drop=True)
    holdout_target = holdout_frame["target"].to_numpy(int)
    holdout_auc = (
        float(roc_auc_score(holdout_target, holdout_probability))
        if len(np.unique(holdout_target)) > 1 else float("nan")
    )
    holdout_summary = _simulate_threshold(holdout_frame, holdout_probability, threshold, direction, symbol)
    windows = _walk_forward_rows(holdout_frame, holdout_probability, threshold, direction, symbol)

    promoted = bool(
        holdout_summary["trades"] >= MIN_CALIBRATION_TRADES
        and all(row["trades"] >= MIN_WINDOW_TRADES and row["profit_factor"] >= 1.30 and row["return_pct"] > 0
                for row in windows)
    )

    # feature importance -- did the model actually use the sentiment columns, or ignore them?
    importances = dict(zip(FEATURE_COLUMNS, getattr(model, "feature_importances_", [None] * len(FEATURE_COLUMNS))))
    sentiment_importance = {col: importances.get(col) for col in SENTIMENT_FEATURE_COLUMNS}

    logger.info(f"{symbol}/{direction}: train_rows={len(train)} holdout_auc={holdout_auc:.3f} "
                f"threshold={threshold:.2f} trades={holdout_summary['trades']} "
                f"return_pct={holdout_summary['return_pct']:.2f} profit_factor={holdout_summary['profit_factor']:.2f}")
    for row in windows:
        logger.info(f"{symbol}/{direction}:   window {row['window']} -> trades={row['trades']:>3} "
                    f"return_pct={row['return_pct']:>7.2f} profit_factor={row['profit_factor']:.2f}")
    logger.info(f"{symbol}/{direction}: sentiment feature importances -> {sentiment_importance}")
    logger.info(f"{symbol}/{direction}: PROMOTED={promoted}")


if __name__ == "__main__":
    for symbol in SYMBOLS:
        for direction in DIRECTIONS:
            train_and_evaluate(symbol, direction)
