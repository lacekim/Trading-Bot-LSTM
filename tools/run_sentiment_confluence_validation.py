"""End-to-end test: does requiring real sentiment confluence beat the
price-only GBM signal tested earlier tonight?

Uses the two symbols where both exist: a trained per-asset GBM price model
(models/experiments/gbm_relative_strength/) and verified, point-in-time-
safe GDELT sentiment coverage (data/sentiment/gdelt_news_tone_daily_verified.csv).
Everything else reuses infrastructure already validated tonight -- the same
real-execution backtest engine, the same calibrate-on-calibration/report-on-
holdout discipline, the same "sweep candidates, don't hand-pick a threshold"
rule _select_threshold already established.

Two sentiment hypotheses are swept, not assumed: "confirming" (bullish tone
backs a LONG, bearish backs a SHORT) and "contrarian" (extremity in the
opposite direction backs the trade instead) -- the small pilot earlier
tonight suggested contrarian might be the real pattern (panic at the COVID
bottom, euphoria before the ETF-approval pullback), but that was too small
a sample to trust; this lets the calibration data decide instead of
assuming it.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.ml.per_asset_trainer import (
    MIN_CALIBRATION_TRADES,
    MIN_WINDOW_TRADES,
    _ENDPOINT_COLUMNS,
)
from trading_bot_v4.signals.confluence import combine_signal_frames
from trading_bot_v4.signals.gdelt_sentiment import load_sentiment_signal_frame
from trading_bot_v4.utils.logger import build_logger
from tools.run_gbm_relative_strength_experiment import (
    FEATURE_COLUMNS,
    _chronological_split,
    _load_symbol_frame,
    _model_path,
)

logger = build_logger("v4_sentiment_confluence_validation")

SYMBOLS = ["BTC", "ETH"]
DIRECTIONS = ["long", "short"]
SENTIMENT_THRESHOLDS = np.arange(0.5, 5.5, 0.5)


def _base_frame(endpoints: pd.DataFrame) -> pd.DataFrame:
    frame = endpoints.copy()
    frame["price"] = frame["Close"].astype(float)
    frame["open"] = frame["Open"].astype(float)
    frame["high"] = frame["High"].astype(float)
    frame["low"] = frame["Low"].astype(float)
    frame["close"] = frame["Close"].astype(float)
    frame["atr"] = frame["ATR"].astype(float)
    frame["_entry_eligible"] = True
    return frame


def _score(frame: pd.DataFrame, symbol: str) -> dict:
    summary, _ = simulate_production_symbol(
        frame, symbol, 100_000.0, entry_eligible=True,
        fee_bps=Config.PAPER_FEE_BPS,
        slippage_bps=Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS,
    )
    return summary


def _sentiment_direction_frame(sentiment: pd.DataFrame, direction: str, threshold: float, contrarian: bool) -> pd.DataFrame:
    label = "LONG" if direction == "long" else "SHORT"
    frame = sentiment[["timestamp"]].copy()
    score = sentiment["sentiment_score"]
    if (direction == "long") != contrarian:
        # long+confirming or short+contrarian -> bullish tone triggers LONG / bearish triggers SHORT confirmation
        hit = score >= threshold if direction == "long" else score <= -threshold
    else:
        # long+contrarian or short+confirming -> opposite extremity triggers
        hit = score <= -threshold if direction == "long" else score >= threshold
    frame["model_direction"] = np.where(hit, label, "HOLD")
    return frame


def evaluate(symbol: str, direction: str) -> None:
    df = _load_symbol_frame(symbol, "1h", direction)
    split = _chronological_split(df)
    if split is None:
        logger.info(f"{symbol}/{direction}: insufficient data for split, skipping")
        return
    train, calibration, holdout = split

    model_path = _model_path(direction, symbol)
    if not model_path.exists():
        logger.info(f"{symbol}/{direction}: no trained model at {model_path}, skipping")
        return
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    label = "LONG" if direction == "long" else "SHORT"

    calibration_probability = model.predict_proba(calibration[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    calibration_endpoints = _base_frame(calibration[_ENDPOINT_COLUMNS].reset_index(drop=True))

    holdout_probability = model.predict_proba(holdout[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    holdout_endpoints = _base_frame(holdout[_ENDPOINT_COLUMNS].reset_index(drop=True))

    # price-only threshold selection (unchanged from tonight's earlier methodology)
    candidates = []
    for threshold in np.arange(0.50, 0.91, 0.02):
        frame = calibration_endpoints.copy()
        frame["model_direction"] = np.where(calibration_probability >= threshold, label, "HOLD")
        summary = _score(frame, symbol)
        if summary["trades"] >= MIN_CALIBRATION_TRADES:
            candidates.append((float(threshold), summary))
    viable = [c for c in candidates if c[1]["profit_factor"] >= 1.20 and c[1]["return_pct"] > 0]
    pool = viable or candidates
    if not pool:
        logger.info(f"{symbol}/{direction}: no viable price threshold found in calibration, skipping")
        return
    price_threshold = max(pool, key=lambda c: (c[1]["profit_factor"], c[1]["return_pct"]))[0]

    calibration_price_frame = calibration_endpoints.copy()
    calibration_price_frame["model_direction"] = np.where(calibration_probability >= price_threshold, label, "HOLD")
    holdout_price_frame = holdout_endpoints.copy()
    holdout_price_frame["model_direction"] = np.where(holdout_probability >= price_threshold, label, "HOLD")

    price_only_holdout = _score(holdout_price_frame, symbol)

    # sentiment confluence: sweep contrarian x threshold on CALIBRATION only, report the winner on HOLDOUT
    calibration_as_of = calibration_price_frame["timestamp"].max()
    calibration_sentiment = load_sentiment_signal_frame(symbol, "1h", calibration_as_of)

    best = None
    for contrarian in (False, True):
        for threshold in SENTIMENT_THRESHOLDS:
            sentiment_direction = _sentiment_direction_frame(calibration_sentiment, direction, float(threshold), contrarian)
            combined = combine_signal_frames(calibration_price_frame, sentiment_direction)
            summary = _score(combined, symbol)
            if summary["trades"] < MIN_CALIBRATION_TRADES:
                continue
            key = (summary["profit_factor"], summary["return_pct"])
            if best is None or key > best["key"]:
                best = {"contrarian": contrarian, "threshold": float(threshold), "key": key, "calibration_summary": summary}

    if best is None:
        logger.info(f"{symbol}/{direction}: no confluence combination cleared MIN_CALIBRATION_TRADES in calibration; "
                     f"sentiment coverage is likely too sparse for this symbol/split")
        logger.info(f"{symbol}/{direction}: PRICE-ONLY holdout -> trades={price_only_holdout['trades']} "
                     f"return_pct={price_only_holdout['return_pct']:.2f} profit_factor={price_only_holdout['profit_factor']:.2f}")
        return

    holdout_as_of = holdout_price_frame["timestamp"].max()
    holdout_sentiment = load_sentiment_signal_frame(symbol, "1h", holdout_as_of)
    holdout_sentiment_direction = _sentiment_direction_frame(holdout_sentiment, direction, best["threshold"], best["contrarian"])
    holdout_combined = combine_signal_frames(holdout_price_frame, holdout_sentiment_direction)
    confluence_holdout = _score(holdout_combined, symbol)

    # 3-window walk-forward robustness check -- the actual promotion bar every
    # other experiment tonight was held to, not just a single holdout score.
    windows = []
    ordered = holdout_combined.sort_values("timestamp").reset_index(drop=True)
    for window, indices in enumerate(np.array_split(np.arange(len(ordered)), 3), start=1):
        section = ordered.iloc[indices].reset_index(drop=True)
        summary = _score(section, symbol)
        windows.append({"window": window, **summary})
    promoted = bool(
        confluence_holdout["trades"] >= MIN_CALIBRATION_TRADES
        and all(row["trades"] >= MIN_WINDOW_TRADES and row["profit_factor"] >= 1.30 and row["return_pct"] > 0
                for row in windows)
    )

    logger.info(f"{symbol}/{direction}: selected sentiment rule -> contrarian={best['contrarian']} threshold={best['threshold']}")
    logger.info(f"{symbol}/{direction}: PRICE-ONLY holdout      -> trades={price_only_holdout['trades']:>4} "
                f"return_pct={price_only_holdout['return_pct']:>7.2f} profit_factor={price_only_holdout['profit_factor']:.2f}")
    logger.info(f"{symbol}/{direction}: SENTIMENT-CONFLUENCE holdout -> trades={confluence_holdout['trades']:>4} "
                f"return_pct={confluence_holdout['return_pct']:>7.2f} profit_factor={confluence_holdout['profit_factor']:.2f}")
    for row in windows:
        logger.info(f"{symbol}/{direction}:   window {row['window']} -> trades={row['trades']:>3} "
                    f"return_pct={row['return_pct']:>7.2f} profit_factor={row['profit_factor']:.2f}")
    logger.info(f"{symbol}/{direction}: PROMOTED={promoted}")


if __name__ == "__main__":
    for symbol in SYMBOLS:
        for direction in DIRECTIONS:
            evaluate(symbol, direction)
