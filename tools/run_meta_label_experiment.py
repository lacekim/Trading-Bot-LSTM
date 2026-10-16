"""Meta-labeling: instead of asking a CNN/LSTM to predict direction on every
candle (what all 8 approaches tonight did, and none found edge -- most
candles are noise), let the rule-based MACD+200MA+RSI+StochRSI signal
(run_macd_200ma_signal_experiment.py) do the filtering -- it only flags the
~12,000 moments across the liquid universe where a real technical setup
exists -- and train the model to answer a narrower question: given this
setup already fired, is it a good one or a bad one.

Pools triggered events across ALL liquid symbols into one training set
(far more usable than any single symbol's 22-193 triggers alone), split
chronologically by global trigger timestamp (not per-symbol) so nothing
trains on a trigger's own future. Each triggered event is labeled by the
same triple-barrier mechanics already validated this session
(run_triple_barrier_target_experiment.py), just re-parameterized to match
the trend-following stop/target/hold used for the rule signal (3.0 ATR
stop, 20.0 ATR target, 240-candle max hold) instead of the short-horizon
defaults.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import list_gmx_symbols, load_gmx_ohlc
from trading_bot_v4.ml.cnn_lstm_model import V4CNNLSTMClassifier
from trading_bot_v4.risk.market_cap_tiers import liquid_symbols
from trading_bot_v4.utils.logger import build_logger
from train_model import DataHandler

logger = build_logger("v4_meta_label_experiment")

TIMEFRAME = "1h"
STARTING_CAPITAL = 100_000.0
CONTEXT_COLUMNS = ["quarter_sin", "quarter_cos", "btc_activity", "arima_forecast"]
ARIMA_ORDER = (2, 0, 2)
ARIMA_TRAIN_FRACTION = 0.80
FEATURE_COLUMNS = list(Config.FEATURE_COLUMNS) + CONTEXT_COLUMNS
SEQUENCE_LENGTH = Config.SEQUENCE_LENGTH
MODEL_ROOT = Path("models/experiments/meta_label_context")

SHORT_WINDOW = 24 * 30   # 30 days of hourly candles
LONG_WINDOW = 24 * 180   # 180 days of hourly candles

_btc_activity_cache: pd.DataFrame | None = None


def _arima_forecast_series(close: pd.Series) -> np.ndarray:
    """ARIMA(2,0,2) one-step-ahead forecasted log return at every bar, fit
    once on this symbol's own first 80% of history (parameters frozen there,
    verified via spot-check against step-by-step .append() to produce
    identical, non-leaking one-step-ahead values), then extended across the
    full series via model.apply()+get_prediction() -- the real, verified
    signal ARIMA found tonight (52.5% directional accuracy, p=0.0000 on BTC),
    too small to trade alone but real, now given to the classifier as a
    feature instead of left unused because it wasn't tradeable standalone."""
    log_returns = np.log(close / close.shift(1)).fillna(0.0)
    n = len(log_returns)
    train_end = max(int(n * ARIMA_TRAIN_FRACTION), 200)
    if train_end >= n - 10:
        return np.zeros(n)
    train = log_returns.iloc[:train_end]
    try:
        model = ARIMA(train, order=ARIMA_ORDER).fit()
        extended = model.apply(log_returns)
        in_sample = extended.get_prediction(start=0, end=train_end - 1, dynamic=False).predicted_mean.to_numpy()
        out_sample = extended.get_prediction(start=train_end, end=n - 1, dynamic=False).predicted_mean.to_numpy()
    except Exception as exc:
        logger.warning(f"ARIMA fit failed, filling zeros: {exc}")
        return np.zeros(n)
    return np.concatenate([in_sample, out_sample])


def _btc_activity_series() -> pd.DataFrame:
    """BTC's own rolling volume/ATR relative to their trailing baseline, as a
    market-wide activity proxy (same construction as
    run_quarter_activity_analysis.py, kept as a continuous score here instead
    of a HIGH/LOW bucket so the model gets the full signal, not a threshold).
    Both windows are strictly backward-looking -- no lookahead."""
    global _btc_activity_cache
    if _btc_activity_cache is not None:
        return _btc_activity_cache
    df = load_gmx_ohlc("BTC", TIMEFRAME)
    df = df.copy()
    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr

    volume_short = df["Volume"].rolling(SHORT_WINDOW).mean()
    volume_long = df["Volume"].rolling(LONG_WINDOW).mean()
    atr_short = df["ATR"].rolling(SHORT_WINDOW).mean()
    atr_long = df["ATR"].rolling(LONG_WINDOW).mean()
    activity_score = 0.5 * (volume_short / volume_long) + 0.5 * (atr_short / atr_long)

    out = df.reset_index()[[df.reset_index().columns[0]]].copy()
    out.columns = ["timestamp"]
    out["btc_activity"] = activity_score.to_numpy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp")
    _btc_activity_cache = out
    return out

# Same trend-following trade definition as the rule-signal experiment, so the
# label matches what a real trade taken at that trigger would have done.
STOP_ATR = 3.0
TARGET_ATR = 20.0
MAX_HOLD_CANDLES = 240

MIN_CALIBRATION_TRADES = 30
MIN_WINDOW_TRADES = 10


def _triple_barrier_outcome(df: pd.DataFrame, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (label, net_return) per row -- label is NaN where unresolved
    within MAX_HOLD_CANDLES. Mirrors run_triple_barrier_target_experiment.py's
    _triple_barrier_target but with this experiment's own stop/target/hold and
    returns the continuous net return too (needed for real profit-factor
    aggregation, not just the binarized win/loss label)."""
    close = df["Close"].to_numpy(np.float64)
    high = df["High"].to_numpy(np.float64)
    low = df["Low"].to_numpy(np.float64)
    atr = df["ATR"].to_numpy(np.float64)
    n = len(df)
    stop_first = Config.PAPER_STOP_TARGET_PRIORITY == "STOP_FIRST"

    stop_distance = atr * STOP_ATR
    target_distance = atr * TARGET_ATR
    if direction == "long":
        stop_price = close - stop_distance
        target_price = close + target_distance
    else:
        stop_price = close + stop_distance
        target_price = close - target_distance

    exit_price = np.full(n, np.nan)
    resolved = np.zeros(n, dtype=bool)
    idx = np.arange(n)

    for offset in range(1, MAX_HOLD_CANDLES + 1):
        forward_idx = idx + offset
        valid = forward_idx < n
        forward_high = np.full(n, np.nan)
        forward_low = np.full(n, np.nan)
        forward_high[valid] = high[forward_idx[valid]]
        forward_low[valid] = low[forward_idx[valid]]
        if direction == "long":
            hit_stop = forward_low <= stop_price
            hit_target = forward_high >= target_price
        else:
            hit_stop = forward_high >= stop_price
            hit_target = forward_low <= target_price
        use_stop = hit_stop & (~hit_target | stop_first)
        use_target = hit_target & ~use_stop
        newly = ~resolved & valid & (use_stop | use_target)
        exit_price[newly] = np.where(use_stop[newly], stop_price[newly], target_price[newly])
        resolved |= newly

    time_idx = idx + MAX_HOLD_CANDLES
    valid_time = time_idx < n
    time_exit = ~resolved & valid_time
    exit_price[time_exit] = close[time_idx[time_exit]]
    resolved |= time_exit

    slip = (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) / 10000.0
    fee_rate = Config.PAPER_FEE_BPS / 10000.0
    round_trip_cost = 2 * (slip + fee_rate)

    raw_return = (exit_price - close) / close
    if direction == "short":
        raw_return = -raw_return
    net_return = raw_return - round_trip_cost

    label = np.where(net_return > 0, 1.0, 0.0)
    label = np.where(resolved, label, np.nan)
    return label, net_return


def _load_triggered_frame(symbol: str, direction: str) -> pd.DataFrame | None:
    df = load_gmx_ohlc(symbol, TIMEFRAME)
    if df is None or len(df) < 250:
        return None
    df = df.copy()

    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr
    df = handler.prepare_features(df, prediction_horizon=1)
    if df.empty:
        return None

    df["ma200"] = df["Close"].rolling(window=200).mean()
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    rsi_min = df["rsi"].rolling(window=14).min()
    rsi_max = df["rsi"].rolling(window=14).max()
    stoch_rsi_raw = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    df["stoch_k"] = stoch_rsi_raw.rolling(window=3).mean()
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    if direction == "long":
        trend_ok = (df["Close"] > df["ma200"]) & (df["macd"] > df["macd_signal"]) & (df["rsi"] < 70)
        stoch_trigger = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) & (df["stoch_k"] < 40)
    else:
        trend_ok = (df["Close"] < df["ma200"]) & (df["macd"] < df["macd_signal"]) & (df["rsi"] > 30)
        stoch_trigger = (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) & (df["stoch_k"] > 60)
    df["triggered"] = trend_ok & stoch_trigger

    label, net_return = _triple_barrier_outcome(df, direction)
    df["meta_label"] = label
    df["net_return"] = net_return

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Context features: calendar quarter (cyclical, so Q4->Q1 isn't an
    # artificial jump) and BTC-derived market-wide activity -- the two
    # factors that showed a real, directionally consistent effect on the
    # rule signal's actual trades, now given to the model as inputs instead
    # of left for it to (fail to) rediscover implicitly.
    quarter = df["timestamp"].dt.quarter
    df["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)
    btc_activity = _btc_activity_series()
    df = pd.merge_asof(df.sort_values("timestamp"), btc_activity, on="timestamp", direction="backward")
    df = df.reset_index(drop=True)  # merge_asof drops the original positional index;
    # build_pooled_dataset's row_pos slicing depends on this matching feature_history's array order

    df["arima_forecast"] = _arima_forecast_series(df["Close"])

    df["symbol"] = symbol
    df["direction"] = direction
    return df


def build_pooled_dataset(direction: str, symbols: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Returns (index_frame, feature_history) where index_frame has one row per
    triggered+resolved event (symbol, timestamp, meta_label, net_return, and
    the integer row-position within that symbol's own feature_history array),
    and feature_history[symbol] is that symbol's full raw (unscaled) feature
    matrix so each trigger's preceding SEQUENCE_LENGTH-bar window can be
    sliced out after the pooled train/val/cal/holdout split is decided."""
    index_rows = []
    feature_history: dict[str, np.ndarray] = {}
    for symbol in symbols:
        frame = _load_triggered_frame(symbol, direction)
        if frame is None:
            continue
        # btc_activity is NaN for the first ~180 days of BTC's own history
        # (rolling-window warmup); zero-fill defensively rather than drop
        # rows, which would break the positional row_pos indexing used below.
        feature_history[symbol] = np.nan_to_num(frame[FEATURE_COLUMNS].to_numpy(np.float32), nan=0.0)
        triggered = frame.index[frame["triggered"] & frame["meta_label"].notna()]
        for row_pos in triggered:
            if row_pos < SEQUENCE_LENGTH:
                continue
            index_rows.append({
                "symbol": symbol, "timestamp": frame.loc[row_pos, "timestamp"],
                "row_pos": int(row_pos), "meta_label": float(frame.loc[row_pos, "meta_label"]),
                "net_return": float(frame.loc[row_pos, "net_return"]),
            })
        logger.info(f"{direction}/{symbol}: {len(frame)} rows, {int(frame['triggered'].sum())} triggers")

    index_frame = pd.DataFrame(index_rows).sort_values("timestamp").reset_index(drop=True)
    return index_frame, feature_history


def _build_sequences(index_frame: pd.DataFrame, feature_history: dict[str, np.ndarray],
                     scaler) -> np.ndarray:
    sequences = np.zeros((len(index_frame), SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, row in enumerate(index_frame.itertuples()):
        history = feature_history[row.symbol]
        window = history[row.row_pos - SEQUENCE_LENGTH + 1: row.row_pos + 1]
        sequences[i] = scaler.transform(window)
    return sequences


def _promotion_check(net_returns: np.ndarray, n_folds: int = 3) -> tuple[bool, list[dict]]:
    windows = []
    for fold_idx, fold in enumerate(np.array_split(net_returns, n_folds), start=1):
        if len(fold) == 0:
            windows.append({"window": fold_idx, "trades": 0, "return_pct": 0.0, "profit_factor": 0.0})
            continue
        gains = fold[fold > 0].sum()
        losses = -fold[fold < 0].sum()
        pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
        windows.append({
            "window": fold_idx, "trades": int(len(fold)),
            "return_pct": float(fold.sum() * 100.0), "profit_factor": pf,
        })
    promoted = bool(
        len(net_returns) >= MIN_CALIBRATION_TRADES
        and all(w["trades"] >= MIN_WINDOW_TRADES and w["profit_factor"] >= 1.30 and w["return_pct"] > 0
                for w in windows)
    )
    return promoted, windows


def run_direction(direction: str, symbols: list[str], force: bool = False) -> dict:
    index_frame, feature_history = build_pooled_dataset(direction, symbols)
    logger.info(f"{direction}: {len(index_frame)} total triggered+resolved events pooled across {len(feature_history)} symbols")
    if len(index_frame) < 200:
        return {"direction": direction, "trained": False, "reason": f"too few pooled events: {len(index_frame)}"}

    n = len(index_frame)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    cal_end = int(n * 0.90)
    train_idx = index_frame.iloc[:train_end]
    val_idx = index_frame.iloc[train_end:val_end]
    cal_idx = index_frame.iloc[val_end:cal_end]
    hold_idx = index_frame.iloc[cal_end:]

    handler = DataHandler()
    train_feature_rows = np.vstack([
        feature_history[row.symbol][row.row_pos] for row in train_idx.itertuples()
    ])
    handler.normalize_data(train_feature_rows, fit=True)
    scaler = handler.scaler

    X_train = _build_sequences(train_idx, feature_history, scaler)
    X_val = _build_sequences(val_idx, feature_history, scaler)
    X_cal = _build_sequences(cal_idx, feature_history, scaler)
    X_hold = _build_sequences(hold_idx, feature_history, scaler)
    y_train = train_idx["meta_label"].to_numpy(np.float32)
    y_val = val_idx["meta_label"].to_numpy(np.float32)

    n_samples = len(y_train)
    n_positive = float(np.sum(y_train))
    n_negative = n_samples - n_positive
    class_weight = {
        0: n_samples / (2 * n_negative) if n_negative > 0 else 1.0,
        1: n_samples / (2 * n_positive) if n_positive > 0 else 1.0,
    }

    classifier = V4CNNLSTMClassifier()
    classifier.build_model((SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))
    classifier.model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=30, batch_size=128, class_weight=class_weight,
        callbacks=[
            EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True, min_delta=0.001),
            ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=3, min_lr=1e-7),
        ],
        verbose=0,
    )

    cal_probability = classifier.model.predict(X_cal, verbose=0).reshape(-1)
    hold_probability = classifier.model.predict(X_hold, verbose=0).reshape(-1)
    hold_target = hold_idx["meta_label"].to_numpy(int)
    holdout_auc = (
        float(roc_auc_score(hold_target, hold_probability))
        if len(np.unique(hold_target)) > 1 else float("nan")
    )

    cal_returns = cal_idx["net_return"].to_numpy(np.float64)
    hold_returns = hold_idx["net_return"].to_numpy(np.float64)

    best_threshold, best_pf = 0.5, -1.0
    for threshold in np.arange(0.40, 0.91, 0.02):
        mask = cal_probability >= threshold
        if mask.sum() < MIN_CALIBRATION_TRADES:
            continue
        selected = cal_returns[mask]
        gains = selected[selected > 0].sum()
        losses = -selected[selected < 0].sum()
        pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
        if selected.sum() > 0 and pf > best_pf:
            best_pf, best_threshold = pf, float(threshold)

    hold_mask = hold_probability >= best_threshold
    hold_selected = hold_returns[hold_mask]
    promoted, windows = _promotion_check(hold_selected)

    gains = hold_selected[hold_selected > 0].sum()
    losses = -hold_selected[hold_selected < 0].sum()
    holdout_pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    classifier.save_model(MODEL_ROOT / f"meta_label_{direction}.h5")
    with (MODEL_ROOT / f"scaler_{direction}.pkl").open("wb") as handle_file:
        pickle.dump(scaler, handle_file)

    result = {
        "direction": direction, "trained": True, "promoted": promoted,
        "threshold": best_threshold, "pooled_events": int(n),
        "train_events": len(train_idx), "holdout_events": len(hold_idx),
        "holdout_trades_selected": int(hold_mask.sum()),
        "holdout_return_pct": float(hold_selected.sum() * 100.0),
        "holdout_profit_factor": holdout_pf, "holdout_auc": holdout_auc,
        "windows": windows,
    }
    logger.info(
        f"{direction}: promoted={promoted} threshold={best_threshold:.2f} "
        f"holdout_trades={result['holdout_trades_selected']}/{result['holdout_events']} "
        f"holdout_return_pct={result['holdout_return_pct']:.2f} "
        f"holdout_profit_factor={holdout_pf:.2f} holdout_auc={holdout_auc:.3f}"
    )
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    directions = ["long", "short"] if args.direction == "both" else [args.direction]

    all_symbols = list_gmx_symbols(TIMEFRAME)
    results = []
    for direction in directions:
        symbols = liquid_symbols(direction.upper(), all_symbols)
        result = run_direction(direction, symbols, force=args.force)
        results.append(result)

    out = Path("logs/v4_meta_label_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
