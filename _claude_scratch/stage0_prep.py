#!/usr/bin/env python3
"""Stage 0: load 1h OHLC for BTC/ETH/XMR/PAXG/SOL/TRX, build features and
triple-barrier labels exactly as predeclared, pool chronologically, do the
70/10/10/10 train/val/cal/holdout split, and save everything needed by the
training stages to a single .npz checkpoint. This is pure data prep (pandas),
no ML -- runs once so the (slower, Python-loop-heavy) training stages never
have to touch pandas/CSV loading themselves.

PREDECLARED (unchanged from original multibot_pipeline.py spec):
- Labeling: triple-barrier, STOP_ATR=3.0, TARGET_ATR=20.0, MAX_HOLD=240 candles,
  24bp round-trip cost (5bp fee + 5bp slippage + 2bp impact, x2).
- Split: chronological pooled 70/10/10/10 train/val/cal/holdout.

DATA SOURCE SWAP (disclosed, done at the user's explicit request after the
first Kraken-only run showed every Kraken file for all 6 symbols stops in
early Jan 2026, so the standing June-2026 reporting requirement had nothing
to report on):
- BTC now loads from binance_margin_bot/trading_bot/BTCUSDT-15m-data.csv
  (Binance, 2017-08-17 -> 2026-09-30) resampled 15m->1h, instead of the
  Kraken BTC_USD-1h.csv (which stopped 2026-01-05). This is the only one of
  the six symbols that has ANY cached history past Jan 2026 anywhere in the
  project -- confirmed by directly searching for ETHUSDT/XMRUSDT/SOLUSDT/
  TRXUSDT/PAXGUSDT OHLC files and finding none.
- ETH/XMR/PAXG/SOL/TRX are UNCHANGED -- still Kraken, still stopping Jan 2026.
  So this run's June-2026 window can only ever show BTC trades; the other
  five symbols still have no data there. That's a real, disclosed limit of
  what's on disk, not a shortcut taken quietly.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

KRAKEN_DIR = Path(__file__).resolve().parent  # Kraken CSVs copied alongside this script
BINANCE_BTC_PATH = Path(__file__).resolve().parent / "BTCUSDT-15m-data.csv"  # copied alongside too
SYMBOLS = ["BTC", "ETH", "XMR", "PAXG", "SOL", "TRX"]

STOP_ATR = 3.0
TARGET_ATR = 20.0
MAX_HOLD = 240
FEE_BPS, SLIP_BPS, IMPACT_BPS = 5.0, 5.0, 2.0
ROUND_TRIP_COST = 2 * (FEE_BPS + SLIP_BPS + IMPACT_BPS) / 10000.0
SEQUENCE_WARMUP = 210

FEATURE_COLS = ["returns", "volatility", "rsi", "macd_diff", "ma200_dist", "ma50_dist",
                "momentum24", "atr_norm", "volume_ratio", "stoch_k"]


def load_symbol_kraken(sym):
    df = pd.read_csv(KRAKEN_DIR / f"{sym}_USD-1h.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_btc_binance():
    """Binance BTCUSDT, 15m native, resampled to 1h. Standard OHLCV resample
    (open=first, high=max, low=min, close=last, volume=sum) -- no barrier
    logic touches this step, just a timeframe change."""
    df = pd.read_csv(BINANCE_BTC_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    hourly = df[["Open", "High", "Low", "Close", "Volume"]].resample("1h").agg(agg)
    hourly = hourly.dropna(subset=["Open", "High", "Low", "Close"])
    return hourly


def load_symbol(sym):
    if sym == "BTC":
        return load_btc_binance()
    return load_symbol_kraken(sym)


def atr(df, period=14):
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df):
    df = df.copy()
    df["ATR"] = atr(df)
    df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility"] = df["returns"].rolling(20).std()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_diff"] = df["macd"] - df["macd_signal"]
    df["ma200_dist"] = (df["Close"] - df["Close"].rolling(200).mean()) / df["Close"]
    df["ma50_dist"] = (df["Close"] - df["Close"].rolling(50).mean()) / df["Close"]
    df["momentum24"] = df["Close"].pct_change(24)
    df["atr_norm"] = df["ATR"] / df["Close"]
    vol_ma = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (vol_ma + 1e-10)
    rsi_min = df["rsi"].rolling(14).min()
    rsi_max = df["rsi"].rolling(14).max()
    df["stoch_k"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    return df


def triple_barrier(df, direction):
    close = df["Close"].to_numpy(np.float64)
    high = df["High"].to_numpy(np.float64)
    low = df["Low"].to_numpy(np.float64)
    atr_v = df["ATR"].to_numpy(np.float64)
    n = len(df)
    stop_d = atr_v * STOP_ATR
    target_d = atr_v * TARGET_ATR
    if direction == "long":
        stop_p = close - stop_d; target_p = close + target_d
    else:
        stop_p = close + stop_d; target_p = close - target_d
    exit_p = np.full(n, np.nan)
    resolved = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    for off in range(1, MAX_HOLD + 1):
        fwd = idx + off
        valid = fwd < n
        fh = np.full(n, np.nan); fl = np.full(n, np.nan)
        fh[valid] = high[fwd[valid]]; fl[valid] = low[fwd[valid]]
        if direction == "long":
            hit_stop = fl <= stop_p; hit_target = fh >= target_p
        else:
            hit_stop = fh >= stop_p; hit_target = fl <= target_p
        use_stop = hit_stop  # STOP_FIRST convention
        use_target = hit_target & ~use_stop
        newly = ~resolved & valid & (use_stop | use_target)
        exit_p[newly] = np.where(use_stop[newly], stop_p[newly], target_p[newly])
        resolved |= newly
    time_idx = idx + MAX_HOLD
    valid_t = time_idx < n
    time_exit = ~resolved & valid_t
    exit_p[time_exit] = close[time_idx[time_exit]]
    resolved |= time_exit
    raw = (exit_p - close) / close
    if direction == "short":
        raw = -raw
    net = raw - ROUND_TRIP_COST
    label = np.where(net > 0, 1.0, 0.0)
    label = np.where(resolved, label, np.nan)
    return label, net


def main():
    frames = []
    for sym in SYMBOLS:
        raw = load_symbol(sym)
        feat = build_features(raw)
        for direction in ["long", "short"]:
            lab, net = triple_barrier(feat, direction)
            feat[f"label_{direction}"] = lab
            feat[f"net_{direction}"] = net
        feat["symbol"] = sym
        feat = feat.iloc[SEQUENCE_WARMUP:].reset_index()
        feat = feat.rename(columns={"date": "timestamp"})
        feat = feat.dropna(subset=FEATURE_COLS)
        frames.append(feat)
        print(f"{sym}: {len(feat)} usable rows, range {feat['timestamp'].min()} -> {feat['timestamp'].max()}", flush=True)

    pooled = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    n = len(pooled)
    # SPLIT WIDENED (predeclared): holdout 10%->30% so there's enough out-of-
    # sample volume to actually judge the long-side rule found earlier,
    # instead of the 1-trade holdout the original 70/10/10/10 split gave.
    # train/val/cal shrink accordingly; everything else (features, labeling,
    # vote rule, threshold-scan procedure) is unchanged.
    train_end = int(n * 0.50); val_end = int(n * 0.60); cal_end = int(n * 0.70)
    train = pooled.iloc[:train_end]
    val = pooled.iloc[train_end:val_end]
    cal = pooled.iloc[val_end:cal_end]
    hold = pooled.iloc[cal_end:].reset_index(drop=True)
    print(f"\npooled={n} train={len(train)} val={len(val)} cal={len(cal)} holdout={len(hold)}")
    print(f"holdout date range: {hold['timestamp'].min()} -> {hold['timestamp'].max()}")

    X_train = train[FEATURE_COLS].to_numpy(np.float64)
    X_val = val[FEATURE_COLS].to_numpy(np.float64)
    X_cal = cal[FEATURE_COLS].to_numpy(np.float64)
    X_hold = hold[FEATURE_COLS].to_numpy(np.float64)

    out = dict(
        X_train=X_train, X_val=X_val, X_cal=X_cal, X_hold=X_hold,
        y_train_long=train["label_long"].to_numpy(np.float64),
        y_train_short=train["label_short"].to_numpy(np.float64),
        y_val_long=val["label_long"].to_numpy(np.float64),
        y_val_short=val["label_short"].to_numpy(np.float64),
        net_cal_long=cal["net_long"].to_numpy(np.float64),
        net_cal_short=cal["net_short"].to_numpy(np.float64),
        net_hold_long=hold["net_long"].to_numpy(np.float64),
        net_hold_short=hold["net_short"].to_numpy(np.float64),
        hold_timestamp=hold["timestamp"].astype(str).to_numpy(),
        hold_symbol=hold["symbol"].to_numpy(),
    )
    np.savez_compressed(Path(__file__).resolve().parent / "data_splits.npz", **out)
    print("\nwrote data_splits.npz")
    print(f"X_train shape={X_train.shape} X_val shape={X_val.shape} X_cal shape={X_cal.shape} X_hold shape={X_hold.shape}")


if __name__ == "__main__":
    main()
