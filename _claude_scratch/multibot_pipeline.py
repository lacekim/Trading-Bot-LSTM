#!/usr/bin/env python3
"""Multi-bot LightGBM-style (from-scratch GBDT, no network/no pyarrow needed)
long/short system on real Kraken 1h data (BTC/ETH/XMR/PAXG/SOL/TRX).

PREDECLARED BEFORE SEEING RESULTS:
- Labeling: triple-barrier, STOP_ATR=3.0, TARGET_ATR=20.0, MAX_HOLD=240 candles,
  24bp round-trip cost (5bp fee + 5bp slippage + 2bp impact, x2) -- same convention
  used throughout this session for comparability.
- Ensemble: 5 GBDT variants per direction (long, short), different random seeds +
  row bootstrap. Signal fires only if >=4/5 vote positive (proba>0.5) AND mean
  ensemble proba clears a threshold selected by max-PF scan on the calibration
  split. This is the "multi-bot cross-validation" structure agreed earlier.
- Split: chronological pooled 70/10/10/10 train/val/cal/holdout, holdout broken
  into 3 equal sub-windows (same convention as the meta-label experiments).
- Promotion bar (reusing project's own established bar): PF>=1.30 AND positive
  return in EVERY holdout sub-window.
- Pull-out/re-entry: rolling PF over last N=20 closed trades; stop opening new
  trades if rolling PF<1.0; resume once rolling PF recovers >=1.1 (hysteresis).
- June 2026 AI-correction window (May15-Jul1 2026) reported separately, per
  standing rule, not folded into the aggregate.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbdt import GBDTClassifier

KRAKEN_DIR = Path.home() / "mnt" / "Javier_Santiago_Gaston_de_Iriarte_canrera" / "freqtrade-develop" / "user_data" / "data" / "kraken"
SYMBOLS = ["BTC", "ETH", "XMR", "PAXG", "SOL", "TRX"]

STOP_ATR = 3.0
TARGET_ATR = 20.0
MAX_HOLD = 240
FEE_BPS, SLIP_BPS, IMPACT_BPS = 5.0, 5.0, 2.0
ROUND_TRIP_COST = 2 * (FEE_BPS + SLIP_BPS + IMPACT_BPS) / 10000.0

SEQUENCE_WARMUP = 210  # need >=200 for MA200 feature
N_ENSEMBLE = 5
AGREE_MIN = 4
MIN_CAL_TRADES = 30
MIN_WINDOW_TRADES = 10
ROLL_N = 20
PULLOUT_PF = 1.0
REENTRY_PF = 1.1

JUNE_START = pd.Timestamp("2026-05-15")
JUNE_END = pd.Timestamp("2026-07-01")


def load_symbol(sym):
    df = pd.read_csv(KRAKEN_DIR / f"{sym}_USD-1h.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["Open", "High", "Low", "Close", "Volume"]]


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


FEATURE_COLS = ["returns", "volatility", "rsi", "macd_diff", "ma200_dist", "ma50_dist",
                "momentum24", "atr_norm", "volume_ratio", "stoch_k"]


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
        use_stop = hit_stop & (~hit_target | True)  # STOP_FIRST
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
    train_end = int(n * 0.70); val_end = int(n * 0.80); cal_end = int(n * 0.90)
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

    results = {}
    ensemble_proba = {}
    for direction in ["long", "short"]:
        y_train = train[f"label_{direction}"].to_numpy(np.float64)
        y_val = val[f"label_{direction}"].to_numpy(np.float64)
        mask_tr = ~np.isnan(y_train); mask_val = ~np.isnan(y_val)
        Xtr, ytr = X_train[mask_tr], y_train[mask_tr]
        Xv, yv = X_val[mask_val], y_val[mask_val]

        models = []
        rng = np.random.default_rng(100)
        for i in range(N_ENSEMBLE):
            boot_idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)
            clf = GBDTClassifier(n_estimators=150, learning_rate=0.08, max_leaves=15,
                                  min_samples_leaf=max(30, len(Xtr)//500),
                                  random_state=100 + i, early_stopping_rounds=10)
            clf.fit(Xtr[boot_idx], ytr[boot_idx], Xv, yv)
            models.append(clf)
            print(f"  [{direction}] model {i}: trees={clf.best_n_trees_}", flush=True)

        def ensemble_predict(X):
            probs = np.stack([m.predict_proba(X) for m in models], axis=0)  # (N_ENSEMBLE, n)
            votes = (probs > 0.5).sum(axis=0)
            mean_p = probs.mean(axis=0)
            return mean_p, votes

        cal_p, cal_votes = ensemble_predict(X_cal)
        hold_p, hold_votes = ensemble_predict(X_hold)
        ensemble_proba[direction] = {"cal_p": cal_p, "cal_votes": cal_votes, "hold_p": hold_p, "hold_votes": hold_votes}

        cal_net = cal[f"net_{direction}"].to_numpy(np.float64)
        best_thr, best_pf = 0.5, -1.0
        for thr in np.arange(0.40, 0.91, 0.02):
            mask = (cal_votes >= AGREE_MIN) & (cal_p >= thr) & ~np.isnan(cal_net)
            if mask.sum() < MIN_CAL_TRADES:
                continue
            sel = cal_net[mask]
            gains = sel[sel > 0].sum(); losses = -sel[sel < 0].sum()
            pf = gains / losses if losses > 0 else (np.inf if gains > 0 else 0.0)
            if sel.sum() > 0 and pf > best_pf:
                best_pf, best_thr = pf, thr
        results[direction] = {"threshold": best_thr, "cal_pf": best_pf}
        print(f"[{direction}] selected threshold={best_thr:.2f} cal_pf={best_pf:.3f}")

    # Combine: at each holdout row, fire long if long ensemble agrees+clears threshold
    # and short doesn't also fire; vice versa; else flat.
    long_thr = results["long"]["threshold"]; short_thr = results["short"]["threshold"]
    long_fire = (ensemble_proba["long"]["hold_votes"] >= AGREE_MIN) & (ensemble_proba["long"]["hold_p"] >= long_thr)
    short_fire = (ensemble_proba["short"]["hold_votes"] >= AGREE_MIN) & (ensemble_proba["short"]["hold_p"] >= short_thr)
    both = long_fire & short_fire
    long_fire = long_fire & ~both
    short_fire = short_fire & ~both

    hold = hold.copy()
    hold["long_fire"] = long_fire
    hold["short_fire"] = short_fire
    hold["trade_return"] = np.where(long_fire, hold["net_long"], np.where(short_fire, hold["net_short"], np.nan))
    hold["direction"] = np.where(long_fire, "long", np.where(short_fire, "short", "flat"))

    trades = hold[hold["direction"] != "flat"].dropna(subset=["trade_return"]).reset_index(drop=True)
    print(f"\ntotal holdout rows={len(hold)}, trades fired={len(trades)} (long={int(long_fire.sum())}, short={int(short_fire.sum())})")

    # Apply rolling pull-out/re-entry on the chronological trade sequence
    active = True
    taken = []
    roll_returns = []
    for _, row in trades.iterrows():
        if active:
            taken.append(True)
            roll_returns.append(row["trade_return"])
        else:
            taken.append(False)
        if len(roll_returns) >= ROLL_N:
            recent = np.array(roll_returns[-ROLL_N:])
            g = recent[recent > 0].sum(); l = -recent[recent < 0].sum()
            pf_roll = g / l if l > 0 else (np.inf if g > 0 else 0.0)
            if active and pf_roll < PULLOUT_PF:
                active = False
            elif not active and pf_roll >= REENTRY_PF:
                active = True
    trades["taken_after_pullout"] = taken

    def pf_of(returns):
        g = returns[returns > 0].sum(); l = -returns[returns < 0].sum()
        return float(g / l) if l > 0 else (float("inf") if g > 0 else 0.0)

    all_returns = trades["trade_return"].to_numpy(np.float64)
    filt_returns = trades.loc[trades["taken_after_pullout"], "trade_return"].to_numpy(np.float64)
    print(f"\nWITHOUT pull-out: trades={len(all_returns)} return_pct={all_returns.sum()*100:.1f} pf={pf_of(all_returns):.3f}")
    print(f"WITH pull-out/re-entry: trades={len(filt_returns)} return_pct={filt_returns.sum()*100:.1f} pf={pf_of(filt_returns):.3f}")

    # 3-way holdout window breakdown (on the no-pullout trade stream, for comparability to promotion bar)
    folds = np.array_split(np.arange(len(trades)), 3)
    windows = []
    promoted_ok = True
    for i, fidx in enumerate(folds, start=1):
        sub = trades.iloc[fidx]
        r = sub["trade_return"].to_numpy(np.float64)
        pf = pf_of(r)
        ret_pct = r.sum() * 100
        lo, hi = (sub["timestamp"].min(), sub["timestamp"].max()) if len(sub) else (None, None)
        overlap = (lo is not None) and (lo <= JUNE_END) and (hi >= JUNE_START)
        ok = len(sub) >= MIN_WINDOW_TRADES and pf >= 1.30 and ret_pct > 0
        promoted_ok = promoted_ok and ok
        windows.append({"window": i, "trades": len(sub), "return_pct": float(ret_pct), "pf": pf,
                         "start": str(lo), "end": str(hi), "june2026_overlap": bool(overlap), "clears_bar": ok})
        tag = "  <-- OVERLAPS June 2026 AI-correction window" if overlap else ""
        print(f"  window {i}: trades={len(sub)} return_pct={ret_pct:.1f} pf={pf:.3f} "
              f"dates=[{lo} -> {hi}] clears_bar={ok}{tag}")

    promoted = bool(len(all_returns) >= MIN_CAL_TRADES and promoted_ok)
    print(f"\nPROMOTION CHECK (PF>=1.30 AND positive return in every window): promoted={promoted}")

    by_symbol = trades.groupby("symbol")["trade_return"].agg(["count", "sum"]).sort_values("sum")
    print("\nper-symbol holdout contribution:")
    for sym, row in by_symbol.iterrows():
        print(f"  {sym}: n={int(row['count'])} sum_return={row['sum']*100:.1f}%")

    out = {
        "thresholds": results,
        "without_pullout": {"trades": len(all_returns), "return_pct": float(all_returns.sum()*100), "pf": pf_of(all_returns)},
        "with_pullout": {"trades": len(filt_returns), "return_pct": float(filt_returns.sum()*100), "pf": pf_of(filt_returns)},
        "windows": windows,
        "promoted": promoted,
        "per_symbol": [{"symbol": s, "n": int(r["count"]), "sum_return_pct": float(r["sum"]*100)} for s, r in by_symbol.iterrows()],
    }
    Path("/tmp/multibot_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote /tmp/multibot_results.json")


if __name__ == "__main__":
    main()
