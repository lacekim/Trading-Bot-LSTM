#!/usr/bin/env python3
"""Diagnostic battery #1 on the FROZEN long-side pipeline from
run_lightgbm_multibot.py. This does NOT touch the original script, the
original thresholds, or the original holdout in any way that would let a
later result retroactively change an earlier one -- it is a separate,
read-only diagnostic run against the same frozen config.

PREDECLARED (stated in chat before this was written, repeated here verbatim
as the versioned record of the experiment):

  1. ENTRY-LAG TEST: identical pipeline (same features, same labels-minus-one
     change, same LightGBM config/seeds, same threshold-scan procedure), with
     exactly one change -- the triple-barrier entry price moves from Close[t]
     (the same bar the features were computed on) to Open[t+1] (the next
     bar's open), and barrier monitoring starts one bar later accordingly.
     Nothing else changes. If performance collapses, that's a timing
     artifact, not a real edge.
  2. SHUFFLED-LABEL NULL MODEL: identical features/config, y_train randomly
     permuted (rng.permutation) before bootstrapping, same threshold-scan on
     calibration, same holdout evaluation. Tests whether the 91-point
     threshold scan alone can manufacture apparent profitability from pure
     noise via multiple-comparisons luck.
  3. SIMPLE BASELINES, same holdout, same cost structure, no ML:
       - buy_and_hold: per-symbol cumulative return, holdout start->end
       - momentum rule: long whenever momentum24>0 AND ma50_dist>0
       - mean_reversion rule: long whenever rsi<30
     Each applied to the existing net_hold_long column (not fired via a
     model) so the comparison uses the exact same frictions/labeling.
  4. VOTE-ONLY (no threshold): the SAME trained frozen ensemble (not
     retrained), decision rule relaxed to votes>=AGREE_MIN only, dropping
     the probability-threshold filter entirely, to isolate how much of the
     frozen result comes from the ensemble vote vs. the threshold scan.

Short side excluded from this batch -- it already failed calibration in the
frozen run (threshold hit the unreachable sentinel, 0 cal trades); retesting
a signal that already produced zero trades adds no information yet.

Scope note on #1: ATR used for stop/target sizing is still ATR[t] (known at
decision time t, the bar whose close triggered the signal) -- only the
*execution reference price* moves to Open[t+1]. This isolates the "zero
latency execution" question without also changing what information the
model is allowed to use.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "tools") \
    else Path(__file__).resolve().parent
KRAKEN_DIR = PROJECT_ROOT / "freqtrade-develop" / "user_data" / "data" / "kraken"
BINANCE_BTC_PATH = PROJECT_ROOT / "binance_margin_bot" / "trading_bot" / "BTCUSDT-15m-data.csv"
OUT_PATH = PROJECT_ROOT / "runtime" / "diagnostics_batch1_results.json"

SYMBOLS = ["BTC", "ETH", "XMR", "PAXG", "SOL", "TRX"]

STOP_ATR = 3.0
TARGET_ATR = 20.0
MAX_HOLD = 240
FEE_BPS, SLIP_BPS, IMPACT_BPS = 5.0, 5.0, 2.0
ROUND_TRIP_COST = 2 * (FEE_BPS + SLIP_BPS + IMPACT_BPS) / 10000.0
SEQUENCE_WARMUP = 210

FEATURE_COLS = ["returns", "volatility", "rsi", "macd_diff", "ma200_dist", "ma50_dist",
                "momentum24", "atr_norm", "volume_ratio", "stoch_k"]

N_ENSEMBLE = 5
AGREE_MIN = 4
MIN_CAL_TRADES = 30
MIN_WINDOW_TRADES = 10
JUNE_START = pd.Timestamp("2026-05-15")
JUNE_END = pd.Timestamp("2026-07-01")

LGB_PARAMS = dict(n_estimators=500, learning_rate=0.08, num_leaves=15,
                   objective="binary", verbosity=-1)
EARLY_STOPPING_ROUNDS = 20
SEED_BASE = 100


# ---------- data loading (byte-identical to run_lightgbm_multibot.py) ----------

def load_symbol_kraken(sym: str) -> pd.DataFrame:
    df = pd.read_feather(KRAKEN_DIR / f"{sym}_USD-1h.feather")
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    df = df.rename(columns={date_col: "date", "open": "Open", "high": "High",
                             "low": "Low", "close": "Close", "volume": "Volume"})
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_btc_binance() -> pd.DataFrame:
    df = pd.read_csv(BINANCE_BTC_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    hourly = df[["Open", "High", "Low", "Close", "Volume"]].resample("1h").agg(agg)
    return hourly.dropna(subset=["Open", "High", "Low", "Close"])


def load_symbol(sym: str) -> pd.DataFrame:
    return load_btc_binance() if sym == "BTC" else load_symbol_kraken(sym)


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
    """UNCHANGED from the frozen script -- entry = Close[t], monitoring from t+1."""
    close = df["Close"].to_numpy(np.float64)
    high = df["High"].to_numpy(np.float64)
    low = df["Low"].to_numpy(np.float64)
    atr_v = df["ATR"].to_numpy(np.float64)
    n = len(df)
    stop_d = atr_v * STOP_ATR; target_d = atr_v * TARGET_ATR
    if direction == "long":
        stop_p = close - stop_d; target_p = close + target_d
    else:
        stop_p = close + stop_d; target_p = close - target_d
    exit_p = np.full(n, np.nan); resolved = np.zeros(n, dtype=bool)
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
        use_stop = hit_stop
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


def triple_barrier_lagged(df, direction):
    """DIAGNOSTIC VARIANT (#1): the only change from triple_barrier() above is
    the execution reference price. Entry = Open[t+1] instead of Close[t].
    ATR used for stop/target SIZING is still ATR[t] (known at decision time
    t) -- only the price the trade is transacted at, and the bar from which
    forward monitoring starts, move one bar later. If this collapses PF, the
    frozen result was relying on same-bar (zero-latency) execution."""
    entry = df["Open"].shift(-1).to_numpy(np.float64)  # Open[t+1], aligned to row t
    high = df["High"].to_numpy(np.float64)
    low = df["Low"].to_numpy(np.float64)
    close = df["Close"].to_numpy(np.float64)
    atr_v = df["ATR"].to_numpy(np.float64)
    n = len(df)
    stop_d = atr_v * STOP_ATR; target_d = atr_v * TARGET_ATR
    if direction == "long":
        stop_p = entry - stop_d; target_p = entry + target_d
    else:
        stop_p = entry + stop_d; target_p = entry - target_d
    exit_p = np.full(n, np.nan); resolved = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    # entry bar is t+1; monitoring (post-entry) starts at t+2, i.e. offset>=2 from t
    for off in range(2, MAX_HOLD + 2):
        fwd = idx + off
        valid = fwd < n
        fh = np.full(n, np.nan); fl = np.full(n, np.nan)
        fh[valid] = high[fwd[valid]]; fl[valid] = low[fwd[valid]]
        if direction == "long":
            hit_stop = fl <= stop_p; hit_target = fh >= target_p
        else:
            hit_stop = fh >= stop_p; hit_target = fl <= target_p
        use_stop = hit_stop
        use_target = hit_target & ~use_stop
        newly = ~resolved & valid & (use_stop | use_target)
        exit_p[newly] = np.where(use_stop[newly], stop_p[newly], target_p[newly])
        resolved |= newly
    time_idx = idx + MAX_HOLD + 1  # MAX_HOLD bars after the entry bar (t+1)
    valid_t = time_idx < n
    time_exit = ~resolved & valid_t
    exit_p[time_exit] = close[time_idx[time_exit]]
    resolved |= time_exit
    valid_entry = ~np.isnan(entry)
    raw = (exit_p - entry) / entry
    if direction == "short":
        raw = -raw
    net = raw - ROUND_TRIP_COST
    label = np.where(net > 0, 1.0, 0.0)
    label = np.where(resolved & valid_entry, label, np.nan)
    net = np.where(valid_entry, net, np.nan)
    return label, net


def pf_of(returns):
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    g = returns[returns > 0].sum(); l = -returns[returns < 0].sum()
    return float(g / l) if l > 0 else (float("inf") if g > 0 else 0.0)


def scan_threshold(votes, mean_p, net, min_trades):
    best_thr, best_pf, best_n = None, -1.0, 0
    for thr in np.arange(0.05, 0.96, 0.01):
        mask = (votes >= AGREE_MIN) & (mean_p >= thr) & ~np.isnan(net)
        if mask.sum() < min_trades:
            continue
        sel = net[mask]
        pf = pf_of(sel)
        if sel.sum() > 0 and pf > best_pf:
            best_pf, best_thr, best_n = pf, thr, int(mask.sum())
    return best_thr, best_pf, best_n


def three_window_report(trades: pd.DataFrame, label: str) -> dict:
    r_all = trades["trade_return"].to_numpy(np.float64)
    folds = np.array_split(np.arange(len(trades)), 3)
    windows = []
    for i, fidx in enumerate(folds, start=1):
        sub = trades.iloc[fidx]
        r = sub["trade_return"].to_numpy(np.float64)
        pf = pf_of(r); ret_pct = r.sum() * 100
        lo, hi = (sub["timestamp"].min(), sub["timestamp"].max()) if len(sub) else (None, None)
        overlap = (lo is not None) and (lo <= JUNE_END) and (hi >= JUNE_START)
        windows.append(dict(window=i, trades=len(sub), return_pct=float(ret_pct), pf=pf,
                             start=str(lo), end=str(hi), june2026_overlap=bool(overlap)))
    june_mask = (trades["timestamp"] >= JUNE_START) & (trades["timestamp"] <= JUNE_END)
    june_r = trades.loc[june_mask, "trade_return"].to_numpy(np.float64)
    by_symbol = trades.groupby("symbol")["trade_return"].agg(["count", "sum"]).sort_values("sum")
    report = dict(
        label=label, trades=len(trades), return_pct=float(r_all.sum() * 100), pf=pf_of(r_all),
        windows=windows,
        june_2026_standalone=dict(trades=int(june_mask.sum()),
                                   return_pct=float(june_r.sum() * 100) if len(june_r) else 0.0,
                                   pf=pf_of(june_r)),
        per_symbol=[dict(symbol=s, n=int(row["count"]), sum_return_pct=float(row["sum"] * 100))
                    for s, row in by_symbol.iterrows()],
    )
    print(f"\n--- {label} ---")
    print(f"  trades={report['trades']} return_pct={report['return_pct']:.1f} pf={report['pf']:.3f}")
    for w in windows:
        print(f"    window {w['window']}: trades={w['trades']} return_pct={w['return_pct']:.1f} "
              f"pf={w['pf']:.3f} [{w['start']} -> {w['end']}]{'  <-- June2026' if w['june2026_overlap'] else ''}")
    print(f"  June 2026 standalone: trades={report['june_2026_standalone']['trades']} "
          f"return_pct={report['june_2026_standalone']['return_pct']:.1f} pf={report['june_2026_standalone']['pf']:.3f}")
    for s in report["per_symbol"]:
        print(f"    {s['symbol']}: n={s['n']} sum_return={s['sum_return_pct']:.1f}%")
    return report


def train_ensemble(X_train, y_train, X_val, y_val, X_cal, X_hold, seed_offset, label):
    """Shared training loop, byte-identical logic to run_lightgbm_multibot.py's
    long-direction training block. seed_offset shifts the RNG stream so the
    shuffled-label variant doesn't reuse the exact same bootstrap draws as
    the frozen run (deliberate -- it's a different experiment, not a replay)."""
    mask_tr = ~np.isnan(y_train); mask_val = ~np.isnan(y_val)
    Xtr, ytr = X_train[mask_tr], y_train[mask_tr]
    Xv, yv = X_val[mask_val], y_val[mask_val]

    cal_probs, hold_probs, base_rates = [], [], []
    for i in range(N_ENSEMBLE):
        seed = SEED_BASE + i + seed_offset
        rng = np.random.default_rng(seed)
        boot_idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)
        boot_X, boot_y = Xtr[boot_idx], ytr[boot_idx]

        clf = lgb.LGBMClassifier(random_state=seed,
                                  min_child_samples=max(30, len(boot_y) // 500),
                                  **LGB_PARAMS)
        clf.fit(boot_X, boot_y, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])

        cal_p = clf.predict_proba(X_cal)[:, 1]
        hold_p = clf.predict_proba(X_hold)[:, 1]
        cal_probs.append(cal_p); hold_probs.append(hold_p)
        base_rates.append(float(boot_y.mean()))
        print(f"  [{label}] member {i}: base_rate={base_rates[-1]:.4f} "
              f"best_iteration={clf.best_iteration_ or LGB_PARAMS['n_estimators']}")

    cal_probs = np.stack(cal_probs, axis=0); hold_probs = np.stack(hold_probs, axis=0)
    base_rates = np.array(base_rates)
    cal_votes = (cal_probs > base_rates[:, None]).sum(axis=0)
    cal_p_mean = cal_probs.mean(axis=0)
    hold_votes = (hold_probs > base_rates[:, None]).sum(axis=0)
    hold_p_mean = hold_probs.mean(axis=0)
    return dict(cal_votes=cal_votes, cal_p_mean=cal_p_mean, hold_votes=hold_votes, hold_p_mean=hold_p_mean)


def fire_trades(hold_df, votes, p_mean, threshold, net_col):
    mask = (votes >= AGREE_MIN) & (p_mean >= threshold)
    out = hold_df.copy()
    out["trade_return"] = np.where(mask, out[net_col], np.nan)
    return out[mask].dropna(subset=["trade_return"]).reset_index(drop=True)


def main():
    print("=== loading data (frozen + lagged labels) ===")
    frames = []
    for sym in SYMBOLS:
        raw = load_symbol(sym)
        feat = build_features(raw)
        lab, net = triple_barrier(feat, "long")
        feat["label_long"] = lab; feat["net_long"] = net
        lab_lag, net_lag = triple_barrier_lagged(feat, "long")
        feat["label_long_lag"] = lab_lag; feat["net_long_lag"] = net_lag
        feat["symbol"] = sym
        feat = feat.iloc[SEQUENCE_WARMUP:].reset_index()
        feat = feat.rename(columns={"date": "timestamp", "index": "timestamp"})
        feat = feat.dropna(subset=FEATURE_COLS)
        frames.append(feat)
        print(f"  {sym}: {len(feat)} usable rows, {feat['timestamp'].min()} -> {feat['timestamp'].max()}")

    pooled = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    n = len(pooled)
    train_end = int(n * 0.50); val_end = int(n * 0.60); cal_end = int(n * 0.70)
    train = pooled.iloc[:train_end]; val = pooled.iloc[train_end:val_end]
    cal = pooled.iloc[val_end:cal_end]; hold = pooled.iloc[cal_end:].reset_index(drop=True)
    print(f"\npooled={n} train={len(train)} val={len(val)} cal={len(cal)} holdout={len(hold)}")
    print(f"holdout date range: {hold['timestamp'].min()} -> {hold['timestamp'].max()}")

    X_train = train[FEATURE_COLS].to_numpy(np.float64)
    X_val = val[FEATURE_COLS].to_numpy(np.float64)
    X_cal = cal[FEATURE_COLS].to_numpy(np.float64)
    X_hold = hold[FEATURE_COLS].to_numpy(np.float64)

    reports = {}

    # ---------- 0. FROZEN reproduction (long side only) ----------
    print("\n=== [0] FROZEN long ensemble (reproduction check) ===")
    y_train = train["label_long"].to_numpy(np.float64)
    y_val = val["label_long"].to_numpy(np.float64)
    frozen = train_ensemble(X_train, y_train, X_val, y_val, X_cal, X_hold, seed_offset=0, label="frozen")
    cal_net = cal["net_long"].to_numpy(np.float64)
    thr, cal_pf, n_fire = scan_threshold(frozen["cal_votes"], frozen["cal_p_mean"], cal_net, MIN_CAL_TRADES)
    if thr is None:
        thr = 1.0
    print(f"[0] selected threshold={thr:.2f} cal_pf={cal_pf:.3f} cal_trades={n_fire} "
          f"(frozen run reported threshold=0.35 cal_pf=5.232 cal_trades=35 -- compare)")
    trades_frozen = fire_trades(hold, frozen["hold_votes"], frozen["hold_p_mean"], thr, "net_long")
    reports["0_frozen_reproduction"] = three_window_report(trades_frozen, "[0] FROZEN (Close[t] entry, calibrated threshold)")

    # ---------- 4. vote-only, no threshold (reuses frozen ensemble, no retrain) ----------
    trades_vote_only = fire_trades(hold, frozen["hold_votes"], frozen["hold_p_mean"], 0.0, "net_long")
    reports["4_vote_only_no_threshold"] = three_window_report(trades_vote_only, "[4] VOTE-ONLY (same ensemble, no probability threshold)")

    # ---------- 1. entry-lag test ----------
    print("\n=== [1] ENTRY-LAG long ensemble (Open[t+1] execution) ===")
    y_train_lag = train["label_long_lag"].to_numpy(np.float64)
    y_val_lag = val["label_long_lag"].to_numpy(np.float64)
    lagged = train_ensemble(X_train, y_train_lag, X_val, y_val_lag, X_cal, X_hold, seed_offset=500, label="lagged")
    cal_net_lag = cal["net_long_lag"].to_numpy(np.float64)
    thr_lag, cal_pf_lag, n_fire_lag = scan_threshold(lagged["cal_votes"], lagged["cal_p_mean"], cal_net_lag, MIN_CAL_TRADES)
    if thr_lag is None:
        thr_lag = 1.0
    print(f"[1] selected threshold={thr_lag:.2f} cal_pf={cal_pf_lag:.3f} cal_trades={n_fire_lag}")
    trades_lag = fire_trades(hold, lagged["hold_votes"], lagged["hold_p_mean"], thr_lag, "net_long_lag")
    reports["1_entry_lag"] = three_window_report(trades_lag, "[1] ENTRY-LAG (Open[t+1] entry, one bar later)")

    # ---------- 2. shuffled-label null ----------
    print("\n=== [2] SHUFFLED-LABEL null ensemble ===")
    rng_shuf = np.random.default_rng(777)
    y_train_shuf = y_train.copy()
    not_nan = ~np.isnan(y_train_shuf)
    perm_vals = y_train_shuf[not_nan].copy()
    rng_shuf.shuffle(perm_vals)
    y_train_shuf[not_nan] = perm_vals
    null_ens = train_ensemble(X_train, y_train_shuf, X_val, y_val, X_cal, X_hold, seed_offset=9000, label="shuffled")
    thr_null, cal_pf_null, n_fire_null = scan_threshold(null_ens["cal_votes"], null_ens["cal_p_mean"], cal_net, MIN_CAL_TRADES)
    if thr_null is None:
        thr_null = 1.0
    print(f"[2] selected threshold={thr_null:.2f} cal_pf={cal_pf_null:.3f} cal_trades={n_fire_null}")
    trades_null = fire_trades(hold, null_ens["hold_votes"], null_ens["hold_p_mean"], thr_null, "net_long")
    reports["2_shuffled_label_null"] = three_window_report(trades_null, "[2] SHUFFLED-LABEL NULL (same pipeline, scrambled y_train)")

    # ---------- 3. simple baselines (no ML) ----------
    print("\n=== [3] simple baselines (no ML, same holdout, same costs) ===")
    mom_mask = (hold["momentum24"] > 0) & (hold["ma50_dist"] > 0)
    trades_mom = hold[mom_mask].copy()
    trades_mom["trade_return"] = trades_mom["net_long"]
    trades_mom = trades_mom.dropna(subset=["trade_return"]).reset_index(drop=True)
    reports["3a_momentum_baseline"] = three_window_report(trades_mom, "[3a] MOMENTUM baseline (momentum24>0 & ma50_dist>0)")

    mr_mask = hold["rsi"] < 30
    trades_mr = hold[mr_mask].copy()
    trades_mr["trade_return"] = trades_mr["net_long"]
    trades_mr = trades_mr.dropna(subset=["trade_return"]).reset_index(drop=True)
    reports["3b_mean_reversion_baseline"] = three_window_report(trades_mr, "[3b] MEAN-REVERSION baseline (rsi<30)")

    print("\n  buy_and_hold per symbol over holdout window:")
    bh = {}
    for sym in SYMBOLS:
        sub = hold[hold["symbol"] == sym]
        if len(sub) < 2:
            continue
        ret = float(sub["Close"].iloc[-1] / sub["Close"].iloc[0] - 1.0) * 100
        bh[sym] = ret
        print(f"    {sym}: buy&hold return over holdout = {ret:.1f}%")
    reports["3c_buy_and_hold"] = bh

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(reports, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print("\n=== SUMMARY TABLE ===")
    print(f"{'variant':45s} {'trades':>7s} {'return_pct':>11s} {'pf':>8s}")
    for key in ["0_frozen_reproduction", "4_vote_only_no_threshold", "1_entry_lag",
                "2_shuffled_label_null", "3a_momentum_baseline", "3b_mean_reversion_baseline"]:
        r = reports[key]
        print(f"{r['label']:45s} {r['trades']:7d} {r['return_pct']:11.1f} {r['pf']:8.3f}")


if __name__ == "__main__":
    main()
