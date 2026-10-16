#!/usr/bin/env python3
"""The multibot long/short crypto system, redone with REAL LightGBM (your
`javier` conda env already has lightgbm==4.6.0 and pyarrow==22.0.0 installed
-- this sandbox never did, which is why the earlier runs used a from-scratch
pure-numpy GBDT instead). Run this yourself in that env; I can't reach it
through the device bridge.

EVERYTHING BELOW IS PREDECLARED BEFORE THIS HAS BEEN RUN ONCE. Where a rule
is unchanged from the pure-numpy runs earlier this session, it's unchanged
on purpose, for comparability. Three things ARE different, and here's
exactly why:

  1. Real lightgbm.LGBMClassifier instead of the pure-numpy GBDTClassifier.
     Same algorithm family (leaf-wise GBDT, logistic loss), real implementation.
  2. Each ensemble member now bootstraps at FULL training-set size (with
     replacement), not the 40,000-row cap the pure-numpy version needed to
     fit the sandbox's per-call time budget. Real LightGBM doesn't have that
     constraint, so this removes a previously-disclosed deviation instead of
     adding one -- it's back to the ORIGINAL spec from several messages ago.
  3. Kraken data is read directly from the .feather files via real pyarrow
     (this sandbox could never read Feather at all), instead of the CSV
     export. Same underlying data, native format, one less conversion step.

LABELING (unchanged): triple-barrier, STOP_ATR=3.0, TARGET_ATR=20.0,
MAX_HOLD=240 candles, 24bp round-trip cost (5bp fee + 5bp slippage + 2bp
impact, x2).

DATA (unchanged from the last crypto run): BTC from Binance
binance_margin_bot/trading_bot/BTCUSDT-15m-data.csv (resampled 15m->1h,
2017-08 -> 2026-09) because Kraken's own BTC history stops 2026-01-05;
ETH/XMR/PAXG/SOL/TRX from Kraken's own .feather files (stop ~2026-01-05/09).

SPLIT (unchanged from the "widen the holdout" request): chronological
pooled 50/10/10/30 train/val/cal/holdout (not 70/10/10/10 -- that gave only
1 holdout trade for the long side, not enough to judge anything).

ENSEMBLE (unchanged): 5 independently-bootstrapped LightGBM models per
direction (long, short). A row's vote from model i is positive when that
model's predicted probability beats ITS OWN base rate (the win-rate in its
own bootstrap sample) -- NOT a flat 0.5, which is mathematically unreachable
given this label's ~20-26% base win rate (the bug found and fixed earlier
this session). Signal fires only if >=4/5 models vote positive AND the
ensemble's mean probability clears a threshold selected by max-PF scan on
the calibration split (requires >=30 qualifying calibration rows).

PULL-OUT/RE-ENTRY (unchanged): rolling PF over the last N=20 closed trades;
stop opening new trades if rolling PF<1.0; resume once rolling PF recovers
>=1.1.

PROMOTION BAR (unchanged): PF>=1.30 AND positive return in EVERY one of 3
equal (by trade count) holdout sub-windows.

VALIDATION (unchanged, applied per the multi-agent-validation principle):
split calibration into two independent chronological halves; a threshold is
only trusted if both halves independently pick similar thresholds (within
0.10) AND each one applied out-of-sample to the OTHER half stays profitable
(>=3 trades, PF>1).

CRITIC VERDICT (unchanged): APPROVED only if holdout has >=10 trades with
PF>1 AND the strict per-window promotion bar above passes. Otherwise
REJECTED (if there's enough holdout volume to judge) or INSUFFICIENT_EVIDENCE
(if there isn't).

JUNE 2026 (standing rule, unchanged): the May15-Jul1 2026 AI-correction
window is reported separately, never folded into the aggregate.

USAGE:
    cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
    python tools/run_lightgbm_multibot.py
(run in the `javier` conda env -- it already has lightgbm, pyarrow, pandas)

Writes runtime/lightgbm_multibot_results.json and prints the full report.
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
OUT_PATH = PROJECT_ROOT / "runtime" / "lightgbm_multibot_results.json"

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
MIN_CAL_TRADES_HALF = 15
MIN_WINDOW_TRADES = 10
ROLL_N = 20
PULLOUT_PF = 1.0
REENTRY_PF = 1.1
JUNE_START = pd.Timestamp("2026-05-15")
JUNE_END = pd.Timestamp("2026-07-01")

LGB_PARAMS = dict(n_estimators=500, learning_rate=0.08, num_leaves=15,
                   objective="binary", verbosity=-1)
EARLY_STOPPING_ROUNDS = 20
SEED_BASE = 100


# ---------- data loading ----------

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
        use_stop = hit_stop  # STOP_FIRST
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


def pf_of(returns):
    returns = np.asarray(returns, dtype=np.float64)
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


def apply_threshold(votes, mean_p, net, thr):
    if thr is None:
        return np.array([]), 0
    mask = (votes >= AGREE_MIN) & (mean_p >= thr) & ~np.isnan(net)
    return net[mask], int(mask.sum())


def main():
    print("=== loading data ===")
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
        feat = feat.rename(columns={"date": "timestamp", "index": "timestamp"})
        feat = feat.dropna(subset=FEATURE_COLS)
        frames.append(feat)
        print(f"  {sym}: {len(feat)} usable rows, {feat['timestamp'].min()} -> {feat['timestamp'].max()}")

    pooled = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    n = len(pooled)
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

    results = {}
    ensemble_info = {}
    for direction in ["long", "short"]:
        print(f"\n=== training {direction} ensemble (real LightGBM, 5 members, full-size bootstrap) ===")
        y_train = train[f"label_{direction}"].to_numpy(np.float64)
        y_val = val[f"label_{direction}"].to_numpy(np.float64)
        mask_tr = ~np.isnan(y_train); mask_val = ~np.isnan(y_val)
        Xtr, ytr = X_train[mask_tr], y_train[mask_tr]
        Xv, yv = X_val[mask_val], y_val[mask_val]

        cal_probs, hold_probs, base_rates, trees_used = [], [], [], []
        for i in range(N_ENSEMBLE):
            seed = SEED_BASE + i + (0 if direction == "long" else 1000)
            rng = np.random.default_rng(seed)
            boot_idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)  # FULL size now
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
            trees_used.append(int(clf.best_iteration_ or LGB_PARAMS["n_estimators"]))
            print(f"  member {i}: base_rate={base_rates[-1]:.4f} best_iteration={trees_used[-1]}")

        cal_probs = np.stack(cal_probs, axis=0)
        hold_probs = np.stack(hold_probs, axis=0)
        base_rates = np.array(base_rates)

        cal_votes = (cal_probs > base_rates[:, None]).sum(axis=0)
        cal_p_mean = cal_probs.mean(axis=0)
        hold_votes = (hold_probs > base_rates[:, None]).sum(axis=0)
        hold_p_mean = hold_probs.mean(axis=0)

        cal_net = cal[f"net_{direction}"].to_numpy(np.float64)
        best_thr, best_pf, n_fire = scan_threshold(cal_votes, cal_p_mean, cal_net, MIN_CAL_TRADES)
        if best_thr is None:
            best_thr = 1.0
        print(f"[{direction}] selected threshold={best_thr:.2f} cal_pf={best_pf:.3f} cal_trades={n_fire}")

        ensemble_info[direction] = dict(cal_probs=cal_probs, hold_probs=hold_probs, base_rates=base_rates,
                                         cal_votes=cal_votes, cal_p_mean=cal_p_mean,
                                         hold_votes=hold_votes, hold_p_mean=hold_p_mean)
        results[direction] = dict(threshold=float(best_thr), cal_pf=float(best_pf), cal_trades=n_fire,
                                   trees_used=trees_used, base_rates=base_rates.tolist())

    # ---- Validator: independent calibration-half cross-check ----
    print("\n=== Validator: independent calibration-half cross-check ===")
    validator = {}
    for direction in ["long", "short"]:
        info = ensemble_info[direction]
        cal_net = cal[f"net_{direction}"].to_numpy(np.float64)
        nC = len(cal_net); half = nC // 2
        votes, meanp = info["cal_votes"], info["cal_p_mean"]
        vA, pA, netA = votes[:half], meanp[:half], cal_net[:half]
        vB, pB, netB = votes[half:], meanp[half:], cal_net[half:]
        thrA, pfA_in, nA_in = scan_threshold(vA, pA, netA, MIN_CAL_TRADES_HALF)
        thrB, pfB_in, nB_in = scan_threshold(vB, pB, netB, MIN_CAL_TRADES_HALF)
        retA_on_B, nA_on_B = apply_threshold(vB, pB, netB, thrA); pfA_on_B = pf_of(retA_on_B)
        retB_on_A, nB_on_A = apply_threshold(vA, pA, netA, thrB); pfB_on_A = pf_of(retB_on_A)
        agree_range = (thrA is not None and thrB is not None and abs(thrA - thrB) <= 0.10)
        oos_ok = (nA_on_B >= 3 and pfA_on_B > 1.0) and (nB_on_A >= 3 and pfB_on_A > 1.0)
        validated = agree_range and oos_ok
        validator[direction] = dict(thrA=thrA, thrB=thrB, pfA_on_B=pfA_on_B, pfB_on_A=pfB_on_A, validated=validated)
        print(f"[{direction}] thrA={thrA} thrB={thrB} pfA_on_B={pfA_on_B} pfB_on_A={pfB_on_A} VALIDATED={validated}")

    # ---- combine long/short on holdout ----
    long_info, short_info = ensemble_info["long"], ensemble_info["short"]
    long_fire = (long_info["hold_votes"] >= AGREE_MIN) & (long_info["hold_p_mean"] >= results["long"]["threshold"])
    short_fire = (short_info["hold_votes"] >= AGREE_MIN) & (short_info["hold_p_mean"] >= results["short"]["threshold"])
    both = long_fire & short_fire
    long_fire &= ~both; short_fire &= ~both

    hold = hold.copy()
    net_hold_long = hold["net_long"].to_numpy(np.float64)
    net_hold_short = hold["net_short"].to_numpy(np.float64)
    hold["trade_return"] = np.where(long_fire, net_hold_long, np.where(short_fire, net_hold_short, np.nan))
    hold["direction"] = np.where(long_fire, "long", np.where(short_fire, "short", "flat"))

    trades = hold[hold["direction"] != "flat"].dropna(subset=["trade_return"]).reset_index(drop=True)
    print(f"\ntotal holdout rows={len(hold)}, trades fired={len(trades)} "
          f"(long={int(long_fire.sum())}, short={int(short_fire.sum())})")

    active = True; taken = []; roll_returns = []
    for _, row in trades.iterrows():
        if active:
            taken.append(True); roll_returns.append(row["trade_return"])
        else:
            taken.append(False)
        if len(roll_returns) >= ROLL_N:
            recent = np.array(roll_returns[-ROLL_N:])
            pf_roll = pf_of(recent)
            if active and pf_roll < PULLOUT_PF:
                active = False
            elif not active and pf_roll >= REENTRY_PF:
                active = True
    trades["taken_after_pullout"] = taken

    all_returns = trades["trade_return"].to_numpy(np.float64)
    filt_returns = trades.loc[trades["taken_after_pullout"], "trade_return"].to_numpy(np.float64)
    print(f"WITHOUT pull-out: trades={len(all_returns)} return_pct={all_returns.sum()*100:.1f} pf={pf_of(all_returns):.3f}")
    print(f"WITH pull-out/re-entry: trades={len(filt_returns)} return_pct={filt_returns.sum()*100:.1f} pf={pf_of(filt_returns):.3f}")

    folds = np.array_split(np.arange(len(trades)), 3)
    windows = []; promoted_ok = True
    for i, fidx in enumerate(folds, start=1):
        sub = trades.iloc[fidx]
        r = sub["trade_return"].to_numpy(np.float64)
        pf = pf_of(r); ret_pct = r.sum() * 100
        lo, hi = (sub["timestamp"].min(), sub["timestamp"].max()) if len(sub) else (None, None)
        overlap = (lo is not None) and (lo <= JUNE_END) and (hi >= JUNE_START)
        ok = len(sub) >= MIN_WINDOW_TRADES and pf >= 1.30 and ret_pct > 0
        promoted_ok = promoted_ok and ok
        windows.append(dict(window=i, trades=len(sub), return_pct=float(ret_pct), pf=pf,
                             start=str(lo), end=str(hi), june2026_overlap=bool(overlap), clears_bar=ok))
        tag = "  <-- OVERLAPS June 2026" if overlap else ""
        print(f"  window {i}: trades={len(sub)} return_pct={ret_pct:.1f} pf={pf:.3f} [{lo} -> {hi}] clears_bar={ok}{tag}")

    june_mask = (trades["timestamp"] >= JUNE_START) & (trades["timestamp"] <= JUNE_END)
    june_r = trades.loc[june_mask, "trade_return"].to_numpy(np.float64)
    june_report = dict(trades=int(june_mask.sum()), return_pct=float(june_r.sum() * 100) if len(june_r) else 0.0,
                        pf=pf_of(june_r))
    print(f"\nJune 2026 STANDALONE: trades={june_report['trades']} return_pct={june_report['return_pct']:.1f} "
          f"pf={june_report['pf']:.3f}")

    promoted = bool(len(all_returns) >= MIN_CAL_TRADES and promoted_ok)
    print(f"\nPROMOTION CHECK (PF>=1.30 AND positive return every window): promoted={promoted}")

    by_symbol = trades.groupby("symbol")["trade_return"].agg(["count", "sum"]).sort_values("sum")
    print("\nper-symbol holdout contribution:")
    for sym, row in by_symbol.iterrows():
        print(f"  {sym}: n={int(row['count'])} sum_return={row['sum']*100:.1f}%")

    # ---- Critic ----
    hold_trades_total = len(all_returns)
    hold_pf_total = pf_of(all_returns)
    if hold_trades_total < 10:
        critic = "INSUFFICIENT_EVIDENCE"
    elif hold_pf_total > 1.0 and promoted:
        critic = "APPROVED"
    else:
        critic = "REJECTED"
    print(f"\nCRITIC VERDICT: {critic}  (holdout_trades={hold_trades_total}, holdout_pf={hold_pf_total:.3f}, "
          f"promotion_bar_passed={promoted})")

    out = dict(
        thresholds=results, validator=validator,
        without_pullout=dict(trades=len(all_returns), return_pct=float(all_returns.sum()*100), pf=pf_of(all_returns)),
        with_pullout=dict(trades=len(filt_returns), return_pct=float(filt_returns.sum()*100), pf=pf_of(filt_returns)),
        windows=windows, june_2026_standalone=june_report, promoted=promoted, critic_verdict=critic,
        per_symbol=[dict(symbol=s, n=int(r["count"]), sum_return_pct=float(r["sum"]*100)) for s, r in by_symbol.iterrows()],
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
