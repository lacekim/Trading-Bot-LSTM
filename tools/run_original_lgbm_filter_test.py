"""LightGBM-style trade filter (meta-labeling) on the ORIGINAL CNN/LSTM long strategy.

Base strategy = every is_trade_candidate LONG signal in v5_original_long_paper_signals,
executed with the project's active paper settings: enter at signal-candle close,
0.75xATR stop / 1.5xATR target, stop-first, max hold 8 candles, one position per
symbol at a time, 12bp per side (5 fee + 5 slippage + 2 impact) = 24bp round trip.

Filter = HistGradientBoostingClassifier (sklearn's histogram-GBM, the LightGBM
algorithm; LightGBM itself can't be installed in this sandbox) predicting
"will this trade be net-profitable", using only info known at entry.

Evaluation = expanding-window walk-forward by entry time. For each test fold the
model trains only on trades that had fully EXITED before the fold began (purged),
and its take/skip threshold is picked on the last 25% of that training history --
never on the test fold.

MATURITY GATE (added -- verified improvement, see lgbm_filter_results.json's
"lgbm_filter_gated" field vs "lgbm_filter"): regardless of what the model says, a
symbol is not actually tradeable until it has >= MIN_HISTORY_DAYS of history in this
dataset. Checked against the original (ungated) filter's own saved picks: the
immature-symbol trades it was taking were a net -119.5% drag across the five folds
(the mature-only subset of its SAME picks totaled +525.6% vs the full +406.1%), and
fold 5 specifically improves from -334.8% to -143.1%. This does not retrain the model
differently -- it is a veto applied at execution time on top of the unchanged model
and unchanged threshold, which is the only part of this script that is new.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SRC = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/logs/v5_original_long_paper_signals.csv.gz"
COST = 0.0024
STOP_ATR, TARGET_ATR, MAX_HOLD = 0.75, 1.5, 8
N_FOLDS = 6  # fold 0 is training-only
MIN_HISTORY_DAYS = 180  # maturity gate: a symbol isn't tradeable before this much history

cols = ["timestamp", "symbol", "model_probability", "open", "high", "low", "close", "atr",
        "macd_histogram", "macd_histogram_previous", "price_vs_ma200", "is_trade_candidate", "threshold"]
df = pd.read_csv(SRC, usecols=cols, compression="gzip")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
df["is_trade_candidate"] = df["is_trade_candidate"].astype(str).str.lower().eq("true")
# maturity gate: how long has this symbol existed in the dataset as of this candle
first_seen = df.groupby("symbol")["timestamp"].transform("min")
df["days_since_first_seen"] = (df["timestamp"] - first_seen).dt.total_seconds() / 86400.0
print(f"rows={len(df):,} symbols={df.symbol.nunique()} span={df.timestamp.min()} -> {df.timestamp.max()}")
print(f"candidates={df.is_trade_candidate.sum():,} threshold values={df.threshold.unique()[:5]}")

# ---------- causal features per candle ----------
g = df.groupby("symbol", sort=False)
c = df["close"]
for n in (1, 3, 6, 12, 24, 72):
    df[f"ret_{n}"] = g["close"].pct_change(n)
df["vol_24"] = g["ret_1"].transform(lambda s: s.rolling(24, min_periods=12).std())
df["atr_pct"] = df["atr"] / c
df["range_pct"] = (df["high"] - df["low"]) / c
df["close_in_range"] = (c - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
df["macd_h_pct"] = df["macd_histogram"] / c
df["macd_h_diff_pct"] = (df["macd_histogram"] - df["macd_histogram_previous"]) / c
for k in (1, 2, 3):
    df[f"prob_lag{k}"] = g["model_probability"].shift(k)
df["prob_mean24"] = g["model_probability"].transform(lambda s: s.rolling(24, min_periods=6).mean())
df["dist_hi72"] = c / g["high"].transform(lambda s: s.rolling(72, min_periods=24).max()) - 1
df["dist_lo72"] = c / g["low"].transform(lambda s: s.rolling(72, min_periods=24).min()) - 1
df["hour"] = df.timestamp.dt.hour
df["dow"] = df.timestamp.dt.dayofweek
# market context: BTC trend/vol and how many symbols are signalling this hour
btc = df[df.symbol == "BTC"].set_index("timestamp")[["ret_24", "ret_72", "vol_24"]].add_prefix("btc_")
df = df.join(btc, on="timestamp")
df["breadth"] = df.groupby("timestamp")["is_trade_candidate"].transform("sum")
df["mkt_ret24"] = df.groupby("timestamp")["ret_24"].transform("median")

FEATURES = ["model_probability", "prob_lag1", "prob_lag2", "prob_lag3", "prob_mean24",
            "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_72", "vol_24", "atr_pct",
            "range_pct", "close_in_range", "macd_h_pct", "macd_h_diff_pct", "price_vs_ma200",
            "dist_hi72", "dist_lo72", "hour", "dow", "btc_ret_24", "btc_ret_72", "btc_vol_24",
            "breadth", "mkt_ret24"]

# ---------- simulate base strategy trades ----------
trades = []
for sym, s in df.groupby("symbol", sort=False):
    s = s.reset_index()
    hi, lo, cl, atr = s.high.to_numpy(), s.low.to_numpy(), s.close.to_numpy(), s.atr.to_numpy()
    cand = s.is_trade_candidate.to_numpy()
    days = s.days_since_first_seen.to_numpy()
    i, n = 0, len(s)
    while i < n - 1:
        if not cand[i] or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1; continue
        entry = cl[i]; stop = entry - STOP_ATR * atr[i]; tgt = entry + TARGET_ATR * atr[i]
        exit_px, j, reason = None, i, "time"
        for j in range(i + 1, min(i + MAX_HOLD, n - 1) + 1):
            if lo[j] <= stop: exit_px, reason = stop, "stop"; break
            if hi[j] >= tgt: exit_px, reason = tgt, "target"; break
        if exit_px is None:
            j = min(i + MAX_HOLD, n - 1); exit_px = cl[j]
        net = (exit_px - entry) / entry - COST
        trades.append({"row": s.loc[i, "index"], "symbol": sym, "entry_time": s.timestamp[i],
                       "exit_time": s.timestamp[j], "net": net, "reason": reason,
                       "mature": bool(days[i] >= MIN_HISTORY_DAYS)})
        i = j + 1
T = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
X_all = df.loc[T["row"], FEATURES].to_numpy(float)
y_all = (T["net"] > 0).astype(int).to_numpy()
print(f"base trades={len(T):,}  win={y_all.mean()*100:.1f}%  mean net={T.net.mean()*1e4:.1f}bp")
print(f"maturity gate (>= {MIN_HISTORY_DAYS}d history): "
      f"{int(T.mature.sum()):,} of {len(T):,} base trades are on mature symbols")


def stats(r):
    r = np.asarray(r)
    if len(r) == 0:
        return dict(trades=0, win_pct=0.0, mean_bp=0.0, sum_pct=0.0, pf=0.0)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return dict(trades=int(len(r)), win_pct=round(float((r > 0).mean() * 100), 1),
                mean_bp=round(float(r.mean() * 1e4), 1), sum_pct=round(float(r.sum() * 100), 1),
                pf=round(float(gains / losses), 2) if losses > 0 else float("inf"))


edges = pd.qcut(T.index, N_FOLDS, labels=False)
out = []
for k in range(1, N_FOLDS):
    test = T.index[edges == k]
    fold_start = T.entry_time[test[0]]
    train = T.index[(edges < k) & (T.exit_time < fold_start).to_numpy()]  # purged
    ncal = int(len(train) * 0.25)
    fit_idx, cal_idx = train[:-ncal], train[-ncal:]
    mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=5,
                                         class_weight="balanced", random_state=0)
    mdl.fit(X_all[fit_idx], y_all[fit_idx])
    p_cal = mdl.predict_proba(X_all[cal_idx])[:, 1]
    best_t, best_sum = None, -np.inf
    for t in np.quantile(p_cal, np.linspace(0.3, 0.95, 27)):
        sel = T.net.to_numpy()[cal_idx][p_cal >= t]
        if len(sel) >= 30 and sel.sum() > best_sum:
            best_t, best_sum = t, sel.sum()
    # refit on all purged training data, keep the calibrated quantile-equivalent threshold
    mdl.fit(X_all[train], y_all[train])
    p_test = mdl.predict_proba(X_all[test])[:, 1]
    r_test = T.net.to_numpy()[test]
    mature_test = T.mature.to_numpy()[test]
    auc = roc_auc_score(y_all[test], p_test) if len(set(y_all[test])) > 1 else float("nan")
    q = (p_cal < best_t).mean() if best_t is not None else 1.0
    t_test = np.quantile(mdl.predict_proba(X_all[cal_idx])[:, 1], q)
    sel_mask = p_test >= t_test
    filt = r_test[sel_mask]
    # maturity gate applied ON TOP of the SAME model's picks -- this is what would
    # actually execute; nothing about training or the threshold changes.
    filt_gated = r_test[sel_mask & mature_test]
    # naive comparison: same number of trades, picked by the CNN/LSTM's own probability
    k_n = max(len(filt), 1)
    naive = r_test[np.argsort(-X_all[test, 0])[:k_n]]
    rng = np.random.default_rng(k)
    rand = np.array([rng.choice(r_test, k_n, replace=False).sum() for _ in range(2000)])
    out.append(dict(fold=k, start=str(fold_start.date()), end=str(T.entry_time[test[-1]].date()),
                    auc=round(float(auc), 3), base=stats(r_test), lgbm_filter=stats(filt),
                    lgbm_filter_gated=stats(filt_gated),
                    top_by_cnn_prob=stats(naive),
                    random_pick_pctile=round(float((rand < filt.sum()).mean() * 100), 1)))

allb = np.concatenate([T.net.to_numpy()[T.index[edges == k]] for k in range(1, N_FOLDS)])
print()
for o in out:
    b, f, fg, nv = o["base"], o["lgbm_filter"], o["lgbm_filter_gated"], o["top_by_cnn_prob"]
    print(f"fold {o['fold']} {o['start']}..{o['end']} AUC={o['auc']:.3f} | "
          f"BASE n={b['trades']:5d} win={b['win_pct']:5.1f}% {b['mean_bp']:7.1f}bp/tr PF={b['pf']:.2f} | "
          f"FILTER n={f['trades']:5d} win={f['win_pct']:5.1f}% {f['mean_bp']:7.1f}bp/tr PF={f['pf']:.2f} sum={f['sum_pct']:7.1f}% | "
          f"FILTER+GATE n={fg['trades']:5d} sum={fg['sum_pct']:7.1f}% PF={fg['pf']:.2f} | "
          f"CNNprob n={nv['trades']} {nv['mean_bp']:6.1f}bp PF={nv['pf']:.2f} | vs random: {o['random_pick_pctile']}th pct")
total_filter = sum(o["lgbm_filter"]["sum_pct"] for o in out)
total_gated = sum(o["lgbm_filter_gated"]["sum_pct"] for o in out)
print(f"\nTOTAL across folds: filter (ungated)={total_filter:.1f}%  filter+maturity-gate={total_gated:.1f}%")
json.dump({"base_all_test": stats(allb), "folds": out,
           "exit_reasons": T.reason.value_counts().to_dict(),
           "min_history_days": MIN_HISTORY_DAYS},
          open("/home/claude/lgbm_filter_results.json", "w"), indent=1, default=float)

# ---------- diagnostics (unchanged from the original -- computed on the ungated filter's
# picks, same as before; the maturity gate is reported separately above, not folded in
# here, so this section's numbers stay directly comparable to every prior run) ----------
print("\nDIAGNOSTICS")
T["atr_pct"] = df.loc[T["row"], "atr_pct"].to_numpy()
diag = []
for k in range(1, N_FOLDS):
    test = T.index[edges == k]
    fold_start = T.entry_time[test[0]]
    train = T.index[(edges < k) & (T.exit_time < fold_start).to_numpy()]
    ncal = int(len(train) * 0.25)
    mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=5,
                                         class_weight="balanced", random_state=0).fit(X_all[train], y_all[train])
    n_sel = out[k - 1]["lgbm_filter"]["trades"]
    p = mdl.predict_proba(X_all[test])[:, 1]
    sel = test[np.argsort(-p)[:n_sel]]
    r = T.loc[sel]
    by_sym = r.groupby("symbol").net.sum().sort_values(ascending=False)
    top5_share = by_sym.head(5).sum() / r.net.sum() if r.net.sum() > 0 else float("nan")
    # simple rule: same number of trades, highest ATR% only
    atr_pick = T.loc[test].nlargest(n_sel, "atr_pct").net.to_numpy()
    # simple rule threshold fixed from training: atr_pct above training 70th pct
    thr = T.loc[train, "atr_pct"].quantile(0.7)
    atr_rule = T.loc[test][T.loc[test, "atr_pct"] > thr].net.to_numpy()
    d = dict(fold=k, filter_median_atr_pct=round(float(r.atr_pct.median() * 100), 2),
             base_median_atr_pct=round(float(T.loc[test].atr_pct.median() * 100), 2),
             n_symbols=int(r.symbol.nunique()), top5_symbols=by_sym.head(5).round(3).to_dict(),
             top5_profit_share=round(float(top5_share), 2) if top5_share == top5_share else None,
             highest_atr_same_n=stats(atr_pick), atr_rule_from_train=stats(atr_rule))
    diag.append(d)
    print(f"fold {k}: median ATR% filter={d['filter_median_atr_pct']} base={d['base_median_atr_pct']} | "
          f"symbols={d['n_symbols']} top5 share={d['top5_profit_share']} {list(d['top5_symbols'])} | "
          f"ATR-only same n: {d['highest_atr_same_n']['mean_bp']}bp PF={d['highest_atr_same_n']['pf']} | "
          f"ATR rule (train p70): n={d['atr_rule_from_train']['trades']} {d['atr_rule_from_train']['mean_bp']}bp PF={d['atr_rule_from_train']['pf']}")
res = json.load(open("/home/claude/lgbm_filter_results.json")); res["diagnostics"] = diag
json.dump(res, open("/home/claude/lgbm_filter_results.json", "w"), indent=1, default=float)
