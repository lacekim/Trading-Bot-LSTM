"""LightGBM trade selection with NO CNN/LSTM input anywhere (no model_probability,
no CNN signals). Same 117-asset 1h OHLC, same execution (0.75xATR stop, 1.5xATR
target, stop-first, 8-candle max hold, 24bp round trip), purged expanding walk-forward.
A) LightGBM filters a cheap rule: close > MA200 and MACD histogram crosses above 0.
B) LightGBM standalone: every candle is a candidate; takes its top-rated trades.
Pass bar (set before running): positive in all 5 test folds, PF > 1.3, >= 50 trades/fold."""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SRC = "/mnt/user-data/uploads/Javier_Santiago_Gaston_de_Iriarte_canrera/logs/v5_original_long_paper_signals.csv.gz"
COST, STOP_ATR, TARGET_ATR, MAX_HOLD, N_FOLDS = 0.0024, 0.75, 1.5, 8, 6
cols = ["timestamp", "symbol", "open", "high", "low", "close", "atr", "macd_histogram", "macd_histogram_previous"]
df = pd.read_csv(SRC, usecols=cols, compression="gzip")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
g = df.groupby("symbol", sort=False); c = df["close"]
for n in (1, 3, 6, 12, 24, 72): df[f"ret_{n}"] = g["close"].pct_change(n)
df["vol_24"] = g["ret_1"].transform(lambda s: s.rolling(24, min_periods=12).std())
df["ma200"] = g["close"].transform(lambda s: s.rolling(200, min_periods=200).mean())
df["px_vs_ma200"] = c / df["ma200"] - 1
df["px_vs_ma50"] = c / g["close"].transform(lambda s: s.rolling(50, min_periods=50).mean()) - 1
df["atr_pct"] = df["atr"] / c
df["range_pct"] = (df["high"] - df["low"]) / c
df["close_in_range"] = (c - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
df["macd_h_pct"] = df["macd_histogram"] / c
df["macd_h_diff_pct"] = (df["macd_histogram"] - df["macd_histogram_previous"]) / c
df["dist_hi72"] = c / g["high"].transform(lambda s: s.rolling(72, min_periods=24).max()) - 1
df["dist_lo72"] = c / g["low"].transform(lambda s: s.rolling(72, min_periods=24).min()) - 1
df["hour"] = df.timestamp.dt.hour; df["dow"] = df.timestamp.dt.dayofweek
btc = df[df.symbol == "BTC"].set_index("timestamp")[["ret_24", "ret_72", "vol_24", "px_vs_ma200"]].add_prefix("btc_")
df = df.join(btc, on="timestamp")
df["mkt_ret24"] = df.groupby("timestamp")["ret_24"].transform("median")
df["mkt_above_ma200"] = df.groupby("timestamp")["px_vs_ma200"].transform(lambda s: (s > 0).mean())
FEATURES = ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_72", "vol_24", "px_vs_ma200", "px_vs_ma50",
            "atr_pct", "range_pct", "close_in_range", "macd_h_pct", "macd_h_diff_pct", "dist_hi72", "dist_lo72",
            "hour", "dow", "btc_ret_24", "btc_ret_72", "btc_vol_24", "btc_px_vs_ma200", "mkt_ret24", "mkt_above_ma200"]

# triple-barrier outcome for EVERY candle (vectorised per symbol)
net = np.full(len(df), np.nan); exit_pos = np.full(len(df), -1)
for sym, idx in g.indices.items():
    hi, lo, cl, atr = (df[k].to_numpy()[idx] for k in ("high", "low", "close", "atr"))
    n = len(idx); stop = cl - STOP_ATR * atr; tgt = cl + TARGET_ATR * atr
    px = np.full(n, np.nan); ex = np.full(n, -1); done = np.zeros(n, bool)
    for o in range(1, MAX_HOLD + 1):
        j = np.arange(n) + o; v = j < n; jj = np.where(v, j, n - 1)
        hs = v & (lo[jj] <= stop); ht = v & (hi[jj] >= tgt) & ~hs
        new = ~done & (hs | ht)
        px[new] = np.where(hs[new], stop[new], tgt[new]); ex[new] = jj[new]; done |= new
    j = np.arange(n) + MAX_HOLD; t = ~done & (j < n)
    px[t] = cl[j[t]]; ex[t] = j[t]; done |= t
    ok = done & np.isfinite(atr) & (atr > 0)
    net[idx[ok]] = (px[ok] - cl[ok]) / cl[ok] - COST; exit_pos[idx[ok]] = idx[ex[ok]]
df["net"] = net
ok = exit_pos >= 0
ts = df["timestamp"]
df["exit_time"] = ts.iloc[np.where(ok, exit_pos, 0)].to_numpy()
df.loc[~ok, "exit_time"] = pd.NaT
rule = (df.px_vs_ma200 > 0) & (df.macd_histogram > 0) & (df.macd_histogram_previous <= 0)


def nonoverlap(rows):
    """one position per symbol at a time, chronological"""
    keep, busy = [], {}
    for r in rows.sort_values("timestamp").itertuples():
        if r.timestamp >= busy.get(r.symbol, pd.Timestamp.min.tz_localize("UTC")):
            keep.append(r.Index); busy[r.symbol] = r.exit_time
    return df.loc[keep]


def stats(r):
    r = np.asarray(r); 
    if len(r) == 0: return dict(trades=0, win_pct=0, mean_bp=0, sum_pct=0, pf=0)
    gn, ls = r[r > 0].sum(), -r[r < 0].sum()
    return dict(trades=int(len(r)), win_pct=round(float((r > 0).mean() * 100), 1), mean_bp=round(float(r.mean() * 1e4), 1),
                sum_pct=round(float(r.sum() * 100), 1), pf=round(float(gn / ls), 2) if ls > 0 else float("inf"))


def run(name, cand, keep_frac_grid, sample=None):
    C = cand[cand.net.notna()].sort_values("timestamp")
    if sample: C = C.iloc[::sample]
    edges = pd.qcut(np.arange(len(C)), N_FOLDS, labels=False)
    X = C[FEATURES].to_numpy(float); y = (C.net > 0).astype(int).to_numpy(); R = C.net.to_numpy()
    res = []
    for k in range(1, N_FOLDS):
        te = np.where(edges == k)[0]; start = C.timestamp.iloc[te[0]]
        tr = np.where((edges < k) & (C.exit_time < start).to_numpy())[0]
        ncal = len(tr) // 4; fit, cal = tr[:-ncal], tr[-ncal:]
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=5, class_weight="balanced", random_state=0)
        m.fit(X[fit], y[fit]); pc = m.predict_proba(X[cal])[:, 1]
        best_q, best = None, -np.inf
        for q in keep_frac_grid:
            t = np.quantile(pc, 1 - q); s = R[cal][pc >= t]
            if len(s) >= 30 and s.sum() > best: best_q, best = q, s.sum()
        m.fit(X[tr], y[tr])
        t = np.quantile(m.predict_proba(X[cal])[:, 1], 1 - best_q)
        p = m.predict_proba(X[te])[:, 1]
        base = nonoverlap(C.iloc[te]).net.to_numpy()
        pick = nonoverlap(C.iloc[te[p >= t]]).net.to_numpy()
        auc = roc_auc_score(y[te], p) if len(set(y[te])) > 1 else float("nan")
        res.append(dict(fold=k, start=str(start.date()), end=str(C.timestamp.iloc[te[-1]].date()), auc=round(float(auc), 3),
                        kept_frac=round(float(best_q), 3), base=stats(base), lgbm=stats(pick)))
    ok_all = all(r["lgbm"]["sum_pct"] > 0 and r["lgbm"]["pf"] > 1.3 and r["lgbm"]["trades"] >= 50 for r in res)
    print(f"\n=== {name}  PASS={ok_all}")
    for r in res:
        b, l = r["base"], r["lgbm"]
        print(f"fold {r['fold']} {r['start']}..{r['end']} AUC={r['auc']:.3f} | base n={b['trades']:6d} {b['mean_bp']:7.1f}bp PF={b['pf']:.2f} | "
              f"LGBM n={l['trades']:5d} win={l['win_pct']:5.1f}% {l['mean_bp']:7.1f}bp PF={l['pf']:.2f} sum={l['sum_pct']:7.1f}%")
    return dict(name=name, passed=ok_all, folds=res)


out = [run("A: LightGBM filtering MA200+MACD-cross rule (no CNN)", df[rule], np.linspace(0.05, 0.7, 27)),
       run("B: LightGBM standalone, every candle (no CNN)", df, np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1]), sample=4)]
json.dump(out, open("/home/claude/lgbm_no_cnn_results.json", "w"), indent=1, default=float)
