"""After-fee P&L test of the bakeoff's lightweight models on the same Kraken BTC/USD data.

Stand-ins (PyPI blocked in this sandbox):
  LightGBM  -> sklearn HistGradientBoostingRegressor (same histogram-GBM algorithm), same lags/params
  ARIMA(1,0,1) -> exact-likelihood-free CSS fit via scipy, one-step-ahead with true past values
  Prophet + order-flow (no seasonality) -> linear trend + linear regressor via least squares
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

FEE_ONE_WAY = 0.0012  # 12bp per side = 24bp round trip (project convention)
LAGS = [1, 2, 3, 5, 10, 20]

trades = pd.read_pickle("/home/claude/btc_trades.pkl")
trades["dt"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
trades = trades.set_index("dt")
trades["signed"] = np.where(trades["side"] == "buy", trades["amount"], -trades["amount"])
trades["pv"] = trades["price"] * trades["amount"]


def bars(tr, tf):
    g = tr.resample(tf)
    close = g["price"].last().ffill()
    vwap = (g["pv"].sum() / g["amount"].sum()).ffill()
    flow = g["signed"].sum().fillna(0.0)
    return close, vwap, flow


# ---------------- models (each returns one-step-ahead forecasts for holdout) ----------------
def arima_css(y, train_end):
    s = 1e4
    ytr = y[:train_end] * s

    def resid(params, series):
        c, phi, th = params
        e = np.zeros(len(series))
        for t in range(1, len(series)):
            e[t] = series[t] - c - phi * series[t - 1] - th * e[t - 1]
        return e

    def css(p):
        return np.sum(resid(p, ytr)[1:] ** 2)

    p = minimize(css, [0.0, 0.1, -0.1], method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-9}).x
    c, phi, th = p
    ys = y * s
    e = resid(p, ys)
    fc = c + phi * ys[train_end - 1:-1] + th * e[train_end - 1:-1]
    return fc / s, dict(c=c, phi=phi, theta=th)


def lag_matrix(y, extra=None):
    m = max(LAGS)
    X = np.column_stack([y[m - l:len(y) - l] for l in LAGS])
    if extra is not None:
        X = np.column_stack([X] + [extra[m - l:len(y) - l] for l in LAGS])
    return X, m


def gbm(y, train_end, extra=None):
    X, m = lag_matrix(y, extra)
    tgt = y[m:]
    split = train_end - m
    mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0,
                                        early_stopping=False)
    mdl.fit(X[:split], tgt[:split])
    return mdl.predict(X[split:])


def linear_flow(y, train_end, reg):
    t = np.arange(len(y), dtype=float) / len(y)
    A = np.column_stack([np.ones(len(y)), t, reg])
    beta, *_ = np.linalg.lstsq(A[:train_end], y[:train_end], rcond=None)
    return A[train_end:] @ beta


# ---------------- scoring ----------------
def score(name, fc, actual, extra=None):
    mask = (fc != 0) & (actual != 0)
    acc = float((np.sign(fc[mask]) == np.sign(actual[mask])).mean())
    ret = np.expm1(actual)

    def sim(pos):
        gross = pos * ret
        turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
        costs = turnover * FEE_ONE_WAY
        net = gross - costs
        entries = int(np.sum((turnover > 0) & (pos != 0)))
        return dict(gross_pct=float(gross.sum() * 100), fees_pct=float(costs.sum() * 100),
                    net_pct=float(net.sum() * 100), trades=entries,
                    net_bp_per_trade=float(net.sum() / max(entries, 1) * 1e4))

    ls = sim(np.sign(fc))                                      # long/short every bar
    lo = sim((fc > 0).astype(float))                           # long-only (Kraken spot)
    thr = 2 * FEE_ONE_WAY                                      # only trade if forecast > round-trip cost
    conf = sim(np.where(np.abs(fc) > thr, np.sign(fc), 0.0))
    # top-decile confidence
    q = np.quantile(np.abs(fc), 0.9)
    top = sim(np.where(np.abs(fc) >= q, np.sign(fc), 0.0))
    return dict(model=name, dir_acc=round(acc * 100, 2), n=int(mask.sum()),
                long_short=ls, long_only=lo, only_if_forecast_beats_fee=conf, top10pct_confidence=top,
                **(extra or {}))


def run_tf(label, close, vwap, flow, train_frac=0.8):
    out = []
    for price_name, price in (("last-trade close", close), ("VWAP", vwap)):
        y = np.log(price / price.shift(1)).dropna()
        fl = flow.reindex(y.index).fillna(0.0).to_numpy()
        fl_lag = np.concatenate([[0.0], fl[:-1]])
        yv = y.to_numpy()
        te = int(len(yv) * train_frac)
        act = yv[te:]
        a_fc, a_p = arima_css(yv, te)
        out.append(score(f"ARIMA(1,0,1)", a_fc, act, {"price": price_name, "tf": label, "params": a_p}))
        g_fc = gbm(yv, te)
        out.append(score("LightGBM-style GBM (lag features)", g_fc, act, {"price": price_name, "tf": label}))
        out.append(score("Prophet-style + SAME-BAR order flow (lookahead)", linear_flow(yv, te, fl), act,
                         {"price": price_name, "tf": label}))
        out.append(score("Prophet-style + LAGGED order flow", linear_flow(yv, te, fl_lag), act,
                         {"price": price_name, "tf": label}))
        out.append(score("LightGBM-style GBM + lagged order flow", gbm(yv, te, fl_lag), act,
                         {"price": price_name, "tf": label}))
        buyhold = float(np.expm1(act).sum() * 100)
        out.append({"model": "buy & hold (reference)", "tf": label, "price": price_name,
                    "net_pct": buyhold - 2 * FEE_ONE_WAY * 100})
    return out


results = []
# 30s: same as original bakeoff -- last 14 days of trades
recent = trades[trades.index >= trades.index.max() - pd.Timedelta(days=14)]
c, v, f = bars(recent, "30s")
results += run_tf("30s", c, v, f)
# 1h and 4h across full trade window
for tf in ("1h", "4h"):
    c, v, f = bars(trades, tf)
    results += run_tf(tf, c, v, f)

json.dump(results, open("/home/claude/pnl_results.json", "w"), indent=1, default=float)
for r in results:
    if "long_short" not in r:
        print(f"{r['tf']:>4} {r['price']:17s} {r['model']:48s} net={r['net_pct']:8.2f}%")
        continue
    print(f"{r['tf']:>4} {r['price']:17s} {r['model']:48s} acc={r['dir_acc']:6.2f}% n={r['n']:5d} | "
          f"L/S gross={r['long_short']['gross_pct']:8.2f}% net={r['long_short']['net_pct']:9.2f}% tr={r['long_short']['trades']:5d} | "
          f"LO net={r['long_only']['net_pct']:8.2f}% | "
          f">fee net={r['only_if_forecast_beats_fee']['net_pct']:7.2f}% tr={r['only_if_forecast_beats_fee']['trades']:4d} | "
          f"top10 net={r['top10pct_confidence']['net_pct']:8.2f}% tr={r['top10pct_confidence']['trades']:4d} "
          f"({r['top10pct_confidence']['net_bp_per_trade']:.1f}bp/tr)")
