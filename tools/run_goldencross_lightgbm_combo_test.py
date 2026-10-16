"""Real test of combining the ONE signal that survived the 1h re-run
(tools/run_model_bakeoff_kraken_lowfreq.py) -- lag-only LightGBM, 54.05%
directional accuracy, p=0.0086 -- with the real golden-cross system, as
an entry filter.

Revision history / why this version looks the way it does:
  v1 used 100%-capital-per-trade compounding and concluded "PF/win-rate
  improved, ignore the total-return drop as a compounding artifact." That
  conclusion didn't survive scrutiny: the arithmetic ($126.06 baseline vs
  $15.46 filtered per starting dollar, ratio ~8.15x) shows the ~32 trades
  the filter removed collectively drove a huge share of the realized
  return, not a rounding effect -- and 100%-capital compounding is exactly
  the fragile, order-sensitive methodology this project's own
  run_ma200_cross_strategy.py already warns against for this reason. This
  version fixes both problems raised:
    1. Explicitly reports which trades got filtered out and whether
       they're concentrated outsized winners vs. genuinely bad trades
       (checkable, not asserted).
    2. Redoes the comparison under fixed-fractional sizing (2% of current
       capital per trade, matching this project's own established
       PER_ASSET_ALLOCATION_PCT convention for cross-asset backtests --
       golden cross has no stop-loss, so risk-based sizing isn't
       applicable here; capital-fraction sizing is the right analog), plus
       CAGR, Sharpe, and max drawdown -- not just profit factor -- since a
       PF move on 36 trades has no reported confidence interval and could
       be noise on its own.

Still the same real, no-lookahead expanding-window walk-forward LightGBM
(refit per calendar year on strictly-prior data) as v1.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from trading_bot_v4.signals.ma200_trend import compute_trend_state

BTC_1H_PATH = "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/BTC_USD-1h.feather"
FEE_BPS = 5
SLIP_BPS = 7
ENTRY_COST = (FEE_BPS + SLIP_BPS) / 10000
EXIT_COST = (FEE_BPS + SLIP_BPS) / 10000
STARTING_CAPITAL = 100_000.0
FIXED_FRACTIONAL_PCT = 2.0  # matches this project's own cross-asset PER_ASSET_ALLOCATION_PCT convention
LAGS = [1, 2, 3, 5, 10, 20]


def load_4h_and_1h() -> tuple[pd.Series, pd.Series]:
    h1 = pd.read_feather(BTC_1H_PATH)
    h1["date"] = pd.to_datetime(h1["date"], utc=True)
    h1 = h1.set_index("date")
    close_1h = h1["close"]
    close_4h = h1.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()["close"]
    return close_4h, close_1h


def expanding_window_lightgbm_predictions(close_1h: pd.Series) -> pd.Series:
    log_returns = np.log(close_1h / close_1h.shift(1)).dropna()
    years = sorted(log_returns.index.year.unique())
    values = log_returns.to_numpy()
    index = log_returns.index
    max_lag = max(LAGS)
    predicted_sign = pd.Series(index=log_returns.index, dtype=float)

    for year in years:
        train_mask = index.year < year
        predict_mask = index.year == year
        if train_mask.sum() < max_lag + 200:
            continue
        train_end = np.where(train_mask)[0].max() + 1
        X_train, y_train = [], []
        for i in range(max_lag, train_end):
            X_train.append([values[i - lag] for lag in LAGS])
            y_train.append(values[i])
        if len(X_train) < 200:
            continue
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, verbosity=-1, random_state=0)
        model.fit(np.array(X_train), np.array(y_train))

        predict_idx = np.where(predict_mask)[0]
        predict_idx = predict_idx[predict_idx >= max_lag]
        if len(predict_idx) == 0:
            continue
        X_pred = np.array([[values[i - lag] for lag in LAGS] for i in predict_idx])
        preds = model.predict(X_pred)
        predicted_sign.iloc[predict_idx] = np.sign(preds)

    return predicted_sign


def collect_all_trades(close_4h: pd.Series, filter_signal: pd.Series) -> pd.DataFrame:
    """Single pass over the REAL golden-cross entries: records every trade
    the baseline would take, tagged with whether the LightGBM filter would
    have agreed at that entry -- so baseline and filtered are guaranteed to
    be the exact same trades in the exact same sequence minus removals,
    not two independently-run backtests that could subtly diverge."""
    trend = compute_trend_state(close_4h)
    position = None
    trades = []

    for ts, row in trend.iterrows():
        if pd.isna(row["ma200"]):
            continue
        if position is None and row["cross_up"]:
            available = filter_signal.loc[:ts].dropna()
            filter_agrees = len(available) > 0 and available.iloc[-1] > 0
            fill = row["close"] * (1 + ENTRY_COST)
            position = {"entry_price": fill, "entry_time": ts, "filter_agrees": filter_agrees}
        elif position is not None and row["cross_down"]:
            exit_price = row["close"] * (1 - EXIT_COST)
            ret = (exit_price / position["entry_price"]) - 1
            trades.append({
                "entry_time": position["entry_time"], "exit_time": ts,
                "ret_pct": ret * 100, "filter_agrees": position["filter_agrees"],
            })
            position = None

    return pd.DataFrame(trades)


def simulate_fixed_fractional(trades: pd.DataFrame, label: str) -> dict:
    capital = STARTING_CAPITAL
    equity_curve = [capital]
    for _, t in trades.iterrows():
        stake = capital * FIXED_FRACTIONAL_PCT / 100
        pnl = stake * (t["ret_pct"] / 100)
        capital += pnl
        equity_curve.append(capital)

    equity = pd.Series(equity_curve)
    trade_returns = trades["ret_pct"].to_numpy() / 100 if len(trades) else np.array([])

    n_years = (trades["exit_time"].max() - trades["entry_time"].min()).days / 365.25 if len(trades) > 1 else np.nan
    cagr = ((capital / STARTING_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years and n_years > 0 else float("nan")

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min() * 100

    sharpe = (trade_returns.mean() / trade_returns.std() * np.sqrt(len(trade_returns))
              if len(trade_returns) > 1 and trade_returns.std() > 0 else float("nan"))

    wins = trades[trades["ret_pct"] > 0] if len(trades) else trades
    gross_win = (wins["ret_pct"] / 100 * STARTING_CAPITAL * FIXED_FRACTIONAL_PCT / 100).sum() if len(wins) else 0
    losses = trades[trades["ret_pct"] <= 0] if len(trades) else trades
    gross_loss = -(losses["ret_pct"] / 100 * STARTING_CAPITAL * FIXED_FRACTIONAL_PCT / 100).sum() if len(losses) else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print(f"\n=== {label} (fixed-fractional {FIXED_FRACTIONAL_PCT}% sizing) ===")
    if len(trades) == 0:
        print("no trades")
        return {}
    print(f"trades: {len(trades)}  win_rate: {len(wins)/len(trades)*100:.1f}%  profit_factor: {pf:.2f}")
    print(f"final_capital: ${capital:,.0f}  total_return: {(capital/STARTING_CAPITAL-1)*100:.1f}%  CAGR: {cagr:.1f}%")
    print(f"max_drawdown: {max_dd:.1f}%  trade-return Sharpe (per-trade, not annualized): {sharpe:.2f}")
    return {"trades": len(trades), "win_rate": len(wins)/len(trades)*100, "pf": pf,
            "total_return": (capital/STARTING_CAPITAL-1)*100, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe}


def main() -> None:
    close_4h, close_1h = load_4h_and_1h()
    print(f"4h bars: {len(close_4h)} ({close_4h.index.min().date()} to {close_4h.index.max().date()})")

    print("\nfitting expanding-window LightGBM (real walk-forward, refit per calendar year)...")
    filter_signal = expanding_window_lightgbm_predictions(close_1h)
    comparable_start = filter_signal.dropna().index.min()
    close_4h_comparable = close_4h[close_4h.index >= comparable_start]
    print(f"comparison window: {comparable_start.date()} to {close_4h_comparable.index.max().date()}")

    all_trades = collect_all_trades(close_4h_comparable, filter_signal)
    filtered_out = all_trades[~all_trades["filter_agrees"]]
    kept = all_trades[all_trades["filter_agrees"]]

    print(f"\n=== Removed-trade breakdown ({len(filtered_out)} of {len(all_trades)} baseline trades) ===")
    if len(filtered_out):
        print(filtered_out[["entry_time", "exit_time", "ret_pct"]].to_string(index=False))
        print(f"\nremoved trades: mean ret {filtered_out['ret_pct'].mean():+.2f}%  "
              f"median ret {filtered_out['ret_pct'].median():+.2f}%  "
              f"sum ret {filtered_out['ret_pct'].sum():+.2f}%  "
              f"biggest winner removed: {filtered_out['ret_pct'].max():+.2f}%  "
              f"count >+50%: {(filtered_out['ret_pct'] > 50).sum()}")
        print(f"kept trades:    mean ret {kept['ret_pct'].mean():+.2f}%  "
              f"median ret {kept['ret_pct'].median():+.2f}%  "
              f"biggest winner kept: {kept['ret_pct'].max():+.2f}%")
        # Is the removed set's mean return significantly different from the kept set's?
        if len(filtered_out) > 1 and len(kept) > 1:
            t_stat, p_val = stats.ttest_ind(filtered_out["ret_pct"], kept["ret_pct"], equal_var=False)
            print(f"Welch t-test (removed vs kept mean return): t={t_stat:.2f} p={p_val:.3f} "
                  f"({'not' if p_val > 0.05 else ''} significant at 0.05)")

    simulate_fixed_fractional(all_trades, "BASELINE: golden cross alone")
    simulate_fixed_fractional(kept, "FILTERED: golden cross + LightGBM 1h agreement")


if __name__ == "__main__":
    main()
