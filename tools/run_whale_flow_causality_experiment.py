"""Tests the direct claim: "if a whale buys, up it goes, sells, down it
goes" -- and the related distribution hypothesis (whales sell into a rally
once retail/crowd activity is elevated, which precedes a reversal).

Uses the full real Kraken tick-level trade history (8M+ rows, 2017-2026),
not the ~11-month Santiment window. Three separate, honest questions, kept
separate because they are NOT the same claim:

  A) Contemporaneous: does whale flow correlate with price change in the
     SAME hour? Expected to be strongly positive almost mechanically -- a
     large market order moves the price it executes at, by definition. This
     is NOT a tradeable signal (you can't act on a bar after it's already
     closed), included only to confirm the data behaves as expected.
  B) Predictive: does whale flow in hour t predict the return in hour t+1
     (and beyond) -- i.e., is there real, tradeable follow-through after a
     whale trade, not just the trade's own immediate price impact?
  C) Distribution: during an established uptrend with elevated retail
     activity (trade count), does whale SELL flow cluster, and does that
     clustering predict a subsequent reversal? This is the actual
     "accumulate low, distribute into the crowd" mechanism, tested directly
     on real trades instead of a social sentiment proxy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc

TRADES_PATH = "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/BTC_USD-trades.feather"
WHALE_PERCENTILE = 99.5
N_FOLDS = 8


def build_hourly() -> pd.DataFrame:
    trades = pd.read_feather(TRADES_PATH)
    trades["dt"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    threshold = trades["cost"].quantile(WHALE_PERCENTILE / 100.0)
    whales = trades[trades["cost"] >= threshold].copy()
    whales["signed_cost"] = np.where(whales["side"] == "buy", whales["cost"], -whales["cost"])
    whales = whales.set_index("dt")

    whale_flow = whales["signed_cost"].resample("1h").sum().fillna(0.0)
    whale_buy_flow = whales[whales["signed_cost"] > 0]["signed_cost"].resample("1h").sum().fillna(0.0)
    whale_sell_flow = whales[whales["signed_cost"] < 0]["signed_cost"].resample("1h").sum().fillna(0.0)

    all_trade_count = trades.set_index("dt")["cost"].resample("1h").count().rename("trade_count")

    price = load_gmx_ohlc("BTC", "1h")
    price.index = pd.to_datetime(price.index, utc=True)
    close = price["Close"].astype(float)

    df = pd.DataFrame({"close": close}).join(whale_flow.rename("whale_flow"), how="left")
    df = df.join(whale_buy_flow.rename("whale_buy_flow"), how="left")
    df = df.join(whale_sell_flow.rename("whale_sell_flow"), how="left")
    df = df.join(all_trade_count, how="left")
    df[["whale_flow", "whale_buy_flow", "whale_sell_flow", "trade_count"]] = df[
        ["whale_flow", "whale_buy_flow", "whale_sell_flow", "trade_count"]
    ].fillna(0.0)

    df["ret_same_hour"] = np.log(df["close"] / df["close"].shift(1))
    for h in [1, 4, 24]:
        df[f"fwd_ret_{h}h"] = np.log(df["close"].shift(-h) / df["close"])

    df["ma20"] = df["close"].rolling(20).mean()
    df["trending_up"] = df["close"] > df["ma20"]
    df["trade_count_z"] = (
        (df["trade_count"] - df["trade_count"].rolling(168).mean())
        / (df["trade_count"].rolling(168).std() + 1e-9)
    )
    return df.dropna()


def main() -> None:
    df = build_hourly()
    print(f"total hourly bars: {len(df)} ({df.index.min()} to {df.index.max()})")

    print("\n=== A) Contemporaneous: whale_flow vs SAME-hour return (expected mechanical, not tradeable) ===")
    corr_same = df["whale_flow"].corr(df["ret_same_hour"])
    print(f"correlation: {corr_same:.4f}")

    print("\n=== B) Predictive: whale_flow[t] vs return at t+1h, t+4h, t+24h ===")
    for h in [1, 4, 24]:
        corr = df["whale_flow"].corr(df[f"fwd_ret_{h}h"])
        print(f"  +{h}h: correlation={corr:.4f}")

    print("\n  Directional accuracy: does whale_flow SIGN predict next-hour return SIGN?")
    n = len(df)
    train_end = int(n * 0.70)
    holdout = df.iloc[train_end:]
    signal = holdout[holdout["whale_flow"] != 0]
    predicted_up = signal["whale_flow"] > 0
    actual_up = signal["fwd_ret_1h"] > 0
    correct = int((predicted_up == actual_up).sum())
    total = len(signal)
    binom_p = stats.binomtest(correct, total, p=0.5, alternative="greater").pvalue
    print(f"  holdout: n={total} dir_acc={correct/total*100:.2f}% binom_p={binom_p:.4f}")

    fold_accs = []
    edges = np.linspace(0, len(signal), N_FOLDS + 1).astype(int)
    for fold in range(N_FOLDS):
        fold_data = signal.iloc[edges[fold]:edges[fold + 1]]
        if len(fold_data) < 20:
            continue
        pu = fold_data["whale_flow"] > 0
        au = fold_data["fwd_ret_1h"] > 0
        acc = (pu == au).mean() * 100
        fold_accs.append(acc)
        print(f"    fold ({fold_data.index.min().date()} to {fold_data.index.max().date()}): "
              f"n={len(fold_data)} dir_acc={acc:.2f}%")

    print("\n=== C) Distribution: whale sell-flow during an uptrend + elevated retail activity ===")
    context = df[(df["trending_up"]) & (df["trade_count_z"] > 1.0)]
    print(f"rows where price trending up AND trade_count_z > 1.0 (elevated activity): {len(context)}")

    mean_sell_flow_context = context["whale_sell_flow"].mean()
    mean_sell_flow_baseline = df[df["trending_up"]]["whale_sell_flow"].mean()
    print(f"mean whale sell-flow when trending up + high activity: ${mean_sell_flow_context:,.0f}")
    print(f"mean whale sell-flow when just trending up (baseline):  ${mean_sell_flow_baseline:,.0f}")

    has_distribution = context[context["whale_sell_flow"] < 0]
    print(f"\nof those {len(context)} rows, {len(has_distribution)} had net whale SELLING")
    for h in [1, 4, 24]:
        col = f"fwd_ret_{h}h"
        reversed_down = (has_distribution[col] < 0).sum()
        total = len(has_distribution)
        if total < 15:
            print(f"  +{h}h: too few rows ({total}), skipping")
            continue
        binom_p = stats.binomtest(int(reversed_down), total, p=0.5, alternative="greater").pvalue
        mean_ret = has_distribution[col].mean()
        print(f"  +{h}h after distribution: reversal_rate={reversed_down/total*100:.1f}% "
              f"(n={total}) binom_p={binom_p:.4f} mean_fwd_ret={mean_ret*100:.4f}%")


if __name__ == "__main__":
    main()
