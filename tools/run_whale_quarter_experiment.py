"""Tests the asymmetric approach described directly by the user: whale trades
as the entry trigger, quarter as context, low win rate expected and accepted
because the payoff structure is wide-target/tight-stop -- catching the rare
large discrete move (like the XRP spike example) rather than predicting the
average bar's direction. Fundamentally different shape than everything else
tested this session (which all assumed a symmetric, high-hit-rate setup).

Whale trades are detected directly from real Kraken tick-level trade data
(8M rows on disk) -- not a proxy or estimate: any single trade whose USD
notional (cost) is above the 99.5th percentile of all trades is flagged as
a whale trade. Entry direction follows the whale trade's side (buy = long,
sell = short). Same real-cost, walk-forward, point-in-time-safe methodology
as everything else tonight -- built to test the hypothesis honestly, not to
confirm it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from train_model import DataHandler

TRADES_PATH = "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/BTC_USD-trades.feather"
TIMEFRAME = "1h"
STARTING_CAPITAL = 100_000.0
WHALE_PERCENTILE = 99.5
STOP_ATR = 2.0    # tight stop -- most trades are expected to lose small
TARGET_ATR = 4.0  # calibrated from real data: median max-favorable-excursion
# after a whale trigger is ~2-3 ATR at 24-48h, ~5 ATR even at the full 240h
# horizon -- the original 20 ATR target was reachable for perhaps 5-10% of
# triggers at best (confirmed empirically), which is why almost nothing hit
# take-profit and the "let it run" mechanic never actually engaged.
MAX_HOLD_CANDLES = 48
N_FOLDS = 8


def load_whale_flagged_bars() -> pd.DataFrame:
    trades = pd.read_feather(TRADES_PATH)
    trades["dt"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)

    threshold = trades["cost"].quantile(WHALE_PERCENTILE / 100.0)
    print(f"whale trade threshold ({WHALE_PERCENTILE}th pct of trade cost): ${threshold:,.0f}")
    whales = trades[trades["cost"] >= threshold].copy()
    print(f"{len(whales)} whale trades out of {len(trades)} total ({len(whales)/len(trades)*100:.3f}%)")

    whales["signed_cost"] = np.where(whales["side"] == "buy", whales["cost"], -whales["cost"])
    whales = whales.set_index("dt")
    hourly_whale_flow = whales["signed_cost"].resample("1h").sum()
    hourly_whale_count = whales["signed_cost"].resample("1h").count()

    price = load_gmx_ohlc("BTC", TIMEFRAME)
    price.index = pd.to_datetime(price.index, utc=True)
    df = price.copy()

    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr

    # Same trend/momentum confirmation as run_macd_200ma_signal_experiment.py
    # -- MACD + 200MA trend + RSI + StochRSI -- required ALONGSIDE the whale
    # trigger, not instead of it. A bare whale trade with no trend/momentum
    # context is exactly the kind of single, unconfirmed trigger that turned
    # out to be too noisy earlier tonight.
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

    df = df.join(hourly_whale_flow.rename("whale_flow"), how="left")
    df = df.join(hourly_whale_count.rename("whale_count"), how="left")
    df["whale_flow"] = df["whale_flow"].fillna(0.0)
    df["whale_count"] = df["whale_count"].fillna(0.0)

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["quarter"] = pd.to_datetime(df["timestamp"], utc=True).dt.quarter
    df = df.dropna(subset=["ATR", "ma200", "macd_signal", "stoch_d"]).reset_index(drop=True)
    return df


def build_signal(df: pd.DataFrame, direction: str, quarters: list[int] | None = None) -> pd.DataFrame:
    df = df.copy()
    whale_long = df["whale_flow"] > 0
    whale_short = df["whale_flow"] < 0

    if direction == "long":
        trend_ok = (df["Close"] > df["ma200"]) & (df["macd"] > df["macd_signal"]) & (df["rsi"] < 70)
        stoch_ok = df["stoch_k"] < 60  # not already deeply overbought
        condition = whale_long & trend_ok & stoch_ok
    else:
        trend_ok = (df["Close"] < df["ma200"]) & (df["macd"] < df["macd_signal"]) & (df["rsi"] > 30)
        stoch_ok = df["stoch_k"] > 40
        condition = whale_short & trend_ok & stoch_ok

    if quarters is not None:
        condition = condition & df["quarter"].isin(quarters)

    label = "LONG" if direction == "long" else "SHORT"
    df["model_direction"] = np.where(condition, label, "HOLD")
    df["open"] = df["Open"].astype(float)
    df["high"] = df["High"].astype(float)
    df["low"] = df["Low"].astype(float)
    df["close"] = df["Close"].astype(float)
    df["price"] = df["Close"].astype(float)
    df["atr"] = df["ATR"].astype(float)
    df["_entry_eligible"] = True
    df["_risk_pct"] = float(Config.RISK_PERCENTAGE)
    df["_max_position_pct"] = float(Config.PAPER_MAX_POSITION_PCT)
    return df


def evaluate(df: pd.DataFrame, direction: str, label: str, quarters: list[int] | None = None) -> dict | None:
    import os
    os.environ["PERSISTENT_POLICY_SYMBOLS"] = "BTC"
    os.environ["PERSISTENT_STOP_ATR"] = str(STOP_ATR)
    os.environ["PERSISTENT_TARGET_ATR"] = str(TARGET_ATR)
    os.environ["PERSISTENT_MAX_HOLD_CANDLES"] = str(MAX_HOLD_CANDLES)

    signals = build_signal(df, direction, quarters=quarters)
    n_triggers = (signals["model_direction"] != "HOLD").sum()
    if n_triggers < 10:
        print(f"{label}: only {n_triggers} triggers, skipping")
        return None

    summary, trades = simulate_production_symbol(
        signals, "BTC", STARTING_CAPITAL, entry_eligible=True,
        fee_bps=Config.PAPER_FEE_BPS,
        slippage_bps=Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS,
    )

    fold_results = []
    edges = np.linspace(0, len(signals), N_FOLDS + 1).astype(int)
    for fold in range(N_FOLDS):
        frame = signals.iloc[edges[fold]:edges[fold + 1]]
        if len(frame) < 20:
            continue
        fold_summary, _ = simulate_production_symbol(
            frame, "BTC", STARTING_CAPITAL, entry_eligible=True,
            fee_bps=Config.PAPER_FEE_BPS,
            slippage_bps=Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS,
        )
        if fold_summary["trades"] == 0:
            continue
        fold_results.append(fold_summary)

    n_folds = len(fold_results)
    n_profitable = sum(1 for r in fold_results if r["return_pct"] > 0)
    binom_p = (
        stats.binomtest(n_profitable, n_folds, p=0.5, alternative="greater").pvalue
        if n_folds > 0 else float("nan")
    )

    result = {
        "label": label, "direction": direction,
        "trades": summary["trades"], "return_pct": summary["return_pct"],
        "profit_factor": summary["profit_factor"], "win_rate_pct": summary["win_rate_pct"],
        "n_folds": n_folds, "n_profitable_folds": n_profitable, "binom_p": binom_p,
    }
    print(f"{label}: trades={result['trades']} return_pct={result['return_pct']:.2f} "
          f"profit_factor={result['profit_factor']:.2f} win_rate={result['win_rate_pct']:.2f}% "
          f"folds={n_profitable}/{n_folds} binom_p={binom_p:.4f}")
    return result


if __name__ == "__main__":
    df = load_whale_flagged_bars()
    print(f"\ntotal bars: {len(df)}, bars with whale activity: {int((df['whale_count']>0).sum())}")

    results = []
    for direction in ["long", "short"]:
        results.append(evaluate(df, direction, f"whale_trigger/{direction}/all_quarters"))

    print("\n=== By quarter ===")
    for direction in ["long", "short"]:
        for q in [1, 2, 3, 4]:
            r = evaluate(df, direction, f"whale_trigger/{direction}/Q{q}", quarters=[q])
            if r:
                results.append(r)

    import json
    out = Path("logs/v4_whale_quarter_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r for r in results if r], indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
