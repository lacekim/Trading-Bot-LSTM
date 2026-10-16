"""The full, integrated strategy described directly by the user, combined
into one system instead of tested piece by piece:

  1. Monthly MACD regime (bullish/bearish) as the macro gate -- only trade
     in the direction of the current regime. Computed strictly from
     COMPLETED monthly bars only (point-in-time safe -- never uses the
     current, still-forming month).
  2. Within that regime, "load the keg": position-building triggers are
     oversold/overbought conditions (RSI) and real whale trades (from the
     actual Kraken tick data), sized up when multiple signals align at once
     rather than a fixed size per trade. Also requires hourly MACD/200MA/
     StochRSI trend confirmation in the regime's direction (same trend
     filter as run_whale_quarter_experiment.py) -- an earlier version of
     this script dropped that confirmation entirely; this is the corrected
     version with the full indicator stack (monthly MACD regime + hourly
     MACD + 200MA + RSI + StochRSI + whale flow).
  3. "Let it explode, take profit on the other side": a trailing stop
     instead of a fixed target -- ride the move as long as it continues,
     lock in gains as it extends, exit on a real reversal rather than an
     arbitrary fixed distance.

This is a custom simulation loop (not simulate_production_symbol, which only
supports fixed stop/target/time exits) specifically to implement the
trailing-stop "explosion then profit-taking" exit the user described.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.gmx_adapter import load_gmx_ohlc
from train_model import DataHandler

TRADES_PATH = "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/BTC_USD-trades.feather"
TIMEFRAME = "1h"
STARTING_CAPITAL = 100_000.0
WHALE_PERCENTILE = 99.5

INITIAL_STOP_ATR = 2.0       # initial protective stop
TRAIL_ACTIVATE_ATR = 3.0     # once price has moved this far in favor, start trailing
TRAIL_DISTANCE_ATR = 2.5     # how far behind the peak/trough the trailing stop sits
MAX_HOLD_CANDLES = 720       # 30 days -- a safety cap, not the intended exit mechanism
BASE_RISK_PCT = 1.0          # base risk per trade, before signal-count scaling


def build_dataset() -> pd.DataFrame:
    price = load_gmx_ohlc("BTC", TIMEFRAME)
    price.index = pd.to_datetime(price.index, utc=True)
    df = price.copy()

    handler = DataHandler()
    atr = handler.calculate_atr(df, Config.ATR_PERIOD)
    df["ATR"] = atr.iloc[:, 0] if isinstance(atr, pd.DataFrame) else atr

    # Monthly MACD regime, point-in-time safe: computed on monthly closes,
    # then forward-filled onto the hourly index using only bars from
    # COMPLETED months (shift by one month so the current, still-forming
    # month's bar is never used before it actually closes).
    monthly = df["Close"].resample("MS").last()
    exp1 = monthly.ewm(span=12, adjust=False).mean()
    exp2 = monthly.ewm(span=26, adjust=False).mean()
    macd_m = exp1 - exp2
    macd_signal_m = macd_m.ewm(span=9, adjust=False).mean()
    regime_m = np.where(macd_m > macd_signal_m, "BULLISH", "BEARISH")
    regime_series = pd.Series(regime_m, index=monthly.index).shift(1)  # only known once month closes
    df["monthly_regime"] = regime_series.reindex(df.index, method="ffill")

    # Hourly RSI for oversold/overbought context
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # StochRSI (same construction as run_whale_quarter_experiment.py)
    rsi_min = df["rsi"].rolling(window=14).min()
    rsi_max = df["rsi"].rolling(window=14).max()
    stoch_rsi_raw = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    df["stoch_k"] = stoch_rsi_raw.rolling(window=3).mean()

    # Hourly MACD + 200MA trend confirmation, same as run_whale_quarter_experiment.py
    df["ma200"] = df["Close"].rolling(window=200).mean()
    exp1_h = df["Close"].ewm(span=12, adjust=False).mean()
    exp2_h = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd_h"] = exp1_h - exp2_h
    df["macd_signal_h"] = df["macd_h"].ewm(span=9, adjust=False).mean()

    # Real whale trades from tick data
    trades = pd.read_feather(TRADES_PATH)
    trades["dt"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    threshold = trades["cost"].quantile(WHALE_PERCENTILE / 100.0)
    whales = trades[trades["cost"] >= threshold].copy()
    whales["signed_cost"] = np.where(whales["side"] == "buy", whales["cost"], -whales["cost"])
    whales = whales.set_index("dt")
    hourly_whale_flow = whales["signed_cost"].resample("1h").sum()
    df = df.join(hourly_whale_flow.rename("whale_flow"), how="left")
    df["whale_flow"] = df["whale_flow"].fillna(0.0)

    df = df.dropna(subset=["ATR", "monthly_regime", "rsi", "stoch_k", "ma200", "macd_signal_h"]).reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    return df


def run_backtest(df: pd.DataFrame) -> dict:
    fee_rate = Config.PAPER_FEE_BPS / 10000.0
    slip = (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) / 10000.0

    capital = STARTING_CAPITAL
    peak_capital = capital
    max_drawdown = 0.0
    position = None  # dict: direction, entry_price, quantity, notional, entry_fee, stop_price, entry_bar, extreme_price
    trades = []

    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    atr = df["ATR"].to_numpy()
    regime = df["monthly_regime"].to_numpy()
    rsi = df["rsi"].to_numpy()
    stoch_k = df["stoch_k"].to_numpy()
    ma200 = df["ma200"].to_numpy()
    macd_h = df["macd_h"].to_numpy()
    macd_signal_h = df["macd_signal_h"].to_numpy()
    whale_flow = df["whale_flow"].to_numpy()
    n = len(df)

    for bar in range(n):
        price_now = close[bar]

        if position is not None:
            direction = position["direction"]
            if direction == "LONG":
                position["extreme_price"] = max(position["extreme_price"], high[bar])
                profit_atr = (position["extreme_price"] - position["entry_price"]) / position["entry_atr"]
                if profit_atr >= TRAIL_ACTIVATE_ATR:
                    trail_stop = position["extreme_price"] - TRAIL_DISTANCE_ATR * position["entry_atr"]
                    position["stop_price"] = max(position["stop_price"], trail_stop)
                hit_stop = low[bar] <= position["stop_price"]
            else:
                position["extreme_price"] = min(position["extreme_price"], low[bar])
                profit_atr = (position["entry_price"] - position["extreme_price"]) / position["entry_atr"]
                if profit_atr >= TRAIL_ACTIVATE_ATR:
                    trail_stop = position["extreme_price"] + TRAIL_DISTANCE_ATR * position["entry_atr"]
                    position["stop_price"] = min(position["stop_price"], trail_stop)
                hit_stop = high[bar] >= position["stop_price"]

            timed_out = bar - position["entry_bar"] >= MAX_HOLD_CANDLES
            if hit_stop or timed_out:
                exit_price = position["stop_price"] if hit_stop else price_now
                exit_price_slipped = exit_price * (1 - slip if direction == "LONG" else 1 + slip)
                gross = (exit_price_slipped - position["entry_price"]) * position["quantity"]
                if direction == "SHORT":
                    gross *= -1
                exit_fee = position["notional"] * fee_rate
                net = gross - position["entry_fee"] - exit_fee
                capital += net
                trades.append({
                    "entry_time": df["timestamp"].iloc[position["entry_bar"]], "exit_time": df["timestamp"].iloc[bar],
                    "direction": direction, "entry_price": position["entry_price"], "exit_price": exit_price_slipped,
                    "net_pnl": net, "reason": "trail_stop" if hit_stop else "timeout",
                    "signal_count": position["signal_count"],
                })
                position = None

        if position is None:
            r = regime[bar]
            oversold = rsi[bar] < 30
            overbought = rsi[bar] > 70
            whale_buy = whale_flow[bar] > 0
            whale_sell = whale_flow[bar] < 0

            long_trend_ok = (close[bar] > ma200[bar]) and (macd_h[bar] > macd_signal_h[bar]) and (stoch_k[bar] < 60)
            short_trend_ok = (close[bar] < ma200[bar]) and (macd_h[bar] < macd_signal_h[bar]) and (stoch_k[bar] > 40)

            signal_count = 0
            direction = None
            if r == "BULLISH" and long_trend_ok and (oversold or whale_buy):
                direction = "LONG"
                signal_count = int(oversold) + int(whale_buy)
            elif r == "BEARISH" and short_trend_ok and (overbought or whale_sell):
                direction = "SHORT"
                signal_count = int(overbought) + int(whale_sell)

            if direction is not None and atr[bar] > 0:
                risk_pct = BASE_RISK_PCT * (1 + 0.5 * (signal_count - 1))  # scale up when both align
                risk_amount = capital * risk_pct / 100.0
                stop_distance = INITIAL_STOP_ATR * atr[bar]
                notional = min(risk_amount / stop_distance * price_now, capital * 0.5)
                if notional >= Config.PAPER_MIN_ORDER_USD:
                    fill = price_now * (1 + slip if direction == "LONG" else 1 - slip)
                    fee = notional * fee_rate
                    quantity = notional / fill
                    stop_price = fill - stop_distance if direction == "LONG" else fill + stop_distance
                    position = {
                        "direction": direction, "entry_price": fill, "quantity": quantity,
                        "notional": notional, "entry_fee": fee, "stop_price": stop_price,
                        "entry_bar": bar, "entry_atr": atr[bar], "extreme_price": fill,
                        "signal_count": signal_count,
                    }

        marked = capital
        if position is not None:
            move = (price_now - position["entry_price"]) * position["quantity"]
            marked += move if position["direction"] == "LONG" else -move
        peak_capital = max(peak_capital, marked)
        max_drawdown = min(max_drawdown, (marked / peak_capital - 1.0) * 100.0)

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        return {"trades": 0}
    wins = trade_frame[trade_frame["net_pnl"] > 0]
    losses = trade_frame[trade_frame["net_pnl"] < 0]
    gross_wins = wins["net_pnl"].sum()
    gross_losses = abs(losses["net_pnl"].sum())
    profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else float("inf")

    return {
        "trades": len(trade_frame), "return_pct": (capital / STARTING_CAPITAL - 1) * 100,
        "profit_factor": profit_factor, "win_rate_pct": len(wins) / len(trade_frame) * 100,
        "max_drawdown_pct": max_drawdown, "avg_win": wins["net_pnl"].mean() if len(wins) else 0,
        "avg_loss": losses["net_pnl"].mean() if len(losses) else 0,
        "trail_stop_exits": int((trade_frame["reason"] == "trail_stop").sum()),
        "timeout_exits": int((trade_frame["reason"] == "timeout").sum()),
        "final_capital": capital, "trade_frame": trade_frame,
    }


if __name__ == "__main__":
    df = build_dataset()
    print(f"total bars: {len(df)}")
    print(f"bullish regime bars: {(df['monthly_regime']=='BULLISH').sum()}, "
          f"bearish regime bars: {(df['monthly_regime']=='BEARISH').sum()}")

    result = run_backtest(df)
    trade_frame = result.pop("trade_frame", None)
    print("\n=== Full backtest, full history ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if trade_frame is not None and not trade_frame.empty:
        print("\n=== By signal count (how many signals aligned at entry) ===")
        for sc, group in trade_frame.groupby("signal_count"):
            wins = group[group["net_pnl"] > 0]
            losses = group[group["net_pnl"] < 0]
            gw, gl = wins["net_pnl"].sum(), abs(losses["net_pnl"].sum())
            pf = gw / gl if gl > 0 else float("inf")
            print(f"  signal_count={sc}: trades={len(group)} win_rate={len(wins)/len(group)*100:.1f}% "
                  f"profit_factor={pf:.2f} total_pnl=${group['net_pnl'].sum():,.0f}")

        trade_frame.to_csv("logs/v4_integrated_regime_trades.csv", index=False)
        print("\nwrote logs/v4_integrated_regime_trades.csv")
