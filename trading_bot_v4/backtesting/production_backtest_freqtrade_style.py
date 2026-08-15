"""Experimental simulator porting the freqtrade OnnxEthusdtStrategy's entry/exit
mechanics onto trading_bot_v4 signals, for direct comparison against
simulate_production_symbol (persistent-until-reversal/window-exit).

Ported, because it's what was actually verified to work in the freqtrade
backtest tonight:
  - EMA200 regime filter (only take a signal if price agrees with the trend)
  - ATR stop-loss, 1.2x ATR, clipped to [0.4%, 5%] of price
  - Trailing stop: activates at +4% favorable move, trails 1.5% behind
  - Fixed 5% ROI take-profit
  - Time-based loss cut: force-exit after 72 bars if still underwater

Deliberately NOT ported: the freqtrade strategy's custom_exit model-probability
logic. It threw a TypeError on every call in tonight's real run and never
fired -- every exit came from stop_loss/trailing_stop/roi. Porting it would be
porting an untested code path, not the thing that was actually shown to work.

Kept as a separate module from production_backtest.py -- this does not touch
or replace the live simulate_production_symbol used by the scheduler/paper
trading path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_bot_v4.config_v4 import V4Config as Config

ROI_TARGET = 0.05
TRAILING_ACTIVATION = 0.04
TRAILING_DISTANCE = 0.015
MAX_HOLD_BARS = 72
REGIME_EMA_SPAN = 200
ATR_STOP_MULTIPLIER = 1.2
MIN_STOP_PCT = 0.004
MAX_STOP_PCT = 0.05


@dataclass
class FreqtradeStylePosition:
    direction: str
    entry_time: pd.Timestamp
    entry_bar: int
    entry_price: float
    quantity: float
    notional: float
    entry_fee: float
    stop_price: float
    peak_favorable_price: float
    trailing_active: bool = False


def simulate_production_symbol_freqtrade_style(
    signals: pd.DataFrame, symbol: str, starting_capital: float,
    fee_bps: float | None = None, slippage_bps: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """signals must have: timestamp, model_direction, open, high, low, price, atr
    -- same schema simulate_production_symbol expects."""
    capital = float(starting_capital)
    peak = capital
    max_drawdown = 0.0
    position: FreqtradeStylePosition | None = None
    trades: list[dict[str, Any]] = []

    effective_slippage_bps = (
        Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS
        if slippage_bps is None else float(slippage_bps)
    )
    effective_fee_bps = Config.PAPER_FEE_BPS if fee_bps is None else float(fee_bps)
    slip = effective_slippage_bps / 10000.0
    fee_rate = effective_fee_bps / 10000.0

    ordered = signals.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ordered["ema200"] = ordered["price"].ewm(span=REGIME_EMA_SPAN, adjust=False).mean()

    def _close_position(position: FreqtradeStylePosition, exit_price: float, reason: str, timestamp, bar: int) -> None:
        nonlocal capital
        exit_fill = exit_price * (1 - slip if position.direction == "LONG" else 1 + slip)
        gross = (exit_fill - position.entry_price) * position.quantity
        if position.direction == "SHORT":
            gross *= -1
        exit_fee = position.notional * fee_rate
        net = gross - position.entry_fee - exit_fee
        capital += net
        trades.append({
            "symbol": symbol, "direction": position.direction,
            "entry_time": position.entry_time, "exit_time": timestamp,
            "entry_price": position.entry_price, "exit_price": exit_fill,
            "notional": position.notional, "gross_pnl": gross,
            "fees": position.entry_fee + exit_fee, "net_pnl": net,
            "return_pct": net / position.notional * 100.0 if position.notional else 0.0,
            "exit_reason": reason, "hold_bars": bar - position.entry_bar,
        })

    for bar, row in ordered.iterrows():
        price = float(row["price"])
        high, low = float(row["high"]), float(row["low"])

        if position is not None:
            hold_bars = bar - position.entry_bar
            exit_price, reason = None, None

            if position.direction == "LONG" and low <= position.stop_price:
                exit_price, reason = position.stop_price, "stop_loss"
            elif position.direction == "SHORT" and high >= position.stop_price:
                exit_price, reason = position.stop_price, "stop_loss"

            if exit_price is None:
                if position.direction == "LONG":
                    position.peak_favorable_price = max(position.peak_favorable_price, high)
                    favorable_move = (position.peak_favorable_price / position.entry_price) - 1.0
                else:
                    position.peak_favorable_price = min(position.peak_favorable_price, low)
                    favorable_move = 1.0 - (position.peak_favorable_price / position.entry_price)
                if favorable_move >= TRAILING_ACTIVATION:
                    position.trailing_active = True
                if position.trailing_active:
                    if position.direction == "LONG":
                        trail_stop = position.peak_favorable_price * (1 - TRAILING_DISTANCE)
                        position.stop_price = max(position.stop_price, trail_stop)
                        if low <= position.stop_price:
                            exit_price, reason = position.stop_price, "trailing_stop"
                    else:
                        trail_stop = position.peak_favorable_price * (1 + TRAILING_DISTANCE)
                        position.stop_price = min(position.stop_price, trail_stop)
                        if high >= position.stop_price:
                            exit_price, reason = position.stop_price, "trailing_stop"

            if exit_price is None:
                current_profit = (
                    (price / position.entry_price) - 1.0 if position.direction == "LONG"
                    else 1.0 - (price / position.entry_price)
                )
                if current_profit >= ROI_TARGET:
                    exit_price, reason = price, "roi"
                elif hold_bars >= MAX_HOLD_BARS and current_profit < 0:
                    exit_price, reason = price, "time_cut"

            if exit_price is not None:
                _close_position(position, exit_price, reason, row["timestamp"], bar)
                position = None

        if position is None:
            direction = str(row["model_direction"]).upper()
            ema200 = float(row["ema200"])
            regime_ok = (direction == "LONG" and price > ema200) or (direction == "SHORT" and price < ema200)
            if direction in {"LONG", "SHORT"} and regime_ok:
                atr = float(row.get("atr", float("nan")))
                stop_pct = float(np.clip((ATR_STOP_MULTIPLIER * atr) / price, MIN_STOP_PCT, MAX_STOP_PCT)) \
                    if np.isfinite(atr) and atr > 0 else MIN_STOP_PCT
                stop_distance = price * stop_pct
                risk_amount = capital * (Config.RISK_PERCENTAGE / 100.0)
                notional = min(risk_amount / stop_pct, capital * (Config.PAPER_MAX_POSITION_PCT / 100.0))
                if notional >= Config.PAPER_MIN_ORDER_USD:
                    fill = price * (1 + slip if direction == "LONG" else 1 - slip)
                    fee = notional * fee_rate
                    quantity = notional / fill
                    stop_price = fill - stop_distance if direction == "LONG" else fill + stop_distance
                    position = FreqtradeStylePosition(
                        direction=direction, entry_time=pd.Timestamp(row["timestamp"]), entry_bar=bar,
                        entry_price=fill, quantity=quantity, notional=notional, entry_fee=fee,
                        stop_price=stop_price, peak_favorable_price=fill,
                    )

        marked = capital
        if position is not None:
            move = (price - position.entry_price) * position.quantity
            marked += move if position.direction == "LONG" else -move
        peak = max(peak, marked)
        max_drawdown = min(max_drawdown, (marked / peak - 1.0) * 100.0)

    if position is not None and not ordered.empty:
        last_row = ordered.iloc[-1]
        _close_position(position, float(last_row["price"]), "end_of_data", last_row["timestamp"], len(ordered) - 1)

    trade_frame = pd.DataFrame(trades)
    wins = trade_frame.loc[trade_frame.get("net_pnl", pd.Series(dtype=float)) > 0] if not trade_frame.empty else trade_frame
    losses = trade_frame.loc[trade_frame.get("net_pnl", pd.Series(dtype=float)) < 0] if not trade_frame.empty else trade_frame
    gross_wins = float(wins["net_pnl"].sum()) if not wins.empty else 0.0
    gross_losses = abs(float(losses["net_pnl"].sum())) if not losses.empty else 0.0
    summary = {
        "symbol": symbol, "candles": int(len(ordered)), "trades": int(len(trade_frame)),
        "long_trades": int((trade_frame["direction"] == "LONG").sum()) if not trade_frame.empty else 0,
        "short_trades": int((trade_frame["direction"] == "SHORT").sum()) if not trade_frame.empty else 0,
        "starting_capital": starting_capital, "final_capital": capital,
        "return_pct": (capital / starting_capital - 1.0) * 100.0 if starting_capital else 0.0,
        "win_rate_pct": len(wins) / len(trade_frame) * 100.0 if len(trade_frame) else 0.0,
        "max_drawdown_pct": max_drawdown,
        "profit_factor": gross_wins / gross_losses if gross_losses else (float("inf") if gross_wins else 0.0),
    }
    return summary, trade_frame
