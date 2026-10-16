"""Honest test of the real Fair Value Gap (FVG) strategy the user pasted a
description of -- NOT the fake "technical_analysis_utils_fvg.py" stub
already in freqtrade-develop/user_data/strategies/, which turned out to be
a 20-period SMA band with a comment admitting it's a placeholder, not real
FVG detection at all.

Real FVG definition used here (three-candle wick gap):
  Bullish FVG (formed during an up-move): high[i-1] < low[i+1]
    zone = (high[i-1], low[i+1])
  Bearish FVG (formed during a down-move): low[i-1] > high[i+1]
    zone = (high[i+1], low[i-1])

Trend filter: reuses trading_bot_v4/signals/ma200_trend.py's already-
verified golden-cross (50MA/200MA) state rather than inventing a new,
unvalidated trend rule -- only take bullish FVGs when the golden cross is
active (ma50 > ma200), only bearish FVGs when it isn't. This matches the
document's own instruction to establish trend on a higher timeframe first.

Entry: first candle after the FVG forms whose low retraces into the zone
(bullish) or high retraces into the zone (bearish), closing back on the
zone's favorable side -- i.e. a reaction/bounce, not a blow-through.
Zones are invalidated (never traded) if price closes fully through the far
side of the zone before ever being touched.

Stop: beyond the FVG's origin candle (candle i-1) extreme, per the
document's explicit recommendation. Target: fixed R multiple of that risk
distance (tested at 1R/2R/3R since the document frames targets this way,
not as a single fixed number).

Same real Kraken 4h data, same combined-portfolio structure, and the same
real fee/slippage costs as tools/run_ma200_cross_strategy.py, for
consistency with everything else already tested this session. Shorts are
tested too, even though every OTHER short mechanism tried this session
found zero real edge -- FVG's entry logic is different enough from those
to warrant its own honest test rather than assuming that finding transfers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from trading_bot_v4.signals.ma200_trend import compute_trend_state

FEE_BPS = 5
SLIP_BPS = 7
ENTRY_COST = (FEE_BPS + SLIP_BPS) / 10000
EXIT_COST = (FEE_BPS + SLIP_BPS) / 10000

STARTING_CAPITAL = 100_000.0
PER_ASSET_ALLOCATION_PCT = 15.0
R_MULTIPLES_TO_TEST = [1, 2, 3]

ASSETS = {
    "BTC": "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/BTC_USD-1h.feather",
    "ETH": "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/ETH_USD-1h.feather",
    "XMR": "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/XMR_USD-1h.feather",
    "PAXG": "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/PAXG_USD-1h.feather",
    "SOL": "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/SOL_USD-1h.feather",
    "TRX": "/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop/user_data/data/kraken/TRX_USD-1h.feather",
}


def load_4h(path: str) -> pd.DataFrame:
    h1 = pd.read_feather(path)
    h1["date"] = pd.to_datetime(h1["date"], utc=True)
    h1 = h1.set_index("date")
    return h1.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def detect_fvgs(df: pd.DataFrame) -> list[dict]:
    """Real three-candle wick-gap FVGs. i is the middle/impulsive candle;
    the gap is between candle i-1 and candle i+1's wicks."""
    zones = []
    high, low = df["high"].values, df["low"].values
    idx = df.index
    for i in range(1, len(df) - 1):
        if low[i + 1] > high[i - 1]:
            zones.append({
                "kind": "bullish", "formed_at": idx[i + 1],
                "zone_low": high[i - 1], "zone_high": low[i + 1],
                "origin_extreme": low[i - 1],  # stop-loss reference
            })
        elif high[i + 1] < low[i - 1]:
            zones.append({
                "kind": "bearish", "formed_at": idx[i + 1],
                "zone_low": high[i + 1], "zone_high": low[i - 1],
                "origin_extreme": high[i - 1],  # stop-loss reference
            })
    return zones


def simulate_asset(label: str, df: pd.DataFrame, trend: pd.DataFrame, r_multiple: float) -> list[dict]:
    zones = detect_fvgs(df)
    trades = []
    open_trade = None
    zone_ptr = 0
    zones_by_formed = sorted(zones, key=lambda z: z["formed_at"])

    for ts, row in df.iterrows():
        # admit newly-formed zones as "pending" up to this bar
        while zone_ptr < len(zones_by_formed) and zones_by_formed[zone_ptr]["formed_at"] <= ts:
            zones_by_formed[zone_ptr]["pending"] = True
            zone_ptr += 1

        if open_trade is not None:
            side = open_trade["side"]
            if side == "long":
                if row["low"] <= open_trade["stop"]:
                    exit_price = open_trade["stop"] * (1 - EXIT_COST)
                    _close(trades, open_trade, exit_price, ts, "stop")
                    open_trade = None
                elif row["high"] >= open_trade["target"]:
                    exit_price = open_trade["target"] * (1 - EXIT_COST)
                    _close(trades, open_trade, exit_price, ts, "target")
                    open_trade = None
            else:
                if row["high"] >= open_trade["stop"]:
                    exit_price = open_trade["stop"] * (1 + EXIT_COST)
                    _close(trades, open_trade, exit_price, ts, "stop")
                    open_trade = None
                elif row["low"] <= open_trade["target"]:
                    exit_price = open_trade["target"] * (1 + EXIT_COST)
                    _close(trades, open_trade, exit_price, ts, "target")
                    open_trade = None

        if open_trade is None:
            if ts not in trend.index or pd.isna(trend.loc[ts, "ma200"]):
                continue
            golden_cross_active = bool(trend.loc[ts, "above200"])

            for z in zones_by_formed:
                if not z.get("pending") or z.get("done") or z["formed_at"] >= ts:
                    continue
                # Trend is re-checked fresh each bar (not baked in at zone
                # formation) -- a zone stays live and re-eligible if trend
                # later aligns, right up until price invalidates it below.
                if z["kind"] == "bullish" and not golden_cross_active:
                    continue
                if z["kind"] == "bearish" and golden_cross_active:
                    continue

                if z["kind"] == "bullish":
                    if row["close"] < z["zone_low"]:
                        z["done"] = True  # closed clean through the far side: invalidated
                        continue
                    if row["low"] <= z["zone_high"] and row["close"] > z["zone_low"]:
                        risk = z["zone_high"] - z["origin_extreme"]
                        if risk <= 0:
                            z["done"] = True
                            continue
                        entry = row["close"] * (1 + ENTRY_COST)
                        open_trade = {
                            "asset": label, "side": "long", "entry_price": entry,
                            "entry_time": ts, "stop": z["origin_extreme"],
                            "target": entry + risk * r_multiple,
                            "notional": STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100.0,
                        }
                        z["done"] = True
                        break
                else:
                    if row["close"] > z["zone_high"]:
                        z["done"] = True
                        continue
                    if row["high"] >= z["zone_low"] and row["close"] < z["zone_high"]:
                        risk = z["origin_extreme"] - z["zone_low"]
                        if risk <= 0:
                            z["done"] = True
                            continue
                        entry = row["close"] * (1 - ENTRY_COST)
                        open_trade = {
                            "asset": label, "side": "short", "entry_price": entry,
                            "entry_time": ts, "stop": z["origin_extreme"],
                            "target": entry - risk * r_multiple,
                            "notional": STARTING_CAPITAL * PER_ASSET_ALLOCATION_PCT / 100.0,
                        }
                        z["done"] = True
                        break
    return trades


def _close(trades: list[dict], open_trade: dict, exit_price: float, ts, reason: str) -> None:
    entry = open_trade["entry_price"]
    ret = (exit_price / entry - 1) if open_trade["side"] == "long" else (entry / exit_price - 1)
    trades.append({
        "asset": open_trade["asset"], "side": open_trade["side"],
        "entry_time": open_trade["entry_time"], "exit_time": ts,
        "ret_pct": ret * 100, "pnl": open_trade["notional"] * ret,
        "exit_reason": reason,
    })


def main() -> None:
    price, trends = {}, {}
    for label, path in ASSETS.items():
        df = load_4h(path)
        price[label] = df
        trends[label] = compute_trend_state(df["close"])
        print(f"{label}: {len(df)} bars, {df.index.min().date()} to {df.index.max().date()}")

    for r_multiple in R_MULTIPLES_TO_TEST:
        print(f"\n{'='*70}\nR MULTIPLE = {r_multiple}\n{'='*70}")
        all_trades = []
        for label, df in price.items():
            trades = simulate_asset(label, df, trends[label], r_multiple)
            all_trades.extend(trades)

        if not all_trades:
            print("no trades at all -- zero FVG setups triggered under these rules")
            continue

        trades_df = pd.DataFrame(all_trades).sort_values("exit_time")
        wins = trades_df[trades_df.pnl > 0]
        losses = trades_df[trades_df.pnl <= 0]
        gross_win = wins.pnl.sum()
        gross_loss = -losses.pnl.sum()
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        top3 = trades_df.nlargest(3, "pnl")
        trades_excl_top3 = trades_df.drop(top3.index)
        wins_excl = trades_excl_top3[trades_excl_top3.pnl > 0]
        losses_excl = trades_excl_top3[trades_excl_top3.pnl <= 0]
        gross_win_excl = wins_excl.pnl.sum()
        gross_loss_excl = -losses_excl.pnl.sum()
        pf_excl = gross_win_excl / gross_loss_excl if gross_loss_excl > 0 else float("inf")

        print(f"total trades: {len(trades_df)}  win rate: {len(wins)/len(trades_df)*100:.1f}%")
        print(f"profit factor: {pf:.2f}   profit factor excl. top-3 winners: {pf_excl:.2f}")
        print(f"total pnl: ${trades_df.pnl.sum():,.0f}")

        print("\nby asset / side:")
        for (label, side), g in trades_df.groupby(["asset", "side"]):
            w = g[g.pnl > 0]
            wr = len(w) / len(g) * 100 if len(g) else 0
            gw = w.pnl.sum()
            gl = -g[g.pnl <= 0].pnl.sum()
            asset_pf = gw / gl if gl > 0 else float("inf")
            print(f"  {label:5} {side:5} n={len(g):4} win_rate={wr:5.1f}% "
                  f"pf={asset_pf:5.2f} total_pnl=${g.pnl.sum():>10,.0f}")

        print("\nby exit reason:")
        for reason, g in trades_df.groupby("exit_reason"):
            print(f"  {reason:8} n={len(g):4} win_rate={len(g[g.pnl>0])/len(g)*100:5.1f}%")


if __name__ == "__main__":
    main()
