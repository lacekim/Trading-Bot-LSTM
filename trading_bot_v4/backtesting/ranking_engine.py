"""V4 backtest ranking mode for GMX symbols."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from trading_bot import list_gmx_symbols, load_gmx_ohlc
from trading_bot_v4.backtesting.reporting import write_v4_backtest_html_report
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler, build_legacy_data_handler
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache

logger = build_logger("v4_backtest_ranking")


STABLECOIN_TOKENS = {"USDC", "USDT", "DAI", "USD", "SUSD", "LUSD", "FRAX", "TUSD", "BUSD", "USDP", "MIM"}
SYNTHETIC_COMMODITY_TOKENS = {"XAU", "XAG", "OIL", "WTI", "GOLD", "SILVER", "COMMODITY", "PAXG", "XAUT"}


def _looks_like_stablecoin(symbol: str) -> bool:
    normalized = str(symbol or "").upper().replace("-", "")
    return any(token in normalized for token in STABLECOIN_TOKENS) or normalized.endswith("USD")


def _looks_like_synthetic_commodity(symbol: str) -> bool:
    normalized = str(symbol or "").upper().replace("-", "")
    return any(token in normalized for token in SYNTHETIC_COMMODITY_TOKENS)


def _safe_numeric(series: Any, default: float = 0.0) -> float:
    return float(series) if pd.notna(series) else default


def _compute_trade_metrics(trades: pd.DataFrame | None, starting_capital: float, final_capital: float, drawdown: float = 0.0) -> dict[str, float]:
    if trades is None or trades.empty:
        return {
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "average_trade_return": 0.0,
            "expectancy": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "calmar_ratio": 0.0,
        }

    returns = []
    for _, trade in trades.iterrows():
        entry = trade.get("entry_price")
        exit_price = trade.get("exit_price")
        direction = str(trade.get("direction", "LONG")).upper()
        if entry is None or exit_price is None:
            continue
        entry = float(entry)
        exit_price = float(exit_price)
        if direction == "LONG":
            returns.append(((exit_price / entry) - 1) * 100.0)
        else:
            returns.append(((entry / exit_price) - 1) * 100.0)

    if not returns:
        return {
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "average_trade_return": 0.0,
            "expectancy": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "calmar_ratio": 0.0,
        }

    arr = np.array(returns, dtype=float)
    winning = arr[arr > 0]
    losing = arr[arr < 0]
    gross_profit = float(winning.sum()) if len(winning) else 0.0
    gross_loss = abs(float(losing.sum())) if len(losing) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    mean_return = float(arr.mean())
    std_return = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0
    downside = losing
    downside_std = float(downside.std(ddof=0)) if len(downside) > 1 else 0.0
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
    sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0
    avg_win = float(winning.mean()) if len(winning) else 0.0
    avg_loss = float(losing.mean()) if len(losing) else 0.0
    win_rate = len(winning) / len(arr) if len(arr) else 0.0
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)
    consecutive_wins = 0
    consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    for value in arr:
        if value > 0:
            current_wins += 1
            current_losses = 0
            consecutive_wins = max(consecutive_wins, current_wins)
        elif value < 0:
            current_losses += 1
            current_wins = 0
            consecutive_losses = max(consecutive_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    calmar_ratio = (float(final_capital - starting_capital) / max(float(starting_capital), 1.0)) / max(abs(float(drawdown)), 1e-9) if drawdown != 0 else 0.0
    return {
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "average_trade_return": mean_return,
        "expectancy": expectancy,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "largest_win": float(winning.max()) if len(winning) else 0.0,
        "largest_loss": float(losing.min()) if len(losing) else 0.0,
        "consecutive_wins": int(consecutive_wins),
        "consecutive_losses": int(consecutive_losses),
        "calmar_ratio": calmar_ratio,
    }


def filter_and_rank_backtest_summaries(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=[
            "symbol",
            "candles",
            "signals_traded",
            "return_pct",
            "final_capital",
            "total_profit",
            "win_rate_pct",
            "max_drawdown_pct",
            "profit_factor",
            "average_trade_return",
            "return_to_drawdown_ratio",
        ])

    ranked = summary_df.copy()
    ranked["symbol"] = ranked["symbol"].astype(str)
    ranked["candles"] = pd.to_numeric(ranked.get("candles", 0), errors="coerce").fillna(0)
    ranked["signals_traded"] = pd.to_numeric(ranked.get("signals_traded", 0), errors="coerce").fillna(0)
    ranked["return_pct"] = pd.to_numeric(ranked.get("return_pct", 0), errors="coerce").fillna(0)
    ranked["max_drawdown_pct"] = pd.to_numeric(ranked.get("max_drawdown_pct", 0), errors="coerce").fillna(0)
    ranked["profit_factor"] = pd.to_numeric(ranked.get("profit_factor", 0), errors="coerce").fillna(0)
    ranked["return_to_drawdown_ratio"] = pd.to_numeric(ranked.get("return_to_drawdown_ratio", 0), errors="coerce").fillna(0)

    ranked["exclude"] = False
    ranked.loc[ranked["symbol"].map(_looks_like_stablecoin), "exclude"] = True
    ranked.loc[ranked["symbol"].map(_looks_like_synthetic_commodity), "exclude"] = True
    ranked.loc[ranked["candles"] < 1000, "exclude"] = True
    ranked.loc[ranked["signals_traded"] < 50, "exclude"] = True

    filtered = ranked.loc[~ranked["exclude"]].copy()
    if filtered.empty:
        return filtered

    filtered["has_positive_return"] = filtered["return_pct"] > 0
    filtered["meets_drawdown_filter"] = filtered["max_drawdown_pct"] >= -15.0
    filtered["meets_profit_factor_filter"] = filtered["profit_factor"] >= 1.2

    filtered = filtered.sort_values(
        by=[
            "has_positive_return",
            "meets_drawdown_filter",
            "meets_profit_factor_filter",
            "return_to_drawdown_ratio",
            "return_pct",
            "profit_factor",
        ],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    )
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def run_symbol_v4_backtest_ranking(model: Any, data_handler: Any, symbol: str, timeframe: str, starting_capital: float):
    df_raw = load_gmx_ohlc(symbol, timeframe)
    df = data_handler.prepare_features(df_raw)
    df["ATR"] = data_handler.calculate_atr(df_raw, Config.ATR_PERIOD)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) <= Config.SEQUENCE_LENGTH + 1:
        raise ValueError(f"Insufficient data for {symbol}: {len(df)} clean candles")

    features = df[Config.FEATURE_COLUMNS].values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        scaled = data_handler.normalize_data(features, fit=False)
    except TypeError:
        scaled = data_handler.normalize_data(features)

    seq_len = Config.SEQUENCE_LENGTH
    X = np.array([scaled[i - seq_len:i] for i in range(seq_len, len(scaled))])
    probs = model.predict(X, verbose=0).reshape(-1)
    signal_index = df.index[seq_len:]

    signals = pd.DataFrame(index=signal_index)
    signals["prob"] = probs
    signals["Close"] = df.loc[signal_index, "Close"]
    signals["High_next"] = df["High"].shift(-1).loc[signal_index]
    signals["Low_next"] = df["Low"].shift(-1).loc[signal_index]
    signals["Close_next"] = df["Close"].shift(-1).loc[signal_index]
    signals["ATR"] = df.loc[signal_index, "ATR"]
    signals = signals.dropna()

    capital = float(starting_capital)
    trade_log = []
    equity = []
    threshold = Config.MIN_SIGNAL_THRESHOLD

    for timestamp, row in signals.iterrows():
        probability = float(row["prob"])
        entry_price = float(row["Close"])
        atr = float(row["ATR"])

        if probability > threshold:
            direction = "LONG"
            stop_loss = entry_price - atr * Config.ATR_SL_MULTIPLIER
            take_profit = entry_price + atr * Config.ATR_TP_MULTIPLIER
        elif probability < (1 - threshold):
            direction = "SHORT"
            stop_loss = entry_price + atr * Config.ATR_SL_MULTIPLIER
            take_profit = entry_price - atr * Config.ATR_TP_MULTIPLIER
        else:
            equity.append((timestamp, capital))
            continue

        stop_distance = abs(entry_price - stop_loss)
        risk_amount = capital * (Config.RISK_PERCENTAGE / 100)
        risk_size = risk_amount / stop_distance if stop_distance else 0
        max_affordable = capital / entry_price
        units = min(risk_size, max_affordable * 0.95)
        if units <= 0:
            equity.append((timestamp, capital))
            continue

        high_next = float(row["High_next"])
        low_next = float(row["Low_next"])
        close_next = float(row["Close_next"])

        if direction == "LONG":
            if low_next <= stop_loss:
                exit_price = stop_loss
                reason = "Stop Loss"
            elif high_next >= take_profit:
                exit_price = take_profit
                reason = "Take Profit"
            else:
                exit_price = close_next
                reason = "Next Candle Close"
            profit = (exit_price - entry_price) * units
        else:
            if high_next >= stop_loss:
                exit_price = stop_loss
                reason = "Stop Loss"
            elif low_next <= take_profit:
                exit_price = take_profit
                reason = "Take Profit"
            else:
                exit_price = close_next
                reason = "Next Candle Close"
            profit = (entry_price - exit_price) * units

        capital += profit
        trade_log.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "direction": direction,
                "probability": probability,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "units": units,
                "profit": profit,
                "capital": capital,
                "reason": reason,
            }
        )
        equity.append((timestamp, capital))

    trades = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(equity, columns=["timestamp", "equity"]).set_index("timestamp")

    if trades.empty:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": len(df),
            "predictions": len(signals),
            "signals_traded": 0,
            "starting_capital": starting_capital,
            "final_capital": starting_capital,
            "total_profit": 0.0,
            "return_pct": 0.0,
            "win_rate_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "average_trade_return": 0.0,
            "return_to_drawdown_ratio": 0.0,
        }, trades

    wins = trades[trades["profit"] > 0]
    drawdown = (equity_df["equity"] / equity_df["equity"].cummax() - 1).min() * 100
    gross_profit = float(wins["profit"].sum()) if not wins.empty else 0.0
    gross_loss = float(trades.loc[trades["profit"] < 0, "profit"].abs().sum()) if not trades.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    trade_returns = []
    for _, trade in trades.iterrows():
        entry_price = float(trade["entry_price"])
        exit_price = float(trade["exit_price"])
        if trade["direction"] == "LONG":
            trade_return = ((exit_price / entry_price) - 1) * 100
        else:
            trade_return = ((entry_price / exit_price) - 1) * 100
        trade_returns.append(trade_return)

    average_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0
    return_pct = ((capital / starting_capital) - 1) * 100 if starting_capital else 0.0
    return_to_drawdown_ratio = return_pct / abs(drawdown) if drawdown < 0 else (float("inf") if return_pct > 0 else 0.0)

    metrics = _compute_trade_metrics(trades, starting_capital, capital, drawdown)
    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": len(df),
        "predictions": len(signals),
        "signals_traded": len(trades),
        "starting_capital": starting_capital,
        "final_capital": capital,
        "total_profit": capital - starting_capital,
        "return_pct": return_pct,
        "win_rate_pct": (len(wins) / len(trades) * 100) if len(trades) else 0.0,
        "max_drawdown_pct": drawdown,
        "profit_factor": metrics["profit_factor"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "sortino_ratio": metrics["sortino_ratio"],
        "average_trade_return": metrics["average_trade_return"],
        "expectancy": metrics["expectancy"],
        "average_win": metrics["average_win"],
        "average_loss": metrics["average_loss"],
        "largest_win": metrics["largest_win"],
        "largest_loss": metrics["largest_loss"],
        "consecutive_wins": metrics["consecutive_wins"],
        "consecutive_losses": metrics["consecutive_losses"],
        "calmar_ratio": metrics["calmar_ratio"],
        "return_to_drawdown_ratio": return_to_drawdown_ratio,
    }
    return summary, trades


def run_v4_backtest_ranking(args: Any | None = None) -> pd.DataFrame:
    if args is None:
        args = type("Args", (), {"timeframe": Config.TIMEFRAME, "capital": 100000.0})()

    timeframe = getattr(args, "timeframe", Config.TIMEFRAME)
    starting_capital = float(getattr(args, "capital", 100000.0))

    handler = V4DataHandler()
    handler.refresh_gmx_cache(force=False)

    cache = ModelScalerCache()
    model, scaler = cache.load()
    data_handler = build_legacy_data_handler(str(cache.scaler_path), scaler=scaler)

    symbols = list_gmx_symbols(timeframe)
    if not symbols:
        raise FileNotFoundError(f"No GMX {timeframe} files found in {Config.GMX_OHLC_DIR}")

    summaries = []
    trades_by_symbol = {}
    for symbol in symbols:
        if _looks_like_stablecoin(symbol) or _looks_like_synthetic_commodity(symbol):
            logger.info("Skipping %s due to stablecoin/commodity filter", symbol)
            continue
        try:
            summary, trades = run_symbol_v4_backtest_ranking(model, data_handler, symbol, timeframe, starting_capital)
            summaries.append(summary)
            trades_by_symbol[symbol] = trades
        except Exception as exc:
            logger.warning("Backtest ranking failed for %s: %s", symbol, exc)

    summary_df = pd.DataFrame(summaries)
    ranked_df = filter_and_rank_backtest_summaries(summary_df)

    output_path = Path("v4_ranked_backtest_summary.csv")
    ranked_df.to_csv(output_path, index=False)
    report_path = Path("v4_backtest_research_report.html")
    write_v4_backtest_html_report(ranked_df, trades_by_symbol, report_path, title="V4 Ranked Backtest Research Report")

    print(f"V4 ranked backtest saved to {output_path}")
    print(f"V4 HTML report saved to {report_path}")
    display_columns = [
        "symbol",
        "return_pct",
        "final_capital",
        "total_profit",
        "signals_traded",
        "win_rate_pct",
        "max_drawdown_pct",
        "profit_factor",
        "average_trade_return",
        "return_to_drawdown_ratio",
    ]
    available_columns = [col for col in display_columns if col in ranked_df.columns]
    if not ranked_df.empty:
        print(ranked_df[available_columns].head(20).to_string(index=False))

    return ranked_df
