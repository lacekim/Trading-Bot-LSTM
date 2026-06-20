from argparse import ArgumentParser
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import talib
except ImportError:
    talib = None


GMX_OHLC_DIR = Path(
    "/Users/mike/Documents/GitHub/Algorithmic-Trading-with-Deep-Learning/data/GMX_OHLCVT"
)


def gmx_ohlc_path(symbol="GMX", timeframe="15m", data_dir=GMX_OHLC_DIR):
    path = Path(data_dir) / f"gmx_arbitrum_{symbol.upper()}_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"GMX OHLC file not found: {path}")
    return path


def list_gmx_symbols(timeframe="15m", data_dir=GMX_OHLC_DIR):
    data_dir = Path(data_dir)
    suffix = f"_{timeframe}"
    symbols = []
    for path in data_dir.glob(f"gmx_arbitrum_*{suffix}.csv"):
        symbol = path.stem.removeprefix("gmx_arbitrum_").removesuffix(suffix)
        symbols.append(symbol)
    return sorted(symbols)


def load_gmx_ohlc_file(path):
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "open_time": "Date",
        }
    )

    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns)
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["Open", "High", "Low", "Close", "Volume"]], path


def load_gmx_ohlc(symbol="GMX", timeframe="15m", data_dir=GMX_OHLC_DIR):
    path = gmx_ohlc_path(symbol, timeframe, data_dir)
    return load_gmx_ohlc_file(path)


def rsi(series, period=14):
    if talib is not None:
        return pd.Series(talib.RSI(series.to_numpy(dtype=float), timeperiod=period), index=series.index)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stoch_rsi(series, period=14, smooth_k=3):
    if talib is not None:
        fast_k, _ = talib.STOCHRSI(
            series.to_numpy(dtype=float),
            timeperiod=period,
            fastk_period=period,
            fastd_period=smooth_k,
            fastd_matype=0,
        )
        return pd.Series(fast_k, index=series.index)

    rsi_values = rsi(series, period)
    lowest = rsi_values.rolling(period).min()
    highest = rsi_values.rolling(period).max()
    raw_stoch = 100 * (rsi_values - lowest) / (highest - lowest).replace(0, np.nan)
    return raw_stoch.rolling(smooth_k).mean()


def add_indicators(df, ma_window=200):
    df = df.copy()
    df["MA_200"] = df["Close"].rolling(window=ma_window).mean()
    df["RSI"] = rsi(df["Close"], period=14)
    df["StochRSI"] = stoch_rsi(df["Close"], period=14)
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Trend"] = df["Volume"] / df["Volume_MA"].replace(0, np.nan)
    df["Volume_Trend"] = df["Volume_Trend"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return df.dropna(subset=["MA_200", "RSI", "StochRSI"])


def generate_signals(df):
    df = df.copy()

    long_signal = (
        (df["Close"] > df["MA_200"]) &
        (df["StochRSI"].shift(1) < 20) &
        (df["StochRSI"] >= 20)
    )

    short_signal = (
        (df["Close"] < df["MA_200"]) &
        (df["StochRSI"].shift(1) > 80) &
        (df["StochRSI"] <= 80)
    )

    df["Signal"] = np.select([long_signal, short_signal], [-1, 1], default=0)
    return df


def run_backtest(
    df,
    starting_capital=100000.0,
    stop_loss_percent=0.02,
    take_profit_percent=0.05,
):
    capital = starting_capital
    position = 0
    units = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    equity_curve = []
    trade_log = []

    for timestamp, row in df.iterrows():
        close_price = float(row["Close"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        signal = int(row["Signal"])

        if position == 0 and signal != 0:
            position = signal
            entry_price = close_price
            units = capital / entry_price
            stop_loss = entry_price * (1 - stop_loss_percent if position == 1 else 1 + stop_loss_percent)
            take_profit = entry_price * (1 + take_profit_percent if position == 1 else 1 - take_profit_percent)
            trade_log.append(
                {
                    "timestamp": timestamp,
                    "action": "Enter Long" if position == 1 else "Enter Short",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "capital": capital,
                }
            )

        elif position == 1:
            exit_price = None
            exit_reason = None
            if low_price <= stop_loss:
                exit_price = stop_loss
                exit_reason = "Stop Loss"
            elif high_price >= take_profit:
                exit_price = take_profit
                exit_reason = "Take Profit"
            elif signal == -1:
                exit_price = close_price
                exit_reason = "Opposite Signal"

            if exit_price is None:
                unrealized = (close_price - entry_price) * units
                equity_curve.append({"timestamp": timestamp, "equity": capital + unrealized})
                continue

            pnl = (exit_price - entry_price) * units
            capital += pnl
            trade_log.append(
                {
                    "timestamp": timestamp,
                    "action": f"Exit Long ({exit_reason})",
                    "exit_price": exit_price,
                    "profit": pnl,
                    "capital": capital,
                }
            )
            position = 0

        elif position == -1:
            exit_price = None
            exit_reason = None
            if high_price >= stop_loss:
                exit_price = stop_loss
                exit_reason = "Stop Loss"
            elif low_price <= take_profit:
                exit_price = take_profit
                exit_reason = "Take Profit"
            elif signal == 1:
                exit_price = close_price
                exit_reason = "Opposite Signal"

            if exit_price is None:
                unrealized = (entry_price - close_price) * units
                equity_curve.append({"timestamp": timestamp, "equity": capital + unrealized})
                continue

            pnl = (entry_price - exit_price) * units
            capital += pnl
            trade_log.append(
                {
                    "timestamp": timestamp,
                    "action": f"Exit Short ({exit_reason})",
                    "exit_price": exit_price,
                    "profit": pnl,
                    "capital": capital,
                }
            )
            position = 0

        unrealized = 0.0
        if position == 1:
            unrealized = (close_price - entry_price) * units
        elif position == -1:
            unrealized = (entry_price - close_price) * units
        equity_curve.append({"timestamp": timestamp, "equity": capital + unrealized})

    if position != 0:
        final_price = float(df["Close"].iloc[-1])
        pnl = (final_price - entry_price) * units if position == 1 else (entry_price - final_price) * units
        capital += pnl
        trade_log.append(
            {
                "timestamp": df.index[-1],
                "action": "Exit Long (End)" if position == 1 else "Exit Short (End)",
                "exit_price": final_price,
                "profit": pnl,
                "capital": capital,
            }
        )
        equity_curve[-1]["equity"] = capital

    trades = pd.DataFrame(trade_log)
    equity = pd.DataFrame(equity_curve).set_index("timestamp")
    return capital, trades, equity


def summarize_backtest(starting_capital, final_capital, trades, equity):
    metrics = backtest_metrics(starting_capital, final_capital, trades, equity)

    print(f"Final Capital: ${metrics['final_capital']:,.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Closed Trades: {metrics['closed_trades']}")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")


def backtest_metrics(starting_capital, final_capital, trades, equity):
    exits = trades[trades["action"].str.startswith("Exit")] if not trades.empty else trades
    wins = exits[exits["profit"] > 0] if not exits.empty else exits
    return {
        "final_capital": final_capital,
        "total_return_pct": (final_capital / starting_capital - 1) * 100,
        "closed_trades": len(exits),
        "win_rate_pct": (len(wins) / len(exits) * 100) if len(exits) else 0.0,
        "max_drawdown_pct": (equity["equity"] / equity["equity"].cummax() - 1).min() * 100,
    }


def plot_results(df, equity, output_path="gmx_backtest.png"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(df.index, df["Close"], label="Close Price")
    axes[0].plot(df.index, df["MA_200"], label="200-period MA")
    axes[0].scatter(df.index[df["Signal"] == 1], df.loc[df["Signal"] == 1, "Close"], marker="^", label="Long Signal")
    axes[0].scatter(df.index[df["Signal"] == -1], df.loc[df["Signal"] == -1, "Close"], marker="v", label="Short Signal")
    axes[0].legend()
    axes[0].set_title("GMX OHLC Backtest Signals")

    axes[1].plot(equity.index, equity["equity"], label="Equity")
    axes[1].legend()
    axes[1].set_title("Equity Curve")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return output_path


def parse_args():
    parser = ArgumentParser(description="Run a GMX OHLC backtest.")
    parser.add_argument("--symbol", default="GMX", help="GMX market symbol, e.g. GMX, ADA, BTC")
    parser.add_argument("--timeframe", default="15m", help="CSV timeframe suffix, e.g. 5m or 15m")
    parser.add_argument("--all-assets", action="store_true", help="Backtest every GMX asset for the selected timeframe")
    parser.add_argument("--capital", type=float, default=100000.0, help="Starting capital")
    parser.add_argument("--data-dir", default=str(GMX_OHLC_DIR), help="Directory containing GMX OHLC CSV files")
    parser.add_argument("--plot", default="gmx_backtest.png", help="Output path for the backtest plot")
    return parser.parse_args()

def evaluate_signal_accuracy(df, horizon=5):
    results = []

    for i in range(len(df) - horizon):
        signal = df["Signal"].iloc[i]
        if signal == 0:
            continue

        entry_price = df["Close"].iloc[i]
        future_price = df["Close"].iloc[i + horizon]

        if signal == 1:
            win = future_price > entry_price
        else:
            win = future_price < entry_price

        results.append(win)

    if len(results) == 0:
        print("No trades")
        return

    accuracy = sum(results) / len(results) * 100
    print(f"Signal Accuracy ({horizon} bars): {accuracy:.2f}%")
    print(f"Total Signals: {len(results)}")


def signal_accuracy(df, horizon=5):
    results = []
    for i in range(len(df) - horizon):
        signal = df["Signal"].iloc[i]
        if signal == 0:
            continue

        entry_price = df["Close"].iloc[i]
        future_price = df["Close"].iloc[i + horizon]
        results.append(future_price > entry_price if signal == 1 else future_price < entry_price)

    return (sum(results) / len(results) * 100, len(results)) if results else (0.0, 0)


def run_asset_backtest(symbol, timeframe, data_dir, capital):
    df, path = load_gmx_ohlc(symbol, timeframe, data_dir)
    df = generate_signals(add_indicators(df))
    final_capital, trades, equity = run_backtest(df, starting_capital=capital)
    metrics = backtest_metrics(capital, final_capital, trades, equity)
    accuracy_3, signal_count = signal_accuracy(df, horizon=3)
    accuracy_5, _ = signal_accuracy(df, horizon=5)
    accuracy_10, _ = signal_accuracy(df, horizon=10)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "signals": signal_count,
        "signal_accuracy_3_pct": accuracy_3,
        "signal_accuracy_5_pct": accuracy_5,
        "signal_accuracy_10_pct": accuracy_10,
        **metrics,
        "source": path,
    }


def run_all_assets(args):
    rows = []
    symbols = list_gmx_symbols(args.timeframe, args.data_dir)
    if not symbols:
        raise FileNotFoundError(f"No GMX {args.timeframe} files found in {args.data_dir}")

    for symbol in symbols:
        try:
            rows.append(run_asset_backtest(symbol, args.timeframe, args.data_dir, args.capital))
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": args.timeframe,
                    "error": str(exc),
                }
            )

    summary = pd.DataFrame(rows)
    if "total_return_pct" in summary.columns:
        summary = summary.sort_values("total_return_pct", ascending=False, na_position="last")

    output_path = f"gmx_all_assets_{args.timeframe}_summary.csv"
    summary.to_csv(output_path, index=False)

    print(f"Backtested {len(symbols)} assets for {args.timeframe}")
    print(f"Summary saved to {output_path}")
    display_columns = [
        "symbol",
        "total_return_pct",
        "final_capital",
        "closed_trades",
        "win_rate_pct",
        "max_drawdown_pct",
    ]
    available_columns = [col for col in display_columns if col in summary.columns]
    print(summary[available_columns].head(20).to_string(index=False))


def main():
    args = parse_args()
    if args.all_assets:
        run_all_assets(args)
        return

    df, path = load_gmx_ohlc(args.symbol, args.timeframe, args.data_dir)
    df = generate_signals(add_indicators(df))
    evaluate_signal_accuracy(df, horizon=3)
    evaluate_signal_accuracy(df, horizon=5)
    evaluate_signal_accuracy(df, horizon=10)
    final_capital, trades, equity = run_backtest(df, starting_capital=args.capital)

    print(f"Loaded {len(df):,} candles from {path}")
    print(f"Backtest range: {df.index.min()} -> {df.index.max()}")
    summarize_backtest(args.capital, final_capital, trades, equity)

    if not trades.empty:
        trades.to_csv("gmx_trade_log.csv", index=False)
        print("Trade log saved to gmx_trade_log.csv")

    plot_path = plot_results(df, equity, args.plot)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
