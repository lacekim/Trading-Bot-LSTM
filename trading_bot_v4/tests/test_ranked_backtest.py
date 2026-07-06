import pandas as pd

from trading_bot_v4.backtesting.ranking_engine import filter_and_rank_backtest_summaries


def test_filter_and_rank_backtest_summaries_excludes_invalid_symbols_and_sorts_by_risk_metrics():
    summaries = pd.DataFrame([
        {
            "symbol": "GMX",
            "candles": 1500,
            "signals_traded": 80,
            "return_pct": 18.5,
            "final_capital": 118500.0,
            "total_profit": 18500.0,
            "win_rate_pct": 60.0,
            "max_drawdown_pct": -10.0,
            "profit_factor": 1.7,
            "average_trade_return": 1.4,
            "return_to_drawdown_ratio": 1.85,
        },
        {
            "symbol": "ETH",
            "candles": 1200,
            "signals_traded": 70,
            "return_pct": 8.0,
            "final_capital": 108000.0,
            "total_profit": 8000.0,
            "win_rate_pct": 55.0,
            "max_drawdown_pct": -20.0,
            "profit_factor": 1.1,
            "average_trade_return": 0.8,
            "return_to_drawdown_ratio": 0.4,
        },
        {
            "symbol": "USDC",
            "candles": 1500,
            "signals_traded": 80,
            "return_pct": 5.0,
            "final_capital": 105000.0,
            "total_profit": 5000.0,
            "win_rate_pct": 50.0,
            "max_drawdown_pct": -5.0,
            "profit_factor": 1.5,
            "average_trade_return": 0.5,
            "return_to_drawdown_ratio": 1.0,
        },
        {
            "symbol": "XAU",
            "candles": 1500,
            "signals_traded": 80,
            "return_pct": 6.0,
            "final_capital": 106000.0,
            "total_profit": 6000.0,
            "win_rate_pct": 50.0,
            "max_drawdown_pct": -5.0,
            "profit_factor": 1.5,
            "average_trade_return": 0.6,
            "return_to_drawdown_ratio": 1.2,
        },
        {
            "symbol": "BTC",
            "candles": 900,
            "signals_traded": 80,
            "return_pct": 9.0,
            "final_capital": 109000.0,
            "total_profit": 9000.0,
            "win_rate_pct": 50.0,
            "max_drawdown_pct": -6.0,
            "profit_factor": 1.5,
            "average_trade_return": 0.7,
            "return_to_drawdown_ratio": 1.5,
        },
        {
            "symbol": "SOL",
            "candles": 1500,
            "signals_traded": 40,
            "return_pct": 12.0,
            "final_capital": 112000.0,
            "total_profit": 12000.0,
            "win_rate_pct": 58.0,
            "max_drawdown_pct": -8.0,
            "profit_factor": 1.6,
            "average_trade_return": 1.1,
            "return_to_drawdown_ratio": 1.5,
        },
    ])

    ranked = filter_and_rank_backtest_summaries(summaries)

    assert list(ranked["symbol"]) == ["GMX", "SOL", "ETH"]
    assert "USDC" not in ranked["symbol"].values
    assert "XAU" not in ranked["symbol"].values
    assert "BTC" not in ranked["symbol"].values
