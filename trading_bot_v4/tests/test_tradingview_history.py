import pandas as pd

from trading_bot_v4.features.tradingview_history import (
    import_tradingview_csv,
    prepend_tradingview_history,
)


def test_import_and_prepend_preserves_gmx_overlap(tmp_path):
    source = tmp_path / "BINANCE_XRPUSDT.csv"
    pd.DataFrame(
        {
            "time": [1704067200, 1704070800, 1704074400],
            "open": [1.0, 1.1, 99.0],
            "high": [1.2, 1.3, 100.0],
            "low": [0.9, 1.0, 98.0],
            "close": [1.1, 1.2, 99.0],
            "Volume": [10, 11, 12],
        }
    ).to_csv(source, index=False)

    result = import_tradingview_csv(source, tmp_path / "history")
    assert result.symbol == "XRP"
    assert result.rows == 3

    gmx = pd.DataFrame(
        {"Open": [2.0], "High": [2.2], "Low": [1.9], "Close": [2.1], "Volume": [20]},
        index=pd.DatetimeIndex(["2024-01-01 02:00:00"], name="Date"),
    )
    combined = prepend_tradingview_history(gmx, "XRP", "1h", tmp_path / "history")
    assert len(combined) == 3
    assert combined.loc[pd.Timestamp("2024-01-01 02:00:00"), "Close"] == 2.1
