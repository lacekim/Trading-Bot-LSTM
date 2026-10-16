"""Standalone diagnostic: calls the exact function freqUI's equities chart
bridge uses (fetch_alpaca_or_crypto_candles) directly for XOM and WFC, to
confirm the G12 NameError fix (originally verified only for VZ) also
recovered the other bridged equities pairs. Run from the project root:

    python tools/check_equities_candle_bridge.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_combined_status_report import fetch_alpaca_or_crypto_candles

for symbol in ("XOM", "WFC"):
    print(f"=== {symbol} ===")
    df, tf = fetch_alpaca_or_crypto_candles(symbol, "4h", 500)
    print("resolved_timeframe:", tf)
    print("rows:", len(df))
    if not df.empty:
        print(df.tail(3))
    else:
        print("EMPTY DATAFRAME")
    print()
