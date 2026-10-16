"""Standalone diagnostic: calls the exact function freqUI's chart bridge
uses (fetch_alpaca_or_crypto_candles) for BTC, to spot-check the
crypto-bars fallback path after the G12 fix (that edit only added a
flush=True/EMPTY print here, no logic change -- this confirms it still
works). Run from the project root:

    python tools/check_crypto_candle_bridge.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_combined_status_report import fetch_alpaca_or_crypto_candles

df, tf = fetch_alpaca_or_crypto_candles("BTC", "4h", 500)
print("resolved_timeframe:", tf)
print("rows:", len(df))
if not df.empty:
    print(df.tail(3))
else:
    print("EMPTY DATAFRAME")
