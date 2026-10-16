"""Standalone diagnostic: is Alpaca actually reachable from wherever this
is run? Prints whether the keys are visible, and if so, hits Alpaca's
real API for account equity + open position count. No other project
imports needed -- run this directly, from any directory:

    python tools/check_alpaca_env.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    loaded = load_dotenv(PROJECT_ROOT / ".env")
    print(f".env path checked: {PROJECT_ROOT / '.env'} (exists: {(PROJECT_ROOT / '.env').exists()}, load_dotenv returned: {loaded})")
except Exception as exc:
    print(f"FAILED to import/run dotenv: {exc}")
    sys.exit(1)

api_key = os.environ.get("ALPACA_API_KEY")
secret_key = os.environ.get("ALPACA_SECRET_KEY")
print(f"ALPACA_API_KEY visible in os.environ: {bool(api_key)} (len={len(api_key) if api_key else 0})")
print(f"ALPACA_SECRET_KEY visible in os.environ: {bool(secret_key)} (len={len(secret_key) if secret_key else 0})")

if not api_key or not secret_key:
    print("RESULT: keys not visible in this process's environment -- this is why "
          "_alpaca_positions()/alpaca_total_equity() return empty/None. Check "
          "whatever shell/session actually launched the process against this .env.")
    sys.exit(1)

try:
    from alpaca.trading.client import TradingClient
    client = TradingClient(api_key, secret_key, paper=True)
    account = client.get_account()
    positions = client.get_all_positions()
    print(f"RESULT: Alpaca API call succeeded. equity=${account.equity}, cash=${account.cash}, open_positions={len(positions)}")
except Exception as exc:
    print(f"RESULT: keys ARE visible but the Alpaca API call itself failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
