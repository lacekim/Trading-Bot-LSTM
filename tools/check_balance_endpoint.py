"""Standalone diagnostic: hit the REAL /api/v1/balance endpoint on the
running freqtrade webserver (port from config.json) and print the raw
result -- the definitive check for whether ALPACA_BALANCE_BRIDGE (in
api_v1.py) is actually folding Alpaca's equity in, since freqUI's own
Dashboard page apparently has no separate Balance/pie-chart panel to
eyeball in this build.

Run from anywhere:  python tools/check_balance_endpoint.py
"""
import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "freqtrade-develop" / "user_data" / "config.json"

cfg = json.loads(CONFIG_PATH.read_text())
api = cfg.get("api_server", {})
port = api.get("listen_port", 8080)
username = api.get("username")
password = api.get("password")
base = f"http://127.0.0.1:{port}/api/v1"

if not username or not password:
    print(f"FAILED: no api_server username/password in {CONFIG_PATH}")
    sys.exit(1)

login = requests.post(f"{base}/token/login", auth=(username, password), timeout=10)
if login.status_code != 200:
    print(f"FAILED to log in: HTTP {login.status_code} {login.text[:300]}")
    sys.exit(1)
token = login.json()["access_token"]

resp = requests.get(f"{base}/balance", headers={"Authorization": f"Bearer {token}"}, timeout=10)
if resp.status_code != 200:
    print(f"FAILED to fetch /balance: HTTP {resp.status_code} {resp.text[:300]}")
    sys.exit(1)

data = resp.json()
print(f"total: {data.get('total')}")
print(f"total_bot: {data.get('total_bot')}")
print(f"value: {data.get('value')}")
print(f"value_bot: {data.get('value_bot')}")
print(f"note: {data.get('note')}")
print(f"currency count: {len(data.get('currencies', []))}")
alpaca_rows = [c for c in data.get("currencies", []) if c.get("currency") == "ALPACA"]
if alpaca_rows:
    print(f"RESULT: found ALPACA row -> {alpaca_rows[0]}")
else:
    print("RESULT: no ALPACA row found in currencies -- bridge did not add it.")
    print("Full currencies list:")
    for c in data.get("currencies", []):
        print(f"  {c.get('currency')}: balance={c.get('balance')}")
