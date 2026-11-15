"""Pull hourly Santiment social history for the six Kraken assets the golden cross trades and save it to disk.

Why: the two earlier Santiment experiments fetched their history live and never saved it; only 45 days of BTC are on
disk (market_data_collectors/data/santiment_btc_social.csv). A sentiment test against the golden cross bar needs a
dated series per asset.

The API key is read from .env (SANTIMENT_API_KEY=...). It is never printed and never written anywhere else.
Output: runtime/santiment/<ASSET>_social_1h.csv (columns: datetime, social_volume_total, sentiment_weighted_total,
social_dominance_total) and runtime/santiment/manifest.json. runtime/ is in .gitignore, so the data is not committed.
Same call the project's working scripts use: san.get("<metric>/<slug>", from_date=..., to_date=..., interval="1h").
Santiment restricts how far back and how close to today a plan may query; the script asks in 60-day pieces, records
exactly which pieces the API refused and why, and keeps whatever it was given.

Run from the project root, in the conda env that has sanpy (the one the collector ran in):
    python tools/pull_santiment_history.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "santiment"
SLUGS = {"BTC": "bitcoin", "ETH": "ethereum", "XMR": "monero", "PAXG": "pax-gold", "SOL": "solana", "TRX": "tron"}
METRICS = ["social_volume_total", "sentiment_weighted_total", "social_dominance_total"]
START = datetime(2023, 1, 1, tzinfo=timezone.utc)   # a year before the 2024-01-01 test window, for baselines
PIECE_DAYS = 60
PAUSE_SECONDS = 1.0


def read_key() -> str:
    env = ROOT / ".env"
    if not env.exists():
        sys.exit("No .env file in the project root.")
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("SANTIMENT_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key:
                return key
    sys.exit("SANTIMENT_API_KEY is not set in .env. Add a line: SANTIMENT_API_KEY=your_token")


def main() -> None:
    try:
        import san
    except ImportError:
        sys.exit("The 'san' package is not installed in this environment. Run: pip install sanpy")
    san.ApiConfig.api_key = read_key()
    OUT.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc)
    manifest = {"pulled_at": end.isoformat(), "start_requested": START.isoformat(), "assets": {}}
    for asset, slug in SLUGS.items():
        frames, refused = {}, []
        for metric in METRICS:
            parts = []
            t0 = START
            while t0 < end:
                t1 = min(t0 + timedelta(days=PIECE_DAYS), end)
                try:
                    df = san.get(f"{metric}/{slug}", from_date=t0.isoformat(), to_date=t1.isoformat(), interval="1h")
                    if df is not None and not df.empty:
                        s = df["value"]
                        s.index = pd.to_datetime(s.index, utc=True)
                        parts.append(s)
                except Exception as exc:  # the API explains plan limits in the error text; keep it
                    refused.append({"metric": metric, "from": t0.date().isoformat(), "to": t1.date().isoformat(),
                                    "reason": str(exc)[:300]})
                time.sleep(PAUSE_SECONDS)
                t0 = t1
            if parts:
                s = pd.concat(parts)
                frames[metric] = s[~s.index.duplicated(keep="last")].sort_index()
        if frames:
            table = pd.DataFrame(frames)
            table.index.name = "datetime"
            table.to_csv(OUT / f"{asset}_social_1h.csv")
            info = {"slug": slug, "rows": int(len(table)), "first": str(table.index[0]), "last": str(table.index[-1]),
                    "metrics_received": list(frames), "pieces_refused": len(refused), "refusals": refused[:12]}
        else:
            info = {"slug": slug, "rows": 0, "pieces_refused": len(refused), "refusals": refused[:12]}
        manifest["assets"][asset] = info
        print(f"{asset:5s} rows={info['rows']:6d} first={info.get('first', '-')} last={info.get('last', '-')} "
              f"refused pieces={info['pieces_refused']}")
        if refused:
            print(f"      first refusal: {refused[0]['from']}..{refused[0]['to']} {refused[0]['reason'][:160]}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}/  (manifest.json lists every refused piece)")


if __name__ == "__main__":
    main()
