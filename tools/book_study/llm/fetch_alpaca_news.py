"""Step 1 of adding LLMs in Finance to the AFML bot: the DATA SOURCE for the book's headline scorer.

The book's Listing (Chapter 7, p242-243, `score_headlines`) reads
    news_df = pd.read_csv("news_ticker.csv")  # columns: date, ticker, headline
This script builds that file for the bot's own symbols on BOTH venues, from Alpaca's historical
news API (it covers equities and crypto). Changing the data source is the one change allowed;
nothing in the book's code is changed here, and no existing project file is read for writing.

Symbols:  Kraken = the 27 files in runtime/dollar_bars/kraken/bars (BTC -> news symbol BTCUSD)
          Alpaca = the files in runtime/dollar_bars/alpaca/bars (AAPL -> AAPL)
Output:   runtime/llm_news/<venue>/<SYMBOL>.csv  with columns
              date (UTC calendar day of created_at), ticker (bot symbol), headline,
              created_at (full UTC timestamp, kept for the point-in-time rule a_j <= t),
              news_id, source
          runtime/llm_news/<venue>/_done/<SYMBOL>.json  (count, first, last) -> resumable
          runtime/llm_news/fetch_report.json
Keys:     ALPACA_API_KEY / ALPACA_SECRET_KEY from the environment or the project's .env.
          They are only sent to data.alpaca.markets in request headers; never printed or written.

Usage (from the project root, in a terminal with internet access):
    python tools/book_study/llm/fetch_alpaca_news.py --venue kraken
    python tools/book_study/llm/fetch_alpaca_news.py --venue alpaca
    optional: --symbols BTC,ETH   --start 2015-01-01   --end 2026-11-01
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
URL = "https://data.alpaca.markets/v1beta1/news"
PAGE = 50                 # API maximum per page
PAUSE = 0.32              # ~190 requests/minute, under the 200/minute limit


def load_keys() -> tuple[str, str]:
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    env = ROOT / ".env"
    if (not key or not sec) and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if k.strip() == "ALPACA_API_KEY" and not key:
                    key = v
                elif k.strip() == "ALPACA_SECRET_KEY" and not sec:
                    sec = v
    if not key or not sec:
        sys.exit("ALPACA_API_KEY / ALPACA_SECRET_KEY not found in the environment or .env")
    return key, sec


def bot_symbols(venue: str) -> list[str]:
    folder = ROOT / "runtime" / "dollar_bars" / venue / "bars"
    syms = sorted(p.stem for p in folder.glob("*.csv") if not p.name.startswith("._"))
    if not syms:
        sys.exit(f"no symbol files in {folder}")
    return syms


def news_symbol(venue: str, sym: str) -> str:
    return f"{sym}USD" if venue == "kraken" else sym


def get(params: dict, key: str, sec: str) -> dict:
    req = urllib.request.Request(
        URL + "?" + urllib.parse.urlencode(params),
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec, "Accept": "application/json"},
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            sys.exit(f"HTTP {e.code} from the news API for {params.get('symbols')}: {e.read()[:300]!r}")
        except urllib.error.URLError:
            time.sleep(min(60, 2 ** attempt))
    sys.exit(f"news API unreachable after retries for {params.get('symbols')}")


def fetch_symbol(venue: str, sym: str, start: str, end: str, key: str, sec: str, out: Path) -> dict:
    ns = news_symbol(venue, sym)
    tmp = out / f"{sym}.csv.part"
    n, first, last, token = 0, None, None, None
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "ticker", "headline", "created_at", "news_id", "source"])
        while True:
            p = {"symbols": ns, "start": start, "end": end, "limit": PAGE, "sort": "asc",
                 "include_content": "false"}
            if token:
                p["page_token"] = token
            data = get(p, key, sec)
            for a in data.get("news", []):
                ts = a.get("created_at", "")
                head = (a.get("headline") or "").replace("\r", " ").replace("\n", " ").strip()
                if not ts or not head:
                    continue
                w.writerow([ts[:10], sym, head, ts, a.get("id", ""), a.get("source", "")])
                n += 1
                first = first or ts
                last = ts
            token = data.get("next_page_token")
            time.sleep(PAUSE)
            if not token:
                break
    tmp.replace(out / f"{sym}.csv")
    info = {"venue": venue, "symbol": sym, "news_symbol": ns, "headlines": n, "first": first, "last": last,
            "start": start, "end": end, "fetched_at": datetime.now(timezone.utc).isoformat()}
    (out / "_done" / f"{sym}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["kraken", "alpaca"], required=True)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    a = ap.parse_args()

    key, sec = load_keys()
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] or bot_symbols(a.venue)
    out = ROOT / "runtime" / "llm_news" / a.venue
    (out / "_done").mkdir(parents=True, exist_ok=True)
    start, end = f"{a.start}T00:00:00Z", f"{a.end}T23:59:59Z"

    rows = []
    for i, s in enumerate(syms, 1):
        done = out / "_done" / f"{s}.json"
        if done.exists():
            info = json.loads(done.read_text(encoding="utf-8"))
            print(f"[{i}/{len(syms)}] {s}: already done ({info['headlines']} headlines)")
        else:
            info = fetch_symbol(a.venue, s, start, end, key, sec, out)
            print(f"[{i}/{len(syms)}] {s} ({info['news_symbol']}): {info['headlines']} headlines, "
                  f"{info['first']} .. {info['last']}")
        rows.append(info)

    report = ROOT / "runtime" / "llm_news" / "fetch_report.json"
    old = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    old[a.venue] = {"symbols": len(rows), "with_news": sum(r["headlines"] > 0 for r in rows),
                    "headlines": sum(r["headlines"] for r in rows),
                    "zero_news": [r["symbol"] for r in rows if r["headlines"] == 0]}
    report.write_text(json.dumps(old, indent=2), encoding="utf-8")
    print(json.dumps(old[a.venue], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
