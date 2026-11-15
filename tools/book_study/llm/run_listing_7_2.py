"""Run LLMs in Finance Listing 7.2 (p242-243, headline-level sentiment scoring) exactly as printed, on the bot's symbols.

SOURCE TABLE (every line of what runs, and where it comes from)
| what                                                         | source                                                        |
|--------------------------------------------------------------|---------------------------------------------------------------|
| the scoring code: FinBERT (ProsusAI/finbert), score_headlines, | the book, Listing 7.2, p242-243; executed UNCHANGED from the  |
|   per-ticker per-day mean (daily_sent)                       | user's mirror LLMs-in-Finance/book_mirror/ch07.../02_p242-243.py |
| news_ticker.csv (date, ticker, headline)                     | DATA SOURCE, the one allowed change: Alpaca news headlines   |
|                                                              | from runtime/llm_news/<venue>/ (fetch_alpaca_news.py), only   |
|                                                              | the three columns the book names                              |
| writing daily_sent to daily_sent.csv after the book's code   | NOT in the book: the book leaves daily_sent in memory. Saving |
|   ends                                                       | it changes nothing in the computation; needed to check it.    |
Nothing else. No threshold, trading rule or merge with returns: the book prints none for Listing 7.2.
If the book's code fails as printed (for example memory: the book scores every headline in ONE call), this script
stops and the failure is reported. It is NOT patched.

Run on the Mac, from the project root, after fetch_alpaca_news.py has finished for the venue:
    python tools/book_study/llm/run_listing_7_2.py --venue kraken
    python tools/book_study/llm/run_listing_7_2.py --venue alpaca
Needs: pip install transformers torch   (FinBERT is downloaded from Hugging Face on first use)
Output: runtime/llm_news/<venue>/news_ticker.csv, daily_sent.csv, listing_7_2_run.json, listing_7_2_stdout.txt
"""
from __future__ import annotations
import argparse, json, os, runpy, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BOOK = Path.home() / "Documents/GitHub/LLMs-in-Finance/book_mirror/ch07_applications_of_llms_in_finance/02_p242-243.py"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["kraken", "alpaca"], required=True)
    ap.add_argument("--book-file", default=str(BOOK))
    a = ap.parse_args()
    book = Path(a.book_file)
    if not book.exists():
        sys.exit(f"Book file not found: {book}")
    folder = ROOT / "runtime" / "llm_news" / a.venue
    files = sorted(p for p in folder.glob("*.csv") if not p.name.startswith(("._", "news_ticker", "daily_sent")))
    if not files:
        sys.exit(f"No headline files in {folder}. Run fetch_alpaca_news.py --venue {a.venue} first.")
    news = pd.concat([pd.read_csv(f, usecols=["date", "ticker", "headline"]) for f in files], ignore_index=True)
    news = news.dropna(subset=["headline"])
    news[["date", "ticker", "headline"]].to_csv(folder / "news_ticker.csv", index=False)
    info = {"venue": a.venue, "book_file": str(book), "symbols_with_files": len(files), "headlines": int(len(news)),
            "tickers": int(news.ticker.nunique()), "first_date": str(news.date.min()), "last_date": str(news.date.max()),
            "started": datetime.now(timezone.utc).isoformat()}
    print(json.dumps(info, indent=1), flush=True)
    os.chdir(folder)                      # the book reads "news_ticker.csv" from the working directory
    t0 = time.time()
    try:
        g = runpy.run_path(str(book))     # the book's code, unchanged
    except BaseException as exc:          # report, do not patch
        info.update(status="FAILED AS PRINTED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-3000:],
                    seconds=round(time.time() - t0, 1))
        (folder / "listing_7_2_run.json").write_text(json.dumps(info, indent=1))
        print(f"\nSTOPPED: the book's code failed as printed: {info['error']}\nDetails in {folder/'listing_7_2_run.json'}")
        sys.exit(1)
    d = g["daily_sent"]; d.to_csv(folder / "daily_sent.csv", index=False)
    info.update(status="RAN AS PRINTED", seconds=round(time.time() - t0, 1), daily_rows=int(len(d)),
                sentiment_columns=[c for c in d.columns if c.startswith("sent_")], id2label={str(k): v for k, v in g["id2label"].items()})
    (folder / "listing_7_2_run.json").write_text(json.dumps(info, indent=1))
    print(f"\nRAN AS PRINTED in {info['seconds']} s: {len(d)} ticker-days -> {folder/'daily_sent.csv'}")

if __name__ == "__main__":
    main()
