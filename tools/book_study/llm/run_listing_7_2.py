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
| scoring the headlines 16 at a time and stacking the results  | the BOOK's own pattern, Listing 7.1 p239-241 (embed_text:     |
|   (one line of the book's 7.2 replaced, see BATCH_LINE)      | batch_size=16, loop over texts[i:i + batch_size], np.vstack). |
|                                                              | Needed because the single call as printed was killed for      |
|                                                              | memory on 53,800 headlines (BOOK_NOTES section 10). The user  |
|                                                              | said "Fallow the book" (2026-11-15).                          |
| equivalence check before the full run                        | NOT in the book: the first 256 headlines are scored by the    |
|                                                              | book's single call and by the 16-at-a-time loop; the run      |
|                                                              | stops if any probability differs by more than 1e-4.          |
| a progress line every 500 batches                            | NOT in the book: printing only; changes nothing computed.     |
Nothing else. No threshold, trading rule or merge with returns: the book prints none for Listing 7.2.
If the book's code fails, this script stops and the failure is reported. It is NOT patched further.

Run on the Mac, from the project root, after fetch_alpaca_news.py has finished for the venue:
    python tools/book_study/llm/run_listing_7_2.py --venue kraken
    python tools/book_study/llm/run_listing_7_2.py --venue alpaca
Needs: pip install transformers torch   (FinBERT is downloaded from Hugging Face on first use)
Output: runtime/llm_news/<venue>/news_ticker.csv, daily_sent.csv, listing_7_2_run.json, listing_7_2_stdout.txt
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BOOK = Path.home() / "Documents/GitHub/LLMs-in-Finance/book_mirror/ch07_applications_of_llms_in_finance/02_p242-243.py"

BOOK_LINE = 'scores = score_headlines(news_df["headline"].tolist())'
BATCH_LINE = (  # Listing 7.1's batching (embed_text, batch_size=16), applied to the book's own score_headlines
    "import numpy as np\n"
    "_texts = news_df[\"headline\"].tolist()\n"
    "_all = []\n"
    "for _i in range(0, len(_texts), 16):\n"
    "    _all.append(score_headlines(_texts[_i:_i + 16]))\n"
    "    if (_i // 16) % 500 == 0:\n"
    "        print(f\"scored {_i + len(_all[-1])} of {len(_texts)}\", flush=True)\n"
    "scores = np.vstack(_all)"
)


def book_code(book: Path) -> str:
    src = book.read_text(encoding="utf-8")
    if src.count(BOOK_LINE) != 1:
        sys.exit("The book file does not contain the expected scoring line exactly once; stopping.")
    return src.replace(BOOK_LINE, BATCH_LINE)


def run_code(code: str, name: str) -> dict:
    g = {"__name__": "__main__", "__file__": name}
    exec(compile(code, name, "exec"), g)
    return g


def equivalence_check(book: Path, folder: Path, news: pd.DataFrame) -> dict:
    """Score the first 256 headlines with the book's single call and with the 16-at-a-time loop; compare."""
    import numpy as np
    chk = folder / "_equivalence"
    chk.mkdir(exist_ok=True)
    news.head(256)[["date", "ticker", "headline"]].to_csv(chk / "news_ticker.csv", index=False)
    here = os.getcwd(); os.chdir(chk)
    try:
        g = run_code(book.read_text(encoding="utf-8"), str(book))      # the book exactly as printed, on 256 rows
        single = np.asarray(g["scores"])
        g2 = run_code(book_code(book), str(book))                     # the same with Listing 7.1's batching
        batched = np.asarray(g2["scores"])
    finally:
        os.chdir(here)
    diff = float(np.abs(single - batched).max())
    return {"rows": int(len(single)), "max_abs_difference": diff, "passed": bool(diff <= 1e-4)}


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
    eq = equivalence_check(book, folder, news)
    info["equivalence_check"] = eq
    print(f"equivalence check (book's single call vs 16 at a time, first {eq['rows']} headlines): "
          f"max difference {eq['max_abs_difference']:.2e} -> {'PASSED' if eq['passed'] else 'FAILED'}", flush=True)
    if not eq["passed"]:
        (folder / "listing_7_2_run.json").write_text(json.dumps(info, indent=1))
        sys.exit("STOPPED: the batched scores differ from the book's single call.")
    os.chdir(folder)                      # the book reads "news_ticker.csv" from the working directory
    t0 = time.time()
    try:
        g = run_code(book_code(book), str(book))   # the book's code, with Listing 7.1's batching on one line
    except BaseException as exc:          # report, do not patch
        info.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-3000:],
                    seconds=round(time.time() - t0, 1))
        (folder / "listing_7_2_run.json").write_text(json.dumps(info, indent=1))
        print(f"\nSTOPPED: the book's code failed: {info['error']}\nDetails in {folder/'listing_7_2_run.json'}")
        sys.exit(1)
    d = g["daily_sent"]; d.to_csv(folder / "daily_sent.csv", index=False)
    info.update(status="RAN (book code, Listing 7.1 batching)", seconds=round(time.time() - t0, 1), daily_rows=int(len(d)),
                sentiment_columns=[c for c in d.columns if c.startswith("sent_")], id2label={str(k): v for k, v in g["id2label"].items()})
    (folder / "listing_7_2_run.json").write_text(json.dumps(info, indent=1))
    print(f"\nRAN in {info['seconds']} s: {len(d)} ticker-days -> {folder/'daily_sent.csv'}")

if __name__ == "__main__":
    main()
