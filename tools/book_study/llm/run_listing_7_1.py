"""Run LLMs in Finance Listing 7.1 (p239-241, minimal hybrid forecasting pipeline: prices + news) per symbol, both venues.

SOURCE TABLE (every line of what runs, and where it comes from)
| what                                                          | source                                                       |
|---------------------------------------------------------------|--------------------------------------------------------------|
| the pipeline: next-day return target, daily headlines joined, | the book, Listing 7.1, p239-241; executed from the user's    |
|   MiniLM CLS embeddings (batch 16), lagged return, inner join, | mirror LLMs-in-Finance/book_mirror/ch07.../01_p239-241.py    |
|   70/30 time split, RandomForestRegressor(200, depth 6, 42)   |                                                              |
| 4 lines replaced in memory, DATA SOURCE only:                 | the one allowed change                                       |
|   import yfinance as yf          -> removed (no download)     |                                                              |
|   ticker = "SPY"                 -> each of the bot's symbols |                                                              |
|   data = yf.download(ticker, period="10y", interval="1d")    |                                                              |
|        -> the last 10 years of daily closes from the project: |                                                              |
|           Kraken: runtime/venue_candles_v2/kraken_1h resampled to UTC days; Alpaca: runtime/venue_candles/alpaca_1d |
|   news_df = pd.read_csv("daily_news.csv", parse_dates=["date"])                                                    |
|        -> that symbol's (date, headline) rows from runtime/llm_news/<venue>/news_ticker.csv                      |
| saving y_pred and y_test per date after the book's code ends  | NOT in the book: the book stops at y_pred; saved for testing |
| a symbol whose run raises an error is recorded and skipped    | NOT in the book (it runs one ticker); nothing is patched     |
Symbols: Kraken BTC ETH XMR PAXG SOL TRX; Alpaca the 55 golden-cross instruments. The book file is never edited.
Run on the Mac, from the project root:
    python tools/book_study/llm/run_listing_7_1.py --venue kraken
    python tools/book_study/llm/run_listing_7_1.py --venue alpaca
Needs transformers, torch, scikit-learn (all in the javier env). Downloads sentence-transformers/all-MiniLM-L6-v2 once.
Output: runtime/llm_news/<venue>/listing_7_1_preds.csv (ticker, date, y_pred, y_test), listing_7_1_run.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BOOK = Path.home() / "Documents/GitHub/LLMs-in-Finance/book_mirror/ch07_applications_of_llms_in_finance/01_p239-241.py"
KRAKEN = ["BTC", "ETH", "XMR", "PAXG", "SOL", "TRX"]
SUBS = [
    ('import yfinance as yf', '# import yfinance as yf  (data source replaced: prices come from the project)'),
    ('ticker = "SPY"', 'ticker = TICKER'),
    ('data = yf.download(ticker, period="10y", interval="1d")', 'data = LOAD_PRICES(ticker)'),
    ('news_df = pd.read_csv("daily_news.csv", parse_dates=["date"])', 'news_df = LOAD_NEWS(ticker)'),
]


def book_code(book: Path) -> str:
    src = book.read_text(encoding="utf-8")
    for old, new in SUBS:
        if src.count(old) != 1:
            sys.exit(f"Expected this line exactly once in the book file, stopping: {old}")
        src = src.replace(old, new)
    return src


def alpaca_symbols() -> list[str]:
    data = json.loads((ROOT / "runtime" / "cross_asset_golden_cross_results.json").read_text())
    fut = {"GC=F": "GLD", "PA=F": "PPLT"}; eq = {"BRK-B": "BRK.B"}
    skip = {"BTC-USD", "ETH-USD", "SOL-USD", "PAXG-USD"}
    return [fut.get(r["symbol"]) or eq.get(r["symbol"]) or r["symbol"] for r in data["results"] if r["robust"] and r["symbol"] not in skip]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--venue", choices=["kraken", "alpaca"], required=True)
    ap.add_argument("--book-file", default=str(BOOK)); a = ap.parse_args()
    book = Path(a.book_file)
    if not book.exists(): sys.exit(f"Book file not found: {book}")
    code = compile(book_code(book), str(book), "exec")
    news = pd.read_csv(ROOT / "runtime" / "llm_news" / a.venue / "news_ticker.csv")
    if a.venue == "kraken":
        syms = KRAKEN
        def load_prices(t):
            h = pd.read_csv(ROOT / "runtime" / "venue_candles_v2" / "kraken_1h" / f"{t}_1h.csv", parse_dates=["date"]).set_index("date")
            d = h["close"].resample("1D").last().dropna().to_frame("Close")
            return d[d.index > d.index[-1] - pd.DateOffset(years=10)]
    else:
        syms = alpaca_symbols()
        px = pd.read_csv(ROOT / "runtime" / "venue_candles" / "alpaca_1d" / "alpaca_daily.csv.gz", parse_dates=["date"])
        have = set(px.symbol.unique()); syms = [s for s in syms if s in have]
        def load_prices(t):
            d = px[px.symbol == t].set_index("date")["close"].sort_index().to_frame("Close")
            return d[d.index > d.index[-1] - pd.DateOffset(years=10)]
    def load_news(t):
        n = news[news.ticker == t][["date", "headline"]].copy(); n["date"] = pd.to_datetime(n["date"]); return n
    out = ROOT / "runtime" / "llm_news" / a.venue
    rows, report = [], {"venue": a.venue, "book_file": str(book), "symbols": syms, "started": datetime.now(timezone.utc).isoformat(), "per_symbol": {}}
    t0 = time.time()
    for i, s in enumerate(syms, 1):
        g = {"__name__": "__main__", "TICKER": s, "LOAD_PRICES": load_prices, "LOAD_NEWS": load_news}
        try:
            exec(code, g)
            yp = pd.Series(g["y_pred"], index=g["y_test"].index); yt = g["y_test"]
            for dt in yt.index:
                rows.append({"ticker": s, "date": str(pd.Timestamp(dt).date()), "y_pred": float(yp[dt]), "y_test": float(yt[dt])})
            report["per_symbol"][s] = {"status": "ran", "rows": int(len(g["X"])), "train": int(len(g["X_train"])), "test": int(len(yt)),
                                       "test_first": str(pd.Timestamp(yt.index[0]).date()), "test_last": str(pd.Timestamp(yt.index[-1]).date())}
            print(f"[{i}/{len(syms)}] {s}: {len(g['X'])} rows, test {len(yt)} from {pd.Timestamp(yt.index[0]).date()}", flush=True)
        except Exception as exc:
            report["per_symbol"][s] = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-1500:]}
            print(f"[{i}/{len(syms)}] {s}: ERROR {type(exc).__name__}: {exc}", flush=True)
    pd.DataFrame(rows).to_csv(out / "listing_7_1_preds.csv", index=False)
    report["seconds"] = round(time.time() - t0, 1); report["ran"] = sum(v["status"] == "ran" for v in report["per_symbol"].values())
    (out / "listing_7_1_run.json").write_text(json.dumps(report, indent=1))
    print(f"\nDone in {report['seconds']} s: {report['ran']} of {len(syms)} symbols ran -> {out/'listing_7_1_preds.csv'}")


if __name__ == "__main__":
    main()
