"""AFML Snippet 4.9 (p41, "Multi-threaded Monte Carlo") run with the book's own call: mainMC() with its printed defaults
numObs=10, numBars=100, maxH=5, numIters=1E6, numThreads=24.  No market data is involved, so there is no data source to swap.

SOURCE TABLE
| what | source |
|---|---|
| mainMC, auxMC (Snippet 4.8), getIndMatrix / getAvgUniqueness / seqBootstrap (4.3-4.5), processJobs / processJobs_ (Snippets 20.8-20.10) | book_port_a.py / book_mp.py, the October ports; their forced Python 3 changes are listed in each file header (xrange -> range; print -> the describe() table is returned) and are still open items for the user's ruling |
| the call: mainMC() with every printed default | the book, p41 |
| printing the returned table and saving it to runtime/snippet_4_9/ | NOT in the book (the book prints it inside mainMC); needed to keep the result |
| `if __name__ == "__main__":` guard | NOT in the book: required by Python's multiprocessing on macOS so the 24 worker processes do not re-run this script |
What this replaces: the earlier ports run (ports_sections_a.py) used 300 iterations and venue-calibrated numObs / numBars / maxH; neither is in the book.
Run on the Mac from the project root:  python tools/book_study/run_snippet_4_9_as_printed.py
"""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import book_port_a as PA

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "runtime" / "snippet_4_9"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); print("Snippet 4.9: mainMC() with the printed defaults (1,000,000 iterations, 24 threads) ...", flush=True)
    tab = PA.mainMC()
    print(tab)
    tab.to_csv(out / "mainMC_describe.csv")
    (out / "run.json").write_text(json.dumps({"call": "mainMC()", "defaults": {"numObs": 10, "numBars": 100, "maxH": 5, "numIters": 1e6, "numThreads": 24},
        "seconds": round(time.time() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat(), "table": tab.to_dict()}, indent=1, default=str))
    print(f"\nsaved to {out} in {time.time() - t0:.0f} s")
