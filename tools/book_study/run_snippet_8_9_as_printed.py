"""AFML Snippet 8.9 (p75, "Calling all components") run with the book's own defaults:
testFunc(n_features=40, n_informative=10, n_redundant=10, n_estimators=1000, n_samples=10000, cv=10).
Synthetic data from Snippet 8.7 (getTestData); no market data, so there is no data source to swap.

SOURCE TABLE
| what | source |
|---|---|
| testFunc (8.9), getTestData (8.7), featImportance (8.8), plotFeatImportance (8.10), the MDI / MDA / SFI functions (8.2-8.4) | book_port_b.py, the October ports; their forced changes (izip -> zip; base_estimator -> estimator; max_samples as an integer number of draws for scikit-learn >= 1.7; Agg plotting backend) are listed in the file header and remain open items for the user's ruling |
| the call: testFunc() with every printed default; output written by the book's own code to ./testFunc/ (stats.csv and one plot per method) | the book, p75 |
| running inside runtime/snippet_8_9/ so ./testFunc/ lands there; os.makedirs of that folder (inside the port) | NOT in the book (the book assumes the folder exists) |
| `if __name__ == "__main__":` guard | NOT in the book: macOS multiprocessing needs it |
What this replaces: the October ports run did 8.7-8.9 "at a reduced size" (fewer samples, 100 trees); not in the book.
Run on the Mac from the project root:  python tools/book_study/run_snippet_8_9_as_printed.py
"""
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import book_port_b as PB

if __name__ == "__main__":
    out = HERE.parents[1] / "runtime" / "snippet_8_9"; out.mkdir(parents=True, exist_ok=True); os.chdir(out)
    t0 = time.time(); print("Snippet 8.9: testFunc() with the printed defaults (40 features, 10,000 samples, 1,000 trees, 10-fold CV) ...", flush=True)
    stats = PB.testFunc()
    print(stats.to_string())
    (out / "run.json").write_text(json.dumps({"call": "testFunc()", "defaults": {"n_features": 40, "n_informative": 10, "n_redundant": 10, "n_estimators": 1000, "n_samples": 10000, "cv": 10},
        "seconds": round(time.time() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat(), "stats": stats.to_dict(orient="records")}, indent=1, default=str))
    print(f"\nsaved to {out} in {time.time() - t0:.0f} s")
