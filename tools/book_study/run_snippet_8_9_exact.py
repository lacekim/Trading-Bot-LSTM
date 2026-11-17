"""Runs AFML Snippet 8.9's testFunc() from the exact copy (snippet_8_9_as_printed.py) with every printed default
(40 features, 10 informative, 10 redundant, 1,000 trees, 10,000 samples, 10-fold CV).
NOT in the book: running inside runtime/snippet_8_9_exact/ and creating its ./testFunc/ folder (the book writes there but
assumes the folder exists); the `if __name__ == "__main__":` guard (macOS multiprocessing needs it); the timing line.
Replaces the 2026-11-16 run, which went through the October port (book_port_b.py) and its extra arguments.
Run on the Mac from the project root:  python tools/book_study/run_snippet_8_9_exact.py
"""
import os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if __name__ == "__main__":
    out = HERE.parents[1] / "runtime" / "snippet_8_9_exact"; (out / "testFunc").mkdir(parents=True, exist_ok=True); os.chdir(out)
    import snippet_8_9_as_printed as S
    t0 = time.time()
    S.testFunc()
    print(f"# done in {time.time() - t0:.0f} s; files in {out / 'testFunc'}")
    print(open("testFunc/stats.csv").read())
