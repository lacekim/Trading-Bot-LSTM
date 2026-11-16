"""Runs AFML Snippet 13.1's main() (snippet_13_as_printed.py) and keeps everything it prints.
NOT in the book: writing the printed lines to runtime/snippet_13_1/printed_output.txt as well as the screen, a header line
per parameter pair (the book's main() prints nothing between pairs), and saving the returned last batch. Single process, as printed.
Run on the Mac from the project root:  python tools/book_study/run_snippet_13_1_as_printed.py
"""
import builtins, json, sys, time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import snippet_13_as_printed as S
out = HERE.parents[1] / "runtime" / "snippet_13_1"; out.mkdir(parents=True, exist_ok=True)
log = open(out / "printed_output.txt", "w", buffering=1)
_print = builtins.print
def tee(*a, **k):
    _print(*a, **k, flush=True); _print(*a, **{**k, "file": log})
S.print = tee                                     # the snippet's own print() lines go to the screen and the file
pairs = list(product([10,5,0,-5,-10],[5,10,25,50,100])); i = {"n": 0}
_batch = S.batch
def batch_with_header(coeffs, **kw):
    i["n"] += 1; tee(f"# pair {i['n']} of 25: forecast {coeffs['forecast']} half-life {coeffs['hl']} sigma {coeffs['sigma']}")
    return _batch(coeffs, **kw)
S.batch = batch_with_header
t0 = time.time()
last = S.main()
(out / "run.json").write_text(json.dumps({"call": "main()", "seconds": round(time.time() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat(),
    "returned_last_batch_rows": len(last)}, indent=1))
tee(f"# done in {time.time() - t0:.0f} s")
