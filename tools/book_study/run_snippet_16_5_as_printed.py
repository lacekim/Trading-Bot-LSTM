"""Calls AFML Snippet 16.5's hrpMC() (snippet_16_5_as_printed.py) once, as printed (numIters=1e4, about an hour).
NOT in the book: running from runtime/snippet_16_5/ (so the book's stats.csv lands there), and copying the printed lines to
runtime/snippet_16_5/printed_output.txt.
Needs the gitignored _claude_scratch/cla_2013/CLA.py (the paper's code, ruled 2026-11-18).
Run on the Mac from the project root:  python tools/book_study/run_snippet_16_5_as_printed.py
"""
import builtins, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
out = HERE.parents[1] / "runtime" / "snippet_16_5"; out.mkdir(parents=True, exist_ok=True)
log = open(out / "printed_output.txt", "w", buffering=1); _print = builtins.print
def tee(*a, **k): _print(*a, **k, flush=True); _print(*a, **{**k, "file": log})
import numpy, pandas, scipy
tee(f"# python {sys.version.split()[0]} numpy {numpy.__version__} pandas {pandas.__version__} scipy {scipy.__version__}")
import snippet_16_5_as_printed as S
S.print = tee
os.chdir(out)
S.hrpMC()
