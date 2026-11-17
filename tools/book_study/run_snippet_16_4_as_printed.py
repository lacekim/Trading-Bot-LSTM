"""Runs AFML Snippet 16.4's main() (snippet_16_4_as_printed.py) as printed.
NOT in the book: running inside runtime/snippet_16_4/ (so the two PNGs the snippet saves land there), printing the library
versions first, and copying every printed line and warning to runtime/snippet_16_4/printed_output.txt.
Run on the Mac from the project root:  python tools/book_study/run_snippet_16_4_as_printed.py
"""
import builtins, logging, os, sys, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
out = HERE.parents[1] / "runtime" / "snippet_16_4"; out.mkdir(parents=True, exist_ok=True); os.chdir(out)
log = open(out / "printed_output.txt", "w", buffering=1)
_print = builtins.print
def tee(*a, **k):
    _print(*a, **k, flush=True); _print(*a, **{**k, "file": log})
def showwarning(message, category, filename, lineno, file=None, line=None):
    tee(f"WARNING {category.__name__}: {message} ({Path(filename).name}:{lineno})")
warnings.showwarning = showwarning
import numpy, pandas, scipy, matplotlib
tee(f"# python {sys.version.split()[0]} numpy {numpy.__version__} pandas {pandas.__version__} scipy {scipy.__version__} matplotlib {matplotlib.__version__}")
import snippet_16_4_as_printed as S
S.print = tee
S.main()
tee("# done; files: " + ", ".join(sorted(p.name for p in out.iterdir())))
