"""Calls AFML Snippet 4.6's main() (snippet_4_6_as_printed.py) once, as printed.
NOT in the book: the call itself (the book prints none), and copying the printed lines to runtime/snippet_4_6/printed_output.txt.
Run on the Mac from the project root:  python tools/book_study/run_snippet_4_6_as_printed.py
"""
import builtins, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
out = HERE.parents[1] / "runtime" / "snippet_4_6"; out.mkdir(parents=True, exist_ok=True)
log = open(out / "printed_output.txt", "w", buffering=1); _print = builtins.print
def tee(*a, **k): _print(*a, **k, flush=True); _print(*a, **{**k, "file": log})
import numpy, pandas
tee(f"# python {sys.version.split()[0]} numpy {numpy.__version__} pandas {pandas.__version__}")
import snippet_4_6_as_printed as S
S.print = tee
S.main()
