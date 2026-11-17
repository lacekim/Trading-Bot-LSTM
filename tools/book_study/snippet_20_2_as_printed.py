"""AFML Snippet 20.2, "Vectorized Cartesian product" (PDF page 187), copied from the PDF.
RULING (user 2026-11-17, "ok"): `from itertoolsimport izip,product` (space lost in typesetting) -> `from itertools import product`
with izip -> zip (Python 3's built-in; as in 8.9); print i -> print(i).
Run on the Mac from the project root:  python tools/book_study/snippet_20_2_as_printed.py
"""
# Cartesian product of dictionary of lists
from itertools import product
dict0={'a':['1','2'],'b':['+','*'],'c':['!','@']}
jobs=(dict(zip(dict0,i)) for i in product(*dict0.values()))
for i in jobs:print(i)
