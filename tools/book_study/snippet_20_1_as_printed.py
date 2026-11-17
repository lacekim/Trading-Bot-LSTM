"""AFML Snippet 20.1, "Un-vectorized Cartesian product" (PDF page 187), copied from the PDF.
RULING (user 2026-11-17, "ok"): print -> print() is the only change.
Run on the Mac from the project root:  python tools/book_study/snippet_20_1_as_printed.py
"""
# Cartesian product of dictionary of lists
dict0={'a':['1','2'],'b':['+','*'],'c':['!','@']}
for a in dict0['a']:
    for b in dict0['b']:
        for c in dict0['c']:
            print({'a':a,'b':b,'c':c})
