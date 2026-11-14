# Successful Algorithmic Trading ch.14 (moving average crossover): every change made to the book's code

`book_code_printed/` is the book's code as printed; `book_code/` is the copy that runs. Each change below is one commit,
with its diff in `changes/`. The user said on 2026-11-14: "do what you need to, write it all down and commit EACH thing you do".
The book's own source files are not committed (public repo, commercial book); the diffs and this log are.

### SAT ch.14 change 1: data.py line 130 .sort() -> .sort_index()
Why: pandas 2.3.3 has no DataFrame.sort (the book targets pandas 0.16.2); sort_index is the same operation (sort rows by the date index) under its current name. Diff: changes/01_data_py_sort_index.patch. Run after this change (run_step1.txt): next stop at mac.py line 61, 'bars != []' raises ValueError (truth value of an empty array is ambiguous) on numpy 2.2.6.
