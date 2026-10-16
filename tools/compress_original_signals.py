"""Read-only helper: writes a gzip-compressed copy of the original CNN/LSTM
signal file so it fits under the 400MB transfer limit for the LightGBM
trade-filter test. Does not modify the original file or touch the bot.

Run from the repo root:
    python3 tools/compress_original_signals.py
"""
import gzip
import os
import shutil
import time

SRC = "logs/v5_original_long_paper_signals.csv"
DST = "logs/v5_original_long_paper_signals.csv.gz"
PREVIEW = "logs/v5_original_long_paper_signals_preview.csv"

t0 = time.time()
with open(SRC, "rb") as f, open(PREVIEW, "wb") as p:
    for _ in range(6):
        p.write(f.readline())

with open(SRC, "rb") as f_in, gzip.open(DST, "wb", compresslevel=6) as f_out:
    shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)

mb = os.path.getsize(DST) / 1e6
print(f"done in {time.time() - t0:.0f}s -> {DST} ({mb:.0f} MB)")
print("OK to transfer" if mb < 400 else "still over 400MB -- tell Claude")
