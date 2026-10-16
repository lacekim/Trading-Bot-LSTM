"""Independent check of book_study_writeup.pdf: saved-run numbers must appear in the PDF text; every Contents id must appear as a heading."""
import json, re, subprocess, sys
from pathlib import Path
R = Path("runtime"); OUT = R / "book_writeup"
txt = subprocess.run(["pdftotext", "-layout", str(OUT / "book_study_writeup.pdf"), "-"], capture_output=True, text=True).stdout
flat = re.sub(r"\s+", " ", txt)
bad, ok = [], 0
def need(label, s):
    global ok
    if s in flat: ok += 1
    else: bad.append((label, s))
for v in ("kraken", "alpaca"):
    m = json.load(open(next(R.glob(f"book_bot3/{v}_*/meta.json"))))
    d = m["dsr"]["last_365_days"]
    for label, s in [("wf auc", f"{m['walk_forward']['auc']:.4f}"), ("cpcv auc", f"{m['cpcv']['auc_mean']:.4f}"), ("gain", f"{m['ch14']['gain_pct']:+.2f}"),
                     ("sharpe", f"{d['sr_daily']*365**.5:+.2f}"), ("psr", f"{d['psr_vs_0']:.4f}"), ("dsr", f"{d['dsr']:.4f}"), ("maxdd", f"{m['ch14']['drawdowns_14_4']['max_pct']:.2f}"),
                     ("events", f"{m['events']:,}"), ("symbols", f"{m['symbols']}")]:
        need(f"{v} {label}", s)
con = json.load(open(OUT / "contents.json"))
ids = [c[0] if isinstance(c, list) else c for c in con]
for i in ids:
    if i.lower() not in flat.lower(): bad.append(("heading", i))
chk = json.load(open(OUT / "checks_ch14_15.json"))
mm = [c for c in chk if abs(float(c[1]) - float(c[2])) > 1e-6 * max(1.0, abs(float(c[1])))]
print("numbers/headings ok:", ok, "contents ids:", len(ids), "mismatches:", bad, "ch14_15 failed checks:", len(mm))
sys.exit(1 if bad or mm else 0)
