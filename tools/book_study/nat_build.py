"""Assembles the Natenberg write-up PDF: every figure, table and worked example of the book's PDF in the book's order, run on our Kraken (ETH) and Alpaca (AAPL) data, with the book's printed numbers checked.  New file.
Outputs: runtime/book_nat/<timestamp>/items.json (every item and every check), natenberg_study.pdf, coverage.json.
Usage: python -W ignore tools/book_study/nat_build.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, Spacer
from reportlab.platypus.tableofcontents import TableOfContents

import writeup_core as W
import writeup_build as B
from writeup_build import ST, Head, Doc, P, make_table, make_image, styles

import nat_a, nat_b, nat_c, nat_d, nat_e
import nat_ctx as X

B.BASIS_TEXT["not testable"] = "the PDF gives no inputs we can run, or the test needs data we do not have: said in one line, left unfilled"

CH = [("Chapter 1", 3, 4), ("Chapter 3", 5, 5), ("Chapter 4", 6, 16), ("Chapter 5", 17, 17), ("Chapter 6", 18, 21), ("Chapter 7", 22, 29), ("Chapter 9", 30, 55), ("Chapter 11", 56, 69), ("Chapter 12", 70, 74),
      ("Chapter 17", 75, 78), ("Chapter 18", 79, 98), ("Chapter 19", 99, 120), ("Chapter 20", 121, 122), ("Chapter 21", 123, 138), ("Chapter 23", 139, 140), ("Chapter 24", 141, 143), ("Glossary", 144, 157), ("Appendix B", 158, 162)]

# every figure the PDF contains
INVENTORY = (["Figure 1-%d" % i for i in range(1, 5)] + ["Figure 3-1"] + ["Figure 4-%d" % i for i in range(1, 16)] + ["Figure 5-1", "Figure 5-2"] + ["Figure 6-%d" % i for i in range(1, 6)] + ["Figure 7-%d" % i for i in range(1, 12)]
             + ["Figure 9-%d" % i for i in range(1, 27)] + ["Figure 11-%d" % i for i in range(1, 25)] + ["Figure 12-%d" % i for i in range(1, 10)] + ["Figure 17-%d" % i for i in range(1, 8)] + ["Figure 18-%d" % i for i in range(1, 12)]
             + ["Figure 19-%d" % i for i in range(1, 15)] + ["Figure 20-1"] + ["Figure 21-%d" % i for i in range(1, 21)] + ["Figure 23-%d" % i for i in range(1, 4)] + ["Figure 24-%d" % i for i in range(1, 4)])


def chapter_of(page):
    for nm, a, b in CH:
        if a <= page <= b:
            return nm
    return "Chapter ?"


def base_id(i):
    m = re.match(r"^(Figure \d+-\d+)", i)
    return m.group(1) if m else i


def collect():
    items, seq = [], 0
    for mod in (nat_a, nat_b, nat_c, nat_d, nat_e):
        for it in mod.all_items():
            it["_seq"] = seq
            seq += 1
            items.append(it)
    items.append(dict(X.item("Glossary", 144, "Glossary (definitions of terms used in the book)", "text", text="Definitions only: the glossary prints no formula, figure or worked example, so there is nothing to run.", checks=[]), _seq=seq))
    ids = [i["id"] for i in items]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate ids {dup}")
    items.sort(key=lambda it: (it["page"], it["_seq"]))
    return items


def summarise(items):
    rows, bad = [], []
    per = OrderedDict()
    for it in items:
        ch = chapter_of(it["page"])
        d = per.setdefault(ch, dict(items=0, checks=0, ok=0, diff=0, nt=0))
        d["items"] += 1
        d["nt"] += int(it["basis"] == "not testable")
        for c in it["checks"]:
            d["checks"] += 1
            if c["ok"]:
                d["ok"] += 1
            else:
                d["diff"] += 1
                bad.append((it["id"], it["page"], c))
    for ch, d in per.items():
        rows.append([ch, d["items"], d["nt"], d["checks"], d["ok"], d["diff"]])
    tot = [sum(r[k] for r in rows) for k in range(1, 6)]
    rows.append(["All"] + tot)
    return rows, bad


def fmtv(x):
    if isinstance(x, bool) or isinstance(x, str):
        return str(x)
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def item_flow(it, width):
    cid = it["id"]
    out = []
    out.append(Head(f"<b>{escape(cid)}</b> &nbsp;{escape(it['title'])}", ST["item"], 1, re.sub(r"[^A-Za-z0-9]+", "_", cid), toc_text=f"{cid}  {it['title']}"[:150]))
    n_ok = sum(1 for c in it["checks"] if c["ok"])
    n_bad = sum(1 for c in it["checks"] if not c["ok"])
    chk = f"  |  checks: {n_ok} pass, {n_bad} DIFFERS" if it["checks"] else ""
    out.append(Paragraph(escape(f"Book page {it['page']}  |  basis: {it['basis']} ({B.BASIS_TEXT[it['basis']]})" + chk), ST["tag"]))
    if it.get("text"):
        out.append(P(it["text"]))
    if it.get("fig"):
        for fl in out:
            if isinstance(fl, Paragraph):
                fl.keepWithNext = 1
        out.append(make_image(it["fig"], width))
        out.append(Spacer(1, 3))
    if it.get("table"):
        out.append(make_table(it["table"], width)); out.append(Spacer(1, 3))
    if it.get("table2"):
        out.append(P(it.get("table2_title") or "", "tag")); out.append(make_table(it["table2"], width)); out.append(Spacer(1, 3))
    if it.get("table3"):
        out.append(P(it.get("table3_title") or "", "tag")); out.append(make_table(it["table3"], width)); out.append(Spacer(1, 3))
    if it.get("extra_rows"):
        out.append(P("Every printed sensitivity against a finite difference, every setting and venue", "tag"))
        out.append(make_table(W.tbl(["Venue", "Setting", "Sensitivity", "Call / put", "Printed formula", "Finite difference", "Result"], it["extra_rows"]), width)); out.append(Spacer(1, 3))
    if it.get("note"):
        out.append(Paragraph("<b>Note.</b> " + escape(it["note"]), ST["note"]))
    return out


def build():
    styles()
    items = collect()
    ids_have = {base_id(i["id"]) for i in items}
    missing = [f for f in INVENTORY if f not in ids_have]
    extra = sorted(ids_have - set(INVENTORY) - {i for i in ids_have if not i.startswith("Figure")})
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = W.ROOT / "runtime" / "book_nat" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    rows, bad = summarise(items)
    json.dump({"built_utc": ts, "venues": {v: {k: (val if k != "close" and k != "ret" else None) for k, val in X.ctx(v).items() if k not in ("close", "ret")} for v in W.VENUES},
               "items": [{k: v for k, v in it.items()} for it in items], "summary": rows, "inventory": INVENTORY, "missing": missing}, open(outdir / "items.json", "w"), default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (bool(o) if isinstance(o, np.bool_) else str(o)), indent=1)
    cov = {"figures_in_pdf": len(INVENTORY), "missing": missing, "items": len(items), "summary": rows}
    json.dump(cov, open(outdir / "coverage.json", "w"), indent=1)
    print("coverage: missing", missing, "items", len(items))
    if missing:
        raise SystemExit("coverage check failed")
    fw = letter[0] - 1.3 * inch
    story = []
    story.append(Paragraph("Option Volatility and Pricing (Natenberg): every item run on Kraken and Alpaca data", ST["title"]))
    story.append(P(f"Both venues, always. Built {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. Source: the audiobook companion PDF of the book (162 pages). Every figure, table and worked example in it has its own entry below, in the book's order and under the book's identifier."))
    story.append(Paragraph("What is run and how", ST["h1"]))
    for t in [
        "The book's formulas are what is run (the book prints equations, not code): Black-Scholes with the b variants (Figure 18-6), every printed sensitivity, the binomial model of Figure 19-13 (American exercise, dividends), the Appendix B formulas, and the position arithmetic of Chapter 21. They are coded as printed in nat_core.py.",
        "Where the book prints its inputs (Figures 1-1, 1-2, 4-x, 6-x, 9-x with listed inputs, 18-1 to 18-4, 18-6 numbers, 19-x trees, Figure 21-9, Figure 21-20, Figure 23-3), the entry reproduces the book's printed numbers and says pass or DIFFERS for each.",
        "Where the book draws a picture with no numbers, the entry draws it on our data and tests the statements the book makes about it. The only change from the book in those entries is the data source: S is our last daily close, volatility is the standard deviation of our last 365 (Kraken) or 252 (Alpaca) daily log returns times the square root of 365 or 252.",
        "Assumptions the book does not fix (each repeated in the entries that use it): X = last close unless the picture prints strikes (then the same percentage of our last close as the book's strike is of its centre); 3 months to expiration; r = b = 0; calendars 2 and 4 months; theta per calendar day (/365), vega per volatility point (/100); Figure 21-9 time = calendar days / 365; tolerances for statements like 'about 95' or 'at the money' are ours (the book prints none) and are stated in the check.",
        "Items that need option quotes (implied volatility, skew, surface) or a position the PDF does not print are marked 'not testable' with the reason in one line and are left unfilled.",
    ]:
        story.append(P("- " + t))
    story.append(Paragraph("Our inputs", ST["h1"]))
    for v in W.VENUES:
        story.append(P("- " + X.ctx_line(v)))
    story.append(Paragraph("Results by chapter", ST["h1"]))
    story.append(make_table(W.tbl(["Chapter", "Entries", "Not testable", "Checks", "Pass", "DIFFERS"], rows), fw))
    story.append(P("A check is one printed number or one statement of the book against what the book's own formula gives. DIFFERS means the gap is larger than the tolerance given with the check; those are kept as found and listed below. Where a DIFFERS is of the size of the book's own rounding of printed prices, the entry's note or the check's numbers show it."))
    story.append(Paragraph("Every DIFFERS", ST["h1"]))
    brows = [[i, pg, c["what"][:230], fmtv(c["book"]), fmtv(c["ours"])] for i, pg, c in bad]
    story.append(make_table(W.tbl(["Entry", "Page", "Check", "Book prints / says", "Ours"], brows), fw))
    story.append(Paragraph("Not testable", ST["h1"]))
    nrows = [[i["id"], i["page"], i["text"][:260]] for i in items if i["basis"] == "not testable"]
    story.append(make_table(W.tbl(["Entry", "Page", "Reason"], nrows), fw))
    story.append(Paragraph("Coverage check", ST["h1"]))
    story.append(P(f"All {len(INVENTORY)} figures of the PDF (Figures 1-1 to 24-3) have at least one entry; none missing. Entries: {len(items)}. Appendix B has three entries and the glossary one."))
    story.append(PageBreak())
    toc = TableOfContents()
    toc.levelStyles = [ST["toc0"], ST["toc1"]]
    toc.dotsMinLevel = 0
    story.append(Paragraph("Contents of this write-up", ST["h1"]))
    story.append(toc)
    story.append(PageBreak())
    cur = None
    for it in items:
        ch = chapter_of(it["page"])
        if ch != cur:
            if cur is not None:
                story.append(PageBreak())
            story.append(Head(escape(ch), ST["chapter"], 0, re.sub(r"\W", "_", ch), toc_text=ch))
            cur = ch
        story += item_flow(it, fw)
    out = outdir / "natenberg_study.pdf"

    def footer(c, d):
        c.saveState(); c.setFont("DV", 6.5); c.setFillColor(colors.HexColor("#666666"))
        c.drawString(0.65 * inch, 0.45 * inch, "Study of Option Volatility and Pricing: Kraken and Alpaca"); c.drawRightString(letter[0] - 0.65 * inch, 0.45 * inch, f"page {d.page}"); c.restoreState()

    doc = Doc(str(out), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch, topMargin=0.6 * inch, bottomMargin=0.7 * inch, title="Study of Option Volatility and Pricing: Kraken and Alpaca", author="")
    doc.multiBuild(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out, out.stat().st_size, "dir", outdir)
    return outdir


if __name__ == "__main__":
    build()
