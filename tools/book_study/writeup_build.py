"""Assembles the book-format write-up PDF: every entry of the book's Contents, in the book's order, with the book's identifier and our Kraken and Alpaca results.  New file.
Inputs: runtime/book_writeup/items_*.json (written by writeup_ch*.py), the Contents list (runtime/book_writeup/contents.json, parsed from the book's text), saved run outputs.
Outputs: runtime/book_writeup/book_study_writeup.pdf and coverage.json.
Usage: python -W ignore tools/book_study/writeup_build.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents

import writeup_core as W

OUT = W.OUT
FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", FD + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

CHAPTERS = {1: "Financial Machine Learning as a Distinct Subject", 2: "Financial Data Structures", 3: "Labeling", 4: "Sample Weights", 5: "Fractionally Differentiated Features", 6: "Ensemble Methods",
            7: "Cross-Validation in Finance", 8: "Feature Importance", 9: "Hyper-Parameter Tuning with Cross-Validation", 10: "Bet Sizing", 11: "The Dangers of Backtesting", 12: "Backtesting through Cross-Validation",
            13: "Backtesting on Synthetic Data", 14: "Backtest Statistics", 15: "Understanding Strategy Risk", 16: "Machine Learning Asset Allocation", 17: "Structural Breaks", 18: "Entropy Features",
            19: "Microstructural Features", 20: "Multiprocessing and Vectorization", 21: "Brute Force and Quantum Computers", 22: "High-Performance Computational Intelligence and Forecasting Technologies"}
CH_PAGES = {1: (11, 12), 2: (13, 24), 3: (25, 34), 4: (35, 45), 5: (46, 55), 6: (56, 61), 7: (62, 67), 8: (68, 78), 9: (79, 84), 10: (85, 92), 11: (93, 94), 12: (95, 98), 13: (99, 119), 14: (120, 129),
            15: (130, 135), 16: (136, 155), 17: (156, 166), 18: (167, 177), 19: (178, 186), 20: (187, 199), 21: (200, 206), 22: (207, 218)}
BASIS_TEXT = {"data": "computed on our Kraken and Alpaca data or run outputs", "formula": "closed form or simulation with the book's own inputs", "code": "a code listing: where it lives in our code and how it was checked",
              "diagram": "a drawing or another group's result: shown or described, not computed on our data", "text": "text of the book, with what it means for our bot"}


def chapter_of(page: int) -> int:
    for c, (a, b) in CH_PAGES.items():
        if a <= page <= b:
            return c
    return 22


# --------------------------------------------------------------------------------------------------------------------------------------------------
def load_items():
    items = OrderedDict()
    for f in sorted(glob.glob(str(OUT / "items_ch*.json"))):
        for it in json.load(open(f)):
            if it["id"] in items:
                raise SystemExit(f"duplicate item id {it['id']} (in {f})")
            items[it["id"]] = it
    for f in sorted(glob.glob(str(OUT / "items_patch_*.json"))):
        for it in json.load(open(f)):
            if it["id"] not in items:
                raise SystemExit(f"patch for unknown item {it['id']}")
            items[it["id"]] = it
    return items


def parse_contents():
    p = OUT / "contents.json"
    t = Path("/tmp/book.txt")
    if t.exists():
        lines = t.read_text().split("\n")
        end = [i for i, l in enumerate(lines) if "Figure 22.10" in l][0]
        con = []
        for l in lines[4:end + 1]:
            m = re.match(r"^\s*((?:Table|Figure|Equation|Expression|Snippet|Snipper|Section|Series|Log|Function|Matrix|Definition|Chapter)\s+[0-9A-Za-z.\-ab ]+?)\s+(\d{1,3})\s*$", l)
            if m:
                con.append((m.group(1).strip(), int(m.group(2))))
        json.dump(con, open(p, "w"))
    return [tuple(x) for x in json.load(open(p))]


def parent_of(extra_id: str):
    m = re.match(r"^(.*?) \(", extra_id)
    if m:
        return m.group(1)
    return None


# --------------------------------------------------------------------------------------------------------------------------------------------------
ST = {}


def styles():
    base = dict(fontName="DV", fontSize=8, leading=10.2, textColor=colors.black)
    ST["body"] = ParagraphStyle("body", **base, spaceAfter=3)
    ST["small"] = ParagraphStyle("small", **{**base, "fontSize": 7, "leading": 8.8})
    ST["note"] = ParagraphStyle("note", **{**base, "fontName": "DV-I", "fontSize": 7.2, "leading": 9.2, "textColor": colors.HexColor("#333333")}, spaceAfter=3, leftIndent=8)
    ST["tag"] = ParagraphStyle("tag", **{**base, "fontSize": 6.6, "leading": 8, "textColor": colors.HexColor("#555555")}, spaceAfter=2)
    ST["item"] = ParagraphStyle("item", **{**base, "fontName": "DV-B", "fontSize": 9.2, "leading": 11.4}, spaceBefore=9, spaceAfter=1)
    ST["chapter"] = ParagraphStyle("chapter", **{**base, "fontName": "DV-B", "fontSize": 15, "leading": 18}, spaceBefore=0, spaceAfter=8)
    ST["h1"] = ParagraphStyle("h1", **{**base, "fontName": "DV-B", "fontSize": 12, "leading": 15}, spaceBefore=10, spaceAfter=4, keepWithNext=1)
    ST["title"] = ParagraphStyle("title", **{**base, "fontName": "DV-B", "fontSize": 22, "leading": 27}, spaceAfter=6)
    ST["cell"] = ParagraphStyle("cell", **{**base, "fontSize": 6.6, "leading": 8})
    ST["cellh"] = ParagraphStyle("cellh", **{**base, "fontName": "DV-B", "fontSize": 6.6, "leading": 8})
    ST["toc0"] = ParagraphStyle("toc0", **{**base, "fontName": "DV-B", "fontSize": 8.5, "leading": 12}, spaceBefore=4)
    ST["toc1"] = ParagraphStyle("toc1", **{**base, "fontSize": 6.8, "leading": 8.2}, leftIndent=12)


class Head(Paragraph):
    def __init__(self, text, style, level, key, toc_text=None):
        super().__init__(text, style)
        self.level, self.key, self.toc_text = level, key, toc_text or re.sub("<[^>]+>", "", text)


class Doc(SimpleDocTemplate):
    def afterFlowable(self, fl):
        if isinstance(fl, Head):
            self.canv.bookmarkPage(fl.key)
            self.canv.addOutlineEntry(fl.toc_text, fl.key, fl.level, closed=fl.level > 0)
            self.notify("TOCEntry", (fl.level, fl.toc_text, self.page, fl.key))


def P(text, st="body"):
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), ST[st])


def make_table(t, width):
    cols = t["cols"]
    rows = t["rows"]
    n = len(cols)
    lens = []
    for j in range(n):
        vals = [len(str(r[j])) if j < len(r) else 0 for r in rows] + [len(str(cols[j]))]
        lens.append(min(60, max(6, (sum(vals) / len(vals)) * 0.7 + max(vals) * 0.3)))
    tot = sum(lens)
    cw = [width * x / tot for x in lens]
    data = [[Paragraph(escape(str(c)), ST["cellh"]) for c in cols]]
    for r in rows:
        data.append([Paragraph(escape(str(v)), ST["cell"]) for v in (list(r) + [""] * (n - len(r)))[:n]])
    tb = Table(data, colWidths=cw, repeatRows=1)
    tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return tb


def make_image(path, width, max_h=6.6 * inch):
    im = PILImage.open(path)
    w, h = im.size
    scale = min(width / w, max_h / h, 1.0 if False else 9e9)
    sc = min(width / w, max_h / h)
    return Image(path, width=w * sc, height=h * sc)


def item_flowables(cid, page, it, width, extra=False):
    out = []
    key = re.sub(r"[^A-Za-z0-9]+", "_", cid)
    title = f"{escape(cid)} &nbsp;&nbsp;{escape(it['title'])}"
    out.append(Head(f"<b>{escape(cid.upper() if not extra else cid)}</b> &nbsp;{escape(it['title'])}", ST["item"], 1, key, toc_text=f"{cid}  {it['title']}"[:150]))
    tag = f"Book page {page if page else '-'}  |  basis: {it['basis']} ({BASIS_TEXT[it['basis']]})" + ("  |  our addition (not a Contents entry)" if extra else "")
    out.append(Paragraph(escape(tag), ST["tag"]))
    if it.get("text"):
        out.append(P(it["text"]))
    if it.get("fig"):
        for fl in out:
            if isinstance(fl, Paragraph):
                fl.keepWithNext = 1
        out.append(make_image(it["fig"], width))
        out.append(Spacer(1, 3))
    if it.get("table"):
        out.append(make_table(it["table"], width))
        out.append(Spacer(1, 3))
    if it.get("table2"):
        out.append(P(it.get("table2_title") or "", "tag"))
        out.append(make_table(it["table2"], width))
        out.append(Spacer(1, 3))
    if it.get("note"):
        out.append(Paragraph("<b>Note.</b> " + escape(it["note"]), ST["note"]))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def headline():
    rows = []
    for v in W.VENUES:
        m = W.run_json(v, "meta.json")
        d = m["dsr"]["last_365_days"]
        rows.append([v, f"{m['symbols']}", f"{m['events']:,}", f"{m['first_day']} to {m['last_day']}", f"{m['walk_forward']['auc']:.4f}", f"{m['cpcv']['auc_mean']:.4f}", f"{m['ch14']['gain_pct']:+.2f}",
                     f"{d['sr_daily'] * 365 ** .5:+.2f}", f"{d['psr_vs_0']:.4f}", f"{d['dsr']:.4f}", f"{m['ch14']['drawdowns_14_4']['max_pct']:.2f}"])
    heads = ["Venue", "Symbols", "Events", "Daily data span", "Walk-forward AUC", "CPCV AUC (mean of 5 paths)", "Last-365-day gain %", "Annualised Sharpe", "PSR vs 0", "DSR (11 trials)", "Max drawdown %"]
    t = [[heads[j]] + [r[j] for r in rows] for j in range(1, len(heads))]
    return W.tbl(["Measure"] + [r[0] for r in rows], t)


def build():
    styles()
    items = load_items()
    con = parse_contents()
    ids_con = [c for c, _ in con]
    pages = dict(con)
    missing = [c for c in ids_con if c not in items]
    extras = [i for i in items if i not in pages]
    attach = {}
    for e in extras:
        par = parent_of(e)
        if e.startswith("Figure 13.26") or e.startswith("Figure 13.27"):
            par = "Figure 13.25"
        attach.setdefault(par, []).append(e)
    orphan = [e for e in extras if not any(e in v for v in attach.values())]
    cnt = Counter(items[c]["basis"] for c in ids_con if c in items)
    cov = {"contents_entries": len(ids_con), "with_item": len(ids_con) - len(missing), "missing": missing, "extra_items": extras, "orphan_extras": orphan, "by_basis": dict(cnt),
           "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    json.dump(cov, open(OUT / "coverage.json", "w"), indent=1)
    print("coverage:", {k: v for k, v in cov.items() if k != "extra_items"})
    if missing or orphan:
        raise SystemExit("coverage check failed")

    fw = letter[0] - 1.3 * inch
    story = []
    story.append(Paragraph("Advances in Financial Machine Learning: the study written up item by item", ST["title"]))
    story.append(P(f"Kraken (crypto) and Alpaca (equities), both always. Built {datetime.now(timezone.utc).strftime('%Y-%m-%d')} from the saved runs in runtime/book_bot3 (one bot per venue) and the dollar bars in runtime/dollar_bars.", "body"))
    story.append(Spacer(1, 6))
    story.append(Paragraph("How to read this", ST["h1"]))
    story.append(P(f"The book's Contents lists {len(ids_con)} items (tables, figures, equations, expressions, snippets, sections, series, a log, functions, a matrix and a definition). Each has its own entry below, in the book's order and under the book's identifier, "
                   "with a short title in our own words and the book's page. Under it: what we computed on our Kraken and Alpaca data, or, where the item is a formula, a code listing or a drawing, what it is and where it lives in our code. "
                   "Every entry is tagged with its basis: data (computed on our data or run outputs), formula (closed form or simulation with the book's own inputs), code (a listing: where it lives and how it was checked), "
                   "diagram (a drawing or another group's result, not computed on our data) or text. Where our result differs from what the book prints, or the book's data is not in the project, the entry says so in a note."))
    story.append(Paragraph("Where the bot stands (from the saved run files)", ST["h1"]))
    story.append(make_table(headline(), fw))
    story.append(Spacer(1, 4))
    story.append(P("These are out-of-sample results of the book's strategy as built: walk-forward trained on events labelled before 2024-01-01, then tested; the last-365-day column is the account path of those bets, after costs (Kraken 12 bp, Alpaca 10 bp per side). "
                   "The AUCs are near 0.5 and the deflated Sharpe ratios are near 0: no edge is shown. The reason on record is below (Table 1.2, rows 1 and 6)."))
    story.append(Paragraph("What is not covered or not as in the book", ST["h1"]))
    gaps = [
        "Table 1.2 rows 1 and 6 (the meta-strategy paradigm, meta-labeling): not met as the book states them. Our labels are side-free (getEvents with side=None, Snippets 3.3 and 3.5), so the forest learns side and size together and Snippet 3.7's meta-labeling step is not exercised.",
        "Section 2.3 (Equations 1-15, imbalance and run bars): the book prints the equations and no code for them, so there is nothing to copy; they are listed with the formula only and not computed. The bot trades the dollar bars of Section 2.2.",
        "Equations 16-18 and Expression 1 (the futures-basket accounting): not applicable to spot symbols; the three rules at the end of Section 2.4.1 (Equations 19-21) are applied.",
        "Equations 79-80 and Sections 19.5.1-19.5.2 (PIN, VPIN): not computed; they need a buy / sell split of volume that our sources do not have.",
        "Chapter 21 and 22 figures that report another group's experiment (Figures 22.1-22.7 and 22.9): described only. Figure 22.8 and 22.10 are shown with an analogue on our own data.",
        "Figures 17.2 and 17.3: the extract prints the pictures and no definition of CADF or C-dot, so ours are stand-ins (stated in each entry).",
        "Simulation figures (4.2, 8.2-8.4, 11.1-11.2, 13.1-13.25, 15.1 curves, 16.7, 18.1-18.2) are samples with fewer iterations or fixed seeds; the book's runs are unseeded, so the pictures agree in shape, not pixel for pixel. Where the book prints numbers (Snippet 10.4, Figure 12.1, Figure 16.5, Table 16.1) ours are identical (see their entries).",
        "Kraken's BTC bars end on 2026-01-05 (ICX on 2026-01-04) while other symbols run to 2026-10-12, so BTC drops out of any last-year window.",
        "Assumptions stated in the entries (each flagged 'assumption'): Figure 15.2 and 15.3 use pi+ = 0.01; Equation 61's tau0 = 0.1; Equation 66's phi = 0.5; Equation 77's loading f = V'w; Equation 97's cost c = per-side cost; Figure 9.2's loss form; Figure 16.1's constructed matrices.",
    ]
    for g in gaps:
        story.append(P("- " + g, "body"))
    story.append(Paragraph("Coverage check", ST["h1"]))
    story.append(P(f"{cov['with_item']} of {cov['contents_entries']} Contents entries have an entry here; none missing. Basis of the entries: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + "."))
    story.append(PageBreak())
    toc = TableOfContents()
    toc.levelStyles = [ST["toc0"], ST["toc1"]]
    toc.dotsMinLevel = 0
    story.append(Paragraph("Contents of this write-up", ST["h1"]))
    story.append(toc)
    story.append(PageBreak())

    cur = None
    for cid, page in con:
        ch = chapter_of(page)
        if ch != cur:
            if cur is not None:
                story.append(PageBreak())
            story.append(Head(f"Chapter {ch}. {escape(CHAPTERS[ch])}", ST["chapter"], 0, f"ch{ch}", toc_text=f"Chapter {ch}. {CHAPTERS[ch]}"))
            cur = ch
        it = items[cid]
        story += item_flowables(cid, page, it, fw)
        # entries that are not Contents entries are not rendered (the book's Contents only)

    out = OUT / "book_study_writeup.pdf"

    def footer(c, d):
        c.saveState()
        c.setFont("DV", 6.5)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(0.65 * inch, 0.45 * inch, "Study of Advances in Financial Machine Learning: Kraken and Alpaca")
        c.drawRightString(letter[0] - 0.65 * inch, 0.45 * inch, f"page {d.page}")
        c.restoreState()

    doc = Doc(str(out), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch, topMargin=0.6 * inch, bottomMargin=0.7 * inch, title="Study of Advances in Financial Machine Learning: Kraken and Alpaca", author="")
    doc.multiBuild(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out, out.stat().st_size)


if __name__ == "__main__":
    build()
