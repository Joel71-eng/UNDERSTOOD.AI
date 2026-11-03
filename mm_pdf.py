"""
mm_pdf.py - a clean printable PDF of what the learner has just made: the diagram (or molecule / equation),
the explanation in sections, a table of the parts, and where the facts came from.  Built with reportlab
(pure Python, works on Windows with `pip install reportlab`).
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUBS = "₀₁₂₃₄₅₆₇₈₉"
_REPL = {"→": "->", "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "•": "•",
         "…": "...", " ": " ", "⇌": "<=>", "−": "-"}


def pdf_text(s: str) -> str:
    """Markdown-ish text -> reportlab paragraph markup that survives the built-in Helvetica font."""
    s = (s or "")
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"([" + _SUBS + r"]+)", lambda m: "<sub>" + m.group(1).translate(_SUB) + "</sub>", s)
    for k, v in _REPL.items():
        s = s.replace(k, v)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"<(?!/?(?:b|i|sub|sup)>)", "&lt;", s)
    return s.encode("latin-1", "replace").decode("latin-1")


def _img(path: str, max_w: float, max_h: float) -> RLImage:
    with Image.open(path) as im:
        w, h = im.size
    s = min(max_w / w, max_h / h)
    return RLImage(path, width=w * s, height=h * s)


def build_pdf(out_path: Path, title: str, level_label: str, images: list[tuple[str, str]], explanation_md: str,
              parts: list[dict], facts: Optional[list[str]] = None, source_note: str = "", equation: str = "") -> str:
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=28, leading=33, spaceAfter=6, alignment=0)
    meta = ParagraphStyle("meta", parent=ss["Normal"], fontSize=11.5, leading=15, textColor=colors.HexColor("#666666"), spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=18, leading=22, spaceBefore=14, spaceAfter=5, textColor=colors.HexColor("#1f3a5f"))
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=13, leading=19, spaceAfter=6)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=18, bulletIndent=4)
    cap = ParagraphStyle("cap", parent=ss["Normal"], fontSize=11, leading=14, textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=10)
    eq = ParagraphStyle("eq", parent=body, fontSize=18, leading=26, alignment=1, spaceBefore=6, spaceAfter=10)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(2 * cm, 1.2 * cm, "Understood")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title=title, author="Understood")
    W = A4[0] - 4 * cm
    story = [Paragraph(pdf_text(title), h1),
             Paragraph(pdf_text(f"{level_label}  |  {_dt.date.today().strftime('%d %B %Y')}"), meta)]
    if equation:
        story.append(Paragraph(equation.replace("→", "->").encode("latin-1", "replace").decode("latin-1") if "<sub>" in equation else pdf_text(equation), eq))
    for path, caption in images:
        if path and Path(path).exists():
            story.append(_img(path, W, 11.5 * cm))
            if caption:
                story.append(Paragraph(pdf_text(caption), cap))
    for line in (facts or []):
        story.append(Paragraph(pdf_text(line), body))

    for raw in (explanation_md or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            story.append(Spacer(1, 3))
        elif line.lstrip().startswith("#"):
            story.append(Paragraph(pdf_text(line.lstrip("# ").strip()), h2))
        elif re.match(r"^\s*([-*•]|\d+[.)])\s+", line):
            story.append(Paragraph(pdf_text(re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", line)), bullet, bulletText="•"))
        else:
            story.append(Paragraph(pdf_text(line), body))

    rows = [p for p in parts if p.get("name")]
    if rows:
        story.append(Paragraph("Parts", h2))
        data = [["", Paragraph("<b>Part</b>", body), Paragraph("<b>What it is / does</b>", body)]]
        style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
                 ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        for i, p in enumerate(rows, 1):
            data.append(["", Paragraph(pdf_text(p["name"]), body), Paragraph(pdf_text(p.get("text") or ""), body)])
            c = p.get("colour") or (120, 120, 120)
            style.append(("BACKGROUND", (0, i), (0, i), colors.Color(c[0] / 255, c[1] / 255, c[2] / 255)))
        t = Table(data, colWidths=[0.45 * cm, 4.2 * cm, W - 4.65 * cm], repeatRows=1)
        t.setStyle(TableStyle(style))
        story.append(t)
    if source_note:
        story += [Spacer(1, 10), Paragraph(pdf_text(source_note), meta)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return str(out_path)


def build_doc_pdf(out_path: Path, blocks: list[dict], level_label: str = "", data_dir: Optional[Path] = None) -> str:
    """Render a document-model block list (see mm_doc.py) to a PDF."""
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=28, leading=33, spaceAfter=6, alignment=0)
    meta = ParagraphStyle("meta", parent=ss["Normal"], fontSize=11.5, leading=15, textColor=colors.HexColor("#666666"), spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=18, leading=22, spaceBefore=16, spaceAfter=5, textColor=colors.HexColor("#1f3a5f"))
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#3b6aa0"))
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=13, leading=19, spaceAfter=7)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=18, bulletIndent=4)
    cap = ParagraphStyle("cap", parent=ss["Normal"], fontSize=11, leading=14, textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=10)
    eq = ParagraphStyle("eq", parent=body, fontSize=18, leading=26, alignment=1, spaceBefore=6, spaceAfter=10)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(2 * cm, 1.2 * cm, "Understood")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    title = next((b["text"] for b in blocks if b["t"] == "h1"), "Understood notes")
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title=title, author="Understood")
    W = A4[0] - 4 * cm
    story: list = []
    answers: list = []
    qn = 0
    for b in blocks:
        t = b["t"]
        if t == "h1":
            story.append(Paragraph(pdf_text(b["text"]), h1))
        elif t == "h2":
            story.append(Paragraph(pdf_text(b["text"]), h2))
        elif t == "h3":
            story.append(Paragraph(pdf_text(b["text"]), h3))
        elif t == "note":
            story.append(Paragraph(pdf_text(b["text"]), meta))
        elif t == "p":
            story.append(Paragraph(pdf_text(b["text"]), body))
        elif t == "equation":
            story.append(Paragraph(b["text"].replace("→", "->").encode("latin-1", "replace").decode("latin-1") if "<sub>" in b["text"] else pdf_text(b["text"]), eq))
        elif t == "bullets":
            for it in b["items"]:
                story.append(Paragraph(pdf_text(it), bullet, bulletText="•"))
        elif t == "image":
            path = Path(b["path"])
            if not path.is_absolute() and data_dir:
                path = Path(data_dir) / path
            if path.exists():
                story.append(_img(str(path), W, 11.5 * cm))
                if b.get("caption"):
                    story.append(Paragraph(pdf_text(b["caption"]), cap))
        elif t == "terms":
            data = [[Paragraph("<b>Term</b>", body), Paragraph("<b>Meaning</b>", body)]]
            for it in b["items"]:
                data.append([Paragraph(pdf_text(it["term"]), body), Paragraph(pdf_text(it["meaning"]), body)])
            tb = Table(data, colWidths=[4.2 * cm, W - 4.2 * cm], repeatRows=1)
            tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
                                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7"))]))
            story.append(tb)
        elif t == "parts":
            data = [["", Paragraph("<b>Part</b>", body), Paragraph("<b>What it is / does</b>", body)]]
            style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd"))]
            for i, it in enumerate(b["items"], 1):
                data.append(["", Paragraph(pdf_text(it["name"]), body), Paragraph(pdf_text(it.get("text") or ""), body)])
                c = it.get("colour") or (120, 120, 120)
                style.append(("BACKGROUND", (0, i), (0, i), colors.Color(c[0] / 255, c[1] / 255, c[2] / 255)))
            tb = Table(data, colWidths=[0.45 * cm, 4.2 * cm, W - 4.65 * cm], repeatRows=1)
            tb.setStyle(TableStyle(style))
            story.append(tb)
        elif t == "qa":
            for it in b["items"]:
                qn += 1
                story.append(Paragraph(pdf_text(f"{qn}. {it['q']}"), body))
                story.append(Spacer(1, 10))
                answers.append(f"{qn}. {it['a']}")
    if answers:
        story.append(PageBreak())
        story.append(Paragraph("Answer key", h2))
        for a in answers:
            story.append(Paragraph(pdf_text(a), body))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return str(out_path)
