"""
mm_slides.py - slide decks as PDF and as an EDITABLE PowerPoint.  Independent of the narrated video: the video is built from the same
scenes (mm_media.scenes_for_figure), so the slides and the video always agree.

deck = {"title", "subtitle", "slides": [{"title", "bullets": [..], "image": path|None, "notes": str, "colour": (r,g,b)}]}
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

import mm_core as C

PW, PH = 1600, 900


def short_bullets(text: str, max_bullets: int = 3, max_words: int = 26) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    return [C.words_trim(s, max_words) for s in parts[:max_bullets]]


def deck_from_scenes(fig: dict, data_dir: Path, work: Path, scenes: list[dict], depth: str = "standard") -> dict:
    """One slide per video scene: its finished picture (post) and its narration sentence(s)."""
    work.mkdir(parents=True, exist_ok=True)
    slides = []
    for k, sc in enumerate(scenes):
        name = sc["name"]
        title = fig["title"] if name == "intro" else ("Recap" if name == "outro" else name)
        img_path = work / f"slide_{k}.png"
        sc["post"].convert("RGB").save(img_path)
        nb = 1 if name in ("intro", "outro") else {"brief": 2, "standard": 3, "deep": 4}.get(depth, 3)
        slides.append({"title": title, "bullets": short_bullets(sc.get("short") or sc["text"], nb, 24) if name not in ("intro", "outro") else short_bullets(sc["text"], 2, 24),
                       "image": str(img_path), "notes": sc["text"], "colour": tuple(sc.get("colour") or (60, 60, 60))})
    return {"title": fig["title"], "subtitle": fig.get("level_label") or C.get_level(fig.get("level"))["label"], "slides": slides}


def deck_from_text(title: str, subtitle: str, text_slides: list[dict], pictures: list[tuple[str, str]]) -> dict:
    """Key-point slides (title + bullets) followed by one slide per picture of the discussion."""
    slides = []
    for k, sl in enumerate(text_slides):
        slides.append({"title": sl["title"], "bullets": sl.get("bullets", []), "image": None, "notes": sl.get("narration", ""),
                       "colour": C.colour_for(k)})
    for k, (path, cap) in enumerate(pictures):
        slides.append({"title": cap or "Figure", "bullets": [], "image": str(path), "notes": cap, "colour": C.colour_for(k + 3)})
    return {"title": title, "subtitle": subtitle, "slides": slides}


# --------------------------------------------------------------------------------------------------------------- PDF
def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    s = min(w / img.width, h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)


def _wrap(d, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for wd in text.split():
        t = (cur + " " + wd).strip()
        if cur and d.textlength(t, font=font) > max_w:
            lines.append(cur)
            cur = wd
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


def slide_page(sl: dict, n: int, total: int) -> Image.Image:
    col = tuple(sl.get("colour") or (60, 60, 60))
    page = Image.new("RGB", (PW, PH), (252, 252, 254))
    d = ImageDraw.Draw(page)
    d.rectangle([0, 0, 26, PH], fill=col)
    big, small = C.get_font(54, True), C.get_font(22)
    body = C.get_font(36)
    d.text((70, 50), C.words_trim(sl["title"], 12), font=big, fill=(30, 40, 60))
    top = 150
    has_img = bool(sl.get("image") and Path(sl["image"]).exists())
    bullets = sl.get("bullets") or []
    if has_img:
        img = Image.open(sl["image"]).convert("RGB")
        box_w = int(PW * (0.58 if bullets else 0.9))
        fit = _fit(img, box_w, PH - top - 70)
        x = 70 if bullets else (PW - fit.width) // 2
        page.paste(fit, (x, top))
    if bullets:
        x0 = int(PW * 0.64) if has_img else 90
        for size in (36, 32, 28, 25, 22):                         # largest type that still fits the page
            body = C.get_font(size)
            need = sum(len(_wrap(d, b, body, PW - x0 - 70)) * int(size * 1.4) + 26 for b in bullets)
            if need <= PH - top - 80:
                break
        lh = int(size * 1.4)
        y = top + 10
        for b in bullets:
            lines = _wrap(d, b, body, PW - x0 - 70)
            d.ellipse([x0 - 34, y + 14, x0 - 18, y + 30], fill=col)
            for ln in lines:
                d.text((x0, y), ln, font=body, fill=(45, 45, 52))
                y += lh
            y += 26
    d.text((PW - 40, PH - 36), f"{n} / {total}", font=small, fill=(150, 150, 158), anchor="rm")
    return page


def build_pdf(deck: dict, out: Path) -> Path:
    pages = [slide_page(s, i + 1, len(deck["slides"])) for i, s in enumerate(deck["slides"])]
    if not pages:
        raise ValueError("no slides")
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:], resolution=150.0)
    return out


# ----------------------------------------------------------------------------------------------------------- PowerPoint
def build_pptx(deck: dict, out: Path) -> Path:
    """Editable: real title and bullet text boxes, the picture as a picture, the narration as speaker notes."""
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    for sl in deck["slides"]:
        s = prs.slides.add_slide(blank)
        col = RGBColor(*[int(v) for v in (sl.get("colour") or (60, 60, 60))])
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.22), prs.slide_height)
        bar.fill.solid()
        bar.fill.fore_color.rgb = col
        bar.line.fill.background()
        tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.2), Inches(1.0))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = sl["title"]
        tf.paragraphs[0].runs[0].font.size = Pt(34)
        tf.paragraphs[0].runs[0].font.bold = True
        tf.paragraphs[0].runs[0].font.color.rgb = RGBColor(30, 40, 60)
        has_img = bool(sl.get("image") and Path(sl["image"]).exists())
        bullets = sl.get("bullets") or []
        if has_img:
            im = Image.open(sl["image"])
            box_w, box_h = (7.6 if bullets else 11.6), 5.4
            sc = min(box_w / im.width, box_h / im.height)
            w_in, h_in = im.width * sc, im.height * sc
            left = 0.6 if bullets else (13.333 - w_in) / 2
            s.shapes.add_picture(str(sl["image"]), Inches(left), Inches(1.55), Inches(w_in), Inches(h_in))
        if bullets:
            x0, wd = (8.5, 4.4) if has_img else (0.9, 11.6)
            bx = s.shapes.add_textbox(Inches(x0), Inches(1.7), Inches(wd), Inches(5.0))
            bf = bx.text_frame
            bf.word_wrap = True
            for k, b in enumerate(bullets):
                para = bf.paragraphs[0] if k == 0 else bf.add_paragraph()
                para.text = "• " + b
                para.font.size = Pt(22 if sum(len(x) for x in bullets) < 330 else (18 if sum(len(x) for x in bullets) < 520 else 15))
                para.space_after = Pt(14)
        if sl.get("notes"):
            s.notes_slide.notes_text_frame.text = sl["notes"]
    prs.save(str(out))
    return out
