"""
mm_files.py - turn an uploaded file into something the tutor can use.

    image  -> saved (downscaled), the tutor looks at it
    pdf    -> text via pypdf; if the PDF is scanned (almost no text) the first pages are rendered as images instead
    docx   -> paragraphs + tables as text
    txt / md / csv -> text
"""
from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

from PIL import Image

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
TEXT_EXT = {".txt", ".md", ".csv", ".tsv", ".json", ".py", ".html", ".htm"}
MAX_TEXT = 60000
MAX_IMG = 1600


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:                                                   # noqa: BLE001
            continue
    return data.decode("utf-8", "replace")


def ingest(name: str, data: bytes, dest_dir: Path) -> dict:
    """Returns {name, kind: image|pdf|docx|text|unknown, path, text, pages:[image paths], note}"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(name).suffix.lower()
    uid = uuid.uuid4().hex[:10]
    out = {"name": name, "kind": "unknown", "path": None, "text": "", "pages": [], "note": ""}

    if ext in IMAGE_EXT:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:                                              # noqa: BLE001  (corrupt or renamed file: say so, do not crash the request)
            return {**out, "note": f"this picture could not be opened ({type(e).__name__})"}
        if max(img.size) > MAX_IMG:
            img.thumbnail((MAX_IMG, MAX_IMG), Image.LANCZOS)
        p = dest_dir / f"{uid}.png"
        img.save(p)
        return {**out, "kind": "image", "path": str(p)}

    if ext == ".pdf":
        p = dest_dir / f"{uid}.pdf"
        p.write_bytes(data)
        text, pages = "", []
        try:
            from pypdf import PdfReader
            rd = PdfReader(str(p))
            text = "\n\n".join((pg.extract_text() or "") for pg in rd.pages[:80])
        except Exception as e:                                              # noqa: BLE001
            out["note"] = f"could not read text ({type(e).__name__})"
        if len(re.sub(r"\s+", "", text)) < 200:                             # scanned PDF: show the first pages to the tutor
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(str(p))
                for i in range(min(3, len(pdf))):
                    im = pdf[i].render(scale=1.6).to_pil().convert("RGB")
                    q = dest_dir / f"{uid}_p{i + 1}.png"
                    im.save(q)
                    pages.append(str(q))
                out["note"] = "scanned PDF: showing the first pages"
            except Exception as e:                                          # noqa: BLE001
                out["note"] = (out["note"] + f" {type(e).__name__}").strip()
        return {**out, "kind": "pdf", "path": str(p), "text": text[:MAX_TEXT], "pages": pages}

    if ext == ".docx":
        p = dest_dir / f"{uid}.docx"
        p.write_bytes(data)
        text = ""
        try:
            import docx
            d = docx.Document(str(p))
            parts = [para.text for para in d.paragraphs if para.text.strip()]
            for t in d.tables:
                for row in t.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
            text = "\n\n".join(parts)
        except Exception as e:                                              # noqa: BLE001
            out["note"] = f"could not read ({type(e).__name__})"
        return {**out, "kind": "docx", "path": str(p), "text": text[:MAX_TEXT]}

    if ext in TEXT_EXT or not ext:
        text = _decode(data)
        if ext in (".csv", ".tsv"):
            text = "\n".join(text.splitlines()[:80])
        p = dest_dir / f"{uid}.txt"
        p.write_text(text[:MAX_TEXT], encoding="utf-8")
        return {**out, "kind": "text", "path": str(p), "text": text[:MAX_TEXT]}

    out["note"] = f"{ext or 'this'} files are not supported yet"
    return out
