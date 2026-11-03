"""
bank_raster.py - the raster route of the bank builder: a labelled PNG/JPG -> a clean picture + the position every label points at.

    1. OCR (English only) finds the label words and where they sit.
    2. Straight line segments that START next to a label are its leader line; the far end is the ANCHOR (the part it points at).
    3. The words and those leader lines are painted out (inpainting), leaving the bare drawing.
Everything here is a best effort and is ALWAYS flagged for human review: OCR can misread, and a drawing with no leader lines
(labels written inside the parts) gives names without anchors.

Needs on Colab:  !apt -q install tesseract-ocr   and   !pip -q install pytesseract opencv-python-headless
"""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np
from PIL import Image

WORD = re.compile(r"[A-Za-z][A-Za-z'’\-/]+")
STATS: dict = {}


def available() -> tuple[bool, str]:
    """(ok, reason).  Checked in three separate steps so the reason points at exactly what to fix:
    the opencv-python package, the pytesseract package, or the tesseract-ocr system binary itself."""
    try:
        import cv2  # noqa: F401
    except Exception as e:                                                        # noqa: BLE001
        return False, f"opencv import failed: {type(e).__name__}: {e}  (pip install opencv-python-headless)"
    try:
        import pytesseract
    except Exception as e:                                                        # noqa: BLE001
        return False, f"pytesseract import failed: {type(e).__name__}: {e}  (pip install pytesseract)"
    try:
        v = pytesseract.get_tesseract_version()
    except Exception as e:                                                        # noqa: BLE001
        return False, (f"the tesseract-ocr program itself was not found: {type(e).__name__}: {e}  "
                       "(apt install tesseract-ocr; on Colab this is lost if the runtime restarted after you installed it, or it ran in a "
                       "different session than this one)")
    return True, f"tesseract {v}"


def _prepare(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    long_side = max(img.size)
    if long_side > 1800:
        s = 1800 / long_side
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    elif long_side < 1100:
        s = 1100 / long_side
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return img


def ocr_labels(img: Image.Image, min_conf: int = 55) -> list[dict]:
    """[{name, box (x0,y0,x1,y1)}]: English words with good confidence, neighbours on one line joined into one label."""
    import pytesseract
    d = pytesseract.image_to_data(img, lang="eng", config="--psm 11", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf < min_conf or not WORD.fullmatch(t) or len(t) < 3 and t.lower() not in ("of", "a"):
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        if h < 0.008 * img.height or h > 0.08 * img.height:
            continue
        words.append({"t": t, "x0": x, "y0": y, "x1": x + w, "y1": y + h, "h": h})
    labels: list[dict] = [dict(w, ws=[(w["x0"], w["t"])]) for w in words]
    merged = True
    while merged:                                                                # join words of one line, whatever order OCR listed them in
        merged = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                same_line = abs((a["y0"] + a["y1"]) / 2 - (b["y0"] + b["y1"]) / 2) < 0.6 * max(a["h"], b["h"])
                gap = max(b["x0"] - a["x1"], a["x0"] - b["x1"])
                if same_line and gap < 1.1 * max(a["h"], b["h"]):
                    a["ws"] += b["ws"]
                    a["x0"], a["x1"] = min(a["x0"], b["x0"]), max(a["x1"], b["x1"])
                    a["y0"], a["y1"], a["h"] = min(a["y0"], b["y0"]), max(a["y1"], b["y1"]), max(a["h"], b["h"])
                    del labels[j]
                    merged = True
                    break
            if merged:
                break
    for lb in labels:
        lb["t"] = " ".join(t for _, t in sorted(lb["ws"]))
    labels.sort(key=lambda l: (round((l["y0"] + l["y1"]) / 2 / 40), l["x0"]))
    out = []
    for lb in labels:
        name = re.sub(r"\s+", " ", lb["t"]).strip(" -/")
        if len(name) >= 3:
            out.append({"name": name[:60], "box": (lb["x0"], lb["y0"], lb["x1"], lb["y1"]), "h": lb["h"]})
    return out


def _dist_point_box(px, py, box) -> float:
    x0, y0, x1, y1 = box
    dx = max(x0 - px, 0, px - x1)
    dy = max(y0 - py, 0, py - y1)
    return math.hypot(dx, dy)


def leader_segments(img: Image.Image, labels: list[dict]) -> list[dict]:
    """Segments whose one end touches a label; a label gets the longest one pointing away from it.  Nearest-first, one segment per label."""
    import cv2
    rgb = np.asarray(img.convert("RGB"))
    a = rgb.max(-1)
    dark = (a < 100).astype(np.uint8) * 255                                      # black-ish strokes only: coloured fills are not lines
    for lb in labels:                                                            # the words themselves are not lines
        x0, y0, x1, y1 = [int(v) for v in lb["box"]]
        dark[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = 0
    H, W = a.shape
    segs = cv2.HoughLinesP(dark, 1, np.pi / 180, threshold=int(0.025 * max(W, H)), minLineLength=int(0.045 * max(W, H)), maxLineGap=int(0.008 * max(W, H)))
    segs = [] if segs is None else [tuple(map(float, r)) for r in np.asarray(segs).reshape(-1, 4)]
    STATS.update(segments=len(segs), labels=len(labels))
    cands = []
    for li, lb in enumerate(labels):
        reach = max(2.2 * lb["h"], 0.012 * max(W, H))
        bx0, by0, bx1, by1 = lb["box"]
        pad = 0.35 * lb["h"]
        for (x0, y0, x1, y1) in segs:
            for (nx, ny, fx, fy) in ((x0, y0, x1, y1), (x1, y1, x0, y0)):
                dn = _dist_point_box(nx, ny, lb["box"])
                df = _dist_point_box(fx, fy, lb["box"])
                length = math.hypot(fx - nx, fy - ny)
                if dn <= reach and df > dn + 0.2 * length and length > 0:
                    ux, uy = (nx - fx) / length, (ny - fy) / length          # the line, run backwards from its near end, must pass through the label
                    hit = dn <= 1.1 * lb["h"] or any(bx0 - pad <= nx + ux * t <= bx1 + pad and by0 - pad <= ny + uy * t <= by1 + pad for t in range(0, int(reach + lb["h"] * 2), 2))
                    if hit:
                        cands.append((dn - 0.15 * length, li, (nx, ny, fx, fy)))
    cands.sort(key=lambda c: c[0])
    taken_l, taken_s, out = set(), [], []
    for _, li, (nx, ny, fx, fy) in cands:
        if li in taken_l:
            continue
        if any(math.hypot(fx - tx, fy - ty) < 0.02 * max(W, H) for tx, ty in taken_s):      # two labels cannot point at the same tip
            continue
        taken_l.add(li)
        taken_s.append((fx, fy))
        out.append({"label": li, "near": (nx, ny), "tip": (fx, fy)})
    return out


def extract(raw: Image.Image) -> tuple[Image.Image, list[dict], dict]:
    """(clean image, parts [{name, anchor_xy|None, text_xy}], info).  Parts are in reading order."""
    import cv2
    img = _prepare(raw)
    labels = ocr_labels(img)
    segs = leader_segments(img, labels) if labels else []
    by_label = {s["label"]: s for s in segs}
    mask = np.zeros((img.height, img.width), np.uint8)
    for lb in labels:
        x0, y0, x1, y1 = [int(v) for v in lb["box"]]
        mask[max(0, y0 - 3):y1 + 4, max(0, x0 - 3):x1 + 4] = 255
    for s in segs:
        nx, ny = s["near"]
        tx, ty = s["tip"]
        L = max(1.0, math.hypot(tx - nx, ty - ny))
        ux, uy = (tx - nx) / L, (ty - ny) / L
        ext = 0.02 * max(img.size)                                                # Hough ends fall a little short of the real stroke
        cv2.line(mask, (int(nx - ux * ext), int(ny - uy * ext)), (int(tx + ux * ext * 0.5), int(ty + uy * ext * 0.5)), 255, max(7, int(0.006 * max(img.size))))
    clean = cv2.inpaint(np.asarray(img)[:, :, ::-1].copy(), mask, 4, cv2.INPAINT_TELEA)[:, :, ::-1]
    parts = []
    for i, lb in enumerate(labels):
        s = by_label.get(i)
        cx, cy = (lb["box"][0] + lb["box"][2]) / 2, (lb["box"][1] + lb["box"][3]) / 2
        parts.append({"name": lb["name"], "anchor_xy": s["tip"] if s else None, "text_xy": (cx, cy)})
    return Image.fromarray(np.ascontiguousarray(clean)), parts, {"labels": len(labels), "anchored": len(by_label), "size": img.size}
