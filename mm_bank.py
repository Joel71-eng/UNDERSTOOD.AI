"""
mm_bank.py - the diagram bank at question time (read-only; built offline by bank_build.py).

Bank folder (UNDERSTOOD_BANK_DIR, default <data>/bank), a local copy of the Drive master:
    <id>.png  <id>.json  approved.txt  index.json
A topic is served from the bank only when its id is listed in approved.txt.  Otherwise the generator is used and the miss is
logged to <data>/miss_log.jsonl, so the bank grows from real demand.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import mm_core as C

_DROP = re.compile(r"\b(a|an|the|of|labell?ed|diagram|drawing|picture|image|structure|parts?|anatomy|anatomical|illustration|show|me|draw|human)\b")


def topic_key(t: str) -> str:
    """'a labelled diagram of the human heart' -> 'heart'.  Exact words only: 'heart attack' stays 'heart attack'."""
    return re.sub(r"\s+", " ", _DROP.sub(" ", C._norm(t))).strip()


class Bank:
    def __init__(self, root: Path, miss_log: Optional[Path] = None):
        self.root = Path(root)
        self.miss_path = miss_log
        self._keys: dict[str, str] = {}
        self._mtime = None

    def _load(self):
        ap = self.root / "approved.txt"
        if not ap.exists():
            self._keys = {}
            return
        m = ap.stat().st_mtime
        if m == self._mtime:
            return
        self._mtime = m
        keys: dict[str, str] = {}
        for bid in [x.strip() for x in ap.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]:
            jf = self.root / f"{bid}.json"
            if not (jf.exists() and (self.root / f"{bid}.png").exists()):
                continue
            try:
                rec = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:                                                    # noqa: BLE001
                continue
            for a in [rec.get("title", ""), *rec.get("aliases", [])]:
                if topic_key(a):
                    keys.setdefault(topic_key(a), bid)                  # if several are approved for a topic, the first one listed wins
        self._keys = keys

    def enabled(self) -> bool:
        self._load()
        return bool(self._keys)

    def lookup(self, topic: str) -> Optional[dict]:
        self._load()
        bid = self._keys.get(topic_key(topic))
        if not bid:
            return None
        rec = json.loads((self.root / f"{bid}.json").read_text(encoding="utf-8"))
        rec["png"] = str(self.root / f"{bid}.png")
        return rec

    def miss(self, topic: str):
        if not self.miss_path or not self.enabled():
            return
        try:
            with open(self.miss_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "topic": topic}) + "\n")
        except Exception:                                                        # noqa: BLE001
            pass


def prepare(rec: dict, size: int = 1024) -> tuple[Image.Image, dict]:
    """Deterministic clean-up: crop the white margin, pad to a square, standard size.  Returns (image, {part: (x, y) in pixels})."""
    im = Image.open(rec["png"]).convert("RGB")
    W, H = im.size
    a = np.asarray(im)
    ys, xs = np.where(a.min(-1) < 245)
    if len(xs):
        m = int(0.04 * max(W, H))
        x0, y0, x1, y1 = max(0, xs.min() - m), max(0, ys.min() - m), min(W, xs.max() + m), min(H, ys.max() + m)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    side = max(x1 - x0, y1 - y0)
    padx, pady = (side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(im.crop((x0, y0, x1, y1)), (padx, pady))
    scale = size / side
    out = canvas.resize((size, size), Image.LANCZOS)
    pts = {}
    for p in rec.get("parts", []):
        if p.get("anchor"):
            x, y = p["anchor"][0] * W, p["anchor"][1] * H
            pts[p["name"]] = (float((x - x0 + padx) * scale), float((y - y0 + pady) * scale))
    return out, pts


def bank_regions(img: Image.Image, anchors: dict) -> tuple[list[dict], np.ndarray, dict]:
    """Regions of the bank picture plus ONE region per part, built around the source's own leader-line end:
    the arrow tip is that point, snapped inside the region (nearest region pixel if it fell on an outline).
    A part whose point lies in a region another part already owns gets a small disc of its own.
    Returns (regions, label_map, {part: (region id, (x, y))})."""
    regions, lmap = C.find_regions(img)
    H, W = lmap.shape
    regs = {r["id"]: dict(r) for r in regions}
    lm = lmap.copy()
    dist, (iy, ix) = ndi.distance_transform_edt(lm == 0, return_indices=True)
    claimed: dict[int, str] = {}
    out: dict = {}
    yy, xx = np.mgrid[0:H, 0:W]
    for part, (x, y) in anchors.items():
        xi, yi = int(min(W - 1, max(0, round(x)))), int(min(H - 1, max(0, round(y))))
        rid = int(lm[yi, xi])
        pt = (float(xi), float(yi))
        if rid == 0 and dist[yi, xi] <= 0.05 * W:                        # on an outline: take the nearest region pixel
            nx, ny = int(ix[yi, xi]), int(iy[yi, xi])
            rid, pt = int(lm[ny, nx]), (float(nx), float(ny))
        if rid and rid not in claimed:
            claimed[rid] = part
            out[part] = (rid, pt)
            continue
        new = max(regs, default=0) + 1
        if new > 250:
            continue
        r = max(10, int(0.03 * W))
        disc = (xx - xi) ** 2 + (yy - yi) ** 2 <= r * r
        lm[disc] = new
        regs[new] = dict(id=new, area=int(disc.sum()), anchor=(float(xi), float(yi)))
        claimed[new] = part
        out[part] = (new, (float(xi), float(yi)))
    return list(regs.values()), lm, out


def tint(img: Image.Image, amount: Optional[float] = None) -> Image.Image:
    """Lighter colours for bank pictures: every fill is blended toward white, thin dark outlines are kept (so parts stay separated).
    amount 0 = source colours, 1 = white.  UNDERSTOOD_BANK_TINT changes the default 0.45."""
    a = float(os.environ.get("UNDERSTOOD_BANK_TINT", "0.45")) if amount is None else amount
    if a <= 0:
        return img.convert("RGB")
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    dark = arr.max(-1) < 110
    thin = dark & ~ndi.binary_opening(dark, iterations=2)          # outlines are thin; a big dark FILL is not an outline
    out = arr * (1 - a) + 255 * a
    out[thin] = arr[thin]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
