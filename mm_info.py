"""
mm_info.py - compose process infographics from per-step pictures (all text, numbers and arrows are drawn by code).

    compose_overview(fig, data_dir, lit=None)  -> PIL image: all steps laid out with arrows (ring for cycles, rows otherwise)
    ribbon(fig, current, lit_to, width)        -> PIL strip: numbered dots joined by arrows (used under each video step)
    step_scene_image(...)                      -> a step picture with its ribbon underneath

`lit` is the number of arrows already lit (0 = none, n-1 = all for a line, n = all for a cycle).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import mm_core as C

GREY = (205, 208, 214)
INK = (40, 40, 40)


def step_colour(i: int) -> tuple:
    return C.colour_for(i)


def load_step_image(fig: dict, i: int, data_dir: Path, size: int) -> Image.Image:
    st = fig["steps"][i]
    if st.get("image_path") and st.get("status") == "ready":
        try:
            return Image.open(Path(data_dir) / st["image_path"]).convert("RGB").resize((size, size), Image.LANCZOS)
        except Exception:                                                       # noqa: BLE001
            pass
    ph = Image.new("RGB", (size, size), (243, 244, 247))
    d = ImageDraw.Draw(ph)
    d.text((size / 2, size / 2), "not drawn" if st.get("status") == "failed" else "…", font=C.get_font(max(14, size // 9)), fill=(150, 150, 158), anchor="mm")
    return ph


def _arrow(d: ImageDraw.ImageDraw, p0, p1, colour, width: int = 5):
    C._arrow(d, p0, p1, colour, width)


def _edge_point(c, half, toward):
    """Point where the line from centre c towards `toward` leaves the square of half-size `half`."""
    dx, dy = toward[0] - c[0], toward[1] - c[1]
    if dx == 0 and dy == 0:
        return c
    k = half / max(abs(dx), abs(dy))
    return (c[0] + dx * k, c[1] + dy * k)


def _badge(d: ImageDraw.ImageDraw, xy, n: int, colour, r: int):
    d.ellipse([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r], fill=colour, outline="white", width=3)
    d.text(xy, str(n), font=C.get_font(int(r * 1.15), True), fill="white", anchor="mm")


def compose_overview(fig: dict, data_dir: Path, lit: Optional[int] = None, cell: int = 300) -> Image.Image:
    steps = fig["steps"]
    n = len(steps)
    shape = fig.get("shape", "linear")
    cycle = shape == "cycle" and n >= 3
    lit = (n if cycle else n - 1) if lit is None else lit
    title_h = 92
    font_t = C.get_font(int(cell * 0.095), True)
    label_h = int(cell * 0.26)
    gap = int(cell * 0.36)

    pos: list[tuple[float, float]] = []
    if cycle:
        R = max(cell * 0.95, (cell + gap) * n / (2 * math.pi))
        W = H = int(2 * R + cell + 2 * gap)
        cx, cy = W / 2, title_h + (H - title_h) / 2 + 10
        H = H + title_h
        for i in range(n):
            a = -math.pi / 2 + 2 * math.pi * i / n
            pos.append((cx + R * math.cos(a), cy + R * math.sin(a)))
        H = int(cy + R + cell / 2 + label_h + gap)
    else:
        cols = 3 if n <= 6 else 4
        rows = math.ceil(n / cols)
        W = cols * cell + (cols + 1) * gap
        H = title_h + rows * (cell + label_h) + (rows + 1) * gap
        for i in range(n):
            r, c = divmod(i, cols)
            pos.append((gap + c * (cell + gap) + cell / 2, title_h + gap + r * (cell + label_h + gap) + cell / 2))
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((W / 2, title_h / 2 + 4), fig["title"], font=C.get_font(int(cell * 0.15), True), fill=INK, anchor="mm")

    half = cell / 2
    # arrows first (under the pictures)
    for i in range(n - (0 if cycle else 1)):
        j = (i + 1) % n
        a, b = pos[i], pos[j]
        col = step_colour(j) if i < lit else GREY
        if cycle:
            p0, p1 = _edge_point(a, half + 6, b), _edge_point(b, half + 10, a)
            _arrow(d, p0, p1, col)
        else:
            same_row = abs(a[1] - b[1]) < 5
            if same_row:
                _arrow(d, (a[0] + half + 6, a[1]), (b[0] - half - 8, b[1]), col)
            else:                                           # elbow from the end of one row to the start of the next
                y1 = a[1] + half + label_h * 0.9
                ymid = (a[1] + half + label_h + b[1] - half) / 2
                for p, q in (((a[0], y1), (a[0], ymid)), ((a[0], ymid), (b[0], ymid))):
                    d.line([p, q], fill=(255, 255, 255), width=9)
                    d.line([p, q], fill=col, width=5)
                _arrow(d, (b[0], ymid), (b[0], b[1] - half - 8), col)
    for i, (x, y) in enumerate(pos):
        im = load_step_image(fig, i, data_dir, cell)
        canvas.paste(im, (int(x - half), int(y - half)))
        d.rectangle([x - half, y - half, x + half, y + half], outline=step_colour(i), width=4)
        _badge(d, (x - half + 22, y - half + 22), i + 1, step_colour(i), 20)
        t = steps[i]["title"]
        while d.textlength(t, font=font_t) > cell + gap * 0.7 and len(t) > 6:
            t = t[:-2]
        d.text((x, y + half + label_h * 0.5), t, font=font_t, fill=INK, anchor="mm")
    return canvas


def ribbon(fig: dict, current: int, lit_to: int, width: int = 1200, height: int = 78) -> Image.Image:
    """Numbered dots joined by arrows.  Arrows with index < lit_to are lit; `current` dot is ringed; later dots are dim."""
    n = len(fig["steps"])
    cycle = fig.get("shape") == "cycle" and n >= 3
    img = Image.new("RGB", (width, height), (250, 250, 252))
    d = ImageDraw.Draw(img)
    r = min(22, int(width / n / 3.2))
    margin = 50
    xs = [margin + (width - 2 * margin) * i / max(1, n - 1) for i in range(n)]
    y = height * 0.42
    for i in range(n - 1):
        col = step_colour(i + 1) if i < lit_to else GREY
        _arrow(d, (xs[i] + r + 4, y), (xs[i + 1] - r - 6, y), col, 4)
    if cycle:                                              # return path under the dots, from the last step back to the first
        col = step_colour(0) if lit_to >= n else GREY
        yb = y + r + 16
        d.line([(xs[-1], y + r + 3), (xs[-1], yb), (xs[0], yb)], fill=col, width=4)
        _arrow(d, (xs[0], yb), (xs[0], y + r + 6), col, 4)
    for i, x in enumerate(xs):
        st = fig["steps"][i]["status"] if "status" in fig["steps"][i] else "ready"
        reached = i <= current
        col = step_colour(i) if reached else (186, 189, 196)
        if i == current:
            d.ellipse([x - r - 5, y - r - 5, x + r + 5, y + r + 5], outline=col, width=4)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        d.text((x, y), str(i + 1), font=C.get_font(int(r * 1.1), True), fill="white", anchor="mm")
    return img


def step_scene_image(fig: dict, i: int, data_dir: Path, lit_to: int, size: int = 520, width: int = 1200) -> Image.Image:
    """The step picture centred above its ribbon - one video scene image."""
    pic = load_step_image(fig, i, data_dir, size)
    rb = ribbon(fig, i, lit_to, width)
    img = Image.new("RGB", (width, size + rb.height + 6), "white")
    img.paste(pic, ((width - size) // 2, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([(width - size) // 2, 0, (width + size) // 2 - 1, size - 1], outline=step_colour(i), width=5)
    img.paste(rb, (0, size + 6))
    return img


# =====================================================================================================
# A process step drawn on the ONE verified diagram of the organ / system
# =====================================================================================================
def _rect_exit(cx, cy, hw, hh, tx, ty):
    """Where the line from the centre of a box towards (tx, ty) leaves the box."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    k = min(hw / abs(dx) if dx else 1e9, hh / abs(dy) if dy else 1e9)
    return cx + dx * k, cy + dy * k


def compose_system_step(system: dict, step: dict, idx: int, n_steps: int, data_dir: Path) -> Image.Image:
    """The same picture in every step (so the organ never changes between steps).  What changes, drawn by code:
    the parts this step is about keep their full colour and an outline, the rest are washed out; every located part carries its name;
    arrows run between the parts named in the step, labelled with what moves.  Nothing outside the diagram can appear."""
    data_dir = Path(data_dir)
    clean = Image.open(data_dir / system["clean_path"]).convert("RGB")
    lmap = np.array(Image.open(data_dir / system["labelmap_path"]))
    W, H = clean.size
    parts = {p["name"]: p for p in system["parts"] if p.get("visible") and p.get("anchor")}
    active = [n for n in step.get("parts", []) if n in parts]
    col = step_colour(idx)
    arr = np.asarray(clean).astype(np.float32)
    on = np.isin(lmap, [parts[n]["region_id"] for n in active]) if active else np.zeros((H, W), bool)
    arr[~on] = arr[~on] * 0.6 + 255 * 0.4
    if on.any():
        edge = on & ~ndi.binary_erosion(on, iterations=max(2, W // 200))
        arr[edge] = col
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    fs = max(13, W // 42)
    font, fbold = C.get_font(max(11, int(fs * 0.85))), C.get_font(fs, True)
    flows = [dict(f) for f in step.get("flows", []) if f.get("from") in parts and f.get("to") in parts]
    anchor = {n: (float(parts[n]["anchor"][0]), float(parts[n]["anchor"][1])) for n in parts}

    # arrows run from part to part (region centres); the name tags sit on the side AWAY from the arrows so nothing hides them
    dirs: dict[str, float] = {}
    for f in flows:
        (ax, ay), (bx, by) = anchor[f["from"]], anchor[f["to"]]
        dirs[f["from"]] = dirs.get(f["from"], 0.0) + (by - ay)
        dirs[f["to"]] = dirs.get(f["to"], 0.0) + (ay - by)
    pair_count: dict[tuple, int] = {}
    for f in flows:
        a, b = f["from"], f["to"]
        key = tuple(sorted((a, b)))
        k = pair_count.get(key, 0)
        pair_count[key] = k + 1
        (ax, ay), (bx, by) = anchor[a], anchor[b]
        vx, vy = bx - ax, by - ay
        ln = max(1.0, (vx * vx + vy * vy) ** 0.5)
        ux, uy = vx / ln, vy / ln
        p0, p1 = (ax + ux * 12, ay + uy * 12), (bx - ux * 14, by - uy * 14)
        bow = ln * (0.12 + 0.12 * k)
        ctrl = ((p0[0] + p1[0]) / 2 - uy * bow, (p0[1] + p1[1]) / 2 + ux * bow)
        if ln > 30:
            C._arrow(d, p0, p1, col, max(3, W // 190), ctrl=ctrl)
        f["_mid"] = (0.25 * p0[0] + 0.5 * ctrl[0] + 0.25 * p1[0], 0.25 * p0[1] + 0.5 * ctrl[1] + 0.25 * p1[1], ln)
    for n in active:                                                           # a dot where each active part is
        x, y = anchor[n]
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline="white", width=2)

    placed: list[tuple] = []
    order = active + [n for n in parts if n not in active]
    for n in order:
        f_ = fbold if n in active else font
        w_, h_ = d.textlength(n, font=f_) + 12, f_.size + 8
        x, y = anchor[n]
        side = -1 if dirs.get(n, 0.0) > 0 else 1                               # partners below -> tag above, otherwise below
        x = min(max(x, w_ / 2 + 2), W - w_ / 2 - 2)
        r = None
        for k in range(0, 6):
            for sgn in (side, -side):
                yy = min(max(y + sgn * (h_ * 0.5 + 12 + k * h_), h_ / 2 + 2), H - h_ / 2 - 2)
                cand = (x - w_ / 2, yy - h_ / 2, x + w_ / 2, yy + h_ / 2)
                if not any(cand[0] < q[2] and cand[2] > q[0] and cand[1] < q[3] and cand[3] > q[1] for q in placed):
                    r = cand
                    break
            if r:
                break
        r = r or (x - w_ / 2, y - h_ / 2, x + w_ / 2, y + h_ / 2)
        placed.append(r)
        on_ = n in active
        d.rounded_rectangle(r, radius=7, fill="white" if on_ else (246, 246, 248), outline=col if on_ else (200, 200, 206), width=3 if on_ else 1)
        d.text(((r[0] + r[2]) / 2, (r[1] + r[3]) / 2), n, font=f_, fill=(25, 25, 30) if on_ else (140, 140, 148), anchor="mm")

    for f in flows:                                                            # what moves, written on the arrow
        mx, my, ln = f.pop("_mid")
        what = f.get("what")
        if what and ln > 60:
            tw = d.textlength(what, font=font)
            d.rounded_rectangle([mx - tw / 2 - 6, my - font.size / 2 - 3, mx + tw / 2 + 6, my + font.size / 2 + 3], radius=6, fill="white", outline=col, width=2)
            d.text((mx, my), what, font=font, fill=(30, 30, 35), anchor="mm")
    _badge(d, (fs + 8, fs + 8), idx + 1, col, fs)
    return img
