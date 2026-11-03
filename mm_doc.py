"""
mm_doc.py - the document model.  One ordered list of blocks feeds the on-screen view, the PDF and the slide video.

Block types (dicts with key "t"):
    h1 {text}              h2 {text}              p {text}               bullets {items:[str]}
    image {path, caption}  equation {text}        terms {items:[{term, meaning}]}
    qa {items:[{q, a}]}    (printed as questions; the answers go in an Answer key at the end)
    note {text}            parts {items:[{name, text, colour}]}           (colour-keyed table, used for figures)
"""
from __future__ import annotations

import re
from typing import Optional

import mm_core as C


def h1(text): return {"t": "h1", "text": text}
def h2(text): return {"t": "h2", "text": text}
def h3(text): return {"t": "h3", "text": text}
def p(text): return {"t": "p", "text": text}
def bullets(items): return {"t": "bullets", "items": [i for i in items if i]}
def image(path, caption=""): return {"t": "image", "path": path, "caption": caption}
def note(text): return {"t": "note", "text": text}


def md_to_blocks(md: str) -> list[dict]:
    """Markdown with '## Heading', '- bullets' and paragraphs -> blocks."""
    out: list[dict] = []
    buf: list[str] = []
    items: list[str] = []

    def flush_p():
        if buf:
            out.append(p(" ".join(buf).strip()))
            buf.clear()

    def flush_b():
        if items:
            out.append(bullets(list(items)))
            items.clear()

    for raw in (md or "").splitlines():
        line = raw.strip()
        if not line:
            flush_p()
            flush_b()
        elif line.startswith("#"):
            flush_p()
            flush_b()
            out.append(h2(line.lstrip("# ").strip()))
        elif re.match(r"^([-*•]|\d+[.)])\s+", line):
            flush_p()
            items.append(re.sub(r"^([-*•]|\d+[.)])\s+", "", line))
        else:
            flush_b()
            buf.append(line)
    flush_p()
    flush_b()
    return out


def blocks_to_markdown(blocks: list[dict], answers: bool = False) -> str:
    out: list[str] = []
    for b in blocks:
        t = b["t"]
        if t == "h1":
            out.append(f"# {b['text']}")
        elif t == "h2":
            out.append(f"## {b['text']}")
        elif t == "h3":
            out.append(f"### {b['text']}")
        elif t in ("p", "note", "equation"):
            out.append(b["text"])
        elif t == "bullets":
            out.append("\n".join(f"- {i}" for i in b["items"]))
        elif t == "terms":
            out.append("\n".join(f"- **{i['term']}**: {i['meaning']}" for i in b["items"]))
        elif t == "qa":
            out.append("\n".join(f"{n}. {i['q']}" + (f"\n   *Answer:* {i['a']}" if answers else "") for n, i in enumerate(b["items"], 1)))
        elif t == "parts":
            out.append("\n".join(f"- **{i['name']}**: {i['text']}" for i in b["items"]))
        elif t == "image":
            out.append(f"*{b.get('caption') or 'Picture'}*")
    return "\n\n".join(x for x in out if x)


def first_sentence(text: str, max_words: int = 26) -> str:
    s = re.split(r"(?<=[.!?])\s+", (text or "").strip())[0]
    return C.words_trim(s, max_words)


# --------------------------------------------------------------------------------------------- builders
def guide_blocks(title: str, level_label: str, explanation_md: str, terms: list[dict], qa: list[dict], images: list[tuple[str, str]],
                 source_note: str, with_questions: bool = True) -> list[dict]:
    blocks = [h1(title), note(level_label)]
    for path, cap in images:
        blocks.append(image(path, cap))
    body = md_to_blocks(explanation_md)
    blocks += body
    if terms:
        blocks += [h2("Key terms"), {"t": "terms", "items": terms}]
    secs = [b for i, b in enumerate(body) if b["t"] == "h2"]
    recap = []
    for i, b in enumerate(body):
        if b["t"] == "h2" and i + 1 < len(body) and body[i + 1]["t"] in ("p", "bullets"):
            nxt = body[i + 1]
            recap.append(first_sentence(nxt["text"]) if nxt["t"] == "p" else (nxt["items"][0] if nxt["items"] else ""))
    if recap:
        blocks += [h2("Remember"), bullets(recap[:5])]
    if with_questions and qa:
        blocks += [h2("Practice questions"), {"t": "qa", "items": qa}]
    if source_note:
        blocks.append(note(source_note))
    return blocks


def notes_blocks(title: str, level_label: str, summary_md: str, images: list[tuple[str, str]], transcript: Optional[list[dict]] = None) -> list[dict]:
    blocks = [h1(title), note(level_label)]
    blocks += md_to_blocks(summary_md)
    for path, cap in images:
        blocks.append(image(path, cap))
    if transcript:
        blocks.append(h2("Full discussion"))
        for m in transcript:
            who = "You" if m["role"] == "user" else "Mwalimu"
            if m.get("text"):
                blocks.append(p(f"{who}: {m['text']}"))
    return blocks


def process_blocks(fig: dict, data_dir, overview_path: Optional[str]) -> list[dict]:
    blocks = [h1(fig["title"]), note(fig.get("level_label", ""))]
    if overview_path:
        blocks.append(image(overview_path, fig["title"]))
    blocks.append(h2("Step by step"))
    for i, st in enumerate(fig["steps"], 1):
        if st.get("image_path") and st.get("status") == "ready":
            blocks.append(image(st["image_path"], f"Step {i}: {st['title']}"))
        blocks.append(p(f"Step {i} - {st['title']}. {st['text']}"))
        if st.get("flows"):
            blocks.append(p("Arrows: " + "; ".join(f"{f['from']} to {f['to']}" + (f" ({f['what']})" if f.get("what") else "") for f in st["flows"]) + "."))
    if fig.get("source_note"):
        blocks.append(note(fig["source_note"]))
    return blocks


# How much each export says.  One table so notes, slides and video always agree.
DEPTHS = {
    "brief":    dict(label="Brief",    bullets=2, bullet_words=18, narration=55,  slides=8,  answer_sections=("short answer",)),
    "standard": dict(label="Standard", bullets=3, bullet_words=24, narration=90,  slides=14, answer_sections=("short answer", "how it works", "explanation", "a real example", "a common mistake", "key points")),
    "deep":     dict(label="Deep",     bullets=4, bullet_words=28, narration=130, slides=22, answer_sections=None),     # None = every section
}
DEFAULT_DEPTH = "standard"


def depth_params(depth: Optional[str]) -> dict:
    return DEPTHS.get(depth or DEFAULT_DEPTH, DEPTHS[DEFAULT_DEPTH])


def sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if x.strip()]


def _answer_blocks(answer: str, keep: Optional[tuple]) -> list[dict]:
    """The tutor's own answer as notes: its section headings become sub-headings, nothing is re-summarised."""
    out: list[dict] = []
    skip = False
    for b in md_to_blocks(answer):
        if b["t"] == "h2":
            skip = keep is not None and C._norm(b["text"]) not in keep
            if not skip:
                out.append(h3(b["text"]))
        elif not skip:
            out.append(b)
    return out


def discussion_blocks(title: str, level_label: str, turns: list[dict], depth: str = DEFAULT_DEPTH, transcript: bool = False) -> list[dict]:
    """Notes that follow the discussion: for every question the learner asked, the answer's own sections (and its pictures).
    turns = [{"role", "text", "images": [(path, caption)]}].  No new summarising call, so nothing is invented."""
    dp = depth_params(depth)
    blocks = [h1(title), note(f"{level_label} - {dp['label']} notes".strip(" -"))]
    key_points: list[str] = []
    q = ""
    for t in turns:
        if t["role"] == "user":
            q = t["text"]
            continue
        head = C.words_trim(q, 16).rstrip(".") if q else "Overview"
        blocks.append(h2(head))
        body = _answer_blocks(t["text"], dp["answer_sections"])
        blocks += body or [p(t["text"])]
        for pth, cap in t.get("images", []):
            blocks.append(image(pth, cap))
        first = next((b for b in body if b["t"] in ("p", "bullets")), None)
        if first:
            key_points.append(first_sentence(first["text"] if first["t"] == "p" else (first["items"][0] if first["items"] else ""), 30))
        q = ""
    if key_points:
        blocks += [h2("Key points to remember"), bullets([k for k in key_points if k])]
    if transcript:
        blocks.append(h2("Full discussion"))
        for m in turns:
            if m.get("text"):
                blocks.append(p(f"{'You' if m['role'] == 'user' else 'Mwalimu'}: {m['text']}"))
    return blocks


def slides_from_blocks(blocks: list[dict], max_slides: Optional[int] = None, narration_words: Optional[int] = None, depth: Optional[str] = None) -> list[dict]:
    """[{title, bullets, narration}]: one slide per section.  Bullets are the section's first sentences; the narration is the whole section
    (up to the depth's word limit), so the video says what the notes say."""
    dp = depth_params(depth)
    max_slides = max_slides or dp["slides"]
    narration_words = narration_words or dp["narration"]
    title = next((b["text"] for b in blocks if b["t"] == "h1"), "Lesson")
    cands: list[dict] = []
    h2t = ""
    cur: Optional[dict] = None

    def start(t):
        nonlocal cur
        cur = {"title": t, "sents": [], "extra": []}
        cands.append(cur)

    for b in blocks:
        if b["t"] == "h2":
            h2t = b["text"]
            start(h2t)
        elif b["t"] == "h3":
            start(C.words_trim(f"{h2t}: {b['text']}", 12).rstrip(".") if h2t and h2t.lower() not in ("overview",) else b["text"])
        elif cur is None:
            continue
        elif b["t"] == "p":
            cur["sents"] += sentences(b["text"])
        elif b["t"] == "bullets":
            cur["sents"] += [i for i in b["items"] if i]
        elif b["t"] == "terms":
            cur["extra"] += [f"{i['term']}: {first_sentence(i['meaning'], 16)}" for i in b["items"]]
        elif b["t"] == "qa":
            cur["extra"] += [i["q"] for i in b["items"]]
    keep = [c for c in cands if (c["sents"] or c["extra"]) and c["title"].lower() not in ("full discussion",)]
    if len(keep) > max_slides:                                    # thin evenly instead of cutting off the end of the discussion
        step = len(keep) / max_slides
        keep = [keep[int(i * step)] for i in range(max_slides)]
    out = [{"title": title, "bullets": [], "narration": f"Let's learn about {title}."}]
    for c in keep:
        src = c["sents"] or c["extra"]
        bullets_ = [C.words_trim(x, dp["bullet_words"]) for x in (src[:dp["bullets"]] if c["sents"] else c["extra"][:dp["bullets"]])]
        spoken = " ".join(c["sents"] or c["extra"])
        out.append({"title": c["title"], "bullets": bullets_, "narration": C.words_trim(f"{c['title']}. {spoken}", narration_words)})
    out.append({"title": "Well done", "bullets": [], "narration": "That is the key points. Read the notes again and try the practice questions."})
    return out
