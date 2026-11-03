"""
bank_build.py - OFFLINE builder for the Understood diagram bank.  Run it on Colab (internet + Linux + Drive), never in the app.

    !pip -q install cairosvg lxml pillow requests pytesseract opencv-python-headless langdetect
    !apt -q install -y tesseract-ocr
    from google.colab import drive; drive.mount('/content/drive')
    !python bank_build.py --out /content/drive/MyDrive/understood_mm/bank --topics "human heart,human brain"

What it does, per topic
  1. looks for labelled drawings on Wikimedia Commons with a free licence (NC / ND refused, share-alike kept and credited):
       - file search for SVG drawings AND for PNG/JPG pictures (several pages deep),
       - the Commons CATEGORIES that match the topic (and their sub-categories), which is where most pictures live
     English only: file names with another language's code or script are skipped, and labels that are not English are rejected.
  2. SVG: every <text> is a label; the leader line next to it gives the part's ANCHOR (the tip end of the line).
     PNG/JPG (used when no SVG gives enough labels): OCR reads the words, line detection finds each pointer, both are painted out
     (bank_raster.py).  Raster results are always flagged "check carefully" in the review app.
  3. removes all text + the leader lines, renders a clean PNG (cairosvg)
  4. writes   <id>.png   <id>.json   (title, aliases, credit, parts[{name, anchor(0-1), side}])   and a numbered contact sheet
     the best picture is saved as <id>; the next best ones as <id>__alt2, <id>__alt3 (look at all of them in the review app)
  5. run  streamlit run bank_review.py -- --bank <this folder>  : look, fix a name, Approve / Reject (writes approved.txt for you)
The app only ever reads approved ids.  Nothing here runs at question time, so no SVG tools are needed on the PC.

Resumable: finished topics are skipped on the next run, the index is written after every topic, and every downloaded file is cached in
<out>/_cache, so a crash or a closed Colab costs nothing.  Originals are kept in <out>/source/<id>.svg|jpg|png (the bank PNG is the CLEANED render).

Layout of --out:   bank/<id>.png  <id>.json  index.json  approved.txt  review/<id>_review.png  miss_log.jsonl (written by the app)
"""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import requests
from lxml import etree

import os
UA = "UnderstoodBankBuilder/1.0 (non-commercial education project" + (f"; contact: {os.environ['UNDERSTOOD_CONTACT']}" if os.environ.get("UNDERSTOOD_CONTACT") else "") + ")"
API = "https://commons.wikimedia.org/w/api.php"
SVG_NS = "http://www.w3.org/2000/svg"
RENDER_W = 1400

# id -> (aliases, search query).  Start small; grow from miss_log.jsonl.
TOPICS = {
    "human_heart": (["human heart", "heart", "heart anatomy"], "human heart labelled diagram"),
    "human_brain": (["human brain", "brain", "brain anatomy"], "human brain labelled diagram"),
    "digestive_system": (["digestive system", "digestion", "alimentary canal"], "human digestive system labelled diagram"),
    "respiratory_system": (["respiratory system", "lungs", "human lungs"], "human respiratory system labelled diagram"),
    "kidney": (["kidney", "human kidney", "urinary system"], "human kidney labelled diagram"),
    "eye": (["human eye", "eye", "eye anatomy"], "human eye anatomy labelled diagram"),
    "ear": (["human ear", "ear", "ear anatomy"], "human ear anatomy labelled diagram"),
    "skin": (["skin", "human skin", "skin layers"], "human skin layers labelled diagram"),
    "neuron": (["neuron", "nerve cell"], "neuron structure labelled diagram"),
    "skeleton": (["human skeleton", "skeleton", "bones"], "human skeleton labelled diagram"),
    "plant_cell": (["plant cell"], "plant cell labelled diagram"),
    "animal_cell": (["animal cell"], "animal cell labelled diagram"),
    "flower": (["flower", "flower structure"], "flower parts labelled diagram"),
    "male_reproductive": (["male reproductive system"], "male reproductive system labelled diagram"),
    "female_reproductive": (["female reproductive system"], "female reproductive system labelled diagram"),
    "stomach": (["stomach", "human stomach"], "human stomach anatomy labelled diagram"),
    "liver": (["liver", "human liver"], "human liver anatomy labelled diagram"),
    "nephron": (["nephron", "kidney nephron"], "nephron structure labelled diagram"),
    "alveolus": (["alveolus", "alveoli", "gas exchange in the lungs"], "alveolus gas exchange labelled diagram"),
    "nervous_system": (["nervous system", "human nervous system"], "human nervous system labelled diagram"),
    "spinal_cord": (["spinal cord"], "spinal cord cross section labelled diagram"),
    "circulatory_system": (["circulatory system", "blood circulation"], "human circulatory system labelled diagram"),
    "tooth": (["tooth", "human tooth"], "human tooth anatomy labelled diagram"),
    "bone": (["bone structure", "long bone"], "long bone structure labelled diagram"),
    "muscle": (["muscle", "skeletal muscle"], "skeletal muscle structure labelled diagram"),
    "endocrine_system": (["endocrine system", "glands"], "human endocrine system labelled diagram"),
    "lymphatic_system": (["lymphatic system"], "human lymphatic system labelled diagram"),
    "mitochondrion": (["mitochondrion", "mitochondria"], "mitochondrion structure labelled diagram"),
    "chloroplast": (["chloroplast"], "chloroplast structure labelled diagram"),
    "bacterial_cell": (["bacterial cell", "bacterium", "prokaryotic cell"], "bacterial cell structure labelled diagram"),
    "virus": (["virus", "virus structure"], "virus structure labelled diagram"),
    "dna": (["dna", "dna structure", "double helix"], "dna double helix structure labelled diagram"),
    "leaf": (["leaf", "leaf structure", "leaf cross section"], "leaf cross section labelled diagram"),
    "root": (["root", "root structure"], "plant root structure labelled diagram"),
    "stem": (["stem", "plant stem"], "plant stem cross section labelled diagram"),
    "seed": (["seed", "seed structure"], "seed structure labelled diagram"),
    "fish": (["fish", "fish anatomy"], "fish external anatomy labelled diagram"),
    "frog": (["frog", "frog anatomy"], "frog anatomy labelled diagram"),
    "insect": (["insect", "insect anatomy"], "insect body parts labelled diagram"),
    "atom": (["atom", "atomic structure"], "atom structure labelled diagram"),
    "earth_layers": (["earth layers", "structure of the earth"], "layers of the earth labelled diagram"),
    "volcano": (["volcano"], "volcano cross section labelled diagram"),
    "solar_system": (["solar system"], "solar system planets labelled diagram"),
    "electric_circuit": (["electric circuit", "simple circuit"], "simple electric circuit labelled diagram"),
    "microscope": (["microscope", "light microscope"], "compound light microscope labelled diagram"),
}


# ---------------------------------------------------------------------------------------------- licence + Commons
def free_license(short: str) -> bool:
    s = (short or "").strip().lower()
    if not s or re.search(r"\b(nc|nd)\b|non-?commercial|no-?deriv|fair use|all rights", s):
        return False
    return s.startswith(("cc by", "cc-by", "cc0", "public domain", "pd"))


def strip_html(s: str) -> str:
    import html
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


EXTRA_QUERIES = {
    "human_brain": ["human brain anatomy diagram", "brain lobes cerebrum cerebellum diagram", "brain sagittal section labelled", "brain regions diagram"],
}


def queries_for(tid: str, aliases: list[str], query: str) -> list[str]:
    qs = [query] + EXTRA_QUERIES.get(tid, []) + [f"{aliases[0]} anatomy diagram", f"{aliases[0]} labeled", f"{aliases[0]} structure diagram"]
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


# ISO language codes that mean "not English" when they end a Commons file name:  Heart-fr.svg, Heart_(de).png, Heart es.svg
_NON_EN = set("""ar az be bg bn bs ca cs cy da de el eo es et eu fa fi fr ga gl gu he hi hr hu hy id is it ja ka kk km kn ko lt lv mk ml mn mr ms my
nb ne nl nn no oc pa pl pt ro ru si sk sl sq sr sv sw ta te th tl tr uk ur uz vi zh ast als arz azb bar bat-smg be-tarask bho cbk-zam ckb diq
fy ha hsb ilo jv ku lb lmo mai mg min mrj nds new nov pms scn sco sh simple ta tt war yi yo zh-cn zh-tw zh-hans zh-hant pt-br es-mx fr-ca""".split())
_AMBIGUOUS = {"it", "no", "is", "be", "my", "id", "he", "hi", "me", "as", "an", "ca", "el", "ha", "pa", "si", "sh", "new", "ku", "min", "war", "simple", "sco"}
_FOREIGN_WORDS = {"coeur", "corazon", "herz", "cuore", "serce", "hjerte", "hjarta", "sydan", "jantung", "kalp", "srdce", "cerebro", "cerveau", "gehirn",
                  "cervello", "cerebelo", "systeme", "sistema", "corpo", "humano", "humain", "menschliche", "cellule", "celula", "zelle", "oeil", "ojo", "auge",
                  "oreille", "oreja", "ohr", "poumon", "pulmon", "lunge", "rein", "riñon", "niere", "foie", "higado", "leber", "squelette", "esqueleto", "skelett"}


def english_title(title: str) -> bool:
    """File names only: a quick, strict screen before anything is downloaded.  Anything doubtful is skipped (there are plenty of files)."""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", re.sub(r"^File:", "", title or ""))
    if re.search(r"[^\x00-\x7f]", stem):                                      # accents or another script in the name
        return False
    low = stem.lower()
    toks = [t for t in re.split(r"[\s_\-().,]+", low) if t]
    if any(t in _FOREIGN_WORDS for t in toks):
        return False
    m = re.search(r"([-_(\s])\s*([a-z]{2,3}(?:-[a-z0-9]{2,8})?)\s*\)?$", low)
    if m and m.group(2) in _NON_EN and (m.group(2) not in _AMBIGUOUS or m.group(1) in "-_("):
        return False
    if re.search(r"\((?:%s)\)" % "|".join(sorted(_NON_EN - _AMBIGUOUS)), low):
        return False
    return True


def english_labels(names: list[str]) -> tuple[bool, str]:
    """Labels of a candidate: all plain Latin letters, and (when there is enough text and langdetect is installed) detected as English."""
    clean = [n for n in names if n]
    if not clean:
        return False, "no labels"
    bad = [n for n in clean if re.search(r"[^\x00-\x7f]", n)]
    if bad:
        return False, "non-English characters: " + ", ".join(bad[:3])
    text = ". ".join(clean)
    if len(text.split()) >= 25:
        try:
            from langdetect import DetectorFactory, detect_langs
            DetectorFactory.seed = 0
            top = detect_langs(text)[0]
            if top.lang != "en" and top.prob > 0.8:
                return False, f"looks like {top.lang} ({top.prob:.0%})"
        except Exception:                                                       # noqa: BLE001  (langdetect not installed: the ASCII rule still applies)
            pass
    return True, ""


def _info_to_cand(pg: dict) -> dict | None:
    info = (pg.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata") or {}
    lic = (meta.get("LicenseShortName") or {}).get("value", "")
    mime = info.get("mime")
    url = info.get("url", "")
    if mime not in ("image/svg+xml", "image/png", "image/jpeg") or not free_license(lic) or not url.startswith("https://upload.wikimedia.org/"):
        return None
    if not english_title(pg.get("title", "")):
        return None
    kind = "svg" if mime == "image/svg+xml" else "raster"
    if kind == "raster":
        if min(info.get("width") or 0, info.get("height") or 0) < 500:           # too small to read labels from
            return None
        if (info.get("width") or 0) > 2000:
            url = info.get("thumburl") or url                                    # a 1600 px copy: faster, and plenty for OCR
    return {"title": pg["title"], "url": url, "page_url": info.get("descriptionurl", ""), "license": lic, "kind": kind,
            "author": strip_html((meta.get("Artist") or {}).get("value", ""))[:80] or "unknown author"}


_IMG_PROPS = {"prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiextmetadatafilter": "LicenseShortName|Artist", "iiurlwidth": 1600}


class RateLimited(Exception):
    pass


OFFLINE = [False]                          # True = never touch the network: use only what is in the cache (see bank_fetch.py)
_LAST = [0.0]
CACHE: list = [None]                       # set to <out>/_cache by main(): every file is downloaded once, ever


def _polite(gap: float = 1.2):
    wait = _LAST[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _LAST[0] = time.time()


def _get(url: str, **kw):
    """GET with a pause between requests and a patient retry on 429 / 5xx (Wikimedia asks for this; Colab shares one address with many users)."""
    for attempt in range(7):
        _polite()
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60, **kw)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code not in (429, 500, 502, 503, 504):
            return r
        try:
            ra = float(r.headers.get("Retry-After", "")) if r is not None else 0
        except ValueError:
            ra = 0
        if ra > 120:                                                            # Wikimedia asked for a long pause: stop cleanly, nothing is lost
            raise RateLimited(f"Wikimedia asked us to wait {ra / 60:.0f} minutes")
        pause = max(ra, min(90, 4 * 2 ** attempt))
        print(f"   (busy {r.status_code if r is not None else 'no answer'}: waiting {pause:.0f}s, try {attempt + 1}/7)")
        time.sleep(pause)
    return r


def download(url: str) -> bytes | None:
    """File bytes from the disk cache, else from Commons (then cached).  None if it cannot be had."""
    import hashlib
    path = None
    if CACHE[0]:
        path = Path(CACHE[0]) / (hashlib.sha1(url.encode()).hexdigest()[:20] + Path(url.split("?")[0]).suffix[:6])
        if path.exists() and path.stat().st_size > 0:
            return path.read_bytes()
    if OFFLINE[0]:
        return None
    r = _get(url)
    if r is None or r.status_code != 200 or len(r.content) > 12_000_000:
        return None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return r.content


def _api(params: dict) -> dict:
    """Commons API call.  Answers are cached on disk too, so a build can run later without the network (bank_fetch.py fills the cache from a PC)."""
    import hashlib
    path = None
    if CACHE[0]:
        path = Path(CACHE[0]) / ("api_" + hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20] + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    if OFFLINE[0]:
        raise LookupError("not in the offline cache")
    r = _get(API, params={"format": "json", "formatversion": 2, **params})
    r.raise_for_status()
    j = r.json()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(j), encoding="utf-8")
    return j


def search_files(query: str, kind: str, pages: int = 2) -> list[dict]:
    """File search.  kind: 'drawing' (SVG) or 'bitmap' (PNG/JPG).  Follows the continue token for `pages` pages of 50."""
    out: list[dict] = []
    cont: dict = {}
    for _ in range(pages):
        j = _api({"action": "query", "generator": "search", "gsrnamespace": 6, "gsrlimit": 50, "gsrsearch": f"{query} filetype:{kind}", **_IMG_PROPS, **cont})
        for pg in j.get("query", {}).get("pages", []):
            c = _info_to_cand(pg)
            if c:
                out.append(c)
        cont = j.get("continue") or {}
        if not cont:
            break
    return out


def find_categories(topic: str, limit: int = 6) -> list[str]:
    """Commons categories whose NAME matches the topic words, e.g. 'Category:Diagrams of the human heart'."""
    j = _api({"action": "query", "list": "search", "srsearch": f"{topic} diagram", "srnamespace": 14, "srlimit": limit})
    return [h["title"] for h in j.get("query", {}).get("search", []) if re.search(r"diagram|anatomy|labell?ed|structure|illustration|drawing|svg", h["title"], re.I)]


def category_files(cat: str, depth: int = 1, per_cat: int = 100) -> list[dict]:
    """Files of a category and of its sub-categories (depth levels down)."""
    out: list[dict] = []
    todo, seen = [(cat, 0)], set()
    while todo:
        c, d = todo.pop(0)
        if c in seen:
            continue
        seen.add(c)
        try:
            j = _api({"action": "query", "generator": "categorymembers", "gcmtitle": c, "gcmtype": "file", "gcmlimit": per_cat, **_IMG_PROPS})
            for pg in j.get("query", {}).get("pages", []):
                cand = _info_to_cand(pg)
                if cand:
                    out.append(cand)
            if d < depth:
                sub = _api({"action": "query", "list": "categorymembers", "cmtitle": c, "cmtype": "subcat", "cmlimit": 12})
                todo += [(m["title"], d + 1) for m in sub.get("query", {}).get("categorymembers", [])]
        except RateLimited:
            raise
        except Exception as e:                                                  # noqa: BLE001
            print(f"   category {c}: {type(e).__name__}")
    return out


def gather_candidates(tid: str, aliases: list[str], query: str) -> list[dict]:
    """Everything worth trying for a topic: SVG search, PNG/JPG search, matching categories.  Unique by title; SVGs first."""
    cands: list[dict] = []
    seen: set = set()

    def add(items):
        for c in items:
            if c["title"] not in seen:
                seen.add(c["title"])
                cands.append(c)
    for q in queries_for(tid, aliases, query):
        for kind in ("drawing", "bitmap"):
            try:
                add(search_files(q, kind, pages=2 if q == query else 1))
            except RateLimited:
                raise
            except Exception as e:                                              # noqa: BLE001
                print(f"   search failed ({q}, {kind}): {type(e).__name__}")
    try:
        for cat in find_categories(aliases[0]):
            add(category_files(cat))
    except RateLimited:
        raise
    except Exception as e:                                                      # noqa: BLE001
        print(f"   category search failed: {type(e).__name__}")
    cands.sort(key=lambda c: 0 if c["kind"] == "svg" else 1)
    print(f"   {len(cands)} candidates ({sum(1 for c in cands if c['kind'] == 'svg')} SVG, {sum(1 for c in cands if c['kind'] == 'raster')} PNG/JPG), English file names only")
    return cands


# ---------------------------------------------------------------------------------------------- tiny SVG geometry
def _mul(a, b):                                   # affine 2x3 as (a,b,c,d,e,f)
    return (a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1], a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3],
            a[0]*b[4]+a[2]*b[5]+a[4], a[1]*b[4]+a[3]*b[5]+a[5])


def parse_transform(s: str):
    m = (1, 0, 0, 1, 0, 0)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate":
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            a = math.radians(v[0]); c, s_ = math.cos(a), math.sin(a)
            t = (c, s_, -s_, c, 0, 0)
            if len(v) == 3:
                t = _mul(_mul((1, 0, 0, 1, v[1], v[2]), t), (1, 0, 0, 1, -v[1], -v[2]))
        else:
            continue
        m = _mul(m, t)
    return m


def ctm(el):
    chain = []
    while el is not None:
        chain.append(el.get("transform"))
        el = el.getparent()
    m = (1, 0, 0, 1, 0, 0)
    for t in reversed(chain):
        if t:
            m = _mul(m, parse_transform(t))
    return m


def apply(m, x, y):
    return m[0]*x+m[2]*y+m[4], m[1]*x+m[3]*y+m[5]


def num(s, default=0.0):
    mt = re.match(r"\s*(-?[\d.]+(?:e-?\d+)?)", s or "")
    return float(mt.group(1)) if mt else default


def viewbox(root):
    vb = root.get("viewBox")
    if vb:
        x, y, w, h = [float(t) for t in re.split(r"[\s,]+", vb.strip())]
        return x, y, w, h
    return 0.0, 0.0, num(root.get("width"), 1000), num(root.get("height"), 1000)


def local(el) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


def text_of(t) -> str:
    return re.sub(r"\s+", " ", "".join(t.itertext())).strip()


def path_ends(d: str):
    """(start, end) of a SHORT OPEN path made of at most 4 segments - absolute or relative M L H V C S Q T A - else None.
    Closed shapes (Z) and long outlines are not leader lines."""
    toks = re.findall(r"([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)", d or "")
    cmd, args, segs, closed = None, [], 0, False
    cur = start = None
    need = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "A": 7}
    out_start = None
    for c, n in toks:
        if c:
            if c in "Zz":
                closed = True
                cmd, args = None, []
                continue
            cmd, args = c, []
            continue
        if cmd is None:
            continue
        args.append(float(n))
        k = need[cmd.upper()]
        if len(args) < k:
            continue
        rel = cmd.islower()
        u = cmd.upper()
        x0, y0 = cur if cur else (0.0, 0.0)
        if u == "H":
            pt = ((x0 if not rel else x0) + args[0] if rel else args[0], y0)
        elif u == "V":
            pt = (x0, y0 + args[0] if rel else args[0])
        else:
            ex, ey = args[-2], args[-1]
            pt = (x0 + ex, y0 + ey) if rel else (ex, ey)
        if u == "M":
            if cur is None:
                out_start = pt
            else:
                segs += 99                                    # a second sub-path: not a simple leader
            cmd = "l" if rel else "L"                         # further pairs after M are implicit lines
        else:
            segs += 1
        cur = pt
        args = []
    if closed or out_start is None or cur is None or segs < 1 or segs > 4:
        return None
    return out_start, cur


def line_ends(el):
    """Both end points (user space of the element, then CTM applied) of a <line> or a simple <path>/<polyline>."""
    tag = local(el)
    m = ctm(el)
    if tag == "line":
        p = [(num(el.get("x1")), num(el.get("y1"))), (num(el.get("x2")), num(el.get("y2")))]
    elif tag in ("polyline", "polygon"):
        nums = [float(n) for n in re.findall(r"-?[\d.]+", el.get("points", ""))]
        pts = list(zip(nums[::2], nums[1::2]))
        p = [pts[0], pts[-1]] if len(pts) >= 2 else []
    elif tag == "path":
        pp = path_ends(el.get("d", ""))
        if not pp:
            return None
        p = [pp[0], pp[1]]
    else:
        return None
    return [apply(m, *q) for q in p] if len(p) == 2 else None


def text_pos(t):
    m = ctm(t)
    x, y = num(t.get("x")), num(t.get("y"))
    if (not t.get("x")) and len(t):
        for ch in t:                                  # label positioned by its first <tspan>
            if ch.get("x"):
                x, y = num(ch.get("x")), num(ch.get("y"), y)
                break
    return apply(m, x, y)


# ---------------------------------------------------------------------------------------------- languages
def _lang(el) -> str | None:
    return el.get("systemLanguage") or el.get("{http://www.w3.org/XML/1998/namespace}lang")


def _is_en(v: str | None) -> bool:
    return bool(v) and any(x.strip().lower().startswith("en") for x in v.split(","))


def keep_english(root) -> int:
    """Multilingual SVGs repeat every label per language (<switch> / systemLanguage).  Keep ONE: English if present, else the
    language-less child, else the first.  The others are removed, so they neither steal leader lines nor show in the clean picture."""
    removed = 0
    for sw in [e for e in root.iter() if local(e) == "switch"]:
        kids = [k for k in sw if isinstance(k.tag, str)]
        if len(kids) < 2:
            continue
        pick = None
        for test in (lambda k: _is_en(_lang(k)), lambda k: not _lang(k), lambda k: True):
            pick = next((k for k in kids if test(k)), None)
            if pick is not None:
                break
        for k in kids:
            if k is not pick:
                sw.remove(k)
                removed += 1
    for e in [e for e in root.iter() if isinstance(e.tag, str) and _lang(e) and not _is_en(_lang(e))]:
        if e.getparent() is not None and local(e) in ("text", "g", "tspan"):
            e.getparent().remove(e)
            removed += 1
    return removed


# ---------------------------------------------------------------------------------------------- extraction
STATS: dict = {}


def extract_parts(root) -> tuple[list[dict], list]:
    """-> (parts[{name, text_xy, anchor_xy|None}], elements to delete).  Coordinates in the SVG's user space."""
    STATS["removed_other_languages"] = keep_english(root)
    texts = [t for t in root.iter() if local(t) == "text" and text_of(t)]
    lines = []
    for el in root.iter():
        if local(el) in ("line", "polyline", "path", "polygon"):
            ends = line_ends(el)
            if ends:
                a, b = ends
                length = math.dist(a, b)
                if 6 < length < 0.45 * max(viewbox(root)[2:]):          # leader lines are short; outlines are not
                    lines.append((el, a, b))
    big = max(viewbox(root)[2:])
    cand = []                                            # every (distance, text, leader line, far end) - matched nearest-first, once each
    info = []
    for ti, t in enumerate(texts):
        name = text_of(t)
        if len(name) < 3 or len(name) > 40 or re.fullmatch(r"[\d\W]+", name):
            info.append(None)
            continue
        tx, ty = text_pos(t)
        mfs = re.search(r"font-size:\s*([\d.]+)", t.get("style", ""))
        fs = float(mfs.group(1)) if mfs else (num(t.get("font-size"), 12) or 12)
        wid = len(name) * fs * 0.55                                       # rough text width; leaders touch either end
        ends = [(tx, ty - fs * 0.3), (tx + wid, ty - fs * 0.3), (tx + wid / 2, ty - fs * 0.3)]
        info.append((name, (tx, ty)))
        for li, (el, a_, b_) in enumerate(lines):
            for near, far in ((a_, b_), (b_, a_)):
                d = min(math.dist(near, e) for e in ends)
                if d < 0.07 * big:
                    cand.append((d, ti, li, far))
    cand.sort(key=lambda c: c[0])
    got_t, got_l, anchor = set(), set(), {}
    for d, ti, li, far in cand:
        if ti in got_t or li in got_l:
            continue
        got_t.add(ti)
        got_l.add(li)
        anchor[ti] = (li, far)
    STATS.update(texts=len([i for i in info if i]), lines=len(lines), matched=len(anchor))
    parts, drop = [], []
    for ti, t in enumerate(texts):
        if info[ti] is None:
            continue
        name, txy = info[ti]
        if ti in anchor:
            li, far = anchor[ti]
            drop.append(lines[li][0])
            parts.append({"name": name, "text_xy": txy, "anchor_xy": far})
        else:
            parts.append({"name": name, "text_xy": txy, "anchor_xy": None})   # label with no leader: review by hand
        drop.append(t)
    return parts, drop


def _finish(tid: str, aliases: list[str], cand: dict, im, rec_parts: list[dict], extra_flags: list[str], out: Path) -> dict:
    """Common tail for SVG and raster pictures: checks, the .png/.json record, the numbered review sheet."""
    from PIL import ImageDraw
    import numpy as np
    W, H = im.size
    rec = {"id": tid, "title": aliases[0], "aliases": aliases, "size": [W, H], "parts": rec_parts, "kind": cand.get("kind", "svg"),
           "source": {"title": cand["title"], "page_url": cand["page_url"], "author": cand["author"], "license": cand["license"]},
           "credit": f"Diagram: {re.sub(r'^File:', '', cand['title'])} by {cand['author']} ({cand['license']}), Wikimedia Commons; labels redrawn"}
    arr = np.asarray(im)
    flags, blank = list(extra_flags), []
    for p_ in rec_parts:
        if p_["anchor"]:
            x, y = int(p_["anchor"][0] * W), int(p_["anchor"][1] * H)
            win = arr[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
            if win.size and win.min(-1).max() >= 245:
                blank.append(p_["name"])
    names = [p_["name"].lower() for p_ in rec_parts]
    nan = [p_["name"] for p_ in rec_parts if not p_["anchor"]]
    good = len(rec_parts) - len(nan)
    if good < 5:
        flags.append(f"only {good} labels with a leader line")
    if nan:
        flags.append("no leader line: " + ", ".join(nan[:5]))
    if blank:
        flags.append("anchor on blank background: " + ", ".join(blank[:5]))
    if len(set(names)) < len(names):
        flags.append("duplicate label names")
    ok_en, why = english_labels([p_["name"] for p_ in rec_parts])
    if not ok_en:
        flags.append("NOT ENGLISH? " + why)
    rec["flags"] = flags
    im.save(out / f"{tid}.png")
    (out / f"{tid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    sheet = im.copy()
    dr = ImageDraw.Draw(sheet)
    for i, p in enumerate(rec_parts, 1):
        if p["anchor"]:
            x, y = p["anchor"][0] * W, p["anchor"][1] * H
            dr.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(220, 0, 0), width=3)
            dr.text((x + 12, y - 8), f"{i}", fill=(220, 0, 0))
    (out / "review").mkdir(exist_ok=True)
    sheet.save(out / "review" / f"{tid}_review.png")
    with open(out / "review" / f"{tid}_parts.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(rec_parts, 1):
            f.write(f"{i}. {p['name']}  {'ANCHOR' if p['anchor'] else 'NO LEADER LINE - fix by hand'}\n")
    return rec


def render_svg(root) -> bytes:
    """SVG -> PNG.  Some Commons files point at arrowhead <marker>s that do not exist, which crashes cairosvg (AttributeError in draw_markers).
    Try as is; if that fails, drop every marker reference (arrowheads on the leader lines are being removed anyway) and render again."""
    import cairosvg
    try:
        return cairosvg.svg2png(bytestring=etree.tostring(root), output_width=RENDER_W, background_color="white")
    except Exception:                                                           # noqa: BLE001
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            for k in ("marker-start", "marker-mid", "marker-end", "marker"):
                el.attrib.pop(k, None)
            st = el.get("style")
            if st and "marker" in st:
                el.set("style", ";".join(x for x in st.split(";") if not x.strip().startswith("marker")))
        return cairosvg.svg2png(bytestring=etree.tostring(root), output_width=RENDER_W, background_color="white")


def build_one(tid: str, aliases: list[str], cand: dict, svg_bytes: bytes, out: Path) -> dict | None:
    import cairosvg
    from PIL import Image
    root = etree.fromstring(svg_bytes, etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False))
    parts, drop = extract_parts(root)
    good = [p for p in parts if p["anchor_xy"]]
    if len(good) < 3:
        return None
    vx, vy, vw, vh = viewbox(root)
    for el in drop:
        if el.getparent() is not None:
            el.getparent().remove(el)
    png = render_svg(root)
    im = Image.open(io.BytesIO(png)).convert("RGB")
    rec_parts = []
    for p in parts:
        ax, ay = p["anchor_xy"] if p["anchor_xy"] else (None, None)
        tx, ty = p["text_xy"]
        rec_parts.append({
            "name": p["name"],
            "anchor": [round((ax - vx) / vw, 4), round((ay - vy) / vh, 4)] if ax is not None else None,
            "side": "left" if (tx - vx) / vw < 0.5 else "right"})
    return _finish(tid, aliases, dict(cand, kind="svg"), im, rec_parts, [], out)


def build_one_raster(tid: str, aliases: list[str], cand: dict, data: bytes, out: Path) -> dict | None:
    """PNG/JPG route: OCR + pointer detection + inpainting (bank_raster.py).  Always flagged for careful review."""
    from PIL import Image
    import bank_raster as R
    im0 = Image.open(io.BytesIO(data)).convert("RGB")
    clean, parts, info = R.extract(im0)
    good = [p for p in parts if p["anchor_xy"]]
    if len(good) < 3:
        return None
    W, H = clean.size
    rec_parts = [{"name": p["name"], "anchor": [round(p["anchor_xy"][0] / W, 4), round(p["anchor_xy"][1] / H, 4)] if p["anchor_xy"] else None,
                  "side": "left" if p["text_xy"][0] / W < 0.5 else "right"} for p in parts]
    return _finish(tid, aliases, dict(cand, kind="raster"), clean, rec_parts,
                   ["RASTER picture (OCR + line detection): check every label position and spelling"], out)


def score_candidate(cand: dict, data: bytes) -> tuple[int, str]:
    """(number of anchored English labels, note) without writing anything."""
    if cand["kind"] == "svg":
        root = etree.fromstring(data, etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False))
        parts, _ = extract_parts(root)
        ok, why = english_labels([p["name"] for p in parts])
        n = sum(1 for p in parts if p["anchor_xy"])
        return (n if ok else 0), (f"{STATS.get('texts')} labels ({STATS.get('removed_other_languages', 0)} other-language items dropped), "
                                  f"{STATS.get('lines')} leader candidates, {n} anchored" + ("" if ok else f" - rejected: {why}"))
    from PIL import Image
    import bank_raster as R
    clean, parts, info = R.extract(Image.open(io.BytesIO(data)).convert("RGB"))
    ok, why = english_labels([p["name"] for p in parts])
    return (info["anchored"] if ok else 0), f"PNG/JPG: {info['labels']} words read, {info['anchored']} with a pointer" + ("" if ok else f" - rejected: {why}")


def build_topic(tid: str, aliases: list[str], query: str, out: Path, tries: int = 12, svg_keep: int = 3, raster_keep: int = 3, raster: bool = True) -> dict | None:
    """Score candidates from BOTH pools independently and save up to svg_keep SVGs and up to raster_keep PNG/JPGs, so you always get a choice
    from each route rather than one ranking swallowing the other.  Best SVG saved as <id>, its runners-up as <id>__alt2, __alt3, ...;
    best raster as <id>__raster1, __raster2, ...  Fewer than the target is fine (saved as found); none at all for a pool is fine too."""
    cands = gather_candidates(tid, aliases, query)
    r_ok = False
    if raster:
        try:
            import bank_raster
            r_ok, why = bank_raster.available()
        except Exception as e:                                                  # noqa: BLE001
            r_ok, why = False, f"bank_raster.py import failed: {type(e).__name__}: {e}  (is bank_raster.py uploaded next to this file?)"
        if not r_ok:
            print(f"   (PNG/JPG route off: {why})")

    def score_pool(batch: list[dict], keep: int) -> list[tuple[int, dict, bytes]]:
        """Downloads and scores until `keep` good (>=8 anchored) candidates are found or the batch runs out.  Returns EVERY candidate that
        scored >=3, not just the top `keep`: save() below needs that margin, because a candidate that scores well can still fail to render
        (a broken SVG, an OCR crash), and it must fall through to the next-best one of the SAME kind rather than leaving the slot empty."""
        scored: list[tuple[int, dict, bytes]] = []
        for cand in batch:
            try:
                content = download(cand["url"])
                if not content:
                    print(f"   skip {cand['title'][:58]}: could not download")
                    continue
                n, note = score_candidate(cand, content)
                print(f"   [{cand['kind']}] {cand['title'][:58]}: {note}")
            except RateLimited:
                raise
            except Exception as e:                                              # noqa: BLE001
                print(f"   skip {cand['title'][:58]}: {type(e).__name__}")
                continue
            if n >= 3:
                scored.append((n, cand, content))
            if len([x for x in scored if x[0] >= 8]) >= keep:                    # enough good ones: stop downloading more of this kind
                break
        scored.sort(key=lambda x: -x[0])
        return scored

    svg_scored = score_pool([c for c in cands if c["kind"] == "svg"][:tries * 2], svg_keep)
    raster_scored = score_pool([c for c in cands if c["kind"] == "raster"][:tries], raster_keep) if r_ok else []

    def save(scored: list, target: int, prefix_first: str, prefix_rest: str) -> dict | None:
        """Builds candidates in score order until `target` are successfully saved (or the pool runs out) - a build failure costs a candidate,
        not a slot, so a weak field of pictures still yields as many saved ones as actually exist."""
        best, made = None, 0
        for n, cand, data in scored:
            if made >= target:
                break
            rid = prefix_first if made == 0 else f"{prefix_rest}{made + 1}"
            try:
                rec = (build_one if cand["kind"] == "svg" else build_one_raster)(rid, aliases, cand, data, out)
            except Exception as e:                                              # noqa: BLE001
                print(f"   could not build {cand['title'][:50]}: {type(e).__name__}: {str(e)[:80]} - trying the next one")
                continue
            if not rec:
                continue
            (out / "source").mkdir(exist_ok=True)                                # the original file, kept for credit and re-processing
            (out / "source" / f"{rid}{'.svg' if cand['kind'] == 'svg' else Path(cand['url'].split('?')[0]).suffix or '.jpg'}").write_bytes(data)
            if made == 0:
                best = rec
            made += 1
        return best

    best_svg = save(svg_scored, svg_keep, tid, f"{tid}__alt")
    best_raster = save(raster_scored, raster_keep, f"{tid}__raster1", f"{tid}__raster")
    return best_svg or best_raster


def write_index(out: Path):
    idx = {}
    for jf in sorted(out.glob("*.json")):
        if jf.name != "index.json":
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
                idx[d["id"]] = d["aliases"]
            except Exception:                                                   # noqa: BLE001
                pass
    (out / "index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--topics", default="", help="comma list of ids or aliases; empty = all")
    ap.add_argument("--svg", help="build one topic from a local SVG instead of Commons (needs --id)")
    ap.add_argument("--id")
    ap.add_argument("--force", action="store_true", help="rebuild topics you rejected earlier")
    ap.add_argument("--svg-keep", type=int, default=3, help="how many SVG pictures to save per topic for you to choose from (default 3)")
    ap.add_argument("--raster-keep", type=int, default=3, help="how many PNG/JPG pictures to save per topic for you to choose from (default 3)")
    ap.add_argument("--no-raster", action="store_true", help="SVG only (skip the PNG/JPG route)")
    ap.add_argument("--offline", action="store_true", help="never use the network: build only from <out>/_cache (filled by bank_fetch.py on your PC)")
    ap.add_argument("--png", help="build one topic from a local PNG/JPG (needs --id)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    CACHE[0] = str(out / "_cache")
    OFFLINE[0] = a.offline
    have = sorted(f.stem for f in out.glob("*.json") if f.name != "index.json" and "__alt" not in f.stem)
    print(f"Bank folder: {out.resolve()}\n  already built ({len(have)}): {', '.join(have) if have else 'none - this folder is empty, so everything will be built'}"
          + ("\n  --force is ON: these will be rebuilt" if a.force else ""))
    if a.svg:
        tid = a.id or "custom"
        rec = build_one(tid, TOPICS.get(tid, ([tid.replace("_", " ")], ""))[0],
                        {"title": Path(a.svg).name, "page_url": "", "author": "local file", "license": "check licence"},
                        Path(a.svg).read_bytes(), out)
        print("ok" if rec else "no usable labelled leader lines found")
        return
    if a.png:
        tid = a.id or "custom"
        rec = build_one_raster(tid, TOPICS.get(tid, ([tid.replace("_", " ")], ""))[0],
                               {"title": Path(a.png).name, "page_url": "", "author": "local file", "license": "check licence"}, Path(a.png).read_bytes(), out)
        print("ok" if rec else "could not find enough labels with pointer lines")
        return
    want = [w.strip().lower() for w in a.topics.split(",") if w.strip()]
    for tid, (aliases, q) in TOPICS.items():
        if want and not any(w == tid or w in aliases for w in want):
            continue
        rej = out / "rejected.txt"
        if rej.exists() and tid in rej.read_text(encoding="utf-8").split() and not a.force:
            print("skip (rejected earlier; use --force):", tid)
            continue
        if (out / f"{tid}.json").exists() and not a.force:
            print("already built, skipping (use --force to rebuild):", tid)
            continue
        print("building", tid)
        try:
            rec = build_topic(tid, aliases, q, out, svg_keep=a.svg_keep, raster_keep=a.raster_keep, raster=not a.no_raster)
        except RateLimited as e:
            write_index(out)
            print(f"\nSTOPPED: {e}. Everything built so far is saved. Run the same command again in a few minutes; finished topics are skipped.")
            return
        except Exception as e:                                                  # noqa: BLE001  (one bad topic never stops the run)
            print(f"  -> FAILED: {type(e).__name__}: {str(e)[:100]} - go on with the next topic")
            continue
        print("  ->", f"{len(rec['parts'])} labels, {sum(1 for p in rec['parts'] if p['anchor'])} anchored ({rec.get('kind')})" if rec else "NOTHING USABLE (try another query or add by hand)")
        write_index(out)                                                         # saved after every topic: a crash or a closed Colab loses nothing
    write_index(out)
    (out / "approved.txt").touch()
    print(f"\nNext: on the PC run   streamlit run bank_review.py -- --bank <the bank folder>   to look, fix names and approve.")


if __name__ == "__main__":
    sys.exit(main())
