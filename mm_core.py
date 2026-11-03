"""
mm_core.py - pure, deterministic logic for "Understood" (no network, no GPU, no models).

Everything here is testable on its own.  The engine (LangGraph) calls these helpers; the tutor model is only
ever asked narrow questions whose answers are VERIFIED by code from this file.

Contents
    1. Study levels (Ordinary / Advanced / University) and what each changes
    2. Small text helpers
    3. Routing rules (fast, free) + validation of the tutor's routing answer (agentic part)
    4. Parts: parsing + literal verification against a source text, and short image-prompt assembly
    5. Regions: split a clean diagram into regions, numbered marks for the tutor, label/arrow drawing
    6. Sections formatting for tutor answers, narration helpers, lesson timeline
    7. Document passage selection (lightweight retrieval for uploaded files)
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

# =====================================================================================================
# 1. Study levels
# =====================================================================================================
LEVELS: dict[str, dict] = {
    "ordinary": dict(
        key="ordinary", label="Ordinary level (S1-S4)", max_parts=6,
        style="simple textbook diagram, bold outlines, flat colours, large clearly separated shapes",
        vocab="simple everyday words a 14 year old can follow",
        explain="Use simple words and short sentences. Say what each part does in one sentence.",
        narration_words=22, mol_style="displayed", gen_size=768,
    ),
    "advanced": dict(
        key="advanced", label="Advanced level (S5-S6)", max_parts=9,
        style="detailed textbook cross-section diagram, thin dark outlines, flat colours, parts clearly separated",
        vocab="the correct scientific terms used in A-level syllabuses",
        explain="Use correct scientific terms. Explain the function of each part and how the parts work together.",
        narration_words=32, mol_style="skeletal_groups", gen_size=768,
    ),
    "university": dict(
        key="university", label="University", max_parts=13,
        style="detailed scientific illustration, fine dark outlines, flat colours, accurate relative proportions, parts clearly separated",
        vocab="precise university-level terminology",
        explain="Use precise technical terms. Explain mechanisms and how the parts interact, and mention one common misconception.",
        narration_words=42, mol_style="skeletal_full", gen_size=1024,
    ),
}
DEFAULT_LEVEL = "university"      # learning levels are retired for now: one standard, university-usable depth
LEVEL_LABELS = {k: v["label"] for k, v in LEVELS.items()}


def get_level(key: Optional[str] = None) -> dict:
    """Levels are retired for now: every request gets the same standard (university-usable, 13 parts).  The key is ignored."""
    return LEVELS[DEFAULT_LEVEL]


_RE_LEVEL = [
    ("university", re.compile(r"\b(university|undergrad\w*|degree level|college level)\b", re.I)),
    ("advanced", re.compile(r"\b(a[- ]?level|advanced level|s\.?[56]\b|senior (five|six|5|6))", re.I)),
    ("ordinary", re.compile(r"\b(o[- ]?level|ordinary level|s\.?[1-4]\b|senior (one|two|three|four|[1-4])\b|lower secondary)", re.I)),
]


def level_from_text(text: str) -> Optional[str]:
    return None                                                      # levels retired: wording like "o-level" no longer changes anything
    for key, rx in _RE_LEVEL:
        if rx.search(text or ""):
            return key
    return None


# Words that must never reach the learner-facing interface (the backend is not revealed).
BANNED_UI_WORDS = ["qwen", "colab", "gpu", "lora", "tunnel", "cloudflare", "huggingface", "hugging face", "diffusion",
                   "diffusers", "bitsandbytes", "nf4", "quantiz", "controlnet", "langgraph", "checkpoint", "t4 "]


def scrub_ui_text(text: str) -> str:
    """Last line of defence: never show backend vocabulary to the learner."""
    out = text or ""
    for w in BANNED_UI_WORDS:
        out = re.sub(re.escape(w), "the learning service", out, flags=re.I)
    return out


# =====================================================================================================
# 2. Text helpers
# =====================================================================================================
STOP = set("a an the of and or to in on for with from by at as is are was were be been it its this that these those "
           "me my i you your we our can could would should please show give make draw create generate produce want need "
           "about into onto some any more less very just also then than so if what which who how why when where".split())


def est_tokens(text: str) -> int:
    return max(1, int(len(text or "") / 3.6))


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", (text or "").lower()) if w not in STOP and len(w) > 1]


def words_trim(text: str, max_words: int) -> str:
    w = (text or "").split()
    if len(w) <= max_words:
        return " ".join(w)
    cut = " ".join(w[:max_words])
    m = re.search(r"^(.*[.!?])\s", cut + " ", re.S)
    return m.group(1) if m and len(m.group(1).split()) >= max(6, max_words // 2) else cut.rstrip(",;: ") + "."


def trim_to_sentence(text: str) -> str:
    m = re.search(r"^(.*[.!?])\s", (text or "") + " ", re.S)
    return m.group(1) if m and len(m.group(1)) > 20 else (text or "").rstrip() + " ..."


def safe_name(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_")
    return (s or "item")[:maxlen]


def parse_json(text: str):
    """Pull the first JSON object or array out of a model answer (handles ```json fences and chatter)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.S)
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = t.find(open_c), t.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except Exception:                                       # noqa: BLE001
                continue
    return None


# =====================================================================================================
# 3. Routing
# =====================================================================================================
TOOLS = ("learn_diagram", "learn_process", "chem_molecule", "chem_reaction", "studio_image", "edit_image", "explain", "chat",
         "notes", "guide", "lesson", "pdf", "slides", "relabel", "clarify")
TUTOR_MODE_TOOLS = ("chat", "explain", "notes", "guide", "pdf", "lesson", "slides")
NEEDS_FIGURE = ("relabel",)


@dataclass
class RouteCtx:
    mode: str = "assistant"            # assistant | tutor | studio | edit
    has_figure: bool = False           # a labelled figure from this conversation exists
    has_explanation: bool = False
    image_attachments: int = 0
    doc_attachments: int = 0
    has_current_image: bool = False    # any image to edit (studio result, edit result, figure)
    part_names: list = field(default_factory=list)   # labels on the current figure
    known_molecule: Optional[object] = None          # callable(name) -> bool, supplied by the chemistry module
    has_doc: bool = False                            # a guide / notes document from this conversation exists
    has_history: bool = False                        # there is a conversation to turn into notes


_VERB_DRAW = r"(draw|generate|create|make|show|produce|sketch|illustrate|design|render|give me|get me|i need|i want)"
RE_DRAW_START = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could|would|will) you\s+)?(?:please\s+)?(?:i (?:want|need|would like)\s+(?:you to\s+|a\s+|an\s+)?)?"
                           r"(draw|generate|create|make|show|produce|sketch|illustrate|design|render|give me|get me)\b", re.I)
RE_NOTES = re.compile(r"\b(discussion|conversation|chat|our talk|what we (?:discussed|covered|did|learn(?:ed|t))|transcript|this session|so far)\b", re.I)
RE_FULL = re.compile(r"\b(full|whole|complete|word for word|verbatim|transcript)\b", re.I)
RE_GUIDE = re.compile(r"\b(study guide|learning guide|revision (?:guide|notes)|study notes|lesson notes|learning notes|notes (?:on|about|for)|make notes)\b", re.I)
RE_PROCESS_STRONG = re.compile(r"\binfographic\b|\bstep[- ]by[- ]step\b", re.I)
RE_PROCESS_WEAK = re.compile(r"\b(cycle|process|stages?|steps?|flows?|journey|life ?cycle|how .{2,40}(?:works?|happens?|occurs?|forms?|is made|flows?|pumps?|travels?|moves?|passes|handles|responds?|transmits?))\b", re.I)
RE_SLIDES = re.compile(r"\b(slides?|slide ?deck|power ?point|pptx|presentation)\b", re.I)
RE_PDF = re.compile(r"\bpdf\b|\bprintable\b|\bhandout\b", re.I)
RE_LESSON = re.compile(
    r"\b(play|start|run|make|give me|create)\b.{0,25}\b(lesson|narration|audio|video|animation|animated)\b|"
    r"\b(and|then|plus|with)\s+(a\s+|an\s+|the\s+)?(lesson|narration|audio|video|animation)\b|\bvideo (of|on|about)\b|"
    r"\b(narrate|read (it|this|that) (out|aloud)|audio (description|explanation|version)|talk me through)\b", re.I)
RE_RENAME = re.compile(r"^\s*(?:please\s+)?(?:rename|change)\s+(?:the\s+)?(?:label\s+)?['\"]?(.+?)['\"]?\s+(?:label\s+)?to\s+['\"]?(.+?)['\"]?\s*\.?$", re.I)
RE_REMOVE_LABEL = re.compile(r"^\s*(?:please\s+)?(?:remove|delete|hide)\s+(?:the\s+)?(?:label\s+)?['\"]?(.+?)['\"]?(?:\s+label)?\s*\.?$", re.I)
RE_DIAGRAM = re.compile(r"\b(diagram|labell?ed|schematic|cross[- ]?section|apparatus|set[- ]?up|parts of|anatomy of|"
                        r"flow ?chart of)\b", re.I)
RE_STUDIO = re.compile(r"\b(photo|photograph|realistic|poster|logo|cartoon|wallpaper|portrait|landscape|painting|"
                       r"picture of|image of|3d|render of)\b", re.I)
RE_EDIT = re.compile(r"\b(edit|change|make it|make the|remove|erase|replace|add|turn (it|this)|colou?r|brighter|darker|"
                     r"background|crop|blur|sharpen|restore|enhance|fix|swap|style)\b", re.I)
RE_QUESTION = re.compile(r"^\s*(what|why|how|when|where|who|which|explain|describe|define|tell me|difference|compare|"
                         r"is|are|does|do|can|could|should|summari[sz]e|identify|read)\b|\?\s*$", re.I)
RE_GREET = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|ok(ay)?)\b[\s!.]*$", re.I)
RE_REFERS = re.compile(r"\b(this|that|the|my)\s+(\w+\s+)?(diagram|image|picture|photo|figure|drawing|file|document|pdf|page|notes?|text|reaction|molecule)\b|\b(it|above)\b", re.I)
RE_MOLWORDS = re.compile(r"\b(molecule|molecular|structural formula|displayed formula|skeletal|isomers?|functional groups?)\b", re.I)
RE_RXNWORDS = re.compile(r"\b(reaction|equation|combustion|neutrali[sz]ation|balance|balanced|redox|esterification|displacement)\b|->|→", re.I)
RE_PLAIN_REF = re.compile(r"^[\s\W]*(what|explain|describe)\b", re.I)

ELEMENTS = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh "
               "Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra "
               "Ac Th Pa U Np Pu Am Cm".split())
COMMON_FORMULAS = {"HCl", "NaCl", "NaOH", "KOH", "KCl", "CaO", "MgO", "CuO", "ZnO", "HF", "HBr", "HI", "CO", "NO", "NaF", "KBr", "KI", "AgCl",
                   "CaCl2", "MgCl2", "BaSO4", "NaBr", "FeS", "PbO", "SO2", "CuS"}
_RE_FORM_TOKEN = re.compile(r"(?<![A-Za-z0-9])((?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*){2,})(?![A-Za-z0-9])")


def formula_tokens(text: str) -> list[str]:
    """Chemical formulas found in a text (validated against real element symbols, so 'ATP' or 'DNA' do not count)."""
    out = []
    for m in _RE_FORM_TOKEN.finditer(text or ""):
        tok = m.group(1)
        syms = re.findall(r"[A-Z][a-z]?", tok)
        if syms and all(x in ELEMENTS for x in syms) and (re.search(r"\d", tok) or tok in COMMON_FORMULAS):
            out.append(tok)
    return out


def _step(tool: str, **args) -> dict:
    return {"tool": tool, "args": {k: v for k, v in args.items() if v not in (None, "")}}


_TOPIC_STOP = set("pdf video lesson handout audio narration and a an the me give play make please of it this that our chat discussion with".split())


def extract_topic(text: str) -> str:
    """The subject of a drawing request, with command words stripped."""
    t = (text or "").strip().rstrip("?.! ")
    t = re.sub(r"(\s*[,;]?\s+(and|then|also|plus)\s+|\s*,\s*)(give|make|export|play|create|download|save|send|get|add|include|produce)\b.*$", "", t, flags=re.I)
    t = re.sub(r"\s+(with|in|as)\s+(a\s+|the\s+)?(pdf|audio|narration|lesson)\b.*$", "", t, flags=re.I)
    t = re.sub(r"^(please\s+)?(can|could|would) you\s+", "", t, flags=re.I)
    t = re.sub(r"^i (want|need|would like)\s+(you to\s+)?", "", t, flags=re.I)
    t = re.sub(rf"^(please\s+)?{_VERB_DRAW}\s+(me\s+)?", "", t, flags=re.I)
    t = re.sub(r"^(a|an|the|some)\s+", "", t, flags=re.I)
    t = re.sub(r"^(pdf|handout)(\s+(file|document))?\s+(of|about|on|for)\s+", "", t, flags=re.I)
    t = re.sub(r"^(a|an|the|some)\s+", "", t, flags=re.I)
    t = re.sub(r"\b(labell?ed|detailed|simple|clear|clean|scientific|biology|chemistry)\s+", "", t, flags=re.I)
    t = re.sub(r"^(diagram|picture|image|drawing|figure|illustration|schematic|sketch)\s+(of|showing)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+(diagram|drawing|figure|illustration|schematic)$", "", t, flags=re.I)
    t = re.sub(r"^(the )?(structure|structural formula|displayed formula|skeletal formula|molecular formula|formula|molecule|parts|anatomy|process|life cycle|cycle) of\s+", "", t, flags=re.I)
    t = re.sub(r"^(a|an|the)\s+", "", t, flags=re.I)
    t = re.sub(r"^(a |an |the )?(study guide|learning guide|revision guide|revision notes|study notes|lesson notes|learning notes|notes|infographic|step[- ]by[- ]step( infographic)?)\s+(of|on|about|for|to)\s+", "", t, flags=re.I)
    t = re.sub(r"^(a |an |the )?(pdf|handout|video|lesson|study guide|infographic)$", "", t, flags=re.I)
    t = re.sub(r"^(our|this|the|my)\s+(discussion|conversation|chat)$", "", t, flags=re.I)
    t = re.sub(r"\s+(and|then|plus)\s+(a\s+|the\s+)?(pdf|video|lesson|handout)\b.*$", "", t, flags=re.I)
    t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.I)
    t = re.sub(r"^how\s+(?:a|an|the)?\s*(.+?)\s+(?:works?|happens?|occurs?|forms?)$", r"\1", t, flags=re.I) if re.match(r"^how\s+", t, re.I) else t
    t = re.sub(r"^step[- ]by[- ]step\s+", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" ,.-")[:80]
    if t and all(w in _TOPIC_STOP for w in re.findall(r"[a-z]+", t.lower())):
        return ""
    return t


def doc_commands(t: str, ctx: RouteCtx) -> Optional[list[dict]]:
    """Notes / guide / bare pdf-or-video commands.  They never need a generated image."""
    low = t.lower()
    want_pdf, want_lesson = bool(RE_PDF.search(t)), bool(RE_LESSON.search(t))
    tail = [_step("lesson")] if want_lesson else []
    topic = extract_topic(t)
    if RE_SLIDES.search(low) and (ctx.has_figure or ctx.has_doc or ctx.has_history) and len(low.split()) <= 14:
        return [_step("slides")]
    if RE_GUIDE.search(low):
        return [_step("guide", topic=topic)] + tail
    if RE_NOTES.search(low) and (want_pdf or want_lesson or re.search(r"\b(notes|summar\w+|transcript)\b", low)):
        return [_step("notes", mode="full" if RE_FULL.search(low) else "condensed")] + tail
    return None


def rule_route(text: str, ctx: RouteCtx) -> Optional[list[dict]]:
    """Cheap, deterministic routing.  Returns a list of steps, or None when the request is genuinely ambiguous
    (the engine then asks the tutor to decide - that is the agentic part)."""
    t = (text or "").strip()
    low = t.lower()

    # ---- modes that bypass the tutor entirely -------------------------------------------------
    if ctx.mode == "studio":
        return [_step("studio_image", prompt=t)] if t else []
    if ctx.mode == "edit":
        return [_step("edit_image", instruction=t)] if t else []
    if ctx.mode == "tutor":
        cmd = doc_commands(t, ctx)
        if cmd:
            return cmd
        if (RE_PDF.search(t) or RE_LESSON.search(t)) and not ctx.image_attachments and len(t.split()) <= 8:
            return ([_step("lesson")] if RE_LESSON.search(t) else []) + ([_step("pdf")] if RE_PDF.search(t) else [])
        if ctx.image_attachments or ctx.doc_attachments or (ctx.has_figure and RE_REFERS.search(t)):
            return [_step("explain", question=t)]
        return [_step("chat")]

    # ---- empty text with an attachment: explain it -------------------------------------------
    if not t:
        return [_step("explain", question="")] if (ctx.image_attachments or ctx.doc_attachments) else []

    want_pdf = bool(RE_PDF.search(t))
    want_lesson = bool(RE_LESSON.search(t))
    draw_cmd = bool(RE_DRAW_START.search(t))
    forms = formula_tokens(t)
    topic = extract_topic(t)
    steps: list[dict] = []

    # ---- direct commands on the current figure -------------------------------------------------
    if ctx.has_figure:
        known = {_norm(p) for p in ctx.part_names}
        m = RE_RENAME.match(t)
        if m and (_norm(m.group(1)) in known or "label" in low):
            return [_step("relabel", op="rename", old=m.group(1).strip(), new=m.group(2).strip())]
        m = RE_REMOVE_LABEL.match(t)
        if m and len(m.group(1).split()) <= 4 and (_norm(m.group(1)) in known or "label" in low):
            return [_step("relabel", op="remove", old=m.group(1).strip())]
    if RE_GREET.match(t):
        return [_step("chat")]

    # ---- notes / guides / bare pdf or video: never need a generated image ---------------------------------
    cmd = doc_commands(t, ctx)
    if cmd:
        return cmd
    if (want_pdf or want_lesson) and not RE_DIAGRAM.search(low) and not forms and not RE_STUDIO.search(low) \
            and not RE_PROCESS_STRONG.search(low):
        refers = bool(re.search(r"\b(this|that|it|our|above|these|here)\b", low))
        draw_verb = re.match(r"^\s*(?:please\s+)?(?:(?:can|could|would|will) you\s+)?(?:please\s+)?(draw|sketch|illustrate|render|design|generate|create|show|make)\b", low)
        if (re.search(r"\b(and|then)\s+(give|make|export|create)\b", low) and topic) or (draw_verb and topic):
            pass                                                     # a drawing plus an add-on: handled below
        elif topic and not refers and not ctx.image_attachments and not ctx.doc_attachments:
            return [_step("guide", topic=topic)] + ([_step("lesson")] if want_lesson else [])
        else:
            return ([_step("lesson")] if want_lesson else []) + ([_step("pdf")] if want_pdf else [])

    # ---- chemistry: reactions, then molecules -------------------------------------------------------
    if RE_RXNWORDS.search(low) and (forms or re.search(r"\b(of|between|for)\b", low) or "->" in t or "→" in t) \
            and (draw_cmd or re.search(r"\b(equation|balance|balanced|reaction)\b|->|→", low)) and not RE_DIAGRAM.search(low):
        steps.append(_step("chem_reaction", text=t))
    elif not RE_DIAGRAM.search(low) and (
            RE_MOLWORDS.search(low) or (forms and (draw_cmd or "structure" in low or "formula" in low))
            or (ctx.known_molecule and topic and ctx.known_molecule(topic) and (draw_cmd or "structure" in low))):
        steps.append(_step("chem_molecule", query=topic or t))

    # ---- images that already exist: edit or explain ------------------------------------------------
    if not steps and (ctx.image_attachments or ctx.has_current_image):
        asks = bool(RE_QUESTION.search(t)) or bool(RE_PLAIN_REF.match(t))
        editing = bool(RE_EDIT.search(low)) and not (asks and not re.search(r"\b(edit|change|make it|remove|replace|add)\b", low))
        if ctx.image_attachments and editing and not RE_DIAGRAM.search(low):
            steps.append(_step("edit_image", instruction=t))
        elif ctx.image_attachments and asks:
            steps.append(_step("explain", question=t))
        elif ctx.has_current_image and not ctx.image_attachments and editing and RE_REFERS.search(low) and not (want_lesson or want_pdf):
            steps.append(_step("edit_image", instruction=t))
    if not steps and ctx.doc_attachments and (RE_QUESTION.search(t) or RE_REFERS.search(low)):
        steps.append(_step("explain", question=t))

    # ---- new diagrams / images ------------------------------------------------------------------------
    if not steps:
        asks = bool(RE_QUESTION.search(t))
        proc = bool(RE_PROCESS_STRONG.search(low)) or (bool(RE_PROCESS_WEAK.search(low)) and (
            draw_cmd or (not asks and bool(re.search(r"\b(life ?cycle|cycle of|process of|stages of|steps of)\b", low)))))
        if proc and (not RE_DIAGRAM.search(low) or RE_PROCESS_STRONG.search(low)) and topic:
            pm = re.search(r"\b(life ?cycle of (?:a |an |the )?[\w\-]+(?: [\w\-]+)?)", low)
            steps.append(_step("learn_process", topic=pm.group(1) if pm else topic))
        elif RE_DIAGRAM.search(low) and (draw_cmd or not asks):
            steps.append(_step("learn_diagram", topic=topic))
        elif RE_STUDIO.search(low) and (draw_cmd or not asks):
            steps.append(_step("studio_image", prompt=t))
        elif draw_cmd:
            return None                                              # "draw a heart": diagram or picture? let the tutor decide

    # ---- add-ons (explain / lesson / pdf) ---------------------------------------------------------------
    if steps and steps[0]["tool"] in ("learn_diagram", "chem_molecule", "chem_reaction") and not any(x["tool"] == "explain" for x in steps):
        steps.append(_step("explain", question=""))
    if steps:
        if want_lesson:
            steps.append(_step("lesson"))
        if want_pdf:
            steps.append(_step("pdf"))
        return steps

    # ---- plain follow-ups ------------------------------------------------------------------------------------
    if ctx.has_figure and RE_REFERS.search(low) and RE_QUESTION.search(t):
        return [_step("explain", question=t)]
    if RE_QUESTION.search(t):
        return [_step("chat")]
    return None


def validate_route(obj, ctx: RouteCtx, user_text: str) -> Optional[list[dict]]:
    """Turn the tutor's JSON routing answer into safe steps (whitelisted tools, sane arguments, max 4)."""
    if isinstance(obj, dict):
        items = obj.get("steps") if isinstance(obj.get("steps"), list) else [obj]
    elif isinstance(obj, list):
        items = obj
    else:
        return None
    allowed = set(TOOLS)
    if ctx.mode == "tutor":
        allowed = set(TUTOR_MODE_TOOLS)
    out: list[dict] = []
    for it in items[:4]:
        if not isinstance(it, dict):
            continue
        tool = str(it.get("tool", "")).strip().lower()
        if tool not in allowed:
            continue
        args = it.get("args") if isinstance(it.get("args"), dict) else {k: v for k, v in it.items() if k not in ("tool", "args")}
        args = {k: str(v).strip()[:300] for k, v in args.items() if isinstance(v, (str, int, float)) and str(v).strip()}
        if tool in ("learn_diagram", "learn_process", "clarify"):
            args.setdefault("topic", extract_topic(user_text))
            if not args["topic"]:
                continue
        elif tool == "guide":
            args.setdefault("topic", extract_topic(user_text))
        elif tool == "notes":
            args.setdefault("mode", "condensed")
        elif tool == "chem_molecule":
            args.setdefault("query", extract_topic(user_text) or user_text)
        elif tool == "chem_reaction":
            args.setdefault("text", user_text)
        elif tool == "studio_image":
            args.setdefault("prompt", user_text)
        elif tool == "edit_image":
            args.setdefault("instruction", user_text)
        elif tool == "relabel" and not ctx.has_figure:
            continue
        if out and out[-1]["tool"] == tool and tool in ("chat", "explain"):
            continue
        out.append({"tool": tool, "args": args})
    lvl = None
    if isinstance(obj, dict) and obj.get("level") in LEVELS:
        lvl = obj["level"]
    if out and out[0]["tool"] in ("learn_diagram", "chem_molecule", "chem_reaction") and not any(s["tool"] == "explain" for s in out):
        out.insert(1, {"tool": "explain", "args": {}})
    if out and lvl:
        out[0]["args"].setdefault("level", lvl)
    return out or None


# =====================================================================================================
# 4. Parts + prompt assembly
# =====================================================================================================
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _variants(name: str) -> set[str]:
    n = _norm(name)
    v = {n}
    if n.endswith("ies"):
        v.add(n[:-3] + "y")
    if n.endswith("s"):
        v.add(n[:-1])
    else:
        v.add(n + "s")
    if n.endswith("y"):
        v.add(n[:-1] + "ies")
    return {x for x in v if x}


def parse_parts_list(answer: str) -> list[str]:
    """Accept JSON list/object or a comma/line separated list from the tutor."""
    obj = parse_json(answer)
    raw: list = []
    if isinstance(obj, list):
        raw = obj
    elif isinstance(obj, dict):
        for k in ("parts", "structures", "components", "list"):
            if isinstance(obj.get(k), list):
                raw = obj[k]
                break
    if not raw:
        raw = re.split(r"[,;\n]|(?:^|\s)[-*\u2022]\s+|\s\d+[.)]\s", answer or "")
    out = []
    for x in raw:
        if isinstance(x, dict):
            x = x.get("name") or x.get("part") or ""
        s = re.sub(r"^[\s\-*\u2022\d.)]+", "", str(x)).strip().strip("\"'.:")
        if 2 <= len(s) <= 40 and len(s.split()) <= 4:
            out.append(s)
    return out


def verify_parts(candidates: list[str], source_text: str, max_parts: int, known: Optional[list[str]] = None) -> list[str]:
    """Keep only names that literally appear in the source text (or in a trusted list), dedupe, cap."""
    hay = " " + _norm(source_text) + " "
    trusted = {_norm(k) for k in (known or [])}
    out, seen = [], set()
    for c in candidates:
        key = _norm(c)
        if not key or key in seen:
            continue
        ok = key in trusted or any(f" {v} " in hay for v in _variants(c))
        if ok:
            seen.add(key)
            out.append(c.strip().lower() if c.islower() or c.isupper() else c.strip())
        if len(out) >= max_parts:
            break
    return out


BASE_CLAUSE = ("white background, clean boundaries between parts, each part a distinct flat colour, "
               "no text, no letters, no numbers, no labels, no captions, no arrows")


# Lighter look (UNDERSTOOD_PROMPT_STYLE=light): colour only where a part must stand apart, closed dark outlines carry the separation.
# Light is now the default (UNDERSTOOD_PROMPT_STYLE=classic brings the old bold colours back); ab_style.py compares the two.
LIGHT_CLAUSE = ("white background, pale soft tones with colour only where a part must stand apart, thin dark closed outline around every part, "
                "no gradients, no shadows, no text, no letters, no numbers, no labels, no captions, no arrows")


def build_diagram_prompt(topic: str, parts: list[str], level_key: str, style: str = "") -> str:
    """Short and unambiguous: subject + level style + the part list + clean-boundary clause.  Nothing invented."""
    lv = get_level(level_key)
    shown = ", ".join(parts)
    clause = LIGHT_CLAUSE if (style or os.environ.get("UNDERSTOOD_PROMPT_STYLE", "light")) == "light" else BASE_CLAUSE
    return f"{topic}, {lv['style']}, showing {shown}, {clause}"


def parts_in_prompt(prompt: str) -> list[str]:
    m = re.search(r"showing (.+?), white background", prompt or "")
    return [p.strip() for p in m.group(1).split(",")] if m else []


# =====================================================================================================
# 5. Regions, marks, labels
# =====================================================================================================
PALETTE = [(213, 94, 0), (0, 114, 178), (0, 128, 96), (170, 60, 130), (200, 120, 0), (30, 140, 200),
           (140, 90, 40), (0, 128, 128), (190, 40, 120), (110, 130, 20), (50, 60, 150), (120, 120, 120),
           (180, 40, 40), (80, 160, 60)]


def colour_for(i: int) -> tuple[int, int, int]:
    return PALETTE[i % len(PALETTE)]


_FONT_FILES = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/Library/Fonts/Arial.ttf"]
_FONT_BOLD = ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]


def get_font(size: int, bold: bool = False):
    for p in (_FONT_BOLD if bold else []) + _FONT_FILES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:                                       # noqa: BLE001
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def find_regions(img: Image.Image, max_regions: int = 14) -> tuple[list[dict], np.ndarray]:
    """Split a clean, flat-colour diagram into regions (deterministic).  Outlines and colour edges separate regions;
    the white background touching the border is ignored.  Each region gets an anchor point that is always INSIDE it
    (deepest point), so ring shapes such as a cell wall are handled.  Returns (regions, label_map)."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    H, W = a.shape[:2]
    sm = np.stack([ndi.gaussian_filter(a[..., c], 1.0) for c in range(3)], -1)
    dark = sm.max(-1) < 80                                          # near-black outlines only (a dark green part is not an outline)
    dx = np.abs(np.diff(sm, axis=1, append=sm[:, -1:])).max(-1)
    dy = np.abs(np.diff(sm, axis=0, append=sm[-1:])).max(-1)
    sep = ndi.binary_dilation(dark | (np.maximum(dx, dy) > 28), iterations=1)
    lab_w, _ = ndi.label(sm.min(-1) > 235)
    border = set(np.unique(np.concatenate([lab_w[0], lab_w[-1], lab_w[:, 0], lab_w[:, -1]]))) - {0}
    background = np.isin(lab_w, list(border))
    free = ndi.binary_opening(~sep & ~background, iterations=2)
    lab, n = ndi.label(free)
    found = []
    for k in range(1, n + 1):
        m = lab == k
        area = int(m.sum())
        if area < 0.004 * W * H or area > 0.6 * W * H:
            continue
        dt = ndi.distance_transform_edt(m)
        y, x = np.unravel_index(dt.argmax(), dt.shape)
        found.append(dict(k=k, area=area, anchor=(float(x), float(y))))
    found = sorted(sorted(found, key=lambda r: -r["area"])[:max_regions], key=lambda r: (round(r["anchor"][1] / 40), r["anchor"][0]))
    lmap = np.zeros((H, W), dtype=np.uint8)
    regions = []
    for i, r in enumerate(found, 1):
        lmap[lab == r["k"]] = i
        regions.append(dict(id=i, area=r["area"], anchor=r["anchor"]))
    return regions, lmap


def draw_marks(img: Image.Image, regions: list[dict]) -> Image.Image:
    """Temporary copy with a numbered circle on each region.  Only the tutor sees it, never the learner."""
    m = img.convert("RGB").copy()
    d = ImageDraw.Draw(m)
    r = max(12, m.size[0] // 40)
    f = get_font(int(r * 1.2), bold=True)
    for g in regions:
        x, y = g["anchor"]
        d.ellipse([x - r, y - r, x + r, y + r], fill="white", outline="black", width=2)
        try:
            d.text((x, y), str(g["id"]), font=f, fill="black", anchor="mm")
        except Exception:                                           # noqa: BLE001
            d.text((x - r / 2, y - r / 2), str(g["id"]), font=f, fill="black")
    return m


def accept_points(raw: list[tuple], W: int, H: int, min_sep: float = 0.02) -> tuple[list[tuple], list[str]]:
    ok, bad = [], []
    diag = (W * W + H * H) ** 0.5
    for name, pt in raw:
        if pt is None:
            bad.append(name)
        elif any(((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2) ** 0.5 < min_sep * diag for _, q in ok):
            bad.append(name)
        else:
            ok.append((name, pt))
    return ok, bad


def to_region_id(v, valid) -> Optional[int]:
    try:
        i = int(str(v).strip())
    except Exception:                                               # noqa: BLE001
        return None
    return i if i in valid else None


def parse_assignment(answer: str, parts: list[str], valid_ids) -> dict[str, int]:
    """Tutor answer {"nucleus": 3, "vacuole": null} -> {part: region id}.  Each region is used at most once."""
    obj = parse_json(answer)
    out, used = {}, set()
    if isinstance(obj, dict):
        lookup = {_norm(p): p for p in parts}
        for k, v in obj.items():
            p = lookup.get(_norm(str(k)))
            i = to_region_id(v, valid_ids) if v is not None else None
            if p and i and i not in used and p not in out:
                out[p] = i
                used.add(i)
    return out


# -----------------------------------------------------------------------------------------------------
# 5b. Verified naming.  The tutor looks at ONE outlined region at a time and names it in its own words (no numbers);
#     code matches that name to the verified part list; a probability check then confirms or rejects every pair.
#     A region nobody can confirm stays unlabelled - a missing label is better than a wrong one.
# -----------------------------------------------------------------------------------------------------
_IRREGULAR = {"alveoli": "alveolus", "bronchi": "bronchus", "nuclei": "nucleus", "stomata": "stoma", "cristae": "crista",
              "grana": "granum", "mitochondria": "mitochondrion", "bacteria": "bacterium", "vertebrae": "vertebra",
              "ova": "ovum", "hyphae": "hypha", "villi": "villus", "ventricles": "ventricle", "atria": "atrium"}
_UNSURE = {"unclear", "unknown", "none", "not sure", "unsure", "cannot tell", "can not tell", "na", "n a", "nothing",
           "background", "something", "a shape", "shape", "a round shape", "round shape", "structure", "part", "region"}


def _canon_basic(s: str) -> str:
    """Comparison form of a part name: lower case, no articles, British = American spelling, singular words."""
    t = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower().replace("-", " "))
    t = re.sub(r"\b(the|a|an|of|human|mammalian)\b", " ", t)
    words = []
    for w in t.split():
        w = w.replace("oe", "e").replace("ae", "e")                      # oesophagus = esophagus, haemoglobin = hemoglobin
        if w in _IRREGULAR:
            w = _IRREGULAR[w]
        elif len(w) > 3:
            if w.endswith("ies"):
                w = w[:-3] + "y"
            elif w.endswith(("ches", "shes", "xes", "sses")):
                w = w[:-2]
            elif w.endswith("s") and not w.endswith(("ss", "us", "is", "as")):
                w = w[:-1]
        words.append(w)
    return " ".join(words)


_SYN_RAW = {"gullet": "oesophagus", "food pipe": "oesophagus", "gallbladder": "gall bladder", "colon": "large intestine",
            "large bowel": "large intestine", "windpipe": "trachea", "plasma membrane": "cell membrane",
            "central vacuole": "vacuole", "large vacuole": "vacuole", "sap vacuole": "vacuole", "golgi body": "golgi apparatus",
            "golgi complex": "golgi apparatus", "er": "endoplasmic reticulum", "mitral valve": "bicuspid valve",
            "soma": "cell body", "voltaic cell": "electrochemical cell"}
SYNONYMS = {_canon_basic(k): _canon_basic(v) for k, v in _SYN_RAW.items()}


def canon_name(s: str) -> str:
    c = _canon_basic(s)
    return SYNONYMS.get(c, c)


def _has_phrase(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def match_part(free_name: str, parts: list[str]) -> tuple[Optional[str], str]:
    """Map a name the tutor wrote to one of the VERIFIED parts.  Returns (part, how) or (None, why).
    Ambiguous names ('intestine' when both small and large intestine are parts) are refused, never guessed."""
    f = canon_name(free_name)
    if not f or f in _UNSURE:
        return None, "empty"
    cans = {p: canon_name(p) for p in parts}
    exact = [p for p, c in cans.items() if c == f]
    if len(exact) == 1:
        return exact[0], "exact"
    ft = f.split()
    inside = [(p, len(c.split())) for p, c in cans.items() if _has_phrase(ft, c.split())]       # part name appears in the answer
    if inside:
        top = max(n for _, n in inside)
        best = [p for p, n in inside if n == top]
        return (best[0], "contains") if len(best) == 1 else (None, "ambiguous")
    wider = [p for p, c in cans.items() if _has_phrase(c.split(), ft)]                         # answer is a piece of a part name
    if len(wider) == 1:
        return wider[0], "contains"
    if len(wider) > 1:
        return None, "ambiguous"
    import difflib
    close = difflib.get_close_matches(f, list(cans.values()), n=2, cutoff=0.88)
    if close and (len(close) == 1 or difflib.SequenceMatcher(None, f, close[0]).ratio() - difflib.SequenceMatcher(None, f, close[1]).ratio() > 0.05):
        return next(p for p, c in cans.items() if c == close[0]), "fuzzy"
    return None, "no match"


def parse_region_name(answer: str) -> str:
    """The short name the tutor gave for an outlined region ('' when it said it is unsure or rambled)."""
    lines = [x for x in (answer or "").strip().splitlines() if x.strip()]
    t = lines[0] if lines else ""
    t = re.sub(r"^(the )?(structure|part|region|label|outlined (structure|part))( inside the red outline| outlined in red)?\s*(is|=|:)?\s*", "", t.strip(), flags=re.I)
    t = re.sub(r"^(it is|this is|that is|it's|this looks like|it looks like|looks like|answer:)\s*", "", t, flags=re.I)
    t = re.sub(r"^(a |an |the )", "", t.strip(" .:;\"'`*-"), flags=re.I).strip(" .:;\"'`*-")
    t = re.sub(r"\s+", " ", t)
    if not t or len(t.split()) > 6 or len(t) > 48 or _canon_basic(t) in {_canon_basic(x) for x in _UNSURE}:
        return ""
    return t


def highlight_region(img: Image.Image, mask: np.ndarray, dim: float = 0.35, outline=(220, 30, 30),
                     max_side: Optional[int] = None) -> Image.Image:
    """The picture with everything except `mask` faded and a thin red outline around the region (what the tutor is asked about)."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    if mask.shape != a.shape[:2]:
        raise ValueError("mask and picture sizes differ")
    m = mask[..., None].astype(np.float32)
    out = a * m + (a * dim + 255.0 * (1 - dim)) * (1 - m)
    out[ndi.binary_dilation(mask, iterations=3) & ~mask] = outline
    res = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    if max_side and max(res.size) > max_side:
        res.thumbnail((max_side, max_side), Image.LANCZOS)
    return res


def assign_with_abstain(P, accept: float = 0.6) -> list[tuple[int, int, float]]:
    """P[r][c] = probability that region r is candidate c.  One-to-one assignment (Hungarian) in which a region may also
    abstain: it only takes a candidate when the probability is at least `accept`.  Returns [(r, c, p)]."""
    from scipy.optimize import linear_sum_assignment
    A = np.asarray(P, dtype=float)
    if A.ndim != 2 or A.size == 0:
        return []
    R, K_ = A.shape
    cost = np.hstack([-np.log(np.clip(A, 1e-6, 1.0)), np.full((R, R), -math.log(max(1e-6, accept)))])
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(A[r, c])) for r, c in zip(rows, cols) if c < K_ and A[r, c] >= accept]


def describe_plan(steps: list[dict]) -> str:
    """'I understood this as: ...' in the learner's words (never tool names)."""
    words = {"learn_diagram": "a labelled diagram of {t}", "learn_process": "a step-by-step process: {t}", "chem_molecule": "the molecule {t}",
             "chem_reaction": "a chemical reaction", "studio_image": "a picture", "edit_image": "a change to the picture on screen",
             "explain": "an explanation", "chat": "an answer", "notes": "notes of our discussion", "guide": "a study guide on {t}",
             "lesson": "a narrated lesson", "pdf": "a PDF", "slides": "slides", "relabel": "a change to the labels", "clarify": "a question about what you want"}
    bits = []
    for st in steps:
        t = (st.get("args") or {}).get("topic") or (st.get("args") or {}).get("query") or ""
        bits.append(words.get(st["tool"], st["tool"].replace("_", " ")).format(t=t or "this topic").replace(" of this topic", "").replace(": this topic", ""))
    return ", then ".join(bits)


def assign_soft(P, none=None, accept: float = 0.35, strong: float = 0.6, margin: float = 0.08) -> list[tuple[int, int, float]]:
    """Soft one-to-one labelling.  P[r][c] = probability that region r is part c, none[r] = probability it is 'none of these'.
    A pair is kept when it is the best one-to-one match AND either p >= strong, or p >= accept while clearly ahead of the
    region's next best answer (margin) and ahead of 'none of these'.  A flat distribution therefore never gets a label,
    while a confident-but-not-extreme 0.45 against 0.15 does.  Returns [(r, c, p)]."""
    A = np.asarray(P, dtype=float)
    if A.ndim != 2 or A.size == 0:
        return []
    R, K_ = A.shape
    nn = np.zeros(R) if none is None else np.asarray(none, dtype=float)
    from scipy.optimize import linear_sum_assignment
    cost = np.hstack([-np.log(np.clip(A, 1e-6, 1.0)), np.full((R, R), -math.log(max(1e-6, accept)))])
    rows, cols = linear_sum_assignment(cost)
    out = []
    for r, c in zip(rows, cols):
        if c >= K_:
            continue
        p = float(A[r, c])
        rival = max([float(x) for k, x in enumerate(A[r]) if k != c] + [float(nn[r])])
        if p >= strong or (p >= accept and p - rival >= margin):
            out.append((int(r), int(c), p))
    return out


def position_words(anchor, size) -> str:
    """Where a point is in the picture, in plain words (as you look at the picture, not anatomical left/right)."""
    x, y = anchor[0] / max(1, size[0]), anchor[1] / max(1, size[1])
    h = "left" if x < 0.38 else ("right" if x > 0.62 else "centre")
    v = "upper" if y < 0.38 else ("lower" if y > 0.62 else "middle")
    if v == "middle" and h == "centre":
        return "centre"
    if v == "middle":
        return f"{h} side"
    return f"{v} {h}" if h != "centre" else f"{v} middle"


CONCRETE_RULES = ("Be concrete and specific: name the exact parts or steps involved, say what each one does and what happens next or what it "
                  "connects to, and give one everyday example a learner in East Africa would know (farming, food, health, weather). "
                  "Never write vague lines such as 'it is very important' without saying why.")


@dataclass
class LabelGeom:
    name: str
    colour: tuple
    point: tuple                 # arrow tip, canvas coordinates
    text_xy: tuple               # top-left of the label text block
    anchor: tuple                # where the arrow leaves the label
    side: str
    size: int
    text_w: int = 0
    lines: tuple = ()            # the label split into 1-2 lines
    ctrl: tuple = (0.0, 0.0)     # control point of the curved arrow
    line_h: int = 0


_PROBE = ImageDraw.Draw(Image.new("RGB", (4, 4)))


def _wrap_label(name: str, font, max_w: float) -> Optional[list[str]]:
    """The label on one line, or balanced over two; None when even that is too wide."""
    if _PROBE.textlength(name, font=font) <= max_w:
        return [name]
    words = name.split()
    best = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        w = max(_PROBE.textlength(a, font=font), _PROBE.textlength(b, font=font))
        if best is None or w < best[0]:
            best = (w, [a, b])
    return best[1] if best and best[0] <= max_w else None


def _title_font(text: str, width: float, max_h: int):
    size = max(16, max_h)
    f = get_font(size, True)
    while size > 14 and _PROBE.textlength(text, font=f) > width:
        size -= 2
        f = get_font(size, True)
    return f


def _centered(d: ImageDraw.ImageDraw, xy, text: str, font, fill):
    try:
        d.text(xy, text, font=font, fill=fill, anchor="mm")
    except Exception:                                               # noqa: BLE001
        w = d.textlength(text, font=font)
        d.text((xy[0] - w / 2, xy[1] - 10), text, font=font, fill=fill)


def title_band_height(width: int) -> int:
    return max(60, int(width * 0.085))


def _draw_title(d: ImageDraw.ImageDraw, width: int, band: int, title: str):
    d.rectangle([0, 0, width, band], fill=(244, 246, 250))
    d.line([0, band, width, band], fill=(205, 210, 220), width=3)
    _centered(d, (width / 2, band / 2), title, _title_font(title, width - 48, int(band * 0.62)), (30, 40, 60))


def add_title(img: Image.Image, title: str) -> Image.Image:
    """The picture with a title band on top (used for molecules, reactions and any other study picture)."""
    img = img.convert("RGB")
    W, H = img.size
    band = title_band_height(W)
    out = Image.new("RGB", (W, H + band), "white")
    _draw_title(ImageDraw.Draw(out), W, band, title)
    out.paste(img, (0, band))
    return out


def layout_labels(size: tuple[int, int], items: list[dict], pad_ratio: float = 0.42, title: Optional[str] = None) -> dict:
    """items: [{name, anchor:(x,y) in image coords, colour}] -> geometry in canvas coords.
    Labels sit in the left and right margins, evenly spread over the full height (so they never bunch up), in the same
    top-to-bottom order as their parts (so arrows do not cross).  The font is the largest that fits the room each label
    has (margin width x slot height), wrapping long names over two lines.  Arrows are gentle curves, not straight rules."""
    W, H = size
    pad = int(W * pad_ratio)
    tb = title_band_height(W + 2 * pad) if title else 0
    order = sorted(items, key=lambda i: i["anchor"][0])
    left = [i for i in order if i["anchor"][0] < W / 2]
    right = [i for i in order if i["anchor"][0] >= W / 2]
    while len(left) - len(right) > 1:
        right.insert(0, left.pop())
    while len(right) - len(left) > 1:
        left.append(right.pop(0))
    m = max(len(left), len(right), 1)
    slot = H / m
    max_w = pad - 90
    fsize, wraps = 14, {}
    for fs in range(min(80, int(slot * 0.6)), 13, -1):
        font = get_font(fs, True)
        ok, w_ = True, {}
        for it in items:
            ln = _wrap_label(it["name"], font, max_w)
            if ln is None or len(ln) * fs * 1.2 > slot * 0.92:
                ok = False
                break
            w_[it["name"]] = ln
        if ok:
            fsize, wraps = fs, w_
            break
    if not wraps:
        font = get_font(fsize, True)
        wraps = {it["name"]: [it["name"]] for it in items}
    font = get_font(fsize, True)
    line_h = int(fsize * 1.2)
    chip = max(10, fsize // 2)
    CW = W + 2 * pad
    lw = max(3, W // 220)
    geoms: list[LabelGeom] = []
    for side, group in (("L", left), ("R", right)):
        group = sorted(group, key=lambda i: i["anchor"][1])
        k = len(group)
        if not k:
            continue
        hs = [len(wraps[i["name"]]) * line_h for i in group]
        ys = [tb + 0.8 * ((j + 0.5) * H / k) + 0.2 * i["anchor"][1] for j, i in enumerate(group)]
        for j in range(1, k):                                       # keep a clear gap between neighbours
            ys[j] = max(ys[j], ys[j - 1] + (hs[j] + hs[j - 1]) / 2 + fsize * 0.35)
        for j in range(k - 1, -1, -1):
            top, bottom = tb + hs[j] / 2 + 4, tb + H - hs[j] / 2 - 4
            ys[j] = min(max(ys[j], top), bottom)
            if j < k - 1:
                ys[j] = min(ys[j], ys[j + 1] - (hs[j] + hs[j + 1]) / 2 - fsize * 0.35)
        for it, y, h in zip(group, ys, hs):
            lines = wraps[it["name"]]
            tw = int(max(_PROBE.textlength(x, font=font) for x in lines))
            tx = 16 + chip + 8 if side == "L" else CW - 16 - chip - 8 - tw
            ax = tx + tw + 14 if side == "L" else tx - 14
            px, py = it["anchor"][0] + pad, it["anchor"][1] + tb
            vx, vy = px - ax, py - y
            ln = math.hypot(vx, vy) or 1.0
            bow = 0.16 * ln * (1 if vy >= 0 else -1) * (1 if side == "L" else -1)
            ctrl = ((ax + px) / 2 - vy / ln * bow, (y + py) / 2 + vx / ln * bow)
            geoms.append(LabelGeom(it["name"], tuple(it["colour"]), (px, py), (tx, y - h / 2), (ax, y), side, fsize, tw, tuple(lines), ctrl, line_h))
    return {"pad": pad, "canvas": (CW, H + tb), "geoms": geoms, "fsize": fsize, "tb": tb, "title": title, "lw": lw, "chip": chip}


def _bezier(p0, c, p1, n: int = 30) -> list[tuple]:
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1]) for t in (i / n for i in range(n + 1))]


def _arrow(d: ImageDraw.ImageDraw, p0, p1, colour, width: int, ctrl=None):
    """Arrow from p0 to p1; a gentle curve when `ctrl` is given.  White halo so it reads on any background."""
    pts = _bezier(p0, ctrl, p1) if ctrl else [p0, p1]
    ang = math.atan2(p1[1] - pts[-3][1], p1[0] - pts[-3][0]) if ctrl else math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    head = max(18, width * 6)
    base = (p1[0] - head * math.cos(ang), p1[1] - head * math.sin(ang))
    line = pts[:-1] + [base] if ctrl else [p0, base]
    for col, w in (((255, 255, 255), width + 4), (colour, width)):
        d.line(line, fill=col, width=w, joint="curve")
    tri = [p1, (base[0] + head * 0.45 * math.sin(ang), base[1] - head * 0.45 * math.cos(ang)),
           (base[0] - head * 0.45 * math.sin(ang), base[1] + head * 0.45 * math.cos(ang))]
    d.polygon(tri, fill=colour, outline=(255, 255, 255))


def render_labels(img: Image.Image, layout: dict, show: Optional[set] = None, spotlight: Optional[np.ndarray] = None,
                  spot_colour=(255, 200, 0), faded: Optional[set] = None) -> Image.Image:
    """Draw (a subset of) the labels with coloured curved arrows, and the title band if the layout has one.  `spotlight` =
    boolean mask in IMAGE coords: everything else is dimmed (lesson animation).  `faded` names are drawn lighter."""
    pad, tb = layout["pad"], layout.get("tb", 0)
    base = img.convert("RGB")
    if spotlight is not None:
        arr = np.asarray(base).astype(np.float32)
        arr[~spotlight] = arr[~spotlight] * 0.42 + 255 * 0.58 * 0.35
        edge = spotlight & ~ndi.binary_erosion(spotlight, iterations=3)
        arr[edge] = spot_colour
        base = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    canvas = Image.new("RGB", layout["canvas"], "white")
    canvas.paste(base, (pad, tb))
    d = ImageDraw.Draw(canvas)
    if layout.get("title"):
        _draw_title(d, canvas.width, tb, layout["title"])
    font = get_font(layout["fsize"], True)
    lw, chip = layout.get("lw", 3), layout.get("chip", 10)
    for g in layout["geoms"]:
        if show is not None and g.name not in show:
            continue
        col = g.colour
        is_faded = bool(faded and g.name in faded)
        if is_faded:
            col = tuple(int(c * 0.45 + 255 * 0.55) for c in g.colour)
        r = max(4, lw * 1.5)
        d.ellipse([g.point[0] - r - 1, g.point[1] - r - 1, g.point[0] + r + 1, g.point[1] + r + 1], fill="white")
        d.ellipse([g.point[0] - r, g.point[1] - r, g.point[0] + r, g.point[1] + r], fill=col)
        ux, uy = g.point[0] - g.ctrl[0], g.point[1] - g.ctrl[1]
        un = math.hypot(ux, uy) or 1.0
        tip = (g.point[0] - ux / un * (r + 3), g.point[1] - uy / un * (r + 3))        # arrow ends at the dot, not under it
        _arrow(d, g.anchor, tip, col, lw, ctrl=g.ctrl)
        block_h = len(g.lines) * g.line_h
        cx = g.text_xy[0] - chip - 8 if g.side == "L" else g.text_xy[0] + g.text_w + 8
        cy = g.text_xy[1] + block_h / 2 - chip / 2
        d.rectangle([cx, cy, cx + chip, cy + chip], fill=col)
        for k, line in enumerate(g.lines):
            tx = g.text_xy[0] if g.side == "L" else g.text_xy[0] + g.text_w - d.textlength(line, font=font)
            d.text((tx, g.text_xy[1] + k * g.line_h), line, font=font, fill=(120, 120, 120) if is_faded else (25, 25, 30))
    return canvas


# =====================================================================================================
# 5c. Processes on ONE verified diagram: scope, parts and flows are decided by code from the verified part list
# =====================================================================================================
MOVE_CUE = re.compile(r"\b(to|into|towards?|through|from|travels?|moves?|passes|carries|flows?|sends?|relays?|reaches|enters?|leaves?|"
                      r"down|up|back|pumps?|pushes?|along|returns?|arrives?)\b", re.I)
FLOW_THINGS = ["oxygenated blood", "deoxygenated blood", "blood", "nerve impulse", "impulse", "signals", "signal", "information", "food", "bile",
               "water", "air", "oxygen", "carbon dioxide", "urine", "waste", "sugar", "glucose", "light", "message", "enzymes"]


def mentioned_parts(text: str, components: list[str]) -> list[str]:
    """Components named in the text, in the order they are first mentioned.  A shorter name that sits inside a longer matched one
    ('atrium' inside 'right atrium') is not counted separately.  Synonyms count ('gullet' -> oesophagus)."""
    low = " " + _norm(text) + " "
    for syn, canon in SYNONYMS.items():                                 # let a synonym in the text count as the canonical word
        if syn != canon and f" {syn} " in low:
            low = low.replace(f" {syn} ", f" {canon} ")
    hits: list[tuple[int, str]] = []
    for c in components:
        pos = [low.find(f" {v} ") for v in _variants(canon_name(c)) | _variants(c)]
        pos = [p for p in pos if p >= 0]
        if pos:
            hits.append((min(pos), c))
    names = [_norm(c) for _, c in hits]
    keep = [(p, c) for p, c in hits if not any(_norm(c) != o and f" {_norm(c)} " in f" {o} " for o in names)]
    return [c for _, c in sorted(keep)]


def _span(text_low: str, name: str) -> int:
    pos = [text_low.find(f" {v} ") for v in _variants(canon_name(name)) | _variants(name)]
    pos = [p for p in pos if p >= 0]
    return min(pos) if pos else -1


def flow_thing(text: str, declared: str = "") -> str:
    low = " " + _norm(text) + " "
    d = _norm(declared)
    if d and len(d.split()) <= 3 and all(f" {w} " in low for w in d.split()):
        return d
    for t in FLOW_THINGS:
        if f" {t} " in low or f" {t}s " in low:
            return t
    return ""


def step_graph(title: str, text: str, components: list[str], declared_parts=None, declared_flows=None) -> dict:
    """Which parts a step is about and which arrows it shows - all grounded in the step's own wording.
    parts  = components the step literally names (else the ones the tutor declared, max 2)
    flows  = the tutor's arrows if both ends are named in the step, else arrows along the order the parts are named when the
             sentence describes movement.  Direction follows the words ('receives ... from X' flows from X)."""
    full = f"{title}. {text}"
    low = " " + _norm(text) + " "                                       # order and direction come from the sentence, not the title
    ment_text = mentioned_parts(text, components)
    ment = ment_text + [m for m in mentioned_parts(title, components) if m not in ment_text]
    decl = []
    for d in declared_parts or []:
        p, _ = match_part(str(d), components)
        if p and p not in decl:
            decl.append(p)
    parts = ment or decl[:2]
    flows: list[dict] = []
    for f in declared_flows or []:
        if not isinstance(f, dict):
            continue
        a, _ = match_part(str(f.get("from", "")), components)
        b, _ = match_part(str(f.get("to", "")), components)
        if a and b and a != b and a in ment and b in ment and not any(x["from"] == a and x["to"] == b for x in flows):
            flows.append({"from": a, "to": b, "what": flow_thing(full, str(f.get("what", "")))})
    if not flows and len(ment_text) >= 2 and MOVE_CUE.search(text or ""):
        for a, b in zip(ment_text, ment_text[1:]):
            pa, pb = _span(low, a), _span(low, b)
            seg = low[pa:pb] if 0 <= pa < pb else ""
            rev = bool(re.search(r"\bfrom\b", seg)) and not re.search(r"\b(to|into|towards?|through|pass\w*|sends?|pumps?|relays?)\b", seg)
            flows.append({"from": b if rev else a, "to": a if rev else b, "what": flow_thing(full)})
    return {"parts": parts, "flows": flows[:3]}


def foreign_terms(components: list[str], subject: str, all_part_names) -> set:
    """Structure names that belong to OTHER organs or systems (taken from the built-in notes).  If a step about the brain
    names the 'spinal cord' or 'optic nerve', it has wandered out of scope."""
    inside = [_norm(c) for c in components] + [_norm(subject)]
    out = set()
    for n in all_part_names:
        n = _norm(n)
        if len(n) < 6 or any(f" {n} " in f" {i} " or f" {i} " in f" {n} " for i in inside if i):
            continue
        out.add(n)
    return out


def out_of_scope(text: str, foreign: set) -> list[str]:
    low = " " + _norm(text) + " "
    return sorted(t for t in foreign if f" {t} " in low or f" {t}s " in low)


def verify_plan(steps: list[dict], components: list[str], source_text: str, max_steps: int, foreign: Optional[set] = None,
                trusted: bool = False) -> tuple[list[dict], list[str]]:
    """Keep the steps that (1) are supported by the source, (2) stay inside the organ/system, (3) involve at least one of its parts.
    Returns (steps with parts and flows, out-of-scope terms that caused drops - for one corrective retry)."""
    kept, bad = [], []
    for st in steps:
        txt = f"{st['title']}. {st['text']}"
        if not trusted:
            if foreign:
                oos = out_of_scope(txt, foreign)
                if oos:                                                 # checked first, so the retry can be told exactly what to leave out
                    bad += oos
                    continue
            if not verify_steps([st], source_text, 1):
                continue
        g = step_graph(st["title"], st["text"], components, st.get("parts"), st.get("flows"))
        if not g["parts"]:
            continue
        kept.append({**st, "parts": g["parts"], "flows": g["flows"]})
        if len(kept) >= max_steps:
            break
    return kept, sorted(set(bad))


def parse_plan(answer: str) -> Optional[dict]:
    """Like parse_process, but keeps the tutor's 'parts' and 'flows' per step (they are only suggestions: step_graph checks them)."""
    base = parse_process(answer)
    obj = parse_json(answer)
    if not base or not isinstance(obj, (dict, list)):
        return base
    raw = obj.get("steps") if isinstance(obj, dict) else obj
    if isinstance(raw, list):
        j = 0
        for it in raw:
            if not isinstance(it, dict):
                continue
            title = re.sub(r"\s+", " ", str(it.get("title") or it.get("name") or "")).strip(" .:-")[:60]
            text = re.sub(r"\s+", " ", str(it.get("text") or it.get("description") or "")).strip()
            if title and len(text.split()) >= 3 and j < len(base["steps"]):
                base["steps"][j]["parts"] = [str(x) for x in it.get("parts", [])][:6] if isinstance(it.get("parts"), list) else []
                base["steps"][j]["flows"] = [f for f in it.get("flows", []) if isinstance(f, dict)][:4] if isinstance(it.get("flows"), list) else []
                j += 1
    return base


RE_DEEPER = re.compile(r"\b(deeper|in depth|in more detail|more detail|elaborate|go on|tell me more|expand on|dig deeper|why is that|but why|how come)\b", re.I)
DEEP_SECTIONS = ["Short answer", "How it works", "A real example", "A common mistake", "Think about it"]
_Q_LEAD = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could|would) you\s+)?(?:tell me\s+|explain\s+|describe\s+)?(?:why|how|what|when|where|who|which)?\s*"
                     r"(?:is|are|was|were|does|do|did|can|could|would|should|happens?)?\s*(?:the|a|an)?\s*", re.I)


def question_topic(q: str, max_words: int = 5) -> str:
    """The subject words of a science question, for looking up trusted notes ('Why is the sky blue?' -> 'sky blue')."""
    t = _Q_LEAD.sub("", (q or "").strip().rstrip("?.! "))
    words = content_words(t)
    return " ".join(words[:max_words])


# =====================================================================================================
# 6. Sections, narration, lesson timeline
# =====================================================================================================
DIAGRAM_SECTIONS = ["What you see", "The parts and what they do", "Why it matters", "Check yourself"]
PROCESS_SECTIONS = ["The big picture", "Step by step", "Why it matters", "Check yourself"]
CHAT_SECTIONS = ["Short answer", "Explanation", "Key points", "Try this"]
_HEAD = re.compile(r"^\s*#{1,4}\s*(.+?)\s*:?\s*$")


def parse_sections(text: str) -> list[tuple[str, str]]:
    out, cur_t, cur_b = [], None, []
    for line in (text or "").splitlines():
        m = _HEAD.match(line)
        if m:
            if cur_t is not None or "".join(cur_b).strip():
                out.append((cur_t or "", "\n".join(cur_b).strip()))
            cur_t, cur_b = m.group(1).strip(), []
        else:
            cur_b.append(line)
    if cur_t is not None or "".join(cur_b).strip():
        out.append((cur_t or "", "\n".join(cur_b).strip()))
    return [(t, b) for t, b in out if t or b]


def has_sections(text: str, expected: list[str], at_least: int = 3) -> bool:
    heads = {_norm(t) for t, _ in parse_sections(text) if t}
    return sum(1 for e in expected if _norm(e) in heads) >= at_least


def sections_markdown(text: str, expected: list[str]) -> str:
    """Clean `## Title` blocks in the expected order; anything outside the expected titles is kept, never dropped."""
    secs = parse_sections(text)
    if not secs:
        return text.strip()
    by = {_norm(t): (t, b) for t, b in secs if t}
    out = []
    for e in expected:
        if _norm(e) in by:
            out.append(f"## {e}\n{by.pop(_norm(e))[1]}".rstrip())
    for t, b in secs:
        if not t:
            out.insert(0, b) if b else None
        elif _norm(t) in by:
            out.append(f"## {t}\n{b}".rstrip())
    return "\n\n".join(x for x in out if x.strip())


def wrap_as_sections(text: str, first_title: str) -> str:
    """Deterministic fallback when the tutor ignores the format: keep everything under one clean heading."""
    return f"## {first_title}\n{(text or '').strip()}"


def sentence_for(part: str, source_text: str, max_words: int) -> str:
    """Narration fallback: the first sentence of the source that mentions the part."""
    for s in re.split(r"(?<=[.!?])\s+", source_text or ""):
        if any(f" {v} " in " " + _norm(s) + " " for v in _variants(part)) and 4 <= len(s.split()) <= 70:
            return words_trim(re.sub(r"\s+", " ", s).strip(), max_words)
    return f"This is the {part}."


def parse_narration(answer: str, parts: list[str]) -> dict[str, str]:
    obj = parse_json(answer)
    out = {}
    if isinstance(obj, dict):
        lookup = {_norm(p): p for p in parts}
        for k, v in obj.items():
            p = lookup.get(_norm(str(k)))
            if p and isinstance(v, str) and len(v.split()) >= 3:
                out[p] = re.sub(r"\s+", " ", v).strip()
    return out


def build_timeline(segments: list[dict], lead: float = 0.8, tail: float = 1.2, gap: float = 0.35) -> dict:
    """segments: [{name, text, dur}] -> start/end times.  Returns {"items": [...], "total": seconds}."""
    t, items = lead, []
    for s in segments:
        a = t
        b = a + max(0.8, float(s["dur"]))
        items.append({**s, "start": a, "end": b})
        t = b + gap
    return {"items": items, "total": t - gap + tail, "lead": lead}


def word_progress(text: str, frac: float) -> int:
    """How many words have been spoken after `frac` of the segment (weighted by word length)."""
    words = text.split()
    if not words:
        return 0
    w = np.array([len(x) + 2 for x in words], dtype=float)
    cum = np.cumsum(w) / w.sum()
    return int(np.searchsorted(cum, min(max(frac, 0.0), 1.0) - 1e-9) + 1) if frac > 0 else 0


# =====================================================================================================
# 7. Document passages (lightweight retrieval for uploaded files)
# =====================================================================================================
def chunk_text(text: str, size: int = 700) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\r\n\r\n", text or "") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            while len(p) > size:
                chunks.append(p[:size])
                p = p[size:]
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def select_passages(text: str, query: str, budget_chars: int = 5000) -> str:
    chunks = chunk_text(text)
    if sum(len(c) for c in chunks) <= budget_chars:
        return "\n\n".join(chunks)
    q = set(content_words(query))
    scored = []
    for i, c in enumerate(chunks):
        cw = set(content_words(c))
        scored.append((len(q & cw) / (1 + 0.02 * len(cw)) if q else 0.0, -i, i))
    keep, used = [], 0
    for _, _, i in sorted(scored, reverse=True):
        if used + len(chunks[i]) > budget_chars:
            continue
        keep.append(i)
        used += len(chunks[i])
    return "\n\n".join(chunks[i] for i in sorted(keep))


# =====================================================================================================
# 8. Processes (infographic steps) - verification and panel prompts
# =====================================================================================================
PANEL_STYLE = ("simple flat illustration, bold dark outlines, flat colours, one clear scene, white background, "
               "no text, no letters, no numbers, no labels, no arrows")
PROCESS_SHAPES = ("linear", "cycle", "io")
PROCESS_STEPS = {"ordinary": (4, 5), "advanced": (5, 7), "university": (6, 9)}


def parse_process(answer: str) -> Optional[dict]:
    """Tutor JSON {"shape":..., "steps":[{"title","text","scene"}]} -> cleaned dict, or None."""
    obj = parse_json(answer)
    if isinstance(obj, list):
        obj = {"steps": obj}
    if not isinstance(obj, dict) or not isinstance(obj.get("steps"), list):
        return None
    steps = []
    for it in obj["steps"]:
        if isinstance(it, str):
            it = {"title": it, "text": it}
        if not isinstance(it, dict):
            continue
        title = re.sub(r"\s+", " ", str(it.get("title") or it.get("name") or "")).strip(" .:-")[:60]
        text = re.sub(r"\s+", " ", str(it.get("text") or it.get("description") or "")).strip()
        scene = re.sub(r"\s+", " ", str(it.get("scene") or it.get("visual") or title)).strip()[:160]
        if title and len(text.split()) >= 3:
            steps.append({"title": title, "text": text, "scene": scene or title})
    shape = str(obj.get("shape", "linear")).lower()
    return {"shape": shape if shape in PROCESS_SHAPES else "linear", "steps": steps}


def verify_steps(steps: list[dict], source_text: str, max_steps: int) -> list[dict]:
    """Keep steps whose wording is supported by the source (>= 2 shared content words, or 1 for short text)."""
    src = set(content_words(source_text))
    out = []
    for st in steps:
        words = set(content_words(st["title"] + " " + st["text"]))
        need = 2 if len(words) >= 6 else 1
        if len(words & src) >= need:
            out.append(st)
        if len(out) >= max_steps:
            break
    return out


def build_panel_prompt(scene: str, subject: str = "") -> str:
    lead = f"{subject}, " if subject else ""
    return f"{lead}{scene}, {PANEL_STYLE}"


def verify_qa(items: list, source_text: str, max_items: int) -> list[dict]:
    """Practice questions: the answer must be supported by the source (>= half its content words appear there)."""
    src = set(content_words(source_text))
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        q, a = str(it.get("q") or it.get("question") or "").strip(), str(it.get("a") or it.get("answer") or "").strip()
        aw = set(content_words(a))
        if len(q.split()) >= 3 and aw and len(aw & src) / len(aw) >= 0.5:
            out.append({"q": q, "a": a})
        if len(out) >= max_items:
            break
    return out


def verify_terms(items: list, source_text: str, max_items: int) -> list[dict]:
    hay = " " + _norm(source_text) + " "
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        term, meaning = str(it.get("term") or "").strip(), str(it.get("meaning") or it.get("definition") or "").strip()
        if term and len(meaning.split()) >= 3 and any(f" {v} " in hay for v in _variants(term)):
            out.append({"term": term, "meaning": meaning})
        if len(out) >= max_items:
            break
    return out
