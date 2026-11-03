"""
mm_app.py - Streamlit front end for "Understood".
Run:  streamlit run mm_app.py
ONE conversation does everything: the learner types what they want (a question, "draw the heart", "show how blood flows through it",
"make it brighter", "make a study guide") and the tutor decides what to do, remembering the whole conversation.  The sidebar holds the
specialised tools (picture studio, picture editing, guides, exports) for people who prefer a form.
The learner never sees models, servers or GPUs.  Connection settings live at  ...?admin=1  (optional PIN via
UNDERSTOOD_ADMIN_PIN).  Requests run in a background thread, so the page stays usable (step through a process,
read, chat) while pictures are still being drawn.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import streamlit as st

import mm_core as C
import mm_doc
from mm_engine import Engine, Settings

DATA_DIR = Path(os.environ.get("UNDERSTOOD_DATA", Path.home() / "understood_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Understood", page_icon="🔬", layout="wide")

SUGGESTIONS = ["Draw a labelled diagram of the human heart", "Show step by step how blood flows through the heart",
               "Why do we sweat when it is hot?", "Make a study guide on photosynthesis"]


@st.cache_resource
def get_engine(url: str) -> Engine:
    return Engine(DATA_DIR, url)


settings = Settings(DATA_DIR)
admin = st.query_params.get("admin") == "1"
pin = os.environ.get("UNDERSTOOD_ADMIN_PIN", "")
if admin and pin and st.session_state.get("pin_ok") is not True:
    with st.sidebar:
        if st.text_input("Admin PIN", type="password") == pin:
            st.session_state.pin_ok = True
            st.rerun()
    admin = False

# ------------------------------------------------------------------ connection (hidden from learners)
if not settings.server_url:
    st.title("Understood")
    st.info("First-time setup: paste the address of your learning service.")
    url = st.text_input("Service address", placeholder="https://…")
    if st.button("Save") and url.strip():
        settings.set(server_url=url.strip().rstrip("/"))
        st.rerun()
    st.stop()

engine = get_engine(settings.server_url)
engine.set_server(settings.server_url)

if admin:
    with st.sidebar.expander("Admin", expanded=False):
        new = st.text_input("Service address", value=settings.server_url)
        c1, c2, c3 = st.columns(3)
        if c1.button("Save"):
            settings.set(server_url=new.strip().rstrip("/"))
            st.rerun()
        if c2.button("Check"):
            try:
                st.json(engine.client.health())
            except Exception as e:                                               # noqa: BLE001
                st.error(str(e)[:200])
        if c3.button("Warm up"):
            with st.spinner("Loading…"):
                try:
                    st.json(engine.client.warmup())
                except Exception as e:                                           # noqa: BLE001
                    st.error(str(e)[:200])
        st.json(engine.bank_status())
        st.checkbox("Use online sources (Wikipedia, PubChem)", value=True, key="online")
        engine.online_sources = st.session_state.get("online", True)

# ------------------------------------------------------------------ session + sidebar
if "tid" not in st.session_state:
    ts = engine.store.threads()
    st.session_state.tid = ts[0][0] if ts else engine.store.new_thread()
tid = st.session_state.tid
running = engine.status(tid) == "running"
level_key = "university"                                      # learning levels are retired for now: one standard depth
OUTPUTS = ["Auto", "Diagram", "Process", "Picture", "Explain", "Edit"]

def go(request: dict | None):
    """Start (or resume) a background run and refresh the page."""
    if request is not None:
        request.setdefault("level", level_key)
        request.setdefault("mode", "assistant")
    engine.start(tid, request)
    st.rerun()


def attach(files) -> list:
    return [{"name": f.name, "data_b64": base64.b64encode(f.getvalue()).decode()} for f in (files or [])]


with st.sidebar:
    st.title("Understood")
    st.caption("Learn it. See it. Understand it.")
    if st.button("➕ New chat", width="stretch"):
        st.session_state.tid = engine.store.new_thread()
        st.rerun()
    for t_id, title, _ts in engine.store.threads()[:12]:
        if st.button(("▸ " if t_id == tid else "") + title[:34], key=f"th_{t_id}", width="stretch"):
            st.session_state.tid = t_id
            st.rerun()
    st.caption(f"Chats are kept for {engine.retention_hours:g} hours after you last use them." if engine.retention_hours else "Chats are kept until you delete them.")
    with st.expander("This chat"):
        kept = (engine.tdir(tid) / ".keep").exists()
        if st.toggle("Keep this chat", value=kept, key=f"keep_{tid}") != kept:
            engine.pin(tid, not kept)
            st.rerun()
        if st.button("🗑 Delete this chat", disabled=running, width="stretch"):
            engine.delete_thread(tid)
            ts_ = engine.store.threads()
            st.session_state.tid = ts_[0][0] if ts_ else engine.store.new_thread()
            st.rerun()
    st.divider()
    with st.expander("🧰 More tools", expanded=False):
        st.caption("Everything here can also be asked for by typing in the chat. Exports are at the end of each answer.")
        t_pic, t_edit, t_learn = st.tabs(["Picture", "Edit", "Learn"])
        with t_pic:
            i_prompt = st.text_area("Describe the picture", key="i_prompt", height=80)
            c1, c2 = st.columns(2)
            aspect = c1.selectbox("Shape", ["square", "wide", "tall"], key="i_aspect")
            quality = "high" if c2.toggle("Higher quality (slower)", key="i_q") else "fast"
            if st.button("Create picture", key="i_go", disabled=running or not i_prompt.strip(), width="stretch"):
                go({"text": i_prompt.strip(), "mode": "studio", "options": {"aspect": aspect, "quality": quality}, "shown_text": f"Picture: {i_prompt.strip()}"})
        with t_edit:
            e_file = st.file_uploader("Picture to change (empty = the last picture in this chat)", type=["png", "jpg", "jpeg", "webp"], key="e_file")
            e_text = st.text_input("What should change?", key="e_text")
            if st.button("Change it", key="e_go", disabled=running or not e_text.strip(), width="stretch"):
                go({"text": e_text.strip(), "mode": "edit", "attachments": attach([e_file] if e_file else []), "shown_text": e_text.strip()})
        with t_learn:
            l_topic = st.text_input("Topic", key="l_topic", placeholder="e.g. the human brain")
            c1, c2 = st.columns(2)
            if c1.button("Diagram", key="l_diag", disabled=running or not l_topic.strip(), width="stretch"):
                go({"action": "show_as", "kind": "diagram", "text": l_topic.strip(), "shown_text": f"Labelled diagram: {l_topic.strip()}"})
            if c2.button("Process", key="l_proc", disabled=running or not l_topic.strip(), width="stretch"):
                go({"action": "process", "text": l_topic.strip(), "shown_text": f"Step by step: {l_topic.strip()}"})
            g_q = st.checkbox("Study guide with practice questions", True, key="g_q")
            if st.button("Study guide", key="g_go", disabled=running, width="stretch"):
                go({"action": "guide", "text": l_topic.strip(), "options": {"questions": g_q}, "shown_text": f"Study guide: {l_topic.strip() or 'my notes'}"})


# ------------------------------------------------------------------ rendering
def render_process(path: str, key: str, live: bool):
    try:
        rec = json.loads((DATA_DIR / path).read_text(encoding="utf-8"))
    except Exception:                                                            # noqa: BLE001
        st.info("Preparing the steps…")
        return
    steps = rec["steps"]
    n = len(steps)
    k = f"seg_{key}"
    first_pending = next((i for i, s in enumerate(steps) if s["status"] == "pending"), None)

    def icon(i):
        s = steps[i]["status"]
        if s == "ready":
            return "✓"
        if s == "failed":
            return "✗"
        return "⏳" if (live and i == first_pending) else "·"

    st.markdown(f"##### {rec['title']} - step by step" + (f" · inside the {rec['subject']}" if rec.get("system") and rec.get("subject") else ""))
    if k not in st.session_state or st.session_state[k] is None or st.session_state[k] >= n:
        st.session_state[k] = 0

    def nav(d):
        st.session_state[k] = max(0, min(n - 1, (st.session_state.get(k) or 0) + d))

    st.segmented_control("Steps", list(range(n)), key=k, format_func=lambda i: f"{i + 1} {icon(i)}", label_visibility="collapsed")
    i = st.session_state.get(k) or 0
    st_ = steps[i]
    left, right = st.columns([3, 2])
    with left:
        if st_["status"] == "ready" and st_.get("image_path") and (DATA_DIR / st_["image_path"]).exists():
            st.image(str(DATA_DIR / st_["image_path"]), width="stretch")
        elif st_["status"] == "failed":
            st.warning(f"Step {i + 1} could not be drawn.")
            st.button("Try drawing this step again", key=f"retry_{key}_{i}", disabled=running,
                      on_click=lambda i=i: engine.start(tid, {"action": "retry_step", "n": i, "level": level_key, "mode": "assistant", "shown_text": f"Redraw step {i + 1}"}))
        else:
            st.info(f"Step {i + 1} is still to come - it will appear here as soon as it is drawn." if (live or running) else
                    f"Step {i + 1} has not been drawn yet.")
    with right:
        st.markdown(f"**Step {i + 1} of {n}: {st_['title']}**")
        st.write(st_["text"])
        if st_.get("flows"):
            st.caption("Arrows: " + " · ".join(f"{f['from']} → {f['to']}" + (f" ({f['what']})" if f.get("what") else "") for f in st_["flows"]))
        c1, c2 = st.columns(2)
        c1.button("◀ Previous", key=f"prev_{key}", disabled=i == 0, on_click=nav, args=(-1,), width="stretch")
        c2.button("Next ▶", key=f"next_{key}", disabled=i >= n - 1, on_click=nav, args=(1,), width="stretch")
    if rec.get("overview_path") and (DATA_DIR / rec["overview_path"]).exists():
        with st.expander("See the whole process in one picture"):
            p = DATA_DIR / rec["overview_path"]
            st.image(str(p), width="stretch")
            st.download_button("Download this infographic", p.read_bytes(), file_name=f"{C.safe_name(rec['title'])}.png", mime="image/png", key=f"ov_{key}")
    if rec.get("source_note"):
        st.caption(rec["source_note"])


def render_doc(path: str, key: str):
    try:
        rec = json.loads((DATA_DIR / path).read_text(encoding="utf-8"))
    except Exception:                                                            # noqa: BLE001
        return
    with st.expander(f"📘 {rec['title']} - read it here", expanded=False):
        blocks = rec["blocks"]
        chunk: list = []
        for b in blocks:
            if b["t"] == "image":
                if chunk:
                    st.markdown(mm_doc.blocks_to_markdown(chunk))
                    chunk = []
                p = DATA_DIR / b["path"]
                if p.exists():
                    st.image(str(p), caption=b.get("caption") or None, width="stretch")
            else:
                chunk.append(b)
        if chunk:
            st.markdown(mm_doc.blocks_to_markdown(chunk))
        if any(b["t"] == "qa" for b in blocks):
            with st.expander("Answer key"):
                st.markdown(mm_doc.blocks_to_markdown([b for b in blocks if b["t"] == "qa"], answers=True))


def render_outputs(outs: list, key: str, live: bool = False):
    for i, o in enumerate(outs):
        t, k = o["type"], f"{key}_{i}"
        if t == "text":
            st.markdown(o["text"])
        elif t == "note":
            st.info(o["text"])
        elif t == "understood":
            st.caption(f"I understood this as: {o['text']}")
        elif t == "image":
            p = DATA_DIR / o["path"]
            if p.exists():
                st.image(str(p), caption=o.get("caption") or None, width="stretch")
                st.download_button("Download picture", p.read_bytes(), file_name=p.name, mime="image/png", key=f"{k}_dl")
        elif t == "video":
            p = DATA_DIR / o["path"]
            if p.exists():
                st.video(p.read_bytes(), format="video/mp4")
                st.download_button("Download lesson video", p.read_bytes(), file_name=p.name, mime="video/mp4", key=f"{k}_v")
        elif t == "file":
            p = DATA_DIR / o["path"]
            if p.exists():
                st.download_button(f"📄 Download {o.get('name', p.name)}", p.read_bytes(), file_name=o.get("name", p.name),
                                   mime=o.get("mime", "application/octet-stream"), key=f"{k}_f")
        elif t == "sources":
            st.caption(" · ".join(o["items"]))
        elif t == "process":
            render_process(o["path"], Path(o["path"]).stem, live)
        elif t == "doc":
            render_doc(o["path"], k)
        elif t == "choices":
            st.write(o["text"])
            cols = st.columns(len(o["options"]))
            labels = {"diagram": "🧬 Parts diagram", "process": "🔁 Step-by-step process", "picture": "🖼 Just a picture"}
            for c, opt in zip(cols, o["options"]):
                if c.button(labels.get(opt, opt), key=f"{k}_{opt}", disabled=running):
                    go({"action": "show_as", "kind": opt, "text": o.get("topic", ""), "shown_text": labels.get(opt, opt)})


# ------------------------------------------------------------------ conversation
st.header("Understood")
chat = engine.chat_get(tid)
if not chat and not running:
    st.markdown("Ask me anything in science, or tell me what to make - I work out the rest. For example:")
    cols = st.columns(2)
    for k_, sug in enumerate(SUGGESTIONS):
        if cols[k_ % 2].button(sug, key=f"sug_{k_}", width="stretch"):
            go({"text": sug, "shown_text": sug})
for n_, m in enumerate(chat):
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        if m["role"] == "user":
            st.markdown(m["text"] or "_(sent a file)_")
            if m.get("files"):
                st.caption("Attached: " + ", ".join(m["files"]))
        else:
            render_outputs(m["outputs"], f"m{n_}")
            if m.get("status") == "parked":
                st.warning("The connection was lost. Your request is saved - press Resume below.")
            elif m.get("status") == "failed":
                st.error("Sorry, I could not finish that. Please try again or rephrase.")

if running:
    def live_view():
        if engine.status(tid) != "running":
            st.rerun()
        lv = engine.live(tid)
        with st.chat_message("assistant"):
            with st.status("Working on it…", expanded=True):
                for _ts, txt in lv["log"][-5:]:
                    st.write(txt)
            render_outputs(lv["outputs"], "live", live=True)
        if st.button("⏹ Stop", key="stop_run", help="Stop now and undo this request"):
            engine.cancel(tid)
    st.fragment(run_every=2)(live_view)()

status = engine.status(tid)
if not running and status == "cancelled":
    st.info("Stopped. I undid that request, so the chat is as it was.")
if not running and (status == "parked" or engine.parked(tid)):
    if st.button("▶ Resume where I stopped", type="primary"):
        go(None)

last = chat[-1] if chat else None
if not running and last and last["role"] == "assistant" and last.get("topic") and last.get("status") == "done":
    sc = st.columns([1.3, 1, 1, 1, 3])
    sc[0].caption("Show it as:")
    for c, (kind, label) in zip(sc[1:4], [("diagram", "Diagram"), ("process", "Process"), ("picture", "Picture")]):
        if c.button(label, key=f"showas_{kind}_{len(chat)}"):
            go({"action": "show_as", "kind": kind, "text": last["topic"], "shown_text": f"Show it as a {kind}"})

with st.expander("Activity"):
    rows = engine.store.activity(tid, 40)
    st.text("\n".join(f"{time.strftime('%H:%M:%S', time.localtime(ts))}  {txt}" for ts, txt in rows) or "Nothing yet.")

# ------------------------------------------------------------------ export bar (end of the conversation flow)
if not running and last and last["role"] == "assistant" and last.get("status") == "done":
    n_ex = sum(1 for m in chat if m["role"] == "user")
    has_fig = any(o["type"] in ("image", "process") for m in chat if m["role"] == "assistant" for o in m.get("outputs", []))
    with st.container(border=True):
        st.markdown("**Export**")
        with st.expander("What to include", expanded=True):
            scope_label = st.radio("Export", ["This picture / answer", "Our discussion"], index=0 if has_fig else 1, horizontal=True,
                                   key="exp_scope", label_visibility="collapsed")
            c1, c2 = st.columns([2, 1])
            whole = c2.checkbox("Whole chat", key="exp_all", value=False)
            n_pick = c1.number_input("How many of the last exchanges to include", min_value=1, max_value=max(1, n_ex), value=min(4, max(1, n_ex)),
                                     step=1, key="exp_n", disabled=whole or scope_label.startswith("This"),
                                     help="One exchange = one thing you asked and the answer. Used for discussion notes, slides and video of the discussion.")
        depth_label = st.radio("How much detail", ["Brief", "Standard", "Deep"], index=1, horizontal=True, key="exp_depth",
                               help="Brief: a few key lines. Standard: the main sections of every answer. Deep: every section, in full. "
                                    "Applies to the notes, the slides and the video, which always say the same thing.")
        ex_opts = {"scope": "figure" if scope_label.startswith("This") else "chat", "history": None if whole else int(n_pick),
                   "depth": depth_label.lower()}
        b = st.columns(4)
        if b[0].button("📄 PDF", key="ex_pdf", width="stretch"):
            go({"action": "pdf", "text": "", "shown_text": "Make a PDF", **ex_opts})
        if b[1].button("🖥 Slides", key="ex_sl", width="stretch", help="PDF slides and an editable PowerPoint"):
            go({"action": "slides", "text": "", "shown_text": "Make slides", **ex_opts})
        if b[2].button("🔊 Video", key="ex_vid", width="stretch"):
            go({"action": "lesson", "text": "", "shown_text": "Play the narrated lesson", **ex_opts})
        if b[3].button("📝 Notes", key="ex_nt", width="stretch", help="Revision notes of the discussion"):
            go({"action": "notes", "text": "", "mode_notes": "condensed", "shown_text": "Make notes of our discussion", "scope": "chat", "history": ex_opts["history"], "depth": ex_opts["depth"]})

# ------------------------------------------------------------------ settings row + prompt box, pinned together at the bottom
with st.bottom:
    s2, s3 = st.columns([1.1, 3.5])
    out_kind = s2.selectbox("Output", OUTPUTS, key="out_kind", disabled=running,
                            help="Auto: I work out what you want and tell you. Or pick exactly what to make.")
    s3.caption({"Auto": "I decide what you need from what you type, and tell you what I understood.",
                "Diagram": "I draw a labelled diagram of what you type.", "Process": "I show it step by step.",
                "Picture": "I make a picture from your description.", "Explain": "I explain the picture on screen, or answer your question.",
                "Edit": "I change the picture on screen as you describe (or attach a picture to change)."}[out_kind])
    prompt = st.chat_input("Ask a question, or say what to draw, explain, change or make…",
                           accept_file="multiple", file_type=["png", "jpg", "jpeg", "webp", "pdf", "docx", "txt", "md", "csv"], disabled=running)
if prompt:
    text = (prompt.text or "").strip()
    req = {"text": text, "attachments": attach(prompt.files), "options": {}, "shown_text": text}
    if out_kind == "Auto":
        req.update(mode="assistant", auto_note=True)
    elif out_kind == "Diagram":
        req.update(action="show_as", kind="diagram", mode="assistant")
    elif out_kind == "Process":
        req.update(action="process", mode="assistant")
    elif out_kind == "Picture":
        req.update(action="picture", mode="assistant")
    elif out_kind == "Explain":
        req.update(action="explain", mode="assistant")
    elif out_kind == "Edit":
        req.update(action="edit", mode="assistant")
    go(req)
