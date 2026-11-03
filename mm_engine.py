"""
mm_engine.py - the LangGraph engine of "Understood" (runs on the learner's PC, talks to the GPU server over HTTP).

Graph (one request = one pass):  prepare -> route -> run (repeats, one step at a time) -> finish
    prepare   resets the per-request work file, saves uploads
    route     rules first (free); if ambiguous the tutor decides (agentic); result is a short list of steps
    run       executes ONE step (learn_diagram, chem_molecule, chem_reaction, studio_image, edit_image, explain, chat,
              lesson, pdf, relabel) then loops until all steps are done
    finish    stores the conversation, compacts old history, hands the outputs back

Why a per-request work file?  If the connection drops in the middle of a drawing, the step raises, LangGraph keeps its
checkpoint, and resume() re-enters the same step.  The work file remembers the job already submitted (phase data), so
nothing is drawn twice and nothing is lost.

Everything shown to the learner passes through scrub_ui_text, and errors are generic: the backend is never revealed.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import sqlite3
import contextvars
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, TypedDict

import numpy as np
import requests
from PIL import Image

import mm_chem as K
import mm_core as C
import mm_doc
import mm_bank
import mm_info
import mm_slides
import mm_files
import mm_media
import mm_pdf
import mm_sources as S


# =====================================================================================================
# Server client
# =====================================================================================================
class ServerOffline(Exception):
    """Cannot reach the service (network, sleeping runtime...).  The run is parked and can be resumed."""


class ServerError(Exception):
    """The service answered but failed."""


class JobLost(Exception):
    """The service restarted and forgot the job."""


class Cancelled(Exception):
    """The learner pressed Stop."""


class VerifyUnavailable(Exception):
    """The service has no label-check endpoint (old server notebook).  Labels are then left off rather than guessed."""


class ServerClient:
    def __init__(self, url: str):
        self.url = (url or "").rstrip("/")

    def _req(self, method: str, path: str, timeout: int, **kw):
        if not self.url:
            raise ServerOffline("no server address")
        headers = {"ngrok-skip-browser-warning": "1", **(kw.pop("headers", None) or {})}    # free ngrok shows a warning page to browsers only
        try:
            r = requests.request(method, self.url + path, timeout=timeout, headers=headers, **kw)
        except requests.exceptions.SSLError as e:                       # a dead or expired tunnel address often fails the handshake
            raise ServerOffline("the service address looks expired or unreachable") from e
        except (requests.ConnectionError, requests.Timeout) as e:
            raise ServerOffline(str(e)) from e
        if r.status_code == 404 and path.startswith("/jobs/"):
            raise JobLost(path)
        if r.status_code in (502, 503, 504, 530):                       # tunnel up but nothing behind it
            raise ServerOffline(f"HTTP {r.status_code}")
        if r.status_code >= 400:
            raise ServerError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def health(self) -> dict:
        return self._req("GET", "/health", 10)

    def warmup(self) -> dict:
        return self._req("POST", "/warmup", 600)

    def tutor(self, messages: list[dict], max_new_tokens: int = 400, temperature: float = 0.0) -> tuple[str, str]:
        j = self._req("POST", "/tutor", 300, json={"messages": messages, "max_new_tokens": max_new_tokens, "temperature": temperature})
        return j["text"], j.get("finish_reason", "stop")

    def choose(self, items: list[dict], perms: int = 2, seed: int = 0) -> list[dict]:
        """Probability of every option for each item [{image_b64, question, options}] -> [{probs, mass}].  Sent in small batches."""
        out: list[dict] = []
        for k in range(0, len(items), 8):
            try:
                j = self._req("POST", "/choose", 300, json={"items": items[k:k + 8], "perms": perms, "seed": seed + k})
            except ServerError as e:
                if "404" in str(e) or "405" in str(e):
                    raise VerifyUnavailable(str(e)) from e
                raise
            out += j["results"]
        return out

    def submit_image(self, **payload) -> str:
        return self._req("POST", "/image", 60, json=payload)["job_id"]

    def job(self, job_id: str) -> dict:
        return self._req("GET", f"/jobs/{job_id}", 20)

    def cancel_job(self, job_id: str) -> None:
        """Ask the service to stop an image job.  Best effort: an old service without the endpoint just finishes the job unseen."""
        try:
            self._req("POST", f"/jobs/{job_id}/cancel", 10)
        except Exception:                                                        # noqa: BLE001
            pass


def img_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_img(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


# =====================================================================================================
# Settings and storage
# =====================================================================================================
class Settings:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "settings.json"
        self.d = {"server_url": ""}
        if self.path.exists():
            try:
                self.d.update(json.loads(self.path.read_text(encoding="utf-8")))
            except Exception:                                                    # noqa: BLE001
                pass

    @property
    def server_url(self) -> str:
        import os
        return os.environ.get("UNDERSTOOD_SERVER") or self.d.get("server_url", "")

    def set(self, **kw):
        self.d.update(kw)
        self.path.write_text(json.dumps(self.d), encoding="utf-8")


class Store:
    """Thread list + activity log (SQLite)."""

    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS threads(id TEXT PRIMARY KEY, title TEXT, ts REAL)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS activity(id INTEGER PRIMARY KEY AUTOINCREMENT, thread TEXT, ts REAL, text TEXT)")
            self.conn.commit()

    def new_thread(self) -> str:
        tid = uuid.uuid4().hex[:10]
        with self.lock:
            self.conn.execute("INSERT INTO threads VALUES(?,?,?)", (tid, "New chat", time.time()))
            self.conn.commit()
        return tid

    def title(self, tid: str, title: str):
        with self.lock:
            self.conn.execute("UPDATE threads SET title=? WHERE id=? AND title='New chat'", (title[:60], tid))
            self.conn.commit()

    def threads(self) -> list[tuple]:
        with self.lock:
            return self.conn.execute("SELECT id, title, ts FROM threads ORDER BY ts DESC LIMIT 50").fetchall()

    def log(self, tid: str, text: str):
        with self.lock:
            self.conn.execute("INSERT INTO activity(thread, ts, text) VALUES(?,?,?)", (tid, time.time(), text))
            self.conn.commit()

    def activity(self, tid: str, limit: int = 60) -> list[tuple]:
        with self.lock:
            rows = self.conn.execute("SELECT ts, text FROM activity WHERE thread=? ORDER BY id DESC LIMIT ?", (tid, limit)).fetchall()
        return rows[::-1]

    def all_threads(self) -> list[tuple]:
        """(id, created ts, newest activity ts or None) for every conversation, no limit - used by the retention purge."""
        with self.lock:
            return self.conn.execute("SELECT t.id, t.ts, (SELECT MAX(ts) FROM activity a WHERE a.thread = t.id) FROM threads t").fetchall()

    def forget(self, tid: str):
        with self.lock:
            self.conn.execute("DELETE FROM activity WHERE thread=?", (tid,))
            self.conn.execute("DELETE FROM threads WHERE id=?", (tid,))
            self.conn.commit()


# =====================================================================================================
# Graph state (persisted by the checkpointer between requests)
# =====================================================================================================
class State(TypedDict, total=False):
    thread_id: str
    request: dict                 # {text, level, mode, attachments:[ingested], action, options}
    messages: list                # [{role, content}] plain text history
    summary: str
    figure: Optional[dict]
    current_image: Optional[str]
    last_explanation: str
    docs: list                    # ingested documents of this conversation [{name, text}]
    error: str
    last_topic: str               # subject of the last diagram / process / guide (for "Show as" buttons)
    doc: Optional[dict]           # last notes / guide document {path, title, pdf}
    focus: str                    # "figure" or "doc": what "make a PDF / video" refers to


ASPECTS = {"square": (768, 768), "wide": (1024, 576), "tall": (576, 1024)}
HISTORY_BUDGET = 2400
COMPACT_AT = 3200


BUILD = "2026-10-29 stop+depth+bank-v3"


class Engine:
    def __init__(self, data_dir: Path, server_url: str):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, StateGraph

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "threads").mkdir(exist_ok=True)
        self.client = ServerClient(server_url)
        self.store = Store(self.data_dir / "understood.db")
        self.sources = S.SourceStore(self.data_dir / "sources.db")
        self.voice = mm_media.Voice(self.data_dir / "voices")
        self.on_event: Optional[Callable[[str], None]] = None
        self.online_sources = True
        self.bank = mm_bank.Bank(Path(os.environ.get("UNDERSTOOD_BANK_DIR") or (self.data_dir / "bank")), self.data_dir / "miss_log.jsonl")
        self.runs: dict = {}
        self.run_lock = threading.Lock()
        self.cancels: dict[str, threading.Event] = {}                  # one Stop flag per conversation
        self.pre: dict[str, dict] = {}                                 # what to put back if a run is stopped
        self._tid = contextvars.ContextVar("understood_tid", default=None)   # the conversation this call belongs to (follows into graph workers)

        conn = sqlite3.connect(str(self.data_dir / "checkpoints.db"), check_same_thread=False)
        self.ck_conn = conn
        self.saver = SqliteSaver(conn)
        self.retention_hours = float(os.environ.get("UNDERSTOOD_RETENTION_HOURS", "24"))     # 0 = keep everything
        g = StateGraph(State)
        g.add_node("prepare", self.n_prepare)
        g.add_node("route", self.n_route)
        g.add_node("run", self.n_run)
        g.add_node("finish", self.n_finish)
        g.add_edge(START, "prepare")
        g.add_edge("prepare", "route")
        g.add_edge("route", "run")
        g.add_conditional_edges("run", self.more_steps, {"run": "run", "finish": "finish"})
        g.add_edge("finish", END)
        self.graph = g.compile(checkpointer=self.saver)
        try:
            self.purge_old()                                             # at start-up ...
        except Exception:                                                # noqa: BLE001  (never block the app on housekeeping)
            pass
        threading.Thread(target=self._purge_loop, daemon=True).start()   # ... and every hour while the app runs

    # ---------------------------------------------------------------- public API
    def set_server(self, url: str):
        self.client = ServerClient(url)

    # ---------------------------------------------------------------- retention: conversations are deleted after a day
    # Only this code deletes (the tutor has no tool for it).  A conversation goes as a whole: messages, summary, pictures,
    # uploads, videos, PDFs, checkpoints and log rows - half-deleted history would corrupt the context.  Running and
    # pinned (".keep" file) conversations are skipped.  Files the learner downloaded are theirs and stay.
    def _purge_loop(self):
        while True:
            time.sleep(3600)
            try:
                self.purge_old()
            except Exception:                                            # noqa: BLE001
                pass

    def pin(self, tid: str, on: bool = True):
        keep = self.tdir(tid) / ".keep"
        if on:
            keep.write_text("keep", encoding="utf-8")
        elif keep.exists():
            keep.unlink()

    def last_activity(self, tid: str, created: float = 0.0, logged: Optional[float] = None) -> float:
        t = max(created or 0.0, logged or 0.0)
        d = self.data_dir / "threads" / tid
        if d.exists():
            try:
                t = max(t, d.stat().st_mtime)
                for p in d.rglob("*"):
                    t = max(t, p.stat().st_mtime)
            except OSError:
                pass
        return t

    def delete_thread(self, tid: str):
        base = (self.data_dir / "threads").resolve()
        d = (base / tid).resolve()
        if d.parent == base and d.exists():                              # never anything outside data/threads/<id>
            shutil.rmtree(d, ignore_errors=True)
        self.store.forget(tid)
        try:
            self.saver.delete_thread(tid)
        except Exception:                                                # noqa: BLE001  (older checkpoint package: delete the rows ourselves)
            for table in ("checkpoints", "writes"):
                try:
                    with self.saver.lock:
                        self.ck_conn.execute(f"DELETE FROM {table} WHERE thread_id=?", (tid,))
                        self.ck_conn.commit()
                except Exception:                                        # noqa: BLE001
                    pass
        self.runs.pop(tid, None)

    def prune_log(self, cutoff: float):
        p = self.data_dir / "naming_log.jsonl"
        if not p.exists():
            return
        keep = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("ts", 0) >= cutoff:
                    keep.append(line)
            except Exception:                                            # noqa: BLE001
                continue
        p.write_text("".join(x + "\n" for x in keep), encoding="utf-8")

    def purge_old(self, max_age_hours: Optional[float] = None, now: Optional[float] = None) -> list[str]:
        """Delete conversations whose last activity is older than the limit.  Returns the ids that were deleted."""
        hours = self.retention_hours if max_age_hours is None else max_age_hours
        if not hours or hours <= 0:
            return []
        cutoff = (now or time.time()) - hours * 3600
        known = {tid: (c, a) for tid, c, a in self.store.all_threads()}
        for p in (self.data_dir / "threads").iterdir():
            if p.is_dir():
                known.setdefault(p.name, (0.0, None))
        gone = []
        for tid, (created, logged) in known.items():
            if not re.fullmatch(r"[0-9a-f]{10}", tid):                   # only ids this app made
                continue
            run = self.runs.get(tid)
            if (run and run["thread"].is_alive()) or (self.data_dir / "threads" / tid / ".keep").exists():
                continue
            if self.last_activity(tid, created, logged) >= cutoff:
                continue
            self.delete_thread(tid)
            gone.append(tid)
        self.prune_log(cutoff)
        if gone:
            for conn, lock in ((self.store.conn, self.store.lock), (self.ck_conn, getattr(self.saver, "lock", None))):
                try:                                                     # give the freed space back and drop the deleted bytes
                    with (lock or threading.Lock()):
                        conn.execute("VACUUM")
                except Exception:                                        # noqa: BLE001
                    pass
        return gone

    # ---- Stop: end the run at the next safe point and put everything back as it was -----------------------------------
    def cancel(self, tid: str) -> bool:
        """Stop the running request.  The in-flight picture job is cancelled on the service; a tutor call that is already
        being answered finishes first (it cannot be interrupted half-way) and its answer is thrown away."""
        r = self.runs.get(tid)
        if not (r and r["thread"].is_alive()):
            return False
        self.cancels.setdefault(tid, threading.Event()).set()
        job = (self.work_get(tid).get("phase") or {}).get("job")
        if job:
            self.client.cancel_job(job)
        self.log(tid, "Stopping…")
        return True

    def _check(self, tid: Optional[str] = None):
        tid = tid or self._tid.get()
        if tid is None:                                                          # fallback: the only run in progress
            live = [t for t, r in self.runs.items() if r["thread"].is_alive()]
            tid = live[0] if len(live) == 1 else None
        ev = self.cancels.get(tid) if tid else None
        if ev is not None and ev.is_set():
            raise Cancelled()

    def _snapshot(self, tid: str):
        """Remember the conversation state before a request starts (only for a fresh request, not a resume)."""
        try:
            vals = dict(self.graph.get_state({"configurable": {"thread_id": tid}}).values or {})
        except Exception:                                                        # noqa: BLE001
            vals = {}
        self.pre[tid] = {"state": vals, "work": self.work_get(tid)}

    def _revert(self, tid: str):
        pre = self.pre.get(tid)
        if pre is None:                                                          # app restarted mid-run: keep the state, drop the half-done work only
            self.work_put(tid, {"outputs": [], "steps": [], "i": 0, "phase": {}})
            try:
                self.graph.update_state({"configurable": {"thread_id": tid}}, {}, as_node="finish")
            except Exception:                                                    # noqa: BLE001
                pass
            return
        blank = {"thread_id": tid, "request": {}, "messages": [], "summary": "", "figure": None, "current_image": None, "last_explanation": "",
                 "docs": [], "error": "", "last_topic": "", "doc": None, "focus": ""}
        vals = {**blank, **pre["state"]}
        try:
            self.graph.update_state({"configurable": {"thread_id": tid}}, vals, as_node="finish")   # nothing is left waiting to resume
        except Exception:                                                        # noqa: BLE001
            pass
        self.work_put(tid, {"outputs": [], "steps": [], "i": 0, "phase": {}})
        r = self.runs.get(tid) or {}
        if r.get("chat_len") is not None:
            self.chat_put(tid, self.chat_get(tid)[: r["chat_len"]])              # the question being answered goes too

    def run(self, tid: str, request: dict, on_event: Optional[Callable[[str], None]] = None) -> dict:
        self.on_event = on_event
        self._snapshot(tid)
        self.store.title(tid, request.get("text") or "New chat")
        return self._invoke(tid, {"thread_id": tid, "request": request})

    def resume(self, tid: str, on_event: Optional[Callable[[str], None]] = None) -> dict:
        self.on_event = on_event
        return self._invoke(tid, None)

    # ---- background runs: the page stays usable while a request is being worked on ------------------------------
    def chat_path(self, tid: str) -> Path:
        return self.tdir(tid) / "chat.json"

    def bank_status(self) -> dict:
        self.bank._load()
        return {"build": BUILD, "bank_dir": str(self.bank.root), "approved_topics": sorted(set(self.bank._keys.values())), "bank_on": self.bank.enabled()}

    def chat_get(self, tid: str) -> list:
        try:
            return json.loads(self.chat_path(tid).read_text(encoding="utf-8"))
        except Exception:                                                        # noqa: BLE001
            return []

    def chat_put(self, tid: str, msgs: list):
        self.chat_path(tid).write_text(json.dumps(msgs), encoding="utf-8")

    def start(self, tid: str, request: Optional[dict]) -> bool:
        """Begin (or resume, when request is None) a run in a background thread.  False if one is already running."""
        with self.run_lock:
            r = self.runs.get(tid)
            if r and r["thread"].is_alive():
                return False
            msgs = self.chat_get(tid)
            chat_len = len(msgs)
            if request is None and (self.runs.get(tid) or {}).get("chat_len") is not None:
                chat_len = self.runs[tid]["chat_len"]                            # a resumed run still undoes back to the original question
            if request is None:
                if msgs and msgs[-1]["role"] == "assistant" and msgs[-1].get("status") != "done":
                    msgs.pop()
                    self.chat_put(tid, msgs)
            else:
                files = [a["name"] for a in request.get("attachments") or []]
                shown = request.get("shown_text", request.get("text", ""))
                if shown or files:
                    msgs.append({"role": "user", "text": shown, "files": files})
                    self.chat_put(tid, msgs)
            t = threading.Thread(target=self._bg, args=(tid, request), daemon=True)
            self.cancels[tid] = threading.Event()
            self.runs[tid] = {"thread": t, "result": None, "started": time.time(), "chat_len": chat_len}
            t.start()
            return True

    def _bg(self, tid: str, request: Optional[dict]):
        res = self.resume(tid) if request is None else self.run(tid, request)
        w = self.work_get(tid)
        self.runs[tid]["result"] = res
        if res["status"] == "cancelled":
            return                                                               # nothing is added: the chat is exactly as before
        if res["outputs"] or res["status"] != "done":
            msgs = self.chat_get(tid)
            msgs.append({"role": "assistant", "outputs": res["outputs"], "status": res["status"], "topic": w.get("topic", "")})
            self.chat_put(tid, msgs)

    def status(self, tid: str) -> str:
        r = self.runs.get(tid)
        if r and r["thread"].is_alive():
            return "running"
        if r and r["result"]:
            return r["result"]["status"]
        return "idle"

    def live(self, tid: str) -> dict:
        w = self.work_get(tid)
        return {"outputs": w.get("outputs", []), "steps": w.get("steps", []), "i": w.get("i", 0), "log": self.store.activity(tid, 40)}

    def parked(self, tid: str) -> bool:
        try:
            st = self.graph.get_state({"configurable": {"thread_id": tid}})
            return bool(st and st.next)
        except Exception:                                                        # noqa: BLE001
            return False

    def state(self, tid: str) -> dict:
        try:
            return dict(self.graph.get_state({"configurable": {"thread_id": tid}}).values or {})
        except Exception:                                                        # noqa: BLE001
            return {}

    def _invoke(self, tid: str, inp) -> dict:
        cfg = {"configurable": {"thread_id": tid}, "recursion_limit": 40}
        self._tid.set(tid)
        try:
            self._check(tid)
            self.graph.invoke(inp, cfg)
            w = self.work_get(tid)
            return {"status": "done", "outputs": w.get("outputs", []), "route": w.get("steps", [])}
        except Cancelled:
            self._revert(tid)
            self.log(tid, "Stopped - I undid that request, so nothing was changed.")
            return {"status": "cancelled", "outputs": [], "message": "stopped"}
        except (ServerOffline, JobLost) as e:
            self.log(tid, "Connection lost - your request is saved. Press Resume when you are back online.")
            return {"status": "parked", "outputs": self.work_get(tid).get("outputs", []), "message": "offline"}
        except ServerError as e:
            self.log(tid, "The learning service had a problem with this request.")
            return {"status": "failed", "outputs": self.work_get(tid).get("outputs", []), "message": "server"}
        except Exception as e:                                                   # noqa: BLE001
            self.log(tid, "Something went wrong with this request.")
            return {"status": "failed", "outputs": self.work_get(tid).get("outputs", []), "message": f"{type(e).__name__}: {e}"}

    # ---------------------------------------------------------------- helpers: files, work, log
    def tdir(self, tid: str) -> Path:
        d = self.data_dir / "threads" / tid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, p: Path) -> str:
        return str(Path(p).relative_to(self.data_dir)).replace("\\", "/")

    def abs(self, rel: str) -> Path:
        return self.data_dir / rel

    def work_get(self, tid: str) -> dict:
        p = self.tdir(tid) / "work.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                                        # noqa: BLE001
            return {"outputs": [], "steps": [], "i": 0, "phase": {}}

    def work_put(self, tid: str, w: dict):
        (self.tdir(tid) / "work.json").write_text(json.dumps(w), encoding="utf-8")

    def log(self, tid: str, text: str):
        text = C.scrub_ui_text(text)
        self.store.log(tid, text)
        if self.on_event:
            try:
                self.on_event(text)
            except Exception:                                                    # noqa: BLE001
                pass

    def save_img(self, tid: str, img: Image.Image, stem: str) -> str:
        p = self.tdir(tid) / f"{stem}_{uuid.uuid4().hex[:6]}.png"
        img.convert("RGB").save(p)
        return self.rel(p)

    # ---------------------------------------------------------------- helpers: tutor
    def ask(self, task: str, system: str, user: str, images: Optional[list[Image.Image]] = None,
            history: Optional[list[dict]] = None, max_tokens: int = 300, temperature: float = 0.0) -> tuple[str, str]:
        self._check()
        msgs = [{"role": "system", "content": f"TASK:{task}\n{system}"}] + list(history or [])
        content = [{"type": "image", "b64": img_b64(im)} for im in (images or [])] + [{"type": "text", "text": user}]
        msgs.append({"role": "user", "content": content if images else user})
        out = self.client.tutor(msgs, max_tokens, temperature)
        self._check()                                                            # Stop pressed during the call: drop its answer
        return out

    # ---------------------------------------------------------------- nodes
    def n_prepare(self, state: State) -> dict:
        tid, req = state["thread_id"], dict(state["request"])
        saved = []
        docs = list(state.get("docs") or [])
        for a in req.get("attachments") or []:
            if a.get("data_b64"):
                info = mm_files.ingest(a["name"], base64.b64decode(a["data_b64"]), self.tdir(tid) / "uploads")
                info["path"] = self.rel(Path(info["path"])) if info.get("path") else None
                info["pages"] = [self.rel(Path(p)) for p in info.get("pages", [])]
                saved.append(info)
                if info["kind"] in ("pdf", "docx", "text") and info.get("text"):
                    docs.append({"name": info["name"], "text": info["text"]})
                self.log(tid, f"Received {info['name']}" + (f" ({info['note']})" if info.get("note") else ""))
            elif a.get("path"):
                saved.append(a)
        req["attachments"] = saved
        self.work_put(tid, {"outputs": [], "steps": [], "i": 0, "phase": {}, "request": req,
                            "level": req.get("level") or C.DEFAULT_LEVEL})
        return {"request": {k: v for k, v in req.items() if k != "attachments"} | {"n_attach": len(saved)}, "docs": docs[-6:]}

    def ctx(self, state: State, req: dict, w: dict) -> C.RouteCtx:
        fig = state.get("figure")
        extra = dict(has_doc=bool(state.get("doc")), has_history=bool(state.get("messages") or state.get("summary") or self.chat_get(state["thread_id"])))
        atts = w["request"].get("attachments", [])
        return C.RouteCtx(
            mode=req.get("mode", "assistant"), has_figure=bool(fig), has_explanation=bool(state.get("last_explanation")),
            image_attachments=sum(1 for a in atts if a["kind"] == "image"),
            doc_attachments=sum(1 for a in atts if a["kind"] in ("pdf", "docx", "text")) + (1 if (state.get("docs") and not atts and False) else 0),
            has_current_image=bool(state.get("current_image") or fig),
            part_names=[p["name"] for p in (fig or {}).get("parts", [])], known_molecule=K.known_molecule, **extra)

    def n_route(self, state: State) -> dict:
        tid = state["thread_id"]
        w = self.work_get(tid)
        req = w["request"]
        text = (req.get("text") or "").strip()
        ctx = self.ctx(state, req, w)
        action = req.get("action")
        steps = None
        if action in ("pdf", "lesson", "slides"):
            steps = [{"tool": action, "args": {}}]
        elif action == "explain":                                # the Output chooser: Explain
            steps = [{"tool": "explain", "args": {"question": text}}] if state.get("figure") else [{"tool": "chat", "args": {}}]
        elif action == "edit" and text:                          # the Output chooser: Edit the picture on screen
            steps = [{"tool": "edit_image", "args": {"instruction": text}}]
        elif action == "notes":
            steps = [{"tool": "notes", "args": {"mode": req.get("mode_notes", "condensed")}}]
        elif action == "guide":
            steps = [{"tool": "guide", "args": {"topic": text}}]
        elif action == "picture":
            steps = [{"tool": "studio_image", "args": {"prompt": text}}] if text else None
        elif action == "process":
            steps = [{"tool": "learn_process", "args": {"topic": text}}] if text else None
        elif action == "show_as":
            topic = text or state.get("last_topic") or ""          # what the learner just typed always wins over the previous topic
            kind = req.get("kind")
            if kind == "process":
                steps = [{"tool": "learn_process", "args": {"topic": topic}}]
            elif kind == "diagram":
                steps = [{"tool": "learn_diagram", "args": {"topic": topic}}, {"tool": "explain", "args": {}}]
            else:
                steps = [{"tool": "studio_image", "args": {"prompt": topic}}]
        elif action == "retry_step":
            steps = [{"tool": "retry_step", "args": {"n": req.get("n", 0)}}]
        elif action == "variation":
            steps = [{"tool": "studio_image", "args": {"prompt": req.get("text", "")}}]
        else:
            steps = C.rule_route(text, ctx)
        if steps is None:
            self.log(tid, "Working out what you need…")
            steps = self.tutor_route(state, text, ctx)
        if not steps:
            steps = [{"tool": "chat", "args": {}}]
        lvl = req.get("level_explicit") and req.get("level") or C.level_from_text(text) or req.get("level") or C.DEFAULT_LEVEL
        w["level"], w["steps"], w["i"] = lvl, steps, 0
        if req.get("auto_note") and not action and steps and steps[0]["tool"] != "chat":
            self.out(tid, w, type="understood", text=C.describe_plan(steps))
        self.work_put(tid, w)
        self.log(tid, "Plan: " + ", ".join(s["tool"].replace("_", " ") for s in steps))
        return {}

    def route_context(self, state: State, ctx: C.RouteCtx) -> str:
        """What is on screen and what was just said, so 'it', 'that' and 'again' can be resolved."""
        bits = []
        fig = state.get("figure")
        if fig:
            bits.append(f"On screen: a {fig.get('kind')} - {fig.get('title')}" + (f" with labels: {', '.join(ctx.part_names)}" if ctx.part_names else ""))
        elif state.get("current_image"):
            bits.append("On screen: a picture")
        if ctx.image_attachments:
            bits.append(f"The learner attached {ctx.image_attachments} picture(s).")
        if ctx.doc_attachments:
            bits.append(f"The learner attached {ctx.doc_attachments} document(s).")
        recent = self.history(state)[-4:]
        if recent:
            bits.append("Recent turns:\n" + "\n".join(f"{m['role']}: {m['content'][:160]}".replace("\n", " ") for m in recent))
        return "\n".join(bits)

    def tutor_route(self, state: State, text: str, ctx: C.RouteCtx) -> list[dict]:
        sysmsg = ("Decide what the learner wants. Use what is on screen and the recent turns to understand words like 'it', 'that' or 'again'. A request that names a NEW subject is never an edit of what is on screen. "
                  "A question about what is on screen is explain or chat; a request to change the picture on screen is edit_image. "
                  "Reply with JSON only: {\"steps\":[{\"tool\":NAME,\"args\":{...}}],\"level\":null}. "
                  "Tools: learn_diagram(topic) labelled science diagram of a biology or chemistry apparatus/structure; "
                  "chem_molecule(query) molecular structure; chem_reaction(text) balanced equation picture; "
                  "learn_process(topic) step-by-step infographic of a process or cycle; "
                  "studio_image(prompt) any other picture; edit_image(instruction) change an existing picture; "
                  "explain(question); chat(); lesson() narrated video; pdf(); notes() notes of this conversation; "
                  "guide(topic) study guide with practice questions; clarify(topic) when unsure whether a diagram, a process or a picture is wanted; "
                  "relabel(op,old,new). Use at most 3 steps.")
        try:
            context = self.route_context(state, ctx)
            ans, _ = self.ask("ROUTE", sysmsg, (context + "\n" if context else "") + f"Request: {text}", max_tokens=200)
            steps = C.validate_route(C.parse_json(ans), ctx, text)
        except (ServerOffline, JobLost):
            raise
        except Exception:                                                        # noqa: BLE001
            steps = None
        if steps:
            return steps
        topic = C.extract_topic(text)
        if C.RE_DRAW_START.search(text) and topic:
            return [{"tool": "clarify", "args": {"topic": topic}}]
        return [{"tool": "chat", "args": {}}]

    def more_steps(self, state: State) -> str:
        w = self.work_get(state["thread_id"])
        return "run" if (w["i"] < len(w["steps"]) and not w.get("abort")) else "finish"

    def n_run(self, state: State) -> dict:
        tid = state["thread_id"]
        self._check(tid)
        w = self.work_get(tid)
        step = w["steps"][w["i"]]
        fn = getattr(self, "t_" + step["tool"], None)
        upd: dict = {}
        if fn is None:
            w["i"] += 1
            self.work_put(tid, w)
            return {}
        upd = fn(state, step.get("args", {}), w) or {}
        w = self.work_get(tid)                                  # tools write their own progress into the work file
        tool, targs = step["tool"], step.get("args", {})
        if tool in ("learn_diagram", "learn_process", "guide") and targs.get("topic"):
            upd.setdefault("last_topic", targs["topic"])
            w["topic"] = targs["topic"]
        elif tool in ("chem_molecule", "chem_reaction") and (targs.get("query") or targs.get("text")):
            w["topic"] = targs.get("query") or targs.get("text")
        if tool in ("learn_diagram", "learn_process", "chem_molecule", "chem_reaction", "retry_step"):
            upd["focus"] = "figure"
        elif tool in ("guide", "notes"):
            upd["focus"] = "doc"
        w["i"] += 1
        w["phase"] = {}
        self.work_put(tid, w)
        return upd

    def n_finish(self, state: State) -> dict:
        tid = state["thread_id"]
        w = self.work_get(tid)
        req = w["request"]
        texts = [o["text"] for o in w["outputs"] if o["type"] == "text"]
        msgs = list(state.get("messages") or [])
        user_line = (req.get("text") or "").strip() or "(sent a file)"
        msgs.append({"role": "user", "content": user_line[:1200]})
        msgs.append({"role": "assistant", "content": (" ".join(texts) or "Here is your result.")[:1500]})
        summary = state.get("summary", "")
        if sum(C.est_tokens(m["content"]) for m in msgs) > COMPACT_AT and len(msgs) > 6:
            old, keep = msgs[:-4], msgs[-4:]
            try:
                transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
                summary, _ = self.ask("SUMMARY", "Summarise this tutoring conversation in under 120 words. Keep names, topics and decisions.",
                                      (summary + "\n" + transcript)[-6000:], max_tokens=220)
                msgs = keep
            except (ServerOffline, ServerError, JobLost):
                pass
        self.log(tid, "Done.")
        return {"messages": msgs, "summary": summary}

    # ---------------------------------------------------------------- helpers: output
    def out(self, tid: str, w: dict, **part):
        w.setdefault("outputs", []).append(part)
        self.work_put(tid, w)

    def history(self, state: State) -> list[dict]:
        out, used = [], 0
        for m in reversed(state.get("messages") or []):
            t = C.est_tokens(m["content"])
            if used + t > HISTORY_BUDGET:
                break
            out.append(m)
            used += t
        hist = out[::-1]
        if state.get("summary"):
            hist = [{"role": "user", "content": "Earlier in this conversation: " + state["summary"]},
                    {"role": "assistant", "content": "Understood."}] + hist
        return hist

    def run_image_job(self, tid: str, w: dict, label: str, payload: dict) -> Image.Image:
        """Submit once, then poll.  The job id lives in the work file so a resume continues instead of resubmitting."""
        ph = w.setdefault("phase", {})
        if not ph.get("job"):
            self.log(tid, f"{label}…")
            ph["job"] = self.client.submit_image(**payload)
            ph["t0"] = time.time()
            self.work_put(tid, w)
        last = None
        while True:
            if (ev := self.cancels.get(tid)) is not None and ev.is_set():
                self.client.cancel_job(ph["job"])
                raise Cancelled()
            try:
                j = self.client.job(ph["job"])
            except JobLost:
                ph.pop("job", None)
                self.work_put(tid, w)
                raise
            if j["status"] == "done":
                self.log(tid, f"{label} - done in {int(time.time() - ph['t0'])}s")
                return b64_img(j["image_b64"])
            if j["status"] == "cancelled":
                raise Cancelled()
            if j["status"] == "error":
                ph.pop("job", None)
                self.work_put(tid, w)
                raise ServerError(j.get("error", "job failed"))
            msg = f"{label}… {j.get('progress', 0)}/{j.get('steps', 1)}" if j["status"] == "running" else f"{label}… waiting"
            if msg != last:
                self.log(tid, msg)
                last = msg
            time.sleep(1.5)
            if time.time() - ph["t0"] > 1500:
                raise ServerError("timeout")

    def section_answer(self, task: str, system: str, user: str, expected: list[str], images=None, history=None, max_tokens=520) -> str:
        ans, fin = self.ask(task, system, user, images=images, history=history, max_tokens=max_tokens, temperature=0.2)
        if fin == "length":
            ans = C.trim_to_sentence(ans)
        if C.has_sections(ans, expected):
            return C.sections_markdown(ans, expected)
        ans2, _ = self.ask("REFORMAT", "Rewrite the text under exactly these markdown headings, in this order: "
                           + ", ".join(f"## {h}" for h in expected) + ". Keep all the facts, add nothing.", ans, max_tokens=max_tokens)
        return C.sections_markdown(ans2, expected) if C.has_sections(ans2, expected) else C.wrap_as_sections(ans, expected[0])

    def level_of(self, w: dict) -> dict:
        return C.get_level(w.get("level"))

    # ================================================================ TOOLS
    # ---- learn_diagram -------------------------------------------------------------------------------------
    def build_diagram(self, tid: str, w: dict, topic: str, lv: dict, quiet: bool = False, prefer: str = "") -> Optional[tuple]:
        """Notes -> parts -> picture -> verified naming.  Returns (fig, verified, ph) or None (the learner was already told why).
        Shared by the labelled diagram and by processes, so both stand on exactly the same verified picture."""
        ph = w.setdefault("phase", {})
        if "parts" not in ph:
            hit = self.bank.lookup(topic)                               # the diagram bank first: a real, already-labelled drawing
            if hit:
                self.log(tid, f"Bank: using the reference drawing '{hit['title']}' ({hit['id']}) - no generator needed")
            elif self.bank.enabled():
                self.log(tid, f"Bank: nothing for '{topic}' - it will be drawn instead")
            else:
                self.log(tid, f"Bank: not set up (no approved pictures in {self.bank.root}) - drawing it")
            self.log(tid, f"Looking up reliable notes on {topic}")
            src = S.fetch_topic(topic, lv["key"], self.sources, self.online_sources)
            if not src and not hit:
                if self.bank.enabled():
                    self.bank.miss(topic)
                if not quiet:
                    self.out(tid, w, type="note", text="I could not find reliable notes on that topic. Try a more common name, or tell me which parts you need.")
                    w["abort"] = True
                    self.work_put(tid, w)
                return None
            if hit:
                bank_names = [p_["name"] for p_ in hit["parts"] if p_.get("anchor")]
                cap = lv["max_parts"] + (4 if prefer else 0)
                mapping: dict = {}
                if src and src.get("curated"):
                    sentences = dict(src["sentences"])
                    order = list(sentences) if prefer else list(src["parts"])
                    if prefer:
                        pri = C.mentioned_parts(prefer, order)
                        order = pri + [p_ for p_ in order if p_ not in pri]
                    for cp in order:
                        bm, _ = C.match_part(cp, bank_names)
                        if bm and bm not in mapping.values():
                            mapping[cp] = bm
                else:
                    sentences = {}
                if len(mapping) < 3:                                     # notes that name few of the drawn parts: show the drawing's own labels
                    for bn in bank_names:
                        if bn not in mapping.values() and len(mapping) < cap:
                            mapping[bn] = bn
                parts = list(mapping)[:cap]
                mapping = {p_: mapping[p_] for p_ in parts}
                if len(parts) < 2:
                    hit = None
            if hit:
                base_note = f"Source: {src.get('source') or 'notes'}" if src else "Source: reference diagram"
                ph.update(parts=parts, title=hit["title"], sentences=sentences, source_text=(src["text"][:4000] if src else ""),
                          source_note=f"{base_note}. {hit['credit']}", prompt="", seed=0,
                          bank={"id": hit["id"], "map": mapping})
                self.work_put(tid, w)
            else:
                if self.bank.enabled():
                    self.bank.miss(topic)
                if not src:
                    if not quiet:
                        self.out(tid, w, type="note", text="I could not find reliable notes on that topic. Try a more common name, or tell me which parts you need.")
                        w["abort"] = True
                        self.work_put(tid, w)
                    return None
                if src.get("curated"):
                    parts, sentences = src["parts"], src["sentences"]
                    if prefer:                                          # a process needs ITS parts on the drawing: those named in it come first
                        allp = list(sentences)
                        pri = C.mentioned_parts(prefer, allp)            # in the order the process names them
                        parts = (pri + [p for p in allp if p not in pri])[:lv["max_parts"] + 4]
                else:
                    self.log(tid, "Choosing the parts to show")
                    ans, _ = self.ask("PARTS", f"List the main visible structures of the topic as a JSON array of short names (max {lv['max_parts']}). "
                                      "Use only names that appear in the text.", f"Topic: {topic}\nText: {src['text'][:3500]}", max_tokens=160)
                    parts = C.verify_parts(C.parse_parts_list(ans), src["text"], lv["max_parts"])
                    sentences = {}
                if len(parts) < 2:
                    if not quiet:
                        self.out(tid, w, type="note", text="The notes I found did not name enough clear parts to draw. Try a more specific topic.")
                        w["abort"] = True
                        self.work_put(tid, w)
                    return None
                ph.update(parts=parts, title=src.get("title") or topic, sentences=sentences, source_text=src["text"][:4000],
                          source_note=f"Source: {src.get('source') or 'notes'}" + (f" ({src['license']})" if src.get("license") and src.get("url") else ""),
                          prompt=C.build_diagram_prompt(topic, parts, lv["key"]), seed=int(time.time()) % 100000)
                self.work_put(tid, w)
        if "ref_done" not in ph:                                       # a reference picture helps the tutor see what the parts look like
            ph["ref"] = None                                            # no reference-picture check any more: it cost time and did not improve labelling
            ph["ref_done"] = True
            self.work_put(tid, w)
        if ph.get("bank") and not ph.get("img_path"):                  # bank picture: no generator call at all
            rec = self.bank.lookup(topic) or json.loads((self.bank.root / f"{ph['bank']['id']}.json").read_text(encoding="utf-8"))
            rec["png"] = str(self.bank.root / f"{ph['bank']['id']}.png")
            bimg, bpts = mm_bank.prepare(rec, lv["gen_size"])
            ph["img_path"] = self.save_img(tid, bimg, "raw")
            ph["bank"]["anchors"] = {k: list(v) for k, v in bpts.items()}
            self.work_put(tid, w)
        if not ph.get("img_path"):
            img = self.run_image_job(tid, w, "Drawing the diagram", dict(mode="generate", prompt=ph["prompt"], width=lv["gen_size"], height=lv["gen_size"],
                                                                     quality="fast", seed=ph["seed"]))
            ph["img_path"] = self.save_img(tid, img, "raw")            # kept on this computer: a resume never needs the service's copy
            self.work_put(tid, w)
        img = Image.open(self.abs(ph["img_path"])).convert("RGB")
        self.log(tid, "Locating the parts")
        parts = ph["parts"]
        assign: dict = {}
        info: dict = {}
        part_anchor: dict = {}
        verified = True
        if ph.get("bank"):                                              # labels come from the source drawing; code only places them
            bmap = ph["bank"]["map"]
            anchors = {p: tuple(ph["bank"]["anchors"][bmap[p]]) for p in parts if bmap.get(p) in ph["bank"]["anchors"]}
            regions, lmap, placed = mm_bank.bank_regions(img, anchors)       # regions from the source colours ...
            img = mm_bank.tint(img)                                           # ... then lighter colours for display
            for p, (rid, pt) in placed.items():
                assign[p], part_anchor[p] = rid, pt
                info[p] = {"p": 1.0, "how": "from the source drawing"}
            self.log(tid, f"Placed {len(assign)} labels from the reference drawing")
        else:
            regions, lmap = C.find_regions(img)
        if regions and not ph.get("bank"):
            try:
                assign, info = self.locate_parts(tid, w, img, regions, ph)
                parts = ph["parts"]                                      # may now include parts the tutor found that the notes did not list
            except (VerifyUnavailable, ServerError):
                verified = False
                self.log(tid, "Reading the drawing failed, so no labels are drawn")
        by_id = {r["id"]: r for r in regions}
        items, fparts = [], []
        for i, p in enumerate(parts):
            col = C.colour_for(i)
            rid = assign.get(p)
            sent = ph["sentences"].get(p) or C.sentence_for(p, ph["source_text"], lv["narration_words"])
            if rid:
                a = part_anchor.get(p) or by_id[rid]["anchor"]
                items.append({"name": p, "anchor": (a[0], a[1]), "colour": col})
            fparts.append({"name": p, "colour": list(col), "anchor": [float((part_anchor.get(p) or by_id[rid]["anchor"])[0]), float((part_anchor.get(p) or by_id[rid]["anchor"])[1])] if rid else None,
                           "region_id": int(rid) if rid else None, "text": sent, "visible": bool(rid),
                           "confidence": (round(info[p]["p"], 2) if rid and p in info and info[p].get("p") is not None else None), "how": info[p]["how"] if rid and p in info else ""})
        clean = self.save_img(tid, img, "clean")
        lm = self.tdir(tid) / f"labelmap_{uuid.uuid4().hex[:6]}.png"
        Image.fromarray(lmap.astype(np.uint8), "L").save(lm)
        fig = {"kind": "diagram", "title": ph["title"].title() if ph["title"].islower() else ph["title"], "level": lv["key"], "clean_path": clean,
               "labelmap_path": self.rel(lm), "parts": fparts, "source_text": ph["source_text"], "source_note": ph["source_note"], "facts": []}
        return fig, verified, ph

    def t_learn_diagram(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        lv = self.level_of(w)
        topic = (args.get("topic") or w["request"].get("text") or "").strip()
        built = self.build_diagram(tid, w, topic, lv)
        if built is None:
            return {}
        fig, verified, ph = built
        fparts, parts = fig["parts"], ph["parts"]
        labelled = self.render_diagram(tid, fig)
        shown = [p["name"] for p in fparts if p["visible"]]
        missing = [p["name"] for p in fparts if not p["visible"]]
        self.out(tid, w, type="image", path=labelled, caption=f"{fig['title']} - {lv['label']}")
        if not verified:
            self.out(tid, w, type="note", text="I could not double-check the labels just now, so I left them off rather than risk a wrong one. Ask me to draw it again in a moment.")
        elif missing:
            self.out(tid, w, type="note", text="Not clearly shown in this drawing, so not labelled: " + ", ".join(missing) + ". You can ask for a new version.")
        if verified and not shown:
            self.out(tid, w, type="note", text="I could not match the parts to this drawing. Try 'draw it again'.")
        self.out(tid, w, type="sources", items=[ph["source_note"]] + ([S.credit_line(ph["ref"]).replace("Reference picture:", "Checked against a reference picture:", 1)] if ph.get("ref") else []))
        self.log(tid, f"Labelled {len(shown)} of {len(parts)} parts")
        w["figure"] = True
        self.work_put(tid, w)
        return {"figure": fig, "current_image": labelled, "last_explanation": ""}

    # ---- locating the parts of a GENERATED drawing --------------------------------------------------------------
    # The drawing is already segmented (code) and drawn in light colours.  The tutor looks at the whole picture ONCE, with a small numbered
    # circle on every region, and names each numbered structure in its own words.  No part list, no per-region questions, no checking
    # loops: a name is used as given.  Code only (a) maps a name onto the notes' part name when it is a synonym, so the explanation
    # sentences attach, (b) lets the bigger region keep a name the tutor gave twice, (c) drops names that mean "nothing".
    def locate_parts(self, tid: str, w: dict, img: Image.Image, regions: list[dict], ph: dict) -> tuple[dict, dict]:
        title = ph["title"]
        parts = list(ph["parts"])
        got = ph.get("located")
        if got is None:
            self.log(tid, "Reading the drawing")
            marked = C.draw_marks(img, regions)
            marked.thumbnail((768, 768))
            ans, _ = self.ask("LOCATE", "You are a science tutor looking at a textbook diagram. Small white circles with numbers mark distinct regions of the drawing.",
                              f"This is a diagram of {title}. Look at the whole picture first: the shape, colour and position of each structure. Then, for every numbered circle, "
                              "write the name of the structure it sits on, the way a textbook would label it (1 to 4 words). Use null for a circle that is not on a distinct structure. "
                              "Reply with JSON only, like {\"1\": \"name\", \"2\": null}.",
                              images=[marked], max_tokens=16 * len(regions) + 40)
            got = {}
            obj = C.parse_json(ans)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    m = re.match(r"\s*(\d+)", str(k))
                    if m and isinstance(v, str):
                        got[m.group(1)] = C.parse_region_name(v)
            ph["located"] = got
            self.work_put(tid, w)
        area = {r["id"]: r["area"] for r in regions}
        assign: dict = {}
        info: dict = {}
        for rid_s, name in sorted(got.items(), key=lambda kv: -area.get(int(kv[0]), 0)):      # the bigger region keeps a name given twice
            rid = int(rid_s)
            if rid not in area or not name:
                continue
            part, _ = C.match_part(name, parts)
            if not part:
                part = re.sub(r"\s+", " ", name.lower()).strip()
                if part in assign:
                    continue
                if part not in parts:
                    parts.append(part)
            if part in assign:
                continue
            assign[part] = rid
            info[part] = {"p": None, "how": "named by the tutor from the picture", "free": name}
        ph["parts"] = parts
        self.log_naming(ph, regions, got, assign, parts)
        self.log(tid, f"Located {len(assign)} parts")
        return assign, info

    def log_naming(self, ph: dict, regions: list[dict], got: dict, assign: dict, parts: list[str]):
        """One JSON line per diagram: what the tutor said for every region and what was used."""
        try:
            by_part = {rid: p for p, rid in assign.items()}
            rec = {"ts": time.time(), "title": ph.get("title"), "parts": parts,
                   "regions": [{"id": r["id"], "area": r["area"], "anchor": [round(r["anchor"][0]), round(r["anchor"][1])],
                                "name": got.get(str(r["id"])), "final": by_part.get(r["id"])} for r in regions],
                   "not_located": [p for p in parts if p not in assign]}
            with open(self.data_dir / "naming_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:                                                        # noqa: BLE001
            pass

    def render_diagram(self, tid: str, fig: dict, titled: bool = True) -> str:
        clean = Image.open(self.abs(fig["clean_path"])).convert("RGB")
        items = [{"name": p["name"], "anchor": tuple(p["anchor"]), "colour": tuple(p["colour"])} for p in fig["parts"] if p["visible"] and p["anchor"]]
        if not items:
            return self.save_img(tid, C.add_title(clean, fig["title"]) if titled else clean, "labelled")
        layout = C.layout_labels(clean.size, items, title=fig["title"] if titled else None)
        return self.save_img(tid, C.render_labels(clean, layout), "labelled")

    # ---- relabel -------------------------------------------------------------------------------------------
    def t_relabel(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        fig = json.loads(json.dumps(state.get("figure") or {}))
        if fig.get("kind") != "diagram":
            self.out(tid, w, type="note", text="There is no labelled diagram to change yet.")
            return {}
        key = C._norm(args.get("old", ""))
        for p in fig["parts"]:
            if C._norm(p["name"]) == key:
                if args.get("op") == "rename" and args.get("new"):
                    p["name"] = args["new"][:40]
                else:
                    p["visible"] = False
                break
        else:
            self.out(tid, w, type="note", text=f"I could not find a label called '{args.get('old', '')}'.")
            return {}
        path = self.render_diagram(tid, fig)
        self.out(tid, w, type="image", path=path, caption=fig["title"])
        self.log(tid, "Updated the labels")
        return {"figure": fig, "current_image": path}

    # ---- chemistry -----------------------------------------------------------------------------------------
    def chem_lookup(self, name: str) -> Optional[str]:
        try:
            smi = S.pubchem_smiles(name, self.sources) if self.online_sources else None
        except Exception:                                                        # noqa: BLE001
            smi = None
        if smi:
            return smi
        try:
            ans, _ = self.ask("SMILES", "Reply with the SMILES string of the named compound and nothing else. If unsure reply NONE.", name, max_tokens=60)
            ans = ans.strip().split()[0] if ans.strip() else ""
            return None if ans.upper() == "NONE" else ans
        except (ServerOffline, ServerError, JobLost):
            return None

    def t_chem_molecule(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        if not K.HAVE_RDKIT:
            self.out(tid, w, type="note", text="Molecule drawing is not available on this computer yet (install the chemistry package).")
            w["abort"] = True
            self.work_put(tid, w)
            return {}
        lv = self.level_of(w)
        q = args.get("query") or w["request"].get("text", "")
        self.log(tid, f"Finding the structure of {q}")
        sp = K.resolve_species(q, self.chem_lookup)
        if not sp or not sp.get("smiles"):
            self.out(tid, w, type="note", text=f"I could not work out the structure of '{q}'. Try its common chemical name or its formula.")
            w["abort"] = True
            self.work_put(tid, w)
            return {}
        d = K.draw_molecule(sp, lv["key"])
        base = self.save_img(tid, d["image"], "mol")
        groups = []
        for g in d["groups"]:
            groups.append({"name": g["name"], "colour": list(g["colour"]), "image_path": self.save_img(tid, g["image"], "grp"),
                           "text": K.GROUP_NOTES.get(g["name"], f"This is the {g['name']} group.")})
        f = d["facts"]
        title = (f.get("name") or sp["label"]).capitalize()
        facts = [f"Formula: {K.formula_markup(f['formula'])}", f"Molar mass: {f['molar_mass']} g/mol"] + [f"Functional group: {g['name']}" for g in groups]
        fig = {"kind": "molecule", "title": title, "level": lv["key"], "clean_path": base, "groups": groups, "parts": [], "facts": facts,
               "source_note": "Structure drawn from its formula", "source_text": " ".join(g["text"] for g in groups)}
        shown = self.save_img(tid, C.add_title(d["image"], f"{title}  ({K.formula_markup(f['formula'])})"), "mol_t")
        self.out(tid, w, type="image", path=shown, caption=f"{title} - {K.formula_markup(f['formula'])}")
        self.out(tid, w, type="note", text=" | ".join(facts))
        for g, gd in zip(groups, d["groups"]):
            gp = self.save_img(tid, C.add_title(gd["image"], f"{title}: {g['name']} group"), "grp_t")
            self.out(tid, w, type="image", path=gp, caption=f"{g['name']} group highlighted")
        w["figure"] = True
        self.work_put(tid, w)
        return {"figure": fig, "current_image": shown, "last_explanation": ""}

    def t_chem_reaction(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        if not K.HAVE_RDKIT:
            self.out(tid, w, type="note", text="Reaction drawing is not available on this computer yet (install the chemistry package).")
            w["abort"] = True
            self.work_put(tid, w)
            return {}
        lv = self.level_of(w)
        text = args.get("text") or w["request"].get("text", "")
        self.log(tid, "Balancing the equation")
        plan = K.plan_reaction(text, self.chem_lookup)
        if plan.get("error") == "need_equation":
            ans, _ = self.ask("EQUATION", "Write the chemical equation for the described reaction as: A + B -> C + D, using formulas or common names. Reply with the equation only.",
                              text, max_tokens=80)
            plan = K.plan_reaction(text, self.chem_lookup, tutor_equation=ans.strip().splitlines()[0] if ans.strip() else None)
        if plan.get("error"):
            msg = plan["error"] if plan["error"] != "need_equation" else "I could not tell which reaction you mean. Write it like: methane + oxygen -> carbon dioxide + water."
            self.out(tid, w, type="note", text=msg)
            w["abort"] = True
            self.work_put(tid, w)
            return {}
        d = K.draw_reaction(plan, lv["key"])
        path = self.save_img(tid, d["image"], "rxn")
        regions = []
        for r in d["regions"]:
            role = "reactant" if r["role"] == "reactant" else "product"
            regions.append({"name": r["name"], "bbox": [int(x) for x in r["bbox"]], "colour": list(r["colour"]),
                            "text": f"{r['name']} is a {role}. {r['coeff']} {'unit' if r['coeff'] == 1 else 'units'} of it {'react' if role == 'reactant' else 'form'} in the balanced equation."})
        eq_pdf = K.equation_text(plan, "reportlab")
        bal = ", ".join(f"{b['element']}: {b['left']} = {b['right']}" for b in K.atom_balance(plan))
        names = lambda items: " + ".join((x.get("name") or x["label"]).capitalize() if x.get("name") else K.formula_markup(x["label"]) for x in items)
        n_r = plan["n_reactants"]
        title = f"{names(plan['species'][:n_r])} → {names(plan['species'][n_r:])}"
        fig = {"kind": "reaction", "title": title, "level": lv["key"], "clean_path": path, "regions": regions, "parts": [], "equation": eq_pdf,
               "facts": [f"Atoms on each side - {bal}"], "source_note": f"Equation from {plan['source']}", "source_text": f"{K.equation_text(plan)}. Atoms balance: {bal}."}
        shown = self.save_img(tid, C.add_title(d["image"], title), "rxn_t")
        self.out(tid, w, type="image", path=shown, caption=K.equation_text(plan))
        self.out(tid, w, type="note", text=f"Balanced: {K.equation_text(plan)}  ({bal}). Equation from {plan['source']}.")
        w["figure"] = True
        self.work_put(tid, w)
        return {"figure": fig, "current_image": shown, "last_explanation": ""}

    # ---- studio / edit -----------------------------------------------------------------------------------------
    def t_studio_image(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        opts = w["request"].get("options") or {}
        W, H = ASPECTS.get(opts.get("aspect", "square"), ASPECTS["square"])
        quality = "better" if opts.get("quality") == "high" else "fast"
        prompt = args.get("prompt") or w["request"].get("text", "")
        ph = w.setdefault("phase", {})
        ph.setdefault("seed", int(time.time()) % 100000)
        img = self.run_image_job(tid, w, "Creating the image", dict(mode="generate", prompt=prompt, width=W, height=H, quality=quality, seed=ph["seed"]))
        path = self.save_img(tid, img, "studio")
        self.out(tid, w, type="image", path=path, caption="")
        return {"current_image": path}

    def edit_source(self, state: State, w: dict) -> Optional[str]:
        for a in w["request"].get("attachments", []):
            if a["kind"] == "image":
                return a["path"]
        return state.get("current_image")

    def resolve_edit(self, tid: str, instruction: str) -> str:
        """Editing used to see only the latest sentence.  'Make it bigger' or 'now blue' is rewritten into a complete instruction
        using the recent conversation, and the rewrite is shown to the learner so a wrong guess is visible."""
        refers = re.search(r"\b(it|that|this|them|those|again|more|less|bigger|smaller|same|previous|before|undo|now|also|too|instead)\b", instruction, re.I)
        chat = self.chat_get(tid)
        if not chat or not (refers or len(instruction.split()) <= 4):
            return instruction
        lines = []
        for m in chat[-8:]:
            if m["role"] == "user" and m.get("text"):
                lines.append(f"Learner: {m['text'][:160]}")
            elif m["role"] == "assistant":
                caps = [o.get("caption") for o in m.get("outputs", []) if o.get("type") == "image" and o.get("caption")]
                txt = " ".join(o["text"] for o in m.get("outputs", []) if o.get("type") == "text")[:140]
                if caps or txt:
                    lines.append("Tutor: " + (f"[picture: {caps[-1]}] " if caps else "") + txt)
        if not lines:
            return instruction
        try:
            ans, _ = self.ask("EDIT_PROMPT", "Rewrite the learner's latest instruction as ONE complete, self-contained instruction for an image editor that cannot see the "
                              "conversation. Keep what they asked and add only details the conversation clearly implies (which object, which colour). "
                              "Reply with the instruction only, under 40 words.", "Conversation:\n" + "\n".join(lines) + f"\nLatest instruction: {instruction}", max_tokens=90)
        except ServerError:
            return instruction
        new = re.sub(r"\s+", " ", ans.strip().splitlines()[0]).strip(" \"'") if ans and ans.strip() else ""
        if 2 <= len(new.split()) <= 45:
            if new.lower() != instruction.lower():
                self.log(tid, f"Understood the edit as: {new}")
            return new
        return instruction

    def t_edit_image(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        src = self.edit_source(state, w)
        if not src:
            self.out(tid, w, type="note", text="Upload a picture to edit, or create one first.")
            w["abort"] = True
            self.work_put(tid, w)
            return {}
        im = Image.open(self.abs(src)).convert("RGB")
        im.thumbnail((896, 896), Image.LANCZOS)
        W, H = max(256, im.width - im.width % 16), max(256, im.height - im.height % 16)
        im = im.resize((W, H))
        ph = w.setdefault("phase", {})
        ph.setdefault("seed", int(time.time()) % 100000)
        if "instr" not in ph:                                           # decided once, so a resume sends the same instruction
            ph["instr"] = self.resolve_edit(tid, args.get("instruction") or w["request"].get("text", ""))
            self.work_put(tid, w)
        out = self.run_image_job(tid, w, "Editing the image", dict(mode="edit", prompt=ph["instr"],
                                                                 width=W, height=H, quality="better", seed=ph["seed"], image_b64=img_b64(im)))
        path = self.save_img(tid, out, "edit")
        self.out(tid, w, type="image", path=path, caption="Edited picture")
        return {"current_image": path}

    # ---- explain / chat ----------------------------------------------------------------------------------------
    def figure_facts(self, fig: dict) -> str:
        """Everything the tutor may say about the figure on screen - verified notes, plus where each labelled part is."""
        kind = fig.get("kind")
        if kind == "diagram":
            try:
                size = Image.open(self.abs(fig["clean_path"])).size
            except Exception:                                                    # noqa: BLE001
                size = None
            lines = []
            for p in fig.get("parts", []):
                if p.get("visible") and p.get("anchor"):
                    where = f" (in the {C.position_words(p['anchor'], size)} of the picture)" if size else ""
                    lines.append(f"- {p['name']}{where}: {p['text']}")
            missing = [p["name"] for p in fig.get("parts", []) if not p.get("visible")]
            out = f"Diagram: {fig['title']}. Labelled parts, with where they are as you look at the picture:\n" + "\n".join(lines)
            if missing:
                out += f"\nNot labelled in this picture - do not say where they are: {', '.join(missing)}."
            return out + f"\nReference notes: {fig.get('source_text', '')[:1800]}"
        if kind == "molecule":
            return f"Molecule: {fig['title']}. " + " ".join(fig.get("facts", [])) + " " + fig.get("source_text", "")[:800]
        if kind == "process":
            def line(i, s):
                extra = ""
                if s.get("parts"):
                    extra += f" [parts shown: {', '.join(s['parts'])}"
                    if s.get("flows"):
                        extra += "; arrows: " + ", ".join(f"{f['from']} to {f['to']}" + (f" ({f['what']})" if f.get("what") else "") for f in s["flows"])
                    extra += "]"
                return f"{i}. {s['title']}: {s['text']}{extra}"
            steps = "\n".join(line(i, s) for i, s in enumerate(fig.get("steps", []), 1))
            scope = (f" Every step happens inside the {fig['subject']} and is drawn on the same diagram of it; do not mention anything outside it."
                     if fig.get("system") else "")
            return (f"Process: {fig['title']} ({fig.get('shape', 'linear')}).{scope} Steps in order:\n{steps}"
                    + (" The last step leads back to the first." if fig.get("shape") == "cycle" else ""))
        return f"Reaction: {fig.get('source_text', '')}"

    def explain_figure(self, state: State, fig: dict, lv: dict, question: str, tokens: int, fresh: bool) -> str:
        """A concrete discussion of a diagram, process, molecule or reaction: exact names, where things are, what happens next."""
        sections = C.PROCESS_SECTIONS if fig.get("kind") == "process" else C.DIAGRAM_SECTIONS
        sysmsg = (f"You are a friendly science tutor talking about what is on the learner's screen. {lv['explain']} Use {lv['vocab']}. {C.CONCRETE_RULES} "
                  "Use only the facts given and never mention parts or steps that are not listed. Refer to labels and steps by their exact names"
                  + (" and say where a part is in the picture." if fig.get("kind") == "diagram" else ".")
                  + " Use exactly these markdown headings: " + ", ".join(f"## {h}" for h in sections)
                  + ". Under 'Check yourself' ask two short questions about named parts or steps.")
        return self.section_answer("EXPLAIN", sysmsg, f"{self.figure_facts(fig)}\nQuestion: {question or 'Explain this to me.'}", sections,
                                   history=None if fresh else self.history(state), max_tokens=tokens)

    def t_explain(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        lv = self.level_of(w)
        question = (args.get("question") or "").strip() or (w["request"].get("text") or "").strip()
        req_atts = w["request"].get("attachments", [])
        imgs = [Image.open(self.abs(a["path"])).convert("RGB") for a in req_atts if a["kind"] == "image"][:2]
        imgs += [Image.open(self.abs(p)).convert("RGB") for a in req_atts for p in a.get("pages", [])][:2]
        doc_text = "\n\n".join(a["text"] for a in req_atts if a.get("text"))
        if not doc_text and not imgs and not state.get("figure") and state.get("docs"):
            doc_text = "\n\n".join(d["text"] for d in state["docs"])
        fig = state.get("figure")
        tokens = {"ordinary": 380, "advanced": 520, "university": 700}[lv["key"]]
        self.log(tid, "Writing the explanation")
        fresh_fig = bool(w.get("figure"))
        if fig and (fresh_fig or not (imgs or doc_text)):
            text = self.explain_figure(state, fig, lv, question, tokens, fresh_fig)
        else:
            body = C.select_passages(doc_text, question, 4500) if doc_text else ""
            sysmsg = (f"You are a friendly science tutor. {lv['explain']} Use {lv['vocab']}. {C.CONCRETE_RULES} Base your answer on what you can see or read, "
                      "quote the exact labels, numbers or words you rely on, and say so if something is unclear. "
                      "Use exactly these markdown headings: " + ", ".join(f"## {h}" for h in C.CHAT_SECTIONS) + ".")
            user = (f"Document:\n{body}\n\n" if body else "") + f"Question: {question or 'Explain this to me.'}"
            text = self.section_answer("EXPLAIN", sysmsg, user, C.CHAT_SECTIONS, images=imgs or None, history=self.history(state), max_tokens=tokens)
        self.out(tid, w, type="text", text=C.scrub_ui_text(text))
        return {"last_explanation": text}

    def trusted_notes(self, question: str, lv: dict) -> str:
        """Notes from the built-in library or Wikipedia (cached) that are really about the question - so a deeper answer rests on facts."""
        topic = C.question_topic(question)
        if not topic:
            return ""
        try:
            src = S.fetch_topic(topic, lv["key"], self.sources, self.online_sources)
        except Exception:                                                        # noqa: BLE001
            return ""
        if not src:
            return ""
        q = {w[:5] for w in C.content_words(question)}                          # stems, so 'sweat' matches 'sweating'
        hay = {w[:5] for w in C.content_words((src.get("title") or "") + " " + src["text"][:1500])}
        if len(q & hay) < (1 if len(q) <= 3 else 2):
            return ""
        return f"Trusted notes ({src.get('source') or 'notes'}):\n{src['text'][:1800]}"

    def dialogue_act(self, state: State, text_in: str) -> str:
        """greeting | answer (the learner is replying to the tutor's last question) | deeper | question | follow_up"""
        if C.RE_GREET.match(text_in):
            return "greeting"
        words = text_in.split()
        last_tutor = next((m["content"] for m in reversed(state.get("messages") or []) if m["role"] == "assistant"), "")
        asked = "?" in last_tutor.strip()[-200:]
        is_q = bool(C.RE_QUESTION.search(text_in)) or "?" in text_in
        if C.RE_DEEPER.search(text_in) and len(words) <= 10:
            return "deeper"
        if asked and not is_q and len(words) >= 2:
            return "answer"
        if len(words) >= 5 and is_q:
            return "question"
        return "follow_up"

    def t_chat(self, state: State, args: dict, w: dict) -> dict:
        """A conversation that goes somewhere: it answers, explains the mechanism, gives a real example, names a common mistake and ends
        with a question that makes the learner reason; it grades the learner's reply to its own question; and 'tell me more' goes one level down."""
        tid = state["thread_id"]
        lv = self.level_of(w)
        text_in = w["request"].get("text", "")
        act = self.dialogue_act(state, text_in)
        extra = ""
        if state.get("docs"):
            extra += "\n\nDocument excerpts you may use:\n" + C.select_passages("\n\n".join(d["text"] for d in state["docs"]), text_in, 3000)
        if state.get("figure"):                                                  # the conversation stays grounded in what is on screen
            extra += "\n\nOn the learner's screen now (use it only if the question is about it):\n" + self.figure_facts(state["figure"])[:2600]
        if act in ("question", "deeper") and not state.get("docs"):
            q_for_notes = text_in if act == "question" else (state.get("last_topic") or next((m["content"] for m in reversed(state.get("messages") or []) if m["role"] == "user"), ""))
            notes = self.trusted_notes(q_for_notes, lv)
            if notes:
                extra += "\n\n" + notes + "\n(Use these notes for facts; do not contradict them.)"
        base = f"You are a friendly, accurate science tutor for African learners. {lv['explain']} Use {lv['vocab']}. "
        deep_tokens = {"ordinary": 460, "advanced": 620, "university": 800}[lv["key"]]
        if act == "greeting":
            ans, _ = self.ask("CHAT", base + "Reply in one or two friendly sentences and invite the learner to ask a science question.", text_in,
                              history=self.history(state), max_tokens=80, temperature=0.3)
        elif act == "answer":
            ans, fin = self.ask("CHAT", base + C.CONCRETE_RULES + " The learner is replying to the question you asked at the end of your last message. "
                                "First say what is right in their answer, quoting their words. If something is missing or wrong, correct it gently with the exact fact. "
                                "Then add ONE new idea that goes a step deeper, and end with a new question that needs reasoning, not a yes or no.",
                                text_in + extra, history=self.history(state), max_tokens=deep_tokens, temperature=0.3)
            if fin == "length":
                ans = C.trim_to_sentence(ans)
        elif act == "deeper":
            ans, fin = self.ask("CHAT", base + C.CONCRETE_RULES + " The learner wants to go deeper on what you just explained. Do not repeat it. Explain the mechanism underneath: "
                                "what happens at the smaller scale or in the step before and after. Give one real example from daily life in East Africa or from a school laboratory, "
                                "and one exception or limit. End with a question that makes the learner predict something.",
                                text_in + extra, history=self.history(state), max_tokens=deep_tokens, temperature=0.3)
            if fin == "length":
                ans = C.trim_to_sentence(ans)
        elif act == "question":
            sysmsg = (base + C.CONCRETE_RULES + " Answer the question directly first. Then explain how it works, cause and then effect, one step at a time. "
                      "Give one real example, from East African daily life where it fits. Name one common mistake and correct it. "
                      "Use exactly these markdown headings: " + ", ".join(f"## {h}" for h in C.DEEP_SECTIONS)
                      + ". Under 'Think about it' ask ONE question that needs reasoning, not a yes or no.")
            ans = self.section_answer("CHAT", sysmsg, text_in + extra, C.DEEP_SECTIONS, history=self.history(state), max_tokens=deep_tokens)
        else:                                                                    # a short follow-up: direct, concrete, linked to what came before
            ans, fin = self.ask("CHAT", base + C.CONCRETE_RULES + " Answer in three to six sentences, directly and concretely, link it to what was said earlier, "
                                "and end with one short question that checks understanding or invites the next step.", text_in + extra,
                                history=self.history(state), max_tokens=300, temperature=0.3)
            if fin == "length":
                ans = C.trim_to_sentence(ans)
        self.out(tid, w, type="text", text=C.scrub_ui_text(ans))
        return {}


    # ---- process infographic -----------------------------------------------------------------------------------
    def process_load(self, rec_rel: str) -> dict:
        return json.loads(self.abs(rec_rel).read_text(encoding="utf-8"))

    def process_save(self, fig: dict):
        self.abs(fig["record_path"]).write_text(json.dumps(fig), encoding="utf-8")

    def draw_step(self, tid: str, w: dict, fig: dict, i: int) -> bool:
        """Draw one panel.  Offline/lost-job errors propagate (the run parks); a failed panel is marked and skipped."""
        st = fig["steps"][i]
        n = len(fig["steps"])
        try:
            img = self.run_image_job(tid, w, f"Drawing step {i + 1} of {n}", dict(
                mode="generate", prompt=C.build_panel_prompt(st["scene"], fig.get("subject", "")), width=512, height=512, quality="fast", seed=st["seed"]))
            st["image_path"], st["status"] = self.save_img(tid, img, f"step{i + 1}"), "ready"
            ok = True
        except ServerError:
            st["status"] = "failed"
            self.log(tid, f"Step {i + 1} could not be drawn")
            ok = False
        w.get("phase", {}).pop("job", None)
        self.work_put(tid, w)
        self.process_save(fig)
        return ok

    def finish_overview(self, tid: str, fig: dict):
        try:
            ov = mm_info.compose_overview(fig, self.data_dir)
            fig["overview_path"] = self.save_img(tid, ov, "overview")
        except Exception:                                                        # noqa: BLE001
            fig["overview_path"] = None
        self.process_save(fig)

    PROCESS_SYSTEM_MIN = 3        # parts that must be located on the drawing before steps are drawn on it

    def t_learn_process(self, state: State, args: dict, w: dict) -> dict:
        """A process is drawn as the SAME verified diagram of its organ or system, once per step: the parts involved in the step are
        highlighted and arrows (drawn by code) show what moves between them.  Nothing outside the organ can appear and the picture
        cannot change between steps.  Processes that do not happen inside one object fall back to one small picture per step."""
        tid = state["thread_id"]
        lv = self.level_of(w)
        topic = (args.get("topic") or w["request"].get("text") or "").strip()
        ph = w.setdefault("phase", {})
        if "mode" not in ph:
            ph["sys"] = self.resolve_system(tid, topic, lv)
            ph["mode"] = "system" if ph["sys"] else "panels"
            self.work_put(tid, w)
        if ph["mode"] == "system":
            res = self.process_on_system(state, w, lv, topic)
            if res is not None:
                return res
            ph["mode"] = "panels"                                      # not enough parts could be placed: fall back
            self.work_put(tid, w)
        return self.process_panels(state, w, lv, topic)

    def resolve_system(self, tid: str, topic: str, lv: dict) -> str:
        """The organ / system / apparatus the process happens INSIDE ('' when it is not inside one object)."""
        cp = S.curated_process(topic, lv["key"])
        if cp:
            return cp.get("system", "")
        d = S.curated_topic(topic)
        if d and d["name"] != "water cycle":
            return d["name"]
        try:
            ans, _ = self.ask("SYSTEM", "Which single organ, body system or piece of apparatus does this process happen inside? Reply with its name only, "
                              "copied from the request, or NONE if it does not happen inside one object (for example the water cycle).",
                              f"Request: {topic}", max_tokens=16)
        except (ServerOffline, ServerError, JobLost):
            return ""
        name = re.sub(r"[^a-zA-Z' \-]", "", ans or "").strip().lower()
        if not name or name.startswith("none") or len(name.split()) > 4 or C._norm(name) not in C._norm(topic):
            return ""
        return name

    def plan_in_system(self, w: dict, topic: str, lv: dict, base: dict, components: list[str]) -> tuple[list[dict], dict]:
        """Ordered steps that stay inside the system.  Curated processes keep their vetted wording; otherwise the tutor plans from the notes
        and code rejects any step that is unsupported, names another organ's structures, or involves none of the system's parts."""
        ph = w["phase"]
        lo, hi = C.PROCESS_STEPS[lv["key"]]
        cp = S.curated_process(topic, lv["key"])
        meta = {"dropped": 0, "wandered": []}
        if cp and cp.get("system"):
            steps = [{**st, "parts": [], "flows": []} for st in cp["steps"]]
            kept, _ = C.verify_plan(steps, components, cp["text"], hi + 2, trusted=True)
            meta.update(dropped=len(steps) - len(kept), title=cp["title"], shape=cp["shape"], text=cp["text"], note="Source: built-in notes")
            return kept, meta
        all_parts = S.all_part_names()
        foreign = C.foreign_terms(components, ph.get("sys", ""), all_parts)
        sentences = "\n".join(f"- {p['name']}: {p['text']}" for p in base["parts"] if p["visible"])
        bad: list[str] = []
        kept: list[dict] = []
        parsed: Optional[dict] = None
        for attempt in range(2):
            ans, _ = self.ask("PROCESS_PLAN", f"Plan the steps of a process that happens entirely INSIDE the {ph['sys']}. Use ONLY the parts listed and only the facts in the text. "
                              f"Every step must name at least one listed part and say where things go from and to. Do not mention any structure outside the {ph['sys']}; "
                              "if something arrives from or leaves to the outside, say only that it arrives at or leaves from a listed part. Reply with JSON only: "
                              "{\"shape\":\"linear|cycle|io\",\"steps\":[{\"title\":short name,\"text\":one sentence using the exact part names,\"parts\":[part names],"
                              "\"flows\":[{\"from\":part,\"to\":part,\"what\":what moves, one or two words}],\"scene\":a simple picture of the step}]}. "
                              f"Give {lo} to {hi} steps. Use {lv['vocab']}.",
                              f"Process: {topic}\nParts you may use:\n{sentences}\nText: {base['source_text'][:3000]}"
                              + (f"\nDo not mention: {', '.join(bad)}." if bad else ""), max_tokens=900)
            parsed = C.parse_plan(ans)
            if not parsed:
                continue
            got, wandered = C.verify_plan(parsed["steps"], components, base["source_text"] + " " + sentences, hi, foreign)
            if len(got) >= len(kept):                                       # keep the better of the two attempts
                kept, meta["dropped"], meta["wandered"] = got, len(parsed["steps"]) - len(got), wandered
            if not wandered:
                meta["wandered"] = []
                break
            bad = wandered                                                  # one corrective retry, naming what to leave out
        meta.update(title=re.sub(r"\s+", " ", topic).strip(" ?.!").title()[:70],
                    shape=(parsed or {}).get("shape", "linear"), text=base["source_text"], note=base["source_note"])
        return kept, meta

    def process_on_system(self, state: State, w: dict, lv: dict, topic: str) -> Optional[dict]:
        tid = state["thread_id"]
        ph = w["phase"]
        cp_ = S.curated_process(topic, lv["key"])
        built = self.build_diagram(tid, w, ph["sys"], lv, quiet=True, prefer=((cp_["text"] + " " + " ".join(s_["title"] for s_ in cp_["steps"])) if cp_ else topic))
        if built is None:
            return None
        base, verified, _ = built
        located = [p for p in base["parts"] if p["visible"]]
        if not verified or len(located) < self.PROCESS_SYSTEM_MIN:
            self.out(tid, w, type="note", text=f"I could not place enough parts of the {ph['sys']} on the drawing, so each step is drawn as its own small picture instead.")
            return None
        components = [p["name"] for p in located]
        if "plan" not in ph:
            self.log(tid, f"Planning the steps inside the {ph['sys']}")
            kept, meta = self.plan_in_system(w, topic, lv, base, components)
            ph["plan"], ph["meta"] = kept, meta
            self.work_put(tid, w)
        kept, meta = ph["plan"], ph["meta"]
        if len(kept) < 2:
            msg = f"I could not build clear steps that stay inside the {ph['sys']} from the notes I found."
            if meta.get("wandered"):
                msg += " The notes keep moving to other parts of the body (" + ", ".join(meta["wandered"][:3]) + "). Try asking for one stage of it."
            self.out(tid, w, type="note", text=msg)
            w["abort"] = True
            self.work_put(tid, w)
            return {}
        if not ph.get("rec"):                                          # (a resume skips this: the record already exists)
            rec = self.tdir(tid) / f"process_{uuid.uuid4().hex[:6]}.json"
            system = {"title": base["title"], "clean_path": base["clean_path"], "labelmap_path": base["labelmap_path"],
                      "parts": [{k: p[k] for k in ("name", "colour", "anchor", "region_id", "visible")} for p in base["parts"]]}
            fig = {"kind": "process", "title": meta["title"], "shape": meta["shape"] if len(kept) >= 3 else "linear", "level": lv["key"], "level_label": lv["label"],
                   "subject": ph["sys"], "system": system,
                   "steps": [{"n": i + 1, "title": st["title"], "text": st["text"], "scene": st.get("scene", ""), "parts": st["parts"], "flows": st["flows"],
                              "image_path": None, "status": "pending", "seed": 0} for i, st in enumerate(kept)],
                   "source_text": meta["text"][:4000], "source_note": meta["note"], "facts": [], "record_path": self.rel(rec), "overview_path": None}
            self.render_system_steps(tid, fig)
            self.finish_overview(tid, fig)
            self.out(tid, w, type="process", path=fig["record_path"])
            if meta.get("dropped"):
                self.out(tid, w, type="note", text=f"I kept to the {ph['sys']} only: {meta['dropped']} step(s) were left out because they happen outside it or need parts that are not on the drawing.")
            ph["rec"] = fig["record_path"]
            self.work_put(tid, w)
        fig = self.process_load(ph["rec"])
        if not ph.get("discussed"):
            try:
                ph["disc"] = self.explain_figure(state, fig, lv, "", {"ordinary": 380, "advanced": 520, "university": 700}[lv["key"]], True)
                self.out(tid, w, type="text", text=C.scrub_ui_text(ph["disc"]))
            except ServerError:
                ph["disc"] = ""
            ph["discussed"] = True
            self.work_put(tid, w)
        self.log(tid, f"Drew {len(fig['steps'])} steps on the {ph['sys']}")
        return {"figure": fig, "current_image": fig.get("overview_path"), "last_explanation": ph.get("disc", "")}

    def render_system_steps(self, tid: str, fig: dict, only: Optional[int] = None):
        n = len(fig["steps"])
        for i, st in enumerate(fig["steps"]):
            if only is not None and i != only:
                continue
            img = mm_info.compose_system_step(fig["system"], st, i, n, self.data_dir)
            st["image_path"], st["status"] = self.save_img(tid, img, f"step{i + 1}"), "ready"
        self.process_save(fig)

    def process_panels(self, state: State, w: dict, lv: dict, topic: str) -> dict:
        """Fallback for processes that do not happen inside one organ or apparatus (water cycle, mitosis ...): one small picture per step."""
        tid = state["thread_id"]
        ph = w.setdefault("phase", {})
        if "rec" not in ph:
            self.log(tid, f"Planning the steps of {topic}")
            lo, hi = C.PROCESS_STEPS[lv["key"]]
            src = S.curated_process(topic, lv["key"])
            if src:
                steps, shape, text, subject, note = src["steps"], src["shape"], src["text"], src.get("subject", ""), "Source: built-in notes"
                title = src["title"]
            else:
                wiki = S.fetch_topic(topic, lv["key"], self.sources, self.online_sources)
                if not wiki:
                    self.out(tid, w, type="note", text="I could not find reliable notes on that process. Try a more common name.")
                    w["abort"] = True
                    self.work_put(tid, w)
                    return {}
                ans, _ = self.ask("PROCESS", f"Break the process into {lo} to {hi} ordered steps using only the text. Reply with JSON only: "
                                  "{\"shape\":\"linear|cycle|io\",\"steps\":[{\"title\":short name,\"text\":one sentence,\"scene\":a simple picture of the step}]}. "
                                  f"Use {lv['vocab']}.", f"Process: {topic}\nText: {wiki['text'][:3500]}", max_tokens=800)
                parsed = C.parse_process(ans)
                steps = C.verify_steps(parsed["steps"], wiki["text"], hi) if parsed else []
                if len(steps) < 3:
                    self.out(tid, w, type="note", text="I could not break that into clear steps from the notes I found. Try a more specific process.")
                    w["abort"] = True
                    self.work_put(tid, w)
                    return {}
                shape, text, subject = parsed["shape"], wiki["text"], ""
                title = (wiki.get("title") or topic).title()
                note = f"Source: {wiki.get('source') or 'notes'}" + (f" ({wiki['license']})" if wiki.get("license") and wiki.get("url") else "")
            steps = steps[:hi]
            base = int(time.time()) % 90000
            rec = self.tdir(tid) / f"process_{uuid.uuid4().hex[:6]}.json"
            fig = {"kind": "process", "title": title, "shape": shape, "level": lv["key"], "level_label": lv["label"], "subject": subject,
                   "steps": [{"n": i + 1, "title": st["title"], "text": st["text"], "scene": st["scene"], "image_path": None, "status": "pending", "seed": base + i}
                             for i, st in enumerate(steps)],
                   "source_text": text[:4000], "source_note": note, "facts": [], "record_path": self.rel(rec), "overview_path": None}
            self.process_save(fig)
            self.out(tid, w, type="process", path=fig["record_path"])
            ph["rec"] = fig["record_path"]
            self.work_put(tid, w)
        fig = self.process_load(ph["rec"])
        if not ph.get("discussed"):                       # the discussion comes first (it is fast), so the learner reads while the panels are drawn
            try:
                ph["disc"] = self.explain_figure(state, fig, lv, "", {"ordinary": 380, "advanced": 520, "university": 700}[lv["key"]], True)
                self.out(tid, w, type="text", text=C.scrub_ui_text(ph["disc"]))
            except ServerError:
                ph["disc"] = ""
            ph["discussed"] = True
            self.work_put(tid, w)
        for i, st in enumerate(fig["steps"]):
            if st["status"] != "ready":
                self.draw_step(tid, w, fig, i)
        self.finish_overview(tid, fig)
        ready = sum(1 for s_ in fig["steps"] if s_["status"] == "ready")
        self.log(tid, f"Drew {ready} of {len(fig['steps'])} steps")
        return {"figure": fig, "current_image": fig.get("overview_path"), "last_explanation": ph.get("disc", "")}

    def t_retry_step(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        fig0 = state.get("figure") or {}
        if fig0.get("kind") != "process":
            self.out(tid, w, type="note", text="There is no step-by-step process to retry.")
            return {}
        fig = self.process_load(fig0["record_path"])
        try:
            i = int(args.get("n", 0))
        except (TypeError, ValueError):
            i = 0
        if not 0 <= i < len(fig["steps"]):
            return {}
        if fig.get("system"):                                                    # drawn by code: re-render, no service needed
            self.render_system_steps(tid, fig, only=i)
            self.finish_overview(tid, fig)
            return {"figure": fig, "current_image": fig.get("overview_path")}
        fig["steps"][i]["seed"] = int(time.time()) % 90000
        fig["steps"][i]["status"] = "pending"
        self.draw_step(tid, w, fig, i)
        self.finish_overview(tid, fig)
        return {"figure": fig, "current_image": fig.get("overview_path")}

    def t_clarify(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        self.out(tid, w, type="choices", topic=args.get("topic", ""),
                 text=f"How would you like '{args.get('topic', '')}' shown?", options=["diagram", "process", "picture"])
        w["abort"] = True
        self.work_put(tid, w)
        return {"last_topic": args.get("topic", "")}

    # ---- documents: notes of the discussion, learning guides ---------------------------------------------------
    def doc_save(self, tid: str, kind: str, title: str, level_key: str, blocks: list[dict]) -> dict:
        lv = C.get_level(level_key)
        p = self.tdir(tid) / f"doc_{uuid.uuid4().hex[:6]}.json"
        pdf = self.tdir(tid) / f"{C.safe_name(title)}_{uuid.uuid4().hex[:4]}.pdf"
        mm_pdf.build_doc_pdf(pdf, blocks, lv["label"], self.data_dir)
        rec = {"kind": kind, "title": title, "level": level_key, "blocks": blocks, "pdf": self.rel(pdf)}
        p.write_text(json.dumps(rec), encoding="utf-8")
        return {"path": self.rel(p), "title": title, "pdf": rec["pdf"]}

    def doc_outputs(self, tid: str, w: dict, d: dict):
        self.out(tid, w, type="doc", path=d["path"])
        self.out(tid, w, type="file", path=d["pdf"], name=Path(d["pdf"]).name, mime="application/pdf")

    EXPORT_ACTIONS = ("pdf", "lesson", "slides", "notes")

    def chat_slice(self, tid: str, w: dict) -> list:
        """The part of the chat an export should cover: the last N exchanges (the learner's choice), or the whole chat."""
        chat = self.chat_get(tid)
        if chat and chat[-1]["role"] == "user" and w["request"].get("action") in self.EXPORT_ACTIONS:
            chat = chat[:-1]                                     # the export command itself is not part of the discussion
        n = w["request"].get("history")
        users = [i for i, m in enumerate(chat) if m["role"] == "user"]
        if isinstance(n, int) and n > 0 and len(users) > n:
            chat = chat[users[-n]:]
        return chat

    def depth_of(self, w: dict) -> str:
        d = (w.get("request") or {}).get("depth")
        return d if d in mm_doc.DEPTHS else mm_doc.DEFAULT_DEPTH

    def notes_material(self, tid: str, w: dict, lv: dict, mode: str = "condensed", depth: str = "brief"):
        """(summary, turns, images) of the chosen part of the chat, or None when there is nothing to summarise.
        Only Brief notes ask the tutor for a summary; Standard and Deep use the answers as they were given."""
        turns, images = [], []
        for m in self.chat_slice(tid, w):
            if m["role"] == "user" and m.get("text"):
                turns.append({"role": "user", "text": m["text"]})
            elif m["role"] == "assistant":
                txt = " ".join(o["text"] for o in m.get("outputs", []) if o["type"] == "text")
                mine: list = []
                for o in m.get("outputs", []):
                    if o["type"] == "image" and o.get("path"):
                        mine.append((o["path"], o.get("caption", "")))
                    elif o["type"] == "process":
                        try:
                            rec = self.process_load(o["path"])
                            if rec.get("overview_path"):
                                mine.append((rec["overview_path"], rec["title"]))
                        except Exception:                                        # noqa: BLE001
                            pass
                images += mine
                if txt or mine:
                    turns.append({"role": "assistant", "text": txt or "Here is the picture.", "images": mine})
        if not turns:
            return None
        summary = ""
        if mode != "full" and depth == "brief":
            convo = "\n".join(f"{'Learner' if t['role'] == 'user' else 'Tutor'}: {t['text']}" for t in turns)[-6500:]
            heads = ["What we covered", "Key points", "Questions to revisit"]
            summary = self.section_answer("NOTES", f"Write short revision notes from this tutoring conversation. {lv['explain']} Use only what is in the conversation. "
                                          "Use exactly these markdown headings: " + ", ".join(f"## {h}" for h in heads) + ".", convo, heads, max_tokens=520)
        return summary, turns, images

    def discussion_doc(self, tid: str, w: dict, lv: dict, mode: str = "condensed"):
        """(blocks, summary, images) of the discussion at the chosen depth, or None."""
        depth = self.depth_of(w)
        mat = self.notes_material(tid, w, lv, mode, depth)
        if not mat:
            return None
        summary, turns, images = mat
        if depth == "brief":
            return mm_doc.notes_blocks("Discussion notes", lv["label"], summary, images[-6:], turns if mode == "full" else None), summary, images
        return mm_doc.discussion_blocks("Discussion notes", lv["label"], turns, depth, transcript=(mode == "full")), summary, images

    def t_notes(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        lv = self.level_of(w)
        mode = args.get("mode", "condensed")
        self.log(tid, "Writing the discussion notes")
        mat = self.discussion_doc(tid, w, lv, mode)
        if not mat:
            self.out(tid, w, type="note", text="There is no discussion to make notes from yet. Ask me something first.")
            w["abort"] = True
            self.work_put(tid, w)
            return {}
        blocks, summary, images = mat
        d = self.doc_save(tid, "notes", "Discussion notes", lv["key"], blocks)
        self.doc_outputs(tid, w, d)
        return {"doc": d, "last_explanation": summary or state.get("last_explanation", "")}

    def t_guide(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        lv = self.level_of(w)
        topic = (args.get("topic") or "").strip()
        atts = w["request"].get("attachments", [])
        doc_text = "\n\n".join(a["text"] for a in atts if a.get("text"))
        fig = state.get("figure") or {}
        text, title, note = "", topic, ""
        if doc_text or (not topic and state.get("docs")):
            body = doc_text or "\n\n".join(d["text"] for d in state["docs"])
            text = C.select_passages(body, topic, 5000)
            title = topic or (atts[0]["name"].rsplit(".", 1)[0] if atts else (state["docs"][0]["name"].rsplit(".", 1)[0]))
            note = "Source: your document"
        else:
            topic = topic or state.get("last_topic") or fig.get("title") or ""
            title = topic
            if not topic:
                self.out(tid, w, type="note", text="Tell me the topic for the guide, for example: study guide on photosynthesis.")
                w["abort"] = True
                self.work_put(tid, w)
                return {}
            self.log(tid, f"Looking up reliable notes on {topic}")
            cp = S.curated_process(topic, lv["key"])
            src = {"title": cp["title"], "text": cp["text"], "source": "built-in notes", "url": None} if cp else S.fetch_topic(topic, lv["key"], self.sources, self.online_sources)
            if not src and fig and C._norm(topic) in C._norm(fig.get("title", "") + " " + fig.get("source_text", "")[:200]):
                src = {"title": fig["title"], "text": fig.get("source_text", ""), "source": fig.get("source_note", "")}
            if not src:
                self.out(tid, w, type="note", text="I could not find reliable notes on that topic. Try a more common name, or attach your own notes.")
                w["abort"] = True
                self.work_put(tid, w)
                return {}
            text = src["text"][:5000]
            title = src.get("title") or topic
            note = f"Source: {src.get('source') or 'notes'}" + (f" ({src['license']})" if src.get("license") and src.get("url") else "")
        title = title.title() if title.islower() else title
        n_terms, n_q = {"ordinary": (5, 4), "advanced": (7, 5), "university": (9, 6)}[lv["key"]]
        self.log(tid, "Writing the study guide")
        heads = ["Overview", "How it works", "Why it matters"]
        expl = self.section_answer("GUIDE", f"You are writing a study guide. {lv['explain']} Use {lv['vocab']}. Use only the notes. "
                                   "Use exactly these markdown headings: " + ", ".join(f"## {h}" for h in heads) + ".",
                                   f"Topic: {title}\nNotes: {text}", heads, max_tokens={"ordinary": 420, "advanced": 560, "university": 720}[lv["key"]])
        terms, qa = [], []
        try:
            ans, _ = self.ask("TERMS", f"List up to {n_terms} key terms from the notes. Reply with JSON only: [{{\"term\":..., \"meaning\":one short sentence}}].",
                              f"Topic: {title}\nNotes: {text}", max_tokens=450)
            terms = C.verify_terms(C.parse_json(ans) or [], text, n_terms)
        except (ServerError,):
            pass
        opts = w["request"].get("options") or {}
        with_q = opts.get("questions", True)
        if with_q:
            try:
                ans, _ = self.ask("QUESTIONS", f"Write {n_q} practice questions with short answers taken from the notes. Reply with JSON only: [{{\"q\":..., \"a\":...}}].",
                                  f"Topic: {title}\nNotes: {text}", max_tokens=600)
                qa = C.verify_qa(C.parse_json(ans) or [], text, n_q)
            except (ServerError,):
                pass
        images = []
        if fig and C._norm(title) and (C._norm(title) in C._norm(fig.get("title", "")) or C._norm(fig.get("title", "")) in C._norm(title)):
            pic = fig.get("overview_path") or state.get("current_image")
            if pic:
                images.append((pic, fig.get("title", "")))
        blocks = mm_doc.guide_blocks(title, lv["label"], expl, terms, qa, images, note, with_q)
        d = self.doc_save(tid, "guide", f"Study guide - {title}", lv["key"], blocks)
        self.doc_outputs(tid, w, d)
        if with_q and not qa:
            self.out(tid, w, type="note", text="I could not make practice questions that I could check against the notes, so I left them out.")
        return {"doc": d, "last_explanation": expl, "last_topic": topic or title}

    # ---- lesson (narrated video) -----------------------------------------------------------------------------------
    def video_figure(self, state: State, w: dict, tid: str) -> Optional[dict]:
        """What the video is about: the current figure, else the current document, else notes of the discussion."""
        lv = self.level_of(w)
        fig, doc = state.get("figure"), state.get("doc")
        focus = state.get("focus") or ("figure" if fig else "doc")
        if w["request"].get("scope") == "chat":                  # the export bar's "our discussion" choice
            fig, doc, focus = None, None, "doc"
        if focus == "figure" and fig:
            fig = json.loads(json.dumps(fig))
            if fig["kind"] == "process":
                fig = self.process_load(fig["record_path"])
                if not any(s_["status"] == "ready" for s_ in fig["steps"]):
                    return None
            self.deepen_narration(tid, fig, self.depth_of(w))
            return fig
        if doc:
            rec = json.loads(self.abs(doc["path"]).read_text(encoding="utf-8"))
            blocks = rec["blocks"]
        else:
            nd = self.t_notes(state, {"mode": "condensed"}, w)
            if not nd.get("doc"):
                return None
            rec = json.loads(self.abs(nd["doc"]["path"]).read_text(encoding="utf-8"))
            blocks = rec["blocks"]
        return {"kind": "slides", "title": rec["title"], "level": rec["level"], "slides": mm_doc.slides_from_blocks(blocks, depth=self.depth_of(w))}

    def t_lesson(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        fig = self.video_figure(state, w, tid)
        if not fig:
            if not any(o["type"] == "note" for o in w["outputs"]):
                self.out(tid, w, type="note", text="There is nothing to narrate yet. Ask me for a diagram, a process, a study guide or some questions first.")
            return {}
        lv = C.get_level(fig.get("level"))
        out_path = self.tdir(tid) / f"lesson_{uuid.uuid4().hex[:6]}.mp4"

        def prog(msg, frac):
            self.log(tid, f"{msg}…")
        self.log(tid, "Preparing the narrated lesson")
        try:
            res = mm_media.make_lesson(fig, self.data_dir, out_path, self.voice, progress=prog)
        except Exception as e:                                                   # noqa: BLE001
            self.out(tid, w, type="note", text="I could not build the video on this computer (video tools missing).")
            self.log(tid, f"Lesson failed: {type(e).__name__}: {str(e)[:120]}")
            return {}
        self.out(tid, w, type="video", path=self.rel(out_path), seconds=res.get("seconds"))
        if res.get("voice") == "none":
            self.out(tid, w, type="note", text="No voice is installed on this computer, so the lesson has captions only.")
        return {}

    NARRATION_WORDS = {"brief": 40, "standard": 75, "deep": 110}

    def deepen_narration(self, tid: str, fig: dict, depth: str):
        """Spoken text for the video (and speaker notes of the slides): an introduction, 2-4 sentences per part or step, and a recap.
        Written by the tutor from the notes only, a few items per call, and cached so a second export costs nothing."""
        if fig.get("kind") not in ("diagram", "process"):
            return
        nw = self.NARRATION_WORDS.get(depth, 75)
        key = C.safe_name(Path(fig.get("clean_path") or fig.get("record_path") or fig["title"]).stem)
        cache = self.tdir(tid) / f"narration_{depth}_{key}.json"
        got: dict = {}
        if cache.exists():
            try:
                got = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:                                                    # noqa: BLE001
                got = {}
        if not got:
            if fig["kind"] == "diagram":
                items = [(p["name"], p.get("text") or "") for p in fig["parts"] if p.get("visible", True)]
                src = fig.get("source_text", "")
            else:
                items = [(s_["title"], s_.get("text") or "") for s_ in fig["steps"] if s_.get("status") == "ready"]
                src = fig.get("text", "")
            names = [n for n, _ in items]
            title = fig["title"]
            if depth == "brief" or not items:
                return
            chunks = [items[i:i + 4] for i in range(0, len(items), 4)]
            self.log(tid, "Writing the narration")
            ans, _ = self.ask("NARRATE", f"Write the spoken opening and closing of a lesson video about {title}. Reply with JSON "
                              f"{{\"intro\": 2 or 3 sentences saying what it is and why it matters, \"recap\": 2 or 3 sentences showing how the parts fit together}}. "
                              f"At most {nw} words each. Use only the notes.", f"Notes: {src[:2500]}\nParts in the order they are shown: {', '.join(names)}", max_tokens=420)
            obj = C.parse_json(ans)
            for k in ("intro", "recap"):
                if isinstance(obj, dict) and isinstance(obj.get(k), str) and len(obj[k].split()) >= 6:
                    got[k] = C.words_trim(re.sub(r"\s+", " ", obj[k]).strip(), nw + 20)
            items_out: dict = {}
            for n, ch in enumerate(chunks, 1):
                self.log(tid, f"Writing the narration {n} of {len(chunks)}")
                ans, _ = self.ask("NARRATE", f"Write the spoken narration of a lesson video about {title}. For each item below write {nw // 2} to {nw} words "
                                  "(2 to 4 plain sentences): what it is, what it does, and how it connects to the items around it. Use ONLY the notes. "
                                  "Reply with JSON {\"item name\": \"narration\"} using the item names exactly.",
                                  f"Notes: {src[:2500]}\nItems:\n" + "\n".join(f"- {nm}: {tx}" for nm, tx in ch),
                                  max_tokens=min(880, int(len(ch) * nw * 1.7) + 80), temperature=0.2)
                for nm, tx in C.parse_narration(ans, [x for x, _ in ch]).items():
                    items_out[nm] = C.words_trim(tx, nw + 15)
            got["items"] = items_out
            if got["items"] or got.get("intro"):
                cache.write_text(json.dumps(got), encoding="utf-8")
        if got.get("intro"):
            fig["lesson_intro"] = got["intro"]
        if got.get("recap"):
            fig["lesson_outro"] = got["recap"]
        it = got.get("items") or {}
        if fig["kind"] == "diagram":
            for p in fig["parts"]:
                if it.get(p["name"]):
                    p["narration"] = it[p["name"]]
        else:
            for s_ in fig["steps"]:
                if it.get(s_["title"]):
                    s_["narration"] = it[s_["title"]]

    # ---- slides (PDF + editable PowerPoint) -----------------------------------------------------------------------
    def t_slides(self, state: State, args: dict, w: dict) -> dict:
        """Key-point slides of the picture on screen, or of the chosen part of the discussion.  Independent of the video; the video is
        built from the same scenes, so the two always agree."""
        tid = state["thread_id"]
        lv = self.level_of(w)
        fig, doc = state.get("figure"), state.get("doc")
        scope = w["request"].get("scope") or ("figure" if (fig or doc) and (state.get("focus") or "figure") == "figure" and fig else "chat")
        self.log(tid, "Building the slides")
        work = self.tdir(tid) / f"slides_{uuid.uuid4().hex[:6]}"
        deck = None
        if scope == "figure" and fig:
            vf = self.video_figure(state, w, tid)
            if vf and vf["kind"] == "slides":
                deck = mm_slides.deck_from_text(vf["title"], lv["label"], vf["slides"], [])
            elif vf:
                deck = mm_slides.deck_from_scenes(vf, self.data_dir, work, mm_media.scenes_for_figure(vf, self.data_dir), self.depth_of(w))
        if deck is None:
            mat = self.discussion_doc(tid, w, lv, "condensed")
            if not mat:
                self.out(tid, w, type="note", text="There is nothing to make slides from yet. Ask me something or make a diagram first.")
                return {}
            blocks, summary, images = mat
            blocks = [dict(b, text="Discussion") if b["t"] == "h1" else b for b in blocks]
            text_slides = mm_doc.slides_from_blocks(blocks, depth=self.depth_of(w))
            work.mkdir(parents=True, exist_ok=True)
            pics = [(str(self.abs(pth)), cap) for pth, cap in images[-4:] if self.abs(pth).exists()]
            deck = mm_slides.deck_from_text("Discussion", lv["label"], text_slides, pics)
        stem = C.safe_name(deck["title"]) or "slides"
        made = []
        try:
            pdf = self.tdir(tid) / f"{stem}_slides_{uuid.uuid4().hex[:4]}.pdf"
            mm_slides.build_pdf(deck, pdf)
            made.append((pdf, "application/pdf"))
        except Exception as e:                                                   # noqa: BLE001
            self.log(tid, f"Slide PDF failed: {type(e).__name__}")
        try:
            pp = self.tdir(tid) / f"{stem}_slides_{uuid.uuid4().hex[:4]}.pptx"
            mm_slides.build_pptx(deck, pp)
            made.append((pp, "application/vnd.openxmlformats-officedocument.presentationml.presentation"))
        except Exception as e:                                                   # noqa: BLE001
            self.log(tid, f"PowerPoint failed: {type(e).__name__}")
            self.out(tid, w, type="note", text="I could not make the PowerPoint version on this computer (the PowerPoint tool is missing). The PDF slides are ready.")
        if not made:
            self.out(tid, w, type="note", text="I could not build the slides on this computer.")
            return {}
        for path_, mime in made:
            self.out(tid, w, type="file", path=self.rel(path_), name=path_.name, mime=mime)
        return {}

    # ---- pdf -----------------------------------------------------------------------------------------------------
    def t_pdf(self, state: State, args: dict, w: dict) -> dict:
        tid = state["thread_id"]
        fig, doc = state.get("figure") or {}, state.get("doc")
        focus = state.get("focus") or ("figure" if fig else "doc")
        if w["request"].get("scope") == "chat":
            fig, doc, focus = {}, None, "doc"
        if any(o["type"] == "file" for o in w["outputs"]):                      # notes / guide already produced its PDF in this request
            return {}
        if focus == "doc" and doc:
            self.out(tid, w, type="file", path=doc["pdf"], name=Path(doc["pdf"]).name, mime="application/pdf")
            return {}
        if focus == "figure" and fig.get("kind") == "process":
            rec = self.process_load(fig["record_path"])
            d = self.doc_save(tid, "process", rec["title"], rec["level"], mm_doc.process_blocks(rec, self.data_dir, rec.get("overview_path")))
            self.out(tid, w, type="file", path=d["pdf"], name=Path(d["pdf"]).name, mime="application/pdf")
            return {}
        if fig:
            explanation = state.get("last_explanation") or ""
            for o in w["outputs"]:
                if o["type"] == "text":
                    explanation = o["text"]
            if not explanation:
                explanation = self.t_explain(dict(state), {"question": ""}, w).get("last_explanation", "")
            # the PDF has its own heading, so use the picture without the title band
            lab = self.render_diagram(tid, fig, titled=False) if fig["kind"] == "diagram" else fig.get("clean_path")
            images = [(str(self.abs(lab)), fig["title"])]
            for g in fig.get("groups", []):
                images.append((str(self.abs(g["image_path"])), f"{g['name']} group"))
            parts = [{"name": p["name"], "colour": p["colour"], "text": p["text"]} for p in fig.get("parts", []) if p["visible"]]
            parts += [{"name": g["name"], "colour": g["colour"], "text": g["text"]} for g in fig.get("groups", [])]
            parts += [{"name": r["name"], "colour": r["colour"], "text": r["text"]} for r in fig.get("regions", [])]
            out_path = self.tdir(tid) / f"{C.safe_name(fig['title'])}_{uuid.uuid4().hex[:4]}.pdf"
            self.log(tid, "Building your PDF")
            mm_pdf.build_pdf(out_path, fig["title"], C.get_level(fig.get("level") or w.get("level"))["label"], images, explanation, parts, fig.get("facts"),
                             fig.get("source_note", ""), fig.get("equation", ""))
            self.out(tid, w, type="file", path=self.rel(out_path), name=out_path.name, mime="application/pdf")
            return {"last_explanation": explanation}
        res = self.t_notes(state, {"mode": "condensed"}, w)                      # no figure or document: PDF of the discussion
        return {"doc": res["doc"]} if res.get("doc") else {}
