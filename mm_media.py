"""
mm_media.py - the "Play lesson" feature: offline English voice + simple animation -> one MP4 shown in the app.

Voice (all run on YOUR PC, no internet):
    1. Piper      - natural-sounding neural voice (pip install piper-tts + one .onnx voice file in understood_data/voices)
    2. Windows SAPI via pyttsx3 - built into Windows, robotic but needs nothing
    3. none       - the lesson still plays with animated captions (and says so)

Animation: for each part the region is spotlighted, a coloured arrow draws in, and the words of the narration light up
as they are spoken.  Frames are drawn with Pillow and piped to ffmpeg (a copy ships with `imageio-ffmpeg`).
"""
from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw

import mm_core as C

SR = 22050


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                                       # noqa: BLE001
        return "ffmpeg"


# =====================================================================================================
# Voice
# =====================================================================================================
class Voice:
    def __init__(self, voices_dir: Optional[Path] = None):
        self.voices_dir = Path(voices_dir) if voices_dir else None
        self._piper = None
        self.engine = self._detect()

    def _piper_model(self) -> Optional[str]:
        env = os.environ.get("UNDERSTOOD_PIPER_VOICE")
        if env and Path(env).exists():
            return env
        if self.voices_dir and self.voices_dir.exists():
            found = sorted(self.voices_dir.glob("*.onnx"))
            return str(found[0]) if found else None
        return None

    def _detect(self) -> str:
        if self._piper_model():
            try:
                import piper  # noqa: F401
                return "piper"
            except Exception:                                               # noqa: BLE001
                pass
        try:
            import pyttsx3  # noqa: F401
            return "sapi"
        except Exception:                                                   # noqa: BLE001
            return "none"

    def synth(self, text: str, out_wav: Path) -> bool:
        """Write `text` as speech to out_wav (PCM wav).  False if no voice is available or it failed."""
        try:
            if self.engine == "piper":
                from piper import PiperVoice
                if self._piper is None:
                    self._piper = PiperVoice.load(self._piper_model())
                with wave.open(str(out_wav), "wb") as wf:
                    try:
                        self._piper.synthesize_wav(text, wf)               # piper-tts >= 1.3
                    except AttributeError:
                        self._piper.synthesize(text, wf)                    # older piper-tts
                return out_wav.exists() and out_wav.stat().st_size > 1000
            if self.engine == "sapi":
                import pyttsx3
                eng = pyttsx3.init()
                eng.setProperty("rate", 165)
                eng.save_to_file(text, str(out_wav))
                eng.runAndWait()
                return out_wav.exists() and out_wav.stat().st_size > 1000
        except Exception:                                                   # noqa: BLE001
            return False
        return False


def _norm_wav(src: Path, dst: Path) -> Optional[bytes]:
    """Convert any audio file to mono 16-bit 22.05 kHz PCM and return its raw frames."""
    r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(dst)],
                       capture_output=True)
    if r.returncode != 0 or not dst.exists():
        return None
    with wave.open(str(dst), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(SR * max(0.0, seconds))


# =====================================================================================================
# Scenes (what is drawn for each narration segment)
# =====================================================================================================
def spot_rect(img: Image.Image, bbox, colour) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    mask = np.zeros(arr.shape[:2], bool)
    mask[max(0, y0):y1, max(0, x0):x1] = True
    arr[~mask] = arr[~mask] * 0.42 + 255 * 0.58 * 0.35
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    ImageDraw.Draw(out).rectangle([x0, y0, x1, y1], outline=tuple(colour), width=5)
    return out


def scenes_for_figure(fig: dict, data_dir: Path) -> list[dict]:
    """[{name, text, colour, pre: PIL, post: PIL}] - pre = before the arrow appears, post = after."""
    kind = fig["kind"]
    out: list[dict] = []
    intro_text = fig.get("lesson_intro") or f"Let's look at {fig['title']}."
    outro_text = fig.get("lesson_outro") or "That is the whole diagram. Look at each label again and try to say what it does."

    if kind == "diagram":
        clean = Image.open(data_dir / fig["clean_path"]).convert("RGB")
        lmap = np.array(Image.open(data_dir / fig["labelmap_path"]))
        parts = [p for p in fig["parts"] if p.get("visible", True)]
        items = [{"name": p["name"], "anchor": tuple(p["anchor"]), "colour": tuple(p["colour"])} for p in parts]
        layout = C.layout_labels(clean.size, items)
        blank = C.render_labels(clean, layout, show=set())
        out.append({"name": "intro", "text": intro_text, "colour": (60, 60, 60), "pre": blank, "post": blank})
        done: list[str] = []
        for p in parts:
            mask = lmap == p["region_id"]
            pre = C.render_labels(clean, layout, show=set(done), spotlight=mask, faded=set(done))
            post = C.render_labels(clean, layout, show=set(done) | {p["name"]}, spotlight=mask, faded=set(done))
            out.append({"name": p["name"], "text": p.get("narration") or p.get("text") or f"This is the {p['name']}.", "short": p.get("text") or "",
                        "colour": tuple(p["colour"]), "pre": pre, "post": post})
            done.append(p["name"])
        full = C.render_labels(clean, layout)
        out.append({"name": "outro", "text": outro_text, "colour": (60, 60, 60), "pre": full, "post": full})

    elif kind == "reaction":
        base = Image.open(data_dir / fig["clean_path"]).convert("RGB")
        out.append({"name": "intro", "text": intro_text, "colour": (60, 60, 60), "pre": base, "post": base})
        for r in fig["regions"]:
            img = spot_rect(base, r["bbox"], r["colour"])
            out.append({"name": r["name"], "text": r.get("text") or r["name"], "colour": tuple(r["colour"]), "pre": img, "post": img})
        out.append({"name": "outro", "text": fig.get("lesson_outro") or "The equation is balanced: each kind of atom appears the same number of times on both sides.",
                    "colour": (60, 60, 60), "pre": base, "post": base})

    elif kind == "process":
        import mm_info
        n = len(fig["steps"])
        cyc = fig.get("shape") == "cycle" and n >= 3
        steps = [(i, st) for i, st in enumerate(fig["steps"]) if st.get("status") == "ready"]
        first = mm_info.compose_overview(fig, data_dir, lit=0)
        out.append({"name": "intro", "text": intro_text, "colour": (60, 60, 60), "pre": first, "post": mm_info.compose_overview(fig, data_dir, lit=n if cyc else n - 1)})
        for i, st in steps:
            col = mm_info.step_colour(i)
            pre = mm_info.step_scene_image(fig, i, data_dir, lit_to=i - 1 if i > 0 else 0)
            post = mm_info.step_scene_image(fig, i, data_dir, lit_to=i)
            out.append({"name": st["title"], "text": st.get("narration") or f"{st['title']}. {st['text']}", "short": st.get("text") or "", "colour": col, "pre": pre, "post": post})
        full = mm_info.compose_overview(fig, data_dir)
        out.append({"name": "outro", "text": fig.get("lesson_outro") or ("And then the cycle starts again." if cyc else "That is the whole process, from the first step to the last."),
                    "colour": (60, 60, 60), "pre": full, "post": full})

    elif kind == "slides":
        for k, sl in enumerate(fig["slides"]):
            col = C.colour_for(k)
            title_only = slide_image(sl["title"], [], 0, col)
            full = slide_image(sl["title"], sl["bullets"], len(sl["bullets"]), col)
            out.append({"name": sl["title"], "text": sl["narration"], "colour": col, "pre": title_only, "post": full})

    else:                                                                   # molecule
        base = Image.open(data_dir / fig["clean_path"]).convert("RGB")
        out.append({"name": "intro", "text": intro_text, "colour": (60, 60, 60), "pre": base, "post": base})
        for g in fig.get("groups", []):
            img = Image.open(data_dir / g["image_path"]).convert("RGB")
            out.append({"name": g["name"], "text": g.get("text") or g["name"], "colour": tuple(g["colour"]), "pre": img, "post": img})
        out.append({"name": "outro", "text": fig.get("lesson_outro") or "Functional groups decide how a molecule behaves.", "colour": (60, 60, 60), "pre": base, "post": base})
    return out


def slide_image(title: str, bullets: list[str], shown: int, colour, w: int = 1200, h: int = 540) -> Image.Image:
    """A text slide drawn by code: big title, then the bullets (first `shown` of them)."""
    img = Image.new("RGB", (w, h), (252, 252, 254))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 18, h], fill=tuple(colour))
    big, small = C.get_font(52, True), C.get_font(32)
    words = title.split()
    lines = _wrap(words, big, d, w - 120)[:2]
    y = 40 if bullets else h // 2 - 40 * len(lines)
    for ln in lines:
        d.text((60, y), " ".join(words[i] for i in ln), font=big, fill=(30, 40, 60))
        y += 64
    y += 20
    for b in bullets[:shown]:
        bw = b.split()
        bl = _wrap(bw, small, d, w - 160)[:2]
        d.ellipse([62, y + 12, 78, y + 28], fill=tuple(colour))
        for ln in bl:
            d.text((96, y), " ".join(bw[i] for i in ln), font=small, fill=(45, 45, 52))
            y += 42
        y += 10
    return img


# =====================================================================================================
# Frame composition
# =====================================================================================================
FW, FH, TITLE_H, CAP_H = 1280, 820, 58, 190


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    s = min(w / img.width, h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)


def _wrap(words: list[str], font, d, max_w: int) -> list[list[int]]:
    lines, cur, w = [], [], 0.0
    sp = d.textlength(" ", font=font)
    for i, wd in enumerate(words):
        ww = d.textlength(wd, font=font)
        if cur and w + sp + ww > max_w:
            lines.append(cur)
            cur, w = [], 0.0
        w += (sp if cur else 0) + ww
        cur.append(i)
    if cur:
        lines.append(cur)
    return lines


def compose_frame(fig_img: Image.Image, title: str, text: str, n_spoken: int, colour, total_words: Optional[int] = None) -> Image.Image:
    frame = Image.new("RGB", (FW, FH), "white")
    d = ImageDraw.Draw(frame)
    d.rectangle([0, 0, FW, TITLE_H], fill=(245, 247, 250))
    d.text((24, TITLE_H / 2), title, font=C.get_font(28, True), fill=(40, 40, 40), anchor="lm")
    area_h = FH - TITLE_H - CAP_H
    fit = _fit(fig_img, FW - 40, area_h - 10)
    frame.paste(fit, ((FW - fit.width) // 2, TITLE_H + (area_h - fit.height) // 2))
    y0 = FH - CAP_H
    d.rectangle([0, y0, FW, FH], fill=(250, 250, 252))
    d.line([0, y0, FW, y0], fill=(220, 220, 225), width=2)
    font = C.get_font(30, True)
    words = text.split()
    all_lines = _wrap(words, font, d, FW - 80)
    cur = next((k for k, ln in enumerate(all_lines) if (n_spoken - 1) in ln), 0) if n_spoken else 0
    start = max(0, min(cur - 1, len(all_lines) - 4))                 # long narration scrolls: the line being spoken stays in view
    lines = all_lines[start:start + 4]
    sp = d.textlength(" ", font=font)
    y = y0 + 22
    for ln in lines:
        x = 40
        for i in ln:
            wd = words[i]
            ww = d.textlength(wd, font=font)
            if i == n_spoken - 1:
                d.rounded_rectangle([x - 5, y - 2, x + ww + 5, y + 38], radius=6, fill=tuple(colour))
                d.text((x, y), wd, font=font, fill="white")
            else:
                d.text((x, y), wd, font=font, fill=(35, 35, 35) if i < n_spoken else (160, 160, 165))
            x += ww + sp
        y += 42
    return frame


# =====================================================================================================
# The lesson
# =====================================================================================================
def make_lesson(fig: dict, data_dir: Path, out_path: Path, voice: Voice, fps: int = 10,
                progress: Optional[Callable[[str, float], None]] = None) -> dict:
    """Build the MP4.  Returns {path, seconds, voice: engine used, segments}.  Raises RuntimeError on ffmpeg failure."""
    def say(msg, frac=0.0):
        if progress:
            progress(msg, frac)

    work = Path(out_path).parent / (Path(out_path).stem + "_work")
    work.mkdir(parents=True, exist_ok=True)
    say("Preparing the scenes", 0.02)
    scenes = scenes_for_figure(fig, data_dir)

    pcm, durations, spoke = [], [], False
    for i, sc in enumerate(scenes):
        say(f"Recording narration {i + 1} of {len(scenes)}", 0.05 + 0.25 * i / len(scenes))
        raw = work / f"seg{i}_raw.wav"
        norm = work / f"seg{i}.wav"
        frames = None
        if voice.engine != "none" and voice.synth(sc["text"], raw):
            frames = _norm_wav(raw, norm)
        if frames:
            spoke = True
        else:
            frames = _silence(max(2.5, len(sc["text"].split()) / 2.6))
        pcm.append(frames)
        durations.append(len(frames) / 2 / SR)

    tl = C.build_timeline([{"name": sc["name"], "text": sc["text"], "dur": d} for sc, d in zip(scenes, durations)])
    track = [_silence(tl["lead"])]
    for k, (it, fr) in enumerate(zip(tl["items"], pcm)):
        track.append(fr)
        track.append(_silence(0.35 if k < len(pcm) - 1 else 1.2))
    audio = work / "track.wav"
    with wave.open(str(audio), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"".join(track))

    total, nframes = tl["total"], int(tl["total"] * fps)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FW}x{FH}", "-r", str(fps), "-i", "-",
           "-i", str(audio), "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
           "-shortest", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    title = fig.get("title", "Lesson")
    try:
        for f in range(nframes):
            t = f / fps
            seg = None
            for k, it in enumerate(tl["items"]):
                if it["start"] <= t < it["end"] + 0.35:
                    seg = k
                    break
            if seg is None:
                seg = 0 if t < tl["lead"] else len(scenes) - 1
            it, sc = tl["items"][seg], scenes[seg]
            frac = (t - it["start"]) / max(0.1, it["end"] - it["start"])
            n_spoken = C.word_progress(sc["text"], frac) if t >= it["start"] else 0
            img = sc["post"] if (t - it["start"]) > 0.35 else sc["pre"]
            frame = compose_frame(img, title, sc["text"], n_spoken, sc["colour"])
            proc.stdin.write(frame.tobytes())
            if f % max(1, nframes // 20) == 0:
                say("Drawing the animation", 0.32 + 0.66 * f / nframes)
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.wait(timeout=120)
    except Exception:
        proc.kill()
        raise
    if proc.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError("video encoder failed: " + err[-300:])
    say("Done", 1.0)
    return {"path": str(out_path), "seconds": round(total), "voice": voice.engine if spoke else "none", "segments": len(scenes)}
