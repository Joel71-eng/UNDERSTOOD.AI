"""
mm_server.py - the stateless GPU server for "Understood" (runs in Colab).  The PC owns the conversation; this
server only does work and forgets it.

Two capabilities, three endpoints
    POST /tutor   chat messages (text + optional images) -> text            [vision-language tutor, stays loaded]
    POST /choose  verify: image + question + options -> probability of each option   [reads the tutor's own
                  answer-letter probabilities, so a wrong guess shows up as low confidence instead of a confident lie]
    POST /image   queue an image job (generate | edit)  -> {job_id}         [image model, loaded in stages]
    GET  /jobs/ID poll a job (status, progress, image when done)
    GET  /health  status        POST /warmup  load the tutor now

Memory plan for a 15 GB T4 (all models are the 4-bit copies you saved to Drive):
    tutor (kept loaded)                       ~2.5 GB
    image job, stage 1: text encoder only     ~6 GB   -> encode the prompt, free it
    image job, stage 2: image transformer     ~5-6 GB -> denoise (+ small LoRA), free it
If the 'staged' path fails for edits, the tutor is unloaded for that one job and reloaded afterwards.

Run:  python mm_server.py                    (real models)
      UNDERSTOOD_MOCK=1 python mm_server.py  (fake models: test the whole stack with no GPU)
"""
from __future__ import annotations

import base64
import gc
import io
import os
import queue
import random
import re
import threading
import time
import traceback
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

MOCK = os.environ.get("UNDERSTOOD_MOCK", "0") == "1"
VERSION = "2.2"
BUILD = "2026-10-29 cancel+log"
FEATURES = ["choose", "cancel"]                                              # the PC checks this to know the verification endpoint exists
LOG_LEVEL = os.environ.get("UMM_LOG", "warning")                   # "info" shows every request line (noisy: /jobs is polled)

MODELS_DIR = os.environ.get("UMM_MODELS", "/content/models_local")
TUTOR_DIR = os.environ.get("UMM_TUTOR", f"{MODELS_DIR}/tutor")
IMAGE_DIR = os.environ.get("UMM_IMAGE", f"{MODELS_DIR}/imagegen")
LORA_DIR = os.environ.get("UMM_LORA", "/content/drive/MyDrive/understood_mm/models/lightning_lora")
USE_LORA = os.environ.get("UMM_USE_LORA", "1") == "1"
FAST_STEPS = int(os.environ.get("UMM_FAST_STEPS", "4"))            # the speed-up LoRA's own recipe
BETTER_STEPS = int(os.environ.get("UMM_BETTER_STEPS", "8"))
EDIT_QUALITY = os.environ.get("UMM_EDIT_QUALITY", "better")        # edits are less forgiving than new images
TUTOR_MAX_PIXELS = int(os.environ.get("UMM_TUTOR_MAX_PIXELS", str(1280 * 28 * 28)))

GPU_LOCK = threading.Lock()
app = FastAPI(title="Understood GPU server", version=VERSION)


def b64_to_img(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def lora_strength(steps: int) -> float:
    """Full strength at the LoRA's own 4 steps, weaker as steps grow (heuristic)."""
    return round(max(0.15, min(1.0, (4 / max(1, steps)) ** 0.5)), 2)


# =====================================================================================================
# Real models
# =====================================================================================================
if not MOCK:
    import torch

    def _dtype():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def free_gpu():
        gc.collect()
        torch.cuda.empty_cache()

    class Tutor:
        def __init__(self):
            self.model = None
            self.proc = None

        @property
        def loaded(self) -> bool:
            return self.model is not None

        def load(self):
            if self.loaded:
                return
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(TUTOR_DIR, torch_dtype=_dtype(), device_map={"": 0})
            self.proc = AutoProcessor.from_pretrained(TUTOR_DIR, min_pixels=128 * 28 * 28, max_pixels=TUTOR_MAX_PIXELS)

        def unload(self):
            self.model = None
            self.proc = None
            free_gpu()

        def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> tuple[str, str]:
            from qwen_vl_utils import process_vision_info
            self.load()
            msgs = []
            for m in messages:
                c = m["content"]
                if isinstance(c, str):
                    msgs.append({"role": m["role"], "content": re.sub(r"^TASK:[A-Z_]+\n", "", c)})
                else:
                    parts = []
                    for p in c:
                        if p.get("type") == "image":
                            parts.append({"type": "image", "image": b64_to_img(p["b64"])})
                        else:
                            parts.append({"type": "text", "text": p.get("text", "")})
                    msgs.append({"role": m["role"], "content": parts})
            text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inputs = self.proc(text=[text], images=imgs or None, videos=vids or None, padding=True, return_tensors="pt").to("cuda")
            kw = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
            if temperature > 0:
                kw.update(temperature=temperature, top_p=0.9, repetition_penalty=1.05)
            with torch.no_grad():
                out = self.model.generate(**inputs, **kw)
            new = out[:, inputs.input_ids.shape[1]:]
            answer = self.proc.batch_decode(new, skip_special_tokens=True)[0].strip()
            tok = self.proc.tokenizer
            eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
            finish = "length" if (new.shape[1] >= max_new_tokens and int(new[0, -1]) not in eos) else "stop"
            return answer, finish

        def choose(self, items: list[dict], perms: int, seed: int) -> list[dict]:
            """For each item (image + question + options) return the probability of every option.

            The answer is NOT generated text: we read the model's next-token distribution over the answer letters A, B, C...
            - a model that is unsure shows it (flat probabilities) instead of writing a confident wrong answer;
            - the options are rotated between passes, so the habit of favouring a position cancels out;
            - `mass` is how much of the model's probability went to the letters at all (low = it wanted to say something else).
            """
            from qwen_vl_utils import process_vision_info
            self.load()
            tok = self.proc.tokenizer
            rnd = random.Random(seed)
            out = []
            for it in items:
                opts = [str(o) for o in it["options"]]
                n = len(opts)
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n]
                ids = [tok.encode(ch, add_special_tokens=False)[0] for ch in letters]
                base = list(range(n))
                rnd.shuffle(base)
                acc, mass = [0.0] * n, 0.0
                for k in range(perms):
                    order = base[k % n:] + base[:k % n]                      # rotation: every option visits different positions
                    lines = "\n".join(f"{letters[pos]}. {opts[i]}" for pos, i in enumerate(order))
                    content = []
                    if it.get("image_b64"):
                        content.append({"type": "image", "image": b64_to_img(it["image_b64"])})
                    content.append({"type": "text", "text": f"{it['question']}\n{lines}\nAnswer with the letter only."})
                    msgs = [{"role": "user", "content": content}]
                    text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    imgs, vids = process_vision_info(msgs)
                    inputs = self.proc(text=[text], images=imgs or None, videos=vids or None, padding=True, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        try:
                            logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1]
                        except TypeError:                                    # older transformers: no logits_to_keep
                            logits = self.model(**inputs, use_cache=False).logits[0, -1]
                    sel = torch.softmax(logits.float(), -1)[ids]
                    mass += float(sel.sum()) / perms
                    p = (sel / sel.sum().clamp_min(1e-9)).cpu().tolist()
                    for pos, i in enumerate(order):
                        acc[i] += p[pos] / perms
                out.append({"probs": [round(x, 4) for x in acc], "mass": round(mass, 3)})
            return out

    TUTOR = Tutor()

    class ImageStack:
        """Image generation / editing in two small stages so it fits next to the tutor on a 15 GB GPU."""

        def __init__(self):
            self.staged_ok = True                      # flips to False the first time staged loading fails

        def _encode(self, prompt: str, image: Optional[Image.Image]):
            from diffusers import QwenImage21Pipeline
            pe = QwenImage21Pipeline.from_pretrained(IMAGE_DIR, transformer=None, vae=None, torch_dtype=_dtype(), device_map="cuda")
            try:
                with torch.no_grad():
                    kw = dict(prompt=prompt, device="cuda")
                    if image is not None:
                        kw["image"] = image
                    res = pe.encode_prompt(**kw)
                return res[0].to("cpu"), (res[1].to("cpu") if res[1] is not None else None)
            finally:
                del pe
                free_gpu()

        def _call_args(self, req: "ImageReq", steps: int, progress, image):
            import numpy as np
            kw = dict(width=req.width, height=req.height, num_inference_steps=steps, true_cfg_scale=1.0,
                      generator=torch.Generator("cuda").manual_seed(req.seed))
            if image is not None:
                kw["image"] = image

            def cb(p, i, t, kwargs):
                progress(i + 1, steps)
                return kwargs
            kw["callback_on_step_end"] = cb
            return kw

        def _attach_lora(self, pipe, steps: int, kw: dict, use_lora: bool, strength: Optional[float]):
            if not (use_lora and USE_LORA and os.path.exists(os.path.join(LORA_DIR, "pytorch_lora_weights.safetensors"))):
                return
            import numpy as np
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe.load_lora_weights(LORA_DIR, weight_name="pytorch_lora_weights.safetensors", adapter_name="fast")
            pipe.set_adapters(["fast"], adapter_weights=[strength if strength is not None else lora_strength(steps)])
            if steps <= 8:
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift_terminal=None)
                kw["sigmas"] = [float(x) for x in np.linspace(1.0, 1.0 / steps, steps)]

        def run(self, req: "ImageReq", progress) -> Image.Image:
            from diffusers import QwenImage21Pipeline
            image = b64_to_img(req.image_b64) if req.image_b64 else None
            quality = EDIT_QUALITY if req.mode == "edit" else req.quality
            steps = req.steps or (FAST_STEPS if quality == "fast" else BETTER_STEPS)
            if self.staged_ok:
                try:
                    embeds, mask = self._encode(req.prompt, image)
                    pipe = QwenImage21Pipeline.from_pretrained(IMAGE_DIR, text_encoder=None, torch_dtype=_dtype(), device_map="cuda")
                    try:
                        pipe.vae.enable_tiling()
                        pipe.vae.enable_slicing()
                        kw = self._call_args(req, steps, progress, image)
                        self._attach_lora(pipe, steps, kw, req.use_lora, req.lora_strength)
                        kw.update(prompt_embeds=embeds.to("cuda"), prompt_embeds_mask=(mask.to("cuda") if mask is not None else None))
                        return pipe(**kw).images[0]
                    finally:
                        del pipe
                        free_gpu()
                except Exception:                                                    # noqa: BLE001
                    traceback.print_exc()
                    free_gpu()
                    if req.mode != "edit" and TUTOR.loaded is False:
                        raise
                    self.staged_ok = False                                           # fall through to the all-in-one path
            # all-in-one path: needs the room, so the tutor steps aside for this job
            TUTOR.unload()
            pipe = QwenImage21Pipeline.from_pretrained(IMAGE_DIR, torch_dtype=_dtype(), device_map="cuda")
            try:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
                kw = self._call_args(req, steps, progress, image)
                kw["prompt"] = req.prompt
                self._attach_lora(pipe, steps, kw, req.use_lora, req.lora_strength)
                return pipe(**kw).images[0]
            finally:
                del pipe
                free_gpu()

    STACK = ImageStack()

    def gpu_info() -> dict:
        free, total = torch.cuda.mem_get_info()
        return {"gpu": torch.cuda.get_device_name(0), "vram_free_gb": round(free / 2**30, 1), "vram_total_gb": round(total / 2**30, 1)}

else:
    # ---------------------------------------------------------------------------------------------
    # Mock models: same endpoints, no GPU, no downloads.  Used by the tests.
    # ---------------------------------------------------------------------------------------------
    class _MockTutor:
        loaded = True

        def load(self): pass

        def unload(self): pass

        def generate(self, messages, max_new_tokens, temperature):
            return mock_tutor(messages), "stop"

        def choose(self, items, perms, seed):
            return mock_choose(items, perms, seed)

    TUTOR = _MockTutor()

    # What the mock "sees": every part is drawn in its own colour, so a region's colour tells the mock which part it is.
    # UNDERSTOOD_MOCK_NAMING:  right (default) = names and checks are correct | shifted = free names are wrong but the
    # check is right (tests that verification catches errors) | blind = cannot name anything | unsure = checks are flat.
    MOCK_STATE: dict = {"parts": []}
    NEUTRAL = (222, 222, 222)

    def mock_palette(n: int) -> list:
        import colorsys
        return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / max(1, n), 0.55, 0.92)) for i in range(n)]

    def mock_part_of(image_b64: Optional[str]) -> Optional[int]:
        """Index of the part whose colour dominates the un-faded pixels of a highlighted picture, or None."""
        import numpy as np
        parts = MOCK_STATE["parts"]
        if not image_b64 or not parts:
            return None
        a = np.asarray(b64_to_img(image_b64)).astype(int)
        best, best_n = None, 250
        for i, c in enumerate(mock_palette(len(parts))):
            n = int((np.abs(a - np.array(c)).sum(-1) < 36).sum())
            if n > best_n:
                best, best_n = i, n
        return best

    def mock_choose(items, perms, seed) -> list:
        import mm_core as C
        mode = os.environ.get("UNDERSTOOD_MOCK_NAMING", "right")
        out = []
        for it in items:
            opts = it["options"]
            n = len(opts)
            if mode == "unsure":
                out.append({"probs": [round(1.0 / n, 4)] * n, "mass": 0.9})
                continue
            if it["question"].startswith("Is this picture"):                 # reference-picture check
                out.append({"probs": [0.9] + [0.1 / (n - 1)] * (n - 1), "mass": 0.9})
                continue
            idx = mock_part_of(it.get("image_b64"))
            truth = MOCK_STATE["parts"][idx] if idx is not None else None
            negative = [i for i, o in enumerate(opts) if C._norm(o).startswith(("no ", "none"))]
            hit = next((i for i, o in enumerate(opts) if truth and i not in negative and C._norm(truth) in C._norm(o.split(" (")[0])), None)
            pick = hit if hit is not None else (negative[0] if negative else 0)
            probs = [0.1 / max(1, n - 1)] * n
            probs[pick] = 0.9
            out.append({"probs": [round(p, 4) for p in probs], "mass": 0.9})
        return out

    def gpu_info() -> dict:
        return {"gpu": "mock", "vram_free_gb": 15.0, "vram_total_gb": 15.0}

    def _user_text(messages) -> tuple[str, str, list[str]]:
        system = next((m["content"] for m in messages if m["role"] == "system" and isinstance(m["content"], str)), "")
        last = [m for m in messages if m["role"] == "user"][-1]["content"]
        imgs = []
        if isinstance(last, list):
            imgs = [p["b64"] for p in last if p.get("type") == "image"]
            last = " ".join(p.get("text", "") for p in last if p.get("type") == "text")
        return system, last, imgs

    def mock_tutor(messages) -> str:
        import json
        import mm_core as C
        system, user, imgs = _user_text(messages)
        m = re.match(r"TASK:([A-Z_]+)", system)
        task = m.group(1) if m else "CHAT"
        heads = re.findall(r"## ([^,.\n]+)", system)
        if task in ("EXPLAIN", "REFORMAT", "CHAT") and len(heads) >= 2:          # honour the headings the engine asked for
            return "\n\n".join(f"## {h.strip()}\nA concrete answer about: {' '.join(user.split()[:10])}" for h in heads)
        if task == "ROUTE":
            t = re.search(r"Request:\s*(.+)", user)
            steps = C.rule_route(t.group(1).strip() if t else user, C.RouteCtx())
            if steps is None:
                req_t = t.group(1).strip() if t else user
                topic_ = C.extract_topic(req_t)
                steps = [{"tool": "clarify", "args": {"topic": topic_}}] if (C.RE_DRAW_START.search(req_t) and topic_) else [{"tool": "chat", "args": {}}]
            return json.dumps({"steps": steps, "level": None})
        if task == "SYSTEM":
            m_ = re.search(r"\b(brain|kidney|heart|eye|liver|lung|stomach|ear|skin)\b", user.lower())
            return m_.group(1) if m_ else "NONE"
        if task == "PROCESS_PLAN":
            names = re.findall(r"^- ([^:\n]+):", user.split("Parts you may use:")[-1].split("Text:")[0], re.M)
            wander = os.environ.get("UNDERSTOOD_MOCK_PLAN", "") == "wander" and "Do not mention" not in user
            steps = []
            for k in range(max(1, min(4, len(names) - 1))):
                a_, b_ = names[k], names[(k + 1) % len(names)]
                txt = f"The {a_} passes the signal on to the {b_}."
                if wander and k == 0:
                    txt = f"The spinal cord sends the signal on to the {a_}."
                steps.append({"title": f"Step {k + 1}", "text": txt, "parts": [a_, b_], "flows": [{"from": a_, "to": b_, "what": "signal"}], "scene": f"{a_} and {b_}"})
            return json.dumps({"shape": "linear", "steps": steps})
        if task == "UNDERSTAND":
            m_ = re.search(r"structures:\s*(.+?)\.\s", user + " ")
            return "I can see these structures: " + (m_.group(1) if m_ else "several distinct parts") + ". Each has its own colour and position."
        if task == "LOCATE":                                                   # one call: name every numbered circle (mock: read the colour under each circle)
            import numpy as np
            from scipy import ndimage as ndi
            mode = os.environ.get("UNDERSTOOD_MOCK_NAMING", "right")
            parts_ = MOCK_STATE["parts"]
            if not imgs or not parts_ or mode == "blind":
                return "{}"
            a = np.asarray(b64_to_img(imgs[0])).astype(int)
            H, W = a.shape[:2]
            lab, n = ndi.label(a.min(-1) >= 250)
            edge = set(lab[0].tolist()) | set(lab[-1].tolist()) | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
            sc = W / max(1, MOCK_STATE.get("size", W))
            blobs = []
            for k in range(1, n + 1):
                if k in edge:
                    continue
                ys, xs = np.where(lab == k)
                if len(xs) >= 150:
                    blobs.append((float(ys.mean()), float(xs.mean()), (len(xs) / 3.1416) ** 0.5))
            blobs.sort(key=lambda b_: (round(b_[0] / sc / 40), b_[1]))
            pal = mock_palette(len(parts_))
            out_ = {}
            for i, (cy, cx, r) in enumerate(blobs, 1):
                votes = {}
                for ang in range(0, 360, 20):
                    import math
                    for rr in (r + 6, r + 9):
                        x_, y_ = int(cx + rr * math.cos(math.radians(ang))), int(cy + rr * math.sin(math.radians(ang)))
                        if 0 <= x_ < W and 0 <= y_ < H:
                            for j, c in enumerate(pal):
                                if np.abs(a[y_, x_] - np.array(c)).sum() < 40:
                                    votes[j] = votes.get(j, 0) + 1
                j = max(votes, key=votes.get) if votes else None
                if j is None:
                    out_[str(i)] = None
                elif mode == "collapse":
                    out_[str(i)] = parts_[0]
                elif mode == "shifted":
                    out_[str(i)] = parts_[(j + 1) % len(parts_)]
                else:
                    out_[str(i)] = parts_[j]
            return json.dumps(out_)
        if task == "NAME_REGION":
            mode = os.environ.get("UNDERSTOOD_MOCK_NAMING", "right")
            parts_ = MOCK_STATE["parts"]
            idx = mock_part_of(imgs[0]) if imgs else None
            if mode == "blind" or idx is None or not parts_:
                return "a round shape" if mode == "blind" else "unclear"
            if mode == "collapse":                                   # real 3B behaviour seen in the log: one answer for every region
                return parts_[0]
            return parts_[(idx + 1) % len(parts_)] if mode == "shifted" else parts_[idx]
        if task == "EDIT_PROMPT":
            m_ = re.search(r"Latest instruction:\s*(.+)", user)
            return m_.group(1).strip() if m_ else user.strip()
        if task == "PARTS":
            words = [w for w in re.findall(r"[A-Za-z]{5,}", user.split("Text:")[-1])]
            seen, out = set(), []
            for w in words:
                if w.lower() not in seen:
                    seen.add(w.lower())
                    out.append(w.lower())
                if len(out) >= 6:
                    break
            return json.dumps(out)
        if task == "NAME":
            names = [x.strip() for x in re.findall(r"(?:^|;\s*)([a-z][a-z' \-]+?)(?:\s*\(|;|\s*\?|$)", user.split("structures:")[-1].split("Answer only")[0])]
            names = [n for n in names if n]
            if imgs:
                regs, _ = C.find_regions(b64_to_img(imgs[0]))
                ids = [r["id"] for r in regs]
            else:
                ids = []
            if "the structure is" in user.lower():
                return str(ids[0]) if ids else "none"
            return json.dumps({n: (ids[i] if i < len(ids) else None) for i, n in enumerate(names)})
        if task == "NARRATE" and '"intro"' in system:
            return json.dumps({"intro": "This is an overview of the topic and why it matters to anyone studying it.",
                               "recap": "Together these parts work as one system, and each one depends on the others to do its job."})
        if task == "NARRATE":
            names = re.findall(r"^- (.+?):", user, re.M)
            return json.dumps({n: f"The {n} is an important part, and here it does its job in the whole system." for n in names})
        if task == "PROCESS":
            sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", user.split("Text:")[-1]) if len(x.split()) >= 5][:6]
            return json.dumps({"shape": "linear", "steps": [{"title": " ".join(x.split()[:3]), "text": x, "scene": " ".join(x.split()[:6])} for x in sents]})
        if task == "TERMS":
            words = []
            for w_ in re.findall(r"[A-Za-z]{7,}", user.split("Notes:")[-1]):
                if w_.lower() not in [x.lower() for x in words]:
                    words.append(w_)
                if len(words) >= 4:
                    break
            return json.dumps([{"term": w_.lower(), "meaning": f"{w_.lower()} is explained in the notes of this topic."} for w_ in words])
        if task == "QUESTIONS":
            sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", user.split("Notes:")[-1]) if len(x.split()) >= 6][:4]
            return json.dumps([{"q": f"What do the notes say about {' '.join(x.split()[:2])}?", "a": x} for x in sents])
        if task == "NOTES":
            return "## What we covered\nWe talked about the topics in this chat.\n\n## Key points\n- The main idea was explained.\n- The parts were named.\n\n## Questions to revisit\n- Can you explain it in your own words?"
        if task == "GUIDE":
            return "## Overview\nA short overview of the topic.\n\n## How it works\nThe steps happen in order.\n\n## Why it matters\nIt matters for living things."
        if task == "EXPLAIN":
            return ("## What you see\nA labelled diagram.\n\n## The parts and what they do\nEach labelled part has its own job.\n\n"
                    "## Why it matters\nThe parts work together.\n\n## Check yourself\nCan you name each part from memory?")
        if task == "EQUATION":
            return "CH4 + O2 -> CO2 + H2O"
        if task == "SMILES":
            return "CCO"
        if task == "SUMMARY":
            return " ".join(user.split()[:30])
        if task == "REFORMAT":
            return ("## Short answer\nHere is the answer.\n\n## Explanation\nIt works like this.\n\n## Key points\n- Point one\n\n## Try this\nTry an example.")
        return "Here is a short, friendly answer from the tutor."

    def mock_image(req: "ImageReq", progress) -> Image.Image:
        W, H = req.width, req.height
        if req.image_b64:
            base = b64_to_img(req.image_b64).resize((W, H))
            tint = Image.new("RGB", (W, H), (230, 240, 255))
            for i in range(3):
                progress(i + 1, 3)
                time.sleep(0.05)
            return Image.blend(base, tint, 0.35)
        m = re.search(r"showing (.+?), white background", req.prompt)
        parts = [p.strip() for p in m.group(1).split(",")] if m else []
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        rnd = random.Random(req.seed)
        pal = [(150, 205, 120), (140, 90, 170), (40, 150, 60), (215, 235, 250), (240, 180, 90), (230, 120, 120), (120, 170, 230), (200, 200, 90)]
        if not parts:
            for i in range(4):
                x, y = rnd.randint(40, W - 200), rnd.randint(40, H - 200)
                d.ellipse([x, y, x + 150, y + 150], fill=pal[i % len(pal)], outline=(20, 20, 20), width=4)
        else:
            import math
            n = len(parts)
            MOCK_STATE["parts"] = list(parts)                               # part i is drawn in colour i (see mock_part_of)
            MOCK_STATE["size"] = int(req.width)
            cols = mock_palette(n)
            S = min(W, H)
            d.ellipse([int(W * .06), int(H * .08), int(W * .94), int(H * .92)], fill=cols[0], outline=(20, 20, 20), width=5)
            r = int(min(S * 0.075, 0.4 * 2 * math.pi * 0.27 * S / max(1, n - 1)))
            for i in range(1, n):
                ang = 2 * math.pi * i / max(1, n - 1)
                cx, cy = W / 2 + 0.28 * W * math.cos(ang), H / 2 + 0.26 * H * math.sin(ang)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cols[i], outline=(20, 20, 20), width=4)
            d.ellipse([W / 2 - 55, H / 2 - 55, W / 2 + 55, H / 2 + 55], fill=NEUTRAL, outline=(20, 20, 20), width=4)   # a region that is no listed part
        steps = 4
        for i in range(steps):
            progress(i + 1, steps)
            time.sleep(float(os.environ.get("UNDERSTOOD_MOCK_STEP", "0.05")))
        return img


# =====================================================================================================
# Jobs
# =====================================================================================================
class ImageReq(BaseModel):
    mode: str = Field("generate", pattern="^(generate|edit)$")
    prompt: str
    width: int = 768
    height: int = 768
    quality: str = "fast"                      # fast | better
    seed: int = Field(default_factory=lambda: random.randint(0, 2**31 - 1))
    steps: Optional[int] = None
    use_lora: bool = True
    lora_strength: Optional[float] = None
    image_b64: Optional[str] = None


class TutorReq(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 400
    temperature: float = 0.0


class ChooseItem(BaseModel):
    image_b64: Optional[str] = None
    question: str
    options: list[str] = Field(min_length=2, max_length=12)


class ChooseReq(BaseModel):
    items: list[ChooseItem] = Field(min_length=1, max_length=16)
    perms: int = Field(2, ge=1, le=4)                  # passes with rotated option order (2 is enough to cancel bias for a yes/no)
    seed: int = 0


class JobCancelled(Exception):
    pass


def log(msg: str):
    """One line in server.log (Colab cell 8 shows it): what the server is doing, with a timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


JOBS: dict[str, dict] = {}
QUEUE: "queue.Queue[tuple[str, ImageReq]]" = queue.Queue()
JOB_TTL = 3600


def _clean_jobs():
    now = time.time()
    for k in [k for k, j in JOBS.items() if now - j["created"] > JOB_TTL and j["status"] in ("done", "error")]:
        JOBS.pop(k, None)


def worker():
    while True:
        job_id, req = QUEUE.get()
        job = JOBS[job_id]
        if job.get("cancel"):                                                        # stopped while it was still waiting in the queue
            job.update(status="cancelled")
            log(f"image job {job_id} cancelled before it started")
            continue
        job["status"] = "running"
        t0 = time.time()
        log(f"image job {job_id} started: {req.mode} {req.width}x{req.height}")
        try:
            def progress(i, n):
                job["progress"], job["steps"] = i, n
                if job.get("cancel"):
                    raise JobCancelled()                                             # raised inside the denoising loop: the GPU is released at once
            with GPU_LOCK:
                req.width, req.height = max(256, req.width - req.width % 16), max(256, req.height - req.height % 16)
                img = mock_image(req, progress) if MOCK else STACK.run(req, progress)
            job.update(status="done", image_b64=img_to_b64(img), seed=req.seed)
            log(f"image job {job_id} done in {time.time() - t0:.0f}s")
        except JobCancelled:
            job.update(status="cancelled")
            log(f"image job {job_id} CANCELLED after {time.time() - t0:.0f}s")
            if not MOCK:
                try:
                    free_gpu()
                except Exception:                                                    # noqa: BLE001
                    pass
        except Exception as e:                                                       # noqa: BLE001
            traceback.print_exc()
            job.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
            if not MOCK:
                try:
                    free_gpu()
                except Exception:                                                    # noqa: BLE001
                    pass


threading.Thread(target=worker, daemon=True).start()


# =====================================================================================================
# Endpoints
# =====================================================================================================
@app.middleware("http")
async def _request_log(request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    p = request.url.path
    if p != "/health" and not (p.startswith("/jobs/") and not p.endswith("/cancel")):      # job polling is too chatty to log
        log(f"{request.method} {p} -> {resp.status_code} in {time.time() - t0:.1f}s")
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "build": BUILD, "features": FEATURES, "mock": MOCK, "tutor_loaded": bool(getattr(TUTOR, "loaded", False)),
            "queue": QUEUE.qsize(), "busy": GPU_LOCK.locked(), **gpu_info()}


@app.post("/warmup")
def warmup():
    t0 = time.time()
    with GPU_LOCK:
        TUTOR.load()
    return {"status": "ready", "seconds": round(time.time() - t0, 1), **gpu_info()}


@app.post("/tutor")
def tutor(req: TutorReq):
    if not req.messages:
        raise HTTPException(400, "messages is empty")
    first = req.messages[0].get("content", "") if req.messages else ""
    task = first.split("\n")[0] if isinstance(first, str) and first.startswith("TASK:") else "chat"
    n_img = sum(1 for m in req.messages if isinstance(m.get("content"), list) for p in m["content"] if isinstance(p, dict) and p.get("type") == "image")
    log(f"tutor {task}: {len(req.messages)} messages, {n_img} image(s), up to {req.max_new_tokens} tokens")
    with GPU_LOCK:
        text, finish = TUTOR.generate(req.messages, min(req.max_new_tokens, 900), req.temperature)
    return {"text": text, "finish_reason": finish}


@app.post("/choose")
def choose(req: ChooseReq):
    """Verification: probability of each option for every item (one request = up to 16 items, one GPU turn)."""
    items = [i.model_dump() for i in req.items]
    with GPU_LOCK:
        results = TUTOR.choose(items, req.perms, req.seed)
    return {"results": results}


@app.post("/image")
def image(req: ImageReq):
    if req.mode == "edit" and not req.image_b64:
        raise HTTPException(400, "edit needs image_b64")
    _clean_jobs()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "progress": 0, "steps": 1, "created": time.time()}
    QUEUE.put((job_id, req))
    return {"job_id": job_id}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Stop an image job: a queued one never starts, a running one stops at its next denoising step."""
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    if j["status"] in ("queued", "running"):
        j["cancel"] = True
        log(f"cancel requested for image job {job_id} ({j['status']})")
    return {"job_id": job_id, "status": j["status"], "cancel": bool(j.get("cancel"))}


@app.get("/jobs/{job_id}")
def job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    out = {k: v for k, v in j.items() if k != "created"}
    if j["status"] == "queued":
        out["queue_position"] = QUEUE.qsize()
    return out


if __name__ == "__main__":
    import uvicorn
    log(f"Understood server starting: build {BUILD}, version {VERSION}, features {FEATURES}, mock={MOCK}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level=LOG_LEVEL)
