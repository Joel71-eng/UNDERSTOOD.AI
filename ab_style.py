"""A/B test of the diagram look: classic colours vs lighter colours.  Run on the PC with the Colab server up:
    python ab_style.py  https://your-ngrok-domain  [--topics "human heart,digestive system,..."]
For each topic and style it draws the picture, counts the regions the finder separates and how many of the parts that means, and
writes ab_style/contact_sheet.png (left classic, right light) plus ab_style/report.txt.  Pick the lighter look only if its region
count stays about as high as the classic one."""
import argparse, io, sys, time
from pathlib import Path
from PIL import Image, ImageDraw
import mm_core as C
import mm_sources as S
from mm_engine import ServerClient, b64_img

TOPICS = ["human heart", "digestive system", "human brain", "plant cell", "animal cell", "kidney", "human eye", "flower", "neuron", "respiratory system"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("--topics", default=""); ap.add_argument("--level", default="ordinary")
    a = ap.parse_args()
    topics = [t.strip() for t in a.topics.split(",") if t.strip()] or TOPICS
    cl = ServerClient(a.url); lv = C.get_level(a.level)
    out = Path("ab_style"); out.mkdir(exist_ok=True)
    rows, report = [], []
    for t in topics:
        src = S.fetch_topic(t, a.level, None, True)
        if not src or not src.get("curated"):
            report.append(f"{t}: no curated parts, skipped"); continue
        parts = src["parts"]; pair = []
        for style in ("classic", "light"):
            prompt = C.build_diagram_prompt(t, parts, a.level, style)
            jid = cl.submit_image(mode="generate", prompt=prompt, width=lv["gen_size"], height=lv["gen_size"], quality="fast", seed=7)
            while True:
                j = cl.job(jid)
                if j["status"] in ("done", "error"): break
                time.sleep(2)
            if j["status"] != "done":
                report.append(f"{t} [{style}]: generation failed"); pair.append(None); continue
            im = b64_img(j["image_b64"]).convert("RGB")
            regions, _ = C.find_regions(im)
            report.append(f"{t:22} {style:8} parts asked={len(parts)} regions found={len(regions)}")
            pair.append(im)
        rows.append((t, pair))
    if rows:
        s = 360
        sheet = Image.new("RGB", (2 * s, len(rows) * (s + 24)), "white"); d = ImageDraw.Draw(sheet)
        for r, (t, pair) in enumerate(rows):
            d.text((6, r * (s + 24) + 4), t, fill="black")
            for k, im in enumerate(pair):
                if im: sheet.paste(im.resize((s, s)), (k * s, r * (s + 24) + 24))
        sheet.save(out / "contact_sheet.png")
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8"); print("\n".join(report)); print("\nsaved ab_style/contact_sheet.png")

if __name__ == "__main__":
    main()
