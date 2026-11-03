"""bank_more.py - build ONLY the listed topics into a separate folder (needs the newest bank_build.py and bank_raster.py beside it).
    !python bank_more.py --out /content/drive/MyDrive/understood_mm/bank_more
Finished topics are skipped; Ctrl+M I stops it safely; run it again to continue."""
import argparse, sys
from pathlib import Path
import bank_build as B

MORE = {
    "liver": (["liver", "human liver"], "human liver anatomy labelled diagram"),
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
# Built yesterday and already on your PC, so not repeated here: stomach, nephron, alveolus, nervous_system, spinal_cord.
# To redo one, paste its line from TOPICS in bank_build.py into MORE above.

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="", help="comma list of ids from MORE; empty = all of MORE")
ap.add_argument("--svg-keep", type=int, default=3)
ap.add_argument("--raster-keep", type=int, default=3)
ap.add_argument("--force", action="store_true")
ap.add_argument("--offline", action="store_true", help="use only files fetched on the PC (bank_fetch.py)")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
B.CACHE[0] = str(out / "_cache")
B.OFFLINE[0] = a.offline
want = [w.strip() for w in a.only.split(",") if w.strip()]
todo = {k: v for k, v in MORE.items() if not want or k in want}
have = sorted(f.stem for f in out.glob("*.json") if f.name != "index.json" and "__alt" not in f.stem)
print(f"Folder: {out.resolve()}\n  already built here ({len(have)}): {', '.join(have) or 'none'}\n  to do: {', '.join(k for k in todo if k not in have or a.force)}")
for tid, (aliases, q) in todo.items():
    if (out / f"{tid}.json").exists() and not a.force:
        print("already built, skipping:", tid); continue
    print("building", tid)
    try:
        rec = B.build_topic(tid, aliases, q, out, svg_keep=a.svg_keep, raster_keep=a.raster_keep)
    except B.RateLimited as e:
        B.write_index(out)
        print(f"\nSTOPPED: {e}. Everything so far is saved. Run the same command again in a few minutes."); sys.exit(0)
    except Exception as e:
        print(f"  -> FAILED: {type(e).__name__}: {str(e)[:100]} - next topic"); continue
    print("  ->", f"{len(rec['parts'])} labels, {sum(1 for p in rec['parts'] if p['anchor'])} anchored ({rec.get('kind')})" if rec else "NOTHING USABLE")
    B.write_index(out)
B.write_index(out); (out / "approved.txt").touch()
print("\nDone. Download this folder and merge it into your PC bank (see the guide).")
