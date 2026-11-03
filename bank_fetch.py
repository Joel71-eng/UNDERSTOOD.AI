"""
bank_fetch.py - run this on your PC (home internet is not throttled the way Colab is).  It only DOWNLOADS: the Commons search answers and the candidate
files for each topic, into  <out>/_cache .  Then upload that folder to Drive and build on Colab with --offline (no network, no 429s).

    pip install requests lxml
    set UNDERSTOOD_CONTACT=your-email@example.com        (optional: Wikimedia gives identified programs more room)
    python bank_fetch.py --out C:\\UND\\fetch                      (all topics)
    python bank_fetch.py --out C:\\UND\\fetch --topics "liver,tooth,circulatory system"
    python bank_fetch.py --out C:\\UND\\fetch --more              (only the topics listed in bank_more.py)

Safe to stop (Ctrl+C) and run again: what is downloaded is never downloaded twice.
"""
import argparse
import sys
from pathlib import Path

import bank_build as B

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--topics", default="")
ap.add_argument("--more", action="store_true", help="only the topics of bank_more.py")
ap.add_argument("--svg-tries", type=int, default=24)
ap.add_argument("--raster-tries", type=int, default=12)
a = ap.parse_args()
out = Path(a.out)
B.CACHE[0] = str(out / "_cache")
topics = dict(B.TOPICS)
if a.more:
    import re
    src = Path(__file__).with_name("bank_more.py").read_text(encoding="utf-8")
    ids = re.findall(r'^    "([a-z_]+)": \(\[', src, re.M)
    topics = {k: v for k, v in B.TOPICS.items() if k in ids}
want = [w.strip().lower().replace(" ", "_") for w in a.topics.split(",") if w.strip()]
if want:
    topics = {k: v for k, v in topics.items() if k in want or any(w.replace("_", " ") in [x.lower() for x in v[0]] for w in want)}
print(f"Fetching {len(topics)} topics into {out.resolve() / '_cache'}")
done = 0
for tid, (aliases, q) in topics.items():
    print("topic", tid)
    try:
        cands = B.gather_candidates(tid, aliases, q)
        svgs = [c for c in cands if c["kind"] == "svg"][:a.svg_tries]
        ras = [c for c in cands if c["kind"] == "raster"][:a.raster_tries]
        for i, c in enumerate(svgs + ras, 1):
            ok = B.download(c["url"])
            print(f"   {i}/{len(svgs) + len(ras)} {'ok ' if ok else 'FAILED'} {c['title'][:60]}")
        done += 1
    except B.RateLimited as e:
        print(f"\nSTOPPED: {e}. Run the same command again later; what is downloaded is kept.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nstopped by you; run again to continue")
        sys.exit(0)
    except Exception as e:                                                      # noqa: BLE001
        print(f"   topic failed: {type(e).__name__}: {str(e)[:100]}")
print(f"\nDone ({done} topics). Upload the folder  {out / '_cache'}  to Drive (e.g. understood_mm/bank/_cache) and build on Colab with --offline.")
