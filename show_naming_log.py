"""Print naming_log.jsonl as readable tables.   python show_naming_log.py <path to naming_log.jsonl> [--last N]
One row per segmented region: what the tutor named it, and what was finally labelled."""
import json, sys, time
from pathlib import Path

p = Path(sys.argv[1]); last = int(sys.argv[sys.argv.index("--last") + 1]) if "--last" in sys.argv else 3
recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
for r in recs[-last:]:
    print("\n" + "=" * 90)
    print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r.get('ts', 0)))}  {r.get('title')}")
    print("parts:", ", ".join(r.get("parts", [])))
    print(f"{'id':>3} {'area':>7} {'anchor':>11}  {'tutor said':<26} LABELLED AS")
    for g in r.get("regions", []):
        nm = g.get("name") if "name" in g else (g.get("free") or {}).get("name")
        print(f"{g['id']:>3} {g['area']:>7} {str(g['anchor']):>11}  {str(nm):<26} {g.get('final')}")
    tot = len(r.get("parts", []))
    print(f"labelled {tot - len(r.get('not_located', []))}/{tot};  NOT LOCATED: {r.get('not_located')}")
