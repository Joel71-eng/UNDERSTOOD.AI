r"""
bank_review.py - review the bank in the browser instead of editing files by hand.
    streamlit run bank_review.py -- --bank C:\UND\bank          (or the Drive-mirrored folder)
For each built picture you see the numbered anchors, the automatic flags, and an editable list of part names.
Approve  -> saves your corrections and adds the id to approved.txt (the app serves only approved ids)
Reject   -> adds it to rejected.txt (the builder will not rebuild it unless you pass --force)
"Approve all that passed the checks" approves every unflagged picture in one click - glance at the sheet first.
"""
import argparse
import json
import sys
from pathlib import Path

import streamlit as st

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default="bank")
args, _ = ap.parse_known_args(sys.argv[1:])
BANK = Path(args.bank)
st.set_page_config(page_title="Bank review", layout="wide")


def ids(name: str) -> list[str]:
    f = BANK / name
    return [x.strip() for x in f.read_text(encoding="utf-8").split() if x.strip()] if f.exists() else []


def add(name: str, bid: str):
    cur = ids(name)
    if bid not in cur:
        (BANK / name).write_text("\n".join(cur + [bid]) + "\n", encoding="utf-8")


def remove(name: str, bid: str):
    (BANK / name).write_text("\n".join(x for x in ids(name) if x != bid) + "\n", encoding="utf-8")


def load(bid: str) -> dict:
    return json.loads((BANK / f"{bid}.json").read_text(encoding="utf-8"))


all_ids = sorted(p.stem for p in BANK.glob("*.json") if p.stem != "index" and (BANK / f"{p.stem}.png").exists())
approved, rejected = set(ids("approved.txt")), set(ids("rejected.txt"))
todo = [b for b in all_ids if b not in approved and b not in rejected]
st.title("Diagram bank review")
st.caption(f"{BANK} - {len(approved)} approved, {len(rejected)} rejected, {len(todo)} to review")
if not all_ids:
    st.info("No built pictures here yet. Run bank_build.py first.")
    st.stop()

clean = [b for b in todo if not load(b).get("flags")]
if clean and st.button(f"Approve all {len(clean)} that passed the checks", type="primary"):
    for b in clean:
        add("approved.txt", b)
    st.rerun()

show = st.radio("Show", ["To review", "Approved", "Rejected"], horizontal=True)
pool = {"To review": todo, "Approved": sorted(approved & set(all_ids)), "Rejected": sorted(rejected & set(all_ids))}[show]
if not pool:
    st.success("Nothing here.")
    st.stop()
bid = st.selectbox("Picture", pool, format_func=lambda b: f"{b}" + ("  ⚠" if load(b).get("flags") else "  ✓"))
rec = load(bid)
left, right = st.columns([3, 2])
rv = BANK / "review" / f"{bid}_review.png"
left.image(str(rv if rv.exists() else BANK / f"{bid}.png"), width="stretch")
left.caption(rec.get("credit", ""))
with right:
    for f in rec.get("flags", []):
        st.warning(f)
    if not rec.get("flags"):
        st.success("Passed the automatic checks")
    st.markdown("**Part names** (numbers match the red circles; tick *drop* for labels that are not parts, like 'Figure 1')")
    edited = []
    for i, p in enumerate(rec["parts"], 1):
        c1, c2 = st.columns([4, 1])
        name = c1.text_input(f"{i}", p["name"], key=f"{bid}_n{i}", label_visibility="collapsed", placeholder=f"{i}")
        drop = c2.checkbox("drop", key=f"{bid}_d{i}", value=not p.get("anchor"))
        if not drop:
            edited.append({**p, "name": name.strip() or p["name"]})
    aliases = st.text_input("Topic names that should find this picture (comma separated)", ", ".join(rec.get("aliases", [])), key=f"{bid}_al")
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Approve", type="primary", key=f"{bid}_ok", width="stretch"):
        rec["parts"] = edited
        rec["aliases"] = [a.strip() for a in aliases.split(",") if a.strip()]
        rec["flags"] = []
        (BANK / f"{bid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        add("approved.txt", bid)
        remove("rejected.txt", bid)
        st.rerun()
    if b2.button("🚫 Reject", key=f"{bid}_no", width="stretch"):
        add("rejected.txt", bid)
        remove("approved.txt", bid)
        st.rerun()
    if show != "To review" and b3.button("↩ Back to review", key=f"{bid}_back", width="stretch"):
        remove("approved.txt", bid)
        remove("rejected.txt", bid)
        st.rerun()
