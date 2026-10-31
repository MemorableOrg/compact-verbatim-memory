#!/usr/bin/env python3
"""Rank every LoCoMo question's turns with Snapgraph's frozen arm and dump the ranking, plus recall at several cuts.

    ~/Documents/GitHub/snapgraph/.venv/bin/python snap_rank.py [dev|test|both]

Runs from the Snapgraph tree (src unchanged) with configs/final.json's locomo "snapgraph" arm. Output: snap_rank.json
{sample_id: {qi: [dia_id, ...top 60]}} keyed the way locomo_run.py keys conversations, and a recall table on stdout.
Reads evidence labels only to score, never to rank.
"""
import json, os, sys, time
SG = os.path.expanduser("~/Documents/GitHub/snapgraph")
sys.path.insert(0, os.path.join(SG, "src"))
os.chdir(SG)
import load, loc, score  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "both"
arm = sys.argv[2] if len(sys.argv) > 2 else "snapgraph"
cfg = json.load(open("configs/final.json"))["locomo"]["arms"][arm]
raw = json.load(open("data/locomo10.json"))
sample_of = {ci: c["sample_id"] for ci, c in enumerate(raw)}
out_path = os.path.expanduser("~/Documents/memorable-paper-kickoff/proposal-v2/conv/snap_rank%s.json" % ("" if arm == "snapgraph" else "-" + arm))
out = json.load(open(out_path)) if os.path.exists(out_path) else {}
rows = []
for split in (["dev", "test"] if which == "both" else [which]):
    sysd, scd = load.load_locomo(split)
    for cs, scn in zip(sysd, scd):
        t0 = time.time()
        conv = loc.Conv(cs, True)
        inv = {v: k for k, v in conv.by.items()}  # (sid, ti) -> dia_id
        sid = sample_of[cs["conv"]]
        out.setdefault(sid, {})
        for q, lab in zip(cs["questions"], scn["qa"]):
            if lab["category"] not in (1, 2, 3, 4):
                continue
            r = loc.rank(conv, q["question"], cfg)
            ids = [inv[conv.keys[i]] for i in r[:60]]
            out[sid][str(q["qi"])] = ids
            gold = loc.gold_of(conv, lab)
            if gold:
                gold_ids = {inv[g] for g in gold}
                m = {k: float(set(ids[:k]) >= gold_ids) for k in (5, 10, 20, 40)}
                m.update(split=split, cat=lab["category"], conv=sid)
                rows.append(m)
        json.dump(out, open(out_path, "w"))
        print(f"{split} {sid}: {len(out[sid])} questions in {time.time() - t0:.0f}s", flush=True)
for split in ("dev", "test"):
    rs = [r for r in rows if r["split"] == split]
    if rs:
        print(split, "n=%d" % len(rs), " ".join("all@%d %.1f" % (k, 100 * sum(r[k] for r in rs) / len(rs)) for k in (5, 10, 20, 40)), flush=True)
print("SNAP_RANK_DONE")
