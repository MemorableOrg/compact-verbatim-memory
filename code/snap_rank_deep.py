#!/usr/bin/env python3
"""Like snap_rank.py but keeps the top 300 turns per question and prints recall at deeper cuts.

    ~/Documents/GitHub/snapgraph/.venv/bin/python snap_rank_deep.py [dev|test|both] [arm]

Output: snap_rank300[-arm].json {sample_id: {qi: [dia_id, ...top 300]}}. Evidence labels are read only to score.
"""
import json, os, sys, time
SG = os.path.expanduser("~/Documents/GitHub/snapgraph")
sys.path.insert(0, os.path.join(SG, "src"))
os.chdir(SG)
import load, loc  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "both"
arm = sys.argv[2] if len(sys.argv) > 2 else "nograph_ens"
N = 300
CUTS = (10, 20, 40, 60, 100, 150, 200, 250, 300)
cfg = json.load(open("configs/final.json"))["locomo"]["arms"][arm]
raw = json.load(open("data/locomo10.json"))
sample_of = {ci: c["sample_id"] for ci, c in enumerate(raw)}
out_path = os.path.expanduser("~/Documents/memorable-paper-kickoff/proposal-v2/conv/snap_rank300%s.json" % ("" if arm == "snapgraph" else "-" + arm))
out = json.load(open(out_path)) if os.path.exists(out_path) else {}
rows = []
for split in (["dev", "test"] if which == "both" else [which]):
    sysd, scd = load.load_locomo(split)
    for cs, scn in zip(sysd, scd):
        t0 = time.time()
        conv = loc.Conv(cs, True)
        inv = {v: k for k, v in conv.by.items()}
        sid = sample_of[cs["conv"]]
        out.setdefault(sid, {})
        for q, lab in zip(cs["questions"], scn["qa"]):
            if lab["category"] not in (1, 2, 3, 4):
                continue
            r = loc.rank(conv, q["question"], cfg)
            ids = [inv[conv.keys[i]] for i in r[:N]]
            out[sid][str(q["qi"])] = ids
            gold = loc.gold_of(conv, lab)
            if gold:
                gold_ids = {inv[g] for g in gold}
                m = {k: float(set(ids[:k]) >= gold_ids) for k in CUTS}
                m.update(split=split, cat=lab["category"], conv=sid, nturns=len(conv.keys))
                rows.append(m)
        json.dump(out, open(out_path, "w"))
        print(f"{split} {sid}: {len(out[sid])} questions, {len(conv.keys)} turns, {time.time() - t0:.0f}s", flush=True)
for split in ("dev", "test"):
    rs = [r for r in rows if r["split"] == split]
    if rs:
        print(split, "n=%d" % len(rs), " ".join("all@%d %.1f" % (k, 100 * sum(r[k] for r in rs) / len(rs)) for k in CUTS), flush=True)
        for c in (1, 2, 3, 4):
            cs_ = [r for r in rs if r["cat"] == c]
            print(f"  cat{c} n={len(cs_)}", " ".join("@%d %.1f" % (k, 100 * sum(r[k] for r in cs_) / len(cs_)) for k in CUTS), flush=True)
print("SNAP_RANK_DEEP_DONE")
