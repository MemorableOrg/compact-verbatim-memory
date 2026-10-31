#!/usr/bin/env python3
"""Top-300 ranking per LoCoMo question for the v2 search: the shipped arm with speaker names kept out of the first-stage search text
(NONAME=1; NONAME=2 also keeps them out of the reranker pairs) and a deeper reranker (RRN candidates instead of the arm's 100).

    NONAME=1 RRN=300 ~/Documents/GitHub/snapgraph/.venv/bin/python snap_rank_v2.py [dev|test|both] [arm]

Output: snap_rank300-v2-<arm>.json {sample_id: {qi: [dia_id, ...top 300]}}, the file locomo_run.py reads with --retriever snap.
Evidence labels are read only to print recall at cuts. Settings are chosen on dev; test is ranked once with the frozen settings.
"""
import json, os, re, sys, time
SG = os.path.expanduser("~/Documents/GitHub/snapgraph")
sys.path.insert(0, os.path.join(SG, "src"))
os.chdir(SG)
import load, loc  # noqa: E402
from rankers import rank_desc  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "both"
arm = sys.argv[2] if len(sys.argv) > 2 else "nograph_ens"
NONAME = int(os.environ.get("NONAME", "1"))
RRN = int(os.environ.get("RRN", "300"))
N = 300
CUTS = (10, 20, 40, 60, 100, 150, 200, 300)
cfg = json.load(open("configs/final.json"))["locomo"]["arms"][arm]
cfg["rr"]["N"] = RRN
raw = json.load(open("data/locomo10.json"))
sample_of = {ci: c["sample_id"] for ci, c in enumerate(raw)}
out_path = os.path.expanduser(f"~/Documents/memorable-paper-kickoff/proposal-v2/conv/snap_rank300-v2-{arm}.json")
out = json.load(open(out_path)) if os.path.exists(out_path) else {}

def strip_names(conv, q):
    for n in conv.named(q):
        q = re.sub(r"\b" + re.escape(n) + r"(?:'s)?\b", "", q)
    return re.sub(r"\s+", " ", q).strip()

def rank_v2(conv, q):
    qs = strip_names(conv, q) if NONAME else q
    s = conv.fused(qs, cfg["fuse"])
    s = loc.speaker_prior(s, conv, q, cfg)
    s = loc.rerank_step(s, conv, qs if NONAME == 2 else q, cfg)
    forced = loc.cluster_fill(s, conv, cfg, q)
    if forced is not None:
        return forced
    s, forced = loc.prf_step(s, conv, q, cfg)
    if forced is not None:
        return forced
    return [int(i) for i in rank_desc(s)]

rows = []
print(f"arm {arm} NONAME={NONAME} RRN={RRN} splits {which}", flush=True)
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
            r = rank_v2(conv, q["question"])
            ids = [inv[conv.keys[i]] for i in r[:N]]
            out[sid][str(q["qi"])] = ids
            gold = loc.gold_of(conv, lab)
            if gold:
                gold_ids = {inv[g] for g in gold}
                m = {k: float(set(ids[:k]) >= gold_ids) for k in CUTS}
                m.update(split=split, cat=lab["category"])
                rows.append(m)
        json.dump(out, open(out_path, "w"))
        print(f"{split} {sid}: {len(out[sid])} questions, {len(conv.keys)} turns, {time.time() - t0:.0f}s", flush=True)
for split in ("dev", "test"):
    rs = [r for r in rows if r["split"] == split]
    if rs:
        print(split, f"n={len(rs)}", " ".join(f"all@{k} {100 * sum(r[k] for r in rs) / len(rs):.1f}" for k in CUTS), flush=True)
        for c in (1, 2, 3, 4):
            cs_ = [r for r in rs if r["cat"] == c]
            print(f"  cat{c} n={len(cs_)}", " ".join(f"@{k} {100 * sum(r[k] for r in cs_) / len(cs_):.1f}" for k in CUTS), flush=True)
print("SNAP_RANK_V2_DONE")
