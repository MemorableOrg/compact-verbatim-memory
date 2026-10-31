#!/usr/bin/env python3
"""Wall time per LoCoMo question for the v2 search (names out of the first stage, reranker over RRN candidates), on dev conversations.

    ~/Documents/GitHub/snapgraph/.venv/bin/python time_search_v2.py [n_questions]

Prints median and p95 seconds per question at RRN=100 and RRN=300, no model tokens. Labels are not read.
"""
import json, os, re, statistics, sys, time
SG = os.path.expanduser("~/Documents/GitHub/snapgraph")
sys.path.insert(0, os.path.join(SG, "src"))
os.chdir(SG)
import load, loc  # noqa: E402
from rankers import rank_desc  # noqa: E402

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 40
cfg0 = json.load(open("configs/final.json"))["locomo"]["arms"]["nograph_ens"]


def strip_names(conv, q):
    for n in conv.named(q):
        q = re.sub(r"\b" + re.escape(n) + r"(?:'s)?\b", "", q)
    return re.sub(r"\s+", " ", q).strip()


def rank_v2(conv, q, cfg):
    qs = strip_names(conv, q)
    s = conv.fused(qs, cfg["fuse"])
    s = loc.speaker_prior(s, conv, q, cfg)
    s = loc.rerank_step(s, conv, q, cfg)
    forced = loc.cluster_fill(s, conv, cfg, q)
    if forced is not None:
        return forced
    s, forced = loc.prf_step(s, conv, q, cfg)
    if forced is not None:
        return forced
    return [int(i) for i in rank_desc(s)]


sysd, scd = load.load_locomo("dev")
qs = []
for cs, scn in zip(sysd, scd):
    conv = loc.Conv(cs, True)
    for q, lab in zip(cs["questions"], scn["qa"]):
        if lab["category"] in (1, 2, 3, 4):
            qs.append((conv, q["question"]))
qs = qs[:: max(1, len(qs) // NQ)][:NQ]
print(f"{len(qs)} dev questions over {len(sysd)} conversations", flush=True)
rank_v2(qs[0][0], qs[0][1], dict(cfg0, rr=dict(cfg0["rr"], N=100)))  # warm-up loads the models
for N in (100, 300):
    cfg = dict(cfg0, rr=dict(cfg0["rr"], N=N))
    ts = []
    for conv, q in qs:
        t0 = time.perf_counter()
        rank_v2(conv, q, cfg)
        ts.append(time.perf_counter() - t0)
    ts_sorted = sorted(ts)
    p95 = ts_sorted[min(len(ts) - 1, int(round(0.95 * (len(ts) - 1))))]
    print(f"RRN={N}: n={len(ts)} median {statistics.median(ts):.2f}s mean {statistics.mean(ts):.2f}s p95 {p95:.2f}s", flush=True)
print("TIME_SEARCH_V2_DONE")
