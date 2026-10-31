#!/usr/bin/env python3
"""Rank the LoCoMo test split once with the cascade schedule (MiniLM over 300, bge-reranker-v2-m3 over the top 100 of that,
speaker names out of the first stage) and print all-gold recall at the usual cuts. Registered in Addendum 39 before running.

    ~/Documents/GitHub/snapgraph/.venv/bin/python cascade_test.py

Labels are read only to score. No model tokens. Also writes snap_rank300-cascade-nograph_ens.json for a future reader run.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cascade_dev import SG, cfg, first_stages, cascade  # noqa: E402
os.chdir(SG)
import load, loc  # noqa: E402
from rankers import rank_desc  # noqa: E402

CUTS = (10, 20, 40, 60, 100, 150, 200, 300)
raw = json.load(open("data/locomo10.json"))
sample_of = {ci: c["sample_id"] for ci, c in enumerate(raw)}
out_path = os.path.expanduser("~/Documents/memorable-paper-kickoff/proposal-v2/conv/snap_rank300-cascade-nograph_ens.json")
out, rows = {}, []
sysd, scd = load.load_locomo("test")
for cs, scn in zip(sysd, scd):
    t0 = time.time()
    conv = loc.Conv(cs, True)
    inv = {v: k for k, v in conv.by.items()}
    sid = sample_of[cs["conv"]]
    out[sid] = {}
    for q, lab in zip(cs["questions"], scn["qa"]):
        if lab["category"] not in (1, 2, 3, 4):
            continue
        s = first_stages(conv, q["question"], True)
        r = [int(i) for i in rank_desc(cascade(s, conv, q["question"]))]
        ids = [inv[conv.keys[i]] for i in r[:300]]
        out[sid][str(q["qi"])] = ids
        gold = loc.gold_of(conv, lab)
        if gold:
            gold_ids = {inv[g] for g in gold}
            m = {k: float(set(ids[:k]) >= gold_ids) for k in CUTS}
            m["cat"] = lab["category"]
            rows.append(m)
    json.dump(out, open(out_path, "w"))
    print(f"test {sid}: {len(out[sid])} questions, {time.time() - t0:.0f}s", flush=True)
print("test", f"n={len(rows)}", " ".join(f"all@{k} {100 * sum(r[k] for r in rows) / len(rows):.1f}" for k in CUTS), flush=True)
for c in (1, 2, 3, 4):
    cs_ = [r for r in rows if r["cat"] == c]
    print(f"  cat{c} n={len(cs_)}", " ".join(f"@{k} {100 * sum(r[k] for r in cs_) / len(cs_):.1f}" for k in CUTS), flush=True)
print("CASCADE_TEST_DONE")
