#!/usr/bin/env python3
"""Turn Mem0's published LoCoMo per-question file into our results row format, so rejudge.py can score its answers with another judge.

    python3 mem0_to_rows.py   -> results/mem0-current-locomo-rows.jsonl (1,540 rows; no model call)
    then: .venv/bin/python rejudge.py results/mem0-current-locomo-rows.jsonl gpt-4o-mini 5 mem0v1   (Addendum 35, M2)
"""
import json, os
raw = json.load(open("data/locomo10.json"))
m = json.load(open(os.path.expanduser("~/Documents/(private data folder, not distributed)/beam-repos/mem0mb/results/platform/locomo_results.json")))
by_q = {}
for ci, c in enumerate(raw):
    for j, q in enumerate(c["qa"]):
        by_q[(ci, q["question"].strip())] = (c["sample_id"], j, q)
out = open("results/mem0-current-locomo-rows.jsonl", "w")
n = miss = 0
for e in m["evaluations"]:
    key = (e["conversation_idx"], e["question"].strip())
    if key not in by_q:
        miss += 1; continue
    sid, j, q = by_q[key]
    cr = e["cutoff_results"]["top_200"]
    out.write(json.dumps(dict(sample=sid, qi=j, category=e["category"], question=q["question"], gold=q.get("answer"), answer=str(cr.get("generated_answer", "")),
                              correct=cr["judgment"] == "CORRECT", mem0_judgment=cr["judgment"], source="mem0 memory-benchmarks locomo_results.json")) + "\n")
    n += 1
print(f"wrote {n} rows, {miss} unmatched, to results/mem0-current-locomo-rows.jsonl")
