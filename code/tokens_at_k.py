#!/usr/bin/env python3
"""Count the tokens gpt-5 would read per LoCoMo question under Mem0's current answer prompt when we hand it the top-k turns, for several k, with no model call.

    .venv/bin/python tokens_at_k.py snap_rank300-nograph_ens.json

Prints mean prompt tokens (o200k_base) by split and k, beside Mem0's 6,956.
"""
import json, re, sys, statistics as st
from datetime import datetime as _dt
import tiktoken
import mem0v2_prompts as M2

rank = json.load(open(sys.argv[1]))
CUTS = (40, 60, 100, 150, 200, 250, 300)
DEV = {"conv-26", "conv-30", "conv-41"}
enc = tiktoken.get_encoding("o200k_base")

def human_date(s):
    m = re.search(r"(\d{1,2}) (\w+), (\d{4})", s or "")
    if not m: return s or "unknown date"
    try: return _dt.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y").strftime("%A, %B %d, %Y")
    except ValueError: return s

convs = {}
for c in json.load(open("data/locomo10.json")):
    cv = c["conversation"]; turns = []; n = 1
    while f"session_{n}" in cv:
        date = cv.get(f"session_{n}_date_time", "")
        for t in cv[f"session_{n}"]:
            turns.append(dict(id=t["dia_id"], speaker=t["speaker"], date=date,
                              text=t["text"] + (f" [shared image: {t['blip_caption']}]" if t.get("blip_caption") else "")))
        n += 1
    convs[c["sample_id"]] = dict(turns=turns, qa=c["qa"])

rows = []
for sid, qs in rank.items():
    c = convs[sid]; pos = {t["id"]: i for i, t in enumerate(c["turns"])}
    ref = human_date(c["turns"][-1]["date"])
    qa = {str(i): q for i, q in enumerate(c["qa"])}
    for qi, ids in qs.items():
        q = qa.get(qi)
        if q is None: continue
        rec = dict(split="dev" if sid in DEV else "test", nturns=len(c["turns"]))
        for k in CUTS:
            idx = sorted(pos[d] for d in ids[:k] if d in pos)
            lines = ["The following memories are presented in chronological order (oldest to newest).", ""]
            lines += [f"({human_date(c['turns'][i]['date'])}) {c['turns'][i]['speaker']}: {c['turns'][i]['text']}" for i in idx]
            prompt = M2.ANSWER_GENERATION_PROMPT.format(memories="\n".join(lines), question=q["question"], reference_date=ref)
            rec[k] = len(enc.encode(prompt))
            # grouped: one date line per run of turns from the same session, then "speaker: text" lines
            g = ["The following memories are presented in chronological order (oldest to newest), grouped by the date they were said.", ""]
            last = None
            for i in idx:
                d = c["turns"][i]["date"]
                if d != last:
                    g.append(f"[{human_date(d)}]"); last = d
                g.append(f"{c['turns'][i]['speaker']}: {c['turns'][i]['text']}")
            rec[("g", k)] = len(enc.encode(M2.ANSWER_GENERATION_PROMPT.format(memories="\n".join(g), question=q["question"], reference_date=ref)))
        rows.append(rec)
for split in ("dev", "test", "all"):
    rs = [r for r in rows if split == "all" or r["split"] == split]
    print(split, f"n={len(rs)}", " ".join(f"k{k}: {st.mean(r[k] for r in rs):,.0f}" for k in CUTS))
    print("  grouped dates", " ".join(f"k{k}: {st.mean(r[('g', k)] for r in rs):,.0f}" for k in CUTS))
print("conversation lengths (turns):", sorted({(r['split'], r['nturns']) for r in rows}))
print("Mem0 current reads 6,956 per question (its site, gpt-5 reader, top 200 memories).")
