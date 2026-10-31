#!/usr/bin/env python3
"""Rank of the gold turn after each stage of the turns store, for development LoCoMo questions with one gold turn.

    ~/Documents/GitHub/snapgraph/.venv/bin/python worked_example.py [max_examples]

Stages: BM25 alone, dense alone, fused, + contiguity, + speaker prior, + cross-encoder rerank (names out of the first stage,
300 candidates). Prints questions where the first stage ranks the gold turn past 10 and the final stage brings it into the top 10.
Labels are read only to name the gold turn. No model tokens.
"""
import json, os, re, sys
import numpy as np
SG = os.path.expanduser("~/Documents/GitHub/snapgraph")
sys.path.insert(0, os.path.join(SG, "src"))
os.chdir(SG)
import load, loc  # noqa: E402
from run_turn import z, contig_sess  # noqa: E402
from rankers import rank_desc  # noqa: E402

MAXN = int(sys.argv[1]) if len(sys.argv) > 1 else 4
cfg = json.load(open("configs/final.json"))["locomo"]["arms"]["nograph_ens"]
cfg["rr"]["N"] = 300
fz = cfg["fuse"]


def strip_names(conv, q):
    for n in conv.named(q):
        q = re.sub(r"\b" + re.escape(n) + r"(?:'s)?\b", "", q)
    return re.sub(r"\s+", " ", q).strip()


def rank_of(s, g):
    return int(np.where(rank_desc(s) == g)[0][0]) + 1


sysd, scd = load.load_locomo("dev")
shown = 0
for cs, scn in zip(sysd, scd):
    conv = loc.Conv(cs, True)
    sc = conv.sc
    for q, lab in zip(cs["questions"], scn["qa"]):
        if lab["category"] not in (1, 2, 3, 4):
            continue
        gold = loc.gold_of(conv, lab)
        if len(gold) != 1:
            continue
        g = conv.keys.index(next(iter(gold)))
        qt = q["question"]
        qs = strip_names(conv, qt)
        sc.q = {"question": qs, "question_date": conv.date, "sessions": conv.sessions}
        bm = sc.bm25("turn", fz.get("bm_t", ""), qs)
        de = sc.dense_turn(qs)
        fused = fz["w"]["bm25_t"] * z(bm) + fz["w"]["dense_t"] * z(de)
        bf, bb = fz["contig"]
        cont = contig_sess(sc, fused, bf, bb)
        prior = loc.speaker_prior(cont.copy(), conv, qt, cfg)
        rr = loc.rerank_step(prior.copy(), conv, qt, cfg)
        ranks = [rank_of(x, g) for x in (bm, de, fused, cont, prior, rr)]
        if ranks[2] > 10 and ranks[-1] <= 10:
            print(f"\nQ ({['', 'multi-hop', 'temporal', 'open-domain', 'single-hop'][lab['category']]}): {qt}")
            print(f"gold turn key {conv.keys[g]}; conversation has {len(conv.keys)} turns")
            print("ranks  bm25 %d | dense %d | fused %d | +contiguity %d | +speaker prior %d | +rerank %d" % tuple(ranks))
            print("answer:", lab.get("answer"))
            shown += 1
            if shown >= MAXN:
                sys.exit(0)
