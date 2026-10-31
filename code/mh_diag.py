#!/usr/bin/env python3
"""Where multi-hop retrieval fails on LoCoMo and whether a code-only second search fixes it. Dev split by default; no model calls.

    ~/Documents/GitHub/snapgraph/.venv/bin/python mh_diag.py [arm] [dev|test]
    env: M=5 (seed turns), W=0.5 (weight of each expansion in the fusion), T=8 (terms per expansion)

Three rankings per question: base = the arm as shipped; bridge = base fused with searches for the question plus the content words of each
top-M turn; nbr = base fused with searches for each top-M turn's own text. Evidence labels are read only to score.
"""
import json, os, re, sys, collections
import numpy as np
SG = os.path.expanduser("~/Documents/GitHub/snapgraph")
sys.path.insert(0, os.path.join(SG, "src"))
os.chdir(SG)
import load, loc, query  # noqa: E402
from rankers import rank_desc  # noqa: E402

arm = sys.argv[1] if len(sys.argv) > 1 else "nograph_ens"
split = sys.argv[2] if len(sys.argv) > 2 else "dev"
M = int(os.environ.get("M", "5")); W = float(os.environ.get("W", "0.5")); T = int(os.environ.get("T", "8"))
cfg = json.load(open("configs/final.json"))["locomo"]["arms"][arm]
nlp = query.nlp()
SPK = re.compile(r'^[^:\n]{1,30}: ')

def content(text, exclude):
    out = []
    for t in nlp(SPK.sub('', text)):
        w = t.lemma_.lower()
        if t.pos_ in ('NOUN', 'PROPN', 'NUM', 'ADJ', 'VERB') and not t.is_stop and w not in exclude and len(w) > 2 and w not in out:
            out.append(w)
    return out

def rrf(rankings, weights, c=60):
    sc = collections.defaultdict(float)
    for r, w in zip(rankings, weights):
        for pos, i in enumerate(r):
            sc[i] += w / (c + pos)
    return sorted(sc, key=lambda i: -sc[i])

rows, examples = [], []
sysd, scd = load.load_locomo(split)
for cs, scn in zip(sysd, scd):
    conv = loc.Conv(cs, True)
    key_i = {k: i for i, k in enumerate(conv.keys)}
    for q, lab in zip(cs["questions"], scn["qa"]):
        if lab["category"] == 5:
            continue
        gold = {key_i[k] for k in loc.gold_of(conv, lab) if k in key_i}
        if not gold:
            continue
        qt = q["question"]
        base = loc.rank(conv, qt, cfg)
        qex = {t.lemma_.lower() for t in conv.qdoc(qt)}
        exps, exps2 = [], []
        for i in base[:M]:
            terms = content(conv.texts[i], qex)[:T]
            if terms:
                exps.append([int(j) for j in rank_desc(conv.fused(qt + " " + " ".join(terms), cfg["fuse"]))[:100]])
            exps2.append([int(j) for j in rank_desc(conv.fused(SPK.sub('', conv.texts[i]), cfg["fuse"]))[:100]])
        bridge = rrf([base[:300]] + exps, [1.0] + [W] * len(exps))
        nbr = rrf([base[:300]] + exps2, [1.0] + [W] * len(exps2))
        rec = dict(cat=lab["category"], ngold=len(gold), conv=cs["conv"])
        for name, r in (("base", base), ("bridge", bridge), ("nbr", nbr)):
            for k in (10, 20, 40):
                top = set(r[:k])
                rec[f"{name}@{k}"] = float(gold <= top); rec[f"{name}_share@{k}"] = len(gold & top) / len(gold)
        rows.append(rec)
        if lab["category"] == 1 and not rec["base@40"] and len(examples) < 12:
            pos = {i: p for p, i in enumerate(base)}
            bpos = {i: p for p, i in enumerate(bridge)}
            examples.append((qt, [(pos.get(g, 999), bpos.get(g, 999), conv.sess_i[g], conv.texts[g][:120]) for g in sorted(gold, key=lambda g: pos.get(g, 999))]))
    print(f"{split} conv {cs['conv']} done, {len(rows)} questions so far", flush=True)

def summ(name):
    for c in (1, 2, 3, 4, "all"):
        rs = [r for r in rows if c == "all" or r["cat"] == c]
        print(f"  {name:7s} cat{c}: n={len(rs):4d} " + " ".join(f"all@{k} {100*np.mean([r[f'{name}@{k}'] for r in rs]):5.1f}" for k in (10, 20, 40))
              + f"  share@10 {100*np.mean([r[f'{name}_share@10'] for r in rs]):5.1f}", flush=True)
print(f"arm {arm} split {split} M={M} W={W} T={T}")
for n in ("base", "bridge", "nbr"):
    summ(n)
print("gold turns per multi-hop question:", dict(sorted(collections.Counter(r["ngold"] for r in rows if r["cat"] == 1).items())))
print("\n--- multi-hop misses at 40 under base (base rank, bridge rank, session | text):")
for qt, golds in examples:
    print("Q:", qt)
    for rk, bk, si, tx in golds:
        print(f"   base {rk:>3}  bridge {bk:>3}  sess {si:>2} | {tx}")
print("MH_DIAG_DONE")
