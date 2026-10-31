#!/usr/bin/env python3
"""Code-only selection fixes for multi-hop recall on LoCoMo, scored offline on one split. No model calls.

    ~/Documents/GitHub/snapgraph/.venv/bin/python mh_fix.py [arm] [dev|test]

Variants (each re-orders the arm's own ranking; the top `pin` turns never move):
  base       the arm as shipped
  tailbridge tail re-ordered by fusing base with searches for the question plus each top-5 turn's content words
  mmr_list   list questions only: top 40 chosen greedily for relevance minus similarity to turns already chosen
  mmr_all    the same for every question
  spk2       questions naming two speakers: alternate the best turns of each speaker
  combo      spk2, then mmr_list, then tailbridge
Also prints where the worst-ranked gold turn sits for base misses at 40. Evidence labels are read only to score.
"""
import json, os, re, sys, collections
import numpy as np
SG = os.path.expanduser("~/Documents/GitHub/snapgraph")
sys.path.insert(0, os.path.join(SG, "src"))
os.chdir(SG)
import load, loc, query  # noqa: E402
from rankers import rank_desc, z  # noqa: E402

arm = sys.argv[1] if len(sys.argv) > 1 else "nograph_ens"
split = sys.argv[2] if len(sys.argv) > 2 else "dev"
PIN = int(os.environ.get("PIN", "5")); LAM = float(os.environ.get("LAM", "0.7")); DEPTH = int(os.environ.get("DEPTH", "100")); KK = int(os.environ.get("KK", "40"))
M = 5; W = float(os.environ.get("W", "0.5")); T = 8
cfg = json.load(open("configs/final.json"))["locomo"]["arms"][arm]
if os.environ.get("RRN"):
    cfg["rr"]["N"] = int(os.environ["RRN"])
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

NONAME = int(os.environ.get("NONAME", "0"))  # 1: speaker names dropped from the search text; 2: also from the reranker pairs

def strip_names(conv, q):
    for n in conv.named(q):
        q = re.sub(r"\b" + re.escape(n) + r"(?:'s)?\b", "", q)
    return re.sub(r"\s+", " ", q).strip()

def scores(conv, q):
    qs = strip_names(conv, q) if NONAME else q
    s = conv.fused(qs, cfg['fuse'])
    s = loc.speaker_prior(s, conv, q, cfg)
    return loc.rerank_step(s, conv, qs if NONAME == 2 else q, cfg)

def mmr(s, conv, order, pin=PIN, lam=LAM, depth=DEPTH, kk=KK):
    top = order[:depth]
    V = conv.turn_vecs()[top]
    rel = z(np.array([s[i] for i in top]))
    chosen = list(range(min(pin, len(top))))
    rest = [j for j in range(len(top)) if j >= pin]
    while len(chosen) < kk and rest:
        sims = V[rest] @ V[chosen].T
        val = lam * rel[rest] - (1 - lam) * sims.max(1)
        j = rest[int(np.argmax(val))]
        chosen.append(j); rest.remove(j)
    head = [top[j] for j in chosen]
    hs = set(head)
    return head + [i for i in order if i not in hs]

def tailbridge(conv, q, order, pin=PIN):
    qex = {t.lemma_.lower() for t in conv.qdoc(q)}
    exps = []
    for i in order[:M]:
        terms = content(conv.texts[i], qex)[:T]
        if terms:
            exps.append([int(j) for j in rank_desc(conv.fused(q + " " + " ".join(terms), cfg["fuse"]))[:100]])
    if not exps:
        return order
    head = order[:pin]; hs = set(head)
    fused = rrf([order[:300]] + exps, [1.0] + [W] * len(exps))
    return head + [i for i in fused if i not in hs] + [i for i in order[300:] if i not in hs]

def spk2(conv, q, order, kk=KK):
    names = conv.named(q)
    if len(names) != 2:
        return order
    per = {n: [i for i in order if conv.spk[i] == n] for n in names}
    other = [i for i in order if conv.spk[i] not in names]
    head, used = [], set()
    ptr = {n: 0 for n in names}
    while len(head) < kk and any(ptr[n] < len(per[n]) for n in names):
        for n in names:
            if ptr[n] < len(per[n]) and len(head) < kk:
                head.append(per[n][ptr[n]]); used.add(per[n][ptr[n]]); ptr[n] += 1
    return head + [i for i in order if i not in used]

rows, worst = [], []
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
        s = scores(conv, qt)
        base = [int(i) for i in rank_desc(s)]
        cue = loc.list_cue(conv, qt)
        var = {"base": base}
        var["tailbridge"] = tailbridge(conv, qt, base)
        var["mmr_list"] = mmr(s, conv, base) if cue else base
        var["mmr_all"] = mmr(s, conv, base)
        var["spk2"] = spk2(conv, qt, base)
        c = spk2(conv, qt, base)
        if cue:
            c = mmr(s, conv, c)
        var["combo"] = tailbridge(conv, qt, c)
        rec = dict(cat=lab["category"], ngold=len(gold), cue=cue, two=len(conv.named(qt)) == 2)
        for name, r in var.items():
            for k in (10, 20, 40):
                top = set(r[:k])
                rec[f"{name}@{k}"] = float(gold <= top)
        for k in (30, 60, 100, 150, 200):
            rec[f"base@{k}"] = float(gold <= set(base[:k]))
        rec.update(conv=cs["conv"], qi=q["qi"], q=qt, gold=sorted(gold), top=base[:300], sc=[float(s[i]) for i in base[:300]], sess=[int(conv.sess_i[i]) for i in base[:300]], spk=[conv.spk[i] for i in base[:300]])
        rows.append(rec)
        if not rec["base@40"]:
            pos = {i: p for p, i in enumerate(base)}
            worst.append((lab["category"], max(pos[g] for g in gold)))
    print(f"{split} conv {cs['conv']} done, {len(rows)} questions", flush=True)

names = ["base", "tailbridge", "mmr_list", "mmr_all", "spk2", "combo"]
print(f"arm {arm} split {split} NONAME={NONAME} RRN={cfg['rr'].get('N')} PIN={PIN} LAM={LAM} DEPTH={DEPTH} KK={KK} W={W}")
print(f"{'variant':11s} " + " ".join(f"{'cat'+str(c):>22s}" for c in (1, 2, 3, 4)) + f" {'all':>22s}")
print(f"{'':11s} " + " ".join(f"{'@10':>6s} {'@20':>7s} {'@40':>7s}" for _ in range(5)))
for n in names:
    cells = []
    for c in (1, 2, 3, 4, "all"):
        rs = [r for r in rows if c == "all" or r["cat"] == c]
        cells.append(" ".join(f"{100*np.mean([r[f'{n}@{k}'] for r in rs]):6.1f}" for k in (10, 20, 40)) + " ")
    print(f"{n:11s} " + " ".join(cells))
print("n per cat:", dict(collections.Counter(r["cat"] for r in rows)), "| list-cue questions:", sum(r["cue"] for r in rows), "| two-speaker questions:", sum(r["two"] for r in rows))
print("policy: list-cue questions read KC turns, the rest read KN (base ranking); all-gold recall and mean turns")
for KC, KN in ((40, 40), (100, 20), (150, 20), (100, 10), (150, 30), (200, 20), (60, 20)):
    hit = [r[f"base@{KC}"] if r["cue"] else r[f"base@{KN}"] for r in rows]
    meank = np.mean([KC if r["cue"] else KN for r in rows])
    bycat = " ".join(f"cat{c} {100*np.mean([h for h, r in zip(hit, rows) if r['cat']==c]):5.1f}" for c in (1, 2, 3, 4))
    print(f"  KC={KC:3d} KN={KN:3d}: all {100*np.mean(hit):5.1f}  {bycat}  mean turns {meank:5.1f}")
print("worst gold rank for base misses at 40, by category:")
for c in (1, 2, 3, 4):
    w = [p for cc, p in worst if cc == c]
    if w:
        b = collections.Counter("41-60" if p < 60 else "61-100" if p < 100 else "101-200" if p < 200 else "200+" for p in w)
        print(f"  cat{c}: n={len(w)} " + " ".join(f"{k} {b.get(k, 0)}" for k in ("41-60", "61-100", "101-200", "200+")))
dump = f"mhdump-{arm}-{split}-noname{NONAME}-rrn{cfg['rr'].get('N')}.json"
json.dump(rows, open(dump, "w"))
print("dumped", dump, len(rows))
print("MH_FIX_DONE")
