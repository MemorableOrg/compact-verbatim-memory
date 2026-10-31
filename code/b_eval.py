"""Task B: dev-only (conv-26/30/41), categories 1 and 2. 5r vs 5r+PRF vs 5r+date windows. Hit@1 and R@5 all against evidence ids. No API calls.
    .venv/bin/python b_eval.py dev | test   (test only if a variant beats 5r on dev cat1 R@5 all by >= 3 points)
"""
import json, re, sys, pickle
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
import retrieval_eval as R
from retrieval_eval import top, rrf, tok, stem, STOP, QWORDS, MONTHS

phase = sys.argv[1]
convs = R.load(); names = list(convs); DEV = names[:3]
use = DEV if phase == "dev" else [s for s in names if s not in DEV]
z = np.load("emb_cache_bge-small-en-v1.5.npz"); emb = {k: z[k] for k in z.files if k != "_time"}
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); m.max_seq_length = 256
cp = pickle.load(open("rr_cache.pkl", "rb"))
qs = [(sid, q, [e for x in q["evidence"] for e in re.findall(r"D\d+:\d+", x)]) for sid in use for q in convs[sid]["qa"] if q["category"] in (1, 2) and q.get("evidence")]
qs = [x for x in qs if x[2]]
qv = {}
for sid in use:
    sq = [q["question"] for s, q, _ in qs if s == sid]
    qv[sid] = dict(zip(sq, m.encode(sq, batch_size=32, normalize_embeddings=True)))

def five_r(c, sid, qt):
    n = len(c["turns"]); d = emb[sid] @ qv[sid][qt]
    five = rrf([top(R.bm25_scores(c, qt, True), 200), top(d, 200)], n)
    cand = [int(i) for i in top(five, 100)]
    ordce = sorted(cand, key=lambda j: -cp[(sid, qt, j)])
    return rrf([ordce, cand], n)

def prf(c, sid, qt, base):
    n = len(c["turns"]); hits = top(base, 5)
    txt = " ".join(c["turns"][j]["text"] for j in hits)
    caps = {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", txt)}; nums = set(re.findall(r"\b\d+\b", txt))
    df = Counter(w for dd in c["bm25s"].doc_freqs for w in dd)
    words = {stem(w) for w in tok(txt) if w not in STOP and w not in QWORDS and len(w) > 2}
    rare = sorted((w for w in words if df[w] > 0), key=lambda w: df[w])[:10]
    exp = [stem(w) for w in caps] + list(nums) + rare
    qt_ = [stem(w) for w in tok(qt) if w not in QWORDS and w not in STOP] or [stem(w) for w in tok(qt)]
    s2 = np.array(c["bm25s"].get_scores(qt_)) + 0.5 * (np.array(c["bm25s"].get_scores(exp)) if exp else 0)
    s2 = np.array(R.date_boost(c, qt, list(s2)))
    v2 = qv[sid][qt] + 0.5 * emb[sid][hits].mean(0); v2 /= np.linalg.norm(v2)
    r2 = rrf([top(s2, 200), top(emb[sid] @ v2, 200)], n)
    return rrf([top(base, 200), top(r2, 200)], n)

def named_sessions(c, qt):
    ql = qt.lower(); dates = set()
    for mm in re.finditer(r"(\d{1,2})\s+(" + "|".join(MONTHS) + r")", ql): dates.add((mm.group(1), mm.group(2)))
    for mm in re.finditer(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})", ql): dates.add((mm.group(2), mm.group(1)))
    months = {x for x in MONTHS if re.search(r"\b" + x + r"\b", ql)}; exact = set(); mon = set()
    for t in c["turns"]:
        d = t["date"].lower()
        if any(re.search(r"\b" + dd + r"\s+" + mo + r"\b|\b" + mo + r"\s+" + dd + r"\b", d) for dd, mo in dates): exact.add(t["session"])
        elif months and any(x in d for x in months): mon.add(t["session"])
    return exact or mon

def windows(c, qt, base, frac, budget=10):
    order = [int(i) for i in top(base, len(base))]; sess = named_sessions(c, qt)
    if not sess: return order
    inw = [j for j in order if c["turns"][j]["session"] in sess][: int(budget * frac)]
    s = set(inw); return inw + [j for j in order if j not in s]

V = {
 "5r": lambda c, sid, qt, b: [int(i) for i in top(b, len(b))],
 "5r + PRF": lambda c, sid, qt, b: [int(i) for i in top(prf(c, sid, qt, b), len(b))],
 "5r + date windows 50%": lambda c, sid, qt, b: windows(c, qt, b, 0.5),
 "5r + date windows 100%": lambda c, sid, qt, b: windows(c, qt, b, 1.0),
 "5r + PRF + date windows 50%": lambda c, sid, qt, b: windows(c, qt, prf(c, sid, qt, b), 0.5),
}
res = {v: {cat: {"h1": [], "r5": [], "r10": []} for cat in (1, 2)} for v in V}
for sid, q, ev in qs:
    c = convs[sid]; b = five_r(c, sid, q["question"])
    for v, fn in V.items():
        o = fn(c, sid, q["question"], b); ids = lambda k: {c["turns"][j]["id"] for j in o[:k]}
        r = res[v][q["category"]]; r["h1"].append(c["turns"][o[0]]["id"] in ev); r["r5"].append(all(e in ids(5) for e in ev)); r["r10"].append(all(e in ids(10) for e in ev))
print(f"{phase}: conversations {use}")
print("| variant | cat1 Hit@1 | cat1 R@5 all | cat1 R@10 all | cat2 Hit@1 | cat2 R@5 all | cat2 R@10 all |\n|---|---|---|---|---|---|---|")
for v in V:
    print(f"| {v} | " + " | ".join(f"{np.mean(res[v][cat][k]):.3f}" for cat in (1, 2) for k in ("h1", "r5", "r10")) + " |")
print("n:", {cat: len(res['5r'][cat]['h1']) for cat in (1, 2)})
