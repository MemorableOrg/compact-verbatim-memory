"""Offline: pseudo-relevance feedback and date windows over the 5a retriever. No API calls.
    .venv/bin/python prf_eval.py [rerank]   (with 'rerank' also scores the rerank combos, slow)
"""
import json, re, sys, time, pickle, os
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
import retrieval_eval as R
from retrieval_eval import load, top, rrf, tok, stem, STOP, QWORDS, MONTHS

DO_RR = len(sys.argv) > 1 and sys.argv[1] == "rerank"
convs = load()
z = np.load("emb_cache_bge-small-en-v1.5.npz"); emb = {k: z[k] for k in z.files if k != "_time"}
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); m.max_seq_length = 256
qs = [(sid, q) for sid, c in convs.items() for q in c["qa"] if q["category"] in (1, 2, 3, 4)]
qs = [(s, q) for s, q in qs if [e for x in q.get("evidence", []) for e in re.findall(r"D\d+:\d+", x)]]
qv = {}
for sid in convs:
    sq = [q["question"] for s, q in qs if s == sid]
    qv[sid] = dict(zip(sq, m.encode(sq, batch_size=32, normalize_embeddings=True)))

def bm25_stem(c, toks): return np.array(c["bm25s"].get_scores(toks)) if toks else np.zeros(len(c["turns"]))

def five_a(c, sid, q):
    n = len(c["turns"])
    return rrf([top(R.bm25_scores(c, q, True), 200), top(emb[sid] @ qv[sid][q], 200)], n)

def prf(c, sid, q, base):
    n = len(c["turns"]); hits = top(base, 5)
    txt = " ".join(c["turns"][j]["text"] for j in hits)
    caps = {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", txt)}
    nums = set(re.findall(r"\b\d+\b", txt))
    df = Counter(w for d in c["bm25s"].doc_freqs for w in d)
    words = {stem(w) for w in tok(txt) if w not in STOP and w not in QWORDS and len(w) > 2}
    rare = sorted((w for w in words if df[w] > 0), key=lambda w: df[w])[:10]
    exp = [stem(w) for w in caps] + list(nums) + rare
    qt = [stem(w) for w in tok(q) if w not in QWORDS and w not in STOP] or [stem(w) for w in tok(q)]
    s2 = np.array(c["bm25s"].get_scores(qt)) + 0.5 * bm25_stem(c, exp)
    s2 = np.array(R.date_boost(c, q, list(s2)))
    v2 = qv[sid][q] + 0.5 * emb[sid][hits].mean(0); v2 /= np.linalg.norm(v2)
    r2 = rrf([top(s2, 200), top(emb[sid] @ v2, 200)], n)
    return rrf([top(base, 200), top(r2, 200)], n)

def named_sessions(c, q):
    ql = q.lower(); T = c["turns"]; exact = set(); mon = set()
    dates = set()
    for mm in re.finditer(r"(\d{1,2})\s+(" + "|".join(MONTHS) + r")", ql): dates.add((mm.group(1), mm.group(2)))
    for mm in re.finditer(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})", ql): dates.add((mm.group(2), mm.group(1)))
    months = {x for x in MONTHS if re.search(r"\b" + x + r"\b", ql)}
    for t in T:
        d = t["date"].lower()
        if any(re.search(r"\b" + dd + r"\s+" + mo + r"\b|\b" + mo + r"\s+" + dd + r"\b", d) for dd, mo in dates): exact.add(t["session"])
        elif months and any(x in d for x in months): mon.add(t["session"])
    return exact or mon

def dates(c, q, cat, base, k, frac, proxy):
    sess = named_sessions(c, q)
    if not sess and cat == 2 and proxy: sess = {c["turns"][j]["session"] for j in top(base, 2)}
    order = top(base, len(base))
    if not sess: return order
    inw = [j for j in order if c["turns"][j]["session"] in sess][: int(k * frac)]
    rest = [j for j in order if j not in set(inw)]
    return inw + rest

def evalv(name, fn):
    res = {20: [], 40: [], 60: []}
    for sid, q in qs:
        c = convs[sid]; ev = [e for x in q["evidence"] for e in re.findall(r"D\d+:\d+", x)]
        for k in res:
            order = fn(c, sid, q, k)
            ids = {c["turns"][j]["id"] for j in order[:k]}
            res[k].append((all(e in ids for e in ev), sum(e in ids for e in ev) / len(ev)))
    row = [np.mean([x[i] for x in res[k]]) for k in (20, 40, 60) for i in (0, 1)]
    print(f"| {name} | " + " | ".join(f"{x:.3f}" for x in row) + " |", flush=True); return row

memo = {}
def A(c, sid, q):
    key = (sid, q["question"] if isinstance(q, dict) else q)
    return None
def base_of(c, sid, qt):
    k = ("b", sid, qt)
    if k not in memo: memo[k] = five_a(c, sid, qt)
    return memo[k]
def prf_of(c, sid, qt):
    k = ("p", sid, qt)
    if k not in memo: memo[k] = prf(c, sid, qt, base_of(c, sid, qt))
    return memo[k]
qcat = {(s, q["question"]): q["category"] for s, q in qs}

print("| variant | all@20 | frac@20 | all@40 | frac@40 | all@60 | frac@60 |\n|---|---|---|---|---|---|---|")
rows = {}
def reg(name, fn): rows[name] = evalv(name, fn)
O = lambda sc: list(top(sc, len(sc)))
# fns take (c,sid,q_dict,k); need question text and category
def wrap(f): return lambda c, sid, q, k: f(c, sid, q["question"], q["category"], k)
reg("5a", wrap(lambda c, sid, qt, cat, k: O(base_of(c, sid, qt))))
reg("1 PRF", wrap(lambda c, sid, qt, cat, k: O(prf_of(c, sid, qt))))
for frac in (0.5, 1.0):
    reg(f"2 date windows, named dates only, {int(frac*100)}% of budget", wrap(lambda c, sid, qt, cat, k, fr=frac: dates(c, qt, 0, base_of(c, sid, qt) , k, fr, False) if False else dates(c, qt, cat, base_of(c, sid, qt), k, fr, False)))
    reg(f"2 date windows, + session of top-2 hits for cat2, {int(frac*100)}% of budget", wrap(lambda c, sid, qt, cat, k, fr=frac: dates(c, qt, cat, base_of(c, sid, qt), k, fr, True)))
reg("1+2 PRF then date windows (named, 50%)", wrap(lambda c, sid, qt, cat, k: dates(c, qt, cat, prf_of(c, sid, qt), k, 0.5, False)))
json.dump(rows, open("prf_eval_out.json", "w"))

if DO_RR:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=320)
    cp = pickle.load(open("rr_cache.pkl", "rb")) if os.path.exists("rr_cache.pkl") else {}
    def rr(c, sid, qt, base_sc):
        n = len(c["turns"]); cand = [int(i) for i in top(base_sc, 100)]
        need = [j for j in cand if (sid, qt, j) not in cp]
        if need:
            s = ce.predict([(qt, f"{c['turns'][j]['speaker']} ({c['turns'][j]['date']}): {c['turns'][j]['text']}") for j in need], batch_size=32, show_progress_bar=False)
            for j, v in zip(need, s): cp[(sid, qt, j)] = float(v)
        ordce = sorted(cand, key=lambda j: -cp[(sid, qt, j)])
        return rrf([ordce, cand], n)
    R5 = lambda c, sid, qt: rr(c, sid, qt, base_of(c, sid, qt))
    RP = lambda c, sid, qt: rr(c, sid, qt, prf_of(c, sid, qt))
    reg("5r (rerank+5a)", wrap(lambda c, sid, qt, cat, k: O(R5(c, sid, qt))))
    reg("3 5r + date windows (cat2 proxy, 50%)", wrap(lambda c, sid, qt, cat, k: dates(c, qt, cat, R5(c, sid, qt), k, 0.5, True)))
    reg("3 5r + date windows (named only, 50%)", wrap(lambda c, sid, qt, cat, k: dates(c, qt, cat, R5(c, sid, qt), k, 0.5, False)))
    reg("3 5r over PRF pool", wrap(lambda c, sid, qt, cat, k: O(RP(c, sid, qt))))
    reg("3 5r over PRF pool + date windows (named, 50%)", wrap(lambda c, sid, qt, cat, k: dates(c, qt, cat, RP(c, sid, qt), k, 0.5, False)))
    pickle.dump(cp, open("rr_cache.pkl", "wb"))
    json.dump(rows, open("prf_eval_out.json", "w"))
