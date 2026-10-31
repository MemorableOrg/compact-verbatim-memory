"""Dev/test split evaluation of LoCoMo retrievers. No API calls. Reads evidence ids only to score.
Split: conversations 0-2 in file order (conv-26, conv-30, conv-41) are dev, the other 7 are test.
    .venv/bin/python split_eval.py dev      (all variants on dev, writes split_dev.json)
    .venv/bin/python split_eval.py test W   (winner W and baseline on test, once)
"""
import json, re, sys, os, pickle, time
import numpy as np
from sentence_transformers import SentenceTransformer
import retrieval_eval as R
from retrieval_eval import top, rrf

phase = sys.argv[1]
convs = R.load()
DEV = list(convs)[:3]
z = np.load("emb_cache_bge-small-en-v1.5.npz"); emb = {k: z[k] for k in z.files if k != "_time"}
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); m.max_seq_length = 256
use = DEV if phase == "dev" else [s for s in convs if s not in DEV]
qs = [(sid, q, [e for x in q["evidence"] for e in re.findall(r"D\d+:\d+", x)]) for sid in use for q in convs[sid]["qa"] if q["category"] in (1, 2, 3, 4) and q.get("evidence")]
qs = [x for x in qs if x[2]]
qv = {}
for sid in use:
    sq = [q["question"] for s, q, _ in qs if s == sid]
    qv[sid] = dict(zip(sq, m.encode(sq, batch_size=32, normalize_embeddings=True)))
print(phase, "conversations", use, "questions", len(qs), flush=True)

zs = lambda x: (x - x.mean()) / (x.std() or 1.0)
cp = pickle.load(open("rr_cache.pkl", "rb")) if os.path.exists("rr_cache.pkl") else {}
_ce = [None]
def rerank_scores(c, sid, qt, base):
    cand = [int(i) for i in top(base, 100)]
    need = [j for j in cand if (sid, qt, j) not in cp]
    if need:
        if _ce[0] is None:
            from sentence_transformers import CrossEncoder
            _ce[0] = CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=320)
        s = _ce[0].predict([(qt, f"{c['turns'][j]['speaker']} ({c['turns'][j]['date']}): {c['turns'][j]['text']}") for j in need], batch_size=32, show_progress_bar=False)
        for j, v in zip(need, s): cp[(sid, qt, j)] = float(v)
    ordce = sorted(cand, key=lambda j: -cp[(sid, qt, j)])
    return rrf([ordce, cand], len(c["turns"]))

def scores(name, c, sid, qt):
    d = emb[sid] @ qv[sid][qt]
    if name == "baseline BM25": return np.array(R.bm25_scores(c, qt, False))
    if name == "cleaned BM25": return np.array(R.bm25_scores(c, qt, True))
    if name == "dense bge-small": return d
    five = rrf([top(R.bm25_scores(c, qt, True), 200), top(d, 200)], len(c["turns"]))
    if name == "5a RRF": return five
    if name == "5r rerank+5a": return rerank_scores(c, sid, qt, five)
    if name.startswith("zfuse"):
        w = float(name.split("=")[1]); return w * zs(np.array(R.bm25_scores(c, qt, True))) + (1 - w) * zs(d)
    raise KeyError(name)

KS = (1, 5, 10, 20, 40)
def evaluate(name):
    out = {}
    for cat in (0, 1, 2, 3, 4):
        out[cat] = {f"{t}{k}": [] for k in KS for t in ("all", "any")}
    for sid, q, ev in qs:
        c = convs[sid]; order = top(scores(name, c, sid, q["question"]), 40)
        for cat in (0, q["category"]):
            for k in KS:
                ids = {c["turns"][j]["id"] for j in order[:k]}
                out[cat][f"all{k}"].append(all(e in ids for e in ev)); out[cat][f"any{k}"].append(any(e in ids for e in ev))
    return {cat: {key: (float(np.mean(v)), len(v)) for key, v in d.items()} for cat, d in out.items()}

def row(name, r, cat=0):
    d = r[cat]
    return f"| {name} | {d['all1'][0]:.3f} | {d['all5'][0]:.3f} | {d['any5'][0]:.3f} | {d['all10'][0]:.3f} | {d['any10'][0]:.3f} | {d['all40'][0]:.3f} | {d['all1'][1]} |"
HDR = "| variant | Hit@1 | R@5 all | R@5 any | R@10 all | R@10 any | R@40 all | n |\n|---|---|---|---|---|---|---|---|"

if phase == "dev":
    names = ["baseline BM25", "cleaned BM25", "dense bge-small", "5a RRF", "5r rerank+5a"] + [f"zfuse w={w}" for w in (0.3, 0.4, 0.5, 0.6, 0.7)]
    res = {}
    print(HDR)
    for nm in names:
        t0 = time.time(); res[nm] = evaluate(nm); print(row(nm, res[nm]), f"  ({time.time()-t0:.0f}s)", flush=True)
        pickle.dump(cp, open("rr_cache.pkl", "wb"))
    json.dump(res, open("split_dev.json", "w"))
    zbest = max([n for n in names if n.startswith("zfuse")], key=lambda n: res[n][0]["all5"][0])
    print("best z-fusion weight on dev (by R@5 all):", zbest)
    win = max(names, key=lambda n: res[n][0]["all5"][0]); print("WINNER on dev by R@5 all:", win)
else:
    win = sys.argv[2]; res = {}
    print(HDR)
    for nm in ["baseline BM25", win]:
        res[nm] = evaluate(nm); print(row(nm, res[nm]), flush=True)
    pickle.dump(cp, open("rr_cache.pkl", "wb"))
    for nm in res:
        print("\nper category,", nm); print(HDR)
        for cat in (1, 2, 3, 4): print(row(f"cat{cat}", res[nm], cat))
    json.dump(res, open("split_test.json", "w"))
