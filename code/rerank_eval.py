"""Offline: does a local cross-encoder reranker over the 5a top-100 raise gold-evidence recall at 40 and 60 turns?
No API calls. Writes rerank_eval_out.json and prints a table.

    EMB=BAAI/bge-small-en-v1.5 .venv/bin/python rerank_eval.py [BAAI/bge-reranker-base]
"""
import json, os, re, sys, time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from retrieval_eval import load, bm25_scores, top, rrf

EMB = os.environ.get("EMB", "BAAI/bge-small-en-v1.5")
RERANKER = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-reranker-base"
POOL = 100


def main():
    convs = load()
    z = np.load(f"emb_cache_{EMB.split('/')[-1]}.npz")
    emb = {k: z[k] for k in z.files if k != "_time"}
    m = SentenceTransformer(EMB, device="cpu"); m.max_seq_length = 256
    qs = [(sid, q) for sid, c in convs.items() for q in c["qa"] if q["category"] in (1, 2, 3, 4)]
    qs = [(s, q) for s, q in qs if [e for x in q.get("evidence", []) for e in re.findall(r"D\d+:\d+", x)]]
    qemb = {}
    for sid in convs:
        sq = [q["question"] for s, q in qs if s == sid]
        qemb[sid] = dict(zip(sq, m.encode(sq, batch_size=32, normalize_embeddings=True)))
    ce = CrossEncoder(RERANKER, device="cpu", max_length=320)

    def pool(c, sid, q):
        n = len(c["turns"])
        sc = rrf([top(bm25_scores(c, q, True), 200), top(emb[sid] @ qemb[sid][q], 200)], n)
        return top(sc, POOL), sc

    t0 = time.time()
    rows = {"5a": [], "rerank": [], "rerank+5a": []}
    for i, (sid, q) in enumerate(qs):
        c = convs[sid]
        cand, sc = pool(c, sid, q["question"])
        pairs = [(q["question"], f"{c['turns'][j]['speaker']} ({c['turns'][j]['date']}): {c['turns'][j]['text']}") for j in cand]
        s = ce.predict(pairs, batch_size=32, show_progress_bar=False)
        order_ce = [cand[j] for j in np.argsort(-np.asarray(s), kind="stable")]
        fused = rrf([order_ce, cand], len(c["turns"]))
        order_f = top(fused, POOL)
        ev = [e for x in q["evidence"] for e in re.findall(r"D\d+:\d+", x)]
        for name, order in (("5a", cand), ("rerank", order_ce), ("rerank+5a", order_f)):
            r = {}
            for k in (20, 40, 60):
                ids = {c["turns"][j]["id"] for j in order[:k]}
                r[k] = (all(e in ids for e in ev), sum(e in ids for e in ev) / len(ev))
            rows[name].append(r)
        if i % 200 == 0:
            print(i, f"{time.time()-t0:.0f}s", flush=True)
    print(f"reranked {len(qs)} questions x {POOL} turns with {RERANKER} in {time.time()-t0:.0f}s")
    print("| variant | all@20 | frac@20 | all@40 | frac@40 | all@60 | frac@60 |\n|---|---|---|---|---|---|---|")
    out = {}
    for name, rs in rows.items():
        cells = []
        for k in (20, 40, 60):
            a = np.mean([r[k][0] for r in rs]); f = np.mean([r[k][1] for r in rs])
            cells += [a, f]
        out[name] = cells
        print("| " + name + " | " + " | ".join(f"{x:.3f}" for x in cells) + " |")
    json.dump({"reranker": RERANKER, "emb": EMB, "n": len(qs), "rows": out}, open("rerank_eval_out.json", "w"))


if __name__ == "__main__":
    main()
