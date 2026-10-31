"""LongMemEval_S retrieval, dev/test split, offline. Scoring reuses snapgraph/src/score.py (official-style recall_any/recall_all/ndcg_any,
turn-to-session conversion with 'noans' ids). Abstention questions are excluded from retrieval metrics, as in the official script.
Split: sort question_ids by sha256(question_id) hex ascending; first 100 dev, other 400 test (same hash order as snapgraph/src/split.py).
    .venv/bin/python lme_split_eval.py dev | test
"""
import json, sys, os, pickle, hashlib, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Documents/GitHub/snapgraph/src"))
from score import score_turn, mean_metrics
import bm25s
import retrieval_eval as R
from retrieval_eval import top, rrf, tok
from sentence_transformers import SentenceTransformer, CrossEncoder

phase = sys.argv[1]
data = json.load(open("data/longmemeval_s_cleaned.json"))
order = sorted((x["question_id"] for x in data), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
dev = set(order[:100])
sel = [x for x in data if (x["question_id"] in dev) == (phase == "dev") and "_abs" not in x["question_id"]]
print(phase, "questions scored (abstention excluded):", len(sel), flush=True)
cache = pickle.load(open("lme_emb_cache.pkl", "rb"))
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); m.max_seq_length = 256
qv = m.encode([x["question"] for x in sel], batch_size=32, normalize_embeddings=True)
RRC = "lme_rr_cache.pkl"
rr = pickle.load(open(RRC, "rb")) if os.path.exists(RRC) else {}
ce = [None]
zs = lambda v: (v - v.mean()) / (v.std() or 1.0)

NAMES = ["BM25 (bm25s k1=1.5 b=0.75)", "dense bge-small", "z-fusion 0.4 BM25 + 0.6 dense", "5a (cleaned BM25 + dense, RRF)", "5r (5a + reranker)"]
rows = {n: [] for n in NAMES}; nturn = []; nsess = []
t0 = time.time()
for qi, x in enumerate(sel):
    turns = []
    for sid, date, sess in zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]):
        for ti, t in enumerate(sess): turns.append(dict(sid=sid, ti=ti, speaker=t["role"], date=date, text=t["content"], has=bool(t.get("has_answer"))))
    n = len(turns); nturn.append(n); nsess.append(len(x["haystack_sessions"]))
    E = np.vstack([cache[s] for s in x["haystack_session_ids"]]); d = E @ qv[qi]
    docs = [tok(f"{t['speaker']} {t['date']} {t['text']}") for t in turns]
    bm = bm25s.BM25(k1=1.5, b=0.75); bm.index(docs, show_progress=False)
    b = np.array(bm.get_scores(tok(x["question"])))
    from rank_bm25 import BM25Okapi
    c = dict(turns=turns, bm25s=BM25Okapi([[R.stem(w) for w in tok(f"{t['speaker']} {t['date']} {t['text']}")] for t in turns]))
    cl = R.bm25_scores(c, x["question"], True)
    five = rrf([top(cl, 200), top(d, 200)], n)
    cand = [int(i) for i in top(five, 100)]
    need = [j for j in cand if (x["question_id"], j) not in rr]
    if need:
        if ce[0] is None: ce[0] = CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=320)
        s = ce[0].predict([(x["question"], f"{turns[j]['speaker']} ({turns[j]['date']}): {turns[j]['text']}") for j in need], batch_size=32, show_progress_bar=False)
        for j, v in zip(need, s): rr[(x["question_id"], j)] = float(v)
    ordce = sorted(cand, key=lambda j: -rr[(x["question_id"], j)])
    five_r = rrf([ordce, cand], n)
    lab = dict(abstention=False, gold_sessions_b=set(x["answer_session_ids"]), gold_turns_b={(t["sid"], t["ti"]) for t in turns if t["has"]})
    for name, sc in zip(NAMES, [b, d, 0.4 * zs(b) + 0.6 * zs(d), five, five_r]):
        o = np.argsort(-sc, kind="stable")
        rows[name].append(score_turn([(turns[j]["sid"], turns[j]["ti"]) for j in o], lab, "B"))
    if qi % 20 == 0: print(qi, f"{time.time()-t0:.0f}s", flush=True); pickle.dump(rr, open(RRC, "wb"))
pickle.dump(rr, open(RRC, "wb"))
print(f"haystack per question: mean {np.mean(nturn):.0f} turns, mean {np.mean(nsess):.1f} sessions")
out = {}
for lvl in ("session", "turn"):
    print(f"\n{lvl}-level, {phase}, n={len(sel)}\n| variant | R_any@5 | R_all@5 | R_any@10 | R_all@10 | NDCG_any@10 |\n|---|---|---|---|---|---|")
    for name in NAMES:
        mm = mean_metrics([r[lvl] for r in rows[name]]); out[(lvl, name)] = mm
        print(f"| {name} | {mm['recall_any@5']:.1f} | {mm['recall_all@5']:.1f} | {mm['recall_any@10']:.1f} | {mm['recall_all@10']:.1f} | {mm['ndcg_any@10']:.1f} |")
json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()}, open(f"lme_split_{phase}.json", "w"))
print("DONE")
