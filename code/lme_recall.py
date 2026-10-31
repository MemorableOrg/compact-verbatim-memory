#!/usr/bin/env python3
"""LongMemEval-S, offline, no answer model: where the gold turns sit in the 5a ranking, tokens per cut under Mem0's prompt, and
whether a token budget or a cap on long assistant turns keeps the gold while matching Mem0's 6,787 tokens per question.

    .venv/bin/python lme_recall.py            # prints dev then test; decisions are made on dev only

Labels (has_answer) are read only to score. Ranking is the one lme_run.py uses: RRF of top-200 stemmed BM25 and top-200 bge-small.
"""
import json, hashlib, pickle, os, re, sys, statistics as st, collections
import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi
import retrieval_eval as R
import mem0v2_lme_prompts as ML

enc = tiktoken.get_encoding("o200k_base")
CUTS = (20, 30, 35, 40, 50, 60, 80, 100, 150, 200)
BUDGETS = (5000, 6000, 6800, 8000, 10000, 13000)
CAPS = (60, 100, 150)
data = json.load(open("data/longmemeval_s_cleaned.json"))
order = sorted(data, key=lambda x: hashlib.sha256(x["question_id"].encode()).hexdigest())
dev_ids = {x["question_id"] for x in order[:100]}

CACHE = "lme_emb_cache.pkl"
cache = pickle.load(open(CACHE, "rb"))
missing = {sid for x in data for sid in x["haystack_session_ids"] if sid not in cache}
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); model.max_seq_length = 256
if missing:
    print("embedding", len(missing), "sessions not in the cache", flush=True)
    for x in data:
        for sid, sess in zip(x["haystack_session_ids"], x["haystack_sessions"]):
            if sid in missing and sid not in cache:
                cache[sid] = model.encode([f"{t['role']}: {t['content']}" for t in sess], batch_size=32, normalize_embeddings=True)
    pickle.dump(cache, open(CACHE, "wb"))
qvecs = model.encode([x["question"] for x in data], batch_size=32, normalize_embeddings=True)

def parse_date(s):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})", s)
    return tuple(int(g) for g in m.groups()) if m else (0,)

def prep(x):
    turns = []
    for si, (sid, date, sess) in enumerate(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"])):
        for ti, t in enumerate(sess):
            turns.append(dict(sid=sid, speaker=t["role"], date=date, text=t["content"], key=parse_date(date) + (si, ti), gold=bool(t.get("has_answer"))))
    full = [f"{t['speaker']} {t['date']} {t['text']}" for t in turns]
    return dict(turns=turns, bm25s=BM25Okapi([[R.stem(w) for w in R.tok(d)] for d in full]))

def ranking(x, qv):
    c = prep(x); n = len(c["turns"])
    E = np.vstack([cache[sid] for sid in x["haystack_session_ids"]])
    sc = R.rrf([R.top(R.bm25_scores(c, x["question"], True), 200), R.top(E @ qv, 200)], n)
    return c["turns"], [int(i) for i in R.top(sc, min(300, n))]

def ntok(s):
    return len(enc.encode(s, disallowed_special=()))

SENT = re.compile(r"(?<=[.!?])\s+")
def cap_turn(text, q_terms, cap):
    if ntok(text) <= cap:
        return text
    sents = SENT.split(text)
    sc = [(sum(1 for w in R.tok(s) if R.stem(w) in q_terms), -i) for i, s in enumerate(sents)]
    keep, used = set(), 0
    for _, (score, ni) in sorted(zip(sc, [(s, -i) for i, s in enumerate(sents)]), key=lambda p: (-p[0][0], p[0][1])):
        i = -ni; t = ntok(sents[i])
        if used + t > cap and keep:
            break
        keep.add(i); used += t
    return " ".join(sents[i] for i in sorted(keep))

def prompt_tokens(x, turns, idx, cap=None, q_terms=None):
    lines, cur = [], None
    for i in sorted(idx, key=lambda i: turns[i]["key"]):
        if turns[i]["date"] != cur:
            cur = turns[i]["date"]; lines.append(f"\n[{cur}]")
        txt = turns[i]["text"]
        if cap and turns[i]["speaker"] == "assistant":
            txt = cap_turn(txt, q_terms, cap)
        lines.append(f"{turns[i]['speaker']}: {txt}")
    return ntok(ML.ANSWER_GENERATION_PROMPT.format(memories="\n".join(lines).strip(), question=x["question"], question_date=x["question_date"]))

rows = []
for qi, (x, qv) in enumerate(zip(data, qvecs)):
    turns, rk = ranking(x, qv)
    gold = {i for i, t in enumerate(turns) if t["gold"]}
    if not gold:
        continue
    q_terms = {R.stem(w) for w in R.tok(x["question"]) if w not in R.STOP}
    rec = dict(split="dev" if x["question_id"] in dev_ids else "test", type=x["question_type"], ngold=len(gold),
               gold_roles=collections.Counter(turns[i]["speaker"] for i in gold), worst=max((rk.index(g) + 1 if g in rk else 999) for g in gold))
    for k in CUTS:
        rec[f"r@{k}"] = float(gold <= set(rk[:k]))
    for k in (35, 40, 60, 100, 150):
        rec[f"tok@{k}"] = prompt_tokens(x, turns, rk[:k])
    # token budget: turns in rank order until the prompt would pass B
    line_tok = [ntok(f"{turns[i]['speaker']}: {turns[i]['text']}") + 1 for i in rk]
    base = prompt_tokens(x, turns, [])
    for B in BUDGETS:
        used, n = base, 0
        for lt in line_tok:
            if used + lt > B:
                break
            used += lt; n += 1
        rec[f"b{B}_n"] = n; rec[f"b{B}_r"] = float(gold <= set(rk[:n]))
    # cap long assistant turns, 100 turns read
    for cap in CAPS:
        rec[f"cap{cap}_tok@100"] = prompt_tokens(x, turns, rk[:100], cap, q_terms)
        rec[f"cap{cap}_tok@150"] = prompt_tokens(x, turns, rk[:150], cap, q_terms)
    ans = str(x["answer"]).strip().lower()
    for i in gold:
        if turns[i]["speaker"] == "assistant" and len(ans) > 2 and ans in turns[i]["text"].lower():
            for cap in CAPS:
                rec.setdefault(f"surv{cap}", []).append(ans in cap_turn(turns[i]["text"], q_terms, cap).lower())
    rows.append(rec)
    if qi % 50 == 49:
        print(qi + 1, "questions", flush=True)

def mean(v):
    return 100 * st.mean(v) if v else float("nan")

for split in ("dev", "test"):
    rs = [r for r in rows if r["split"] == split]
    print(f"\n== {split} n={len(rs)}")
    print("all-gold turn recall:", " ".join(f"@{k} {mean([r[f'r@{k}'] for r in rs]):.1f}" for k in CUTS))
    for t in sorted({r["type"] for r in rs}):
        ts = [r for r in rs if r["type"] == t]
        print(f"  {t:28s} n={len(ts):3d} " + " ".join(f"@{k} {mean([r[f'r@{k}'] for r in ts]):5.1f}" for k in (35, 40, 60, 100, 150)))
    print("mean prompt tokens:", " ".join(f"@{k} {st.mean(r[f'tok@{k}'] for r in rs):,.0f}" for k in (35, 40, 60, 100, 150)), "| Mem0 6,787")
    print("token budget (turns added in rank order until the budget):")
    for B in BUDGETS:
        print(f"  B={B:6d}: mean turns {st.mean(r[f'b{B}_n'] for r in rs):5.1f}  all-gold recall {mean([r[f'b{B}_r'] for r in rs]):.1f}")
    print("assistant turns capped (sentences with most question terms kept):")
    for cap in CAPS:
        sv = [v for r in rs for v in r.get(f"surv{cap}", [])]
        print(f"  cap {cap:3d} tokens: prompt @100 {st.mean(r[f'cap{cap}_tok@100'] for r in rs):,.0f}  @150 {st.mean(r[f'cap{cap}_tok@150'] for r in rs):,.0f}  answer string survives in {sum(sv)} of {len(sv)} gold assistant turns that contain it")
    roles = collections.Counter()
    for r in rs:
        roles.update(r["gold_roles"])
    print("gold turns by role:", dict(roles), "| gold turns per question:", dict(sorted(collections.Counter(r["ngold"] for r in rs).items())))
    w = [r["worst"] for r in rs if not r["r@40"]]
    print("misses at 40: n=%d, worst gold rank" % len(w), sorted(w))
json.dump(rows, open("lme_recall_rows.json", "w"))
print("LME_RECALL_DONE")
