#!/usr/bin/env python3
"""LongMemEval-S, offline: which turns to hand the reader so the prompt lands near Mem0's 6,787 tokens without losing gold turns.

    RR=0 .venv/bin/python lme_policy.py       # 5a ranking (BM25 + bge-small, RRF)
    RR=150 .venv/bin/python lme_policy.py     # 5a then bge-reranker-base over the top 150

Gold user turns count when they are in the set. Gold assistant turns count when they are in the set and, if the answer string
appears in the full turn, it still appears after capping. Decisions are made on dev; test is printed for the record.
"""
import json, hashlib, pickle, os, re, statistics as st, collections, time
import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi
import retrieval_eval as R
import mem0v2_lme_prompts as ML

RR = int(os.environ.get("RR", "0"))
enc = tiktoken.get_encoding("o200k_base")
def ntok(s):
    return len(enc.encode(s, disallowed_special=()))

data = json.load(open("data/longmemeval_s_cleaned.json"))
order = sorted(data, key=lambda x: hashlib.sha256(x["question_id"].encode()).hexdigest())
dev_ids = {x["question_id"] for x in order[:100]}
cache = pickle.load(open("lme_emb_cache.pkl", "rb"))
from sentence_transformers import SentenceTransformer, CrossEncoder
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); model.max_seq_length = 256
qvecs = model.encode([x["question"] for x in data], batch_size=32, normalize_embeddings=True)
ce = None
RRC = "lme_policy_rr_cache.pkl"
rrc = pickle.load(open(RRC, "rb")) if os.path.exists(RRC) else {}

def parse_date(s):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})", s)
    return tuple(int(g) for g in m.groups()) if m else (0,)

def prep(x):
    turns = []
    for si, (sid, date, sess) in enumerate(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"])):
        for ti, t in enumerate(sess):
            turns.append(dict(sid=sid, ti=ti, speaker=t["role"], date=date, text=t["content"], key=parse_date(date) + (si, ti), gold=bool(t.get("has_answer"))))
    full = [f"{t['speaker']} {t['date']} {t['text']}" for t in turns]
    return dict(turns=turns, bm25s=BM25Okapi([[R.stem(w) for w in R.tok(d)] for d in full]))

def ranking(x, qv):
    global ce
    c = prep(x); n = len(c["turns"])
    E = np.vstack([cache[sid] for sid in x["haystack_session_ids"]])
    sc = R.rrf([R.top(R.bm25_scores(c, x["question"], True), 200), R.top(E @ qv, 200)], n)
    rk = [int(i) for i in R.top(sc, min(300, n))]
    if RR:
        head = rk[:RR]
        need = [i for i in head if (x["question_id"], c["turns"][i]["sid"], c["turns"][i]["ti"]) not in rrc]
        if need:
            if ce is None:
                try:
                    ce = CrossEncoder("BAAI/bge-reranker-base", device="mps", max_length=320)
                except Exception:
                    ce = CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=320)
            sc2 = ce.predict([(x["question"], f"{c['turns'][i]['speaker']}: {c['turns'][i]['text']}") for i in need], batch_size=32, show_progress_bar=False)
            for i, s in zip(need, sc2):
                rrc[(x["question_id"], c["turns"][i]["sid"], c["turns"][i]["ti"])] = float(s)
        head.sort(key=lambda i: -rrc[(x["question_id"], c["turns"][i]["sid"], c["turns"][i]["ti"])])
        rk = head + rk[RR:]
    return c["turns"], rk

SENT = re.compile(r"(?<=[.!?])\s+")
def cap_turn(text, q_terms, cap):
    if ntok(text) <= cap:
        return text
    sents = SENT.split(text)
    scored = sorted(range(len(sents)), key=lambda i: (-sum(1 for w in R.tok(sents[i]) if R.stem(w) in q_terms), i))
    keep, used = [], 0
    for i in scored:
        t = ntok(sents[i])
        if used + t > cap and keep:
            break
        keep.append(i); used += t
    return " ".join(sents[i] for i in sorted(keep))

def build(x, turns, rk, K, mode, A, C, q_terms):
    """mode: whole | drop | cap | topA (assistant whole only within the top A, else dropped) | topAcap (else capped at C)"""
    chosen = []
    for pos, i in enumerate(rk[:K]):
        t = turns[i]
        if t["speaker"] == "user":
            chosen.append((i, t["text"]))
        elif mode == "whole":
            chosen.append((i, t["text"]))
        elif mode == "drop":
            continue
        elif mode == "cap":
            chosen.append((i, cap_turn(t["text"], q_terms, C)))
        elif mode == "topA":
            if pos < A:
                chosen.append((i, t["text"]))
        elif mode == "topAcap":
            chosen.append((i, t["text"] if pos < A else cap_turn(t["text"], q_terms, C)))
    chosen.sort(key=lambda p: turns[p[0]]["key"])
    lines, cur = [], None
    for i, txt in chosen:
        if turns[i]["date"] != cur:
            cur = turns[i]["date"]; lines.append(f"\n[{cur}]")
        lines.append(f"{turns[i]['speaker']}: {txt}")
    tok = ntok(ML.ANSWER_GENERATION_PROMPT.format(memories="\n".join(lines).strip(), question=x["question"], question_date=x["question_date"]))
    return dict(chosen), tok

POLICIES = [("whole", 0, 0), ("drop", 0, 0), ("cap", 0, 40), ("cap", 0, 60), ("cap", 0, 100), ("topA", 10, 0), ("topA", 20, 0), ("topAcap", 10, 60), ("topAcap", 20, 60), ("topAcap", 20, 40)]
KS = (40, 60, 80, 100, 150)
rows = []
t0 = time.time()
for qi, (x, qv) in enumerate(zip(data, qvecs)):
    turns, rk = ranking(x, qv)
    gold = [i for i, t in enumerate(turns) if t["gold"]]
    if not gold:
        continue
    ans = str(x["answer"]).strip().lower()
    q_terms = {R.stem(w) for w in R.tok(x["question"]) if w not in R.STOP}
    rec = dict(split="dev" if x["question_id"] in dev_ids else "test", type=x["question_type"],
               utok=[ntok(t["text"]) for t in turns if t["speaker"] == "user"], atok=[ntok(t["text"]) for t in turns if t["speaker"] == "assistant"],
               a_in_top100=sum(turns[i]["speaker"] == "assistant" for i in rk[:100]), res={})
    for K in KS:
        for mode, A, C in POLICIES:
            chosen, tok = build(x, turns, rk, K, mode, A, C, q_terms)
            ok = True
            for g in gold:
                if g not in chosen:
                    ok = False; break
                if turns[g]["speaker"] == "assistant" and len(ans) > 2 and ans in turns[g]["text"].lower() and ans not in chosen[g].lower():
                    ok = False; break
            rec["res"][f"{K}|{mode}|{A}|{C}"] = (tok, ok)
    rows.append(rec)
    if qi % 50 == 49:
        print(qi + 1, "questions", f"{time.time() - t0:.0f}s", flush=True)
        if RR:
            pickle.dump(rrc, open(RRC, "wb"))
if RR:
    pickle.dump(rrc, open(RRC, "wb"))

for split in ("dev", "test"):
    rs = [r for r in rows if r["split"] == split]
    print(f"\n== {split} n={len(rs)} RR={RR}")
    print("tokens per turn: user mean %.0f, assistant mean %.0f; assistant turns among the top 100: %.1f" % (
        st.mean(v for r in rs for v in r["utok"]), st.mean(v for r in rs for v in r["atok"]), st.mean(r["a_in_top100"] for r in rs)))
    print(f"{'turns':>5s} {'assistant handling':22s} {'tokens':>7s} {'gold kept':>9s}   | Mem0 reads 6,787")
    for K in KS:
        for mode, A, C in POLICIES:
            key = f"{K}|{mode}|{A}|{C}"
            tok = st.mean(r["res"][key][0] for r in rs); ok = 100 * st.mean(r["res"][key][1] for r in rs)
            label = {"whole": "whole", "drop": "dropped", "cap": f"capped {C} tok", "topA": f"whole in top {A}, else dropped", "topAcap": f"whole in top {A}, else cap {C}"}[mode]
            print(f"{K:5d} {label:22s} {tok:7,.0f} {ok:9.1f}")
json.dump([{k: v for k, v in r.items() if k != "res"} | {"res": r["res"]} for r in rows], open(f"lme_policy_rows_rr{RR}.json", "w"))
print("LME_POLICY_DONE")
