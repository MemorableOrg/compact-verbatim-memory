import argparse, json, os, re, random, time, pickle, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
import retrieval_eval as R
from judge_prompt_lme import get_anscheck_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--k", type=int, default=40)
ap.add_argument("--answer-model", default="gpt-6-sol")
ap.add_argument("--judge-model", default="gpt-4o-2024-08-06")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--answer-provider", choices=["openai", "anthropic"], default="openai")
ap.add_argument("--resume", default=None)
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--max-usd", type=float, default=20.0)
ap.add_argument("--adaptive", default="", help="K1,K2: read K1 turns first; if the reader says NOT FOUND, read K2 turns and answer again")
ap.add_argument("--split", choices=["all", "dev", "test"], default="all", help="dev = first 100 question_ids by sha256 (Addendum 22 task C); test = the other 400")
ap.add_argument("--rerank", type=int, default=0, help="R: re-score the top R turns with bge-reranker-base (local, cached) before the cut")
ap.add_argument("--assistant-top", type=int, default=0, help="A: hand the reader every user turn in the top k, and an assistant turn only when it ranks in the top A (0 = keep all)")
ap.add_argument("--prompt", choices=["official", "mem0v2"], default="official", help="mem0v2: Mem0's current LongMemEval answer prompt (memories grouped by date, mem_thinking stripped)")
a = ap.parse_args()

data = json.load(open("data/longmemeval_s_cleaned.json"))
random.seed(a.seed)
groups = defaultdict(list)
if a.split != "all":
    import hashlib
    order = sorted(data, key=lambda x: hashlib.sha256(x["question_id"].encode()).hexdigest())
    dev_ids = {x["question_id"] for x in order[:100]}
    data = [x for x in data if (x["question_id"] in dev_ids) == (a.split == "dev")]
for x in data: groups[x["question_type"]].append(x)
for g in groups.values(): random.shuffle(g)
sel = []; keys = sorted(groups)
while len(sel) < a.n and any(groups[k] for k in keys):
    for k in keys:
        if groups[k] and len(sel) < a.n: sel.append(groups[k].pop())
prev = [json.loads(l) for l in open(a.resume)] if a.resume else []
_done = {r["question_id"] for r in prev}
sel = [x for x in sel if x["question_id"] not in _done]
print(f"running {len(sel)} of {len(data)} ({len(prev)} resumed)", flush=True)

def parse_date(s):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})", s)
    return tuple(int(g) for g in m.groups()) if m else (0,)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); model.max_seq_length = 256
CACHE = "lme_emb_cache.pkl"
cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
t0 = time.time(); new = 0
for x in sel:
    for sid, sess in zip(x["haystack_session_ids"], x["haystack_sessions"]):
        if sid not in cache:
            cache[sid] = model.encode([f"{t['role']}: {t['content']}" for t in sess], batch_size=32, normalize_embeddings=True); new += 1
print(f"embedded {new} new sessions in {time.time()-t0:.0f}s", flush=True)
pickle.dump(cache, open(CACHE, "wb"))
qvecs = model.encode([x["question"] for x in sel], batch_size=32, normalize_embeddings=True)

def prep(x):
    turns = []
    for si, (sid, date, sess) in enumerate(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"])):
        for ti, t in enumerate(sess):
            turns.append(dict(sid=sid, ti=ti, speaker=t["role"], date=date, text=t["content"], key=parse_date(date) + (si, ti), has_answer=bool(t.get("has_answer")), emb=(sid, ti)))
    full = [f"{t['speaker']} {t['date']} {t['text']}" for t in turns]
    stem = R.stem
    return dict(turns=turns, bm25s=BM25Okapi([[stem(w) for w in R.tok(d)] for d in full]))

def retrieve(x, qv, k):
    c = prep(x); n = len(c["turns"])
    E = np.vstack([cache[sid] for sid in x["haystack_session_ids"]])
    sc = R.rrf([R.top(R.bm25_scores(c, x["question"], True), 200), R.top(E @ qv, 200)], n)
    rk = [int(i) for i in R.top(sc, max(k, a.rerank))]
    if a.rerank:
        head = rk[:a.rerank]
        s2 = rerank_scores(x, c["turns"], head)
        head.sort(key=lambda i: -s2[i])
        rk = head + rk[a.rerank:]
    rk = rk[:k]
    if a.assistant_top:
        rk = [i for pos, i in enumerate(rk) if c["turns"][i]["speaker"] == "user" or pos < a.assistant_top]
    idx = sorted(rk, key=lambda i: c["turns"][i]["key"])
    return c["turns"], idx

_ce = [None]; RRC = "lme_policy_rr_cache.pkl"
_rrc = pickle.load(open(RRC, "rb")) if os.path.exists(RRC) else {}
_rr_lock = threading.Lock()
def rerank_scores(x, turns, cand):
    need = [i for i in cand if (x["question_id"], turns[i]["sid"], turns[i]["ti"]) not in _rrc]
    if need:
        with _rr_lock:
            if _ce[0] is None:
                from sentence_transformers import CrossEncoder
                try: _ce[0] = CrossEncoder("BAAI/bge-reranker-base", device="mps", max_length=320)
                except Exception: _ce[0] = CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=320)
            sc = _ce[0].predict([(x["question"], f"{turns[i]['speaker']}: {turns[i]['text']}") for i in need], batch_size=32, show_progress_bar=False)
            for i, v in zip(need, sc): _rrc[(x["question_id"], turns[i]["sid"], turns[i]["ti"])] = float(v)
    return {i: _rrc[(x["question_id"], turns[i]["sid"], turns[i]["ti"])] for i in cand}

import oa
client = oa.client()
use = {"in": sum(r["tokens"]["ain"] for r in prev), "out": sum(r["tokens"]["aout"] for r in prev), "jin": sum(r["tokens"]["jin"] for r in prev), "jout": sum(r["tokens"]["jout"] for r in prev)}
judge_model = [a.judge_model]

def chat(model, msgs, **kw):
    for attempt in range(14):
        try:
            return client.chat.completions.create(model=oa.dep(model), messages=msgs, **kw)
        except Exception as e:
            if attempt == 13 or "model" in str(e).lower() and "not" in str(e).lower() and "exist" in str(e).lower():
                raise
            time.sleep(min(2 ** attempt, 30))

def cost(): return use["in"] * 2e-6 + use["out"] * 10e-6

ap_provider = a.answer_provider
if ap_provider == "anthropic":
    import anthropic
    from types import SimpleNamespace as _NS
    _ac = anthropic.Anthropic()
    def answer_call(model, prompt):
        for attempt in range(14):
            try:
                m = _ac.messages.create(model=model, max_tokens=400, messages=[{"role": "user", "content": prompt}])
                break
            except Exception as e:
                if attempt == 13 or "not_found" in str(e).lower(): raise
                time.sleep(min(2 ** attempt, 30))
        txt = "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
        return _NS(choices=[_NS(message=_NS(content=txt))], usage=_NS(prompt_tokens=m.usage.input_tokens, completion_tokens=m.usage.output_tokens))
else:
    def answer_call(model, prompt): return chat(model, [{"role": "user", "content": prompt}])

os.makedirs("results", exist_ok=True)
out = f"results/lme-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"

def run(item):
    x, qv = item
    if cost() > a.max_usd: return None
    ks = [int(v) for v in a.adaptive.split(",")] if a.adaptive else [a.k]
    passes, ain, aout = 0, 0, 0
    for k in ks:
        passes += 1
        turns, idx = retrieve(x, qv, k)
        ctx = "\n".join(f"[{turns[i]['date']}] {turns[i]['speaker']}: {turns[i]['text']}" for i in idx)
        rule = " If the history chats do not contain the answer, reply exactly: NOT FOUND." if (len(ks) > 1 and k != ks[-1]) else ""
        if a.prompt == "mem0v2":
            import mem0v2_lme_prompts as ML
            lines, cur = [], None
            for i in idx:
                if turns[i]["date"] != cur:
                    cur = turns[i]["date"]; lines.append(f"\n[{cur}]")
                lines.append(f"{turns[i]['speaker']}: {turns[i]['text']}")
            prompt = ML.ANSWER_GENERATION_PROMPT.format(memories="\n".join(lines).strip(), question=x["question"], question_date=x["question_date"]) + (("\n" + rule) if rule else "")
        else:
            prompt = f"I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.{rule}\n\n\nHistory Chats:\n\n{ctx}\n\nCurrent Date: {x['question_date']}\nQuestion: {x['question']}\nAnswer:"
        r = answer_call(a.answer_model, prompt)
        ans = (r.choices[0].message.content or "").strip()
        if a.prompt == "mem0v2":
            ans = re.sub(r"<mem_thinking>.*?</mem_thinking>", "", ans, flags=re.S).strip()
        ain += r.usage.prompt_tokens; aout += r.usage.completion_tokens
        if "NOT FOUND" not in ans.upper():
            break
    use["in"] += ain; use["out"] += aout
    from types import SimpleNamespace as _NS3
    r = _NS3(usage=_NS3(prompt_tokens=ain, completion_tokens=aout))
    jp = get_anscheck_prompt(x["question_type"], x["question"], x["answer"], ans, abstention="_abs" in x["question_id"])
    try:
        jr = chat(judge_model[0], [{"role": "user", "content": jp}], n=1, temperature=0, max_tokens=10)
    except Exception as e:
        if judge_model[0] != "gpt-4o": judge_model[0] = "gpt-4o"; print("judge fallback to gpt-4o:", str(e)[:150], flush=True)
        jr = chat(judge_model[0], [{"role": "user", "content": jp}], n=1, temperature=0, max_tokens=10)
    use["jin"] += jr.usage.prompt_tokens; use["jout"] += jr.usage.completion_tokens
    label = "yes" in jr.choices[0].message.content.strip().lower()
    got = {turns[i]["sid"] for i in idx}; gold = set(x["answer_session_ids"])
    ans_turns = [(t["sid"], t["ti"]) for t in turns if t["has_answer"]]
    rets = {(turns[i]["sid"], turns[i]["ti"]) for i in idx}
    return dict(question_id=x["question_id"], type=x["question_type"], abstention="_abs" in x["question_id"], question=x["question"], gold=x["answer"],
        retrieved=[f"{turns[i]['sid']}:{turns[i]['ti']}" for i in idx], passes=passes, turns_read=len(idx), gold_sessions=sorted(gold), session_recall=gold <= got,
        session_frac=len(gold & got) / len(gold) if gold else None, turn_recall=all(t in rets for t in ans_turns) if ans_turns else None,
        answer=ans, correct=label, judge_model=judge_model[0],
        tokens=dict(ain=r.usage.prompt_tokens, aout=r.usage.completion_tokens, jin=jr.usage.prompt_tokens, jout=jr.usage.completion_tokens))

res = list(prev); stopped = False
with ThreadPoolExecutor(a.workers) as ex, open(out, "w") as f:
    for r0 in prev: f.write(json.dumps(r0) + "\n")
    for rec in ex.map(run, list(zip(sel, qvecs))):
        if rec is None: stopped = True; continue
        res.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()

n = len(res)
print(f"results: {out}  n={n}" + ("  STOPPED ON COST CAP" if stopped else ""))
print(f"accuracy overall: {sum(r['correct'] for r in res)/n:.3f}")
for t in sorted({r["type"] for r in res}):
    rs = [r for r in res if r["type"] == t]; print(f"  {t}: {sum(r['correct'] for r in rs)/len(rs):.3f} (n={len(rs)})")
ab = [r for r in res if r["abstention"]]
if ab: print(f"  abstention questions (inside the types above): {sum(r['correct'] for r in ab)/len(ab):.3f} (n={len(ab)})")
sr = [r["session_recall"] for r in res]; print(f"session recall@{a.k} (all gold sessions hit): {sum(sr)/n:.3f}")
sf = [r["session_frac"] for r in res if r["session_frac"] is not None]; print(f"session fraction: {sum(sf)/len(sf):.3f}")
tr = [r["turn_recall"] for r in res if r["turn_recall"] is not None]; print(f"answer-turn recall (all has_answer turns hit): {sum(tr)/len(tr):.3f} (n={len(tr)})")
print(f"judge model used: {judge_model[0]}")
print(f"answer tokens in/out: {use['in']}/{use['out']}  judge in/out: {use['jin']}/{use['jout']}")
print(f"answer-model cost: ${cost():.3f}  per question: ${cost()/n:.4f}  full 500 est: ${cost()/n*500:.2f}")
