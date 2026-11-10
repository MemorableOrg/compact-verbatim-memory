import argparse, json, os, re, random, time, hashlib, threading, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
import retrieval_eval as R
from beam_judge_prompt import JUDGE_PROMPT, JUDGE_MODEL

ABILITIES = ["abstention", "contradiction_resolution", "event_ordering", "information_extraction", "instruction_following",
             "knowledge_update", "multi_session_reasoning", "preference_following", "summarization", "temporal_reasoning"]
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.1, 0.5),
          "gpt-4.1-mini": (0.4, 1.6), "gpt-6-sol": (2.0, 10.0)}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/Users/nikhilkrishnaswamy/Documents/fm-data/beam/beam_1M.jsonl")
ap.add_argument("--n", type=int, default=700, help="questions to run (balanced over abilities); 700 = all of 1M")
ap.add_argument("--convs", default="", help="conversation indices, e.g. 0-1,5 (default all)")
ap.add_argument("--abilities", default="")
ap.add_argument("--k", type=int, default=40)
ap.add_argument("--adaptive", default="", help="K1,K2: reader may answer NOT FOUND at K1, then reads K2")
ap.add_argument("--rerank", action="store_true", help="bge-reranker-base over the top 100 fused candidates")
ap.add_argument("--pair", action="store_true", help="add 0.3 of each message's fused score to its user/assistant partner")
ap.add_argument("--standing", type=int, default=0, help="always show the N user messages stating a standing instruction or preference closest to the question")
ap.add_argument("--answer-provider", choices=["openai", "anthropic"], default="anthropic")
ap.add_argument("--answer-model", default="claude-sonnet-5-5")
ap.add_argument("--judge-model", default=JUDGE_MODEL)
ap.add_argument("--thinking", choices=["off", "default"], default="off", help="off: send thinking={type:between_tools} to Sonnet 5.5")
ap.add_argument("--max-answer-tokens", type=int, default=1500)
ap.add_argument("--max-turn-chars", type=int, default=6000)
ap.add_argument("--no-eo-align", action="store_true", help="skip BEAM's LLM-aligned event-ordering tau/F1 (rubric score is still computed)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default="", help="results jsonl of an earlier partial run with the same seed and n: skip its qids, count its rows in the summary")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--max-usd", type=float, default=5.0)
ap.add_argument("--estimate-only", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args()

CPT = 3.5  # chars per token, rough; calibrate from a run's usage
MARK = re.compile(r"\s*->->\s*[\d,]+\s*$")
STANDING = re.compile(r"^\s*(please\s+)?(always|never)\b|\bfrom now on\b|\bI (really )?prefer\b|\bmy preference\b", re.I)

def parse_idx(s, n):
    if not s: return list(range(n))
    out = []
    for p in s.split(","):
        if "-" in p: lo, hi = p.split("-"); out += list(range(int(lo), int(hi) + 1))
        else: out.append(int(p))
    return out

want = None
if a.convs: want = set(parse_idx(a.convs, 35))
abil = a.abilities.split(",") if a.abilities else ABILITIES

def flat_ids(x):
    out = []
    if isinstance(x, bool) or x is None: return out
    if isinstance(x, int): out.append(x)
    elif isinstance(x, (list, tuple)):
        for i in x: out += flat_ids(i)
    elif isinstance(x, dict):
        for i in x.values(): out += flat_ids(i)
    return out

def parse_pq(s):
    import ast
    try: return ast.literal_eval(s)
    except Exception: return json.loads(s)

convs = {}
for ci, line in enumerate(open(a.data)):
    if want is not None and ci not in want: continue
    r = json.loads(line)
    turns = []; date = ""
    for batch in r["chat"]:
        for t in batch:
            if t.get("time_anchor"): date = t["time_anchor"]
            turns.append(dict(id=t["id"], role=t["role"], date=date, text=MARK.sub("", t["content"])))
    full = [f"{t['role']} {t['date']} {t['text']}" for t in turns]
    pq = parse_pq(r["probing_questions"])
    qs = []
    for ab in ABILITIES:
        for qi, q in enumerate(pq.get(ab, [])):
            rub = q["rubric"]
            rub = [str(x) for x in rub] if isinstance(rub, list) else [str(rub)]
            qs.append(dict(conv=ci, qid=f"{ci}_{ab}_{qi}", ability=ab, question=q["question"], rubric=rub,
                           gold_ids=sorted(set(flat_ids(q.get("source_chat_ids"))))))
    convs[ci] = dict(cid=r["conversation_id"], turns=turns, full=full, qs=qs, chars=sum(len(t["text"]) for t in turns))

allq = [q for c in convs.values() for q in c["qs"] if q["ability"] in abil]
random.seed(a.seed)
groups = defaultdict(list)
for q in allq: groups[q["ability"]].append(q)
for g in groups.values(): random.shuffle(g)
sel = []; keys = sorted(groups)
while len(sel) < a.n and any(groups[k] for k in keys):
    for k in keys:
        if groups[k] and len(sel) < a.n: sel.append(groups[k].pop())
prev = [json.loads(l) for l in open(a.resume)] if a.resume else []
if prev:
    done = {r["qid"] for r in prev}; sel = [q for q in sel if q["qid"] not in done]
    print(f"resuming: {len(prev)} questions from {a.resume} skipped, {len(sel)} left", flush=True)
need = sorted({q["conv"] for q in sel})
print(f"conversations loaded {len(convs)}, questions eligible {len(allq)}, running {len(sel)} over {len(need)} conversations", flush=True)

ks = [int(v) for v in a.adaptive.split(",")] if a.adaptive else [a.k]
def price(model): return PRICES.get(model, (2.0, 10.0))
pin, pout = price(a.answer_model); jin, jout = price(a.judge_model)
avg_turn_tok = np.mean([c["chars"] / len(c["turns"]) / CPT for c in convs.values()])
read_tok = ks[0] + (0.5 * ks[1] if len(ks) > 1 else 0)
est_in = read_tok * min(avg_turn_tok, a.max_turn_chars / CPT) + 450
est_out = 350
est_judge_in = 900 * 3.6; est_judge_out = 60 * 3.6
est_q = est_in * pin / 1e6 + est_out * pout / 1e6 + est_judge_in * jin / 1e6 + est_judge_out * jout / 1e6
est_total = est_q * len(sel)
print(f"cost estimate: reader ~{est_in:,.0f} in / {est_out} out tokens per question, judge ~{est_judge_in:,.0f} in per question "
      f"-> ${est_q:.4f}/question, ${est_total:.2f} for {len(sel)} questions (max ${a.max_usd})", flush=True)
if est_total > a.max_usd:
    print("ESTIMATE EXCEEDS --max-usd: stopping"); sys.exit(2)
if a.estimate_only: sys.exit(0)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); model.max_seq_length = 256
os.makedirs("beam_emb", exist_ok=True)
ce = None
if a.rerank:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=320)
lock = threading.Lock()

def conv_index(ci):
    c = convs[ci]
    if "E" in c: return c
    docs = [f"{t['role']}: {t['text']}" for t in c["turns"]]
    h = hashlib.md5("\n".join(docs).encode()).hexdigest()[:12]
    path = f"beam_emb/bge-small_{c['cid']}_{h}.npz"
    if os.path.exists(path): E = np.load(path)["E"]
    else:
        t0 = time.time()
        E = model.encode(docs, batch_size=32, normalize_embeddings=True)
        np.savez(path, E=E); print(f"  embedded conv {ci} ({len(docs)} turns) in {time.time()-t0:.0f}s", flush=True)
    c["E"] = E
    c["pool"] = [i for i, t in enumerate(c["turns"]) if t["role"] == "user" and STANDING.search(t["text"][:400])]
    c["bm25s"] = BM25Okapi([[R.stem(w) for w in R.tok(d)] for d in c["full"]])
    return c

t0 = time.time()
for ci in need: conv_index(ci)
print(f"index ready in {time.time()-t0:.0f}s", flush=True)
qv = dict(zip([q["qid"] for q in sel], model.encode([q["question"] for q in sel], batch_size=32, normalize_embeddings=True)))

def retrieve(q, k):
    c = convs[q["conv"]]; n = len(c["turns"])
    dense = c["E"] @ qv[q["qid"]]
    sc = R.rrf([R.top(R.bm25_scores(c, q["question"], True), 200), R.top(dense, 200)], n)
    if a.pair:
        sc2 = sc.copy()
        for i in range(n):
            j = i + 1 if c["turns"][i]["role"] == "user" else i - 1
            if 0 <= j < n: sc2[j] += 0.3 * sc[i]
        sc = sc2
    if ce is not None:
        cand = [int(i) for i in R.top(sc, 100)]
        pairs = [(q["question"], f"{c['turns'][j]['role']} ({c['turns'][j]['date']}): {c['turns'][j]['text'][:1500]}") for j in cand]
        with lock: s = ce.predict(pairs, batch_size=32, show_progress_bar=False)
        order = [cand[j] for j in np.argsort(-np.asarray(s), kind="stable")]
        sc = R.rrf([order, cand], n)
    ranked = [int(i) for i in R.top(sc, n)]
    if a.standing and c["pool"]:
        keep = sorted(c["pool"], key=lambda i: -dense[i])[:a.standing]
        ranked = keep + [i for i in ranked if i not in set(keep)]
    return sorted(ranked[:k])

use = {"ain": 0, "aout": 0, "jin": 0, "jout": 0}
ulock = threading.Lock()
def add(k, v):
    with ulock: use[k] += v
def cost(): return use["ain"] * pin / 1e6 + use["aout"] * pout / 1e6 + use["jin"] * jin / 1e6 + use["jout"] * jout / 1e6

import oa
oc = oa.client()
def chat_openai(model_, prompt, **kw):
    for at in range(10):
        try: return oc.chat.completions.create(model=oa.dep(model_), messages=[{"role": "user", "content": prompt}], **kw)
        except Exception as e:
            if at == 9: raise
            time.sleep(min(2 ** at, 30))

if a.answer_provider == "anthropic":
    import anthropic
    ac = anthropic.Anthropic()
    use_think_off = [a.thinking == "off" and a.answer_model.startswith("claude-sonnet-5-5")]
    def answer_call(prompt):
        for at in range(10):
            try:
                kw = dict(model=a.answer_model, max_tokens=a.max_answer_tokens, messages=[{"role": "user", "content": prompt}])
                if use_think_off[0]: kw["extra_body"] = {"thinking": {"type": "between_tools"}}
                m = ac.messages.create(**kw); break
            except anthropic.BadRequestError as e:
                if use_think_off[0] and "thinking" in str(e).lower(): use_think_off[0] = False; print("thinking=between_tools rejected, using default:", str(e)[:200], flush=True); continue
                raise
            except Exception as e:
                if at == 9: raise
                time.sleep(min(2 ** at, 30))
        return "".join(b.text for b in m.content if b.type == "text").strip(), m.usage.input_tokens, m.usage.output_tokens
else:
    def answer_call(prompt):
        r = chat_openai(a.answer_model, prompt)
        return (r.choices[0].message.content or "").strip(), r.usage.prompt_tokens, r.usage.completion_tokens

RULES = ("Rules: use only the excerpts. Combine and cross-reference every relevant excerpt. If excerpts contradict, prefer the most recent one "
         "and mention the change. Keep exact names, numbers and dates. Follow any standing instruction or preference the user gave in the excerpts. "
         "For ordering questions list the items in chronological order, one per line. For time questions use the dates shown.")
def ask(q, idx, nf):
    c = convs[q["conv"]]
    ctx = "\n".join(f"[{c['turns'][i]['date']}] (msg {c['turns'][i]['id']}) {c['turns'][i]['role']}: {c['turns'][i]['text'][:a.max_turn_chars]}" for i in idx)
    tail = (" If the excerpts do not contain the answer, reply exactly: NOT FOUND." if nf else
            " If the conversation has no information that answers the question, say there is no information about it in the conversation.")
    prompt = f"Below are excerpts from a long chat history between a user and an AI assistant. Answer the user's question from them.{tail}\n{RULES}\n\nChat excerpts:\n{ctx}\n\nQuestion: {q['question']}\nAnswer:"
    return answer_call(prompt)

def parse_judge(txt):
    t = txt.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try: return float(json.loads(t)["score"])
    except Exception: pass
    m = re.search(r'"score"\s*:\s*([0-9.]+)', t)
    return float(m.group(1)) if m else 0.0

def judge_item(item, resp):
    p = JUDGE_PROMPT.replace("<rubric_item>", item).replace("<llm_response>", resp)
    r = chat_openai(a.judge_model, p, temperature=0)
    add("jin", r.usage.prompt_tokens); add("jout", r.usage.completion_tokens)
    s = parse_judge(r.choices[0].message.content or "")
    return s if s in (0.0, 0.5, 1.0) else min((0.0, 0.5, 1.0), key=lambda v: abs(v - s))

def equiv(x, y):
    p = ("You are a binary classifier.\nIf the TWO snippets describe the SAME event/fact, reply **YES**\nOtherwise reply **NO**. No extra words.\nDO NOT provide any exaplanation.\n\n"
         f"First snippet: {x} \n\nSecond snippet: {y}")
    r = chat_openai(a.judge_model, p, temperature=0, max_tokens=5)
    add("jin", r.usage.prompt_tokens); add("jout", r.usage.completion_tokens)
    return "yes" in (r.choices[0].message.content or "").lower()

def event_ordering(rubric, resp):
    from scipy.stats import kendalltau
    system = [l for l in resp.split("\n") if len(l.strip()) > 2][:30]
    used = set(); sys_c = []
    for s in system:
        mi = None
        for i, r_ in enumerate(rubric):
            if i in used: continue
            if equiv(r_, s): mi = i; break
        if mi is not None: sys_c.append(rubric[mi]); used.add(mi)
        else: sys_c.append(s)
    tp = len(set(rubric) & set(sys_c)); fp = len([x for x in sys_c if x not in rubric]); fn = len([x for x in rubric if x not in sys_c])
    pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
    union = list(dict.fromkeys(rubric + sys_c)); tie = len(union) + 1
    def rank(seq):
        r_ = {it: i + 1 for i, it in enumerate(seq)}
        return [r_.get(u, tie) for u in union]
    try: tau = float(kendalltau(rank(rubric), rank(sys_c), variant="b")[0])
    except Exception: tau = float("nan")
    tn = 0.0 if tau != tau else (tau + 1) / 2
    return dict(eo_f1=f1, tau_norm=tn, eo_final=tn * f1)

os.makedirs("results", exist_ok=True)
out = f"results/beam-{datetime.now().strftime('%Y%m%d-%H%M%S')}{('-' + a.tag) if a.tag else ''}.jsonl"

def run(q):
    if cost() > a.max_usd: return None
    c = convs[q["conv"]]; passes = 0; ain = aout = 0
    for j, k in enumerate(ks):
        passes += 1
        idx = retrieve(q, k)
        nf = len(ks) > 1 and j < len(ks) - 1
        ans, i_, o_ = ask(q, idx, nf)
        ain += i_; aout += o_
        if not (nf and "NOT FOUND" in ans.upper()): break
    add("ain", ain); add("aout", aout)
    ids = [c["turns"][i]["id"] for i in idx]
    scores = [judge_item(it, ans) for it in q["rubric"]]
    score = sum(scores) / len(scores)
    rec = dict(qid=q["qid"], conv=q["conv"], ability=q["ability"], question=q["question"], rubric=q["rubric"], answer=ans, retrieved=ids,
               gold_ids=q["gold_ids"], turns_read=len(idx), passes=passes, item_scores=scores, score=score, passed=score >= 0.5,
               tokens=dict(ain=ain, aout=aout))
    g = q["gold_ids"]
    if g:
        rec["gold_recall"] = sum(x in set(ids) for x in g) / len(g)
        pairs = {i // 2 for i in ids}; rec["gold_recall_pair"] = sum((x // 2) in pairs for x in g) / len(g)
    if q["ability"] == "event_ordering" and not a.no_eo_align:
        rec.update(event_ordering(q["rubric"], ans))
    return rec

res = list(prev); stopped = False
for r in prev: add("ain", r["tokens"]["ain"]); add("aout", r["tokens"]["aout"])
with ThreadPoolExecutor(a.workers) as ex, open(out, "w") as f:
    for rec in ex.map(run, sel):
        if rec is None: stopped = True; continue
        res.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()

n = len(res)
print(f"results: {out}  n={n}" + ("  STOPPED ON COST CAP" if stopped else ""))
print(f"{'ability':26s} {'n':>3s} {'pass':>6s} {'avg':>6s}")
for ab in ABILITIES:
    rs = [r for r in res if r["ability"] == ab]
    if rs: print(f"{ab:26s} {len(rs):3d} {sum(r['passed'] for r in rs)/len(rs):6.3f} {np.mean([r['score'] for r in rs]):6.3f}")
print(f"{'OVERALL':26s} {n:3d} {sum(r['passed'] for r in res)/n:6.3f} {np.mean([r['score'] for r in res]):6.3f}   (pass = rubric mean >= 0.5, Mem0; avg = mean rubric score)")
eo = [r for r in res if "tau_norm" in r]
if eo:
    paper = [r["tau_norm"] if r["ability"] == "event_ordering" else r["score"] for r in res if r["ability"] != "event_ordering" or "tau_norm" in r]
    print(f"event_ordering BEAM-paper metric tau_norm: {np.mean([r['tau_norm'] for r in eo]):.3f} (n={len(eo)}); overall avg with EO=tau_norm: {np.mean(paper):.3f}")
gr = [r["gold_recall"] for r in res if "gold_recall" in r]
if gr: print(f"gold-message recall: {np.mean(gr):.3f}  pair-level (user/assistant partner counts): {np.mean([r['gold_recall_pair'] for r in res if 'gold_recall_pair' in r]):.3f}  (n={len(gr)})")
if len(ks) > 1: print(f"adaptive: {sum(r['passes']==2 for r in res)} of {n} widened; mean turns read {np.mean([r['turns_read'] for r in res]):.1f}")
print(f"answer tokens in/out per question: {use['ain']/n:,.0f}/{use['aout']/n:,.0f}  judge in/out per question: {use['jin']/n:,.0f}/{use['jout']/n:,.0f}")
print(f"reader cost ${use['ain']*pin/1e6+use['aout']*pout/1e6:.3f}  judge cost ${use['jin']*jin/1e6+use['jout']*jout/1e6:.3f}  total ${cost():.3f}  per question ${cost()/n:.4f}")
print(f"projection 700 q: ${cost()/n*700:.2f}   200 q: ${cost()/n*200:.2f} (same K; 10M conversations have 10x more turns but K is fixed)")
