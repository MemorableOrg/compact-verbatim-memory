import argparse, json, os, re, random, time, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from rank_bm25 import BM25Okapi
from openai import OpenAI
from judge_prompt import get_judge_prompt, JUDGE_SYSTEM_PROMPT, preprocess_answer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--answer-model", default="gpt-6-sol")
ap.add_argument("--judge-model", default="gpt-4o-mini")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--retriever", choices=["baseline", "5a", "5r", "snap"], default="baseline")
ap.add_argument("--rank-file", default="snap_rank.json", help="snap: precomputed ranking {sample: {qi: [dia_id,...]}} from snap_rank.py")
ap.add_argument("--adaptive", default="", help="K1,K2: read K1 turns first; if the reader says NOT FOUND, read K2 turns and answer again")
ap.add_argument("--skip-file", default="", help="results jsonl of an earlier partial run: its (sample, qi) questions are skipped so the two files together cover the set")
ap.add_argument("--prompt", choices=["ours", "mem0v2"], default="ours", help="mem0v2: Mem0's current 7-step answer prompt with a reference date and ANSWER: extraction")
ap.add_argument("--judge-prompt", choices=["mem0v1", "mem0v2"], default="mem0v1", help="mem0v2: Mem0's current unified judge prompt")
ap.add_argument("--convs", choices=["all", "dev", "test"], default="all", help="dev = conv-26, conv-30, conv-41; test = the other seven")
ap.add_argument("--answer-provider", choices=["openai", "anthropic"], default="openai")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--full-context", action="store_true")
ap.add_argument("--token-cap", type=int, default=2_000_000)
ap.add_argument("--format", choices=["flat", "grouped"], default="flat", help="mem0v2 prompt only. grouped: one [date] line per run of same-session turns, then speaker: text lines (about 16 percent fewer tokens at 150 turns)")
a = ap.parse_args()

MONTHS = "january february march april may june july august september october november december".split()
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
STOP = set("the a an of to in on at is was were did do does what when who how which and or for with his her their he she they it that this".split())

def load(path="data/locomo10.json"):
    convs = {}
    for c in json.load(open(path)):
        cv = c["conversation"]; turns = []
        n = 1
        while f"session_{n}" in cv:
            date = cv.get(f"session_{n}_date_time", "")
            for t in cv[f"session_{n}"]:
                turns.append(dict(id=t["dia_id"], speaker=t["speaker"], session=n, date=date,
                                  text=t["text"] + (f" [shared image: {t['blip_caption']}]" if t.get("blip_caption") else "")))
            n += 1
        # index text carries speaker and date so name/date questions can hit
        docs = [tok(f"{t['speaker']} {t['date']} {t['text']}") for t in turns]
        convs[c["sample_id"]] = dict(turns=turns, bm25=BM25Okapi(docs), qa=c["qa"])
    return convs

def retrieve(conv, q, k):
    qt = [w for w in tok(q) if w not in STOP] or tok(q)
    scores = list(conv["bm25"].get_scores(qt))
    ql = q.lower()
    dates = set()
    for m in re.finditer(r"(\d{1,2})\s+(" + "|".join(MONTHS) + r")(?:,?\s+(\d{4}))?", ql):
        dates.add((m.group(1), m.group(2)))
    for m in re.finditer(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})", ql):
        dates.add((m.group(2), m.group(1)))
    months = {m for m in MONTHS if re.search(r"\b" + m + r"\b", ql)}
    mx = max(scores) or 1.0
    for i, t in enumerate(conv["turns"]):
        d = t["date"].lower()
        if any(re.search(r"\b" + dd + r"\s+" + mm + r"\b|\b" + mm + r"\s+" + dd + r"\b", d) for dd, mm in dates):
            scores[i] += 0.5 * mx
        elif months and any(m in d for m in months):
            scores[i] += 0.15 * mx
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return sorted(idx)  # chronological (turns are in order)

if a.retriever in ("5a", "5r"):
    import threading, numpy as np, retrieval_eval as R
    from sentence_transformers import SentenceTransformer
    _rc = R.load(); _z = np.load("emb_cache_bge-small-en-v1.5.npz")
    _emb = {k: _z[k] for k in _z.files if k != "_time"}
    _m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); _m.max_seq_length = 256
    _lock = threading.Lock()
    if a.retriever == "5r":
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=320)
    def retrieve5a(sid, q, k):
        with _lock: qv = _m.encode([q], normalize_embeddings=True)[0]
        c = _rc[sid]; n = len(c["turns"])
        sc = R.rrf([R.top(R.bm25_scores(c, q, True), 200), R.top(_emb[sid] @ qv, 200)], n)
        if a.retriever == "5r":
            cand = [int(i) for i in R.top(sc, 100)]
            pairs = [(q, f"{c['turns'][j]['speaker']} ({c['turns'][j]['date']}): {c['turns'][j]['text']}") for j in cand]
            with _lock: s = _ce.predict(pairs, batch_size=32, show_progress_bar=False)
            order_ce = [cand[j] for j in np.argsort(-np.asarray(s), kind="stable")]
            return sorted(int(i) for i in R.top(R.rrf([order_ce, cand], n), k))
        return sorted(int(i) for i in R.top(sc, k))

import oa
client = oa.client()
use = {"in": 0, "out": 0, "jin": 0, "jout": 0}

def chat(model, msgs, **kw):
    for attempt in range(14):
        try:
            return client.chat.completions.create(model=oa.dep(model), messages=msgs, **kw)
        except Exception as e:
            if attempt == 13 or "model" in str(e).lower() and "not" in str(e).lower() and "exist" in str(e).lower():
                raise
            time.sleep(min(2 ** attempt, 30))

convs = load()
qs = []
for sid, c in convs.items():
    for j, q in enumerate(c["qa"]):
        if q["category"] in (1, 2, 3, 4):
            qs.append((sid, j, q))
DEV = {"conv-26", "conv-30", "conv-41"}
if a.convs != "all":
    qs = [x for x in qs if (x[0] in DEV) == (a.convs == "dev")]
if a.skip_file:
    _done = {(json.loads(l)["sample"], json.loads(l)["qi"]) for l in open(a.skip_file) if l.strip()}
    qs = [x for x in qs if (x[0], x[1]) not in _done]
    print(f"skipping {len(_done)} questions already in {a.skip_file}; {len(qs)} left")
random.seed(a.seed)
groups = defaultdict(list)
for x in qs: groups[(x[0], x[2]["category"])].append(x)
for g in groups.values(): random.shuffle(g)
sel = []
keys = sorted(groups)
while len(sel) < a.n and any(groups[k] for k in keys):
    for k in keys:
        if groups[k] and len(sel) < a.n: sel.append(groups[k].pop())
print(f"questions: {len(qs)} eligible, running {len(sel)}")

os.makedirs("results", exist_ok=True)
out = f"results/{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"

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

_snap = json.load(open(a.rank_file)) if a.retriever == "snap" else {}
_adaptive = [int(v) for v in a.adaptive.split(",")] if a.adaptive else []

def pick(sid, j, q, k):
    c = convs[sid]
    if a.full_context: return list(range(len(c["turns"])))
    if a.retriever == "snap":
        pos = {t["id"]: i for i, t in enumerate(c["turns"])}
        return sorted(pos[d] for d in _snap[sid][str(j)][:k] if d in pos)
    return retrieve5a(sid, q["question"], k) if a.retriever in ("5a", "5r") else retrieve(c, q["question"], k)

import mem0v2_prompts as M2
from datetime import datetime as _dt
def human_date(s):
    m = re.search(r"(\d{1,2}) (\w+), (\d{4})", s or "")
    if not m: return s or "unknown date"
    try: return _dt.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y").strftime("%A, %B %d, %Y")
    except ValueError: return s
def ask(c, q, idx, allow_not_found):
    if a.prompt == "mem0v2":
        if a.format == "grouped":
            lines = ["The following memories are presented in chronological order (oldest to newest), grouped by the date they were said.", ""]
            last = None
            for i in idx:
                d = c["turns"][i]["date"]
                if d != last:
                    lines.append(f"[{human_date(d)}]"); last = d
                lines.append(f"{c['turns'][i]['speaker']}: {c['turns'][i]['text']}")
        else:
            lines = ["The following memories are presented in chronological order (oldest to newest).", ""]
            lines += [f"({human_date(c['turns'][i]['date'])}) {c['turns'][i]['speaker']}: {c['turns'][i]['text']}" for i in idx]
        rule = "\nIf no memory contains information relevant to the question, output exactly: ANSWER: NOT FOUND" if allow_not_found else ""
        ref = human_date(c["turns"][-1]["date"])
        prompt = M2.ANSWER_GENERATION_PROMPT.format(memories="\n".join(lines), question=q["question"], reference_date=ref) + rule
        r = answer_call(a.answer_model, prompt)
        use["in"] += r.usage.prompt_tokens; use["out"] += r.usage.completion_tokens
        ans = (r.choices[0].message.content or "").strip()
        if "ANSWER:" in ans: ans = ans.rsplit("ANSWER:", 1)[-1].strip()
        return ans, r
    ctx = "\n".join(f"[{c['turns'][i]['date']}] (session {c['turns'][i]['session']}, {c['turns'][i]['id']}) {c['turns'][i]['speaker']}: {c['turns'][i]['text']}" for i in idx)
    rule = " If the excerpts do not contain the answer, reply exactly: NOT FOUND." if allow_not_found else ""
    prompt = f"Answer the question using only the conversation excerpts below. Be brief; give the answer itself.{rule}\n\nConversation excerpts:\n{ctx}\n\nQuestion: {q['question']}\nAnswer:"
    r = answer_call(a.answer_model, prompt)
    use["in"] += r.usage.prompt_tokens; use["out"] += r.usage.completion_tokens
    return (r.choices[0].message.content or "").strip(), r

def run(x):
    sid, j, q = x
    if use["in"] + use["out"] + use["jin"] + use["jout"] > a.token_cap: return None
    c = convs[sid]
    passes = 1
    if _adaptive:
        idx = pick(sid, j, q, _adaptive[0])
        ans, r = ask(c, q, idx, True)
        if "NOT FOUND" in ans.upper():
            passes = 2
            idx = pick(sid, j, q, _adaptive[1])
            ans, r2 = ask(c, q, idx, False)
            from types import SimpleNamespace as _NS2
            r = _NS2(usage=_NS2(prompt_tokens=r.usage.prompt_tokens + r2.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens + r2.usage.completion_tokens))
    else:
        idx = pick(sid, j, q, a.k)
        ans, r = ask(c, q, idx, False)
    ids = [c["turns"][i]["id"] for i in idx]
    gold = preprocess_answer(q["category"], str(q["answer"]))
    jkw = {"response_format": {"type": "json_object"}}
    if not a.judge_model.startswith("gpt-5"):
        jkw["temperature"] = 0  # gpt-5 models accept only the default temperature
    jp = M2.get_judge_prompt(q["category"], q["question"], M2.preprocess_answer(q["category"], str(q["answer"])), ans) if a.judge_prompt == "mem0v2" else get_judge_prompt(q["category"], q["question"], gold, ans)
    jr = chat(a.judge_model, [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": jp}], **jkw)
    use["jin"] += jr.usage.prompt_tokens; use["jout"] += jr.usage.completion_tokens
    try: label = json.loads(jr.choices[0].message.content)["label"].strip().upper()
    except Exception: label = "WRONG"
    ev = q.get("evidence", [])
    evn = [e for s in ev for e in re.findall(r"D\d+:\d+", s)]
    rec = dict(sample=sid, qi=j, category=q["category"], question=q["question"], gold=q["answer"],
               retrieved=ids, evidence=evn, recall=(all(e in ids for e in evn) if evn else None),
               answer=ans, correct=label == "CORRECT", passes=passes, turns=len(idx),
               tokens=dict(ain=r.usage.prompt_tokens, aout=r.usage.completion_tokens, jin=jr.usage.prompt_tokens, jout=jr.usage.completion_tokens))
    return rec

res = []
with ThreadPoolExecutor(a.workers) as ex, open(out, "w") as f:
    for rec in ex.map(run, sel):
        if rec: res.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()

n = len(res)
print(f"results: {out}  n={n}")
print(f"accuracy overall: {sum(r['correct'] for r in res)/n:.3f}")
for cat, nm in [(1, "multi-hop"), (2, "temporal"), (3, "open-domain"), (4, "single-hop")]:
    rs = [r for r in res if r["category"] == cat]
    if rs: print(f"  cat{cat} {nm}: {sum(r['correct'] for r in rs)/len(rs):.3f} (n={len(rs)})")
if _adaptive: print(f"adaptive: {sum(r['passes'] == 2 for r in res)} of {n} widened; mean turns read {sum(r['turns'] for r in res)/n:.1f}")
rr = [r["recall"] for r in res if r["recall"] is not None]
print(f"evidence recall@{a.k} (all gold turns retrieved): {sum(rr)/len(rr):.3f} (n={len(rr)})")
cr = [sum(e in r["retrieved"] for e in r["evidence"]) / len(r["evidence"]) for r in res if r["evidence"]]
print(f"evidence recall@{a.k} (fraction of gold turns): {sum(cr)/len(cr):.3f}")
print(f"answer tokens in/out: {use['in']}/{use['out']}  judge tokens in/out: {use['jin']}/{use['jout']}  total: {sum(use.values())}")
rate_in, rate_out = (1.25e-6, 10e-6) if a.answer_model.startswith("gpt-5") else (0.15e-6, 0.6e-6) if "4o-mini" in a.answer_model else (2e-6, 10e-6)
cost = use["in"] * rate_in + use["out"] * rate_out
print(f"answer-model cost est: ${cost:.3f}  per question: ${cost/n:.5f}  full 1540 est: ${cost/n*1540:.2f}")
print(f"full 1540 tokens est: answer in {use['in']/n*1540:,.0f}, out {use['out']/n*1540:,.0f}")
