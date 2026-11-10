#!/usr/bin/env python3
"""Open-source Mem0 (mem0ai 2.0.0) on BEAM 1M, answering the same questions our runner (beam_run.py) selects.

Ingestion, time anchors, search depth (top_k 200) and the answer prompt are copied from Mem0's own harness
(mem0mb/benchmarks/beam/run.py + prompts.py). The hosted MemoryClient is replaced by the local mem0.Memory
(LLM extraction at ingest, local qdrant on disk, local bge-small embedder).

    OPENAI_API_KEY=... OPENAI_BASE_URL=... .venv/bin/python mem0_oss_beam.py --convs 0-9 --n 200 --seed 0 --max-usd 40 --tag c0-9
    .venv/bin/python mem0_oss_beam.py --convs 0 --n 20 --fake-llm --max-chunks 50 --tag fake     (free, offline LLM)
    --estimate-only prints chunk and cost estimates and exits. --resume continues a stopped run (same flags, same tag).

Output: results/beam-mem0oss-<tag>.json (Mem0 platform results format: evaluations[].cutoff_results.top_200.generated_answer, no judge score)
and results/beam-mem0oss-<tag>.cost.json (per-conversation LLM call counts, tokens, wall time, dollars). Judge it with
claude_judge.py beam mem0 <file> or beam_rejudge_mem0.py --src <file>.
"""
import os
os.environ["MEM0_TELEMETRY"] = "false"
import argparse, json, random, re, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ABILITIES = ["abstention", "contradiction_resolution", "event_ordering", "information_extraction", "instruction_following",
             "knowledge_update", "multi_session_reasoning", "preference_following", "summarization", "temporal_reasoning"]
PRICES = {"gpt-5-mini": (0.25, 2.00), "gpt-5": (1.25, 10.0), "gpt-5-nano": (0.05, 0.40), "gpt-4.1-mini": (0.4, 1.6),
          "gpt-4o-mini": (0.15, 0.60), "gpt-4.1": (2.0, 8.0)}
CHUNK_SIZE = 2
EMB_MODEL = "BAAI/bge-small-en-v1.5"
EMB_DIMS = 384
MARK = re.compile(r"\s*->->\s*[\d,]+\s*$")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/Users/nikhilkrishnaswamy/Documents/fm-data/beam/beam_1M.jsonl")
ap.add_argument("--convs", default="", help="conversation indices, e.g. 0-9,12 (default all 35)")
ap.add_argument("--n", type=int, default=700, help="questions, balanced over abilities (same selection as beam_run.py)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--llm-model", default="gpt-5-mini", help="model mem0 uses to extract memories")
ap.add_argument("--answer-model", default="gpt-5-mini")
ap.add_argument("--top-k", type=int, default=200)
ap.add_argument("--threshold", type=float, default=0.1, help="mem0 OSS search default")
ap.add_argument("--workers", type=int, default=4, help="conversations in parallel")
ap.add_argument("--max-usd", type=float, default=10.0, help="stop cleanly when the running estimate (ingest + answer) reaches this")
ap.add_argument("--max-chunks", type=int, default=0, help="ingest only the first N chunks of each conversation (testing)")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--fake-llm", action="store_true", help="offline stub LLM for extraction and answers; no network")
ap.add_argument("--estimate-only", action="store_true")
ap.add_argument("--keep-markers", action="store_true", default=True, help="keep the '->-> ids' suffix on turn text, as Mem0's harness does")
ap.add_argument("--strip-markers", dest="keep_markers", action="store_false")
ap.add_argument("--store-dir", default="")
ap.add_argument("--tag", default="run")
a = ap.parse_args()

if os.environ.pop("OPENROUTER_API_KEY", None):
    print("note: OPENROUTER_API_KEY unset for this process (mem0 would route to OpenRouter and ignore OPENAI_BASE_URL)", flush=True)
if not a.fake_llm and not os.environ.get("OPENAI_API_KEY") and not a.estimate_only:
    sys.exit("OPENAI_API_KEY not set")
STORE = a.store_dir or os.path.join(HERE, "mem0_oss_store_fake" if a.fake_llm else "mem0_oss_store")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
OUT = os.path.join(HERE, "results", f"beam-mem0oss-{a.tag}.json")
COST_OUT = os.path.join(HERE, "results", f"beam-mem0oss-{a.tag}.cost.json")


def price(model): return PRICES.get(model, (2.0, 10.0))
def usd(model, pin, pout):
    i, o = price(model); return pin * i / 1e6 + pout * o / 1e6


# --------------------------------------------------------------------------- dataset + selection (same as beam_run.py) ----
def parse_idx(s, n):
    if not s: return list(range(n))
    out = []
    for p in s.split(","):
        if "-" in p: lo, hi = p.split("-"); out += list(range(int(lo), int(hi) + 1))
        else: out.append(int(p))
    return out


def parse_pq(s):
    import ast
    if isinstance(s, dict): return s
    try: return ast.literal_eval(s)
    except Exception: return json.loads(s)


def flat_ids(x):
    out = []
    if isinstance(x, bool) or x is None: return out
    if isinstance(x, int): out.append(x)
    elif isinstance(x, (list, tuple)):
        for i in x: out += flat_ids(i)
    elif isinstance(x, dict):
        for i in x.values(): out += flat_ids(i)
    return out


want = set(parse_idx(a.convs, 35)) if a.convs else None
convs = {}
for ci, line in enumerate(open(a.data)):
    if want is not None and ci not in want: continue
    r = json.loads(line)
    pq = parse_pq(r["probing_questions"])
    qs = []
    mem0_qi = 0
    for ab in ABILITIES:
        for qi, q in enumerate(pq.get(ab, [])):
            rub = q["rubric"]
            rub = [str(x) for x in rub] if isinstance(rub, list) else [str(rub)]
            qs.append(dict(conv=ci, qid=f"{ci}_{ab}_{qi}", mem0_qid=f"1M_{ci}_q{mem0_qi + qi}_{ab}", mem0_qi=mem0_qi + qi, ab_qi=qi, ability=ab,
                           question=q["question"], rubric=rub, difficulty=q.get("difficulty", "unknown"),
                           source_chat_ids=q.get("source_chat_ids", [])))
        mem0_qi += len(pq.get(ab, []))
    convs[ci] = dict(cid=r["conversation_id"], chat=r["chat"], qs=qs)

allq = [q for c in convs.values() for q in c["qs"]]
random.seed(a.seed)
groups = defaultdict(list)
for q in allq: groups[q["ability"]].append(q)
for g in groups.values(): random.shuffle(g)
sel = []; keys = sorted(groups)
while len(sel) < a.n and any(groups[k] for k in keys):
    for k in keys:
        if groups[k] and len(sel) < a.n: sel.append(groups[k].pop())
need = sorted({q["conv"] for q in sel})
sel_by_conv = defaultdict(list)
for q in sel: sel_by_conv[q["conv"]].append(q)


# --------------------------------------------------------------------------- chunking + time anchors (copied from Mem0 harness) ----
def batch_to_chunks(turns, chunk_size=CHUNK_SIZE):
    messages = []
    for turn in turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if not content: continue
        if not a.keep_markers: content = MARK.sub("", content)
        if role not in ("user", "assistant"):
            role = "user" if role.lower() in ("human", "user") else "assistant"
        messages.append({"role": role, "content": content})
    chunks = []
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i: i + chunk_size]
        if chunk: chunks.append(chunk)
    return chunks


def get_time_anchor_epoch(turns):
    for turn in turns:
        anchor = turn.get("time_anchor")
        if anchor:
            try:
                from dateutil.parser import parse as dateparse
                return int(dateparse(anchor.replace("-", " ")).timestamp())
            except Exception:
                pass
    return None


def plan(ci):
    """[(chunk_key, messages, epoch)] in ingestion order."""
    out = []
    for bi, batch in enumerate(convs[ci]["chat"]):
        epoch = get_time_anchor_epoch(batch)
        for k, msgs in enumerate(batch_to_chunks(batch)):
            out.append((f"batch_{bi}_c{k}", msgs, epoch))
    return out


total_chunks = {ci: len(plan(ci)) for ci in need}
if a.max_chunks: total_chunks = {ci: min(n, a.max_chunks) for ci, n in total_chunks.items()}
n_chunks = sum(total_chunks.values())
print(f"conversations {need}; questions {len(sel)}; ingestion chunks (= extraction LLM calls) {n_chunks}: "
      + ", ".join(f"{ci}:{total_chunks[ci]}" for ci in need), flush=True)
EST_IN, EST_OUT = 10500, 800  # rough per extraction call: 8.4k-token system prompt + up to 10 existing memories + last 10 messages; reasoning tokens in out
ANS_IN, ANS_OUT = 12000, 600
ein, eout = price(a.llm_model); ain_, aout_ = price(a.answer_model)
est_ing = n_chunks * (EST_IN * ein + EST_OUT * eout) / 1e6
est_ans = len(sel) * (ANS_IN * ain_ + ANS_OUT * aout_) / 1e6
print(f"rough estimate (not measured): ingestion ${est_ing:.2f} ({n_chunks} calls x {EST_IN} in / {EST_OUT} out), "
      f"answers ${est_ans:.2f} ({len(sel)} x {ANS_IN} in / {ANS_OUT} out), cap ${a.max_usd}", flush=True)
if a.estimate_only: sys.exit(0)


# --------------------------------------------------------------------------- usage accounting ----
lock = threading.Lock()
stop = threading.Event()
USAGE = {}  # ci -> dict


def blank():
    return dict(ingest=dict(calls=0, failed_calls=0, prompt_tokens=0, completion_tokens=0, wall_s=0.0, usd=0.0, chunks_done=0, memories_added=0),
                answer=dict(calls=0, failed_calls=0, prompt_tokens=0, completion_tokens=0, wall_s=0.0, usd=0.0), wall_s_total=0.0)


def running_usd():
    with lock: return sum(u["ingest"]["usd"] + u["answer"]["usd"] for u in USAGE.values())


def record(ci, stage, model, pin, pout, wall, ok=True):
    with lock:
        s = USAGE[ci][stage]
        s["calls"] += 1; s["wall_s"] += wall
        if ok:
            s["prompt_tokens"] += pin; s["completion_tokens"] += pout; s["usd"] += usd(model, pin, pout)
        else:
            s["failed_calls"] += 1


def write_cost():
    with lock:
        tot = dict(ingest=dict(calls=0, failed_calls=0, prompt_tokens=0, completion_tokens=0, wall_s=0.0, usd=0.0, chunks_done=0, memories_added=0),
                   answer=dict(calls=0, failed_calls=0, prompt_tokens=0, completion_tokens=0, wall_s=0.0, usd=0.0))
        for u in USAGE.values():
            for st in tot:
                for k in tot[st]: tot[st][k] += u[st][k]
        doc = dict(system="mem0-oss 2.0.0", llm_model=a.llm_model, answer_model=a.answer_model, fake_llm=a.fake_llm,
                   prices_usd_per_m_tokens={a.llm_model: price(a.llm_model), a.answer_model: price(a.answer_model)},
                   total=tot, total_usd=tot["ingest"]["usd"] + tot["answer"]["usd"], per_conversation={str(k): v for k, v in sorted(USAGE.items())})
        tmp = COST_OUT + ".tmp"
        json.dump(doc, open(tmp, "w"), indent=1); os.replace(tmp, COST_OUT)


# --------------------------------------------------------------------------- mem0 setup ----
import mem0.memory.main as mm
from mem0 import Memory
from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.huggingface import HuggingFaceEmbedding

tls = threading.local()
_orig_prompt = mm.generate_additive_extraction_prompt


def _prompt_with_obs_date(*args, **kw):
    # OSS Memory.add has no timestamp argument; the hosted API turns Mem0's `timestamp` into the observation date. Do the same here.
    obs = getattr(tls, "obs_date", None)
    if obs and "timestamp" not in kw: kw["timestamp"] = obs
    return _orig_prompt(*args, **kw)


mm.generate_additive_extraction_prompt = _prompt_with_obs_date


class SharedEmbedder:
    def __init__(self):
        self.inner = HuggingFaceEmbedding(BaseEmbedderConfig(model=EMB_MODEL, embedding_dims=EMB_DIMS))
        self.config = self.inner.config
        self.lk = threading.Lock()

    def embed(self, text, memory_action=None):
        with self.lk: return self.inner.embed(text, memory_action)

    def embed_batch(self, texts, memory_action="add"):
        with self.lk: return [self.inner.embed(t, memory_action) for t in texts]


class FakeLLM:
    """Stub extraction LLM: one fact per call, built from the last user message in the prompt. Usage is chars/4."""
    def __init__(self, ci):
        self.ci = ci

    def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kw):
        prompt = "\n".join(m["content"] for m in messages)
        m = re.search(r"## New Messages\n(.*?)\n\n## Observation Date", prompt, re.S)
        try:
            msgs = json.loads(m.group(1))
            u = [x["content"] for x in msgs if x.get("role") == "user"]
            body = u[-1] if u else msgs[-1]["content"]
        except Exception:
            body = prompt[-300:]
        text = f"[fake] {body[:160].strip()}"
        tls.usage = (len(prompt) // 4, len(text) // 4 + 20)
        return json.dumps({"memory": [{"id": "0", "text": text, "attributed_to": "user", "linked_memory_ids": []}]})


def make_memory(ci):
    d = os.path.join(STORE, f"conv_{ci}")
    os.makedirs(d, exist_ok=True)

    def cb(llm, response, params):
        u = getattr(response, "usage", None)
        tls.usage = (getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0) if u else (0, 0)

    cfg = {
        "llm": {"provider": "openai", "config": {"model": a.llm_model, "api_key": os.environ.get("OPENAI_API_KEY") or "sk-fake",
                                                  "openai_base_url": os.environ.get("OPENAI_BASE_URL"), "response_callback": cb}},
        "embedder": {"provider": "huggingface", "config": {"model": EMB_MODEL, "embedding_dims": EMB_DIMS}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": f"beam_1M_{ci}", "path": os.path.join(d, "qdrant"),
                                                          "embedding_model_dims": EMB_DIMS, "on_disk": True}},
        "history_db_path": os.path.join(d, "history.db"),
    }
    m = Memory.from_config(cfg)
    m.embedding_model = EMB
    if a.fake_llm:
        m.llm = FakeLLM(ci)
    inner = m.llm.generate_response

    def counted(*args, **kw):
        t0 = time.time(); tls.usage = (0, 0)
        for attempt in range(6):
            try:
                out = inner(*args, **kw)
                record(ci, "ingest", a.llm_model, *tls.usage, time.time() - t0)
                return out
            except Exception as e:
                if attempt == 5:
                    record(ci, "ingest", a.llm_model, 0, 0, time.time() - t0, ok=False)
                    raise
                time.sleep(min(2 ** attempt, 30))

    m.llm.generate_response = counted
    return m


# --------------------------------------------------------------------------- ingestion ----
def ckpt_path(ci): return os.path.join(STORE, f"conv_{ci}", "ckpt.json")


def load_ckpt(ci):
    p = ckpt_path(ci)
    if a.resume and os.path.exists(p): return json.load(open(p))
    return dict(done=[], usage=blank(), complete=False, user_id=f"beam_1M_{ci}_{a.tag}")


def save_ckpt(ci, ck):
    with lock: ck["usage"] = USAGE[ci]
    tmp = ckpt_path(ci) + ".tmp"
    with lock: json.dump(ck, open(tmp, "w"))
    os.replace(tmp, ckpt_path(ci))


def ingest(ci, m, ck):
    done = set(ck["done"])
    t_start = time.time()
    steps = plan(ci)[: total_chunks[ci]]
    for n, (key, msgs, epoch) in enumerate(steps):
        if key in done: continue
        if stop.is_set(): return False
        if running_usd() >= a.max_usd:
            stop.set(); print(f"[{ci}] cost cap ${a.max_usd} reached, stopping", flush=True); return False
        iso = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat() if epoch is not None else None
        tls.obs_date = iso[:10] if iso else None
        meta = {"created_at": iso} if iso else None
        try:
            res = m.add(msgs, user_id=ck["user_id"], metadata=meta)
            added = len(res.get("results", [])) if isinstance(res, dict) else 0
        except Exception as e:
            print(f"[{ci}] add failed at {key}: {str(e)[:150]}", flush=True); added = 0
        with lock: USAGE[ci]["ingest"]["chunks_done"] += 1; USAGE[ci]["ingest"]["memories_added"] += added
        done.add(key); ck["done"] = sorted(done)
        save_ckpt(ci, ck)
        if (n + 1) % 25 == 0:
            u = USAGE[ci]["ingest"]
            print(f"[{ci}] {n + 1}/{len(steps)} chunks, {u['memories_added']} memories, {u['prompt_tokens']:,}/{u['completion_tokens']:,} tokens, "
                  f"conv ${u['usd']:.3f}, total ${running_usd():.2f}", flush=True)
            write_cost()
    with lock: USAGE[ci]["wall_s_total"] += time.time() - t_start
    ck["complete"] = True; save_ckpt(ci, ck)
    return True


# --------------------------------------------------------------------------- answering (copied from Mem0 harness) ----
def get_beam_answer_generation_prompt(question, memories, top_k=None):
    if top_k is not None: memories = memories[:top_k]
    if not memories:
        memories_text = "(No memories available)"
    else:
        def sort_key(m): return m.get("created_at", "") or "" if isinstance(m, dict) else ""
        sorted_mems = sorted(memories, key=sort_key)
        lines = []
        for i, mem in enumerate(sorted_mems, 1):
            text = mem.get("memory", mem) if isinstance(mem, dict) else str(mem)
            created_at = mem.get("created_at", "") if isinstance(mem, dict) else ""
            if created_at:
                date_str = created_at[:10] if len(created_at) >= 10 else created_at
                lines.append(f"{i}. [{date_str}] {text}")
            else:
                lines.append(f"{i}. {text}")
        memories_text = "\n".join(lines)
    return f"""You are an AI assistant with access to stored memories from prior conversations with a user.
Use these memories to answer the following question as accurately and completely as possible.

IMPORTANT RULES:
1. Scan ALL provided memories before answering — do not stop after the first relevant one.
2. If multiple memories contain relevant information, combine and cross-reference them.
3. If the memories contain contradictory information, prefer the more recent one.
4. If the memories don't contain enough information to answer, say exactly: "I don't have enough information to answer this question."
5. For temporal questions: pay attention to dates and relative time references.
6. For ordering questions: present events in chronological order.
7. For preference questions: use the most recently stated preference.
8. Be specific and direct — include exact names, dates, numbers, and details from the memories.
9. Do NOT invent or assume information that isn't in the memories.

QUESTION: {question}

RETRIEVED MEMORIES:
{memories_text}

ANSWER:"""


_oc = None
def openai_client():
    global _oc
    if _oc is None:
        from openai import OpenAI
        _oc = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"), max_retries=2, timeout=300)
    return _oc


def generate_answer(ci, prompt):
    if a.fake_llm:
        n = prompt.count("\n") - 20
        record(ci, "answer", a.answer_model, len(prompt) // 4, 20, 0.0)
        return f"ANSWER: fake answer from {max(n, 0)} memory lines"
    m = a.answer_model.lower()
    kw = {"max_completion_tokens": 4096} if m.startswith(("gpt-5", "o1", "o3", "o4")) else {"max_tokens": 4096, "temperature": 0}
    for attempt in range(5):
        t0 = time.time()
        try:
            r = openai_client().chat.completions.create(model=a.answer_model, messages=[{"role": "user", "content": prompt}], **kw)
            content = r.choices[0].message.content
            record(ci, "answer", a.answer_model, r.usage.prompt_tokens, r.usage.completion_tokens, time.time() - t0)
            if content is None:
                if attempt < 4: continue
                return ""
            return content.strip()
        except Exception as e:
            record(ci, "answer", a.answer_model, 0, 0, time.time() - t0, ok=False)
            if attempt == 4: raise
            time.sleep(2 ** attempt)


def answer_conv(ci, m, ck):
    path = os.path.join(STORE, f"conv_{ci}", "answers.jsonl")
    done = {}
    if a.resume and os.path.exists(path):
        for l in open(path):
            if l.strip(): d = json.loads(l); done[d["question_id"]] = d
    f = open(path, "a" if a.resume else "w")
    for q in sel_by_conv[ci]:
        if q["mem0_qid"] in done: continue
        if stop.is_set(): break
        if running_usd() >= a.max_usd:
            stop.set(); print(f"[{ci}] cost cap ${a.max_usd} reached, stopping", flush=True); break
        t0 = time.monotonic()
        res = m.search(q["question"], top_k=a.top_k, filters={"user_id": ck["user_id"]}, threshold=a.threshold)
        lat = (time.monotonic() - t0) * 1000
        formatted = []
        for r in sorted(res.get("results", []), key=lambda x: x.get("score", 0), reverse=True):
            e = {"memory": r.get("memory", ""), "score": r.get("score", 0), "id": r.get("id", "")}
            if r.get("created_at"): e["created_at"] = r["created_at"]
            if r.get("updated_at"): e["updated_at"] = r["updated_at"]
            formatted.append(e)
        sliced = sorted(formatted[: a.top_k], key=lambda x: x.get("created_at", "") or "")
        gen = generate_answer(ci, get_beam_answer_generation_prompt(q["question"], sliced, top_k=a.top_k))
        if "ANSWER:" in gen: gen = gen.rsplit("ANSWER:", 1)[-1].strip()
        row = {
            "question_id": q["mem0_qid"], "qid": q["qid"], "chat_size": "1M", "conversation_idx": ci, "conversation_id": convs[ci]["cid"],
            "question_type": q["ability"], "question_type_idx": q["mem0_qi"], "difficulty": q["difficulty"], "question": q["question"],
            "rubric": q["rubric"], "ground_truth_answer": " | ".join(q["rubric"]), "source_chat_ids": q["source_chat_ids"], "user_id": ck["user_id"],
            "retrieval": {"search_query": q["question"], "search_results": formatted, "search_latency_ms": round(lat, 1), "total_results": len(formatted)},
            "cutoff_results": {f"top_{a.top_k}": {"judgment": None, "score": None, "generated_answer": gen, "memories_evaluated": len(sliced), "nugget_scores": []}},
        }
        f.write(json.dumps(row) + "\n"); f.flush(); done[q["mem0_qid"]] = row
    f.close()
    return done


def run_conv(ci):
    t0 = time.time()
    ck = load_ckpt(ci)
    with lock:
        USAGE[ci] = ck["usage"] if (a.resume and ck.get("usage")) else blank()
    m = make_memory(ci)
    if not ck["complete"] or len(ck["done"]) < total_chunks[ci]:
        ck["complete"] = False
        if not ingest(ci, m, ck): write_cost(); return ci, {}
    answers = answer_conv(ci, m, ck)
    with lock: USAGE[ci]["wall_s_total"] += time.time() - t0
    write_cost()
    print(f"[{ci}] done: {len(answers)} answers, total ${running_usd():.2f}", flush=True)
    return ci, answers


def write_results(by_conv):
    evals = []
    for ci in sorted(by_conv):
        order = {q["mem0_qid"]: i for i, q in enumerate(convs[ci]["qs"])}
        evals += sorted(by_conv[ci].values(), key=lambda r: order[r["question_id"]])
    doc = {"metadata": {"eval_type": "beam", "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"), "project_name": f"beam-1m-mem0oss-{a.tag}",
                        "system": "mem0ai 2.0.0 open source (local Memory, qdrant on disk, bge-small-en-v1.5)", "answerer_model": a.answer_model,
                        "extraction_model": a.llm_model, "judge_model": None, "provider": "openai", "total_questions": len(evals),
                        "questions_selected": len(sel), "seed": a.seed, "conversations": need, "max_chunks": a.max_chunks, "fake_llm": a.fake_llm,
                        "top_k_cutoffs": [f"top_{a.top_k}"], "chat_sizes": ["1M"], "question_types": ABILITIES},
           "metrics_by_cutoff": {}, "evaluations": evals}
    json.dump(doc, open(OUT, "w"), indent=1)
    return len(evals)


if __name__ == "__main__":
    EMB = SharedEmbedder()
    by_conv = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for ci, answers in ex.map(run_conv, need):
            by_conv[ci] = answers
    n = write_results(by_conv)
    write_cost()
    tot = json.load(open(COST_OUT))
    t = tot["total"]
    print(f"\nwrote {OUT}: {n} of {len(sel)} answers" + ("  (STOPPED ON COST CAP)" if stop.is_set() else ""))
    print(f"ingestion: {t['ingest']['calls']} LLM calls ({t['ingest']['failed_calls']} failed), {t['ingest']['prompt_tokens']:,} in / "
          f"{t['ingest']['completion_tokens']:,} out tokens, {t['ingest']['memories_added']} memories, ${t['ingest']['usd']:.3f}, call wall {t['ingest']['wall_s']:.0f}s")
    print(f"answers:   {t['answer']['calls']} LLM calls, {t['answer']['prompt_tokens']:,} in / {t['answer']['completion_tokens']:,} out tokens, ${t['answer']['usd']:.3f}")
    print(f"total ${tot['total_usd']:.3f}; cost sidecar {COST_OUT}")
    print("MEM0_OSS_BEAM_DONE")
