#!/usr/bin/env python3
"""One judge for both systems: Claude Sonnet 5.5 marks existing answers with the same prompt on both sides (Addendum 41).

    .venv/bin/python claude_judge.py locomo results/20261029-181313.jsonl results/20261029-183439.jsonl results/mem0-current-locomo-rows.jsonl
    .venv/bin/python claude_judge.py beam ours results/beam-20261029-171606-full-1M-gpt5.jsonl
    .venv/bin/python claude_judge.py beam mem0 ~/memory-benchmarks/results/platform/beam_1m_results.json
    .venv/bin/python claude_judge.py lme results/lme-20261029-182856.jsonl results/lme-20261029-182208.jsonl

No answer is generated; every call is a judge call (the installed SDK takes no temperature argument, so the model default applies). Output results/claude-judge-<bench>-<stem>.jsonl, resumable.
Env: --limit N judges only the first N rows (smoke test). Spend is estimated at 3 and 15 dollars per million input and output tokens
and the run stops at 40 dollars.
"""
import json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import anthropic

MODEL = "claude-sonnet-5-5"
WORKERS = 8
CAP = 40.0
LIMIT = None
args = [a for a in sys.argv[1:] if not a.startswith("--limit")]
for a in sys.argv[1:]:
    if a.startswith("--limit="):
        LIMIT = int(a.split("=")[1])
client = anthropic.Anthropic()
use = {"in": 0, "out": 0}
lock = threading.Lock()


def spend():
    return use["in"] / 1e6 * 3 + use["out"] / 1e6 * 15


def ask(user, system=None, max_tokens=150):
    for attempt in range(10):
        try:
            kw = dict(model=MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": user}])
            if system:
                kw["system"] = system
            r = client.messages.create(**kw)
            with lock:
                use["in"] += r.usage.input_tokens; use["out"] += r.usage.output_tokens
                if spend() > CAP:
                    raise SystemExit(f"spend cap reached: {spend():.2f} dollars")
            return "".join(b.text for b in r.content if getattr(b, "text", None))
        except SystemExit:
            raise
        except Exception as e:
            if attempt == 9:
                raise
            time.sleep(min(2 ** attempt, 30))


def out_path(bench, src):
    stem = os.path.basename(src).replace(".jsonl", "").replace(".json", "")
    return f"results/claude-judge-{bench}-{stem}.jsonl"


def run(bench, src, rows, one, key):
    path = out_path(bench, src)
    done = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                d = json.loads(l); done[key(d)] = d
    todo = [r for r in rows if key(r) not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"{bench} {os.path.basename(src)}: {len(rows)} rows, {len(done)} done, judging {len(todo)}", flush=True)
    res = list(done.values())
    with ThreadPoolExecutor(WORKERS) as ex, open(path, "a") as f:
        for o in ex.map(one, todo):
            res.append(o); f.write(json.dumps(o) + "\n"); f.flush()
            if len(res) % 100 == 0:
                print(f"  {len(res)} done, {use['in']:,}/{use['out']:,} tokens, {spend():.2f} dollars", flush=True)
    return res


# ---------------------------------------------------------------- LoCoMo: Mem0's current judge prompt ------------------------
def locomo(srcs):
    from mem0v2_prompts import JUDGE_SYSTEM_PROMPT, get_judge_prompt, preprocess_answer
    gold_by = {}
    for c in json.load(open("data/locomo10.json")):
        for j, q in enumerate(c["qa"]):
            gold_by[(c["sample_id"], j)] = q
    dev = {"conv-26", "conv-30", "conv-41"}

    def one(r):
        q = gold_by.get((r["sample"], r["qi"]), {})
        gold_raw = r.get("gold") if r.get("gold") not in (None, "") else q.get("answer", q.get("adversarial_answer", ""))
        gold = preprocess_answer(r["category"], str(gold_raw))
        txt = ask(get_judge_prompt(r["category"], r["question"], gold, r["answer"]), system=JUDGE_SYSTEM_PROMPT, max_tokens=150)
        m = re.search(r'"label"\s*:\s*"?\s*(CORRECT|WRONG)', txt, re.I)
        label = m.group(1).upper() if m else ("CORRECT" if re.search(r"\bCORRECT\b", txt) and not re.search(r"\bWRONG\b", txt) else "WRONG")
        return dict(sample=r["sample"], qi=r["qi"], category=r["category"], question=r["question"], answer=r["answer"],
                    correct_orig=bool(r["correct"]), correct=label == "CORRECT", judge_model=MODEL)

    for src in srcs:
        rows = [json.loads(l) for l in open(src) if l.strip()]
        rows = [r for r in rows if "correct" in r]
        res = run("locomo", src, rows, one, lambda d: (d["sample"], d["qi"]))
        acc = lambda rs: 100 * sum(x["correct"] for x in rs) / max(1, len(rs))
        acc0 = lambda rs: 100 * sum(x["correct_orig"] for x in rs) / max(1, len(rs))
        agree = 100 * sum(x["correct"] == x["correct_orig"] for x in res) / max(1, len(res))
        print(f"{os.path.basename(src)}: n={len(res)} Claude {acc(res):.1f} (original judge {acc0(res):.1f}); dev {acc([x for x in res if x['sample'] in dev]):.1f} "
              f"test {acc([x for x in res if x['sample'] not in dev]):.1f}; agreement {agree:.1f}; " +
              " ".join(f"cat{c} {acc([x for x in res if x['category'] == c]):.1f}" for c in (1, 2, 3, 4)), flush=True)


# ---------------------------------------------------------------- BEAM: BEAM's rubric prompt, item by item ------------------
def beam(side, src):
    from beam_judge_prompt import JUDGE_PROMPT

    def parse(txt):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip())
        try:
            s = float(json.loads(t)["score"])
        except Exception:
            m = re.search(r'"score"\s*:\s*([0-9.]+)', t)
            s = float(m.group(1)) if m else 0.0
        return min((0.0, 0.5, 1.0), key=lambda v: abs(v - s))

    if side == "ours":
        rows = [json.loads(l) for l in open(src) if l.strip()]
        items = [dict(qid=r["qid"], ability=r["ability"], answer=str(r["answer"]), rubric=r["rubric"] if isinstance(r["rubric"], list) else [str(r["rubric"])],
                      orig_score=r.get("score"), orig_passed=r.get("passed")) for r in rows]
    else:
        ev = json.load(open(os.path.expanduser(src)))["evaluations"]
        items = [dict(qid=e["question_id"], ability=e["question_type"], answer=str(e["cutoff_results"]["top_200"].get("generated_answer", "")),
                      rubric=e["rubric"] if isinstance(e["rubric"], list) else [str(e["rubric"])], orig_score=e["cutoff_results"]["top_200"].get("score"),
                      orig_passed=None) for e in ev]

    def one(it):
        scores = [parse(ask(JUDGE_PROMPT.replace("<rubric_item>", str(x)).replace("<llm_response>", it["answer"]), max_tokens=200)) for x in it["rubric"]]
        score = float(np.mean(scores)) if scores else 0.0
        return dict(qid=it["qid"], ability=it["ability"], item_scores=scores, score=score, passed=score >= 0.5,
                    orig_score=it["orig_score"], orig_passed=it["orig_passed"], judge_model=MODEL)

    res = run(f"beam-{side}", src, items, one, lambda d: d["qid"])
    print(f"BEAM {side}: n={len(res)} pass {100 * sum(r['passed'] for r in res) / len(res):.1f} mean {np.mean([r['score'] for r in res]):.3f}", flush=True)
    for ab in sorted({r["ability"] for r in res}):
        rs = [r for r in res if r["ability"] == ab]
        print(f"  {ab:26s} {len(rs):3d} pass {100 * sum(r['passed'] for r in rs) / len(rs):5.1f} mean {np.mean([r['score'] for r in rs]):.3f}", flush=True)


# ---------------------------------------------------------------- LongMemEval: official per-type prompt, ours only --------------
def lme(srcs):
    from judge_prompt_lme import get_anscheck_prompt

    def one(r):
        txt = ask(get_anscheck_prompt(r["type"], r["question"], r["gold"], r["answer"], abstention=bool(r.get("abstention"))), max_tokens=10)
        return dict(question_id=r["question_id"], type=r["type"], correct_orig=bool(r["correct"]), correct=txt.strip().lower().startswith("yes"), judge_model=MODEL)

    for src in srcs:
        rows = [json.loads(l) for l in open(src) if l.strip()]
        rows = [r for r in rows if "correct" in r]
        res = run("lme", src, rows, one, lambda d: d["question_id"])
        agree = 100 * sum(x["correct"] == x["correct_orig"] for x in res) / max(1, len(res))
        print(f"{os.path.basename(src)}: n={len(res)} Claude {100 * sum(x['correct'] for x in res) / len(res):.1f} "
              f"(official judge {100 * sum(x['correct_orig'] for x in res) / len(res):.1f}); agreement {agree:.1f}", flush=True)


if __name__ == "__main__":
    bench = args[0]
    if bench == "locomo":
        locomo(args[1:])
    elif bench == "beam":
        beam(args[1], args[2])
    elif bench == "lme":
        lme(args[1:])
    print(f"tokens in/out {use['in']:,}/{use['out']:,}; estimated spend {spend():.2f} dollars")
    print("CLAUDE_JUDGE_DONE")
