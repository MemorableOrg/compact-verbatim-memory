#!/usr/bin/env python3
"""Judge our BEAM 1M answers with Mem0's own nugget judge (its prompt, gpt-5 by default), the mirror of beam_rejudge_mem0.py,
so the two systems can be compared under either judge. No answer model is called.

    .venv/bin/python beam_rejudge_ours_mem0judge.py results/beam-20261029-171606-full-1M-gpt5.jsonl [--judge-model gpt-5]

Output: results/<input stem>-mem0judge-<model>.jsonl with item_scores, score (mean), passed (score >= 0.5).
"""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import oa
from beam_judge_prompt_mem0 import BEAM_JUDGE_SYSTEM_PROMPT, get_beam_nugget_judge_prompt

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--judge-model", default="gpt-5")
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()
client = oa.client()
use = {"jin": 0, "jout": 0}

def chat(msgs, **kw):
    for attempt in range(10):
        try:
            return client.chat.completions.create(model=oa.dep(a.judge_model), messages=msgs, **kw)
        except Exception as e:
            if attempt == 9:
                raise
            time.sleep(min(2 ** attempt, 30))

def clamp(v):
    return min((0.0, 0.5, 1.0), key=lambda c: abs(c - v))

def judge(question, nugget, resp):
    r = chat([{"role": "system", "content": BEAM_JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": get_beam_nugget_judge_prompt(question, nugget, resp)}])
    use["jin"] += r.usage.prompt_tokens; use["jout"] += r.usage.completion_tokens
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (r.choices[0].message.content or "").strip())
    try:
        return clamp(float(json.loads(t)["score"]))
    except Exception:
        m = re.search(r'"score"\s*:\s*([0-9.]+)', t)
        return clamp(float(m.group(1))) if m else 0.0

rows = [json.loads(l) for l in open(a.src) if l.strip()]
out = a.src.rsplit(".", 1)[0] + f"-mem0judge-{a.judge_model}.jsonl"
done = {json.loads(l)["qid"] for l in open(out)} if os.path.exists(out) else set()
todo = [r for r in rows if r["qid"] not in done]
print(f"judging {len(todo)} of our answers with Mem0's nugget judge on {a.judge_model}; output {out}", flush=True)

def one(r):
    rub = r["rubric"] if isinstance(r["rubric"], list) else json.loads(r["rubric"]) if str(r["rubric"]).startswith("[") else [str(r["rubric"])]
    scores = [judge(r["question"], str(it), r["answer"]) for it in rub]
    score = float(np.mean(scores)) if scores else 0.0
    return dict(qid=r["qid"], ability=r["ability"], our_score=r["score"], item_scores=scores, score=score, passed=score >= 0.5, judge_model=a.judge_model)

res = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
    for r in ex.map(one, todo):
        res.append(r); f.write(json.dumps(r) + "\n"); f.flush()
        if len(res) % 50 == 0:
            print(len(res), "done", flush=True)
print(f"\nOur BEAM 1M answers under Mem0's judge ({a.judge_model}), n={len(res)}")
for ab in sorted({r["ability"] for r in res}):
    rs = [r for r in res if r["ability"] == ab]
    print(f"{ab:26s} {len(rs):3d} pass {sum(r['passed'] for r in rs) / len(rs):6.3f} mean {np.mean([r['score'] for r in rs]):6.3f}  (BEAM judge: mean {np.mean([r['our_score'] for r in rs]):6.3f})")
print(f"{'OVERALL':26s} {len(res):3d} pass {sum(r['passed'] for r in res) / len(res):6.3f} mean {np.mean([r['score'] for r in res]):6.3f}")
print(f"judge tokens in/out {use['jin']:,}/{use['jout']:,}")
print("BEAM_REJUDGE_OURS_DONE")
