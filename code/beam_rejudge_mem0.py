#!/usr/bin/env python3
"""Judge Mem0's published BEAM 1M answers with the judge our BEAM run used (BEAM's verbatim rubric prompt, gpt-4.1-mini by default),
so both systems are scored by one judge. No answer model is called.

    .venv/bin/python beam_rejudge_mem0.py [--judge-model gpt-4.1-mini] [--n 700] [--resume results/beam-mem0-rejudge-....jsonl]

Input: Mem0's results/platform/beam_1m_results.json (answerer gpt-5, judge gpt-5, 700 questions, 10 abilities).
Output: results/beam-mem0-rejudge-<ts>.jsonl, one row per question with item_scores, score (mean), passed (score >= 0.5).
"""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import oa
from beam_judge_prompt import JUDGE_PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("--judge-model", default="gpt-4.1-mini")
ap.add_argument("--n", type=int, default=700)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--resume", default="")
ap.add_argument("--src", default=os.path.expanduser("~/Documents/fm-data/beam-repos/mem0mb/results/platform/beam_1m_results.json"))
a = ap.parse_args()

client = oa.client()
use = {"jin": 0, "jout": 0}

def chat(prompt, **kw):
    for attempt in range(10):
        try:
            return client.chat.completions.create(model=oa.dep(a.judge_model), messages=[{"role": "user", "content": prompt}], **kw)
        except Exception as e:
            if attempt == 9:
                raise
            time.sleep(min(2 ** attempt, 30))

def parse_judge(txt):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip())
    try:
        return float(json.loads(t)["score"])
    except Exception:
        pass
    m = re.search(r'"score"\s*:\s*([0-9.]+)', t)
    return float(m.group(1)) if m else 0.0

def judge_item(item, resp):
    r = chat(JUDGE_PROMPT.replace("<rubric_item>", item).replace("<llm_response>", resp), temperature=0)
    use["jin"] += r.usage.prompt_tokens; use["jout"] += r.usage.completion_tokens
    s = parse_judge(r.choices[0].message.content or "")
    return s if s in (0.0, 0.5, 1.0) else min((0.0, 0.5, 1.0), key=lambda v: abs(v - s))

src = json.load(open(a.src))
qs = src["evaluations"][: a.n]
prev = [json.loads(l) for l in open(a.resume)] if a.resume else []
done = {r["qid"] for r in prev}
qs = [q for q in qs if q["question_id"] not in done]
out = a.resume or f"results/beam-mem0-rejudge-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
print(f"judging {len(qs)} Mem0 answers with {a.judge_model}; {len(prev)} resumed; output {out}", flush=True)

def one(q):
    cr = q["cutoff_results"]["top_200"]
    resp = str(cr.get("generated_answer", ""))
    items = q["rubric"] if isinstance(q["rubric"], list) else [str(q["rubric"])]
    scores = [judge_item(str(it), resp) for it in items]
    score = float(np.mean(scores)) if scores else 0.0
    return dict(qid=q["question_id"], ability=q["question_type"], question=q["question"], mem0_score=cr.get("score"),
                answer=resp, item_scores=scores, score=score, passed=score >= 0.5, judge_model=a.judge_model)

res = list(prev)
with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
    for r in ex.map(one, qs):
        res.append(r); f.write(json.dumps(r) + "\n"); f.flush()
        if len(res) % 50 == 0:
            print(len(res), "done", f"judge tokens {use['jin']:,}/{use['jout']:,}", flush=True)

print(f"\nMem0 BEAM 1M answers under BEAM's judge ({a.judge_model}), n={len(res)}")
for ab in sorted({r["ability"] for r in res}):
    rs = [r for r in res if r["ability"] == ab]
    print(f"{ab:26s} {len(rs):3d} pass {sum(r['passed'] for r in rs) / len(rs):6.3f} mean {np.mean([r['score'] for r in rs]):6.3f}  (Mem0's own judge: mean {np.mean([r['mem0_score'] or 0 for r in rs]):6.3f})")
print(f"{'OVERALL':26s} {len(res):3d} pass {sum(r['passed'] for r in res) / len(res):6.3f} mean {np.mean([r['score'] for r in res]):6.3f}")
print(f"judge tokens in/out {use['jin']:,}/{use['jout']:,}; results: {out}")
print("BEAM_REJUDGE_MEM0_DONE")
