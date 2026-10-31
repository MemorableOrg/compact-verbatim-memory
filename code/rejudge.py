#!/usr/bin/env python3
"""Re-judge an existing LoCoMo results file with another judge model, same Mem0 judge prompt. Writes results/<stem>-rejudge-<model>.jsonl.
    .venv/bin/python rejudge.py results/20261029-145312.jsonl gpt-5 [workers]
"""
import json, sys, time, re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
sys.argv += [] ; src, judge = sys.argv[1], sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
jp_name = sys.argv[4] if len(sys.argv) > 4 else "mem0v1"
if jp_name == "mem0v2":
    from mem0v2_prompts import JUDGE_SYSTEM_PROMPT, get_judge_prompt, preprocess_answer
else:
    from judge_prompt import JUDGE_SYSTEM_PROMPT, get_judge_prompt, preprocess_answer
import oa
client = oa.client()
rows = [json.loads(l) for l in open(src)]
gold_by = {}
for c in json.load(open("data/locomo10.json")):
    for j, q in enumerate(c["qa"]):
        gold_by[(c["sample_id"], j)] = q
def chat(msgs, **kw):
    for attempt in range(12):
        try: return client.chat.completions.create(model=oa.dep(judge), messages=msgs, **kw)
        except Exception as e:
            if attempt == 11: raise
            time.sleep(min(2 ** attempt, 30))
def one(r):
    q = gold_by[(r["sample"], r["qi"])]
    gold = preprocess_answer(q["category"], str(q["answer"]))
    kw = {"response_format": {"type": "json_object"}}
    if not judge.startswith("gpt-5"): kw["temperature"] = 0
    jr = chat([{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": get_judge_prompt(q["category"], q["question"], gold, r["answer"])}], **kw)
    try: label = json.loads(jr.choices[0].message.content)["label"].strip().upper()
    except Exception: label = "WRONG"
    out = dict(r); out["judge_model"] = judge; out["correct_orig"] = r["correct"]; out["correct"] = label == "CORRECT"
    out["judge_tokens"] = dict(jin=jr.usage.prompt_tokens, jout=jr.usage.completion_tokens)
    return out
out_path = src.replace(".jsonl", f"-rejudge-{judge}-{jp_name}.jsonl")
res = []
with ThreadPoolExecutor(workers) as ex, open(out_path, "w") as f:
    for o in ex.map(one, rows):
        res.append(o); f.write(json.dumps(o) + "\n"); f.flush()
n = len(res); dev = {"conv-26", "conv-30", "conv-41"}
acc = lambda rs: 100 * sum(x["correct"] for x in rs) / len(rs)
print(f"rejudged {n} with {judge}: original {acc([dict(correct=x['correct_orig']) for x in res]):.1f} -> {acc(res):.1f}; test {acc([x for x in res if x['sample'] not in dev]):.1f}")
for c in (1, 2, 3, 4): print(f"  cat{c}: {acc([x for x in res if x['category']==c]):.1f}")
flips = sum(1 for x in res if x["correct_orig"] and not x["correct"]); back = sum(1 for x in res if not x["correct_orig"] and x["correct"])
print(f"  marked wrong now: {flips}; marked right now: {back}; judge tokens in {sum(x['judge_tokens']['jin'] for x in res):,} out {sum(x['judge_tokens']['jout'] for x in res):,}")
print("REJUDGE_DONE")
