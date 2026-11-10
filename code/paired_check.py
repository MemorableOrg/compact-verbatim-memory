import json, random, re, sys, csv
from collections import defaultdict
# 1) paired vs independent bootstrap from the compact verdict csv
rows = list(csv.DictReader(open("results/beam1m-one-judge-verdicts-compact.csv")))
by = defaultdict(lambda: defaultdict(list))
for r in rows:
    c = int(r["conv"])
    for k in ("mem0_pass", "base_pass", "full_pass"): by[k][c].append(int(r[k]))
convs = sorted(by["mem0_pass"])
def rate(k, cs): 
    xs = [p for c in cs for p in by[k][c]]; return 100 * sum(xs) / len(xs)
print("totals", {k: sum(sum(v) for v in by[k].values()) for k in by})
for draws in (2000, 100000):
    rnd = random.Random(0); paired = defaultdict(list); indep = defaultdict(list)
    for _ in range(draws):
        s = [rnd.choice(convs) for _ in convs]
        for a, b in (("full_pass","mem0_pass"),("base_pass","mem0_pass"),("full_pass","base_pass")):
            paired[(a,b)].append(rate(a, s) - rate(b, s))
    rnd = random.Random(0)
    for _ in range(draws):
        for a, b in (("full_pass","mem0_pass"),("base_pass","mem0_pass"),("full_pass","base_pass")):
            sa = [rnd.choice(convs) for _ in convs]; sb = [rnd.choice(convs) for _ in convs]
            indep[(a,b)].append(rate(a, sa) - rate(b, sb))
    for key in paired:
        p = sorted(paired[key]); i = sorted(indep[key]); lo, hi = int(0.025*draws), int(0.975*draws)-1
        print(f"draws {draws:6d} {key[0][:4]}-{key[1][:4]}: paired [{p[lo]:+.2f}, {p[hi]:+.2f}]  independent [{i[lo]:+.2f}, {i[hi]:+.2f}]  point {rate(key[0],convs)-rate(key[1],convs):+.2f}")
# 2) question text mapping: ours vs Mem0 by (conv, ability, index)
ours = [json.loads(l) for l in open("results/beam-20261108-final-v2-all.jsonl")]
import os
mp = os.path.expanduser("~/Documents/fm-data/beam-repos/mem0mb/results/platform/beam_1m_results.json")
m = json.load(open(mp))
items = m if isinstance(m, list) else (m.get("results") or m.get("questions") or list(m.values())[0])
print("mem0 rows", len(items), "keys", sorted(items[0].keys())[:12])
def norm(s): return re.sub(r"\W+", " ", (s or "").lower()).strip()
mq = {}
for it in items:
    qid = it.get("qid") or it.get("question_id") or it.get("id") or ""
    mm = re.match(r"(?:\d+[KM]_)?(\d+)_q(\d+)_(.+)", str(qid))
    if mm: mq[(int(mm.group(1)), mm.group(3), int(mm.group(2)))] = norm(it.get("question") or it.get("query") or "")
    else: print("odd qid", qid)
same = diff = missing = 0; shown = 0
for r in ours:
    c, ab, i = re.match(r"(\d+)_(.+)_(\d+)$", r["qid"]).groups()
    key = (int(c), ab, int(i))
    if key not in mq: missing += 1; continue
    if mq[key] == norm(r["question"]): same += 1
    else:
        diff += 1
        if shown < 3: shown += 1; print("DIFF", key, "| ours:", r["question"][:80], "| mem0:", mq[key][:80])
print(f"question text match: same {same} diff {diff} missing {missing} of {len(ours)}")
