"""10M tier: ours vs Mem0 under the one judge, by ability, with independent (registered) and paired conversation bootstraps,
plus the registered-prediction check. Reads result files only.
    .venv/bin/python -I cmp10m.py results/claude-judge-beam-ours-<stem>.jsonl results/claude-judge-beam-mem0-beam_10m_results.jsonl results/beam-<stem>.jsonl run-beam-10M.log
"""
import json, random, re, sys
from collections import defaultdict

def load(p):
    rows = defaultdict(list)
    for l in open(p):
        r = json.loads(l); m = re.match(r"(?:\d+[KM]_)?(\d+)_", r["qid"])
        rows[int(m.group(1))].append((r["ability"], bool(r["passed"])))
    return rows

A, B = load(sys.argv[1]), load(sys.argv[2])
def rate(rows, cs=None, ab=None):
    cs = sorted(rows) if cs is None else cs
    xs = [p for c in cs for a, p in rows[c] if ab is None or a == ab]
    return 100 * sum(xs) / max(1, len(xs)), len(xs)
abil = sorted({a for c in A for a, _ in A[c]})
print(f"{'ability':26s} {'ours':>9s} {'mem0':>9s} {'gap':>7s}")
for ab in abil + [None]:
    ra, na = rate(A, ab=ab); rb, nb = rate(B, ab=ab)
    print(f"{ab or 'OVERALL':26s} {ra:5.1f}/{na:<3d} {rb:5.1f}/{nb:<3d} {ra-rb:+7.1f}")
convs = sorted(A); assert convs == sorted(B), "conversation sets differ"
for draws in (2000,):
    rnd = random.Random(0); ind = []; par = []
    for _ in range(draws):
        sa = [rnd.choice(convs) for _ in convs]; sb = [rnd.choice(convs) for _ in convs]
        ind.append(rate(A, sa)[0] - rate(B, sb)[0])
    rnd = random.Random(0)
    for _ in range(draws):
        s = [rnd.choice(convs) for _ in convs]; par.append(rate(A, s)[0] - rate(B, s)[0])
    ind.sort(); par.sort(); lo, hi = int(0.025 * draws), int(0.975 * draws) - 1
    print(f"gap interval, {draws} draws: independent (registered) [{ind[lo]:+.1f}, {ind[hi]:+.1f}]; paired [{par[lo]:+.1f}, {par[hi]:+.1f}]")
# run facts
if len(sys.argv) > 3:
    rows = [json.loads(l) for l in open(sys.argv[3])]
    ain = sum(r["tokens"]["ain"] for r in rows); aout = sum(r["tokens"]["aout"] for r in rows)
    print(f"run rows {len(rows)}; reader tokens per question {ain/len(rows):,.0f} in / {aout/len(rows):,.0f} out; reader $ at gpt-5 prices {ain*1.25/1e6 + aout*10/1e6:.2f}; mean turns read {sum(r['turns_read'] for r in rows)/len(rows):.1f}; widened {sum(r['passes']==2 for r in rows)}")
    gr = [r["gold_recall"] for r in rows if "gold_recall" in r]
    if gr: print(f"gold-message recall {sum(gr)/len(gr):.3f} over {len(gr)} questions with gold")
if len(sys.argv) > 4:
    log = open(sys.argv[4]).read()
    for key in ("reader cost", "store total", "index ready", "adaptive:"):
        for line in log.splitlines()[::-1]:
            if key in line: print("LOG:", line[:200]); break
# registered predictions (ledger, Nov 9 2:05am)
ra = rate(A)[0]; rb = rate(B)[0]; gap = ra - rb
ca = rate(A, ab="contradiction_resolution")[0]; cb = rate(B, ab="contradiction_resolution")[0]
ma = rate(A, ab="multi_session_reasoning")[0]; mb = rate(B, ab="multi_session_reasoning")[0]
print("\nregistered predictions:")
print(f"  Mem0 40 to 50: {rb:.1f} -> {'held' if 40 <= rb <= 50 else 'missed'}")
print(f"  ours 48 to 60: {ra:.1f} -> {'held' if 48 <= ra <= 60 else 'missed'}")
print(f"  ours ahead by 5 or more: {gap:+.1f} -> {'held' if gap >= 5 else 'missed'}")
print(f"  contradiction 40 or more vs Mem0 under 20: {ca:.1f} vs {cb:.1f} -> {'held' if ca >= 40 and cb < 20 else 'missed'}")
print(f"  multi-session within 10 of Mem0: {ma:.1f} vs {mb:.1f} -> {'held' if abs(ma-mb) <= 10 else 'missed'}")
