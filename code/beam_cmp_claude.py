#!/usr/bin/env python3
"""Compare two Claude-judged BEAM files (ours vs Mem0) by ability and overall, with a conversation-cluster bootstrap interval on the gap.
    .venv/bin/python beam_cmp_claude.py results/claude-judge-beam-ours-<stem>.jsonl results/claude-judge-beam-mem0-beam_1m_results.jsonl
Reads result files only. qid forms: ours "<conv>_<ability>_<i>", Mem0 "1M_<conv>_q<i>_<ability>"."""
import json, random, re, sys
from collections import defaultdict
def load(p):
    rows = defaultdict(list)
    for l in open(p):
        r = json.loads(l); m = re.match(r"(?:\d+[KM]_)?(\d+)_", r["qid"])
        rows[int(m.group(1))].append((r["ability"], bool(r["passed"])))
    return rows
A, B = load(sys.argv[1]), load(sys.argv[2])
def rate(rows, ab=None):
    xs = [p for c in rows for a, p in rows[c] if ab is None or a == ab]
    return 100 * sum(xs) / max(1, len(xs)), len(xs)
abil = sorted({a for c in A for a, _ in A[c]})
print(f"{'ability':26s} {'ours':>10s} {'mem0':>10s} {'gap':>7s}")
for ab in abil + [None]:
    ra, na = rate(A, ab); rb, nb = rate(B, ab)
    print(f"{ab or 'OVERALL':26s} {ra:6.1f}/{na:<3d} {rb:6.1f}/{nb:<3d} {ra-rb:+7.1f}")
rnd = random.Random(0); ca, cb = sorted(A), sorted(B); gaps = []
for _ in range(2000):
    sa = [c for c in (rnd.choice(ca) for _ in ca)]; sb = [c for c in (rnd.choice(cb) for _ in cb)]
    gaps.append(rate({i: A[c] for i, c in enumerate(sa)})[0] - rate({i: B[c] for i, c in enumerate(sb)})[0])
gaps.sort(); print(f"gap 95% interval (conversation bootstrap, 2000 draws): {gaps[50]:+.1f} to {gaps[1949]:+.1f}")
