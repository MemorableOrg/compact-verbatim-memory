#!/usr/bin/env python3
"""Paired comparison of our GPT-5 LoCoMo answers with Mem0's per-question file on the questions both answered, by category.

    conv/.venv/bin/python h2h_cats.py

Prints n, both/ours-only/mem0-only/neither counts, accuracy of each, the paired difference with a conversation-cluster bootstrap
95% interval (2,000 draws, seed 0) and the exact two-sided McNemar p-value from the discordant counts. Reads result files only.
"""
import json, math, random
from pathlib import Path

CONV = Path(__file__).resolve().parent
MEM0 = Path.home() / "memory-benchmarks/results/platform"
raw = json.load(open(CONV / "data" / "locomo10.json"))
idx_of = {c["sample_id"]: i for i, c in enumerate(raw)}
m = json.load(open(MEM0 / "locomo_results.json"))
mem = {(e["conversation_idx"], e["question"].strip()): e["cutoff_results"]["top_200"]["judgment"] == "CORRECT" for e in m["evaluations"]}
ours = {}
for fname in ("20261029-183439.jsonl", "20261029-181313.jsonl"):
    for l in open(CONV / "results" / fname):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if "correct" in r:
            ours[(idx_of[r["sample"]], r["question"].strip())] = (bool(r["correct"]), r["category"])
rows = [dict(conv=k[0], cat=ours[k][1], o=ours[k][0], m=mem[k]) for k in ours if k in mem]
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def report(name, rs):
    n = len(rs)
    both = sum(r["o"] and r["m"] for r in rs)
    oo = sum(r["o"] and not r["m"] for r in rs)
    mo = sum(r["m"] and not r["o"] for r in rs)
    nn = n - both - oo - mo
    acc_o = 100 * sum(r["o"] for r in rs) / n
    acc_m = 100 * sum(r["m"] for r in rs) / n
    convs = sorted({r["conv"] for r in rs})
    by = {c: [r for r in rs if r["conv"] == c] for c in convs}
    rng = random.Random(0)
    diffs = []
    for _ in range(2000):
        pick = [by[rng.choice(convs)] for _ in convs]
        flat = [r for g in pick for r in g]
        if flat:
            diffs.append(100 * (sum(r["o"] for r in flat) - sum(r["m"] for r in flat)) / len(flat))
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    print(f"{name:12s} n={n:4d} both={both:3d} ours-only={oo:3d} mem0-only={mo:3d} neither={nn:3d} | ours {acc_o:5.1f} mem0 {acc_m:5.1f} "
          f"diff {acc_o - acc_m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}] cluster-boot over {len(convs)} convs | McNemar p={mcnemar(oo, mo):.3f}")


report("all", rows)
for c in (1, 2, 3, 4):
    report(CAT[c], [r for r in rows if r["cat"] == c])

print("--- by split (development = conversations 0 to 2)")
for name, keep in (("dev", lambda r: r["conv"] <= 2), ("test", lambda r: r["conv"] >= 3)):
    sub = [r for r in rows if keep(r)]
    report(name + " all", sub)
    for c in (1, 2, 3, 4):
        report(name + " " + CAT[c], [r for r in sub if r["cat"] == c])
