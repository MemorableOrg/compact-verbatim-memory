#!/usr/bin/env python3
"""Offline, from an mh_fix.py dump: what is left after the search fixes, and whether the ranker's own scores can set the depth.

    python3 mh_policy.py mhdump-nograph_ens-dev-noname1-rrn300.json

Prints: recall at fixed cuts; the remaining multi-hop misses at 40 with worst gold rank and gold count; depth-from-score rules
(read until the reranked score drops by d from the top, or until the largest gap in the top 150, clamped to [kmin, kmax]) against
the fixed-cut curve at the same mean length; a per-session cap. Labels are read only to score. Dev only unless told otherwise.
"""
import json, sys, collections
import numpy as np

rows = json.load(open(sys.argv[1]))
CUTS = (10, 20, 30, 40, 60, 100, 150, 200, 300)
print(f"{sys.argv[1]}: {len(rows)} questions")

def rec_at(r, k):
    return float(set(r["gold"]) <= set(r["top"][:k]))

def table(label, hits, ks):
    cells = []
    for c in (1, 2, 3, 4, "all"):
        rs = [(h, r) for h, r in zip(hits, rows) if c == "all" or r["cat"] == c]
        cells.append(f"cat{c} {100 * np.mean([h for h, _ in rs]):5.1f}" if c != "all" else f"all {100 * np.mean([h for h, _ in rs]):5.1f}")
    print(f"  {label:34s} mean turns {np.mean(ks):6.1f}  " + "  ".join(cells))

print("\nfixed cuts (all-gold recall):")
for k in CUTS:
    table(f"k={k}", [rec_at(r, k) for r in rows], [k] * len(rows))

print("\nremaining multi-hop misses at 40: worst gold rank | gold turns | list cue | question")
miss = [r for r in rows if r["cat"] == 1 and not rec_at(r, 40)]
for r in sorted(miss, key=lambda r: max((r["top"].index(g) + 1 if g in r["top"] else 999) for g in r["gold"])):
    w = max((r["top"].index(g) + 1 if g in r["top"] else 999) for g in r["gold"])
    print(f"  {w:>3} | {r['ngold']} | {'list' if r['cue'] else '    '} | {r['q'][:95]}")
wd = collections.Counter("41-60" if w < 60 else "61-100" if w < 100 else "101-200" if w < 200 else "201-300" if w < 300 else "300+" for w in
                         [max((r["top"].index(g) + 1 if g in r["top"] else 999) for g in r["gold"]) for r in miss])
print("  worst-rank buckets:", dict(wd), "| gold-count among misses:", dict(sorted(collections.Counter(r["ngold"] for r in miss).items())))

print("\ndepth from the reranked score (clamped to [kmin, kmax]); compare with the fixed cut of the same mean length above:")
for kmin, kmax in ((10, 150), (20, 150), (20, 200), (40, 200)):
    for d in (0.5, 1.0, 1.5, 2.0, 3.0):
        ks, hits = [], []
        for r in rows:
            sc = np.array(r["sc"]); top = sc[0]
            k = int(np.sum(sc >= top - d))
            k = min(max(k, kmin), kmax)
            ks.append(k); hits.append(rec_at(r, k))
        table(f"drop d={d} kmin={kmin} kmax={kmax}", hits, ks)
print("\nlargest gap in the top 150 (read up to the gap, clamped):")
for kmin, kmax in ((20, 150), (40, 200)):
    ks, hits = [], []
    for r in rows:
        sc = np.array(r["sc"][:150])
        gaps = sc[:-1] - sc[1:]
        k = int(np.argmax(gaps[kmin - 1:]) + kmin) if len(gaps) >= kmin else kmin
        k = min(max(k, kmin), kmax)
        ks.append(k); hits.append(rec_at(r, k))
    table(f"largest gap kmin={kmin} kmax={kmax}", hits, ks)

print("\nper-session cap inside the top 40 (excess turns from one session replaced by the next turns from other sessions):")
for cap in (4, 6, 8, 12):
    hits = []
    for r in rows:
        chosen, count, rest = [], collections.Counter(), []
        for i, s in zip(r["top"], r["sess"]):
            if len(chosen) >= 40:
                break
            if count[s] < cap:
                chosen.append(i); count[s] += 1
            else:
                rest.append(i)
        hits.append(float(set(r["gold"]) <= set(chosen)))
    table(f"session cap {cap} at 40", hits, [40] * len(rows))

print("\nscore scale check: top score, score at 40, score at 150 (mean over questions):",
      f"{np.mean([r['sc'][0] for r in rows]):.2f} {np.mean([r['sc'][39] for r in rows]):.2f} {np.mean([r['sc'][149] for r in rows if len(r['sc']) > 149]):.2f}")
print("MH_POLICY_DONE")
