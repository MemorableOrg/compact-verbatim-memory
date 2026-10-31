#!/usr/bin/env python3
"""Development-split test of cheaper rerank schedules for the turns store (LoCoMo, 383 questions, labels used only to score).

    ~/Documents/GitHub/snapgraph/.venv/bin/python cascade_dev.py

Settings, all but the first with speaker names out of the first stage:
  shipped   both cross-encoders over the top 100 (names in; the setting every reader row used)
  v2        both cross-encoders over the top 300 (the frozen later change)
  cascade   MiniLM over the top 300, then bge-reranker-v2-m3 over the top 100 of that
  bge300    bge-reranker-v2-m3 alone over the top 300
Prints all-gold recall at 40 and 150 for each. Times printed here are not cold (the score cache is on); see time_schedules.py.
"""
import json, os, re, statistics, sys, time
import numpy as np
SG = os.path.expanduser("~/Documents/GitHub/snapgraph")
sys.path.insert(0, os.path.join(SG, "src"))
os.chdir(SG)
import load, loc, rerank  # noqa: E402
from run_turn import z  # noqa: E402
from rankers import rank_desc  # noqa: E402

cfg = json.load(open("configs/final.json"))["locomo"]["arms"]["nograph_ens"]


def strip_names(conv, q):
    for n in conv.named(q):
        q = re.sub(r"\b" + re.escape(n) + r"(?:'s)?\b", "", q)
    return re.sub(r"\s+", " ", q).strip()


def first_stages(conv, qt, names_out):
    qs = strip_names(conv, qt) if names_out else qt
    s = conv.fused(qs, cfg["fuse"])
    return loc.speaker_prior(s, conv, qt, cfg)


def place(s, cand, comb):
    out = np.array(s, dtype=float)
    big = s.max() - s.min() + 100
    for i, v in zip(cand, comb):
        out[i] = big + v
    return out


def both(s, conv, q, N):
    cand = [int(i) for i in rank_desc(s)[:N]]
    pairs = [(q, conv.texts[i]) for i in cand]
    base = z(np.array([s[i] for i in cand]))
    comb = 0.4 * base + 0.2 * z(rerank.get("msmarco").score(pairs)) + 0.4 * z(rerank.get("bge-v2m3").score(pairs))
    return place(s, cand, comb)


def bge_only(s, conv, q, N):
    cand = [int(i) for i in rank_desc(s)[:N]]
    pairs = [(q, conv.texts[i]) for i in cand]
    comb = 0.6 * z(np.array([s[i] for i in cand])) + 0.4 * z(rerank.get("bge-v2m3").score(pairs))
    return place(s, cand, comb)


def cascade(s, conv, q, N1=300, N2=100):
    cand = [int(i) for i in rank_desc(s)[:N1]]
    pairs = [(q, conv.texts[i]) for i in cand]
    base = z(np.array([s[i] for i in cand]))
    mini = z(rerank.get("msmarco").score(pairs))
    stage1 = 0.6 * base + 0.4 * mini
    top = np.argsort(-stage1)[:N2]
    bge = z(rerank.get("bge-v2m3").score([pairs[j] for j in top]))
    comb = stage1.copy()  # the 200 below keep their stage-1 score
    comb[top] = 0.4 * base[top] + 0.2 * mini[top] + 0.4 * bge + 10  # the 100 rescored stay above the rest
    return place(s, cand, comb)


SETTINGS = {
    "shipped": lambda s, c, q: both(s, c, q, 100),
    "v2": lambda s, c, q: both(s, c, q, 300),
    "cascade": lambda s, c, q: cascade(s, c, q),
    "bge300": lambda s, c, q: bge_only(s, c, q, 300),
}
NAMES_OUT = {"shipped": False, "v2": True, "cascade": True, "bge300": True}
CUTS = (40, 150)


def main():
    sysd, scd = load.load_locomo("dev")
    rows = {k: [] for k in SETTINGS}
    times = {k: [] for k in SETTINGS}
    for cs, scn in zip(sysd, scd):
        conv = loc.Conv(cs, True)
        for q, lab in zip(cs["questions"], scn["qa"]):
            if lab["category"] not in (1, 2, 3, 4):
                continue
            gold = {conv.keys.index(g) for g in loc.gold_of(conv, lab)}
            if not gold:
                continue
            for name, fn in SETTINGS.items():
                t0 = time.perf_counter()
                s = first_stages(conv, q["question"], NAMES_OUT[name])
                r = fn(s, conv, q["question"])
                times[name].append(time.perf_counter() - t0)
                order = [int(i) for i in rank_desc(r)]
                rows[name].append({k: float(set(order[:k]) >= gold) for k in CUTS} | {"cat": lab["category"]})
        print(f"conv done, n={len(rows['v2'])}", flush=True)
    print(f"\n{'setting':8s} n    all@40  all@150  multi@40  multi@150  median s (cached)")
    for name in SETTINGS:
        rs = rows[name]
        mh = [r for r in rs if r["cat"] == 1]
        print(f"{name:8s} {len(rs):4d} {100 * np.mean([r[40] for r in rs]):7.1f} {100 * np.mean([r[150] for r in rs]):8.1f} "
              f"{100 * np.mean([r[40] for r in mh]):9.1f} {100 * np.mean([r[150] for r in mh]):10.1f} "
              f"{statistics.median(times[name]):9.2f}", flush=True)
    print("CASCADE_DEV_DONE")


if __name__ == "__main__":
    main()
