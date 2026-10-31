#!/usr/bin/env python3
"""Cold wall time per LoCoMo question for four rerank schedules, with the cross-encoder score cache bypassed.

    ~/Documents/GitHub/snapgraph/.venv/bin/python time_schedules.py [n_questions]

Models are loaded once (warm), every cross-encoder pair is scored fresh for every question, nothing is saved to the cache.
Settings as in cascade_dev.py. Also times the first stage alone (BM25, dense, contiguity, speaker prior). Labels are not read.
"""
import json, os, re, statistics, sys, time
import numpy as np
SG = os.path.expanduser("~/Documents/GitHub/snapgraph")
sys.path.insert(0, os.path.join(SG, "src"))
os.chdir(SG)
import load, loc, rerank  # noqa: E402
from run_turn import z  # noqa: E402
from rankers import rank_desc  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cascade_dev import strip_names, both, bge_only, cascade, place  # noqa: E402,F401

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 40
cfg = json.load(open("configs/final.json"))["locomo"]["arms"]["nograph_ens"]


def cold():
    for name in ("msmarco", "bge-v2m3"):
        rr = rerank.get(name)
        rr.mem = {}
        rr.dirty = 0


def first(conv, qt, names_out):
    qs = strip_names(conv, qt) if names_out else qt
    s = conv.fused(qs, cfg["fuse"])
    return loc.speaker_prior(s, conv, qt, cfg)


SETTINGS = [
    ("first stage only", False, None),
    ("shipped, both over 100", False, lambda s, c, q: both(s, c, q, 100)),
    ("v2, both over 300", True, lambda s, c, q: both(s, c, q, 300)),
    ("cascade, MiniLM 300 then bge 100", True, lambda s, c, q: cascade(s, c, q)),
    ("bge alone over 300", True, lambda s, c, q: bge_only(s, c, q, 300)),
]

sysd, scd = load.load_locomo("dev")
qs = []
for cs, scn in zip(sysd, scd):
    conv = loc.Conv(cs, True)
    for q, lab in zip(cs["questions"], scn["qa"]):
        if lab["category"] in (1, 2, 3, 4):
            qs.append((conv, q["question"]))
qs = qs[:: max(1, len(qs) // NQ)][:NQ]
print(f"{len(qs)} dev questions, cache bypassed, models warm", flush=True)
for conv, qt in qs[:1]:  # warm the models and the first-stage caches for this question only
    cold(); both(first(conv, qt, True), conv, qt, 300)
for name, names_out, fn in SETTINGS:
    ts = []
    for conv, qt in qs:
        conv._f = {}  # first-stage cache off too
        cold()
        t0 = time.perf_counter()
        s = first(conv, qt, names_out)
        if fn is not None:
            fn(s, conv, qt)
        ts.append(time.perf_counter() - t0)
    ts_sorted = sorted(ts)
    print(f"{name:34s} median {statistics.median(ts):.2f}s  mean {statistics.mean(ts):.2f}s  p95 {ts_sorted[int(0.95 * (len(ts) - 1))]:.2f}s", flush=True)
print("TIME_SCHEDULES_DONE")
