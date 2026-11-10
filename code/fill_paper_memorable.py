#!/usr/bin/env python3
"""Fill the BEAM placeholders in paper-memorable from the one-judge result files. Reads result files only; prints every replacement.
    .venv/bin/python fill_paper_memorable.py
"""
import json, random, re
from collections import defaultdict
from pathlib import Path
R = Path("results"); PAPER = Path("../paper-memorable/sections")
def load(p):
    rows = defaultdict(list)
    for l in open(p):
        r = json.loads(l); m = re.match(r"(?:1M_)?(\d+)_", r["qid"]); rows[int(m.group(1))].append((r["ability"], bool(r["passed"])))
    return rows
def rate(rows, ab=None):
    xs = [p for c in rows for a, p in rows[c] if ab is None or a == ab]; return 100 * sum(xs) / max(1, len(xs))
A = load(R / "claude-judge-beam-ours-beam-20261108-final-v2-all.jsonl"); B = load(R / "claude-judge-beam-mem0-beam_1m_results.jsonl"); C = load(R / "claude-judge-beam-ours-beam-20261108-173414-a47.jsonl")
assert sum(len(v) for v in A.values()) == 700
rnd = random.Random(0); ca, cb = sorted(A), sorted(B); gaps = []
for _ in range(2000):
    sa = [rnd.choice(ca) for _ in ca]; sb = [rnd.choice(cb) for _ in cb]
    gaps.append(rate({i: A[c] for i, c in enumerate(sa)}) - rate({i: B[c] for i, c in enumerate(sb)}))
gaps.sort(); lo, hi = gaps[50], gaps[1949]; full, mem0, base = rate(A), rate(B), rate(C); gap = full - mem0
verdict = "a lead that lies outside its interval" if lo > 0 else ("a loss" if hi < 0 else "a difference inside its interval, a tie")
ABIL = [("abstention", "Abstention"), ("contradiction_resolution", "Contradiction resolution"), ("event_ordering", "Event ordering"),
        ("information_extraction", "Information extraction"), ("instruction_following", "Instruction following"), ("knowledge_update", "Knowledge update"),
        ("multi_session_reasoning", "Multi-session reasoning"), ("preference_following", "Preference following"), ("summarization", "Summarization"),
        ("temporal_reasoning", "Temporal reasoning")]
per = {k: (rate(B, k), rate(C, k), rate(A, k)) for k, _ in ABIL}
print(f"full {full:.1f} mem0 {mem0:.1f} base {base:.1f} gap {gap:+.1f} [{lo:+.1f}, {hi:+.1f}] -> {verdict}")
for k, n in ABIL: print(f"  {n:26s} mem0 {per[k][0]:5.1f} base {per[k][1]:5.1f} full {per[k][2]:5.1f}")
tex = lambda x: f"{x:+.1f}".replace("-", "$-$")
def sub(path, pairs):
    s = (PAPER / path).read_text()
    for old, new in pairs:
        assert old in s, (path, old[:50]); s = s.replace(old, new); print(f"  {path}: {old[:38]!r} -> {new[:50]!r}")
    (PAPER / path).write_text(s)
cr = per["contradiction_resolution"][2]
sub("abstract.tex", [(r"passes \pending{XX.X}\% of questions and Mem0 passes 63.9\%", f"passes {full:.1f}\\% of questions and Mem0 passes 63.9\\%"),
                     (r"resolve contradictions at \pending{XX.X}\% against", f"resolve contradictions at {cr:.1f}\\% against")])
sub("intro.tex", [(r"Memorable passes \pending{XX.X}\% and Mem0 63.9\%", f"Memorable passes {full:.1f}\\% and Mem0 63.9\\%")])
sub("conclusion.tex", [(r"it passes \pending{XX.X}\% of questions", f"it passes {full:.1f}\\% of questions")])
s = (PAPER / "results.tex").read_text()
s = s.replace(r"The full method passes \pending{XX.X}\%, \pending{+X.X [lo, hi]} against Mem0", f"The full method passes {full:.1f}\\%, {tex(gap)} [{tex(lo)}, {tex(hi)}] against Mem0, {verdict}")
for k, n in ABIL:
    m, b, f = per[k]; best = max(m, b, f); cell = lambda v: (r"\ours{%.1f}" % v) if v == best else ("%.1f" % v)
    s, c = re.subn(re.escape(n) + r" & [^\n]*\\pending\{xx\.x\} \\\\", (f"{n} & {cell(m)} & {cell(b)} & {cell(f)} " + r"\\").replace("\\", "\\\\"), s); assert c == 1, n
s = s.replace(r"All 700 & 63.9 & 64.7 & \pending{xx.x} \\", f"All 700 & 63.9 & 64.7 & {full:.1f} " + r"\\")
(PAPER / "results.tex").write_text(s); print("  results.tex: table and sentence filled")
s = (PAPER / "appendix.tex").read_text()
s = s.replace(r"\pending{--}; 73.9; 80.0; 65.7; \pending{--}; \$34.82; 1.77~MB", f"{full:.1f}; 73.9; 80.0; 65.7; {per['abstention'][2]:.1f}; \\$34.82; 1.77~MB")
s = s.replace(r"\pending{--}; missed, low; held; held; \pending{--}; held; missed, high",
              f"{'held' if full >= 68 else 'missed, low'}; missed, low; held; held; {'held' if abs(per['abstention'][2] - 57.1) <= 5 else 'missed'}; held; missed, high")
(PAPER / "appendix.tex").write_text(s); print("  appendix.tex: prediction row filled")
left = sum((PAPER / f).read_text().count(r"\pending{") for f in ["abstract.tex", "intro.tex", "results.tex", "appendix.tex", "conclusion.tex", "method.tex"])
print("placeholders left:", left)
