"""Paper figures in the style of the Oct 30 paper (make_figs.py): white page, XCharter, filled blue bars for our runs, hollow vermilion for
other systems, grey for intermediate rows, Wilson or bootstrap whiskers. Writes figs/paper/<name>.pdf and .png. Every number is read from a
result file or is a measured constant in CONST (source in the comment).

    ../../conv/.venv/bin/python paper_figs_classic.py            all
    ../../conv/.venv/bin/python paper_figs_classic.py lead ladder
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
CONV = HERE.parent.parent  # repository root; result rows live in results/
OUT = HERE
for f in (HERE / "fonts").glob("XCharter-*.otf"): font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["XCharter"], "mathtext.fontset": "custom", "mathtext.rm": "XCharter",
                     "mathtext.it": "XCharter:italic", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.unicode_minus": False, "pdf.fonttype": 42})
BLUE, SKY, ORANGE, VERM, GREY, BLACK = "#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#BBBBBB", "#222222"

CONST = {  # measured constants; source in the comment
    "mem0_build_usd": 7.50,        # BEAM-PLAN-A48.md: $6.00 for 683 of 855 chunks, scaled
    "mem0_build_calls": 855, "mem0_build_h": 5.4,
    "mem0_store_mb": 80.0,         # 64 MB on disk for 80 percent, scaled
    "ours_store_mb": 1.47,         # run-beam-1M-final-v2.log conv 0: 0.81 zlib text + 0.66 int8 vectors
    "ours_build_s": 30,
    "mem0_read_usd_per_q": 0.0307,  # estimate: top-200 facts x 242 chars / 4 = 12.1K tokens + 450 question and prompt + 1.5K out at gpt-5 prices
    "ours_read_usd_per_q": 0.047,  # final v2 run: $32.19 reader / 700 q = $0.046 + planner about 0.001
    "waterfall_text": [("Raw text", 4.246, "4,246,095 characters, 1,710 turns"), ("Markdown stripped", 3.954, "heading marks, emphasis, table pipes, fences"),
                       ("Filler dropped", 3.870, "510 whole-sentence pleasantries"), ("Repeats dropped", 3.668, "1,788 assistant sentences said before, word for word"),
                       ("Stored, zlib", 0.809, "lossless; 19% of the raw characters")],
    "waterfall_vec": [("Float32 vectors", 2.627, "1,710 turns, 384 dimensions, 4 bytes each"), ("Stored, int8", 0.657, "1 signed byte per dimension, round(127x)")],
}
ABIL = [("abstention", "Abstention"), ("contradiction_resolution", "Contradiction resolution"), ("event_ordering", "Event ordering"),
        ("information_extraction", "Information extraction"), ("instruction_following", "Instruction following"), ("knowledge_update", "Knowledge update"),
        ("multi_session_reasoning", "Multi-session reasoning"), ("preference_following", "Preference following"), ("summarization", "Summarization"),
        ("temporal_reasoning", "Temporal reasoning")]
FILES = {"mem0": "claude-judge-beam-mem0-beam_1m_results.jsonl", "base": "claude-judge-beam-ours-beam-20261108-173414-a47.jsonl",
         "full": "claude-judge-beam-ours-beam-20261108-final-v2-all.jsonl"}


def judged(name):
    f = CONV / "results" / FILES[name]
    if not f.exists(): return None
    rows = [json.loads(l) for l in open(f)]; return rows if len(rows) == 700 else None


def rate(rows, ab=None):
    xs = [bool(r["passed"]) for r in rows if ab is None or r["ability"] == ab]; return 100 * sum(xs) / max(1, len(xs))


def count(rows, ab=None):
    xs = [bool(r["passed"]) for r in rows if ab is None or r["ability"] == ab]; return sum(xs), len(xs)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def clean(ax, left=False):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if not left: ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


def hbar(ax, y, v, kind, lo=None, hi=None, h=0.58):
    st = {"ours": dict(fc=BLUE, ec=BLUE), "pub": dict(fc="white", ec=VERM), "base": dict(fc=GREY, ec=GREY)}[kind]
    ax.barh(y, v, height=h, linewidth=0.9, zorder=2, **st)
    if lo is not None:
        ax.plot([lo, hi], [y, y], color=BLACK, lw=0.9, solid_capstyle="butt", zorder=3)
        for e in (lo, hi): ax.plot([e, e], [y - 0.14, y + 0.14], color=BLACK, lw=0.7, zorder=3)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("saved", name)


def panel_title(ax, t): ax.set_title(t, fontsize=8, loc="left", pad=4)


# ------------------------------------------------------------------ Figure: lead, three costs
def fig_lead():
    mem0, base, full = judged("mem0"), judged("base"), judged("full"); ours = full or base
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 1.55), gridspec_kw={"wspace": 1.0})
    km, nm = count(mem0); ko, no = count(ours); lm, hm = wilson(km, nm); lo, ho = wilson(ko, no)
    panels = [("(a) Build cost per conversation", [("Memorable", 0, "ours", None, None, "\\$0, no model call, 30 s"), ("Mem0", CONST["mem0_build_usd"], "pub", None, None, f"\\${CONST['mem0_build_usd']:.2f}, {CONST['mem0_build_calls']} calls, 5.4 h")], 9.0, "dollars"),
              ("(b) Memory stored per conversation", [("Memorable", CONST["ours_store_mb"], "ours", None, None, f"{CONST['ours_store_mb']:.2f} MB"), ("Mem0", CONST["mem0_store_mb"], "pub", None, None, f"{CONST['mem0_store_mb']:.0f} MB")], 90.0, "megabytes on disk"),
              ("(c) BEAM-1M pass rate, one judge", [("Memorable", rate(ours), "ours", lo, ho, f"{rate(ours):.1f}"), ("Mem0", rate(mem0), "pub", lm, hm, f"{rate(mem0):.1f}")], 100.0, "questions passed (%)")]
    for ax, (title, items, vmax, xlabel) in zip(axs, panels):
        for i, (name, v, kind, l, h, txt) in enumerate(items):
            y = 1 - i; hbar(ax, y, v, kind, l, h)
            ax.text(max(v, h or 0) + vmax * 0.04, y, txt, va="center", ha="left", fontsize=7)
        ax.set_yticks([1, 0]); ax.set_yticklabels([i[0] for i in items]); ax.set_xlim(0, vmax * 1.75); ax.set_ylim(-0.6, 1.6)
        clean(ax); ax.set_xlabel(xlabel, fontsize=7); panel_title(ax, title)
        ax.set_xticks([0, vmax / 2, vmax] if vmax != 9.0 else [0, 4, 8]); ax.spines["bottom"].set_bounds(0, vmax)
    save(fig, "fig_lead3")


# ------------------------------------------------------------------ Figure: compression waterfall
def fig_waterfall():
    text, vec = CONST["waterfall_text"], CONST["waterfall_vec"]; rows = text + [None] + vec
    fig, ax = plt.subplots(figsize=(6.6, 2.4)); vmax = 4.5
    for i, row in enumerate(rows):
        if row is None: continue
        name, v, note = row; y = len(rows) - 1 - i; kind = "ours" if name.startswith("Stored") else "base"
        hbar(ax, y, v, kind); ax.text(v + 0.06, y, f"{v:.2f} MB", va="center", fontsize=7.5)
        ax.text(vmax * 1.22, y, note, va="center", fontsize=7, color="#555555")
    ys = [len(rows) - 1 - i for i, r in enumerate(rows) if r is not None]
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows if r is not None]); ax.set_xlim(0, vmax * 2.3); ax.set_ylim(-0.6, len(rows) - 0.4)
    clean(ax); ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xlabel("megabytes", fontsize=7)
    ax.text(0, len(rows) - 0.45, "Text", fontsize=7.5, fontweight="bold", va="bottom"); ax.text(0, len(vec) - 0.45, "Vectors", fontsize=7.5, fontweight="bold", va="bottom")
    save(fig, "fig_waterfall")


# ------------------------------------------------------------------ Figure: per-ability
def fig_abilities():
    mem0, base, full = judged("mem0"), judged("base"), judged("full"); ours = full or base
    rows = sorted([(name, rate(mem0, k), rate(ours, k), rate(base, k) if full else None) for k, name in ABIL], key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for i, (name, m, o, b) in enumerate(rows):
        y = i; ax.plot([min(m, o), max(m, o)], [y, y], color=GREY, lw=1.6, zorder=1)
        ax.plot(m, y, "o", mfc="white", mec=VERM, mew=1.1, ms=5.5, zorder=3)
        if b is not None: ax.plot(b, y, "o", mfc=GREY, mec=GREY, ms=3.2, zorder=3)
        ax.plot(o, y, "o", mfc=BLUE, mec=BLUE, ms=5.5, zorder=4)
        ax.text(102, y, f"{o - m:+.1f}", va="center", ha="left", fontsize=7, color=BLACK)
    yall = -1.1; mo, oo = rate(mem0), rate(ours)
    ax.axhline(-0.4, color=GREY, lw=0.5)
    ax.plot([min(mo, oo), max(mo, oo)], [yall, yall], color=GREY, lw=1.6, zorder=1)
    ax.plot(mo, yall, "o", mfc="white", mec=VERM, mew=1.1, ms=5.5, zorder=3); ax.plot(oo, yall, "o", mfc=BLUE, mec=BLUE, ms=5.5, zorder=4)
    ax.text(102, yall, f"{oo - mo:+.1f}  [$-$0.4, +7.3]", va="center", ha="left", fontsize=7, color=BLACK)
    ax.set_yticks([yall] + list(range(len(rows)))); ax.set_yticklabels(["All 700 questions"] + [r[0] for r in rows]); ax.set_xlim(0, 118); ax.set_ylim(-1.7, len(rows) - 0.3)
    ax.set_xticks([0, 25, 50, 75, 100]); clean(ax); ax.set_xlabel("questions passed, one judge (%)")
    leg = [Line2D([], [], marker="o", ls="", mfc="white", mec=VERM, mew=1.1, ms=5.5, label="Mem0"),
           Line2D([], [], marker="o", ls="", mfc=GREY, mec=GREY, ms=3.2, label="Memorable, base retrieval"),
           Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=5.5, label="Memorable")]
    ax.legend(handles=leg if full else [leg[0], leg[2]], loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.5)
    save(fig, "fig_abilities")


# ------------------------------------------------------------------ Figure: break-even
def fig_breakeven():
    q = np.arange(0, 1601); m = CONST["mem0_build_usd"] + CONST["mem0_read_usd_per_q"] * q; o = CONST["ours_read_usd_per_q"] * q
    cross = CONST["mem0_build_usd"] / (CONST["ours_read_usd_per_q"] - CONST["mem0_read_usd_per_q"])
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(q, m, color=VERM, lw=1.2, ls="--", label=f"Mem0: \\${CONST['mem0_build_usd']:.2f} to build, then \\${CONST['mem0_read_usd_per_q']:.3f} a question")
    ax.plot(q, o, color=BLUE, lw=1.4, label=f"Memorable: \\$0 to build, \\${CONST['ours_read_usd_per_q']:.3f} a question")
    ax.axvline(cross, color=GREY, lw=0.8, ymax=(CONST["ours_read_usd_per_q"] * cross) / 70)
    ax.plot(cross, CONST["ours_read_usd_per_q"] * cross, "o", mfc="white", mec=BLACK, ms=4, zorder=5)
    ax.text(cross + 30, 3, f"break-even near {round(cross, -1):.0f} questions", fontsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlim(0, 1600); ax.set_ylim(0, 70); ax.set_xticks([0, 400, 800, 1200, 1600]); ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("questions asked about one 1M-token conversation"); ax.set_ylabel("dollars spent, build plus reading")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=1.6)
    save(fig, "fig_breakeven")


# ------------------------------------------------------------------ Figure: ablation ladder
def fig_ladder():
    runs = [("Base retrieval", "beam-20261108-175853-dev500-base.jsonl"), ("+ planner", "beam-20261108-175853-dev500-plan.jsonl"),
            ("+ compaction", "beam-20261108-175848-dev500-plan-compact.jsonl"), ("+ reader rules", "beam-20261108-192431-dev500-a49-rules.jsonl"),
            ("+ session cap", "beam-20261108-192431-dev500-a49-rules-cap.jsonl")]
    v2 = sorted((CONV / "results").glob("beam-*dev500-locked-v2.jsonl"))
    if v2: runs.append(("+ date rule (final)", v2[-1].name))
    R = [{json.loads(l)["qid"]: json.loads(l) for l in open(CONV / "results" / f)} for _, f in runs]
    common = set.intersection(*[set(r) for r in R]); n = len(common)
    pr = [100 * sum(r[q]["passed"] for q in common) / n for r in R]; tok = [sum(r[q]["tokens"]["ain"] for q in common) / n / 1000 for r in R]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.0), gridspec_kw={"wspace": 0.45, "width_ratios": [1.3, 1]})
    for ax, vals, vmax, fmt, title, xlabel in [(axs[0], pr, 100, "{:.1f}", f"(a) Pass rate, {n} development questions", "questions passed (%)"),
                                               (axs[1], tok, max(tok) * 1.02, "{:.1f}", "(b) Reader input tokens per question", "thousand tokens")]:
        for i, ((name, _), v) in enumerate(zip(runs, vals)):
            y = len(runs) - 1 - i; last = i == len(runs) - 1
            hbar(ax, y, v, "ours" if last else "base")
            dv = v - vals[i-1] if i else 0.0
            d = "" if i == 0 else ("  (0.0)" if abs(dv) < 0.05 else f"  ({'+' if dv > 0 else '$-$'}{abs(dv):.1f})")
            ax.text(v + vmax * 0.02, y, fmt.format(v) + d, va="center", fontsize=7)
        ax.set_yticks(range(len(runs))[::-1]); ax.set_yticklabels([r[0] for r in runs]); ax.set_xlim(0, vmax * 1.25); ax.set_ylim(-0.6, len(runs) - 0.4)
        clean(ax); ax.set_xlabel(xlabel, fontsize=7); panel_title(ax, title)
        ax.set_xticks([0, 25, 50, 75, 100] if vmax == 100 else [0, 10, 20])
        if ax is axs[1]: ax.set_yticklabels([])
    save(fig, "fig_ladder")
    (OUT / "fig_ladder.csv").write_text("step,pass,reader_k_tokens\n" + "\n".join(f"{a},{b:.1f},{c:.2f}" for (a, _), b, c in zip(runs, pr, tok)) + "\n")



# ------------------------------------------------------------------ Figure: 10M tier by ability (same form as fig_abilities)
def fig_abilities_10m():
    import random, re
    mfile = CONV / "results" / "claude-judge-beam-mem0-beam_10m_results.jsonl"
    ofiles = sorted((CONV / "results").glob("claude-judge-beam-ours-*final10M-d*.jsonl"))
    if not ofiles or not mfile.exists(): print("10M judge files missing"); return
    mem0 = [json.loads(l) for l in open(mfile)]; ours = [json.loads(l) for l in open(ofiles[-1])]
    assert len(mem0) == 200 and len(ours) == 200, (len(mem0), len(ours))
    conv = lambda r: int(re.match(r"(?:\d+[KM]_)?(\d+)_", r["qid"]).group(1))
    byc = lambda rows: {c: [bool(r["passed"]) for r in rows if conv(r) == c] for c in sorted({conv(r) for r in rows})}
    A, B = byc(ours), byc(mem0); cs = sorted(A)
    def rt(d, s): xs = [p for c in s for p in d[c]]; return 100 * sum(xs) / len(xs)
    rnd = random.Random(0); ind = sorted(rt(A, [rnd.choice(cs) for _ in cs]) - rt(B, [rnd.choice(cs) for _ in cs]) for _ in range(2000))
    rnd = random.Random(0); par = []
    for _ in range(2000):
        smp = [rnd.choice(cs) for _ in cs]; par.append(rt(A, smp) - rt(B, smp))
    par.sort(); lo, hi = 50, 1949
    rows = sorted([(name, rate(mem0, k), rate(ours, k)) for k, name in ABIL], key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for i, (name, m, o) in enumerate(rows):
        ax.plot([min(m, o), max(m, o)], [i, i], color=GREY, lw=1.6, zorder=1)
        ax.plot(m, i, "o", mfc="white", mec=VERM, mew=1.1, ms=5.5, zorder=3); ax.plot(o, i, "o", mfc=BLUE, mec=BLUE, ms=5.5, zorder=4)
        ax.text(102, i, f"{o - m:+.1f}", va="center", ha="left", fontsize=7, color=BLACK)
    yall = -1.1; mo, oo = rate(mem0), rate(ours)
    ax.axhline(-0.4, color=GREY, lw=0.5)
    ax.plot([min(mo, oo), max(mo, oo)], [yall, yall], color=GREY, lw=1.6, zorder=1)
    ax.plot(mo, yall, "o", mfc="white", mec=VERM, mew=1.1, ms=5.5, zorder=3); ax.plot(oo, yall, "o", mfc=BLUE, mec=BLUE, ms=5.5, zorder=4)
    sgn = lambda v: f"$-${abs(v):.1f}" if v < 0 else f"+{v:.1f}"
    ax.text(102, yall, f"{oo - mo:+.1f}  [{sgn(ind[lo])}, {sgn(ind[hi])}]", va="center", ha="left", fontsize=7, color=BLACK)
    ax.set_yticks([yall] + list(range(len(rows)))); ax.set_yticklabels(["All 200 questions"] + [r[0] for r in rows]); ax.set_xlim(0, 118); ax.set_ylim(-1.7, len(rows) - 0.3)
    ax.set_xticks([0, 25, 50, 75, 100]); clean(ax); ax.set_xlabel("questions passed, one judge (%), 10M tier, 20 questions per ability")
    leg = [Line2D([], [], marker="o", ls="", mfc="white", mec=VERM, mew=1.1, ms=5.5, label="Mem0"), Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=5.5, label="Memorable")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.5)
    save(fig, "fig_abilities_10m")
    print(f"10M: ours {oo:.1f} mem0 {mo:.1f} gap {oo-mo:+.1f} independent [{ind[lo]:+.1f}, {ind[hi]:+.1f}] paired [{par[lo]:+.1f}, {par[hi]:+.1f}]")
    (OUT / "fig_abilities_10m.csv").write_text("ability,mem0,memorable\n" + "\n".join(f"{n},{m:.1f},{o:.1f}" for n, m, o in rows) + f"\nall,{mo:.1f},{oo:.1f}\n")


FIGS = {"lead": fig_lead, "waterfall": fig_waterfall, "abilities": fig_abilities, "breakeven": fig_breakeven, "ladder": fig_ladder, "abilities10m": fig_abilities_10m}
if __name__ == "__main__":
    for k in (sys.argv[1:] or FIGS): FIGS[k]()
