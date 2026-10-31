"""Figures for the combined paper. Every plotted value is read from a file and written to a CSV beside the figure.

    ../../../conv/.venv/bin/python make_figs.py            all figures
    ../../../conv/.venv/bin/python make_figs.py beat grid  a subset

Inputs: data/*.csv (copies of the two source papers' figure data), conv/results/*.jsonl (LoCoMo and LongMemEval answer runs).
fig_tree, fig_grid and fig_prune belong to an earlier draft and are not used by the paper.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
V2 = HERE.parent.parent
CONV = V2 / "conv"
BLUE, SKY, ORANGE, VERM, GREY, BLACK = "#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#BBBBBB", "#222222"
from matplotlib import font_manager
for _f in (Path(__file__).resolve().parent / "fonts").glob("XCharter-*.otf"):
    font_manager.fontManager.addfont(str(_f))
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["XCharter"], "mathtext.fontset": "custom", "mathtext.rm": "XCharter",
    "mathtext.it": "XCharter:italic", "mathtext.bf": "XCharter:bold", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
})
COL = 3.37
WIDE = 7.0
DEV = {"conv-26", "conv-30", "conv-41"}
DAG = "$^{\\dagger}$"


def save(fig, name, tight=False):
    kw = dict(bbox_inches="tight", pad_inches=0.02) if tight else {}
    fig.savefig(HERE / f"{name}.pdf", **kw)
    fig.savefig(HERE / f"{name}.png", dpi=200, **kw)
    plt.close(fig)


def dump(name, header, data):
    with open(HERE / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in data:
            w.writerow([(f"{v:.3f}" if abs(v) < 2 else f"{v:.2f}") if isinstance(v, (float, np.floating)) else v for v in r])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def cluster_boot(x, clusters, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    clusters = np.asarray(clusters)
    groups = [x[clusters == c] for c in np.unique(clusters)]
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(draws)]
    return 100 * x.mean(), 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)


def clean(ax, left=False):
    for s in ("top", "right") + (() if left else ("left",)):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))[1:]


def read_lme_csv(path):
    out = []
    for r in read_csv(path):
        out.append([",".join(r[:-8])] + r[-8:])
    return out


def qa(fname, test_only=False):
    rows = [json.loads(l) for l in open(CONV / "results" / fname) if l.strip()]
    if test_only:
        rows = [r for r in rows if r["sample"] not in DEV]
    x = [int(bool(r["correct"])) for r in rows]
    c = [r["sample"] for r in rows]
    v, lo, hi = cluster_boot(x, c)
    turns = np.mean([r["turns"] if "turns" in r else len(r["retrieved"]) for r in rows])
    return dict(acc=v, lo=lo, hi=hi, n=len(rows), turns=float(turns))


def hbar(ax, y, v, kind, lo=None, hi=None, h=0.58):
    """kind: ours (filled blue), pub (hollow vermilion), base (grey)."""
    st = {"ours": dict(fc=BLUE, ec=BLUE), "pub": dict(fc="white", ec=VERM), "base": dict(fc=GREY, ec=GREY)}[kind]
    ax.barh(y, v, height=h, linewidth=0.9, zorder=2, **st)
    if lo is not None:
        ax.plot([lo, hi], [y, y], color=BLACK, lw=0.9, solid_capstyle="butt", zorder=3)
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.14, y + 0.14], color=BLACK, lw=0.7, zorder=3)


# ---- Figure 2: like-for-like comparisons --------------------------------------------------------------------------------------
MEM0 = Path.home() / "Documents/(private data folder, not distributed)/beam-repos/mem0mb/results/platform"
LOCOMO_RAW = CONV / "data" / "locomo10.json"

def locomo_h2h():
    """Our gpt-5-reader rows (Mem0's prompts and judge, 40 turns) joined to Mem0's per-question file on conversation and question text."""
    raw = json.load(open(LOCOMO_RAW))
    idx_of = {c["sample_id"]: i for i, c in enumerate(raw)}
    m = json.load(open(MEM0 / "locomo_results.json"))
    mem = {(e["conversation_idx"], e["question"].strip()): dict(ok=e["cutoff_results"]["top_200"]["judgment"] == "CORRECT", cat=e["category"], conv=e["conversation_idx"]) for e in m["evaluations"]}
    ours = {}
    for fname in ("20261029-183439.jsonl", "20261029-181313.jsonl"):
        for l in open(CONV / "results" / fname):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if "correct" in r:
                ours[(idx_of[r["sample"]], r["question"].strip())] = dict(ok=bool(r["correct"]), cat=r["category"], conv=idx_of[r["sample"]])
    keys = [k for k in ours if k in mem]
    return [dict(ours=ours[k]["ok"], mem0=mem[k]["ok"], cat=ours[k]["cat"], conv=ours[k]["conv"]) for k in keys], mem


def fig_beat():
    h2h, mem_all = locomo_h2h()
    lme = [json.loads(l) for l in open(CONV / "results" / "lme-20261029-182856.jsonl") if l.strip()]
    lme = [r for r in lme if "correct" in r]
    kl = sum(int(bool(r["correct"])) for r in lme)
    mlme = json.load(open(MEM0 / "longmemeval_results.json"))["metrics_by_cutoff"]["top_200"]["overall"]
    beam = [json.loads(l) for l in open(CONV / "results" / "beam-20261029-171606-full-1M-gpt5.jsonl") if l.strip()]
    kb = sum(int(bool(r["passed"])) for r in beam)
    mbeam = json.load(open(MEM0 / "beam_1m_results.json"))["metrics_by_cutoff"]["top_200"]["overall"]
    out = []
    fig = plt.figure(figsize=(WIDE, 3.45))
    pos = {}
    for r, y0 in enumerate((0.63, 0.20)):
        for c in range(3):
            pos[(r, c)] = [0.15 + c * 0.33, y0, 0.15, 0.27]

    def panel(r, c, title, w_ext=0.165):
        p = pos[(r, c)]
        ax = fig.add_axes([p[0], p[1], w_ext, p[3]])
        ax.set_title(title, fontsize=7.6, pad=3, loc="left", x=-0.85)
        clean(ax)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
        return ax

    def rows_(ax, items, xmax, ticks, fmtv, xlabel):
        n = len(items)
        for i, (lab, v, kind, lo, hi) in enumerate(items):
            y = n - 1 - i
            hbar(ax, y, v, kind, lo, hi)
            ax.text(max(hi if hi is not None else v, 0.985 if (xmax == 1.25 and i == 0) else 0) + xmax * 0.02, y, fmtv(v), va="center", ha="left", fontsize=7)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)
        ax.set_yticks(range(n))
        ax.set_yticklabels([it[0] for it in items][::-1], fontsize=6.5, linespacing=1.05)
        ax.set_xlabel(xlabel, labelpad=1, fontsize=7)

    # (a) Proced-Mem
    ax = panel(0, 0, "(a) Proced-Mem, MAP of 40 queries")
    passes = [0.954, 0.949, 0.968]
    a_items = [("Ours, code only\n(gpt-5 judge)", 0.957, "ours", None, None), ("MiniLM baseline\n(gpt-5 judge)", 0.796, "base", None, None)]
    rows_(ax, a_items, 1.25, [0, 0.5, 1.0], lambda v: f"{v:.3f}", "mean average precision")
    ax.plot(passes, [1] * 3, "o", ms=2.2, color="white", mec=BLACK, mew=0.6, zorder=4)
    out += [("Proced-Mem", "ours", 0.957, ""), ("Proced-Mem", "MiniLM baseline", 0.796, "")]
    # (b) LoCoMo, same reader, judge and prompts, same questions
    ax = panel(0, 1, "(b) LoCoMo, %d q. both answered" % len(h2h))
    ov, olo, ohi = cluster_boot([r["ours"] for r in h2h], [r["conv"] for r in h2h])
    mv, mlo, mhi = cluster_boot([r["mem0"] for r in h2h], [r["conv"] for r in h2h])
    h2h_t = [r for r in h2h if r["conv"] >= 3]
    tv, tlo, thi = cluster_boot([r["ours"] for r in h2h_t], [r["conv"] for r in h2h_t])
    mtv, mtlo, mthi = cluster_boot([r["mem0"] for r in h2h_t], [r["conv"] for r in h2h_t])
    b_items = [("Ours, 40 turns$^{\\ddagger}$\n(%d q., 385 dev)" % len(h2h), ov, "ours", olo, ohi),
               ("Mem0, current\n(same %d)" % len(h2h), mv, "pub", mlo, mhi),
               ("Ours, 40 turns\n(%d test q.)" % len(h2h_t), tv, "ours", tlo, thi),
               ("Mem0, current\n(same %d)" % len(h2h_t), mtv, "pub", mtlo, mthi)]
    rows_(ax, b_items, 125, [0, 50, 100], lambda v: f"{v:.1f}", "answer accuracy (%)")
    out += [("LoCoMo like-for-like", i[0].replace("\n", " "), i[1], f"{i[3]:.1f}-{i[4]:.1f}" if i[3] else "") for i in b_items]
    # (c) LongMemEval-S session retrieval, every gold session in the top 10, all 470 questions, no reader
    ax = panel(0, 2, "(c) LongMemEval$_{\\mathrm{S}}$ sessions, all gold in top 10")
    c_items = []
    for lab, v, kind in (("Ours, no reader$^{\\ddagger}$\n(470 q.)", 97.45, "ours"), ("MemGAS" + DAG + "\n(470 q.)", 94.47, "pub"), ("EdgeMem" + DAG + "\n(470 q.)", 93.83, "pub")):
        lo, hi = wilson(round(v / 100 * 470), 470)
        c_items.append((lab, v, kind, lo, hi))
    rows_(ax, c_items, 125, [0, 50, 100], lambda v: f"{v:.1f}", "questions (%)")
    out += [("LongMemEval-S sessions top 10", i[0].replace("\n", " "), i[1], f"{i[3]:.1f}-{i[4]:.1f}") for i in c_items]
    # (d) LongMemEval-S answers, GPT-5 reader
    ax = panel(1, 0, "(d) LongMemEval$_{\\mathrm{S}}$ answers, GPT-5 reads")
    lo, hi = wilson(kl, len(lme)); mlo2, mhi2 = wilson(mlme["passed"], mlme["total"])
    d_items = [("Ours, 100 turns\n(%d test q.)" % len(lme), 100 * kl / len(lme), "ours", lo, hi),
               ("Mem0, current\n(500 q.)", mlme["pass_rate"], "pub", mlo2, mhi2)]
    rows_(ax, d_items, 125, [0, 50, 100], lambda v: f"{v:.1f}", "answer accuracy (%)")
    out += [("LongMemEval-S answers", i[0].replace("\n", " "), i[1], f"{i[3]:.1f}-{i[4]:.1f}") for i in d_items]
    # (e) BEAM 1M
    ax = panel(1, 1, "(e) BEAM 1M, GPT-5 reads, pass rate")
    lo, hi = wilson(kb, len(beam)); blo, bhi = wilson(mbeam["correct"], mbeam["total"])
    e_items = [("Ours, 20 then 40 turns\n(%d q.)" % len(beam), 100 * kb / len(beam), "ours", lo, hi),
               ("Mem0, current\n(700 q.)", mbeam["accuracy"], "pub", blo, bhi)]
    rows_(ax, e_items, 125, [0, 50, 100], lambda v: f"{v:.1f}", "questions passed (%)")
    out += [("BEAM 1M", i[0].replace("\n", " "), i[1], f"{i[3]:.1f}-{i[4]:.1f}") for i in e_items]
    # (f) LoCoMo by category, same questions
    ax = panel(1, 2, "(f) LoCoMo by category, same %d q." % len(h2h))
    cats = [(4, "single-hop"), (3, "open-domain"), (2, "temporal"), (1, "multi-hop")]
    for yi, (c, name) in enumerate(cats):
        rs = [r for r in h2h if r["cat"] == c]
        o = 100 * np.mean([r["ours"] for r in rs]); mm = 100 * np.mean([r["mem0"] for r in rs])
        ax.barh(yi + 0.19, o, height=0.34, fc=BLUE, ec=BLUE, zorder=2)
        ax.barh(yi - 0.19, mm, height=0.34, fc="white", ec=VERM, lw=0.9, zorder=2)
        ax.text(o + 2, yi + 0.19, f"{o:.1f}", va="center", fontsize=6.2)
        ax.text(mm + 2, yi - 0.19, f"{mm:.1f}", va="center", fontsize=6.2)
        out += [("LoCoMo by category", f"ours {name} (n={len(rs)})", o, ""), ("LoCoMo by category", f"Mem0 {name} (n={len(rs)})", mm, "")]
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"{name}\n(n={sum(1 for r in h2h if r['cat'] == c)})" for c, name in cats], fontsize=6.5, linespacing=1.05)
    ax.set_ylim(-0.6, len(cats) - 0.4)
    ax.set_xlim(0, 125); ax.set_xticks([0, 50, 100])
    ax.set_xlabel("answer accuracy (%)", labelpad=1, fontsize=7)
    handles = [Rectangle((0, 0), 1, 1, fc=BLUE, ec=BLUE, label="our run, no model writes the memory"),
               Rectangle((0, 0), 1, 1, fc="white", ec=VERM, lw=0.9, label="Mem0 from its published result files; " + DAG + " transcribed from a paper"),
               Rectangle((0, 0), 1, 1, fc=GREY, ec=GREY, label="baseline, or ours with a writer")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3, frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.0, fontsize=6.6)
    dump("fig_beat", ["panel", "system", "value", "interval95"], out)
    save(fig, "fig_beat")


def fig_tokens():
    fig = plt.figure(figsize=(COL, 3.1))
    ax = fig.add_axes([0.40, 0.55, 0.56, 0.40])
    bx = fig.add_axes([0.40, 0.12, 0.56, 0.25])
    read = [("Ours, 10 turns,\nwidened on NOT FOUND", 1263, "ours"), ("Ours, 40 turns", 3191, "ours"), ("Mem0, current" + DAG, 6956, "pub"), ("Mem0, 2025 paper" + DAG, 1764, "pub"), ("Mem0g, 2025 paper" + DAG, 3616, "pub"),
            ("Zep" + DAG, 3911, "pub"), ("Full context" + DAG, 26031, "pub")]
    build = [("Ours", 0, "ours"), ("Mem0, 2025 paper" + DAG, 7000, "pub"), ("Mem0g, 2025 paper" + DAG, 14000, "pub"), ("Zep" + DAG, 600000, "pub")]
    out = []

    def dots(a, items, lo, hi, title, xlabel, fmt):
        n = len(items)
        for i, (lab, v, kind) in enumerate(items):
            y = n - 1 - i
            col = BLUE if kind == "ours" else VERM
            if v == 0:
                a.plot([lo], [y], "o", ms=4.4, mfc=BLUE, mec=BLUE, clip_on=False, zorder=3)
                a.text(lo * 1.25, y, "0, no model call", va="center", fontsize=7)
            else:
                a.plot([lo, v], [y, y], color=col, lw=0.8, zorder=2)
                a.plot([v], [y], "o", ms=4.4, mfc=col if kind == "ours" else "white", mec=col, mew=1.0, zorder=3)
                a.text(v * 1.25, y, fmt(lab, v), va="center", fontsize=7)
            out.append((title, lab.replace("\n", " "), v))
        a.set_xscale("log")
        a.set_xlim(lo, hi)
        a.set_ylim(-0.6, n - 0.4)
        a.set_yticks(range(n))
        a.set_yticklabels([i[0] for i in items][::-1], fontsize=6.8, linespacing=1.05)
        a.set_title(title, fontsize=7.6, loc="left", x=-0.70, pad=3)
        a.set_xlabel(xlabel, labelpad=1, fontsize=7)
        a.xaxis.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
        clean(a)
        a.tick_params(axis="y", length=0)
        a.minorticks_off()

    dots(ax, read, 500, 400000, "(a) Tokens read per question", "answer-model input tokens (log)", lambda l, v: f"{v:,}")
    dots(bx, build, 1000, 5000000, "(b) Memory stored per conversation", "tokens (log)", lambda l, v: ("$>$" if v == 600000 else "$\\sim$") + f"{v:,}")
    dump("fig_tokens", ["panel", "system", "tokens"], out)
    save(fig, "fig_tokens")


# ---- Figure 4: accuracy against turns read --------------------------------------------------------------------------------------
def fig_acc_turns():
    pts = [("5a, 40 turns", "20261029-020923.jsonl", BLUE, "s", True), ("5r, 40 turns", "20261029-022328.jsonl", BLUE, "o", True),
           ("Ours, 10 turns, widened", "20261029-145312.jsonl", BLUE, "D", True), ("Ours with entity graph, widened", "20261029-143122.jsonl", ORANGE, "^", True),
           ("Full context (GPT-6 Sol reader)", "20261029-002936.jsonl", BLACK, "s", False)]
    fig = plt.figure(figsize=(COL, 2.35))
    ax = fig.add_axes([0.14, 0.17, 0.83, 0.78])
    out = []
    hs = []
    off = {"5a, 40 turns": (7, 0, "left"), "5r, 40 turns": (7, 0, "left"), "Ours, 10 turns, widened": (-7, -4, "right"),
           "Ours with entity graph, widened": (-7, 6, "right"), "Full context (GPT-6 Sol reader)": (-7, 0, "right")}
    jit = {"Ours, 10 turns, widened": 1.05, "Ours with entity graph, widened": 0.95}
    for lab, f, col, mk, same in pts:
        s = qa(f, test_only=True)
        mfc = col if same else "white"
        s["turns"] *= jit.get(lab, 1.0)
        ax.plot([s["turns"]] * 2, [s["lo"], s["hi"]], color=col, lw=0.9, zorder=2)
        ax.plot([s["turns"]], [s["acc"]], mk, ms=4.6, mfc=mfc, mec=col, mew=1.0, zorder=3)
        dx, dy, ha = off[lab]
        hs.append(Line2D([], [], marker=mk, ls="", ms=4.6, mfc=mfc, mec=col, mew=1.0, label=lab))
        out.append((lab, s["turns"], s["acc"], s["lo"], s["hi"], s["n"]))
    ax.set_xscale("log")
    ax.set_xlim(7, 1500)
    ax.set_xticks([10, 40, 100, 400, 1000])
    ax.set_xticklabels(["10", "40", "100", "400", "1000"])
    ax.minorticks_off()
    ax.set_ylim(91, 98)
    ax.set_yticks([92, 94, 96, 98])
    ax.set_xlabel("mean turns the answer model reads (log)", labelpad=1)
    ax.set_ylabel("accuracy, 7 test conversations (%)")
    ax.yaxis.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
    clean(ax, left=True)
    ax.legend(handles=hs, loc="lower right", frameon=False, fontsize=6.6, handletextpad=0.3, labelspacing=0.35, borderaxespad=0.2)
    dump("fig_acc_turns", ["system", "mean_turns_test", "acc_test", "lo95", "hi95", "n"], out)
    save(fig, "fig_acc_turns")


# ---- Figure: test recall against turns, shipped ranker against the v2 search ------------------------------------------------
def fig_recall_v2():
    series = {}
    for r in read_csv(DATA / "fig_recall_v2.csv"):
        series.setdefault((r[0], r[1]), []).append((int(r[2]), float(r[3])))
    fig = plt.figure(figsize=(COL, 2.3))
    ax = fig.add_axes([0.15, 0.18, 0.82, 0.74])
    out = []
    for (arm, cat), c, m in ((("shipped", "all"), GREY, "o"), (("v2", "all"), BLUE, "o"), (("shipped", "multi-hop"), GREY, "^"), (("v2", "multi-hop"), BLUE, "^")):
        pts = sorted(p for p in series[(arm, cat)] if p[0] in (10, 20, 40, 60, 100, 150, 200, 300))
        xs, ys = zip(*pts)
        ls = "-" if cat == "all" else "--"
        lab = ("Shipped ranker" if arm == "shipped" else "Names out, 300-deep rerank") + (", all" if cat == "all" else ", multi-hop")
        ax.plot(xs, ys, ls=ls, color=c, lw=1.0, marker=m, ms=3.4, mfc=c if arm == "v2" else "white", mec=c, mew=0.9, label=lab, zorder=3)
        out += [(arm, cat, x, y) for x, y in pts]
    ax.axvline(150, color="#DDDDDD", lw=0.8, zorder=1)
    ax.text(146, 99.5, "150 turns, Mem0's budget", fontsize=6.3, ha="right", va="top", color="#555555")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 40, 100, 150, 300])
    ax.set_xticklabels(["10", "20", "40", "100", "150", "300"])
    ax.minorticks_off()
    ax.set_xlim(8.5, 360)
    ax.set_ylim(30, 101)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xlabel("turns handed to the answer model (log)", labelpad=1)
    ax.set_ylabel("every gold turn retrieved, test (%)")
    ax.yaxis.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
    clean(ax, left=True)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0, handletextpad=0.4, labelspacing=0.3, fontsize=6.2)
    dump("fig_recall_v2", ["arm", "category", "turns", "recall_all_gold_test"], out)
    save(fig, "fig_recall_v2")


# ---- Figure 5: recall against turns -------------------------------------------------------------------------------------------
def fig_recall():
    rows = [r for r in read_csv(DATA / "fig_locomo_recall.csv") if r[3] == ""]
    steps = {}
    for r in rows:
        steps.setdefault(r[0], []).append((int(r[1]), float(r[2])))
    sty = {"BM25 over turns": (GREY, "o", "--"), "+ clean and stem the question": (SKY, "s", "--"),
           "+ bge-small, fused (5a)": (ORANGE, "^", "-"), "+ cross-encoder rerank (5r)": (BLUE, "o", "-")}
    fig = plt.figure(figsize=(WIDE, 2.2))
    ax = fig.add_axes([0.07, 0.17, 0.36, 0.70])
    out = []
    for lab, pts in steps.items():
        c, m, ls = sty[lab]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls=ls, color=c, lw=1.0, marker=m, ms=3.6, mfc=c, mec=c, label=lab, zorder=3)
        out += [("a", lab, x, y) for x, y in pts]
    ax.set_xticks([20, 40, 60])
    ax.set_xlim(16, 64)
    ax.set_ylim(55, 85)
    ax.set_yticks([60, 70, 80])
    ax.set_xlabel("turns handed to the answer model", labelpad=1)
    ax.set_ylabel("every gold turn retrieved (%)")
    ax.yaxis.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
    clean(ax, left=True)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0, handletextpad=0.4, labelspacing=0.3, fontsize=6.6)
    ax.set_title("(a) 1,536 questions, answer-model retrievers", fontsize=7.6, loc="left", pad=3)
    bx = fig.add_axes([0.56, 0.17, 0.41, 0.70])
    ks = (5, 10, 25, 50)
    lab = {"flat": ("Flat hybrid", GREY, "o", "--"), "base_fwd": ("+ forward contiguity", SKY, "s", "--"), "snapgraph_no_rr": ("w/o cross-encoders", VERM, "D", "-"),
           "nograph_ens": ("Ours, graph-free ranker", BLUE, "o", "-"), "snapgraph": ("Ours with entity graph", ORANGE, "^", "-")}
    series = {}
    for r in read_csv(DATA / "B_fig_locomo.csv"):
        series.setdefault(r[0], []).append((int(r[1]), float(r[2])))
    for a, (name, c, m, ls) in lab.items():
        xs, ys = zip(*series[a])
        bx.plot(xs, ys, ls=ls, color=c, lw=1.0, marker=m, ms=3.6, mfc=c if m != "D" else "white", mec=c, mew=0.9, label=name, zorder=3)
        out += [("b", name, x, y) for x, y in series[a]]
    bx.set_xscale("log")
    bx.set_xticks(ks)
    bx.set_xticklabels([str(k) for k in ks])
    bx.minorticks_off()
    bx.set_xlim(4.4, 57)
    bx.set_ylim(50, 95)
    bx.set_yticks(range(50, 96, 10))
    bx.set_xlabel("turns returned, $k$", labelpad=1)
    bx.set_ylabel("share of gold turns in top $k$ (%)")
    bx.yaxis.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
    clean(bx, left=True)
    bx.legend(loc="lower right", frameon=False, handlelength=2.0, handletextpad=0.4, labelspacing=0.3, fontsize=6.6)
    bx.set_title("(b) 1,152 test questions, ranker chains", fontsize=7.6, loc="left", pad=3)
    dump("fig_recall", ["panel", "system", "turns", "value"], out)
    save(fig, "fig_recall")


# ---- Figure 6: an earlier benchmark grid --------------------------------------------------------------------------------------------
TASKS = ["marketing gap analysis", "sales proposal", "partnership contract", "awilix release tier (explicit)", "awilix release tier (implicit)",
         "word2vec pipeline", "cattrs changelog", "cookie-store changelog", "customer delivery prefs", "DAPT intrusion stats",
         "enterprise ops: CSM", "enterprise ops: Teams", "APEX market share", "APEX seat surplus", "APEX EuroGrid"]
OPUS_FRESH = [0, 3, 1, 3, 3, 3, 0, 0, 0, 3, 3, 3, 0, 0, 1]
OPUS_TREE = [0, 0, 2, 0, 0, 3, 0, 0, 0, 1, 3, 0, 0, 0, 0]
OPUS_TREE_TRIALS = {"awilix release tier (implicit)": 2}
TREE_PREF_FAIL = {"awilix release tier (explicit)": 3, "awilix release tier (implicit)": 2, "cattrs changelog": 3, "cookie-store changelog": 3, "customer delivery prefs": 1}
PROBE = {"awilix release tier (explicit)": 1, "awilix release tier (implicit)": 1, "cattrs changelog": 0, "cookie-store changelog": 0, "customer delivery prefs": 0}
PROBE_PREF_FAIL = {"cattrs changelog", "cookie-store changelog", "customer delivery prefs"}
TREE_PROBE = {"awilix release tier (explicit)": 1, "awilix release tier (implicit)": 0, "cattrs changelog": 0, "cookie-store changelog": 0, "customer delivery prefs": 0}
TREE_PROBE_PREF_FAIL = {"awilix release tier (implicit)", "cattrs changelog", "cookie-store changelog"}


def fig_grid():
    gpt = {}
    for r in read_csv(DATA / "fig_fm_tasks.csv"):
        if r[0] == "GPT-6 Sol":
            gpt.setdefault(r[2], {})[r[1]] = (int(r[3]), int(r[4]))
        elif r[2] == "fresh graph build":
            assert OPUS_FRESH[TASKS.index(r[1])] == int(r[3])
    assert sum(OPUS_FRESH) == 23 and sum(OPUS_TREE) == 9
    cols = [("none", "GPT-6 Sol", lambda t: gpt["no memory"][t] + ((),)), ("run\ncache", "GPT-6 Sol", lambda t: gpt["raw run cache"][t] + ((),)),
            ("graph", "GPT-6 Sol", lambda t: gpt["graph memory"][t] + ((),)),
            ("graph", "Claude Opus 5.5", lambda t: (OPUS_FRESH[TASKS.index(t)], 3, ())),
            ("step\ntrees", "Claude Opus 5.5", lambda t: (OPUS_TREE[TASKS.index(t)], OPUS_TREE_TRIALS.get(t, 3), TREE_PREF_FAIL.get(t, 0))),
            ("graph +\nasks", "Claude Opus 5.5", lambda t: (PROBE[t], 1, 1 - PROBE[t] if t in PROBE_PREF_FAIL else 0) if t in PROBE else None),
            ("step trees\n+ asks", "Claude Opus 5.5", lambda t: (TREE_PROBE[t], 1, 1 - TREE_PROBE[t] if t in TREE_PROBE_PREF_FAIL else 0) if t in TREE_PROBE else None)]
    m = len(TASKS)
    fig = plt.figure(figsize=(WIDE, 3.2))
    ax = fig.add_axes([0.01, 0.07, 0.98, 0.92])
    ax.axis("off")
    cw = 3.55
    x0 = 0.0
    out = []
    LAB = 11.0
    for j, (cl, grp, fn) in enumerate(cols):
        xc = LAB + j * cw + (0.55 if j >= 3 else 0)
        ax.text(xc + 1.55, m + 0.55, cl, ha="center", va="bottom", fontsize=6.8, linespacing=1.0)
        for i, t in enumerate(TASKS):
            y = m - 1 - i
            v = fn(t)
            if v is None:
                continue
            if v == "pending":
                for b in range(1):
                    ax.add_patch(Rectangle((xc + 1.15, y + 0.2), 0.8, 0.6, fc="white", ec="#999999", lw=0.5, ls=(0, (2, 1.5))))
                ax.text(xc + 2.05, y + 0.5, "pending", fontsize=5.4, color="#777777", va="center")
                out.append((grp, cl.replace("\n", " "), t, "pending", ""))
                continue
            k, n, pref = v[0], v[1], v[2] if len(v) > 2 else 0
            pref = pref or 0
            nbox = 1 if n == 1 else 3
            bx0 = xc + (1.15 if nbox == 1 else 0.0)
            for b in range(nbox):
                x = bx0 + b * 1.05
                if b < k:
                    ax.add_patch(Rectangle((x, y + 0.2), 0.9, 0.6, fc=BLUE, ec=BLUE, lw=0.5))
                elif b < n:
                    isp = (b - k) < pref
                    ax.add_patch(Rectangle((x, y + 0.2), 0.9, 0.6, fc="white", ec=VERM if isp else "#888888", lw=0.5, hatch="//////" if isp else None))
                else:
                    ax.add_patch(Rectangle((x, y + 0.2), 0.9, 0.6, fc="white", ec="#CCCCCC", lw=0.4, ls=(0, (1.5, 1.5))))
            out.append((grp, cl.replace("\n", " "), t, f"{k} of {n}", pref))
    for i, t in enumerate(TASKS):
        ax.text(LAB - 0.4, m - 1 - i + 0.5, t.replace(": ", ", ").replace("prefs", "preferences").replace("stats", "statistics"), ha="right", va="center", fontsize=6.8)
    ax.plot([-0.5, LAB + 7 * cw + 1], [2.99 - 0.0, 2.99], color="#999999", lw=0.4, ls=":")
    for grp, a, b in (("GPT-6 Sol", 0, 2), ("Claude Opus 5.5", 3, 6)):
        xa = LAB + a * cw + (0.55 if a >= 3 else 0)
        xb = LAB + b * cw + (0.55 if b >= 3 else 0) + 3.1
        ax.plot([xa, xb], [m + 1.95, m + 1.95], color=BLACK, lw=0.6)
        ax.text((xa + xb) / 2, m + 2.15, grp, ha="center", va="bottom", fontsize=7.8)
    ax.set_xlim(0, LAB + 7 * cw + 1)
    ax.set_ylim(-1.0, m + 3.0)
    handles = [Rectangle((0, 0), 1, 1, fc=BLUE, ec=BLUE, label="trial passed"),
               Rectangle((0, 0), 1, 1, fc="white", ec="#888888", label="trial failed"),
               Rectangle((0, 0), 1, 1, fc="white", ec=VERM, hatch="//////", label="failed only the preference check"),
               Rectangle((0, 0), 1, 1, fc="white", ec="#CCCCCC", ls=(0, (1.5, 1.5)), label="no valid trial")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
    dump("fig_grid", ["task_model", "memory", "task", "passed_of", "preference_only_fails"], out)
    save(fig, "fig_grid")


# ---- Figure 7: step tree ----------------------------------------------------------------------------------------------------
def tree_data():
    p = DATA / "fig_tree.json"
    if p.exists():
        return json.load(open(p))
    sys.path.insert(0, str(V2 / "whitepaper-v2" / "figures"))
    from figures2 import pick_task, all_branches  # noqa: E402
    trees = json.load(open(Path.home() / "Documents/(private data folder, not distributed)/v4/mem-tree2/trees.json"))
    t = pick_task(trees)
    bs = all_branches(t)
    d = {"runs": t["runs"], "pass": t["pass"], "branches": [[{"label": n["label"], "pass": n["pass"], "runs": n["runs"]} for n in b] for b in bs]}
    json.dump(d, open(p, "w"))
    return d


def short(label, n=9):
    label = label.replace("sh:", "").replace("tool:", "").replace("api:", "").replace(":noext", "").replace("__", "_")
    return label if len(label) <= n else label[:n - 1] + "."


def fig_tree(max_steps=9, max_branches=6):
    d = tree_data()
    bs = d["branches"]
    bs.sort(key=lambda b: (-b[-1]["pass"], -b[-1]["runs"]))
    main, others, rest = bs[0], bs[1:max_branches], bs[max_branches:]
    S = min(max_steps, max(len(b) for b in [main] + others))
    fig = plt.figure(figsize=(WIDE, 0.55 + 0.42 * (len(others) + 1)))
    ax = fig.add_axes([0.01, 0.05, 0.98, 0.93])
    ax.axis("off")
    ml = [n["label"] for n in main]

    def node(x, y, label, col, filled):
        ax.add_patch(Rectangle((x - 0.42, y - 0.2), 0.84, 0.4, facecolor=col if filled else "white", edgecolor=col, linewidth=0.7, zorder=3))
        ax.text(x, y, short(label), fontsize=5.6, family="monospace", color="white" if filled else BLACK, ha="center", va="center", zorder=4)

    def lane(path, y, k, col, filled):
        end = min(len(path) - 1, S - 1)
        if k:
            xx = np.linspace(k - 1, k, 24)
            ax.plot(xx, y * (1 - np.cos(np.pi * (xx - (k - 1)))) / 2, color=col, lw=0.8, zorder=1)
        ax.plot([k, end], [y, y], color=col, lw=0.8, zorder=1)
        for i in range(k, min(len(path), S)):
            node(i, y, path[i]["label"], col, filled)
        leaf = path[-1]
        more = f", +{len(path) - S} steps" if len(path) > S else ""
        ax.text(end + 0.55, y, f"{'passed' if leaf['pass'] else 'failed'}, {leaf['runs']} run{'s' if leaf['runs'] != 1 else ''}{more}", fontsize=7,
                color=BLUE if leaf["pass"] else VERM, va="center", ha="left")

    lane(main, 0, 0, BLUE, True)
    for j, b in enumerate(others, start=1):
        labels = [n["label"] for n in b]
        k = 0
        while k < min(len(labels), len(ml)) and labels[k] == ml[k]:
            k += 1
        lane(b, -j, k, SKY if b[-1]["pass"] else GREY, False)
    if rest:
        ax.text(-0.4, -len(others) - 1, f"and {len(rest)} other branches: {sum(b[-1]['runs'] for b in rest)} runs, {sum(b[-1]['pass'] for b in rest)} passed", fontsize=7, va="center")
    ax.set_xlim(-0.6, S + 3.6)
    ax.set_ylim(-len(others) - 1.5, 0.6)
    dump("fig_tree", ["task_runs", "task_pass", "branches_shown", "branches_total"], [(d["runs"], d["pass"], len(others) + 1, len(bs))])
    save(fig, "fig_tree")


# ---- Figure 8: pruning --------------------------------------------------------------------------------------------------------
def fig_prune():
    rows = [tuple(int(x) for x in r) for r in read_csv(DATA / "fig_prune.csv")]
    half = (len(rows) + 1) // 2
    H = half
    fig = plt.figure(figsize=(WIDE, 0.14 * H + 0.55))
    for pi, chunk in enumerate((rows[:half], rows[half:])):
        ax = fig.add_axes([0.01 + pi * 0.505, 0.02, 0.485, 0.90])
        ax.axis("off")
        mx = max(r[2] for r in chunk)
        for r, (runs, ok, nbr, cut, af, dom) in enumerate(chunk):
            y = H - 1 - r
            x = 0
            for _ in range(nbr - cut):
                ax.add_patch(Rectangle((x + 0.1, y + 0.15), 0.8, 0.7, fc=BLUE, ec="none"))
                x += 1
            for _ in range(af):
                ax.add_patch(Rectangle((x + 0.1, y + 0.15), 0.8, 0.7, fc="white", ec=VERM, hatch="////", lw=0.5))
                x += 1
            for _ in range(dom):
                ax.add_patch(Rectangle((x + 0.1, y + 0.15), 0.8, 0.7, fc="white", ec=GREY, hatch="....", lw=0.5))
                x += 1
            ax.text(-0.4, y + 0.5, f"{runs} runs, {ok} passed", fontsize=6.3, ha="right", va="center")
            ax.text(mx + 0.6, y + 0.5, f"{cut} of {nbr} cut", fontsize=6.3, ha="left", va="center")
        ax.set_xlim(-6.5, mx + 3.5)
        ax.set_ylim(-0.2, H + 0.1)
    handles = [Rectangle((0, 0), 1, 1, fc=BLUE, ec="none", label="kept"), Rectangle((0, 0), 1, 1, fc="white", ec=VERM, hatch="////", label="cut: only failing runs took it"),
               Rectangle((0, 0), 1, 1, fc="white", ec=GREY, hatch="....", label="cut: a shorter passing branch covers it")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.2, handlelength=1.2)
    dump("fig_prune", ["runs", "passed", "branches", "cut", "cut_all_fail", "cut_dominated"], rows)
    save(fig, "fig_prune")


# ---- Figures 9: LongMemEval published rows, ablation, question types ------------------------------------------------------------
def fig_lme():
    ents = []
    for r in read_lme_csv(DATA / "B_fig_lme.csv"):
        ents.append((r[0].replace(" (all 470, includes tuning)", "$^{\\ddagger}$").replace("Snapgraph", "Ranker"), r[1], [float(x) for x in r[3:]]))
    sty = {"ours": dict(c=BLUE, m="o", mfc=BLUE), "pub_free": dict(c=SKY, m="s", mfc="white"), "pub_llm": dict(c=VERM, m="D", mfc="white")}
    fig = plt.figure(figsize=(COL, 2.35))
    axs = [fig.add_axes([0.385, 0.15, 0.285, 0.72]), fig.add_axes([0.70, 0.15, 0.285, 0.72])]
    axs[1].sharey(axs[0])
    plt.setp(axs[1].get_yticklabels(), visible=False)
    n = len(ents)
    for ax, (idx, xl, tk, k) in zip(axs, ((0, (61, 91), [65, 75, 85], 3), (3, (70, 100), [75, 85, 95], 10))):
        for i, (name, kind, v) in enumerate(ents):
            y = n - 1 - i
            s = sty[kind]
            ax.plot([v[idx + 1], v[idx + 2]], [y, y], color=s["c"], lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot([v[idx]], [y], s["m"], ms=4.2, mfc=s["mfc"], mec=s["c"], mew=1.0, zorder=3)
        ax.set_xlim(*xl)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xticks(tk)
        ax.set_xlabel("R$_{\\mathrm{all}}$@%d (%%)" % k, labelpad=1)
        ax.xaxis.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
        clean(ax)
        ax.tick_params(axis="y", length=0)
        ax.axhline(5.5, color="#999999", lw=0.4, ls=":")
        ax.axhline(2.5, color="#999999", lw=0.4, ls=":")
    axs[0].set_yticks(range(n))
    axs[0].set_yticklabels([e[0] + (DAG if e[1] != "ours" else "") for e in ents][::-1], fontsize=7)
    handles = [Line2D([], [], marker="o", ls="-", ms=4.2, color=BLUE, label="Ours, no LLM"),
               Line2D([], [], marker="s", ls="-", ms=4.2, color=SKY, mfc="white", mew=1.0, label="Published, no LLM at build"),
               Line2D([], [], marker="D", ls="-", ms=4.2, color=VERM, mfc="white", mew=1.0, label="Published, LLM-built")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.875), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8, handlelength=1.4, fontsize=6.5)
    dump("fig_lme", ["system", "kind", "R3", "R3_lo", "R3_hi", "R10", "R10_lo", "R10_hi"], [(e[0], e[1], *e[2][0:3], *e[2][3:6]) for e in ents])
    save(fig, "fig_lme")


def fig_ablation():
    items = [("snapgraph_no_rr", "Both cross-encoders"), ("snapgraph_no_contig", "Contiguity (fwd., bwd.)"), ("nograph_ens", "PPR, entity field, bwd."),
             ("snapgraph_no_spk", "Speaker prior"), ("snapgraph_no_ppr", "PPR alone"), ("snapgraph_m3_only", "MiniLM cross-encoder")]
    d = {(r[0], int(r[1])): [float(x) for x in r[2:]] for r in read_csv(DATA / "B_fig_ablation.csv")}
    ks = (5, 10, 50)
    fig = plt.figure(figsize=(COL, 2.05))
    x0, w, gap = 0.39, 0.17, 0.03
    n = len(items)
    for j, k in enumerate(ks):
        ax = fig.add_axes([x0 + j * (w + gap), 0.2, w, 0.66])
        for i, (a, lab) in enumerate(items):
            dd, lo, hi = d[(a, k)]
            y = n - 1 - i
            sig = hi < 0 or lo > 0
            ax.plot([lo, hi], [y, y], color=BLACK, lw=0.9, solid_capstyle="butt", zorder=2)
            ax.plot([dd], [y], "o", ms=4.0, mfc=BLUE if sig else "white", mec=BLUE, mew=1.0, zorder=3)
        ax.axvline(0, color=BLACK, lw=0.6, zorder=1)
        ax.set_xlim(-11, 1.5)
        ax.set_xticks([-10, -5, 0])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_title("$k=%d$" % k, fontsize=7.5, pad=2)
        ax.xaxis.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
        clean(ax)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(range(n))
        ax.set_yticklabels([lab for _, lab in items][::-1] if j == 0 else [])
    fig.text(x0 + 1.5 * w + gap, 0.03, "Change in recall@$k$ when removed (points)", ha="center", va="bottom", fontsize=7.5)
    handles = [Line2D([], [], marker="o", ls="", ms=4.0, mfc=BLUE, mec=BLUE, label="interval excludes 0"),
               Line2D([], [], marker="o", ls="", ms=4.0, mfc="white", mec=BLUE, mew=1.0, label="interval includes 0")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1)
    dump("fig_ablation", ["removed", "k", "ablated_minus_full", "lo95", "hi95"], [(a, k, *d[(a, k)]) for k in ks for a, _ in items])
    save(fig, "fig_ablation")


def fig_lme_types():
    items = [(r[0], int(r[1]), int(r[2])) for r in read_csv(DATA / "fig_lme_types.csv")]
    m = len(items)
    fig = plt.figure(figsize=(COL, 1.9))
    ax = fig.add_axes([0.46, 0.17, 0.36, 0.80])
    out = []
    for i, (label, k, n) in enumerate(items):
        y = m - 1 - i
        v = 100 * k / n
        lo, hi = wilson(k, n)
        col = BLUE if label == "all 500" else GREY
        ax.plot([lo, hi], [y, y], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot([v], [y], "o", ms=4.0, mfc=col, mec=col, zorder=3)
        ax.text(hi + 1, y, f"{k} of {n}", va="center", ha="left", fontsize=6.8)
        out.append((label, k, n, v, lo, hi))
    ax.axhline(0.5, color="#999999", lw=0.4, ls=":")
    ax.set_xlim(60, 100)
    ax.set_ylim(-0.6, m - 0.4)
    ax.set_xticks([60, 80, 100])
    ax.set_yticks(range(m))
    ax.set_yticklabels([i[0] for i in items][::-1])
    ax.set_xlabel("accuracy, official judge (%)", labelpad=1)
    ax.xaxis.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
    clean(ax)
    ax.tick_params(axis="y", length=0)
    dump("fig_lme_types", ["type", "correct", "n", "acc", "lo95", "hi95"], out)
    save(fig, "fig_lme_types")



# ---- Pareto: accuracy against read tokens, search time and build cost on LoCoMo --------------------------------------------------
def fig_pareto():
    """Every value is a number printed in the paper. Groups share a reader and a judge; only the GPT-5 pair is like-for-like."""
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    # (label, accuracy, tokens read per question, search seconds per question or None, kind, group, dx, dy, ha)
    pts = [
        ("Ours, 40 turns", 90.6, 3294, 1.05, "ours", "GPT-5", 1.09, 1.3, "left"),
        ("Mem0, current", 90.3, 6956, 3.57, "mem0", "GPT-5", 1.09, -1.2, "left"),
        ("Ours, 5a", 94.2, 3089, None, "ours", "Sonnet", 0.93, -1.6, "right"),
        ("Ours, 5r", 96.0, 3191, None, "ours", "Sonnet", 1.08, 1.0, "left"),
        ("Ours, adaptive", 95.7, 1263, None, "ours", "Sonnet", 1.09, 1.4, "left"),
        ("Ours, adaptive", 90.2, 1457, 1.05, "ours", "4o-mini", 1.09, -1.8, "left"),
        ("Mem0, 2025" + DAG, 66.9, 1764, 0.200, "mem0", "4o-mini", 1.09, 0, "left"),
        ("Mem0g, 2025" + DAG, 68.4, 3616, 0.657, "mem0", "4o-mini", 1.09, 0, "left"),
        ("Zep, 2025" + DAG, 75.1, 3911, 0.778, "other", "4o-mini", 1.09, 0, "left"),
        ("Full context" + DAG, 72.9, 26031, None, "other", "4o-mini", 0.92, 0, "right"),
    ]
    marker = {"GPT-5": "o", "Sonnet": "s", "4o-mini": "^"}
    style = {"ours": dict(mfc=BLUE, mec=BLUE), "mem0": dict(mfc="white", mec=VERM), "other": dict(mfc="white", mec=GREY)}
    fig = plt.figure(figsize=(WIDE, 2.75))
    axes = [fig.add_axes([0.075 + i * 0.325, 0.25, 0.255, 0.62]) for i in range(3)]
    out = []
    specs = ((axes[0], 2, "tokens read per question", (1000, 40000), [1000, 3000, 10000, 30000], "(a) accuracy against read cost"),
             (axes[1], 3, "search time per question (s)", (0.15, 6), [0.2, 0.5, 1, 2, 5], "(b) accuracy against search time"))
    for ax, key, xlab, xlim, ticks, title in specs:
        clean(ax, left=True)
        ax.set_xscale("log"); ax.set_xlim(*xlim); ax.set_ylim(60, 100)
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 10 else f"{v:g}"))
        ax.yaxis.grid(True, lw=0.3, color="#DDDDDD"); ax.xaxis.grid(True, lw=0.3, color="#DDDDDD")
        ax.set_xlabel(xlab, fontsize=7, labelpad=1); ax.set_title(title, fontsize=7.6, loc="left"); ax.tick_params(labelsize=6.5)
        if ax is axes[0]:
            ax.set_ylabel("LoCoMo accuracy (%)", fontsize=7)
        for lab, acc, tok, sec, kind, grp, dx, dy, ha in pts:
            x = (tok, sec)[key - 2]
            if x is None:
                continue
            ax.plot(x, acc, marker[grp], ms=5.2, mew=0.9, **style[kind], zorder=3)
            if key == 3 and lab == "Ours, adaptive":
                dx, dy, ha = 0.93, -1.8, "right"
            ax.text(x * dx, acc + dy, lab, fontsize=5.6, va="center", ha=ha)
            if key == 2:
                out.append(("pareto", lab, grp, acc, tok, sec if sec is not None else ""))
    ax = axes[2]; clean(ax); ax.set_title("(c) model tokens to build the memory", fontsize=7.6, loc="left")
    ax.plot(0, 1, "o", ms=5.5, mfc=BLUE, mec=BLUE, zorder=3, clip_on=False)
    ax.text(0.35, 1, "0 tokens; two local encoders,\n36 s for 5,882 turns on a CPU", fontsize=6, va="center")
    ax.barh(0, 11.77 - 0.46, left=0.46, height=0.45, fc="white", ec=VERM, lw=0.9, hatch="////")
    ax.text(0.5, -0.5, "0.46M to 11.77M, six model-built\nmemories (EdgeMem's meter)" + DAG, fontsize=6, va="center")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["model-\nbuilt", "ours"], fontsize=6.5); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 13); ax.set_ylim(-1.0, 1.6); ax.set_xlabel("millions of model tokens, LoCoMo", fontsize=7, labelpad=1); ax.tick_params(labelsize=6.5)
    ax.xaxis.grid(True, lw=0.3, color="#DDDDDD")
    handles = [Line2D([0], [0], marker="o", color="none", mfc=BLUE, mec=BLUE, ms=5, label="ours"),
               Line2D([0], [0], marker="o", color="none", mfc="white", mec=VERM, ms=5, label="Mem0; " + DAG + " transcribed from its 2025 paper"),
               Line2D([0], [0], marker="o", color="none", mfc="white", mec=GREY, ms=5, label="other systems" + DAG),
               Line2D([0], [0], marker="o", color="none", mfc="none", mec=BLACK, ms=5, label="GPT-5 reads and judges (like-for-like)"),
               Line2D([0], [0], marker="s", color="none", mfc="none", mec=BLACK, ms=5, label="Sonnet 5.5 reads, 2025 judge"),
               Line2D([0], [0], marker="^", color="none", mfc="none", mec=BLACK, ms=5, label="gpt-4o-mini reads, 2025 judge")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.0, fontsize=6.3)
    dump("fig_pareto", ["panel", "system", "reader group", "accuracy", "tokens", "seconds"], out)
    save(fig, "fig_pareto")


FIGS = {"recall_v2": fig_recall_v2, "beat": fig_beat, "tokens": fig_tokens, "acc": fig_acc_turns, "recall": fig_recall, "grid": fig_grid, "tree": fig_tree,
        "prune": fig_prune, "lme": fig_lme, "ablation": fig_ablation, "lmetypes": fig_lme_types, "pareto": fig_pareto}

if __name__ == "__main__":
    for w in (sys.argv[1:] or list(FIGS)):
        FIGS[w]()
        print("done", w)
