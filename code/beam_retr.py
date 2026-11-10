"""Offline BEAM retrieval check: no model calls. Scores how often the gold source messages are in the top K.

    python beam_retr.py --tier 100K --method base
    python beam_retr.py --tier 500K --method chunk --dense bge-base --rerank bge-reranker-base

Tiers 100K and 500K are the development tiers; 1M is the test tier and is scored once, after the setting is frozen.
"""
import argparse, ast, hashlib, json, os, re, sys, time
from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi
import Stemmer

ROOT = "/Users/nikhilkrishnaswamy/Documents/fm-data/beam-repos/beam/chats"
ONE_M = "/Users/nikhilkrishnaswamy/Documents/fm-data/beam/beam_1M.jsonl"
MARK = re.compile(r"\s*->->\s*[\d,]+\s*$")
STEM = Stemmer.Stemmer("english")
STOP = set("a an the and or but of to in on at for with by from is are was were be been being do does did have has had i you he she it we they "
           "me my your our their this that these those what which who whom when where why how can could would should will shall may might "
           "about into over after before as if then than so not no yes up down out any all some each".split())
STANDING = re.compile(r"^\s*(please\s+)?(always|never)\b|\bfrom now on\b|\bI (really )?prefer\b|\bmy preference\b", re.I)
DENSE = {"bge-small": "BAAI/bge-small-en-v1.5", "bge-base": "BAAI/bge-base-en-v1.5", "bge-large": "BAAI/bge-large-en-v1.5", "bge-m3": "BAAI/bge-m3"}
CE = {"bge-reranker-base": "BAAI/bge-reranker-base", "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3", "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2"}


def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
def stems(s): return [STEM.stemWord(w) for w in tok(s) if w not in STOP]


def flat_ids(x):
    if isinstance(x, bool) or x is None: return []
    if isinstance(x, int): return [x]
    if isinstance(x, str): return [int(v) for v in re.findall(r"\d+", x)] if re.fullmatch(r"[\d,\s\-]+", x) else []
    if isinstance(x, (list, tuple)): return [i for v in x for i in flat_ids(v)]
    if isinstance(x, dict): return [i for v in x.values() for i in flat_ids(v)]
    return []


def msgs_from(obj):
    """Every message dict in a chat, in order, whatever the nesting."""
    if isinstance(obj, dict):
        if "role" in obj and "content" in obj: yield obj; return
        for k in ("turns", "batches"):
            if k in obj: yield from msgs_from(obj[k]); return
        for v in obj.values(): yield from msgs_from(v)
    elif isinstance(obj, list):
        for v in obj: yield from msgs_from(v)


def make_conv(cid, chat, pq):
    turns, date = [], ""
    for m in msgs_from(chat):
        if m.get("time_anchor"): date = m["time_anchor"]
        turns.append(dict(id=m["id"], role=m["role"], date=date, text=MARK.sub("", m["content"] or "")))
    pq = pq if isinstance(pq, dict) else ast.literal_eval(pq)
    qs = []
    for ab, lst in pq.items():
        for qi, q in enumerate(lst):
            qs.append(dict(qid=f"{cid}_{ab}_{qi}", ability=ab, question=q["question"], gold=sorted(set(flat_ids(q.get("source_chat_ids"))))))
    return dict(cid=str(cid), turns=turns, qs=qs)


def load(tier):
    if tier == "1M":
        for line in open(ONE_M):
            r = json.loads(line)
            yield make_conv(r["conversation_id"], r["chat"], r["probing_questions"])
        return
    d = os.path.join(ROOT, tier)
    for name in sorted(os.listdir(d), key=lambda x: int(x) if x.isdigit() else 0):
        p = os.path.join(d, name)
        if not os.path.isdir(p): continue
        chat = json.load(open(os.path.join(p, "chat.json")))
        pq = json.load(open(os.path.join(p, "probing_questions", "probing_questions.json")))
        yield make_conv(f"{tier}-{name}", chat, pq)


def chunks(text, words, overlap):
    w = text.split()
    if len(w) <= words: return [text]
    step = max(1, words - overlap)
    return [" ".join(w[i:i + words]) for i in range(0, max(1, len(w) - overlap), step)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="100K")
    ap.add_argument("--dense", default="bge-small")
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=0, help="words per chunk; 0 embeds whole messages (truncated at maxlen)")
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--bm25-unit", choices=["msg", "chunk"], default="msg")
    ap.add_argument("--rerank", default="")
    ap.add_argument("--rerank-n", type=int, default=100)
    ap.add_argument("--rerank-chars", type=int, default=1500)
    ap.add_argument("--rerank-mode", choices=["replace", "fuse"], default="fuse")
    ap.add_argument("--pair", action="store_true", help="add a share of each message's score to its partner in the user/assistant pair")
    ap.add_argument("--pair-w", type=float, default=0.3)
    ap.add_argument("--standing", type=int, default=0, help="always show the N user messages that state a standing instruction or preference closest to the question")
    ap.add_argument("--ks", default="10,20,40")
    ap.add_argument("--compact", action="store_true", help="strip markdown, pleasantries and repeated assistant sentences before indexing (compact.py)")
    ap.add_argument("--quant", choices=["none", "int8", "binary"], default="none", help="store message vectors at 1 byte or 1 bit per dimension")
    ap.add_argument("--limit", type=int, default=0, help="conversations")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(DENSE[a.dense], device="cpu"); model.max_seq_length = a.maxlen
    ce = None
    if a.rerank:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(CE[a.rerank], device="cpu", max_length=512)
    os.makedirs("beam_emb", exist_ok=True)

    rows = []
    t0 = time.time()
    for ci, c in enumerate(load(a.tier)):
        if a.limit and ci >= a.limit: break
        T = c["turns"]; n = len(T)
        if a.compact:
            from compact import compact_turns
            before = sum(len(t["text"]) for t in T)
            T = compact_turns(T)
            c["chars"] = (before, sum(len(t["text"]) for t in T))
        id2i = {t["id"]: i for i, t in enumerate(T)}
        units, owner = [], []
        for i, t in enumerate(T):
            for ch in (chunks(t["text"], a.chunk, a.overlap) if a.chunk else [t["text"]]):
                units.append(f"{t['role']}: {ch}"); owner.append(i)
        owner = np.array(owner)
        key = hashlib.md5(("\n".join(units) + a.dense + str(a.maxlen)).encode()).hexdigest()[:16]
        path = f"beam_emb/retr_{a.dense}_{key}.npz"
        if os.path.exists(path): E = np.load(path)["E"]
        else:
            E = model.encode(units, batch_size=32, normalize_embeddings=True); np.savez(path, E=E)
        if a.quant == "int8":  # symmetric 8-bit codes; the query stays full precision
            E = np.round(E * 127).astype(np.int8).astype(np.float32) / 127
        elif a.quant == "binary":  # sign bits only, scored as +1/-1 against the query
            E = np.where(E > 0, 1.0, -1.0).astype(np.float32)
        bm_docs = units if a.bm25_unit == "chunk" else [f"{t['role']} {t['date']} {t['text']}" for t in T]
        bm = BM25Okapi([stems(d) or ["_"] for d in bm_docs])
        bm_owner = owner if a.bm25_unit == "chunk" else np.arange(n)
        pool = [i for i, t in enumerate(T) if t["role"] == "user" and STANDING.search(t["text"][:400])]
        qs = [q for q in c["qs"] if q["gold"]]
        if not qs: continue
        Q = model.encode([q["question"] for q in qs], batch_size=32, normalize_embeddings=True)
        for q, qv in zip(qs, Q):
            dense = np.full(n, -1e9); np.maximum.at(dense, owner, E @ qv)
            raw = bm.get_scores(stems(q["question"]) or tok(q["question"]))
            sparse = np.full(n, -1e9); np.maximum.at(sparse, bm_owner, raw)
            fused = np.zeros(n)
            for s in (dense, sparse):
                for pos, i in enumerate(np.argsort(-s, kind="stable")[:300]): fused[i] += 1.0 / (61 + pos)
            if a.pair:
                f2 = fused.copy()
                for i in range(n):
                    j = i + 1 if T[i]["role"] == "user" else i - 1
                    if 0 <= j < n: f2[j] += a.pair_w * fused[i]
                fused = f2
            order = list(np.argsort(-fused, kind="stable"))
            if ce is not None:
                cand = order[:a.rerank_n]
                pairs = [(q["question"], f"{T[j]['role']} ({T[j]['date']}): {T[j]['text'][:a.rerank_chars]}") for j in cand]
                s = ce.predict(pairs, batch_size=32, show_progress_bar=False)
                ce_order = [cand[j] for j in np.argsort(-np.asarray(s), kind="stable")]
                if a.rerank_mode == "replace": order = ce_order + order[a.rerank_n:]
                else:
                    r = np.zeros(n)
                    for lst in (ce_order, cand):
                        for pos, i in enumerate(lst): r[i] += 1.0 / (61 + pos)
                    order = sorted(cand, key=lambda i: -r[i]) + order[a.rerank_n:]
            if a.standing and pool:
                keep = sorted(pool, key=lambda i: -dense[i])[:a.standing]
                order = keep + [i for i in order if i not in set(keep)]
            gold = [id2i[g] for g in q["gold"] if g in id2i]
            if not gold: continue
            rank = {i: p for p, i in enumerate(order)}
            row = dict(qid=q["qid"], ability=q["ability"], ngold=len(gold))
            for k in ks:
                hit = sum(rank[g] < k for g in gold)
                row[f"all@{k}"] = int(hit == len(gold)); row[f"frac@{k}"] = hit / len(gold)
                row[f"tok@{k}"] = sum(len(T[i]["text"]) for i in order[:k]) / 4
            rows.append(row)
        print(f"  conv {ci} {c['cid']}: {n} msgs, {len(units)} units, {len(qs)} q, {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    by = defaultdict(list)
    for r in rows: by[r["ability"]].append(r)
    print(f"tier {a.tier} dense {a.dense}/{a.maxlen} chunk {a.chunk} bm25 {a.bm25_unit} rerank {a.rerank or '-'} pair {a.pair}  n={len(rows)}")
    head = "".join(f"  all@{k} frac@{k}" for k in ks)
    print(f"{'ability':26s} {'n':>4s}{head}  tok@{ks[1] if len(ks)>1 else ks[0]}")
    for ab in sorted(by) + ["ALL"]:
        rs = rows if ab == "ALL" else by[ab]
        vals = "".join(f"  {100*np.mean([r[f'all@{k}'] for r in rs]):6.1f} {100*np.mean([r[f'frac@{k}'] for r in rs]):6.1f}" for k in ks)
        kt = ks[1] if len(ks) > 1 else ks[0]
        print(f"{ab:26s} {len(rs):4d}{vals}  {np.mean([r[f'tok@{kt}'] for r in rs]):8.0f}")
    if a.out:
        with open(a.out, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
