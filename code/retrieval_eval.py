import json, re, sys, time, os
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi
import Stemmer

MONTHS = "january february march april may june july august september october november december".split()
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
STOP = set("the a an of to in on at is was were did do does what when who how which and or for with his her their he she they it that this".split())
QWORDS = set("what when who whom whose how which where why did do does is was were are has have had would could can will about after before during".split())
stem = Stemmer.Stemmer("english").stemWord
MODEL = os.environ.get("EMB", "BAAI/bge-m3")

def load(path="data/locomo10.json"):
    convs = {}
    for c in json.load(open(path)):
        cv = c["conversation"]; turns = []; n = 1
        while f"session_{n}" in cv:
            date = cv.get(f"session_{n}_date_time", "")
            for t in cv[f"session_{n}"]:
                turns.append(dict(id=t["dia_id"], speaker=t["speaker"], session=n, date=date,
                                  text=t["text"] + (f" [shared image: {t['blip_caption']}]" if t.get("blip_caption") else "")))
            n += 1
        full = [f"{t['speaker']} {t['date']} {t['text']}" for t in turns]
        convs[c["sample_id"]] = dict(turns=turns, full=full, bm25=BM25Okapi([tok(d) for d in full]),
            bm25s=BM25Okapi([[stem(w) for w in tok(d)] for d in full]), qa=c["qa"])
    return convs

def date_boost(conv, q, scores):
    ql = q.lower(); dates = set()
    for m in re.finditer(r"(\d{1,2})\s+(" + "|".join(MONTHS) + r")(?:,?\s+(\d{4}))?", ql): dates.add((m.group(1), m.group(2)))
    for m in re.finditer(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})", ql): dates.add((m.group(2), m.group(1)))
    months = {m for m in MONTHS if re.search(r"\b" + m + r"\b", ql)}
    mx = max(scores) or 1.0
    for i, t in enumerate(conv["turns"]):
        d = t["date"].lower()
        if any(re.search(r"\b" + dd + r"\s+" + mm + r"\b|\b" + mm + r"\s+" + dd + r"\b", d) for dd, mm in dates): scores[i] += 0.5 * mx
        elif months and any(m in d for m in months): scores[i] += 0.15 * mx
    return scores

def bm25_scores(conv, q, clean=False):
    if clean:
        qt = [stem(w) for w in tok(q) if w not in QWORDS and w not in STOP] or [stem(w) for w in tok(q)]
        s = list(conv["bm25s"].get_scores(qt))
    else:
        qt = [w for w in tok(q) if w not in STOP] or tok(q)
        s = list(conv["bm25"].get_scores(qt))
    return np.array(date_boost(conv, q, s))

def top(scores, k): return list(np.argsort(-scores, kind="stable")[:k])

def expand(conv, hits, k, budget):
    T = conv["turns"]; out = []; seen = set()
    for h in hits:
        add = [j for j in (h - 1, h, h + 1) if 0 <= j < len(T) and T[j]["session"] == T[h]["session"] and j not in seen]
        if budget and len(seen) + len(add) > k:
            add = [j for j in add if j == h and j not in seen][: max(0, k - len(seen))]
            if not add and len(seen) >= k: break
        seen.update(add); out += add
        if budget and len(seen) >= k: break
    return sorted(seen)

def session_first(conv, scores, k, nsess=2):
    T = conv["turns"]; by = defaultdict(list)
    for i, s in enumerate(scores): by[T[i]["session"]].append(s)
    sc = {s: sum(sorted(v, reverse=True)[:3]) for s, v in by.items()}
    best = set(sorted(sc, key=lambda s: -sc[s])[:nsess])
    sc2 = np.array([s if T[i]["session"] in best else -1e9 for i, s in enumerate(scores)])
    return sorted(top(sc2, k))

def rrf(ranks_list, n, kk=60):
    r = np.zeros(n)
    for ranking in ranks_list:
        for pos, i in enumerate(ranking): r[i] += 1.0 / (kk + pos + 1)
    return r

def main():
    convs = load()
    emb = {}
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu"); m.max_seq_length = 256
    cache = f"emb_cache_{MODEL.split('/')[-1]}.npz"
    if os.path.exists(cache):
        z = np.load(cache); emb = {k: z[k] for k in z.files}; embtime = float(emb.pop("_time")[0])
    else:
        for sid, c in convs.items():
            emb[sid] = m.encode([f"{t['speaker']}: {t['text']}" for t in c["turns"]], batch_size=32, normalize_embeddings=True)
            print(sid, len(c["turns"]), f"{time.time()-t0:.0f}s", flush=True)
        embtime = time.time() - t0
        np.savez(cache, _time=np.array([embtime]), **emb)
    print(f"embedding all turns: {embtime:.0f}s with {MODEL}")

    qs = [(sid, q) for sid, c in convs.items() for q in c["qa"] if q["category"] in (1, 2, 3, 4)]
    qs = [(s, q) for s, q in qs if [e for x in q.get("evidence", []) for e in re.findall(r"D\d+:\d+", x)]]
    print("questions with evidence:", len(qs))
    qemb = {}
    for sid in convs:
        sq = [q["question"] for s, q in qs if s == sid]
        qemb[sid] = dict(zip(sq, m.encode(sq, batch_size=32, normalize_embeddings=True)))

    def run(name, fn, k):
        allg = []; frac = []; nt = []
        for sid, q in qs:
            c = convs[sid]; ev = [e for x in q["evidence"] for e in re.findall(r"D\d+:\d+", x)]
            idx = fn(c, sid, q["question"], k); ids = {c["turns"][i]["id"] for i in idx}
            allg.append(all(e in ids for e in ev)); frac.append(sum(e in ids for e in ev) / len(ev)); nt.append(len(idx))
        return np.mean(allg), np.mean(frac), np.mean(nt)

    def base(c, sid, q, k): return sorted(top(bm25_scores(c, q), k))
    def nbr_full(c, sid, q, k): return expand(c, top(bm25_scores(c, q), k), k, False)
    def nbr_bud(c, sid, q, k): return expand(c, top(bm25_scores(c, q), k), k, True)
    def sess(c, sid, q, k): return session_first(c, bm25_scores(c, q), k)
    def clean(c, sid, q, k): return sorted(top(bm25_scores(c, q, True), k))
    def dense_rank(c, sid, q): return top(c_emb(sid) @ qemb[sid][q], len(c["turns"]))
    def c_emb(sid): return emb[sid]
    def dense(c, sid, q, k): return sorted(top(emb[sid] @ qemb[sid][q], k))
    def hyb(c, sid, q, k, cl=False):
        n = len(c["turns"]); b = top(bm25_scores(c, q, cl), 200); d = top(emb[sid] @ qemb[sid][q], 200)
        return sorted(top(rrf([b, d], n), k))
    def hyb_clean(c, sid, q, k): return hyb(c, sid, q, k, True)
    def hyb_nbr_bud(c, sid, q, k):
        n = len(c["turns"]); sc = rrf([top(bm25_scores(c, q, True), 200), top(emb[sid] @ qemb[sid][q], 200)], n)
        return expand(c, top(sc, k), k, True)
    def hyb_nbr_full(c, sid, q, k):
        n = len(c["turns"]); sc = rrf([top(bm25_scores(c, q, True), 200), top(emb[sid] @ qemb[sid][q], 200)], n)
        return expand(c, top(sc, k), k, False)
    def hyb_sess(c, sid, q, k):
        n = len(c["turns"]); sc = rrf([top(bm25_scores(c, q, True), 200), top(emb[sid] @ qemb[sid][q], 200)], n)
        return session_first(c, sc, k)
    def clean_nbr_bud(c, sid, q, k): return expand(c, top(bm25_scores(c, q, True), k), k, True)

    variants = [("baseline BM25", base), ("1a neighbors +-1 on k hits (more turns)", nbr_full),
        ("1b neighbors +-1, capped at k turns", nbr_bud), ("2 session-first (2 sessions)", sess),
        ("3 query cleaning + stemming", clean), ("4a dense only", dense), ("4b hybrid BM25+dense RRF", hyb),
        ("5a hybrid with cleaned BM25", hyb_clean), ("5b hybrid cleaned + neighbors capped at k", hyb_nbr_bud),
        ("5c hybrid cleaned + neighbors on k hits", hyb_nbr_full), ("5d hybrid cleaned + session-first", hyb_sess),
        ("5e cleaned BM25 + neighbors capped at k", clean_nbr_bud)]
    rows = []
    for name, fn in variants:
        r20 = run(name, fn, 20); r40 = run(name, fn, 40)
        rows.append((name, r20, r40)); print(f"{name} | {r20[0]:.3f} {r20[1]:.3f} | {r40[0]:.3f} {r40[1]:.3f} | turns {r20[2]:.1f} / {r40[2]:.1f}", flush=True)
    print("\n| variant | recall@20 all | frac@20 | recall@40 all | frac@40 | turns@20 | turns@40 |\n|---|---|---|---|---|---|---|")
    for name, a, b in rows: print(f"| {name} | {a[0]:.3f} | {a[1]:.3f} | {b[0]:.3f} | {b[1]:.3f} | {a[2]:.1f} | {b[2]:.1f} |")
    print("\nmatched-budget control")
    for name, fn in [("baseline BM25", base), ("5a hybrid cleaned", hyb_clean)]:
        for k in (50, 96):
            r = run(name, fn, k); print(f"{name} k={k} | {r[0]:.3f} {r[1]:.3f} | turns {r[2]:.1f}", flush=True)
    json.dump(dict(model=MODEL, embed_seconds=embtime, n=len(qs), rows=rows), open("retrieval_eval_out.json", "w"), default=float)

if __name__ == "__main__":
    main()
