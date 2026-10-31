import json, sys
DEV = {"conv-26", "conv-30", "conv-41"}
for f in sys.argv[1:]:
    r = [json.loads(l) for l in open(f)]
    for nm, rs in (("test", [x for x in r if x["sample"] not in DEV]), ("dev", [x for x in r if x["sample"] in DEV])):
        if not rs: continue
        cats = {c: [x["correct"] for x in rs if x["category"] == c] for c in (1, 2, 3, 4)}
        rec = [x["recall"] for x in rs if x["recall"] is not None]
        print(f"{f.split('/')[-1]} {nm} n={len(rs)} acc={sum(x['correct'] for x in rs)/len(rs):.3f} " + " ".join(f"c{c}={sum(v)/len(v):.3f}({len(v)})" for c, v in cats.items() if v) + f" allgold={sum(rec)/len(rec):.3f}")
