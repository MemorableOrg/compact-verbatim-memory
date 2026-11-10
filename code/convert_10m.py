"""Write ~/Documents/fm-data/beam/beam_10M.jsonl for BEAM-10M chats 1..10 in the runner's row format (conversation_id, chat as a list of
batches of turn dicts, probing_questions as a JSON string). Row i corresponds to chats/10M/<i+1> and to Mem0's conversation_idx i."""
import ast, json, os, sys
ROOT = os.path.expanduser("~/Documents/fm-data/beam-repos/beam/chats/10M")
OUT = os.path.expanduser("~/Documents/fm-data/beam/beam_10M.jsonl")
def parse(x):
    if isinstance(x, str):
        s = x.strip()
        if s[:1] in "[{":
            try: return parse(ast.literal_eval(s))
            except Exception:
                try: return parse(json.loads(s))
                except Exception: return x
        return x
    if isinstance(x, list): return [parse(v) for v in x]
    if isinstance(x, dict): return {k: parse(v) for k, v in x.items()}
    return x
def batches(chat):
    """Yield one flat list of turn dicts per batch, in plan then batch order."""
    chat = parse(chat)
    for plan in chat:
        for key, plist in (plan.items() if isinstance(plan, dict) else [(None, plan)]):
            for b in plist:
                turns = []
                def walk(o):
                    if isinstance(o, dict) and "role" in o and "content" in o: turns.append(o)
                    elif isinstance(o, dict):
                        for v in o.values(): walk(v)
                    elif isinstance(o, list):
                        for v in o: walk(v)
                walk(b.get("turns", b) if isinstance(b, dict) else b)
                if turns: yield turns
n_rows = 0
with open(OUT, "w") as out:
    for d in range(1, 11):
        chat = json.load(open(f"{ROOT}/{d}/chat.json"))
        pq = json.load(open(f"{ROOT}/{d}/probing_questions/probing_questions.json"))
        bl = []
        for turns in batches(chat):
            bl.append([dict(content=t.get("content") or "", id=str(t["id"]), role=t["role"], time_anchor=t.get("time_anchor")) for t in turns])
        nq = sum(len(v) for v in pq.values())
        print(f"chat {d}: {len(bl)} batches, {sum(len(b) for b in bl)} turns, {nq} questions, abilities {len(pq)}", flush=True)
        out.write(json.dumps(dict(conversation_id=d, chat=bl, probing_questions=json.dumps(pq))) + "\n"); n_rows += 1
print("rows", n_rows, "->", OUT)
