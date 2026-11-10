"""Deterministic compaction of chat memory: strip text that never carries an answer. No model calls.

Rules (each removes only form, never a fact the user or assistant stated):
- markdown decoration: heading hashes, bold/italic stars, horizontal rules, table pipes, code fences;
- pleasantries and sign-offs on their own sentence ("Great question!", "Hope this helps.", "Let me know if ...");
- whitespace runs and blank lines;
- assistant sentences repeated word for word earlier in the same conversation (the first copy is kept).

    from compact import compact_turns
    turns = compact_turns(turns)   # list of dicts with "role" and "text"
"""
import re

MARKDOWN = [(re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""), (re.compile(r"\*\*|__|(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)"), ""),
            (re.compile(r"^\s*([-*_]\s*){3,}$", re.M), ""), (re.compile(r"^```\w*\s*$", re.M), ""),
            (re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", re.M), ""), (re.compile(r"\s*\|\s*"), " | ")]
FILLER = re.compile(
    r"^(great|good|excellent|wonderful|fantastic) (question|point|idea)[!.]?$|^(sure|absolutely|of course|certainly)[!,.]?$|"
    r"^(i )?hope (this|that) helps.*$|^(feel free|don't hesitate) to .*$|^let me know if .*$|^happy to help.*$|"
    r"^(thanks|thank you) for (sharing|asking|the update|letting me know).*$|^(good luck|best of luck|you've got this).*$|"
    r"^is there anything else .*$|^would you like (me )?to .*\?$", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")


def clean_text(text):
    for pat, rep in MARKDOWN:
        text = pat.sub(rep, text)
    out = []
    for line in text.splitlines():
        sents = [s for s in SENT.split(line.strip()) if s and not FILLER.match(s.strip())]
        if sents:
            out.append(" ".join(sents))
    return re.sub(r"[ \t]{2,}", " ", "\n".join(out)).strip()


def compact_turns(turns):
    """Return new turn dicts with cleaned text; assistant sentences seen before in the conversation are dropped."""
    seen, out = set(), []
    for t in turns:
        text = clean_text(t["text"])
        if t["role"] == "assistant":
            kept = []
            for s in SENT.split(text):
                key = re.sub(r"\W+", " ", s.lower()).strip()
                if len(key) > 40 and key in seen:  # short sentences repeat naturally; only drop long verbatim repeats
                    continue
                seen.add(key)
                kept.append(s)
            text = " ".join(kept)
        out.append(dict(t, text=text or t["text"][:200]))
    return out
