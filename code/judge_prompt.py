# Source: https://raw.githubusercontent.com/mem0ai/memory-benchmarks/main/benchmarks/locomo/prompts.py (lines 199-327, verbatim, no-evidence path used)
import json
# JUDGE PROMPT (unified — evidence is an optional add-on)
# ===============================================================================

# ── Shared judge system prompt ──
JUDGE_SYSTEM_PROMPT = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."

# ── Evidence chunk (injected when evidence is available) ──
_EVIDENCE_CHUNK = """
## Evidence (actual conversation messages containing the answer)
{evidence_context}
"""

_EVIDENCE_RULE = """
5. **EVIDENCE SUPPORTS ANSWER**: If the evidence corroborates the generated answer, mark CORRECT — even when the generated answer diverges from the gold answer. The gold answer may be wrong or oversimplified; if the generated answer provides a more accurate or better-supported conclusion based on the evidence, that is acceptable. Use evidence only to ACCEPT answers, never to reject them more strictly.
"""

_EVIDENCE_WRONG_CLAUSE = " AND is not supported by evidence"

# ── Unified judge prompt template ──
_JUDGE_TEMPLATE = """Label the generated answer as CORRECT or WRONG.
{evidence_section}
## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.
{evidence_rule}
5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer{evidence_wrong_clause}
- The answer addresses a completely different topic

## Question
Question: {{question}}
Gold answer: {{answer}}
Generated answer: {{response}}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


def _build_judge_prompt(evidence_context: str = None) -> str:
    """Build the judge prompt template, with or without evidence."""
    if evidence_context:
        prompt = _JUDGE_TEMPLATE.format(
            evidence_section=_EVIDENCE_CHUNK.format(evidence_context=evidence_context),
            evidence_rule=_EVIDENCE_RULE,
            evidence_wrong_clause=_EVIDENCE_WRONG_CLAUSE,
        )
        # Renumber rules when evidence rule is inserted (5→6, 6→7, 7→8)
        prompt = prompt.replace("\n5. **SEMANTIC OVERLAP", "\n6. **SEMANTIC OVERLAP")
        prompt = prompt.replace("\n6. **SAME REFERENT", "\n7. **SAME REFERENT")
        prompt = prompt.replace("\n7. **FOCUS ON KNOWLEDGE", "\n8. **FOCUS ON KNOWLEDGE")
    else:
        prompt = _JUDGE_TEMPLATE.format(
            evidence_section="",
            evidence_rule="",
            evidence_wrong_clause="",
        )
    return prompt


# Keep JUDGE_PROMPT as a module-level constant for backward compat (no evidence)
JUDGE_PROMPT = _build_judge_prompt(evidence_context=None)


# ===============================================================================
# DISPATCH
# ===============================================================================

def get_judge_prompt(
    category: int,
    question: str,
    answer: str,
    response: str,
) -> str:
    """Return the formatted unified judge prompt (no evidence).

    Accepts category for backwards compatibility but uses the same
    unified prompt for all categories.
    """
    return JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        response=response,
    )


def get_judge_prompt_with_evidence(
    category: int,
    question: str,
    answer: str,
    response: str,
    evidence_context: str,
) -> str:
    """Return the formatted judge prompt with evidence context.

    Same prompt as get_judge_prompt but with the evidence section
    and evidence-specific rule injected.
    """
    prompt = _build_judge_prompt(evidence_context=evidence_context)
    return prompt.format(
        question=question,
        answer=answer,
        response=response,
    )


def preprocess_answer(category: int, answer: str) -> str:
    """Preprocess ground truth answer based on category.

    Category 3 (open-domain): use only the first part before semicolon.
    """
    if category == 3 and ";" in answer:
        return answer.split(";")[0].strip()
    return answer
