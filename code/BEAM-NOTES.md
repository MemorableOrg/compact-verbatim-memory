# BEAM notes

Sources read 2026-10-29: BEAM repo https://github.com/mohammadtavakoli78/BEAM (commit b2da22e, cloned to (private data folder, not distributed)), paper abstract https://arxiv.org/abs/2510.27246, Mem0 repo https://github.com/mem0ai/memory-benchmarks (cloned to (private data folder)/beam-repos/mem0mb; BEAM code in benchmarks/beam/run.py and prompts.py).

## Data layout
- HF dataset Mohammadta/BEAM has splits 100K (20 chats; README calls it 128K), 500K (35), 1M (35). HF dataset Mohammadta/BEAM-10M has split 10M (10 chats, two parquet files, 154 MB + 190 MB, not downloaded). Total 100 conversations, 2,000 questions: 20 per conversation, 2 per ability.
- Row fields (1M): conversation_id, conversation_seed {category, id, subtopics, theme, title}, narratives, user_profile {user_info, user_relationships}, conversation_plan, user_questions, chat, probing_questions.
- chat is a list of batches, each a list of messages {id (int, global and increasing within the conversation), role (user|assistant), content, index, question_type, time_anchor}. time_anchor (e.g. "March-01-2024") is set on only some messages; later messages inherit the last one. User messages end with a marker like "->-> 1,1" (stripped in beam_run.py).
- probing_questions is a string holding a Python dict literal (parse with ast.literal_eval, as Mem0 does): ability -> 2 questions. Every question has question, rubric (list of strings, 1 to 12 items) and source_chat_ids (message ids of the evidence; a list, or a dict for temporal reasoning; abstention has none; 5 of 630 others are missing or non-int). Gold answer text lives in per-ability fields (answer, ideal_answer, ideal_response, ideal_summary, expected_compliance).
- Local copy of the data is also in the repo under chats/{100K,500K,1M,10M}/<n>/chat.json and probing_questions/probing_questions.json.
- Repo statistics (README): average turns per chat 107 / 416 / 842 / 7,757 for 128K / 500K / 1M / 10M, where a turn is a user plus assistant pair batch. Measured on 1M: 1,556 to 3,008 messages per conversation (mean 2,132), about 1.14M tokens by characters/4.

## Ten abilities
abstention, contradiction_resolution, event_ordering, information_extraction, instruction_following, knowledge_update, multi_session_reasoning, preference_following, summarization, temporal_reasoning.

## Judge (BEAM paper repo)
- Model: gpt-4.1-mini, temperature 0, through langchain ChatOpenAI (src/llm.py: gpt_llm_obj). run_evaluation.py passes model=gpt_llm.
- One call per rubric item, a single user message, <rubric_item> and <llm_response> substituted. The question text is NOT in the prompt. Output JSON {"score": 1.0|0.5|0.0, "reason": ...}. Question score = mean over rubric items.
- Prompt: unified_llm_judge_base_prompt, src/prompts.py line 11547, https://github.com/mohammadtavakoli78/BEAM/blob/b2da22eac88bb0874c64665f13457eb99835774a/src/prompts.py . Copied verbatim by script into conv/beam_judge_prompt.py (JUDGE_PROMPT); read that file for the text.
- event_ordering gets an extra metric in compute_metrics.py: the answer is split on newlines, each line is aligned to a rubric item by gpt-4.1-mini YES/NO equivalence calls (first unused match), then precision/recall/F1 and Kendall tau-b (normalised (tau+1)/2) are computed; final_score = tau_norm * F1.
- BEAM's other metrics (BLEU, ROUGE, embedding similarity, fact-level F1) exist in compute_metrics.py but are not used in the report.

## Scoring: pass rate versus average score
- BEAM paper repo (report_results.py): no pass rate. Per ability, the mean of per-question scores; for event_ordering the mean of tau_norm (not the rubric mean, not final_score); other abilities the mean rubric score (llm_judge_score). Averaged over conversations, written to xlsx.
- Mem0: per question, score = mean of nugget scores; PASS if score >= 0.5. "Pass rate" = passes / questions (e.g. 491/700 = 70.1%); "avg score" = mean question score (0.641). Both reported per ability. Mem0 does not use tau_norm for the main number; it computes a separate Kendall tau-b for event_ordering, combined as (score + (tau+1)/2)/2 and stored as score_with_tau, not used in the headline.
- beam_run.py reports both: pass (>= 0.5) and avg rubric score, plus tau_norm for event_ordering.

## What Mem0 calls 1M and 10M
- "BEAM 1M (700 questions)" = HF Mohammadta/BEAM split 1M: 35 conversations x 20 questions. "BEAM 10M (200 questions)" = Mohammadta/BEAM-10M split 10M: 10 x 20. Their published numbers (platform, top 200 memories): 1M pass 70.1% avg 0.641; 10M pass 50.5% avg 0.486. Top 50: 67.1% / 0.604 and 45.5% / 0.413. Per-question result files: memory-benchmarks/results/platform/beam_{1m,10m}[_top50]_results.json.
- 10M chat format differs: chat is a list of session dicts keyed by plan names, each with batches holding "turns". Mem0's parse_beam_chat handles it; beam_run.py currently reads the 1M format only.

## Differences, BEAM paper repo vs Mem0
- Judge model: gpt-4.1-mini (BEAM) vs gpt-5 for answerer and judge (Mem0 default).
- Judge prompt: BEAM's is the unified prompt without the question; Mem0's is a rewrite ("ported from unified_llm_judge_base_prompt") that adds the question, a system message, explicit rules and a clamp of raw scores to 0/0.5/1; its prompts.py is in benchmarks/beam/prompts.py.
- Metric: mean of scores (BEAM) vs pass at >= 0.5 plus mean (Mem0). Event ordering: tau_norm (BEAM) vs rubric mean (Mem0 headline).
- Event ordering alignment: Mem0 extracts events with an LLM and aligns by index, BEAM splits on newlines and aligns by YES/NO equivalence.
- Answer prompt: Mem0's has 9 rules including exact abstention sentence "I don't have enough information to answer this question." and memories sorted by time; BEAM's RAG reader prompt is in answer_probing_questions (not copied here).
- Retrieval unit: Mem0 ingests with its own extractor into memories (top 50/200 memories); BEAM's RAG baseline chunks pairs of messages (pair_chunk) with dense retrieval, and LIGHT adds working memory and a scratchpad.

## Not found or not verified
- "Mean tokens": the phrase and any token metric are absent from memory-benchmarks (README, benchmarks/, result JSONs have no token fields). beam_run.py counts reader tokens per question itself.
- The BEAM repo does not publish pass rates, so pass-rate comparisons only exist against Mem0.
- BEAM paper PDF tables (LIGHT and baseline scores per tier) were not read; only the abstract.
- Mem0's published runs used top 50 and 200 memories (extracted facts), not raw messages, so token counts are not comparable to ours.

## Our setup (beam_run.py)
Retrieval unit is one message (1,556 to 3,008 per conversation). Index: stemmed BM25 (retrieval_eval.bm25_scores, clean=True) plus bge-small-en-v1.5 (max_seq_length 256, so long assistant messages are truncated for embedding), reciprocal rank fusion, optional bge-reranker-base over top 100. Embeddings cached per conversation in conv/beam_emb/. Top K messages go to the reader in time order. Deviations from BEAM scoring: message text shown to the reader is capped at 6000 characters; event-ordering lines are capped at 30 and empty lines dropped; NaN tau counts as 0; reader prompt is ours (rules close to Mem0's).
