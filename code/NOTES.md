# LoCoMo code-only memory smoke test (2026-10-28)

Files: locomo_run.py, judge_prompt.py (Mem0 judge, verbatim from memory-benchmarks benchmarks/locomo/prompts.py, no-evidence path), results/*.jsonl, run100.log.
Data: locomo10.json, 10 convs, 1986 questions; cat1 282, cat2 321, cat3 96, cat4 841, cat5 446 (dropped) -> 1540.
Memory: every turn verbatim (speaker, session, date, image caption). BM25 (rank_bm25) over speaker+date+text, k=20, date boost (exact day+month +0.5*max score, month only +0.15*max), chronological output.
Models: answer gpt-6-sol, judge gpt-4o-mini (Mem0 unified judge prompt, json label). Stratified round-robin over (conv, category).
Smoke n=8: acc 0.875. Main n=100 (results/20261028-235434.jsonl):
- accuracy 0.770; cat1 0.731 (26), cat2 0.846 (26), cat3 0.696 (23), cat4 0.800 (25)
- evidence recall@20: 0.495 all gold turns retrieved (n=97), 0.580 mean fraction
- tokens: answer 118,596 in / 3,605 out; judge 73,391 in / 3,979 out; total 199,571
- cost answer model: $0.273 ($0.00273/q)
Full 1540: ~1.83M answer-in + 0.056M out tokens, ~$4.2 answer model, ~3.1M tokens incl. judge (judge on 4o-mini extra, small).
Caveats: n=100 gives roughly +-8 points; cat3 gold uses text before first semicolon (Mem0 preprocessing); judge differs from Mem0 only in model choice.

## Retrieval tuning (offline)
Script: retrieval_eval.py (no API calls; reads data/locomo10.json only). Logs: retr_small.log (bge-small), retr_m3.log (bge-m3). Questions: 1536 of the 1540 non-adversarial ones (4 have no parseable evidence id). Baseline reproduces the k=20 run: 0.597 all-gold, 0.653 fraction.
Embedding all 5,882 turns on CPU: bge-small-en-v1.5 36s; bge-m3 280s (about 114s after a 166s load and download).

Table, bge-small for the dense rows. "turns" is the mean number of turns sent at the k=20 / k=40 setting.
| variant | recall@20 | frac@20 | recall@40 | frac@40 | turns |
|---|---|---|---|---|---|
| baseline BM25 | 0.597 | 0.653 | 0.648 | 0.705 | 20 / 40 |
| 1a neighbors +-1 on every top-k hit | 0.712 | 0.773 | 0.767 | 0.827 | 49.5 / 95.8 |
| 1b neighbors +-1, union capped at k | 0.648 | 0.701 | 0.697 | 0.758 | 20 / 40 |
| 2 session-first (best 2 sessions) | 0.583 | 0.632 | 0.661 | 0.711 | 20 / 40 |
| 3 query cleaning + stemming | 0.643 | 0.704 | 0.704 | 0.765 | 20 / 40 |
| 4a dense only | 0.624 | 0.685 | 0.703 | 0.769 | 20 / 40 |
| 4b BM25 + dense, RRF k=60 | 0.642 | 0.707 | 0.716 | 0.782 | 20 / 40 |
| 5a cleaned BM25 + dense RRF | 0.685 | 0.749 | 0.753 | 0.817 | 20 / 40 |
| 5b 5a + neighbors capped at k | 0.681 | 0.742 | 0.760 | 0.823 | 20 / 40 |
| 5c 5a + neighbors on every hit | 0.778 | 0.842 | 0.830 | 0.891 | 49.6 / 95.8 |
| 5d 5a + session-first | 0.592 | 0.649 | 0.614 | 0.670 | 20 / 40 |
| 5e cleaned BM25 + neighbors capped at k | 0.684 | 0.748 | 0.742 | 0.805 | 20 / 40 |

With bge-m3 instead: 4a 0.603/0.694, 4b 0.635/0.695, 5a 0.673/0.733, 5c 0.777/0.827 (recall@20 / @40). bge-m3 is not better than bge-small here and takes 8x longer to embed, so use bge-small.
Same-budget control (bge-m3 5a vs baseline, all-gold): k=50 0.753 vs 0.661; k=96 0.814 vs 0.702. Neighbors (5c, about 50 turns, 0.777) add about 2 points over plain 5a at k=50 (0.753), so most of 1a's gain is just more turns.
Findings: session-first hurts (questions often need turns from more than 2 sessions). Cleaning and stemming alone gives +4.6 points at k=20. Dense alone ties BM25 at k=20 and beats it at k=40. Fusing with the cleaned BM25 gives +8.8 points at k=20 for the same 20 turns.
Token cost: the n=100 run averaged about 1,186 answer-model input tokens per question at 20 turns (about 59 per turn including the prompt). Variant 5a at k=20 costs the same. Variant 5c at about 50 turns would add about 30 turns, roughly +1,780 tokens per question (+150%).

### Answer-model runs: 5a retriever and full context (2026-10-29)
locomo_run.py now has --retriever {baseline,5a}, --full-context and --workers (default 8); retry loop raised to 14 tries with sleeps capped at 30s. Baseline path and output format unchanged. Answer gpt-6-sol, judge gpt-4o-mini, all 1540 questions.
| run | accuracy | cat1 | cat2 | cat3 | cat4 | all-gold recall | answer tokens in / out | answer-model cost |
|---|---|---|---|---|---|---|---|---|
| baseline k=20 | 0.823 | 0.713 | 0.832 | 0.729 | 0.868 | 0.597 | (see run1540-k20.log) | |
| baseline k=40 | 0.858 | 0.794 | 0.882 | 0.719 | 0.887 | 0.648 | 3,529,995 / 56,784 | $7.63 |
| 5a k=40 (results/20261029-001729.jsonl) | 0.911 | 0.876 | 0.922 | 0.750 | 0.937 | 0.753 | 3,453,226 / 51,276 | $7.42 |
| full context (results/20261029-002936.jsonl) | 0.957 | 0.954 | 0.956 | 0.865 | 0.969 | 0.998 | 52,651,729 / 50,176 | $105.81 |
Judge tokens: 5a 1.13M in / 59k out; full 1.13M in / 59k out (gpt-4o-mini, not in the dollar figures). Total tokens: 5a 4.69M, full 53.9M.
Failures: first full-context attempt (8 workers) died at 296 questions on a 429 (gpt-6-sol 4M tokens per minute limit); its log is run1540-full-attempt1-429.log and results/20261029-002510.jsonl is partial and unused. That attempt spent roughly 10M tokens (about $20, estimate) that is not in the figures above. Rerun with 5 workers finished all 1540. Token cap for the full run was raised from 40M to 60M because the estimate was about 42M; real use was 53.9M. No conversation exceeded the context window, no questions skipped. Full-context recall is 0.998, not 1.0, because a few evidence ids in the data do not match any turn id.

## LongMemEval_S
Built: data/longmemeval_s_cleaned.json (277 MB, 500 questions), judge_prompt_lme.py (official judge prompts, verbatim from LongMemEval evaluate_qa.py, abstention handled via "_abs" ids), lme_run.py (5a retriever, bge-small + stemmed BM25 by RRF, k=40 turns, chronological with session dates, answer gpt-6-sol, judge gpt-4o-2024-08-06, 5 workers, --max-usd cap). Embedding cache: lme_emb_cache.pkl (sessions from the 50 smoke questions only).
Ran: one 50-question stratified smoke only (results/lme-20261029-005156.jsonl). Accuracy 0.920 (knowledge-update 1.000 n=9, multi-session 0.778 n=9, single-session-assistant 1.000 n=8, single-session-preference 0.875 n=8, single-session-user 1.000 n=8, temporal-reasoning 0.875 n=8; abstention 0.667 n=3). Session recall 1.000, answer-turn recall 0.936 (n=47). Answer tokens 377,183 in / 3,695 out, $0.79 ($0.0158 per question, about $7.9 projected for 500).
The 500-question run was started and then killed on request during its embedding stage, before any answer or judge call. It produced no results. Log: run-lme.log (ends LME_STOPPED_BY_REQUEST). No further OpenAI calls made. To resume: run lme_run.py with a Claude answer model swapped in.

## Dev/test re-scoring (2026-10-29)
Earlier choices (5a, 5r) were made on all 1,536 questions, which is tuning on test. Redone with split: LoCoMo conv-26, conv-30, conv-41 are dev (383 questions with evidence), the other 7 conversations are test (1153). Selection rule fixed before looking: highest R@5 recall_all on dev. Script split_eval.py (no API calls; reads evidence ids only to score). Hit@1 = top-1 turn is in the evidence set. recall_all = every evidence id in the top k; recall_any = at least one. BM25 here is rank_bm25 (k1=1.5, b=0.75), not bm25s; z-fusion is w*z(cleaned BM25) + (1-w)*z(bge-small), z over all turns of the conversation. Haystack: 419 to 689 turns per conversation (about 19 sessions).
Winner on dev: 5r (bge-reranker-base over the 5a top 100, fused with 5a by RRF). It was also the pick on the old all-1536 table, so the choice does not change. Best z-fusion weight on dev is 0.4 (R@5 all 0.574), below 5r 0.616 and above 5a 0.530 (z-fusion beats RRF on dev at R@5, as the published reports say). Test read once, winner and baseline only.
DEV (conv-26, conv-30, conv-41; 383 questions with evidence)
| variant | Hit@1 | R@5 all | R@5 any | R@10 all | R@10 any | R@40 all | n |
|---|---|---|---|---|---|---|---|
| baseline BM25 | 0.329 | 0.493 | 0.574 | 0.538 | 0.637 | 0.655 | 383 |
| cleaned BM25 | 0.397 | 0.517 | 0.619 | 0.601 | 0.713 | 0.728 | 383 |
| dense bge-small | 0.295 | 0.488 | 0.585 | 0.603 | 0.700 | 0.770 | 383 |
| 5a RRF | 0.373 | 0.530 | 0.637 | 0.642 | 0.747 | 0.775 | 383 |
| 5r rerank+5a | 0.470 | 0.616 | 0.726 | 0.708 | 0.820 | 0.815 | 383 |
| zfuse w=0.3 | 0.413 | 0.572 | 0.684 | 0.674 | 0.773 | 0.830 | 383 |
| zfuse w=0.4 | 0.420 | 0.574 | 0.679 | 0.666 | 0.765 | 0.812 | 383 |
| zfuse w=0.5 | 0.410 | 0.567 | 0.676 | 0.650 | 0.752 | 0.794 | 383 |
| zfuse w=0.6 | 0.410 | 0.561 | 0.663 | 0.645 | 0.752 | 0.775 | 383 |
| zfuse w=0.7 | 0.415 | 0.548 | 0.650 | 0.637 | 0.739 | 0.773 | 383 |

TEST (7 conversations, 1153 questions), read once
| variant | Hit@1 | R@5 all | R@5 any | R@10 all | R@10 any | R@40 all | n |
|---|---|---|---|---|---|---|---|
| baseline BM25 | 0.337 | 0.469 | 0.567 | 0.532 | 0.647 | 0.645 | 1153 |
| 5r rerank+5a | 0.474 | 0.595 | 0.739 | 0.668 | 0.808 | 0.781 | 1153 |

TEST per category, baseline BM25
| variant | Hit@1 | R@5 all | R@5 any | R@10 all | R@10 any | R@40 all | n |
|---|---|---|---|---|---|---|---|
| cat1 | 0.183 | 0.038 | 0.356 | 0.096 | 0.471 | 0.188 | 208 |
| cat2 | 0.355 | 0.528 | 0.597 | 0.589 | 0.675 | 0.714 | 231 |
| cat3 | 0.164 | 0.164 | 0.356 | 0.178 | 0.370 | 0.329 | 73 |
| cat4 | 0.401 | 0.622 | 0.649 | 0.693 | 0.725 | 0.805 | 641 |

TEST per category, 5r rerank+5a
| variant | Hit@1 | R@5 all | R@5 any | R@10 all | R@10 any | R@40 all | n |
|---|---|---|---|---|---|---|---|
| cat1 | 0.404 | 0.135 | 0.683 | 0.221 | 0.750 | 0.423 | 208 |
| cat2 | 0.494 | 0.654 | 0.736 | 0.701 | 0.779 | 0.827 | 231 |
| cat3 | 0.233 | 0.274 | 0.425 | 0.370 | 0.589 | 0.466 | 73 |
| cat4 | 0.518 | 0.760 | 0.794 | 0.835 | 0.863 | 0.916 | 641 |

QA accuracy, answer reader as labelled, judge gpt-4o-mini, 5 categories 1-4, test conversations only (n=1155 incl. 2 questions with no evidence) with dev in brackets:
| run | reader | retriever | test acc | cat1 | cat2 | cat3 | cat4 | dev acc |
|---|---|---|---|---|---|---|---|---|
| 20261029-000510 | gpt-6-sol | baseline k=40 | 0.847 | 0.793 | 0.844 | 0.680 | 0.885 | 0.894 |
| 20261029-001729 | gpt-6-sol | 5a k=40 | 0.904 | 0.865 | 0.909 | 0.733 | 0.934 | 0.932 |
| 20261029-002936 | gpt-6-sol | full context | 0.955 | 0.957 | 0.952 | 0.867 | 0.966 | 0.964 |
| 20261029-020923 | Sonnet 5.5 | 5a k=40 | 0.931 | 0.899 | 0.935 | 0.827 | 0.952 | 0.977 |
Full 1540 for the Sonnet 5a run: 0.942 (cat1 0.922, cat2 0.950, cat3 0.854, cat4 0.956), 4,757,257 answer tokens in / 99,595 out, $10.51, all-gold recall@40 0.753. Variant choice for the QA runs (5a, k=40) was made on all 1,536 questions, so those QA rows were not chosen on a held-out split; the dev-based pick is 5r. 20261029-005057 (1249 of 1540, another process's partial gpt-4o-mini-reader 5a run) is not reported.

### LongMemEval_S, Sonnet 5.5 reader, official judge (gpt-4o-2024-08-06), full 500 (results/lme-20261029-021604.jsonl)
First pass stopped at 460 on the $12 cap (results/lme-20261029-020425.jsonl); the remaining 40 were run with --resume and merged into the new file (460 old rows copied first). 5a retriever, k=40 turns, haystack about 494 turns / 53 sessions per question. Accuracy 0.900 overall: knowledge-update 0.949 (78), multi-session 0.820 (133), single-session-assistant 1.000 (56), single-session-preference 0.800 (30), single-session-user 0.971 (70), temporal-reasoning 0.895 (133). Abstention questions 0.867 (30). Session recall@40 (all gold sessions) 0.984, answer-turn recall 0.942 (n=479). Answer tokens 6,196,224 in / 83,325 out, $13.23 total; judge 116,168 in / 850 out. This is the full 500 including the 30 abstention questions, not split into dev/test; the LongMemEval dev/test split and R@5/R@10/NDCG@10 comparison (task C) are not done yet.

### LoCoMo 5r, Sonnet 5.5 reader, gpt-4o-mini judge (results/20261029-022328.jsonl)
1540 questions, k=40. Accuracy 0.960 (cat1 0.943, cat2 0.960, cat3 0.906, cat4 0.971); test conversations only 0.949 (cat1 0.928, cat2 0.944, cat3 0.880, cat4 0.966), dev 0.992. All-gold recall@40 0.789. Answer tokens 4,913,767 in / 96,919 out, $10.80. Versus 5a on the same reader: 0.942 overall, 0.931 on test, $10.51. 5r was the dev-split winner, so this row's retriever choice is held-out-consistent.

## LongMemEval_S retrieval on the frozen split (task C, 2026-10-29 05:20)

Split: the 500 question_ids sorted by sha256, first 100 dev, 400 test; abstention questions dropped from retrieval scoring (95 dev, 375 test scored). Script lme_split_eval.py. Picked on dev by turn-level R_all@5: 5r. Test, read once, session level R_any@5 / R_all@5 / NDCG_any@10: BM25 (bm25s) 84.0 / 68.3 / 65.6; dense bge-small 92.3 / 76.3 / 64.9; z-fusion 93.6 / 80.0 / 73.2; 5a 95.5 / 81.6 / 72.2; 5r 97.6 / 87.2 / 80.9. Turn level: BM25 78.4 / 55.2 / 64.1; 5r 93.1 / 73.6 / 79.1. Full table in reports/v3/fm-protocol-2.md (Results of Addendum 22, task C) and RUNS.md. The QA accuracy of 90.0 above is on all 500 and is not split.
