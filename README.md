# Memorable: Compact Verbatim Memory for Long Conversations

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](memorable.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Nikhil Krishnaswamy (Stanford University, Memorable) and Advaiyt Sane (Carnegie Mellon University, Memorable).

This repository holds the paper, the code that ran every experiment, the per-question results of every run, and the ledger in which each run was registered with a prediction before it started. Memorable is a memory for long conversations built by fixed code from the turns themselves: every turn is kept as it was said, compacted by three rules, stored with zlib beside an 8-bit embedding, and searched at question time by a small planner, hybrid retrieval and a local cross-encoder. The paper compares it with Mem0 on BEAM under one judge for both systems and reports retrieval on LongMemEval-S.

## Results

Pass rate under one judge (Claude Sonnet 5.5 with BEAM's rubric prompt, scoring both systems' answers). Mem0's answers are the per-question file it published for its Platform v3 run with GPT-5. A gap whose 95% interval includes zero is a tie.

| Benchmark | Questions | Mem0 | Memorable | Gap and interval |
|---|---|---|---|---|
| BEAM-1M | 700 over 35 conversations | 63.9 | 67.3 | +3.4, [-0.4, +7.3], a tie |
| BEAM-10M | 200 over 10 conversations | 46.5 | 56.0 | +9.5, [+0.5, +18.0], ahead in all 10 |
| BEAM-1M, contradiction resolution | 70 | 7.1 | 58.6 | |

On LongMemEval-S, retrieval alone places every gold session in the top 10 for 95.5% of the 375 test questions; EdgeMem and MemGAS report 93.8% and 94.5%.

Building the memory of one 1M-token conversation with Mem0's open-source release costs about 855 model calls, $7.50 and 5.4 hours and takes 80 MB on disk (measured over 80% of the conversation and scaled). Memorable builds the same conversation in about 30 seconds of CPU and stores 1.47 MB. Memorable's reader sees more tokens per question, so Mem0 is cheaper in total after about 460 questions on a conversation.

## How it works

1. Write path, once per conversation and with zero model calls: three compaction rules (markdown, filler sentences, repeated assistant sentences), zlib over the text, and one 8-bit bge-small vector per turn. `code/compact.py` holds the rules.
2. Search path, once per question: a gpt-5-mini planner names the question type and proposes two to four queries; BM25 and embedding rankings for every query are fused by reciprocal rank; each turn lends part of its score to its partner in the exchange; bge-reranker-base rescores the top 100; the user's standing instructions go first and at most six turns come from one session.
3. A frozen reader (GPT-5) answers from 20 turns, or 60 for ordering, summary and aggregate questions, under fixed rules (Appendix D of the paper).

## Layout

| Path | Contents |
|---|---|
| `memorable.pdf` | The paper as built from `paper/`. |
| `paper/` | `main.tex`, `preamble.tex`, `refs.bib`, `sections/*.tex`, `figs/` (each figure as PDF, the CSV of the values it plots where the script writes one, `paper_figs_classic.py`, the TikZ source of the method figure in `figs/method/`, and the XCharter fonts the figures use). |
| `code/` | The scripts behind every run. `beam_run_v2.py` is the BEAM runner (store, search, reader, BEAM's judge in the loop); `claude_judge.py` scores existing answers with the one judge; `beam_cmp_claude.py`, `paired_check.py` and `cmp10m.py` compute the gaps and bootstrap intervals; `beam_retr.py` is the offline retrieval check behind the 8-bit comparison; `mem0_oss_beam.py` builds Mem0's memory with its open-source release and meters every call; `lme_split_eval.py`, `lme_recall.py` and `lme_policy.py` are the LongMemEval-S retrieval measurements; `dev_*.sh` are the development runs on the 100K and 500K tiers. |
| `results/` | One JSONL row per question for every answer run and every judge pass, the paired verdict CSVs (`beam1m-one-judge-verdicts-compact.csv`, `beam10m-one-judge-verdicts-compact.csv`, `ladder-dev500-shared-verdicts.csv`, `int8-check-paired-rows.csv`), the LongMemEval-S split results, the Mem0 build meter (`beam-mem0oss-c0.cost.json`) and the run and judge logs with their cost lines. |
| `protocol/ledger.md` | The run ledger. Every prediction was written before its run; results and decisions were appended afterwards, with the time of day. Appendix A of the paper is read from this file. Two review links and five mail ids were removed before release; nothing else was changed. |

## Setup

Python 3.12. The exact package versions of the run environment are in `requirements.txt`.

```bash
git clone https://github.com/MemorableOrg/compact-verbatim-memory.git
cd compact-verbatim-memory
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Model calls read their keys from the environment: `OPENAI_API_KEY` (or `OPENROUTER_API_KEY` with `USE_OPENROUTER=1`, or the `AZURE_OPENAI_*` variables; see `code/oa.py`) for the reader, planner and BEAM's judge, and `ANTHROPIC_API_KEY` for the one judge. Every script that calls a model prints a token-priced cost estimate first and stops at `--max-usd`.

Benchmark data are not redistributed. The runner expects BEAM's conversations as one JSONL row per conversation at `~/Documents/fm-data/beam/beam_1M.jsonl` (and `beam_100K.jsonl`, `beam_500K.jsonl`, `beam_10M.jsonl`; `code/convert_10m.py` writes the 10M file from BEAM's release), LongMemEval-S as `data/longmemeval_s_cleaned.json`, and Mem0's published per-question files from a clone of [mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks) at `~/Documents/fm-data/beam-repos/mem0mb`. Change the paths at the top of the scripts if yours differ.

## Reproducing the runs

The 1M full method (700 questions, GPT-5 reader, BEAM's judge in the loop), then the one judge on both files:

```bash
cd code
python beam_run_v2.py --data ~/Documents/fm-data/beam/beam_1M.jsonl --n 700 --adaptive 20,40 \
  --answer-provider openai --answer-model gpt-5 --rerank --pair --standing 5 \
  --plan --compact --rules2 --session-cap 6 --quant int8 --store beam_store/1M --tag final
python claude_judge.py beam ours results/beam-<timestamp>-final.jsonl
python claude_judge.py beam mem0 ~/Documents/fm-data/beam-repos/mem0mb/results/platform/beam_1m_results.json
python beam_cmp_claude.py results/claude-judge-beam-ours-<...>.jsonl results/claude-judge-beam-mem0-beam_1m_results.jsonl
```

Base retrieval is the same command without `--plan --compact --rules2 --session-cap 6 --quant int8`. The 10M run uses `beam_10M.jsonl` with `--n 200` and the same flags. The development ladder on the 500K tier is `dev_a48.sh` and `dev_a49.sh` (gpt-5-mini reader, BEAM's judge). The paired and independent bootstraps on the published verdict CSVs are `paired_check.py` (1M) and `cmp10m.py` (10M).

Mem0's build cost, with its open-source release (mem0ai 2.0.0), gpt-5-mini as extractor and a $6 cap:

```bash
python mem0_oss_beam.py --convs 0 --max-usd 6
```

The 8-bit against float32 check (offline, 269 development questions of the 500K tier):

```bash
python beam_retr.py --tier 500K --limit 15 --dense bge-small --rerank bge-reranker-base --pair --compact --ks 20,60 --out results/int8-check-none.jsonl
python beam_retr.py --tier 500K --limit 15 --dense bge-small --rerank bge-reranker-base --pair --compact --quant int8 --ks 20,60 --out results/int8-check-int8.jsonl
```

LongMemEval-S retrieval by split (offline): `python lme_split_eval.py test` and `python lme_split_eval.py dev`.

The figures: `cd paper/figs && python paper_figs_classic.py all` reads the result files in `results/` and rewrites every figure and its CSV. The paper: `cd paper && tectonic main.tex`.

## Citation

```bibtex
@article{krishnaswamy2026memorable,
  title={Memorable: Compact Verbatim Memory for Long Conversations},
  author={Krishnaswamy, Nikhil and Sane, Advaiyt},
  journal={arXiv preprint},
  year={2026}
}
```

## License

The code and the paper source are released under the MIT License (see `LICENSE`). BEAM, LongMemEval and Mem0's published result files remain under their own licenses; the answer files in `results/` repeat BEAM's questions and rubrics for the questions we answered.
