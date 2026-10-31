# Memory Without a Writer: Agent Memory Built by Code

Nikhil Krishnaswamy (Stanford University, Memorable) and Advaiyt Sane (Carnegie Mellon University, Memorable).

This repository holds everything behind the paper so that the numbers can be audited: the LaTeX source and figures, the code that built the store and ran every evaluation, the per-question result rows for every run, and the protocol file in which each run was registered with a prediction before it started.

The paper studies a memory for language-model agents in which no model writes anything. Conversation turns, or whole agent trajectories, are stored exactly as they were and indexed by BM25 and by a local embedding model; at question time a fused score, two dialogue rules and two local cross-encoders reorder the stored units, and a reader model answers from the top of the list. The paper compares this store with Mem0 under Mem0's own reader, judge and prompts on LoCoMo, and reports LongMemEval-S, BEAM and Proced-Mem.

## Layout

| Path | What it holds |
|---|---|
| `memory-without-a-writer.pdf` | The paper as built from this repository. |
| `paper/` | `main.tex`, `refs.bib`, `sections/*.tex`, `figs/` (every figure as PDF and PNG, the CSV of values each figure plots, `make_figs.py`, the TikZ source of the method figure in `figs/variants/method.tex`, and the XCharter font files the figures use). |
| `code/` | The Python scripts that built the indexes, ranked turns, ran the readers and judges, and computed the statistics. `code/results/` holds one JSONL row per question for every answer run and every judge pass, including the one-judge re-scoring. `code/snap_rank*.json` are the ranked turn lists the reader was given. `code/split_*.json` and `code/lme_split_*.json` fix the development and test splits. `code/logs/` are the run logs. `code/NOTES.md`, `code/BEAM-NOTES.md` and `code/RUNS.md` are the working notes. |
| `protocol/protocol.md` | The registrations. Each addendum was written before its run, with numbered predictions and ranges; the results sections were appended afterwards. Appendix A of the paper is read from this file. |

## Building the paper

```
cd paper
tectonic -X compile main.tex
```

Tectonic fetches the LaTeX packages it needs. The figures are prebuilt; to regenerate them, run `python figs/make_figs.py` from `paper/` with `matplotlib`, `numpy` and the result files in `code/results/` in place (the script reads them by relative path, so keep the layout above).

## Reproducing a number

Every value in the paper comes from a file in `code/results/` or `paper/figs/*.csv`, and every figure script writes the values it plots to a CSV beside the figure. The statistics in the paper (conversation-cluster bootstraps, Wilson intervals, exact McNemar counts) are computed by `code/h2h_cats.py`, `code/cascade_test.py` and the figure script. The one-judge re-scoring is `code/claude_judge.py`, with its outputs in `code/results/claude-judge-*.jsonl`.

Running a reader or judge again needs API keys for the model providers, read from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); the scripts print a token-priced cost estimate before they start.

## Data this repository does not include

Benchmark data are not redistributed. `code/data/` is expected to hold `locomo10.json` from the LoCoMo release and `longmemeval_s_cleaned.json` from the LongMemEval release; BEAM is loaded through its own package. Mem0's published per-question result files are read from a clone of `github.com/mem0ai/memory-benchmarks` (the scripts look for it under your home directory; change the `MEM0` path at the top of `code/h2h_cats.py` and `code/mem0_to_rows.py` if yours is elsewhere). Embedding caches (`*.pkl`, `*.npz`) are not included and are rebuilt on first run.

## License

The code and the paper source are released under the MIT License (see `LICENSE`). Benchmark data and the published result files of other systems remain under their own licenses.
