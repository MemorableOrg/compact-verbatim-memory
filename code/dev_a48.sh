#!/bin/zsh
# dev only (BEAM 500K), never 1M: A47 base vs +plan vs +plan+compact, gpt-5-mini reader, BEAM judge gpt-4.1-mini, both on Azure
cd "$(dirname $0)"
export OPENAI_BASE_URL="${OPENAI_BASE_URL}"
D=/Users/nikhilkrishnaswamy/Documents/fm-data/beam/beam_500K.jsonl
C="--data $D --convs 0-14 --n 200 --adaptive 20,40 --answer-provider openai --answer-model gpt-5-mini --rerank --pair --standing 5 --max-usd 12 --workers 4"
.venv/bin/python beam_run_v2.py ${=C} --tag dev500-base > dev500-base.log 2>&1 &
.venv/bin/python beam_run_v2.py ${=C} --plan --tag dev500-plan > dev500-plan.log 2>&1 &
.venv/bin/python beam_run_v2.py ${=C} --plan --compact --tag dev500-plan-compact > dev500-plan-compact.log 2>&1 &
wait
