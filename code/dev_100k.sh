#!/bin/zsh
cd "$(dirname $0)"
export OPENAI_BASE_URL="${OPENAI_BASE_URL}"
D=/Users/nikhilkrishnaswamy/Documents/fm-data/beam/beam_100K.jsonl
C="--data $D --convs 0-19 --n 200 --adaptive 20,40 --answer-provider openai --answer-model gpt-5-mini --rerank --pair --standing 5 --max-usd 15 --workers 3"
.venv/bin/python beam_run_v2.py ${=C} --tag dev100-base > dev100-base.log 2>&1 &
.venv/bin/python beam_run_v2.py ${=C} --plan --compact --rules2 --session-cap 6 --quant int8 --store beam_store/100K --tag dev100-locked > dev100-locked.log 2>&1 &
wait
