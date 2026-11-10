#!/bin/zsh
cd "$(dirname $0)"
export OPENAI_BASE_URL="${OPENAI_BASE_URL}"
B=/Users/nikhilkrishnaswamy/Documents/fm-data/beam
C="--n 200 --adaptive 20,40 --answer-provider openai --answer-model gpt-5-mini --rerank --pair --standing 5 --plan --compact --rules2 --session-cap 6 --quant int8 --max-usd 6 --workers 3"
.venv/bin/python beam_run_v2.py --data $B/beam_100K.jsonl --convs 0-19 ${=C} --tag dev100-locked-v2 > dev100-locked-v2.log 2>&1 &
.venv/bin/python beam_run_v2.py --data $B/beam_500K.jsonl --convs 0-14 ${=C} --tag dev500-locked-v2 > dev500-locked-v2.log 2>&1 &
wait
