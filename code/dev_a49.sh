#!/bin/zsh
cd "$(dirname $0)"
export OPENAI_BASE_URL="${OPENAI_BASE_URL}"
D=/Users/nikhilkrishnaswamy/Documents/fm-data/beam/beam_500K.jsonl
C="--data $D --convs 0-14 --n 200 --adaptive 20,40 --answer-provider openai --answer-model gpt-5-mini --rerank --pair --standing 5 --plan --compact --max-usd 18 --workers 4"
.venv/bin/python beam_run_v2.py ${=C} --rules2 --tag dev500-a49-rules > dev500-a49-rules.log 2>&1 &
.venv/bin/python beam_run_v2.py ${=C} --rules2 --session-cap 6 --tag dev500-a49-rules-cap > dev500-a49-rules-cap.log 2>&1 &
wait
