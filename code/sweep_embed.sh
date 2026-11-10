#!/bin/zsh
cd "$(dirname $0)"
C="--tier 500K --limit 15 --pair --standing 5 --compact --ks 20,60 --rerank-n 100"
run() { .venv/bin/python beam_retr.py ${=C} "$@" 2>&1 | grep -v -i -E "warn|loading|weights|redirect" > "sweep-$1-$2-$3-$4.log"; }
run --dense bge-small --quant none --rerank bge-reranker-base &
run --dense bge-small --quant int8 --rerank bge-reranker-base &
wait
run --dense bge-base --quant int8 --rerank bge-reranker-base &
run --dense bge-large --quant int8 --rerank bge-reranker-base &
wait
run --dense bge-large --quant int8 --rerank bge-reranker-v2-m3 &
run --dense bge-m3 --quant int8 --rerank bge-reranker-base &
wait
echo SWEEP_DONE
