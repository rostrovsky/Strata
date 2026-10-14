#!/bin/zsh
# final.sh: the real-model runs of the final build (build-metal/borrow/strata), each series under trace_any.sh
cd /Users/rafal/Projects/strata-air/Strata
B=bench/results/2026-10-12-m4-air-128k-resume/borrow
O=build-metal/borrow/runs
# 1. the three-turn conversation, HEAD's engine against this one, the same 6,500 MiB cache in every start
$B/trace_any.sh final-turns .venv/bin/python -u $B/runs.py turns --config strata-iq3_s.json --out $O/final-turns --cool 45 --runs 2 \
    --ab "build-metal/borrow/strata-head" "build-metal/borrow/strata" -- --expert-cache-cap 6500 --verbose
# 2. a KV step in the middle of an answer (fed in), no cache cap; --chunk 8: the same passes for the question in every start
$B/trace_any.sh final-kvanswer .venv/bin/python -u $B/runs.py answer --config strata-iq3_s.json --cool 45 --out $O/final-kvanswer \
    --tokens 30 --new 800 --force "STRATA_KV_STEP=1024 build-metal/borrow/strata-head" "STRATA_KV_STEP=1024 build-metal/borrow/strata" \
    "STRATA_KV_STEP=1024 build-metal/borrow/strata" "STRATA_KV_STEP=1024 build-metal/borrow/strata-head" -- --chunk 8 --verbose
# 3. what a cache one KV step smaller costs
$B/trace_any.sh final-smaller $B/smaller.sh 6500
# 4. the exact gate
ENGINE=build-metal/borrow/strata $B/trace_any.sh final-gate $B/gate/gate.sh
