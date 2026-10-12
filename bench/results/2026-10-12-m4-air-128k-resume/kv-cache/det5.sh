#!/bin/zsh
# the two ways the expert cache gives memory to the KV cache while it is there: before a short prompt, and in the middle of an answer
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/resume-128k
D=(.venv/bin/python -u metal/tools/determinism.py --config strata-iq3_s.json --once)
echo "== 900 tokens in one short pass; with a step of 1,024 cells the cache gives memory before it"
$D --out $O/det5-900 --tokens 900 "all-at-once:STRATA_KV_STEP=0 --verbose" "step-1024:STRATA_KV_STEP=1024 --verbose"
echo "== 700 tokens, then 120 written; with a step of 1,024 cells the cache gives memory in the middle of the answer"
$D --out $O/det5-700 --tokens 700 --new 120 "all-at-once:STRATA_KV_STEP=0 --verbose" "step-1024:STRATA_KV_STEP=1024 --verbose"
