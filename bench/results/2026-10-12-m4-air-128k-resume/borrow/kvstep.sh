#!/bin/zsh
# kvstep.sh: a step of the KV cache while the expert cache is there (STRATA_KV_STEP=1024, so that a short
# conversation crosses one), HEAD's engine (A: the cache is made anew, smaller) against the hold (B), each engine
# start afresh, no cache cap (with a cap the step is paid from what the cap left unused and A would not touch the
# cache either); IDs and logits compared with the first run.  Run it under trace_any.sh.
#   answer:   a 30-token question and 800 tokens written, fed in (--force: the model ended its own answer after 178
#             tokens): the step comes in the middle of the answer (token ~738)
#   kv700:    as refill/kvstep.sh: 700 tokens (one pass beside the cache, if the memory for it is free), 120 written
#   kv900:    900 tokens in one pass, 64 written: the step comes before the pass
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/borrow/runs/kvstep
R=(.venv/bin/python -u bench/results/2026-10-12-m4-air-128k-resume/borrow/runs.py answer --config strata-iq3_s.json --cool 45)
A="STRATA_KV_STEP=1024 build-metal/borrow/strata-head"
B="STRATA_KV_STEP=1024 build-metal/borrow/strata"
echo "== answer: 30 tokens, 800 written, A B B A"
$R --out $O/answer --tokens 30 --new 800 --force $A $B $B $A -- --verbose
echo "== kv700: 700 tokens, 120 written, A B"
$R --out $O/kv700 --tokens 700 --new 120 $A $B -- --verbose
echo "== kv900: 900 tokens, 64 written, A B"
$R --out $O/kv900 --tokens 900 --new 64 $A $B -- --verbose
