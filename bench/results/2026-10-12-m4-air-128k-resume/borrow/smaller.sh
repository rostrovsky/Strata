#!/bin/zsh
# smaller.sh C: what a cache one KV step (16,384 cells, 458 MiB) smaller costs in written tok/s: the same build with
# --expert-cache-cap C and C-458, interleaved (A B, B A, A B), the SoC cooled to 45 C before every start; three short
# requests of the benchmark, 200 tokens written each.  Run it under trace_any.sh.
cd /Users/rafal/Projects/strata-air/Strata
C=${1:-6500}
.venv/bin/python -u metal/tools/bench.py --config strata-iq3_s.json --decode --only chat --only code --only reason --new 200 \
    --cool 45 --runs 3 --json build-metal/borrow/runs/smaller.json --log build-metal/borrow/runs/smaller-engine.txt \
    --ab "build-metal/borrow/strata --expert-cache-cap $C" "build-metal/borrow/strata --expert-cache-cap $(( C - 458 ))"
