#!/bin/zsh
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/resume-128k
D=(.venv/bin/python -u metal/tools/determinism.py --config strata-iq3_s.json --once)
echo "== one pass of 8,192 tokens, top-k emitting in index order"
$D --out $O/det3-8k --tokens 8195 "a:--expert-cache-cap 6000 --verbose" "b:--expert-cache-cap 6000 --verbose"
echo "== 10,000 tokens (8,192 + 1,805), caches of 6,000 and 3,000 MiB"
$D --out $O/det3-10k --tokens 10000 "a:--expert-cache-cap 6000 --verbose" "b:--expert-cache-cap 3000 --verbose"
