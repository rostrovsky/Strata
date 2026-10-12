#!/bin/zsh
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/resume-128k
D=(.venv/bin/python -u metal/tools/determinism.py --config strata-iq3_s.json --once)
echo "== one pass of 8,192 tokens"
$D --out $O/det2-8k --tokens 8195 "a:--expert-cache-cap 6000 --verbose" "b:--expert-cache-cap 6000 --verbose"
echo "== one pass of 4,096 tokens"
$D --out $O/det2-4k --tokens 4099 "a:--expert-cache-cap 6000 --verbose" "b:--expert-cache-cap 6000 --verbose"
echo "== the binary from before this evening's changes, one pass of 8,192 tokens"
$D --out $O/det2-old8k --tokens 8195 "a:EXE=$O/strata-before-tidy --expert-cache-cap 6000 --verbose" "b:EXE=$O/strata-before-tidy --expert-cache-cap 6000 --verbose"
