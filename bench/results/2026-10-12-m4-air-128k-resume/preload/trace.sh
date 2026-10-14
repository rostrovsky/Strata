#!/bin/zsh
# trace.sh NAME starts.py-args... : starts.py under the swap watch, with a 1 Hz vm_stat trace beside it (as
# ../refill/trace_any.sh); everything goes to build-metal/preload-start/
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/preload-start
N=$1; shift
( vm_stat 1 | while IFS= read -r line; do print -r -- "$(date +%H:%M:%S) $line"; done > $O/$N-vmstat.txt ) &
VMPID=$!
.venv/bin/python metal/tools/swapwatch.py $N --out $O -- .venv/bin/python -u bench/results/2026-10-12-m4-air-128k-resume/preload/starts.py --config strata-iq3_s.json --out $O --name $N "$@"
rc=$?
pkill -P $VMPID vm_stat 2>/dev/null; kill $VMPID 2>/dev/null
echo "watch rc=$rc"
