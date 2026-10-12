#!/bin/zsh
# trace_run.sh NAME bench-args... : one bench.py run under the swap watch, with a 1 Hz vm_stat trace beside it
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/resume-128k
N=$1; shift
( vm_stat 1 | while IFS= read -r line; do print -r -- "$(date +%H:%M:%S) $line"; done > $O/$N-vmstat.txt ) &
VMPID=$!
.venv/bin/python metal/tools/swapwatch.py $N --out $O -- .venv/bin/python -u metal/tools/bench.py --config strata-iq3_s.json --json $O/$N.json --log $O/$N.log "$@"
rc=$?
pkill -P $VMPID vm_stat 2>/dev/null; kill $VMPID 2>/dev/null   # (only this run's trace: another run may be under way)
echo "watch rc=$rc"
