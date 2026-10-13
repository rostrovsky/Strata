#!/bin/zsh
# trace_any.sh NAME command... : any command under the swap watch, with a 1 Hz vm_stat trace beside it (as trace_run.sh
# does for bench.py); the files go to build-metal/refill/
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/refill
N=$1; shift
( vm_stat 1 | while IFS= read -r line; do print -r -- "$(date +%H:%M:%S) $line"; done > $O/$N-vmstat.txt ) &
VMPID=$!
.venv/bin/python metal/tools/swapwatch.py $N --out $O -- "$@"
rc=$?
pkill -P $VMPID vm_stat 2>/dev/null; kill $VMPID 2>/dev/null
echo "watch rc=$rc"
