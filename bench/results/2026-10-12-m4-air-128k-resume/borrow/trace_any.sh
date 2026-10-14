#!/bin/zsh
# trace_any.sh NAME command... : any command under the swap watch (metal/tools/swapwatch.py), with a 1 Hz vm_stat
# trace and a 1 Hz memtrace (free pages, file cache and the kernel's floor for it) beside it; the files go to
# build-metal/borrow/runs/ (as refill/trace_any.sh); the Mac is kept from sleeping meanwhile (caffeinate: a run of
# 2026-10-14 had a 337 s gap where it slept)
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/borrow/runs
mkdir -p $O
N=$1; shift
caffeinate -i -w $$ &
( vm_stat 1 | while IFS= read -r line; do print -r -- "$(date +%H:%M:%S) $line"; done > $O/$N-vmstat.txt ) &
VMPID=$!
build-metal/borrow/bin/memtrace > $O/$N-memtrace.txt &
MTPID=$!
.venv/bin/python metal/tools/swapwatch.py $N --out $O -- "$@"
rc=$?
pkill -P $VMPID vm_stat 2>/dev/null; kill $VMPID $MTPID 2>/dev/null
echo "watch rc=$rc"
