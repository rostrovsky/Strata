#!/bin/zsh
# abba.sh: the final build with the KV hold (A, the installed configuration as is) against the same build without
# it (B, STRATA_KV_HOLD=0), A B B A, the 130,944-token prompt of sections 8, 9, 14, 18 and 19; each run under
# metal/tools/swapwatch.py with a 1 Hz vm_stat (build-metal/resume-128k/trace_run.sh), the SoC cooled to 45 C before
# each start, the Mac kept awake. Stops before a run if anything else holds the engine's place (the server on 8080,
# another strata, another run), and after a run that moved the swap counter or did not end cleanly.
cd /Users/rafal/Projects/strata-air/Strata
caffeinate -i -w $$ &
O=build-metal/resume-128k
for N in kvh1-on kvh2-off kvh3-off kvh4-on; do
  if lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 || pgrep -f 'bin/strata|/strata |bench.py|determinism.py|turns.py' >/dev/null; then
    echo "$(date +%H:%M:%S) STOP before $N: something else is running"; pgrep -fl 'bin/strata|/strata |bench.py|determinism.py|turns.py'
    exit 2
  fi
  echo "$(date +%H:%M:%S) START $N"
  if [[ $N == *-off ]]; then H=0; else H=1; fi
  STRATA_KV_HOLD=$H build-metal/resume-128k/trace_run.sh $N --prefill 130944 --prefill-new 120 --needle --prefill-repeat --cool 45 -- --verbose
  D=$(grep DONE $O/$N-watch.txt | tail -1)
  echo "$(date +%H:%M:%S) END $N (STRATA_KV_HOLD=$H): $D"
  if [[ $D != *"new swap-outs over the whole run 0 pages"* || $D != *"stopped: False"* || $D != *"exit code 0"* ]]; then
    echo "$(date +%H:%M:%S) STOP after $N: swap-outs, a stop or an error"
    exit 3
  fi
done
echo "$(date +%H:%M:%S) ALL DONE"
