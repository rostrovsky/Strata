#!/bin/zsh
# reserve_runs.sh R...: the installed configuration with --reserve-gib R, each under the swap watch with a vm_stat trace
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/resume-128k
for r in "$@"; do
  $O/trace_run.sh reserve-$r --prefill 500,2000,10000 --prefill-new 24 --cool 45 -- --reserve-gib $r --verbose 2>&1 | tail -2
done
