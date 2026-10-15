#!/bin/zsh
# keep the Mac awake for the whole run, then the same trace as section 14 (swap guard + 1 Hz vm_stat)
caffeinate -i -w $$ &
/Users/rafal/Projects/strata-air/Strata/build-metal/resume-128k/trace_run.sh full-128k-1015 \
  --prefill 130944 --prefill-new 120 --needle --prefill-repeat --cool 42 -- --verbose
