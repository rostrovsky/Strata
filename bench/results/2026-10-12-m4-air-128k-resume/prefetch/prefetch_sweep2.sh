#!/bin/zsh
# prefetch_sweep2.sh: --prefetch N above 10 does nothing by itself (the predictor stops at the predicted 10th score:
# --prefetch-margin 0).  Here the 16 experts it looks at are let through with a negative margin.  Same cache in
# every start (--expert-cache-cap), interleaved, under the swap watch with a vm_stat trace.
cd /Users/rafal/Projects/strata-air/Strata
E=build-metal/bin/strata
build-metal/resume-128k/trace_run.sh prefetch-sweep2 --prefill 500,2000 --prefill-new 96 --cool 45 \
  --ab "$E" "$E --prefetch 16 --prefetch-margin -0.5" "$E --prefetch 16 --prefetch-margin -8" --runs 2 -- --expert-cache-cap 10000 --verbose
python3 build-metal/resume-128k/memsum.py build-metal/resume-128k/prefetch-sweep2
