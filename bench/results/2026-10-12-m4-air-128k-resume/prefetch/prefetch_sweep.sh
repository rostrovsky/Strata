#!/bin/zsh
# prefetch_sweep.sh: how many experts per layer to read ahead, now that the cache is 10-11 GiB (10 was tuned with 6 GiB).
# Interleaved (A B C, then C B A), installed configuration, under the swap watch with a vm_stat trace.
# Run only when no other engine is loaded.
cd /Users/rafal/Projects/strata-air/Strata
E=build-metal/bin/strata
build-metal/resume-128k/trace_run.sh prefetch-sweep --prefill 500,2000 --prefill-new 96 --cool 45 \
  --ab "$E --prefetch 10" "$E --prefetch 14" "$E --prefetch 18" --runs 2 -- --verbose
python3 build-metal/resume-128k/memsum.py build-metal/resume-128k/prefetch-sweep
