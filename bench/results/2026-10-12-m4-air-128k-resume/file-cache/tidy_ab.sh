#!/bin/zsh
# tidy_ab.sh: old binary (file cache kept, rows by page fault) against the new one, A B B A, every run from a file
# cache without the model in it, under the swap watch with a 1 Hz vm_stat trace
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/resume-128k
M=../Strata-data/models/IQ3_S
run() {
  local name=$1; shift
  $O/resident -q --evict $M/*.gguf > /dev/null
  $O/trace_run.sh $name --prefill 500,2000,10000 --prefill-new 24 --cool 45 "$@" -- --verbose 2>&1 | tail -2
  echo "$name: after the run:"; $O/resident -q $M/*.gguf
}
run tidy-a1-old --engine $O/strata-before-tidy
run tidy-b1-new
run tidy-b2-new
run tidy-a2-old --engine $O/strata-before-tidy
