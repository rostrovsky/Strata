#!/bin/zsh
# real-model exact gate at a small, capped cache: reference binary, then the final binary's default, indexed and elastic paths
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/resume-128k
so() { vm_stat | awk '/Swapouts/{gsub("\\.","",$2); print $2}'; }
common=(--config strata-iq3_s.json --context 32768 --new 32 --only chat --only code --only doc8k --ref $O/gate-ref.json)
tail_args=(-- --kv f16 --prefill 4096 --reserve-gib 4 --expert-cache-cap 3000 --chunk 2)
s0=$(so)
.venv/bin/python -u metal/tools/bench.py $common --engine build-metal/lowlevel-128k/strata-reference --gate record --log $O/gate-ref.log --json $O/gate-ref-run.json $tail_args > $O/gate-ref.out 2>&1
echo "record rc=$? swapouts=$(( $(so) - s0 ))"
[ $(( $(so) - s0 )) -gt 0 ] && { echo "swap-outs: stopping"; exit 3; }
STRATA_METAL_INDEXED_QSA=0 .venv/bin/python -u metal/tools/bench.py $common --gate check --exact --log $O/gate-default.log --json $O/gate-default.json $tail_args > $O/gate-default.out 2>&1
echo "default rc=$? swapouts=$(( $(so) - s0 ))"
[ $(( $(so) - s0 )) -gt 0 ] && { echo "swap-outs: stopping"; exit 3; }
STRATA_METAL_INDEXED_QSA=1 .venv/bin/python -u metal/tools/bench.py $common --gate check --exact --log $O/gate-indexed.log --json $O/gate-indexed.json $tail_args > $O/gate-indexed.out 2>&1
echo "indexed rc=$? swapouts=$(( $(so) - s0 ))"
[ $(( $(so) - s0 )) -gt 0 ] && { echo "swap-outs: stopping"; exit 3; }
STRATA_METAL_INDEXED_QSA=1 .venv/bin/python -u metal/tools/bench.py $common --gate check --exact --log $O/gate-elastic.log --json $O/gate-elastic.json $tail_args --elastic on --verbose > $O/gate-elastic.out 2>&1
echo "elastic rc=$? swapouts=$(( $(so) - s0 ))"
