#!/bin/zsh
# gate_preload.sh: as ../borrow/gate/gate_head.sh.  HEAD's engine (built from HEAD's metal/src into
# build-metal/preload-start/head/strata-head) records the gate suite (chat, code, the 7,900-token document) on its
# plain default path; this build is held against it exactly: its default start (the cache filled from the profile),
# then a start with nothing filled (--preload 0) on the plain, indexed, --elastic and --elastic --prefill 8192 paths,
# and a 1 GiB fill with --elastic.  The plain config, a 3,000 MiB cache (GATE_CAP), --chunk 2; the cache's size in
# experts is printed for every engine and must be the same.
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/preload-start/gate
mkdir -p $O
REF=${REF:-build-metal/preload-start/head/strata-head}
E=${ENGINE:-build-metal/preload-start/strata-final}
so() { vm_stat | awk '/Swapouts/{gsub("\\.","",$2); print $2}'; }
common=(--config bench/results/2026-10-12-m4-air-128k-resume/pending/desktop-128k.json --context 32768 --new 32 --only chat --only code --only doc8k --ref $O/gate-ref.json)
tail_args=(-- --kv f16 --prefill 4096 --reserve-gib 2 --expert-cache-cap ${GATE_CAP:-3000} --chunk 2)
s0=$(so)
STRATA_METAL_INDEXED_QSA=0 .venv/bin/python -u metal/tools/bench.py $common --engine $REF --gate record --log $O/gate-ref.log --json $O/gate-ref-run.json $tail_args > $O/gate-ref.out 2>&1
echo "record (HEAD's default path) rc=$? swapouts=$(( $(so) - s0 )): $(grep -o 'expert cache [0-9]* experts' $O/gate-ref.out)"
run() {   # name, environment..., then -- and extra engine options
  local name=$1; shift
  local envs=(); while [[ $1 != -- ]]; do envs+=($1); shift; done; shift
  env $envs .venv/bin/python -u metal/tools/bench.py $common --engine $E --gate check --exact --log $O/gate-$name.log --json $O/gate-$name.json $tail_args "$@" > $O/gate-$name.out 2>&1
  echo "$name rc=$? swapouts=$(( $(so) - s0 )): $(grep GATE $O/gate-$name.out | tail -1); $(grep -o 'expert cache [0-9]* experts' $O/gate-$name.out); $(grep -o 'filled with the profile.s [0-9]* most used\|starts empty' $O/gate-$name.log | head -1)"
  if [ $(( $(so) - s0 )) -gt 0 ]; then echo "swap-outs: stopping"; exit 3; fi
}
run default STRATA_METAL_INDEXED_QSA=0 --
run empty STRATA_PRELOAD=0 STRATA_METAL_INDEXED_QSA=0 --
run empty-indexed STRATA_PRELOAD=0 STRATA_METAL_INDEXED_QSA=1 --
run empty-elastic STRATA_PRELOAD=0 STRATA_METAL_INDEXED_QSA=1 -- --elastic on --verbose
run empty-elastic8k STRATA_PRELOAD=0 STRATA_METAL_INDEXED_QSA=1 -- --prefill 8192 --elastic on --verbose
run fill1-elastic STRATA_PRELOAD=1 STRATA_METAL_INDEXED_QSA=1 -- --elastic on --verbose
