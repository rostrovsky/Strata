#!/bin/zsh
# gate_head.sh: since patch 0009 (the top-k's tie rule) the old reference cannot be met exactly on the 7,900-token
# document by any build (section 15). So HEAD's engine (build-metal/borrow/strata-head, ff1c7a0d) is the reference:
# its plain default path is recorded, and this build's default, indexed, --elastic and --elastic --prefill 8192
# paths are held against it exactly, at the same capped cache (the counts are printed).
cd /Users/rafal/Projects/strata-air/Strata
O=bench/results/2026-10-12-m4-air-128k-resume/borrow/gate
E=${ENGINE:-build-metal/borrow/strata}
so() { vm_stat | awk '/Swapouts/{gsub("\\.","",$2); print $2}'; }
common=(--config bench/results/2026-10-12-m4-air-128k-resume/pending/desktop-128k.json --context 32768 --new 32 --only chat --only code --only doc8k --ref $O/gate-head-ref.json)
tail_args=(-- --kv f16 --prefill 4096 --reserve-gib 2 --expert-cache-cap ${GATE_CAP:-3000} --chunk 2)
s0=$(so)
STRATA_METAL_INDEXED_QSA=0 .venv/bin/python -u metal/tools/bench.py $common --engine build-metal/borrow/strata-head --gate record --log $O/gate-head-ref.log --json $O/gate-head-ref-run.json $tail_args > $O/gate-head-ref.out 2>&1
echo "record (HEAD's default path) rc=$? swapouts=$(( $(so) - s0 )): $(grep -o 'expert cache [0-9]* experts' $O/gate-head-ref.out)"
run() {   # name, environment..., then -- and extra engine options
  local name=$1; shift
  local envs=(); while [[ $1 != -- ]]; do envs+=($1); shift; done; shift
  env $envs .venv/bin/python -u metal/tools/bench.py $common --engine $E --gate check --exact --log $O/gate-head-$name.log --json $O/gate-head-$name.json $tail_args "$@" > $O/gate-head-$name.out 2>&1
  echo "$name rc=$? swapouts=$(( $(so) - s0 )): $(grep GATE $O/gate-head-$name.out | tail -1); $(grep -o 'expert cache [0-9]* experts' $O/gate-head-$name.out)"
  [ $(( $(so) - s0 )) -gt 0 ] && { echo "swap-outs: stopping"; exit 3; }
}
run default STRATA_METAL_INDEXED_QSA=0 --
run indexed STRATA_METAL_INDEXED_QSA=1 --
run elastic STRATA_METAL_INDEXED_QSA=1 -- --elastic on --verbose
run elastic8k STRATA_METAL_INDEXED_QSA=1 -- --prefill 8192 --elastic on --verbose
