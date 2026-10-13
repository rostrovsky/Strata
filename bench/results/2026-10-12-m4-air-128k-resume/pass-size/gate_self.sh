#!/bin/zsh
# gate_self.sh: after the top-k tie rule changed, the old reference cannot be the measure for contexts past 2,048
# tokens.  The current build's plain default path is recorded, and its other paths are held against it exactly:
# indexed attention, --elastic, --elastic with passes of 8,192, and each top-k kernel taken for every row.
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/resume-128k
so() { vm_stat | awk '/Swapouts/{gsub("\\.","",$2); print $2}'; }
common=(--config bench/results/2026-10-12-m4-air-128k-resume/pending/desktop-128k.json --context 32768 --new 32 --only chat --only code --only doc8k --ref $O/gate-self-ref.json)
tail_args=(-- --kv f16 --prefill 4096 --reserve-gib 2 --expert-cache-cap 3000 --chunk 2)
s0=$(so)
run() {   # name, environment..., then -- and extra engine options
  local name=$1; shift
  local envs=(); while [[ $1 != -- ]]; do envs+=($1); shift; done; shift
  env $envs .venv/bin/python -u metal/tools/bench.py $common --gate check --exact --log $O/gate-self-$name.log --json $O/gate-self-$name.json $tail_args "$@" > $O/gate-self-$name.out 2>&1
  echo "$name rc=$? swapouts=$(( $(so) - s0 )): $(grep GATE $O/gate-self-$name.out | tail -1); $(grep -o 'expert cache [0-9]* experts' $O/gate-self-$name.out)"
  [ $(( $(so) - s0 )) -gt 0 ] && { echo "swap-outs: stopping"; exit 3; }
}
STRATA_METAL_INDEXED_QSA=0 .venv/bin/python -u metal/tools/bench.py $common --gate record --log $O/gate-self-ref.log --json $O/gate-self-ref-run.json $tail_args > $O/gate-self-ref.out 2>&1
echo "record (default path of this build) rc=$? swapouts=$(( $(so) - s0 ))"
run indexed STRATA_METAL_INDEXED_QSA=1 --
run elastic STRATA_METAL_INDEXED_QSA=1 -- --elastic on
run elastic8k STRATA_METAL_INDEXED_QSA=1 -- --prefill 8192 --elastic on
run radix-only STRATA_METAL_INDEXED_QSA=1 GGML_METAL_TOP_K=radix --
run bitonic-only STRATA_METAL_INDEXED_QSA=1 GGML_METAL_TOP_K=bitonic --
