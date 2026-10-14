#!/bin/zsh
# build-metal/resume-128k/gate.sh with the engine as ENGINE (default build-metal/bin/strata) and the files in OUT: the real-model
# exact gate at a small, capped cache: reference binary, then ENGINE's default, indexed and elastic paths
cd /Users/rafal/Projects/strata-air/Strata
O=${OUT:-bench/results/2026-10-12-m4-air-128k-resume/borrow/gate}; E=${ENGINE:-build-metal/bin/strata}; mkdir -p $O
so() { vm_stat | awk '/Swapouts/{gsub("\\.","",$2); print $2}'; }
common=(--config bench/results/2026-10-12-m4-air-128k-resume/pending/desktop-128k.json --context 32768 --new 32 --only chat --only code --only doc8k --ref $O/gate-ref.json)
tail_args=(-- --kv f16 --prefill 4096 --reserve-gib 2 --expert-cache-cap ${GATE_CAP:-3000} --chunk 2)
s0=$(so)
.venv/bin/python -u metal/tools/bench.py $common --engine build-metal/lowlevel-128k/strata-reference --gate record --log $O/gate-ref.log --json $O/gate-ref-run.json $tail_args > $O/gate-ref.out 2>&1
echo "record rc=$? swapouts=$(( $(so) - s0 ))"
# Every engine of the gate must have the same expert cache: one that is smaller than --expert-cache-cap (little free
# memory at that moment) cannot serve the 2-token passes of --chunk 2 in every layer, those layers are then computed
# another way, and its logits differ by up to 0.14 from a run with the full cache. (Seen on 2026-10-13: the
# reference got 840 experts instead of 1,529 and every path "failed" against it.)
# (GATE_CAP=N: another cap, for an evening when the reference does not get 3,000 MiB; the count must still be the
# cap's: 0.51 experts per MiB)
want="expert cache $(grep -o 'expert cache [0-9]* experts' $O/gate-ref.out | grep -o '[0-9]*') experts"
got_mib=$(grep -o 'experts ([0-9]* MiB)' $O/gate-ref.out | grep -o '[0-9]*')
[ $(( got_mib > ${GATE_CAP:-3000} - 20 && got_mib <= ${GATE_CAP:-3000} )) -eq 1 ] || { echo "the reference's cache is not the capped one: $(grep 'ready in' $O/gate-ref.out)"; exit 4; }
echo "every engine must report: $want"
[ $(( $(so) - s0 )) -gt 0 ] && { echo "swap-outs: stopping"; exit 3; }
STRATA_METAL_INDEXED_QSA=0 .venv/bin/python -u metal/tools/bench.py $common --engine $E --gate check --exact --log $O/gate-default.log --json $O/gate-default.json $tail_args > $O/gate-default.out 2>&1
echo "default rc=$? swapouts=$(( $(so) - s0 )) $(grep -c "$want" $O/gate-default.out)"
[ $(( $(so) - s0 )) -gt 0 ] && { echo "swap-outs: stopping"; exit 3; }
STRATA_METAL_INDEXED_QSA=1 .venv/bin/python -u metal/tools/bench.py $common --engine $E --gate check --exact --log $O/gate-indexed.log --json $O/gate-indexed.json $tail_args > $O/gate-indexed.out 2>&1
echo "indexed rc=$? swapouts=$(( $(so) - s0 )) $(grep -c "$want" $O/gate-indexed.out)"
[ $(( $(so) - s0 )) -gt 0 ] && { echo "swap-outs: stopping"; exit 3; }
STRATA_METAL_INDEXED_QSA=1 .venv/bin/python -u metal/tools/bench.py $common --engine $E --gate check --exact --log $O/gate-elastic.log --json $O/gate-elastic.json $tail_args --elastic on --verbose > $O/gate-elastic.out 2>&1
echo "elastic rc=$? swapouts=$(( $(so) - s0 )) $(grep -c "$want" $O/gate-elastic.out)"
[ $(( $(so) - s0 )) -gt 0 ] && { echo "swap-outs: stopping"; exit 3; }
# passes of 8,192 tokens read the 7,900-token document in one pass where the reference took two: not expected to be
# bit-identical, the gate says whether a token changed that was not a near-tie
tail8=(-- --kv f16 --prefill 8192 --reserve-gib 2 --expert-cache-cap ${GATE_CAP:-3000} --chunk 2 --elastic on --verbose)
STRATA_METAL_INDEXED_QSA=1 .venv/bin/python -u metal/tools/bench.py $common --engine $E --gate check --log $O/gate-elastic8k.log --json $O/gate-elastic8k.json $tail8 > $O/gate-elastic8k.out 2>&1
echo "elastic, 8192-token passes (near-ties allowed) rc=$? swapouts=$(( $(so) - s0 )) $(grep -c "$want" $O/gate-elastic8k.out)"
