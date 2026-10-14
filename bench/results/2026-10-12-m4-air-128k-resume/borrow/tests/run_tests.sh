#!/bin/zsh
# the tiny-model tests of refill/tests/run_tests.sh against ENGINE (default build-metal/bin/strata), each as it is and
# with STRATA_ELASTIC_REFILL=all, also with STRATA_KV_STEP=256 and the two back-off simulations; and the shape test
# without the hold (STRATA_KV_HOLD=0), with the old borrow rule (STRATA_ELASTIC_BORROW=available), without the wait for
# freed memory (STRATA_ELASTIC_SETTLE=0), without the Short shape's read-ahead (STRATA_ELASTIC_SHORT_AHEAD=0); and
# metal/tests/kv_hold_test.py.
# OUT: where the outputs go (default: this folder)
cd /Users/rafal/Projects/strata-air/Strata
T=${OUT:-bench/results/2026-10-12-m4-air-128k-resume/borrow/tests}
E=${ENGINE:-build-metal/bin/strata}
M=build-metal/tuning-128k/tiny/qwen4exp-moe.gguf
P=.venv/bin/python
mkdir -p $T
fail=0
run() {   # run NAME [VAR=value ...] -- command...
    local name=$1; shift
    local -a envs
    while [[ $1 != "--" ]]; do envs+=($1); shift; done
    shift
    env $envs "$@" > $T/$name.txt 2>&1
    local rc=$?
    echo "$name: rc=$rc, $(tail -1 $T/$name.txt)"
    [[ $rc -ne 0 ]] && fail=1
}
for mode in "" all; do
    s=${mode:+-refill-all}
    e=(${mode:+STRATA_ELASTIC_REFILL=$mode})
    run elastic-shapes$s $e -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
    run elastic-shapes-kv256$s $e STRATA_KV_STEP=256 -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
    run elastic-backoff-1$s $e STRATA_ELASTIC_TEST_BACKOFF=1 -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
    run elastic-backoff-tight$s $e STRATA_ELASTIC_TEST_BACKOFF=tight -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
    run elastic-backoff-1-kv256$s $e STRATA_ELASTIC_TEST_BACKOFF=1 STRATA_KV_STEP=256 -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
    run elastic-backoff-tight-kv256$s $e STRATA_ELASTIC_TEST_BACKOFF=tight STRATA_KV_STEP=256 -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
    run parity$s $e -- $P metal/tests/optimization_parity_test.py --reference build-metal/lowlevel-128k/strata-reference --engine $E --model $M
    run parity-kv256$s $e STRATA_KV_STEP=256 -- $P metal/tests/optimization_parity_test.py --reference build-metal/lowlevel-128k/strata-reference --engine $E --model $M
    run protocol$s $e -- $P metal/tests/serve_protocol_test.py --model $M --engine $E
    run protocol-kv256$s $e STRATA_KV_STEP=256 -- $P metal/tests/serve_protocol_test.py --model $M --engine $E
    run protocol-verify-cache$s $e STRATA_VERIFY_CACHE=1 -- $P metal/tests/serve_protocol_test.py --model $M --engine $E
done
run elastic-shapes-kv256-no-hold STRATA_KV_STEP=256 STRATA_KV_HOLD=0 -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
run elastic-shapes-borrow-available STRATA_ELASTIC_BORROW=available -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
run elastic-shapes-kv1024 STRATA_KV_STEP=1024 -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
run elastic-shapes-short-ahead-0 STRATA_ELASTIC_SHORT_AHEAD=0 -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
run elastic-shapes-settle-0 STRATA_ELASTIC_SETTLE=0 -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
run elastic-backoff-1-kv256-settle-0 STRATA_ELASTIC_TEST_BACKOFF=1 STRATA_KV_STEP=256 STRATA_ELASTIC_SETTLE=0 -- $P metal/tests/elastic_shapes_test.py --model $M --engine $E
run kv-hold -- $P metal/tests/kv_hold_test.py --model $M --engine $E
run kv-hold-refill-all STRATA_ELASTIC_REFILL=all -- $P metal/tests/kv_hold_test.py --model $M --engine $E
shasum -a 256 $E | cut -c1-64 > $T/binary-sha256.txt
echo "binary $(cut -c1-16 $T/binary-sha256.txt); failed: $fail"
exit $fail
