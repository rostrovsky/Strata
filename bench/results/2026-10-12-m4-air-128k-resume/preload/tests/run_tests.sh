#!/bin/zsh
# the tiny-model tests of section 16: those of section 12 (refill/tests/run_tests.sh: the shape test, parity and the
# protocol suite, as they are and with STRATA_ELASTIC_REFILL=all, each also with STRATA_KV_STEP=256 and the back-off
# simulations), and preload_test.py (the start with and without the profile fill), on build-metal/bin/strata
cd /Users/rafal/Projects/strata-air/Strata
T=bench/results/2026-10-12-m4-air-128k-resume/preload/tests
M=build-metal/tuning-128k/tiny/qwen4exp-moe.gguf
P=.venv/bin/python
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
    run elastic-shapes$s $e -- $P metal/tests/elastic_shapes_test.py --model $M
    run elastic-shapes-kv256$s $e STRATA_KV_STEP=256 -- $P metal/tests/elastic_shapes_test.py --model $M
    run elastic-backoff-1$s $e STRATA_ELASTIC_TEST_BACKOFF=1 -- $P metal/tests/elastic_shapes_test.py --model $M
    run elastic-backoff-tight$s $e STRATA_ELASTIC_TEST_BACKOFF=tight -- $P metal/tests/elastic_shapes_test.py --model $M
    run elastic-backoff-1-kv256$s $e STRATA_ELASTIC_TEST_BACKOFF=1 STRATA_KV_STEP=256 -- $P metal/tests/elastic_shapes_test.py --model $M
    run elastic-backoff-tight-kv256$s $e STRATA_ELASTIC_TEST_BACKOFF=tight STRATA_KV_STEP=256 -- $P metal/tests/elastic_shapes_test.py --model $M
    run parity$s $e -- $P metal/tests/optimization_parity_test.py --reference build-metal/lowlevel-128k/strata-reference --engine build-metal/bin/strata --model $M
    run parity-kv256$s $e STRATA_KV_STEP=256 -- $P metal/tests/optimization_parity_test.py --reference build-metal/lowlevel-128k/strata-reference --engine build-metal/bin/strata --model $M
    run protocol$s $e -- $P metal/tests/serve_protocol_test.py --model $M
    run protocol-kv256$s $e STRATA_KV_STEP=256 -- $P metal/tests/serve_protocol_test.py --model $M
    run protocol-verify-cache$s $e STRATA_VERIFY_CACHE=1 -- $P metal/tests/serve_protocol_test.py --model $M
done
run preload -- $P metal/tests/preload_test.py --model $M
shasum -a 256 build-metal/bin/strata | cut -c1-64 > $T/binary-sha256.txt
echo "binary $(cut -c1-16 $T/binary-sha256.txt); failed: $fail"
exit $fail
