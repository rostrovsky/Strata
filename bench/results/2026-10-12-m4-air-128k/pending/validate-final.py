import os, subprocess, time
from pathlib import Path
root = Path('build-metal/lowlevel-128k')
def run(name, args, env=None):
    print(name, flush=True)
    with (root / (name + '.out')).open('w') as log:
        result = subprocess.run(args, env={**os.environ, **(env or {})}, stdout=log, stderr=subprocess.STDOUT)
    print('exit', result.returncode, flush=True)
    print((root / (name + '.out')).read_text()[-1800:], flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
run('final-configure', ['cmake', '-S', 'metal', '-B', 'build-metal', '-DCMAKE_BUILD_TYPE=Release', '-DSTRATA_METAL_BUILD_TESTS=ON'])
run('final-build', ['cmake', '--build', 'build-metal', '-j4'])
run('final-ctest', ['ctest', '--test-dir', 'build-metal', '--output-on-failure'], {'STRATA_METAL_QSA_SIMD_SORT': '0'})
run('simd-qsa-test', ['build-metal/strata-test-indexed-qsa'], {'STRATA_METAL_QSA_SIMD_SORT': '1'})
run('final-parity', ['.venv/bin/python', 'metal/tests/optimization_parity_test.py', '--reference', 'build-metal/lowlevel-128k/strata-before-indexed', '--engine', 'build-metal/bin/strata', '--model', 'build-metal/tuning-128k/tiny/qwen4exp-moe.gguf'])
run('final-protocol', ['.venv/bin/python', 'metal/tests/serve_protocol_test.py', '--engine', 'build-metal/bin/strata', '--model', 'build-metal/tuning-128k/tiny/qwen4exp-moe.gguf'], {'STRATA_METAL_INDEXED_QSA': '1', 'STRATA_METAL_QSA_SIMD_SORT': '1'})
for i, enabled in enumerate([0,1,1,0]):
    run(f'simd-qsa-perf-{i}-{enabled}', ['build-metal/strata-test-indexed-qsa', '--bench'], {'STRATA_METAL_QSA_SIMD_SORT': str(enabled)})
