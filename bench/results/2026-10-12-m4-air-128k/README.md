# M4 Air: 128k context with F16 KV and desktop apps open

Machine: MacBook Air M4, 10 CPU cores, 10 GPU cores, 32 GiB unified memory, AC power, Low Power Mode off.
Model: the existing Qwen3.8-Flash-Next GSQ-RCO IQ3_S GGUF shards. The weights are unchanged. This work follows
Strata `c6127518`, with llama.cpp pinned to `23b0202a189c44a54625aadcb37a946dd1d6278d`.
Measurements are from 2026-10-12. Brave, Ghostty, the user's VM, and other desktop apps stay open.

This F16 profile supersedes the Q8 KV candidate in the [previous report](../2026-10-11-m4-air-128k/README.md).
The requested 120 tok/s prefill and 10 tok/s decode have not been reached. The full-context run completed with
130,944 input tokens, all five records retrieved, 64.9 tok/s prefill, and 4.59 tok/s decode. It incurred 61.7 MiB of
new swap-outs. Optimization stopped at the user's request. See the [full handoff](../../../OPTIMIZATION.md) for
source/binary differences, unbuilt experiments, desktop integration status, and proposed next steps.

## Changes retained

1. **Pass selected attention cells directly to Metal.** QSA already selects its KV cells. The old graph scatters
   those cells into a large F16 mask, adds the causal mask, and scans the mask back into a list for sparse attention.
   `STRATA_METAL_INDEXED_QSA=1` passes the selected indices directly. An integer sorting kernel restores the same
   ascending order. The attention kernel then performs the same floating-point operations over the same cells.
   This path is restricted to one causal sequence, F16 KV, supported head sizes, and at most 4,096 selected cells.
   Other configurations retain the existing path.
2. **Allocate top-k scratch only when needed.** Metal reserved two full-size index buffers for every top-k operation.
   Its radix implementation does not use those buffers. The allocator and dispatcher now share the same predicate,
   so radix receives only its output allocation. The selection algorithm is unchanged.
3. **Reserve only the logits the engine requests.** Normal generation needs the final token's logits. Speculative
   verification reserves one output for the current token and each draft. This change alone did not materially
   reduce the measured peak GPU allocation.

At 131,072-token capacity, F16 KV, and a 4,096-token prefill batch, the context and workspace allocation falls from
**8.39 to 5.58 GiB**, a **2.81 GiB reduction (33.5%)**. Indexed attention accounts for 2.02 GiB of that reduction;
the top-k fix accounts for the remaining 0.79 GiB. The memory planner can use the released space for experts while
keeping the desktop reserve. Cache size still depends on the apps and existing swap at startup.

## Measured candidate, not installed

```text
STRATA_METAL_INDEXED_QSA=1
--max-context 131072 --kv f16 --prefill 4096 --reserve-gib 4 --expert-cache auto
```

The context limit includes input and output. The 4 GiB reserve is intended to leave room for the OS and growing
desktop apps, but the full run still paged and reported warning-level memory pressure in a separate sample. No
system memory limit was raised. The cache is sized at startup. Production Strata and pi remain configured for
32,768 context; the 128k candidate was not installed. The existing 900-second idle unload also loses resident
prefix reuse, which matters when a cold full prefill takes over 33 minutes.

## Completed measurements

Every row below allocates a 131,072-token context. A shorter input does not establish speed at full context.
These are individual runs with different app-memory and thermal states, not a controlled speed comparison.

| Build and input | Maximum prefill batch | Context/workspace | Expert cache | Prefill | Decode | New swap-outs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before indexed attention, 4,096 tokens | 4,096 | 8.39 GiB | 3,350 MiB | 101.9 tok/s | 4.73 tok/s | 0 |
| Indexed attention, before top-k fix, 8,192 tokens | 8,192 | 9.07 GiB | 2,589 MiB | 103.7 tok/s | 4.14 tok/s | 5,870 pages |
| Indexed attention and top-k fix, 16,384 tokens | 4,096 | 5.58 GiB | 5,735 MiB | 77.3 tok/s | 4.85 tok/s | 0 |
| Indexed attention and top-k fix, 130,944 tokens | 4,096 | 5.58 GiB | 5,038 MiB | 64.9 tok/s | 4.59 tok/s | 3,948 pages |

The 8k-batch run caused about 92 MiB of new swap-outs and is not the selected desktop profile. A build ran during
part of the 16k test, so its throughput is not an isolated performance measurement. Its functional checks passed:
all five exact records were retrieved, and a repeated request reused 16,383 input tokens, returned the same 76
output IDs, and produced its first token in 0.38 seconds.

The full run took 2,016.89 seconds to its first token, retrieved all five records, and generated 76 tokens.
A repeated request reused 130,943 tokens, returned identical IDs, and reached its first token in 0.403 seconds;
repeat decode was 4.82 tok/s. New swap-outs reached 3,948 pages by the 81,920-token sample and stayed unchanged
through completion. These are 16 KiB pages, or 61.7 MiB. The engine also logged a compute buffer of 2,201.2842 MiB
against an expected 1,759.2842 MiB: 442 MiB of growth that remains unexplained. The table gives startup allocation,
not a guaranteed runtime peak.

Raw results: [4k baseline](outputs-4096.json), [8k batch](indexed-8192.json), [16k retrieval](topk-16k.json), and
[full-context retrieval and repeat](full-128k.json). The corresponding `*-engine.txt` files record the memory plans.
[Full-run console output](full-128k-console.txt) and a [separate pressure snapshot](full-128k-memory.jsonl) are also
preserved. The full-run JSON was produced before numeric pressure instrumentation was added.

## Correctness

- Twenty attention-kernel cases produce **bit-identical output floats** against the dense-mask implementation.
  Cases cover 128- and 256-element heads, decode and prefill, grouped-query attention, strided index rows, invalid
  padding, 4,096 selected cells, and a 131,072-cell KV cache. See [kernel output](indexed-kernel-test.txt).
- All 533 upstream Metal top-k tests pass. See [CTest output](ctest-topk.txt).
- Tiny-model generation and speculative verification produce the same IDs and reported logits as the reference.
- The initial real-model check produced the same 144 output IDs across chat, code, and a 7,900-token document.
  Chat logits differed when the larger automatic expert cache changed the short-prompt batch size. Fixing the same
  cache cap and batch size made all 48 chat IDs and reported logits identical; code and document probes had already
  matched. See [controlled chat check](indexed-chat-equal-cache.json).

These checks test implementation equivalence and retrieval, not a broad assessment of model quality. F16 KV,
the original weight quantization, expert count, and context length remain intact. MTP and speculative lookup are
not enabled in this profile.

These passes apply to the earlier compiled implementation. The source patches were later consolidated, and an
opt-in SIMD index-sort experiment plus additional tests were added. That final source state has not been built
or validated. The successful full-context run used `build-metal/lowlevel-128k/strata-indexed-topk`; neither the
current source nor `build-metal/bin/strata` should be assumed equivalent to that saved executable.

## Experiments rejected

- Removing six SIMD-group synchronization barriers from the matrix kernels passed correctness tests but did not
  improve throughput. An off/on/on/off kernel comparison gave approximately 0% for F16, -1.6% for BF16, -0.1% for
  Q6_K, and -1.2% for IQ4_XS expert matrices. See [timings](relaxed-perf-summary.json).
- Dispatching only occupied expert tiles changed median 4k prefill from 99.3 to 100.6 tok/s, too small a gain to
  establish an improvement in these runs. See [comparison](compact-moe-ab.json).
- Chunked DeltaNet was slower in the tested 2k workload: 85.4 versus 88.4 tok/s. See [comparison](chunked-ab.json).
- Reusing the dense selection-mask allocation in place saved little memory and showed no useful speed gain.

The rejected kernel changes are absent from the current source patches. Historical patches and timing output
are preserved in `rejected-experiments/`; older indexed-QSA and top-k patch versions there are references, not
rejected algorithms.

## Reproduce

Do not run a benchmark while the real-model server is loaded: two copies exceed this machine's memory budget.
Leave the usual desktop apps open. These commands are for a future requested continuation, not an instruction
to resume the stopped optimization. A rebuilt current tree needs the validation described in the handoff before
its results can be compared to the saved executable.

```sh
cmake -S metal -B build-metal -DCMAKE_BUILD_TYPE=Release -DSTRATA_METAL_BUILD_TESTS=ON
cmake --build build-metal -j4
ctest --test-dir build-metal --output-on-failure

STRATA_METAL_INDEXED_QSA=1 STRATA_METAL_QSA_SIMD_SORT=0 .venv/bin/python metal/tools/bench.py \
  --config strata-iq3_s.json --engine build-metal/lowlevel-128k/strata-indexed-topk --context 131072 \
  --prefill 130944 --prefill-new 120 --needle --prefill-repeat \
  --json build-metal/full-128k.json --log build-metal/full-128k.log \
  -- --kv f16 --prefill 4096 --reserve-gib 4
```

The full-context test places five unique records near 5%, 25%, 50%, 75%, and 95% of the input. It checks all five
values, then repeats the request to check prefix reuse and output identity. The benchmark records progress,
memory pressure, new swap-outs, and the GPU/SSD/CPU contributions to decode time.
