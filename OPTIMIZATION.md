# Strata optimization handoff — M4 MacBook Air, 32 GiB

Updated **2026-10-12**. The user explicitly stopped optimization and requested this handoff. No further tuning,
builds, benchmarks, or deployment were performed while preparing it. Resume optimization only when requested.
The performance goal is **not complete**.

## Read this first

The original goal was a usable **131,072-token total context** with the highest prefill and decode throughput this
machine can sustain while a browser, Ghostty, pi, and other normal desktop apps remain open. The later target was
**above 120 tok/s prefill and above 10 tok/s generation, without sacrificing quality**.

The strongest result is a completed **130,944-token input** at 131,072-token capacity, using the original IQ3_S
weights and **F16 KV**. It retrieved all five exact records spread across the input. Prefill was **64.92 tok/s**
(**33 minutes 36.9 seconds** to the first token); subsequent decode was **4.59 tok/s**. Repeating the request reused
130,943 tokens, returned identical output IDs, and reduced first-token latency to **0.403 seconds**.

This establishes functional use near the full context limit. It does **not** meet the speed target, establish broad
model quality, or prove that the machine has reached an absolute hardware limit. It also was **not swap-free**:
3,948 new 16 KiB swap-out pages, about **61.7 MiB**, appeared by the 81,920-token progress sample. No further
swap-outs were observed through completion. A separate memory sample reported macOS pressure level 2 (warning).

The main useful low-level change reduced the **planned context and compute allocation from 8.39 to 5.58 GiB** at
131,072 capacity with a 4,096-token prefill batch. It removes a redundant dense attention mask and unused top-k
scratch. That leaves more space for cached experts. A final controlled F16 throughput comparison remains undone.

**Do not confuse the following states:**

| Item | State at handoff |
| --- | --- |
| Production Strata configuration | Still **32,768** context; the 128k candidate was not installed |
| pi local-provider configuration | Still **32,768** context; real pi tool use against the candidate was not tested |
| Completed full-context run | Used saved executable `build-metal/lowlevel-128k/strata-indexed-topk` |
| Current repository source | Consolidated patches plus an **unbuilt, untested SIMD-sort experiment** |
| `build-metal/bin/strata` | Older compiled source, including rejected experiments disabled by default; **not equivalent to current source** |
| Generated `build-metal/llama.cpp` | Older patched tree; reconfiguration will replace its patch state |
| Running work | Full-context benchmark ended; its shell session is gone. No optimization process was left running |
| Git | HEAD was `29c52020` (`ngz`) and clean before these handoff/documentation additions |

Commit `29c52020` already contains the optimization source and the initial results report, including the untested
code. A commit is not evidence that its contents were validated. Earlier baseline commit `c6127518` (`u`) contains
the planner and staging fixes. The original planner comparison used the earlier `3497a384` engine. No commit,
push, or pull request was created for this handoff.

## Machine, model, and non-negotiable constraints

| Item | Observed setup |
| --- | --- |
| Machine | Apple M4 MacBook Air, 10 CPU cores (4 performance, 6 efficiency), 10 GPU cores |
| Memory / OS | 32 GiB unified memory; macOS 15.7.9 |
| Power | AC power; Low Power Mode off during measurements |
| Metal recommended working set | About 21.3 GiB; host memory was usually the binding cache limit |
| Measured storage / GPU bandwidth | SSD roughly 3–3.7 GB/s; cool GPU memory-read benchmark 88.7 GiB/s |
| Model | Existing Qwen3.8-Flash-Next GSQ-RCO IQ3_S, two native GGUF shards, unchanged |
| Model storage breakdown | 4.2 GiB dense tensors, 46.8 GiB routed experts, 26.8 GiB lazy n-gram table |
| Dense GPU allocation | 3.89 GiB |
| Experts | 48 layers × 512 experts; 10 selected per token; about 1.95 MiB per expert |
| Attention | Head size 256, 24 query heads, 2 KV heads; 12 QSA layers, every fourth layer; remaining layers use GDN |
| QSA | Top 2,048 pooled cells, pool size 4, plus live tail cells; indexed path must allow the larger resulting list |
| Model context metadata | 262,144; this work validated near 131,072, not 262,144 |

Keep the browser, Ghostty, pi, and the user's other apps available. Observed desktop workloads included Brave,
Citrix Workspace, a VM, Codex, and other apps. Do not close them to obtain a score and then call that the usable
desktop result. Their presence does not by itself prove interactive responsiveness; that still needs an actual
browser/terminal/pi smoke test under load.

Keep the original weights, F16 KV, expert count, and context semantics. Q8 KV changed outputs and was rejected as
the final quality-preserving configuration. Do not silently reduce context, change RoPE, use fewer experts, or
enable MTP to reach a number. The context limit includes input **and** output.

The Air is fanless. Short cool runs are materially faster than sustained work. Earlier measurements showed GPU
time per decoded token near 89 ms when cool and around 130–140 ms when warm, with slowdown after roughly 200 seconds.
Record both cool and sustained rates. Avoid claims of hardware impossibility based on these measurements alone.

## Paths and build model

The workspace is `/Users/rafal/Projects/strata-air`; the actual Git repository is its `Strata` subdirectory.
Run the commands in this document from **`/Users/rafal/Projects/strata-air/Strata`**.

| Purpose | Path |
| --- | --- |
| Strata repository | `/Users/rafal/Projects/strata-air/Strata` |
| Adjacent upstream checkout | `/Users/rafal/Projects/strata-air/llama.cpp` |
| First native model shard | `/Users/rafal/Projects/strata-air/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` |
| Tokenizer | `/Users/rafal/Projects/strata-air/Strata-data/packs/iq3_s/tokenizer` |
| Expert profile | `/Users/rafal/Projects/strata-air/Strata/data/expert-profile.bin` |
| Production config / launcher | `strata-iq3_s.json` / `run-iq3_s.sh` in the repository |
| Main engine / server | `metal/src/main.cpp` / `serve/server.py` |
| Benchmark / tests | `metal/tools/bench.py` / `metal/tests/` |
| Local experiment outputs | `build-metal/lowlevel-128k/` and `build-metal/tuning-128k/` |
| Preserved results | `bench/results/2026-10-12-m4-air-128k/` |
| Earlier planner/Q8 report | `bench/results/2026-10-11-m4-air-128k/` |

The Metal engine is a separate CMake project. llama.cpp is pinned to
**`23b0202a189c44a54625aadcb37a946dd1d6278d`**. CMake copies/fetches that revision into `build-metal/llama.cpp`,
checks it out, and applies `metal/patches/*.patch` in sorted order. Patch hashes trigger reconfiguration. The
adjacent upstream checkout was left untouched. Save lasting changes as patches; edits only to generated sources
will be lost. Do not rerun the old `make_indexed_patch.py` or `make_compact_patch.py` blindly: they represent older
patch layouts.

Future build commands, **not run for this handoff**:

```sh
cd /Users/rafal/Projects/strata-air/Strata
cmake -S metal -B build-metal -DCMAKE_BUILD_TYPE=Release -DSTRATA_METAL_BUILD_TESTS=ON
cmake --build build-metal -j4
ctest --test-dir build-metal --output-on-failure
```

Metal/GPU work and some process/system inspection needed sandbox escalation in this session. File reads and local
documentation edits did not. Do not run another real-model engine while the server is loaded: two copies exceed
the useful memory budget. Do not compile or run GPU microbenchmarks during a throughput measurement.

## What changed and what was actually validated

### Planner and prompt staging — present before the low-level work

The old memory planner measured a temporary context, freed it, sampled free host memory, then subtracted the
context again. Metal could retain those pages long enough to make that a double subtraction. The planner now
samples before and during the temporary allocation and computes host room as:

```text
min(before_context.usable() - context_bytes, during_context.usable()) - reserve_bytes
```

It then respects the Metal limit and any expert-cache cap. Existing swap counts as memory debt. Prompt read-ahead
staging, up to about 1.1 GiB, now ends at `Bulk.finish()` before answer decoding rather than at request destruction.

Historical Q8 KV ABBA comparison: median short-prompt decode increased from **3.58 to 6.36 tok/s**, about 77%, with
no new swap-outs. Cache sizes differed because this tested automatic planning; the first old run was so small that
two layers fell back to the CPU. This is useful evidence for the planner fix, **not an F16 speed result**. See the
2026-10-11 report and its `planner-ab.json`.

Earlier work already provides direct `F_NOCACHE` expert reads, eight read workers, one-layer look-ahead for ten
predicted experts, router bias learning, GPU keepalive, startup cache preload, and overlapping bulk prompt reads.
Preserve those when comparing new changes.

### Output reservation — `metal/src/main.cpp`

`cp.n_outputs_max = drafting() ? (uint32_t) o.spec + 1 : 1` reserves only the requested logits: one for ordinary
generation, and current token plus drafts for speculative verification. Tiny-model parity, including actual
speculative verification, passed on the earlier compiled version. This change alone produced almost no peak GPU
memory saving. Do not attribute the 2.81 GiB improvement to it.

### Indexed QSA — `metal/patches/0002-metal-indexed-qsa.patch`

Opt-in: **`STRATA_METAL_INDEXED_QSA=1`**. Default is off.

The old graph selected KV cells, scattered them into an F16 mask over all KV cells and queries, added the causal
mask, then scanned the mask back into ascending selected indices in Metal. The new path passes the selected I32
indices directly. An integer bitonic sort restores the exact ascending order, moves dead/out-of-range entries to
the end, and emits `-1` padding. The existing attention kernel uses the same cell order and floating-point math.

Guards restrict it to flash attention, a single causal sequence, actual F16 K and V, no ALiBi bias, head size 128 or
256, and at most 4,096 selected entries. Unsupported cases use the original path. The local `FLASH_ATTN_EXT`
extension accepts I32 row-contiguous masks; its CPU backend rejects that extension. Valid selected cells must be
unique. Unallocated dense input masks are no longer filled. Padding contributes the same masked values as before.

The original sort uses a 16 KiB threadgroup array and 78 bitonic compare stages at 4,096 entries. Before top-k was
fixed, indexed QSA reduced the planned 4k-batch allocation from **8.39 to 6.37 GiB**. At 8k batch it reduced
**12.09 to 9.07 GiB**. The final cleaned patch removes the failed compact-mask dependency; **that cleanup has not
been rebuilt or retested**.

Validation on the earlier compiled indexed implementation:

- **20/20 attention cases produced bit-identical output floats**. Coverage includes head sizes 128/256, query
  counts 1/7/33/65, KV lengths 256/5,000/131,072, selected counts 127/2,051/4,096, strided index rows, GQA, and invalid
  padding.
- Tiny-model generation matched IDs and reported logits, including a forced lookup/speculative verification case.
- Real-model chat, code, and a 7,900-token document produced the same **144 output IDs**. Initial strict chat-logit
  comparison failed because changing automatic cache size changed the short-prompt chunk from 2 to 3. Code and
  document probes matched already. At fixed 2,000 MiB cache cap and `--chunk 2`, all 48 chat IDs and reported probes
  matched too. Preserve the failed test as evidence; do not describe the original strict gate as passing.

Probe logits are printed to six decimal places and cover selected logits, not full-vocabulary float arrays. That
is weaker than the kernel test's bit-identical full output comparison. The five-record full-context retrieval is
a functional check, not a comprehensive quality evaluation.

### Top-k workspace — `metal/patches/0003-metal-topk-workspace.patch`

This allocation-only fix is always active. Allocation and dispatch now share `ggml_metal_op_top_k_use_radix`:

```text
ncols > 2048 && (k > 64 || (nrows > 4 && ncols >= 8192))
```

The radix path gets only its output allocation; only the bitonic path gets two full input-sized index workspaces.
No selection algorithm or floating-point arithmetic changes. This reduced the 4k-batch context/workspace plan
from **6.37 to 5.58 GiB**. **533/533 upstream Metal TOP_K cases passed**, the 20 attention cases stayed bit-identical,
and tiny-model parity passed. The consolidated patch layout still needs a fresh build and validation.

### New SIMD index sort — `metal/patches/0004-metal-qsa-simd-sort.patch`

**UNBUILT AND UNTESTED.** Opt-in: `STRATA_METAL_QSA_SIMD_SORT=1`; default is off. It adds
`kernel_flash_attn_ext_vec_idx_i32_simd` while retaining the old kernel. Cross-lane distances of 32 or more use the
old threadgroup compare/exchange and barriers. Smaller distances use `simd_shuffle_xor` on registers with one
barrier per outer stage. The intended result is the same integer order with roughly 78 → 40 barriers at 4,096
entries. This assumes the Apple 32-lane SIMD layout and needs explicit small/partial-group tests. No speedup or
correctness claim is established. Retain it only if tests and a controlled measurement justify it.

Tests were expanded after the last successful run, so current test source is ahead of validation:

- `indexed_qsa_test.cpp` adds selected counts 1/7/32/64 for both head sizes: **28 intended cases**, only the prior 20
  have been run. `--bench` adds six cases: heads 128/256 × queries 1/65/256, KV length 131,072, 2,051 selected cells,
  strided rows, five warmups and 100 timed graph executions.
- `optimization_parity_test.py` now checks default, indexed, and SIMD variants. Five prompts of 15/64/129/257/513
  tokens generate 12 outputs each. A repeating prompt plus `--lookup 1 --spec 3` forces 12 tokens and asserts that
  drafts were actually offered. These latest three-variant tests have not run.
- The existing tiny fixture is `build-metal/tuning-128k/tiny/qwen4exp-moe.gguf` (about 18 MiB). It has 128-element
  attention heads and exercises the indexed path at 1,024 context. Do not assume the tiny-model generator path in
  older docs exists; it was not found at that documented location.

`metal/CMakeLists.txt` adds optional test targets `strata-test-indexed-qsa` and `strata-test-backend-ops`, with CTest
entries for attention and TOP_K. Tests default off; the local build cache had them on.

## Completed F16 measurements

All rows allocate 131,072-token capacity. They use unchanged IQ3_S weights, F16 KV, a 4 GiB reserve, and normal apps
open. Rows differ in desktop memory, temperature, and automatic cache size. They are **not a controlled speed A/B**.
The allocation column is the startup plan, not proof that the runtime allocation cannot grow.

| Build / input tokens | Batch | Planned context/workspace | Expert cache | Prefill tok/s | TTFT | Decode tok/s | New swap-out pages |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Output-reservation fix, 4,096 | 4,096 | 8.39 GiB | 3,350 MiB | 101.9 | 40.20 s | 4.73 | 0 |
| Indexed QSA before top-k fix, 8,192 | 8,192 | 9.07 GiB | 2,589 MiB | 103.7 | 78.97 s | 4.14 | 5,870 |
| Indexed QSA + top-k, 16,384 | 4,096 | 5.58 GiB | 5,735 MiB | 77.35 | 211.84 s | 4.85 | 0 |
| Indexed QSA + top-k, **130,944** | **4,096** | **5.58 GiB** | **5,038 MiB** | **64.92** | **2,016.89 s** | **4.59** | **3,948** |

The 8k-batch run paged about 91.7 MiB and was not selected for the desktop candidate. An earlier unindexed 8k-batch
attempt planned 12.09 GiB and zero expert cache; it was stopped before completion. `outputs-limited.*` is **not** a
completed benchmark. A four-thread C++ build overlapped part of the 16k run, so its speed is confounded. The 16k
functional test still retrieved 5/5 records and repeated all 76 output IDs with 16,383 reused tokens and 0.379 s TTFT.

### Full-context run details

The 130,944-token run was allowed to finish before the handoff. Other builds and GPU benchmarks were paused during
it. It used `strata-indexed-topk`, not the current `build-metal/bin/strata`.

| Measurement | Value |
| --- | --- |
| Start memory | Apps 9.98 GiB, wired 2.39 GiB, free 11.70 GiB, file cache 7.13 GiB, existing swap 1.05 GiB |
| Cached experts | 2,575 slots, 5,038 MiB; about 10% of all experts |
| Full prompt | 130,944 tokens, 35 passes, 2,016.875 s engine prefill time |
| Answer | 76 tokens; 4.5916 tok/s |
| Retrieval positions | 6,560 / 32,736 / 65,457 / 98,177 / 124,353; all five exact values found |
| Repeat | 130,943 reused, 1 read; 0.4033 s TTFT; 4.8241 tok/s; identical output IDs; no additional swap-outs |
| Prefill timing | GPU wait 1,755.415 s; unoverlapped SSD 62.423 s; remainder about 199.04 s |
| GPU busy | 1,825.019 s, overlaps CPU/I/O; do not add it to wall time |
| SSD bytes | 1,396.8 GB for the first request including its answer; 1,275.5 GB read ahead, 1,229.7 GB used |
| Reported power | 18.45 W in the benchmark's measurement |
| New paging | 3,948 × 16 KiB = 61.6875 MiB; existing swap rounded from 1.05 to 1.11 GiB |
| Separate memory sample | At about 94k progress, `pressure_level=2`; temperature sample 66.1 °C SoC / 34.8 °C battery |

First 4,096 tokens took 39.64 s, 32,768 took 429.02 s, 65,536 took 925.94 s, and 81,920 took 1,199.64 s. Most later
4k passes took roughly 65–70 seconds. This is why a short 100+ tok/s prefill result must not be used as the full
context rate.

Decode at full depth averaged **217.79 ms/token**: GPU wait 126.45 ms, unoverlapped SSD 78.91 ms, CPU remainder
12.43 ms. GPU busy was 128.80 ms/token, about 100.21 cache misses and 458.09 MB SSD reads per token; router prediction
took 4.87 ms/token. These measurements suggest that router micro-optimization alone cannot close the gap to 10 tok/s.

**Unresolved memory warning:** the completed engine log ends with:

```text
~llama_context: MTL0 compute buffer size of 2201.2842 MiB, does not match expectation of 1759.2842 MiB
```

That is **442 MiB of additional compute allocation** relative to the expected buffer. Its cause and timing have
not been investigated. It may matter to the planner and paging, but that causal connection is unproven. The final
`memory_after` JSON sample was taken after engine exit and is not its live peak. The main JSON lacks numeric
pressure level because the running script predated that instrumentation; `full-128k-memory.jsonl` is a separate,
single preserved snapshot, not a continuous pressure trace.

## Experiments that did not earn a place in the candidate

| Experiment | Evidence / decision |
| --- | --- |
| Compact QSA mask, `STRATA_METAL_COMPACT_QSA=1` | In-place add and padded views saved only about 0.09 GiB in a controlled check, with no useful speed gain. An earlier roughly 1 GiB apparent saving was host-memory variation. Removed from current patches |
| Compact expert dispatch, `STRATA_METAL_COMPACT_MOE=1` | Prefix counts and a flat grid avoided empty 32-token expert tiles. 867/867 MMID cases passed; ABBA 4k prefill medians 99.3 → 100.6 tok/s, 48 output IDs unchanged. About 1.3% is inconclusive here. Removed |
| Relaxed matrix synchronization, `STRATA_METAL_RELAXED_MM=1` | Removed six SIMD barriers with `mem_none` in inner K loops, retained shared-memory barriers. Correctness passed for the exercised cases, but ABBA timings showed no gain. Removed |
| Chunked GDN, chunk 64 | Tested 2k workload slowed from 88.4 to 85.4 tok/s. Removed |
| Q8 KV | Lower memory use, but output changes. Historical measurements retained; not the quality-preserving candidate |
| Deeper expert prefetch | Two layers ahead lost to one layer ahead / ten predicted experts in tested workloads |
| Alternative eviction | S3-FIFO/LFU experiments did not improve the measured workloads |
| Keepalive off or fetch-only during decode | Earlier decode fell from 6.15 to roughly 3.56–3.70 tok/s; retain normal decode keepalive |
| Smaller keepalive grid / inserted sleeps | No reliable improvement. Earlier small-grid 6.22 vs 6.15; a 60 µs sleep changed sustained 5.33 → 5.44 with lower power, but was not sufficient evidence for a new default; 120 µs slowed it |
| Reading a whole predicted next-layer tensor | Slower than reading only predicted experts; earlier 256-token prompt 16.6 vs 22.8 tok/s |
| MTP speculative head | Adds about 2.04 GiB weights + 0.6 GiB context; open-app measurements often lost. A closed-app sustained run gained about 12%, but the quality gate failed, including a non-near-tie difference. Keep off |

Relaxed-sync median kernel times, off/on/off-order balanced by ABBA:

| Operation | Off µs | On µs | Speed change |
| --- | ---: | ---: | ---: |
| MUL_MAT F16 | 20,274.77 | 20,273.67 | +0.005% |
| MUL_MAT BF16 | 20,910.74 | 21,240.82 | −1.55% |
| MUL_MAT Q6_K | 22,399.84 | 22,421.46 | −0.10% |
| MUL_MAT_ID IQ4_XS | 5,851.80 | 5,923.51 | −1.21% |

Relaxed-sync correctness covered 867 MMID and 689 filtered dense cases plus tiny generation/speculative parity.
The earlier dense filter used lowercase `q6_k` and missed uppercase Q6_K; do not claim that filtered correctness
run independently checked dense Q6_K. The complete MMID suite did include Q6_K. For timing, the dense shape was
`m=4096,n=512,k=14336`, types F16/BF16/Q6_K; the MoE case was IQ4_XS, 128 matrices, 8 used,
`m=768,n=512,k=2048`. Upstream CSV output did not include the desired timings; use console `us/run` rows.

Archived patches are preserved under the results directory's `rejected-experiments/archived-patches/`. The old
numbers do **not** match current patch numbers: old 0004/0005 are earlier indexed-QSA/top-k versions, not rejected
algorithms. Old 0002/0003/0006 are compact mask, compact expert dispatch, and relaxed synchronization. Do not apply
the whole archive to the current tree.

Older baseline comparisons in `docs/APPLE_METAL.md` include useful history: desktop short decode 5.50 → 6.60,
sustained 5.29 → 5.73, and closed-app short decode 7.19 → 8.46 tok/s. Those used smaller contexts and different cache
sizes. They are not current full-128k results and must not be compared directly to 4.59 tok/s at full depth.

## Profiling clues, with limits

`profiling/prefill-profile-engine.txt` preserves a 2,048-token profile at 131,072 F16 capacity, 2,048 batch, and
3,067 MiB expert cache. `GGML_METAL_PROFILE=1` reported about 14.314 s total GPU operator time. GDN was about 12.77%,
sparse flash attention 6.83%, several MoE quantized matrix kernels each 4–8%, dense Q6_K kernels roughly one fifth,
and BF16 low-rank products about 6%. The indexer and TOP_K were only about 0.09% and 0.12% at that short depth.

Profiling serializes operations and perturbs the timing counters: the outer benchmark reported zero GPU-busy time
in that mode. Use the profiler's operator table for attribution, not that zero or its 83.35 tok/s as an isolated
performance result. Full-depth operator shares have not been measured.

Earlier bandwidth accounting estimated 3.9 GiB dense plus about 0.94 GiB selected expert reads per token. At the
measured cool 88.7 GiB/s, the arithmetic memory-read floor is around 54 ms before dispatch, other operations, and
SSD stalls. Roughly 2,000 small operations per token also matter. This is a diagnostic estimate, not proof of an
attainable end-to-end rate or a hard limit. The full run was mainly waiting for GPU work during prefill; eliminating
its 62 seconds of unoverlapped SSD time alone cannot turn a 2,017-second prefill into a 120 tok/s result.

## Next steps if the user resumes optimization

These are proposed experiments, not established improvements. Complete one controlled change at a time and keep
the default path unchanged unless measurements justify a new default.

1. **Reconcile source, binaries, and tests first.** Preserve the saved reference executables. Reconfigure the
   consolidated patches, build, and run both attention paths, TOP_K, tiny generation/speculative parity, and the
   server protocol suite. The prepared `pending/validate-final.py` performs that sequence and an off/on/on/off SIMD
   kernel comparison. It has not run; inspect it before use. It runs relative to the repository root and still
   relies on local ignored reference binaries and the tiny fixture. If SIMD sort fails or produces no repeatable
   gain, archive it and retain the validated integer-sort design.
2. **Account for actual full-depth memory before deployment.** Investigate the 442 MiB compute-buffer growth,
   growing host input/pooling masks, bulk staging, and checkpoint allocations. A reserve of **5 GiB** was an idea,
   not a tested fix. More reserve means fewer cached experts and may reduce decode speed. Staging currently checks
   free memory above a hardcoded 1.5 GiB rather than reserving its full peak separately; it can consume nominal
   desktop headroom. Host masks may approach roughly 0.5 GiB at 128k/4k batch. Check possible double-counting in
   free/file/purgeable estimates rather than assuming it. Auto cache is fixed at startup and cannot accommodate
   arbitrary later app growth.
3. **Try disabling keepalive only during bulk prefill.** The GPU is already busy during large prompt passes; dummy
   keepalive may waste power and thermal headroom on this fanless machine. Start with ABBA `STRATA_KEEPALIVE=off`
   versus `always` on 4k prefill with one generated token, then sustained prefill if promising. Retain keepalive
   during decode. A future `decode` mode could scope the busy guard around answer generation, near `Bulk.finish()`.
   Nothing implementing that mode has been written. The known decode regression from turning keepalive off is not
   evidence against a prefill-only policy.
4. **Run the missing fair final F16 comparison.** Compare the pre-low-level `strata-reference` with the rebuilt
   candidate at 131,072 capacity, F16, batch 4,096, the same chosen reserve, apps open, and `--chunk 2`. Run chat/code
   in ABBA order, then a sustained workload. For automatic-cache benefits, allow cache sizes to differ and record
   them; for kernel/math comparisons, cap both caches at the same fitting value. The existing rows establish
   allocation savings, not a final isolated throughput gain.
5. **Complete desktop integration only after validation.** Merge the tested settings into production and update
   pi's local provider. Check API context/aliases, then actual pi tool use, browser and Ghostty responsiveness, and
   reuse across turns. The staged configuration has not earned a claim of swap-free desktop use.

Further ideas, in approximate order of likely value:

| Idea | Why it might help / what remains unknown |
| --- | --- |
| Reclaim prefill workspace for decode cache | Several GiB of workspace reduce expert cache even when decoding needs less. ggml graph allocation grows rather than shrinking; recreating a scheduler and resizing expert buffers would require preserving KV, GDN, QSA pools, and cache validity. High potential, substantial engineering, no implementation |
| Save session state across idle unload | A repeated 128k request can start in 0.4 s while resident, but cold refill costs 33.6 minutes. The 900 s idle unload loses that benefit. The Metal engine has in-memory recurrent checkpoints, not a full persistent KV/GDN/QSA session format. Disk state would be large and needs correct model/template identity and state restoration. No implementation |
| SIMD integer QSA sort | Current unbuilt patch reduces barriers without intended FP changes. Kernel wins may have little whole-model impact; measure before keeping |
| Full-depth kernel/dispatch profiling | Locate the sustained GPU bottleneck after removing workspace waste. Quantized matrix-vector bandwidth and dispatch overhead are more promising than optimizing a 4–5 ms CPU component alone |
| Router BF16 dot/NEON work | `metal/src/router_predict.hpp` already uses NEON F32 FMA over BF16 weights. Check hardware BF16 instructions and generated assembly first. Rounding predictor inputs changes predictions, not actual model routing, but might hurt prefetch hit rate. Use offline router traces and end-to-end measurements; untested |
| Prefetch confidence margin | Read fewer false positives to save SSD work and heat, at the risk of more blocking misses. Depth/count already have negative results; confidence margin still needs a specific test. Predictions must never replace the real router |
| Pressure-aware cache | Could adapt to apps growing after startup. Safe live resizing and buffer references make this more complex than increasing a reserve. No implementation |
| Better expert-tile grid locality | The rejected flat grid may have disturbed useful expert/M/N tile ordering. Revisit only with a concrete locality hypothesis; previous 1.3% change was inconclusive |
| Use resident cached experts during bulk streaming | Might avoid SSD bytes and heat, but requires copy/synchronization work. Unoverlapped SSD was a small share of full prefill. No demonstrated benefit |
| Metal Tensor API on M4 | `GGML_METAL_TENSOR_ENABLE=1` was not tested; upstream default on this M4 was disabled. Different floating-point order requires a quality gate. Do not assume a gain |
| Lookup speculation on representative code tasks | Tiny forced lookup verification passed, but end-to-end pi workloads and quality at full context were not measured. Batched verification can change rounding just as MTP did; keep it off until measured |

Do not spend another long run repeating relaxed matrix barriers, compact masks, chunk-64 GDN, deeper prefetch, or
cache-policy sweeps without a new hypothesis. Raising Metal's recommended limit or using `--overcommit-gib` was not
validated; host memory was the limit in the observed plans. System-wide overcommit, QoS changes, and closing user
apps are not demonstrated solutions to this task.

## Reproduction and measurement discipline

The command that completed the full-context run was equivalent to the following. This is a reproduction record,
not an instruction to restart it when reading the handoff:

```sh
cd /Users/rafal/Projects/strata-air/Strata
STRATA_METAL_INDEXED_QSA=1 \
STRATA_METAL_COMPACT_QSA=0 STRATA_METAL_COMPACT_MOE=0 STRATA_METAL_RELAXED_MM=0 \
.venv/bin/python -u metal/tools/bench.py \
  --config strata-iq3_s.json \
  --engine build-metal/lowlevel-128k/strata-indexed-topk \
  --context 131072 --prefill 130944 --prefill-new 120 --needle --prefill-repeat \
  --json build-metal/lowlevel-128k/full-128k.json \
  --log build-metal/lowlevel-128k/full-128k.log \
  -- --kv f16 --prefill 4096 --reserve-gib 4
```

The first `--prefill` is benchmark input length; the one after `--` is the engine batch size. The input plus maximum
answer plus boundary margin fits 131,072. The synthetic document uses `docs/DETAILS.md` text with five inserted
records and a tail that closes thinking before retrieval. It is not a 128k agent transcript. Prefix reuse is a
separate latency result and must not be presented as cold-prefill throughput.

For a future throughput ABBA, use `--ab 'ENV=0 EXE ...' 'ENV=1 EXE ...' --runs 2 --pause 8`; two runs per candidate
produce A/B/B/A order. The benchmark can run selected chat/code requests with `--decode --only chat --only code
--new 80`. Eight seconds is a fixed rest interval, not proof of thermal equilibrium. Record temperature and use
sustained runs as well.

For strict real-model parity, record a fresh reference with the same context, batch, fixed cache cap, and
`--chunk 2`, then check with `--exact`. The initial reference in `correctness/indexed-gate-ref.json` came from a
different effective short-prompt chunk; it is not a fair strict chat reference for a new chunk-2 candidate. The
separate controlled chat result resolves that earlier mismatch. Test chat, code, and `doc8k`, then a broader suite
if changing arithmetic. Default near-tie tolerance is 0.05; `--exact` requires all IDs and reported probes to match.

Useful benchmark changes now in source:

- `--gate record --only` respects the requested subset; `--exact` and per-probe comparison expose stricter parity.
- Engine `env` from the config is now inherited by benchmarks as it is by the server; inline A/B assignments win.
- Progress, reuse, per-token GPU/SSD/CPU counters, new swap-outs, and retrieval outcomes are saved in JSON.
- `memory_state()` now reads `vm.swapusage` and `kern.memorystatus_vm_pressure_level`. This last addition was not
  in the already-running full-context script.
- Use `--new`, `--json`, and engine option `--expert-cache-cap`; guessed flags such as `--tokens`, `--out`, and
  `--cap-mib` do not exist.

Record executable identity, environment, exact input depth, total capacity, F16/Q8, batch/chunk, cache slots/MiB,
app memory, existing swap, **new** swap-outs, temperature, power mode, and whether any other GPU/build work ran.
GB in SSD counters is decimal; GiB/MiB in allocations is binary. Page counts here are 16 KiB pages. Do not sum
overlapping GPU-busy and I/O counters as though they were disjoint wall time.

## Production and pi integration still to do

Production `strata-iq3_s.json` points at `build-metal/bin/strata`, uses the native model and expert profile, automatic
expert cache, and `--max-context 32768`. Its API model name is `qwen3.8-flash-next-iq3_s`, backend `metal`, port 8080,
and idle unload 900 seconds. `run-iq3_s.sh` starts `.venv/bin/python serve/server.py --engine strata --config
strata-iq3_s.json --port 8080 --open` from the repository.

The **uninstalled candidate** is preserved at `bench/results/2026-10-12-m4-air-128k/pending/desktop-128k.json` and
also remains in the local build directory. It adds:

```text
--max-context 131072 --kv f16 --prefill 4096 --reserve-gib 4
env: STRATA_METAL_INDEXED_QSA=1
aliases: qwen3.8-27b-iq3_s
```

It leaves idle unload unchanged and does not enable SIMD sort, MTP, or lookup. It points to the ordinary engine
binary, which currently differs from source. Re-read and merge current production settings before later applying
it; do not overwrite intervening user changes. The measured paging and compute-buffer warning remain unresolved.

pi was version **1.0.2**, executable `/Users/rafal/.bun/bin/pi`, linked into the global
`@earendil-works/pi-coding-agent` package. In `/Users/rafal/.pi/agent/models.json`, provider `qwen-local` uses
`http://127.0.0.1:8080/v1`, API `openai-completions`, model ID **`qwen3.8-27b-iq3_s`**, reasoning enabled, text input,
`contextWindow: 32768`, and `maxTokens: 8192`. The old ID/display name misleadingly says 27B; preserve compatibility
through the server alias if the display name is corrected. The global pi default was an OpenRouter DeepSeek model,
not the local provider. Do not change global defaults or print credentials while updating one local entry.

After future validation and installation, set the local provider's context window to 131,072 and check `/health`
and `/v1/models` plus actual tool use. A prepared fixture at `build-metal/lowlevel-128k/pi-smoke.txt` contains
`STRATA_PI_READ_CONFIRMED_7c4a91e2`. A suggested, **not yet run**, pi smoke test is:

```sh
/Users/rafal/.bun/bin/pi \
  --provider qwen-local --model qwen3.8-27b-iq3_s \
  --offline --no-session --thinking off \
  --no-extensions --no-skills --no-prompt-templates --no-context-files \
  --tools read --mode json --print \
  'Use read to open /Users/rafal/Projects/strata-air/Strata/build-metal/lowlevel-128k/pi-smoke.txt, then return its exact content.'
```

Require an actual `read` tool call and the final marker; a plain chat completion is not sufficient. pi tried to
create a settings lock even for help in the restricted environment, so invoking it may require an approved local
execution context. `--offline` avoids unrelated startup network work. No dedicated `run-pi-iq3_s.sh` was created;
such a launcher could select this provider without changing global settings.

Do not blindly raise server timeouts. `STRATA_ENGINE_STALL_S` defaults to 90 seconds and checks CPU/I/O activity,
not just elapsed prompt time. `engine_silence_s` defaults to 300 with prefill adaptation to three times the last
batch; full-run batches were mostly 65–70 seconds. The stall setting is read by the server process, whereas config
`env` applies to the engine child. Keep that distinction if diagnosing a future long-prefill timeout. The 900-second
idle unload reclaims desktop RAM but loses the expensive resident prefix; changing it has a real memory/latency
tradeoff and was not part of the completed work.

## Evidence and local recovery map

The results directory now preserves the finished full-context JSON, engine log, console output, and separate memory
sample, along with earlier measurements, correctness failures and passes, profiling, rejected patches, and pending
scripts. `artifact-manifest.json` records SHA-256 identities for local executables, the tiny fixture, and expert
profile. Large binaries and model weights were not copied into the report.

| Preserved evidence under `bench/results/2026-10-12-m4-air-128k/` | Use |
| --- | --- |
| `full-128k.json`, `full-128k-engine.txt`, `full-128k-console.txt`, `full-128k-memory.jsonl` | Completed full-depth result, paging, reuse, and allocation warning |
| `outputs-4096.json`, `indexed-8192.json`, `topk-16k.json`, corresponding `*-engine.txt` | Planned allocation changes and shorter-input results |
| `indexed-kernel-test.txt`, `ctest-topk.txt` | Earlier 20-case attention and 533-case TOP_K passes |
| `correctness/`, `indexed-chat-equal-cache.json` | Real-model initial strict failure, reference data, controlled chat fix, tiny parity/protocol output |
| `profiling/` | Perturbed 2k prefill operator profile and its metadata |
| `relaxed-perf-summary.json`, `compact-moe-ab.json`, `chunked-ab.json` | Rejected performance hypotheses |
| `rejected-experiments/` | Raw matrix timing output, compact-mask comparison, incomplete 8k attempt, historical source patches |
| `pending/` | Unexecuted validation script, uninstalled config, pi fixture |
| `artifact-manifest.json` | Local artifact identity at the stop |

Local executables to preserve before cleaning `build-metal/`:

| Path relative to repository | Meaning |
| --- | --- |
| `build-metal/tuning-128k/strata-before` | Original pre-planner-fix reference |
| `build-metal/lowlevel-128k/strata-reference` | `c6127518`-era reference before low-level changes |
| `build-metal/lowlevel-128k/strata-full-outputs` | Before the output-reservation fix |
| `build-metal/lowlevel-128k/strata-before-indexed` | Output-reservation fix; old compact experiments available but off; no indexed QSA/top-k fix |
| `build-metal/lowlevel-128k/strata-indexed-topk` | **Executable used for the successful full-context test**; before relaxed-sync and source cleanup |
| `build-metal/bin/strata` | Last compiled experimental tree, includes relaxed-sync; not current consolidated source |

`docs/APPLE_METAL.md` remains useful for architecture and older experiments, but some status statements are stale:
it says the largest exercised context was 40,960, describes the now-fixed planner problem as open, and gives older
batch settings and closed-app advice. The 2026-10-11 report also has historical pending full-context language.
Use this handoff and the finished F16 result for the latest status. The 2026-10-12 report was updated with completion
and a pointer here; its prior correctness numbers still refer to the earlier compiled implementation.

The next agent should be able to start with source/binary reconciliation and the memory warning, without another
blind 34-minute prefill or a repeat of the rejected kernel experiments.
