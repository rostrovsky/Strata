# Strata optimization handoff — M4 MacBook Air, 32 GiB

Updated **2026-10-12, night** (the newest part is the first block of "Read this first"). The work was stopped in the morning at the user's request, resumed in the
afternoon ("continue the OPTIMIZATION.md work"), and stopped again where every next step either takes the Mac for
half an hour or risks swap, which is the user's call. In the evening the user asked for the first of those steps:
a full-context prompt with the corrected plan ran **without a swap-out**, and the 128k configuration was installed.
The performance goal is **not complete**.

Reports: [morning](bench/results/2026-10-12-m4-air-128k/README.md),
[afternoon](bench/results/2026-10-12-m4-air-128k-resume/README.md), [evening](bench/results/2026-10-12-m4-air-128k-resume/README.md#8-evening-the-full-context-prompt-with-the-new-plan-and-the-install) (section 8 of the
afternoon's). Sections below that describe the morning are marked where the afternoon or the evening changed them.

## Read this first

**Night, 2026-10-12 — the newest state; read this before the rest.** The owner asked to "install the faster
config and keep optimizing". Section 10 of the
[report](bench/results/2026-10-12-m4-air-128k-resume/README.md#10-night-the-model-in-the-file-cache-n-gram-rows-and-a-top-k-that-was-not-reproducible)
has the numbers.

- **Installed:** `--max-context 131072 --kv f16 --prefill 8192 --reserve-gib 3 --elastic on`, indexed QSA
  (`strata-iq3_s.json`; the plain one is `bench/results/2026-10-12-m4-air-128k-resume/installed/strata-iq3_s.before-elastic.json`).
- **8 GiB of the Mac's memory was this model, in the OS file cache**: the dense weights a second time (the loader
  reads them through a mapping) and every n-gram row ever looked up. macOS kept those pages and compressed other
  programs instead. The engine gives them back now (`msync(MS_INVALIDATE)`; `ModelFiles::map` in
  `metal/src/expert_io.hpp`; `STRATA_FILE_CACHE=keep` for the old behaviour). Old against new, A B B A: 3.0-4.5 GiB
  of other programs' memory compressed per run before, **none** after; 1-1.6 GiB free where there were 0.01-0.09.
- **n-gram rows are read ahead** (`metal/patches/0006`): llama.cpp's call for that was compiled out on macOS, so
  each row was a page fault, 97 µs against 6. The CPU's part of a 10,000-token prompt went from 9.6-10.0 s to
  4.2 s; the first token of a 500-token prompt comes 10% sooner.
- **The same long prompt did not give the same logits twice** (35.72, 35.64, 35.84 in three starts), since long
  before this evening: llama.cpp's Metal top-k for rows of more than 2,048 candidates handed out its output
  positions with an atomic counter, so ties at the threshold were broken by thread timing. That is every context
  deeper than 8,192 tokens. `metal/patches/0007` writes in index order; two starts now give identical IDs and
  logits (`metal/tools/determinism.py`). **Every earlier result beyond 8,192 tokens was one draw of that
  lottery**, the full-context runs included.
- **Passes of 8,192 tokens give the reference's logits bit for bit** on the gate's 7,900-token document (one pass
  where the reference took two). That question from the evening is closed.
- **`--reserve-gib 3`** instead of 4: now that the reserve is real memory it was measured again. 3 buys 1.3 GiB of
  cache (decode +5% and +11% in the two comparisons) for 0.44 GiB of compression in one run and none in the
  next; 2 buys nothing more and costs (no room for the read-ahead buffer). `--elastic`'s back-off also reacts to
  compression now, before it comes to swap; both of its signs have only fired on the tiny model.
- **No run of the night wrote to swap** (the counter is where it was: 1,586,587).
- **Next, largest first:** the KV cache reserves 3,072 MiB for all 131,072 tokens from the first one. The same
  plan at `--max-context 16384` has a 10.6 GiB expert cache instead of 7.1. A Metal buffer cannot be left
  partly untouched (it is resident as a whole from its first use: `file-cache/kvlazy.m`), so this needs one Metal
  buffer per stretch of cells over the same host memory, in `ggml-metal-device.m` (a set of views per buffer,
  like the ones `ggml_metal_buffer_map` makes), `llama_kv_cache` (no clear of the whole buffer; map more cells
  when a slot beyond the mapped ones is taken) and the engine's plan (the cache shrinks before the KV grows).
  Not started. After that: a full-context prompt on this build (not repeated tonight).
- The targets (120 and 10 tok/s at full depth) are still not reached; a warm Air reads at about 77 tok/s.

**Late evening, 2026-10-12.** The largest finding of the day came last, and it changes
how the sections below should be read:

- **The engine's "direct" SSD reads were not direct.** A `F_NOCACHE` read goes past the OS file cache only if its
  file offset and its buffer are both multiples of 4,096 bytes. The experts' were not, so about 40 GB per pass of
  a long prompt went through the file cache (1 GiB of page-ins a second), and macOS made room by compressing the
  other programs' memory, and at the edge by swapping. Reads are block-aligned now (`metal/src/expert_io.hpp`;
  `STRATA_ALIGNED_READS=0` gives the old ones). One 8,192-token prompt: page-ins 41.5 -> 1.6 GiB, other programs'
  memory compressed 16.5 -> 2.6 GiB. Output is bit-identical (real-model exact gate, three paths).
- **That was what made `--elastic` swap**, and the plain engine too (4 pages in one guarded run with the old
  reads). With aligned reads `--elastic` ran the test that had swapped three times, an A B B A against the plain
  engine, and an A B B A of pass sizes: no swap-out in any of them, 7.71 against 6.38 tok/s written (+21%).
- **Every speed figure below from before this point was measured with the unaligned reads**, on a kernel that was
  compressing gigabytes a minute beside the engine. The plain engine now reads the 12,288-token prompt at 95.5
  and 101.7 tok/s where the afternoon's six runs gave 63.7-79.3 (a warmer afternoon; not a controlled pair).
- **Passes of 8,192 tokens** read a 16,384-token prompt 9% faster than 4,096 (86.0 against 78.9 tok/s) with 44%
  less read from the SSD. They need `--elastic`: only the Long shape has the memory for them.
- **The full-context prompt once more**, with aligned reads, `--elastic` and 8,192-token passes: **70.14 tok/s**
  (31 min 7 s to the first token; 63.68 and 34 min 16 s earlier in the evening), **5.27 tok/s** written at full
  depth (4.54), five of five records, an identical repeat in 0.25 s, 805 GB read from the SSD instead of 1,400,
  **zero swap-outs**, pressure level 1, other programs keeping 2.3-4.3 GiB in RAM. That configuration is
  `pending/desktop-128k-elastic.json`; it is **not installed** (the owner's call; its swap back-off has only run
  on the tiny model).
- **Heat is what is left.** In that run the first 8,192 tokens took 72 s (114 tok/s) and every later pass
  102-142 s; 94% of the time was spent waiting for the GPU. The targets of 120 and 10 tok/s are still far, and
  on a fanless Air with these programs open nothing measured here points to a way there.
- Open, in order of likely value: the owner's decision on installing the elastic configuration; the n-gram
  table's rows, which still go through the file cache (about 0.9 GiB of page-ins per 8,192-token pass);
  `--prefill 8192` against a reference for near-ties; a long session through the server and pi; whether the
  measures that keep `--elastic`'s memory level (ballast, touching at once) are still needed with aligned reads.
- Tools: `metal/tools/swapwatch.py` runs any benchmark under a 5-second watch on the system's swap-out counter
  and kills the engine the moment it moves. **Use it for every real-model run.** Section 9 of the
  [report](bench/results/2026-10-12-m4-air-128k-resume/README.md) has the traces and the test programs.

(At the end of that evening the production config was still the plain 128k one. Since the night it is the
`--elastic` one: see the block above.)

The original goal was a usable **131,072-token total context** with the highest prefill and decode throughput this
machine can sustain while a browser, Ghostty, pi, and other normal desktop apps remain open. The later target was
**above 120 tok/s prefill and above 10 tok/s generation, without sacrificing quality**.

The first full-context result was the morning's: a completed **130,944-token input** at 131,072-token
capacity, original IQ3_S weights, **F16 KV**, all five exact records retrieved. Prefill **64.92 tok/s**
(**33 minutes 36.9 seconds** to the first token), decode **4.59 tok/s**; a repeat reused 130,943 tokens and
answered in **0.403 seconds**. It was **not swap-free** (61.7 MiB) and reported pressure level 2.

**The evening's is the one the installed configuration rests on:** the same prompt with the corrected plan,
`build-metal/bin/strata`, no `--elastic`. Prefill **63.68 tok/s** (**34 minutes 16 seconds** to the first token),
decode **4.54 tok/s**, five of five records, a repeat that reused 130,943 tokens and answered in **0.309 seconds**
with identical output. **Zero swap-outs** over the whole run (the system-wide counter read every 5 seconds from
before the engine's start to a minute after its exit), pressure level 1 in every sample, no compute-buffer warning.
It is one run on one evening's desktop: free memory was under 0.1 GiB for the half hour, and macOS held up to
14.9 GiB of other programs' memory compressed.

What the afternoon established:

- **Why that run's memory was off, and a plan that counts it.** The compute buffer grew by 442 MiB because of how
  ggml's allocator lays out a pass appended to a filled cache; it is now reserved at the start (2,209 MiB instead
  of 1,759 at 131,072 context with 4,096-token passes). 459 MiB of host memory for a pass's inputs was never in
  the plan and now is. The morning's candidate was 0.89 GiB larger than its plan said.
- **4,096-token passes are right** (79.3, 67.4 tok/s against 61.7, 59.5 for 2,048, same prompt, A B B A), and
  keeping the GPU busy during them changes nothing measurable (70.6 against 71.6 tok/s).
- **The Air throttles inside a run when it is warm**: the third 4,096-token pass took 70-93 s where the first took
  39 s. Heat, not the SSD, holds a long prompt near 65 tok/s.
- **`--elastic`** (new, opt-in): the 2.67 GiB of work memory that only long passes need and the expert cache take
  turns in one budget. A B B A: **8.01 against 6.66 tok/s** written (+20%), first token 2-10% sooner for prompts
  up to 1,000 tokens, 3.7 s later for a 2,000-token one.
- **`--elastic` swapped on this desktop**: 1,024 MiB at an engine start (a bug in its first form, fixed), then
  0.75 MiB and 41 MiB in two later runs at `--reserve-gib 4`. It is off by default and **must not be installed
  as it is**. The cause is not one bug: see "The memory this desktop really has".

None of this reaches 120 tok/s or 10 tok/s at full context, and no measurement here suggests a way to on this
Mac with the apps open: the GPU alone takes 82-126 ms per written token, and a warm Air reads at about 65 tok/s.

**Do not confuse the following states:**

| Item | State at handoff |
| --- | --- |
| Production Strata configuration | **131,072** context. Since the night of 2026-10-12: `--kv f16 --prefill 8192 --reserve-gib 3 --elastic on`, indexed QSA, the alias pi asks for (`pending/desktop-128k-elastic.json`). The evening's plain one (`--prefill 4096 --reserve-gib 4`, no `--elastic`) is `installed/strata-iq3_s.before-elastic.json`. The 32,768 file is kept as `bench/results/2026-10-12-m4-air-128k-resume/installed/strata-iq3_s.before.json` |
| pi local-provider configuration | `contextWindow` **131,072** (backup: `~/.pi/agent/models.json.backup-20261012-165047-32k`). The smoke test against the 128k server passed: a real `read` tool call and the marker. A long pi session was not tested |
| Completed full-context run | Two: the morning's (`build-metal/lowlevel-128k/strata-indexed-topk`, old plan, 61.7 MiB swapped) and the evening's (`build-metal/bin/strata`, new plan, **0 swap-outs**) |
| Repository source | Seven patches (`0001`-`0007`), engine with `--elastic`; built and tested (kernel tests, tiny-model parity, protocol suite, shape test). Applied once more to a clean checkout of the pin at the end of the night: 26 files, none differs |
| `build-metal/bin/strata` | Built from the current source (SHA-256 `42045d8f...` at the end of the night) |
| Default path | Same arithmetic as before; **not** the same memory plan: 450 MiB more reserved at 131,072 context with 4,096-token passes, host buffer subtracted |
| Real-model exact gate on the final build | Passed again at the end of the night against `strata-reference` (re-recorded) at 32,768 context with a 3,000 MiB cache: chat, code, 7,900-token document, identical ids and reported logits on the default, indexed and `--elastic` paths, and with `--elastic --prefill 8192` |
| Swap | The afternoon's `--elastic` tests wrote 1,066 MiB to swap (68,248 pages). macOS had 0.92 GiB in swap before, 1.95 GiB after. The evening's full-context run, server start and pi test wrote none (1,975.69 MiB in use before and after). The late evening's experiments before the reads were aligned wrote 6.2 MiB; nothing since (1,941.88 MiB in use at the end of the night) |
| Running work | Nothing was left running; the server was stopped again after each test |
| Git | The afternoon is committed as `8bfdb751` (source) and `cb36a298` (documents and results); the evening's result is the commit after them. `main` is ahead of `origin/main`; nothing was pushed. The night's work is the last commits on `main`; nothing was pushed by the assistant |

Commit `29c52020` contains the morning's optimization source, `2daacf82` its handoff and results. A commit is not
evidence that its contents were validated.

### The memory this desktop really has

The plan takes "free or file cache, minus swap, minus the reserve". Two things make that generous here:

- The other programs (a VM, Citrix, VS Code, the browser) had **14.4 GiB in macOS's compressor, held in 4.9 GiB**.
  The plan sees 4.9. When those programs are used, the rest comes back.
- macOS does not give up the file cache first: in one long-prompt run it compressed 10 GiB of other programs'
  memory while 5 GiB of file cache stayed.

The engine without `--elastic` got away with it all afternoon (no swap-out in about fifteen starts, prompts up to
16,200 tokens): it reserves 2.67 GiB of work memory and touches only part of it unless a long prompt is being
read. `--elastic` fills that memory with experts, and swapped. A full-context prompt touches all of it for half an
hour, which is where the morning's run swapped. **Whatever is tried next should watch `vm_stat`'s Swapouts over
the whole run, not the benchmark's per-request lines** (they do not cover the engine's start).

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
| Local experiment outputs | `build-metal/lowlevel-128k/`, `build-metal/tuning-128k/`, `build-metal/resume-128k/` (afternoon) |
| Preserved results | `bench/results/2026-10-12-m4-air-128k/` (morning), `bench/results/2026-10-12-m4-air-128k-resume/` (afternoon) |
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
**12.09 to 9.07 GiB**. The final cleaned patch removes the failed compact-mask dependency. It was rebuilt and
retested in the afternoon: **28/28** attention cases bit-identical (the 20 below plus selected counts 1/7/32/64 for
both head sizes), tiny-model parity and the protocol suite passed.

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
and tiny-model parity passed. The consolidated patch layout was built and validated in the afternoon (533/533 again).

### SIMD index sort — archived in the afternoon

Built and tested: 28/28 attention cases bit-identical with `STRATA_METAL_QSA_SIMD_SORT=1`, tiny-model parity
identical. Kernel time off/on/on/off at 256-wide heads: 3,580/3,606 against 3,499/3,496 us (65 queries), 15,235/
15,381 against 15,010/14,951 us (256 queries): about 2% of an operation that is 7% of a pass. Not worth a kernel
that assumes 32-lane SIMD groups. The patch and its timings are under
`bench/results/2026-10-12-m4-air-128k/rejected-experiments/`; the environment variable no longer exists.

### Reserve for an appended pass, and the host buffer — `metal/patches/0004-kpool-append-reserve.patch` (afternoon)

Always active. This is the morning's "unresolved memory warning" (2,201 MiB against an expected 1,759 MiB).

llama.cpp reserves the graph of a full cache in which every pool of the QSA index is re-pooled (`n_new` = 32,768 at
131,072 context). A pass appended to a filled cache re-pools about `n_tokens/4 + 1` pools. Every tensor of that
graph is smaller or equal, but ggml-alloc is best-fit: without the large re-pooling tensors early in the graph the
513 MiB index score tensor (`n_pool x n_tokens` F32) finds no hole and extends the buffer. The allocator plans
each pass afresh as `n_pool` grows (a pass at shallow depth uses the bitonic top-k, whose scratch is larger than
the reserved radix one, and from then on every pass is larger than the last plan), so the buffer was freed and
allocated again, larger, on every pass past about 18,000 tokens.

The patch adds `llama_memory_hybrid_idx_context::kpool_reserve_append` and reserves that graph too, first, so the
buffer is still allocated once. Reserved at 131,072 context: 4,096-token passes 1,759 -> **2,209 MiB**; 2,048 and
1,024-token passes unchanged (1,351 and 1,125 MiB: their score tensors fit the holes). It also adds
`llama_get_compute_buffer_sizes(ctx, &device, &host)`; the engine subtracts the host part (459 MiB for 4,096-token
passes, `malloc`ed and untouched when the plan is made) before sizing the cache.

Verified in the evening by a full-context run: 130,944 tokens ended without the mismatch warning, so 2,209 MiB
was not exceeded. (It equals what the morning's run ended with plus the 64 pools of padding; a 16,200-token run at
16,384 context also ended at its reserved size.)

Looked at and dropped: Metal reserves 256 MiB of F16 dequantization scratch per attention operation even for an
F16 KV cache. Skipping it changed the reserved size by 0 MiB.

### `--elastic` — `metal/patches/0005-context-resize.patch`, `metal/src/main.cpp` (afternoon)

Opt-in, off by default, experimental. **It swapped on this desktop until the reads were aligned** (see "Read this
first"); the description below is the afternoon's form, and the late evening's changes are in section 9 of the
report: memory is taken at once on a change to or from the Long shape, the idle part of the cache's memory is
held, the long passes fall back to half or a quarter of `--prefill` if the whole does not fit, and
`LLAMA_MOE_CACHE_NO_CLEAR` is gone.

`llama_context_resize(ctx, n_batch, moe_cache_size)` frees the scheduler's compute buffers and the MoE cache and
makes them anew for another pass size and cache size; a cache whose size does not change is kept with its
contents and callbacks. The memory module is not touched (the patch refuses sliding-window models, whose memory is
sized with the ubatch). One trap is handled in it: `output_ids` is sized once from `n_batch`; growing the batch
without growing it overflowed the heap (found with AddressSanitizer on the tiny model).

The engine keeps the context in one of three shapes:

| Shape | Passes | Cache | When |
| --- | --- | --- | --- |
| Answer | 16 tokens | everything the budget has | writing; stretches of up to 32 new tokens |
| Short | `--elastic-short` (1,024) | kept | a stretch one such pass reads, if its 0.94 GiB + 1.5 GiB are free at that moment |
| Long | `--prefill` | 256 MiB (it only carries the engine's read hooks) | anything longer; the cache is read back afterwards, 1.7-2.4 s for 6.5 GiB |

Why 16 and not 32 for Answer: from 32 tokens llama.cpp computes a layer the cache does not serve on the GPU from
copies of whole expert tensors, which costs 0.56 GiB of work memory and makes the shape's size depend on the
cache. Below 32 the size is the same with and without a cache, so every shape can be measured on the cache-less
probe context and **the cache is allocated exactly once** at start (the first form allocated a provisional cache,
then the final one: 1,024 MiB of swap-outs).

Passes are the ones the engine makes without `--elastic` (same sizes, same kernels), so output is identical:
tiny-model parity (ids and logits), the protocol suite, `metal/tests/elastic_shapes_test.py` (six requests through
all three shapes), the same under AddressSanitizer, and the real-model gate in the afternoon report.

Safety added after the swaps, in this order: the cache comes back from the Long shape only as large as leaves
`--reserve-gib` free at that moment; it is not cleared when made (`LLAMA_MOE_CACHE_NO_CLEAR`: fresh Metal pages
are zero anyway), so it takes memory at the SSD's pace; before a request, if macOS's swap-out count moved since
the last one, the cache gives a quarter of itself back (at most 1 GiB) and stays smaller. **The last one has only
run on the tiny model**, with the count faked (`STRATA_ELASTIC_TEST_BACKOFF=1`). After the gentler allocation went
in, a real-model run still swapped 41 MiB; after that `--elastic` ran on the real model only once more, in the
exact gate with a cache capped at 3,000 MiB (no swap-out, identical output).

Numbers (A B B A, 131,072 context, `--reserve-gib 4`, prompts of 64-2,000 tokens, 24 tokens written each): cache
5,312/6,298 MiB plain against 7,878/8,542 MiB; written 6.66 against 8.01 tok/s; per token 151 ms (GPU 85, SSD 55)
against 125 ms (GPU 82, SSD 34); first token 7.2 -> 6.4 s (64 tokens), 14.5 -> 14.1 s (1,000), 22.9 -> 26.6 s
(2,000, Long shape).

Other engine changes of the afternoon: end of input on stdin now cancels the request under way (a killed server
used to leave the engine reading its prompt to the end, up to half an hour); `--verbose` prints the memory after
every prompt pass and every change of shape.

Tests as they stand, all run on the final binary (`bench/results/2026-10-12-m4-air-128k-resume/tests/`):

- `strata-test-indexed-qsa`: 28 cases. `--bench` adds six timing cases.
- `strata-test-backend-ops -b MTL0 -o TOP_K`: 533 cases.
- `optimization_parity_test.py`: default, indexed, elastic, elastic-long against `strata-before-indexed`; five
  prompts of 15-513 tokens and a forced speculative run.
- `serve_protocol_test.py`: 29 checks, three of them `--elastic`.
- `elastic_shapes_test.py`: new.
- The tiny fixture is `build-metal/tuning-128k/tiny/qwen4exp-moe.gguf` (about 18 MiB). **Do not run it at
  `--max-context 131072`**: its graph reserves 15 GiB.

`metal/CMakeLists.txt` has the optional test targets `strata-test-indexed-qsa` and `strata-test-backend-ops`, with
CTest entries for attention and TOP_K. Tests default off; the local build cache has them on.

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
| Current build, new plan (evening), **130,944** | **4,096** | **6.02 GiB + 0.45 GiB host** | **4,933 MiB** | **63.68** | **2,056.18 s** | **4.54** | **0** |

The 8k-batch run paged about 91.7 MiB and was not selected for the desktop candidate. An earlier unindexed 8k-batch
attempt planned 12.09 GiB and zero expert cache; it was stopped before completion. `outputs-limited.*` is **not** a
completed benchmark. A four-thread C++ build overlapped part of the 16k run, so its speed is confounded. The 16k
functional test still retrieved 5/5 records and repeated all 76 output IDs with 16,383 reused tokens and 0.379 s TTFT.
The last row is the evening's run with `build-metal/bin/strata`; its start memory, pass times and the 5-second swap
watch are in [section 8 of the afternoon report](bench/results/2026-10-12-m4-air-128k-resume/README.md#8-evening-the-full-context-prompt-with-the-new-plan-and-the-install). The details below are the morning's run.

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

**The memory warning, resolved in the afternoon:** the completed engine log ends with:

```text
~llama_context: MTL0 compute buffer size of 2201.2842 MiB, does not match expectation of 1759.2842 MiB
```

See "Reserve for an appended pass" above for the cause and the fix. The final
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
| SIMD index sort (afternoon) | Correct; about 2% of the attention operation. Archived |
| `--prefill 2048` instead of 4096 (afternoon) | 12,288-token prompt, A B B A: 61.7, 59.5 against 79.3, 67.4 tok/s. Needs 1.06 GiB less memory and no append reserve, reads twice the bytes (233 against 149 GB). Kept at 4,096 |
| Keep-alive off during long passes (afternoon) | 71.9, 71.4 against 77.5, 63.7 tok/s with it on: nothing outside the noise. The experimental `STRATA_KEEPALIVE=decode` was removed again |
| No F16 dequant scratch for an F16 KV cache (afternoon) | 0 MiB less reserved. Dropped |
| Reading a prompt through the cache instead of a long pass (afternoon, a first form of `--elastic`) | 27.5 s for 250 tokens, 122 s for 1,000 (8 tok/s, 300 GB read), and the answer after it slower too. Only stretches of up to 32 tokens go that way |

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

The morning's list, with where it stands:

1. ~~Reconcile source, binaries, and tests.~~ **Done** (afternoon report, section 1).
2. ~~Account for actual full-depth memory.~~ **Done for the engine's own memory**: the compute-buffer growth and
   the host buffer are in the plan. Still out of the plan, by design: the read-ahead buffer (1.1 GiB, taken only if
   that much plus 1.5 GiB is free when a long prompt starts) and conversation checkpoints (113 MiB each, up to
   six). **Not done: the other programs' compressed memory** ("The memory this desktop really has").
3. ~~Keep-alive off during bulk prefill.~~ **Measured, no gain, removed.**
4. Fair final F16 comparison: **partly.** The afternoon's A B B A runs compare settings of the current build; the
   real-model gate compares it with `strata-reference` for output. A sustained decode comparison against
   `strata-reference` was not run.
5. Desktop integration: **done in the evening**, after the swap-free full-context run: configuration installed,
   pi's context window raised, smoke test passed. Not tested: a long prompt through the HTTP server or pi.

What is open now, each needing the user's go-ahead because it takes the Mac for half an hour or can swap:

1. ~~A full 130,944-token prompt with the new plan, without `--elastic`, watching Swapouts over the whole run.~~
   **Done in the evening: no swap-out**, 63.68 tok/s; `pending/desktop-128k.json` is installed and pi points at it.
   Free memory was under 0.1 GiB while the prompt was read, so item 3 still matters.
2. **`--elastic` with a larger reserve** (`--reserve-gib 6` in `pending/desktop-128k-elastic.json` is a guess):
   first the five-request run that swapped 41 MiB (`elastic/elastic-long-trips.*` in the afternoon report), then
   a full prompt. Its back-off has never met a real swap-out. Running it with the VM and Citrix closed would show
   how much of the trouble is this desktop.
3. **A plan that knows about compressed memory.** `host_memory()` could count part of "stored in compressor
   minus occupied by compressor" as owed, the way it counts swap. How large a part is not known; it would shrink
   every cache.
4. **`--prefill 8192` in the Long shape.** With `--elastic` the Long shape has about 2.9 GiB to spare at 131,072
   context, enough for 8,192-token passes (3.28 GiB on the GPU + 0.92 GiB host), which halve the SSD traffic of
   a long prompt. Not measured.
5. The read-back after a long prompt (about 2 s for 6.5 GiB) could be shortened by keeping the hottest part of
   the cache through the Long shape, or made free by reading back while the first tokens are written. Neither is
   written.

Further ideas, in approximate order of likely value:

| Idea | Why it might help / what remains unknown |
| --- | --- |
| Reclaim prefill workspace for decode cache | **Implemented in the afternoon as `--elastic`**: +20% written tok/s at the same plan, and swap-outs on this desktop. See above |
| Save session state across idle unload | A repeated 128k request can start in 0.4 s while resident, but cold refill costs 33.6 minutes. The 900 s idle unload loses that benefit. The Metal engine has in-memory recurrent checkpoints, not a full persistent KV/GDN/QSA session format. Disk state would be large and needs correct model/template identity and state restoration. No implementation |
| SIMD integer QSA sort | Built, correct, about 2% of the operation: archived in the afternoon |
| Full-depth kernel/dispatch profiling | Locate the sustained GPU bottleneck after removing workspace waste. Quantized matrix-vector bandwidth and dispatch overhead are more promising than optimizing a 4–5 ms CPU component alone |
| Router BF16 dot/NEON work | `metal/src/router_predict.hpp` already uses NEON F32 FMA over BF16 weights. Check hardware BF16 instructions and generated assembly first. Rounding predictor inputs changes predictions, not actual model routing, but might hurt prefetch hit rate. Use offline router traces and end-to-end measurements; untested |
| Prefetch confidence margin | Read fewer false positives to save SSD work and heat, at the risk of more blocking misses. Depth/count already have negative results; confidence margin still needs a specific test. Predictions must never replace the real router |
| Pressure-aware cache | With `--elastic` only: the cache is re-fitted to the free memory when it comes back from a long prompt, and gives memory back before a request if macOS has swapped. The second has run on the tiny model only. Without `--elastic` the cache is still fixed at start |
| Better expert-tile grid locality | The rejected flat grid may have disturbed useful expert/M/N tile ordering. Revisit only with a concrete locality hypothesis; previous 1.3% change was inconclusive |
| Use resident cached experts during bulk streaming | Might avoid SSD bytes and heat, but requires copy/synchronization work. Unoverlapped SSD was a small share of full prefill. No demonstrated benefit |
| Metal Tensor API on M4 | `GGML_METAL_TENSOR_ENABLE=1` was not tested; upstream default on this M4 was disabled. Different floating-point order requires a quality gate. Do not assume a gain |
| Lookup speculation on representative code tasks | Tiny forced lookup verification passed, but end-to-end pi workloads and quality at full context were not measured. Batched verification can change rounding just as MTP did; keep it off until measured |

Do not spend another long run repeating relaxed matrix barriers, compact masks, chunk-64 GDN, deeper prefetch,
cache-policy sweeps, 2,048-token passes or keep-alive changes without a new hypothesis. Raising Metal's recommended limit or using `--overcommit-gib` was not
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

## Production and pi integration

**Installed on the evening of 2026-10-12.** Production `strata-iq3_s.json` is now `pending/desktop-128k.json`
(131,072 context, no `--elastic`), pi's `contextWindow` is 131,072, `/health` and `/v1/models` report 131,072, and
the pi smoke test below made a real `read` tool call and returned the marker. To go back, copy
`bench/results/2026-10-12-m4-air-128k-resume/installed/strata-iq3_s.before.json` over `strata-iq3_s.json` and
`~/.pi/agent/models.json.backup-20261012-165047-32k` over `~/.pi/agent/models.json`, then start the server again.
The rest of this section was written before the install; where it says 32,768 it describes that state.

Production `strata-iq3_s.json` points at `build-metal/bin/strata`, uses the native model and expert profile, automatic
expert cache, and `--max-context 32768`. Its API model name is `qwen3.8-flash-next-iq3_s`, backend `metal`, port 8080,
and idle unload 900 seconds. `run-iq3_s.sh` starts `.venv/bin/python serve/server.py --engine strata --config
strata-iq3_s.json --port 8080 --open` from the repository.

The candidates are in `bench/results/2026-10-12-m4-air-128k-resume/pending/` (the morning's
`pending/desktop-128k.json` is the same as the first of them, which is the one installed in the evening):

```text
desktop-128k.json          --max-context 131072 --kv f16 --prefill 4096 --reserve-gib 4
desktop-128k-elastic.json  the same with --elastic on and --reserve-gib 6 (experimental; 6 was not run)
both                       env: STRATA_METAL_INDEXED_QSA=1     aliases: qwen3.8-27b-iq3_s
```

They leave idle unload unchanged and do not enable MTP or lookup. They point to the ordinary engine binary, which
is now built from the current source. Re-read and merge current production settings before applying one; do not
overwrite intervening user changes. `desktop-128k.json` has read a full-context prompt with this build (evening);
the `--elastic` one has not.

pi was version **1.0.2**, executable `/Users/rafal/.bun/bin/pi`, linked into the global
`@earendil-works/pi-coding-agent` package. In `/Users/rafal/.pi/agent/models.json`, provider `qwen-local` uses
`http://127.0.0.1:8080/v1`, API `openai-completions`, model ID **`qwen3.8-27b-iq3_s`**, reasoning enabled, text input,
`contextWindow: 32768`, and `maxTokens: 8192`. The old ID/display name misleadingly says 27B; preserve compatibility
through the server alias if the display name is corrected. The global pi default was an OpenRouter DeepSeek model,
not the local provider. Do not change global defaults or print credentials while updating one local entry.

After future validation and installation, set the local provider's context window to 131,072 and check `/health`
and `/v1/models` plus actual tool use. A prepared fixture at `build-metal/lowlevel-128k/pi-smoke.txt` contains
`STRATA_PI_READ_CONFIRMED_7c4a91e2`. The pi smoke test (run in the evening against the 128k server, passed) is:

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

The afternoon's evidence is in `bench/results/2026-10-12-m4-air-128k-resume/`: `tests/` (every test on the final
binary), `memory/` (the per-pass memory trace and the 16k run that ended at its reserved size), `batch-ab.*`,
`rejected/keepalive-ab.*`, `elastic/` (the A B B A run, the first three-engine run with the 1,024 MiB swap, the
five-request run with the 41 MiB swap, a verbose run with the shape changes timed), `gate/` and `pending/`.
The evening's is in the same directory: `full-128k/` (the result JSON, console, engine log, the swap watch and its
script) and `installed/` (the configuration before and after, `/health`, `/v1/models`, pi's transcript, the server's
engine log).

The morning's results directory preserves the finished full-context JSON, engine log, console output, and separate memory
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
| `build-metal/lowlevel-128k/strata-relaxed-experimental` | The morning's last `bin/strata` (relaxed-sync tree), kept before the afternoon's rebuild |
| `build-metal/resume-128k/strata-step1` | The consolidated morning patches as first rebuilt in the afternoon (with the SIMD patch, before `0004`/`0005`) |
| `build-metal/bin/strata` | Current source |

`docs/APPLE_METAL.md` remains useful for architecture and older experiments, but some status statements are stale:
it says the largest exercised context was 40,960, describes the now-fixed planner problem as open, and gives older
batch settings and closed-app advice. The 2026-10-11 report also has historical pending full-context language.
Use this handoff and the finished F16 result for the latest status. The 2026-10-12 report was updated with completion
and a pointer here; its prior correctness numbers still refer to the earlier compiled implementation.

The next agent should be able to start from the open list under "Next steps", with the user's answer to which of
the swap-risking runs they want.
