# M4 Air, 128k context: continuing the optimization (2026-10-12, afternoon)

Machine: MacBook Air M4, 10 CPU cores, 10 GPU cores, 32 GiB unified memory, AC power, Low Power Mode off, fanless.
Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S, unchanged weights, F16 KV cache. llama.cpp is pinned to
`23b0202a189c44a54625aadcb37a946dd1d6278d`. Brave, Citrix, a VM, VS Code and the Claude app stayed open; the Mac had
been under load for hours, so it was warm for every run (battery sensor 34-35.5 C).

This continues the [morning's report](../2026-10-12-m4-air-128k/README.md) and the steps listed in
[OPTIMIZATION.md](../../../OPTIMIZATION.md). The targets (120 tok/s prefill, 10 tok/s decode at 131,072 context)
are still not reached. Three test runs of the new `--elastic` mode made macOS swap (1,066 MiB in all); no other
run did.

**Evening of the same day** ([section 8](#8-evening-the-full-context-prompt-with-the-new-plan-and-the-install)): a
full 130,944-token prompt was read with the corrected plan, without `--elastic`, and macOS did not swap out one page.
After that the 128k configuration was installed and pi tried against it. Sections 1-7 are the afternoon as it was
written.

**Night** ([section 9](#9-later-that-evening-the-engines-direct-reads-were-not-direct),
[section 10](#10-night-the-model-in-the-file-cache-n-gram-rows-and-a-top-k-that-was-not-reproducible)): the
engine's "direct" reads had been going through the file cache, which was what made `--elastic` swap; with that
fixed the `--elastic` configuration was installed. Then: 8 GiB of the model were sitting in the OS file cache
(given back now), the n-gram rows were read one page fault at a time (read ahead now), and the same long prompt
did not give the same logits twice (a top-k kernel of llama.cpp; fixed).

**Late night and the morning of 2026-10-13** ([section 11](#11-late-night-the-kv-cache-takes-its-memory-as-the-conversation-grows),
[section 12](#12-morning-of-2026-10-13-the-expert-cache-is-not-read-back)): the KV cache takes its memory as the
conversation grows, and the expert cache is no longer read back from the SSD after a long prompt: it fills as its
experts are asked for, and the first token after a 2,000-token prompt comes 3.1 s sooner (A B B A, caches of
10.4-11.0 GiB).

## In short

- **The memory plan is honest now.** The 442 MiB by which the compute buffer grew during the morning's full-context
  run is explained and reserved up front, and 459 MiB of host memory that the plan never saw is counted. That is
  0.89 GiB the morning's 128k candidate was using out of its reserve without saying so.
- **Passes of 4,096 tokens stay**: 21% faster on a 12,288-token prompt than 2,048. Keeping the GPU busy during long
  passes stays too (turning it off changed nothing measurable). The SIMD index sort is correct and was archived
  (about 0.15% of a pass).
- **`--elastic` (new, opt-in, experimental)** lets the long passes' work memory and the expert cache take turns in
  one budget. In an A B B A run it wrote answers **20% faster** (8.01 against 6.66 tok/s over twelve requests) with
  a first token 2-10% sooner for prompts up to 1,000 tokens.
- **`--elastic` is not safe to turn on as it is on this desktop: it made macOS swap.** Three of its test runs wrote
  to swap: 1,024 MiB (a start-up bug, fixed), then 0.75 MiB and 41 MiB with the fix in. It uses every byte of its
  plan, and the other programs here hold about 14 GiB in compressed form that the plan counts as 5 GiB. Nothing was
  installed: production Strata and pi are as they were (32,768 context).
- No run without `--elastic` swapped (about fifteen engine starts, prompts up to 16,200 tokens).

## 1. Source, binaries and tests reconciled

The consolidated patches were configured and built from a clean checkout of the pin. On that build:

| Check | Result |
| --- | --- |
| Indexed attention against the dense-mask path, 28 cases (heads 128/256, 1-4,096 selected cells, 131,072 KV cells) | 28/28 bit-identical floats |
| Metal TOP_K (upstream test-backend-ops) | 533/533 |
| Tiny model, 5 prompts of 15-513 tokens and a forced speculative run, default / indexed against `strata-before-indexed` | identical ids and reported logits |
| `--serve` protocol suite, default and indexed | all passed |

The same checks pass on the final build of this session, with two more variants for `--elastic` (section 5).

## 2. SIMD index sort: correct, not kept

The unbuilt `0004-metal-qsa-simd-sort.patch` builds and is correct: 28/28 attention cases bit-identical with
`STRATA_METAL_QSA_SIMD_SORT=1` (the log shows the `_simd` kernel was the one compiled), tiny-model parity identical.
Kernel time, off/on/on/off, 2,051 selected cells of 131,072, 100 runs each:

| Head, queries | Off (us/run) | On (us/run) | Change |
| --- | ---: | ---: | ---: |
| 256, 1 | 529, 577 | 573, 566 | +3% (slower) |
| 256, 65 | 3,580, 3,606 | 3,499, 3,496 | -2.7% |
| 256, 256 | 15,235, 15,381 | 15,010, 14,951 | -2.1% |
| 128, 65 | 2,020, 2,167 | 1,895, 2,076 | -5% |

The model's heads are 256 wide. Two percent of an operation that was 7% of a prompt pass's GPU time is about 0.15%
of the pass, and the kernel assumes Apple's 32-lane SIMD groups. It was moved to
[rejected-experiments](../2026-10-12-m4-air-128k/rejected-experiments/0004-metal-qsa-simd-sort.patch) with its
[timings](../2026-10-12-m4-air-128k/rejected-experiments/simd-sort/).

## 3. The 442 MiB the compute buffer grew by: found, and planned for

The morning's full-context run ended with a compute buffer of 2,201 MiB where 1,759 MiB had been reserved.

**Cause.** llama.cpp reserves the graph of a full cache in which *every* pool of the attention index is re-pooled
(32,768 pools at 131,072 context). A pass that is appended to a filled cache re-pools only the ~1,025 pools it
touches. Its tensors are all smaller or equal, but ggml's allocator is best-fit: without the large re-pooling
tensors early in the graph, the 513 MiB score tensor of the index (pools x tokens) finds no hole to reuse and is
placed at the end of the buffer. The allocator lays every pass out again as the pool count grows, so the buffer
grew pass by pass once the prompt was past about 18,000 tokens, and was freed and allocated anew each time.

**Fix** (`metal/patches/0004-kpool-append-reserve.patch`): the context also reserves the graph of such an appended
pass, and reserves it first so that the buffer is still allocated once. Reserved at 131,072 context, measured with
`--load-only --verbose`:

| Passes of | GPU compute buffer before | after | Host compute buffer |
| ---: | ---: | ---: | ---: |
| 4,096 tokens | 1,759 MiB | **2,209 MiB** | 459 MiB |
| 2,048 tokens | 1,351 MiB | 1,351 MiB | 230 MiB |
| 1,024 tokens | 1,125 MiB | 1,125 MiB | 116 MiB |
| 32 tokens (answering only) | - | 353 MiB | 5 MiB |

Only 4,096-token passes are affected; 2,209 MiB is what the morning's run ended with (2,201 MiB at a slightly
smaller pool count).

**The host buffer was not in the plan either.** The inputs of a pass live in host memory: 459 MiB for 4,096-token
passes. It is `malloc`ed and only backed by pages as passes touch it, so the plan's measurement did not see it.
The same patch adds `llama_get_compute_buffer_sizes`, and the engine now subtracts the host part before sizing the
expert cache. The plan line reads, for example:

```text
strata metal: context 6.02 GiB on the GPU (2.16 of it for the passes) and 0.45 GiB of host memory; expert cache ...
```

Together the plan now accounts for 0.89 GiB that the morning's candidate took out of its 4 GiB reserve unseen.
The price is an expert cache that is smaller by that much (about 465 experts) at 131,072 context with 4,096-token
passes. Section 5 gets that back, and more.

One more reservation was looked at and left alone: Metal always reserves 256 MiB of scratch per attention operation
for dequantizing a quantized KV cache, also when the cache is F16. Skipping it for F16 changed the reserved size
by 0 MiB (the attention operation is not where the buffer peaks), so the patch was dropped.

## 4. Two settings measured and left as they were

Both: 12,288-token prompt at 131,072 context, 4 GiB reserve, each run started below 45 C, order A B B A.

| Question | A | B | Verdict |
| --- | --- | --- | --- |
| Passes of 4,096 or 2,048 tokens? | 4,096: **79.3, 67.4** tok/s | 2,048: 61.7, 59.5 tok/s | 4,096 reads 21% faster (median 73.4 vs 60.6). 2,048 needs 1.06 GiB less memory, but twice the SSD traffic (233 vs 149 GB) |
| GPU keep-alive off during long passes? | always on: 77.5, 63.7 tok/s | off in long passes: 71.9, 71.4 tok/s | 70.6 vs 71.6: no difference outside the noise. Not kept; the experimental mode was removed again |

Raw: [batch-ab.json](batch-ab.json), [rejected/keepalive-ab.json](rejected/keepalive-ab.json).

The third pass of these runs took 70-93 s where the first took 39 s: the warm Mac throttles within two minutes.
That, not the SSD, is what holds a long prompt near 65 tok/s.

## 5. `--elastic`: one budget for the long passes' work memory and the expert cache

A long prompt's passes need 2.67 GiB of work memory at 131,072 context (2.21 on the GPU, 0.46 in host memory) that
answering does not use; answering wants the expert cache, which those passes hardly use. Without `--elastic` both
are held all the time. With it the context has one of three shapes, changed with a new `llama_context_resize`
(`metal/patches/0005-context-resize.patch`); the KV cache and the recurrent state are never touched:

| Shape | Passes | Expert cache | Used for |
| --- | --- | --- | --- |
| Answer | up to 16 tokens | all of the budget | writing, and stretches of up to 32 new tokens |
| Short | up to 1,024 tokens (`--elastic-short`) | kept as it is | a stretch one such pass reads, if 0.94 GiB + 1.5 GiB are free at that moment (the rule the read-ahead buffer already follows); given back after the pass |
| Long | `--prefill` tokens | a 256 MiB token cache | longer prompts; on the way back the cache is read again from the SSD (6.5 GiB: 1.7-2.4 s) |

The plan on the same start, four seconds apart: 4.23 GiB of cache without `--elastic`, **6.50 GiB** with it
(3,401 experts instead of about 2,220).

**Output is unchanged.** The passes are the ones the engine makes without `--elastic`; only where their memory
comes from differs. Tiny model: identical ids and logits in the parity test, in the protocol suite, and in a new
six-request test that goes through all three shapes ([tests/](tests/)); the same under AddressSanitizer.
Real model, against `strata-reference` (the build before the low-level work), 32,768 context, 4,096-token passes,
cache capped at 3,000 MiB, `--chunk 2`, 32 tokens written, `--exact`: **chat, code and a 7,900-token document give
identical ids and identical reported logits** on the default path, with `STRATA_METAL_INDEXED_QSA=1`, and with
`--elastic on` (which took the Short shape for the code prompt and the Long shape for the document). No swap-outs
in these four engine runs ([gate/](gate/)). This is the one `--elastic` run on the real model after the back-off
went in; with a 2.93 GiB cache it was never near the memory's edge.

**Measured** (131,072 context, 4 GiB reserve, prompts of 64 to 2,000 tokens with 24 tokens written after each,
order plain, elastic, elastic, plain; [elastic-short3.json](elastic/elastic-short3.json)):

| | Plain | `--elastic` |
| --- | ---: | ---: |
| Expert cache (two runs) | 5,312 / 6,298 MiB | 7,878 / 8,542 MiB |
| Writing, mean of 12 requests (276 tokens) | 6.66 tok/s | **8.01 tok/s** (+20%) |
| Per written token | 151 ms = GPU 85 + SSD 55 + CPU 11; 58 misses | 125 ms = GPU 82 + SSD 34 + CPU 10; 37 misses |
| First token, 64-token prompt | 7.21, 7.09 s | 6.40, 6.47 s |
| First token, 250 tokens | 8.89, 8.78 s | 8.48, 8.49 s |
| First token, 1,000 tokens | 14.44, 14.56 s | 14.10, 14.20 s |
| First token, 2,000 tokens (Long shape) | 23.00, 22.71 s | 26.09, 27.10 s (+3.7 s) |
| Swap-outs | 0, 0 | **48 pages**, 0 |

Twenty-four tokens per request is a short sample, and the Mac was warm (SoC 49 C at each start); the order of the
runs is what makes the comparison fair. The 2,000-token row is the price of the Long shape: 1.7-2.4 s to read the
cache back, and a pass that finds no cached experts to copy (49 GB read against 44).

An earlier form of the feature showed what does not work: reading a prompt through the cache instead of in a long
pass takes 27.5 s for 250 tokens and 122 s for 1,000, at 8 tok/s
([elastic-short.json](elastic/elastic-short.json), third engine).

### It swapped

| Run | Swap-outs | What happened |
| --- | ---: | --- |
| First build, engine start, 8,247 MiB cache | 65,568 pages = **1,024 MiB** | The start made a provisional 7 GiB cache, measured, freed it and made the final 8 GiB one. Fixed: every shape is now measured on the context that has no cache, and the cache is allocated once |
| Final design, 2,000-token request, 7,878 MiB cache | 48 pages = 0.75 MiB | on the way back from the Long shape |
| Final design, five 2,000-2,400-token requests, 6,047 MiB cache | 2,632 pages = **41 MiB** | in the third and fifth request; these also slowed from 25 s to 47-49 s to the first token |

(16 KiB pages. The per-request lines of the first run all said 0: a request's count does not cover the engine's
start. The third run already had the gentler allocation described below.)

What was done about it, and what was not:

- **Done:** the cache comes back only as large as leaves `--reserve-gib` free at that moment (other programs may
  have grown into the memory it gave up); it is no longer cleared when it is made, so it takes its memory at the
  SSD's 3 GB/s as the experts are read back instead of within half a second; and before every request the engine
  looks at macOS's swap-out count: if it moved, the cache gives a quarter of itself back (1 GiB at most) and stays
  smaller. That last part ran on the tiny model only (with the count faked): identical output, six back-offs.
- **Not done:** `--elastic` was not run at a plan-sized cache on the real model again after the 41 MiB (only the
  capped gate above). Whether a larger reserve makes it swap-free here is not known.

Why it swaps where the plain engine did not: the plain engine reserves its 2.67 GiB of work memory and touches
only part of it unless a long prompt is being read, so about 1.4 GiB more stays free than its plan says.
`--elastic` puts all of it into the cache, which is filled. And the plan's idea of free memory is generous on this
desktop: during these runs the other programs (a VM, Citrix, VS Code, the browser) had 14.4 GiB in the compressor,
squeezed into 4.9 GiB. The plan counts the 4.9. When those programs are used, that memory comes back. A run in the
afternoon also showed macOS compress 10 GiB of other programs' memory during a long prompt while 5 GiB of file
cache stayed (no swap-out then).

## 6. Smaller things in the engine

- **A request ends when nobody is left to read it.** When stdin closed (the server or a benchmark killed), a
  request under way went on to its end, up to half an hour with 16 GiB held. End of input now sets the stop flag.
- **`--verbose` prints the memory after every prompt pass** (`pass of 4096 tokens to 20480: ... MiB on the GPU`).
- `metal/tests/optimization_parity_test.py` has the variants default, indexed, elastic and elastic-long;
  `metal/tests/serve_protocol_test.py` three `--elastic` cases; `metal/tests/elastic_shapes_test.py` is new.

## 7. State at the end of the afternoon

Superseded in the evening for the rows "Production config, pi", "Candidates" and "Git": see section 8.

| | |
| --- | --- |
| Patches | `0001` (unchanged), `0002-metal-indexed-qsa`, `0003-metal-topk-workspace`, `0004-kpool-append-reserve` (new), `0005-context-resize` (new). They were applied to a clean checkout of the pin once more at the end, and the engine built from that |
| `build-metal/bin/strata` | built from the current source (SHA-256 `7835bd8c...`, in [artifact-manifest.json](artifact-manifest.json)); every file in [tests/](tests/) is from this binary. The real-model gate and the measurements ran on builds before the last one-line change (a guard in the context's destructor for a failed resize) |
| Default path | not byte-identical to the morning's: `0004` reserves 450 MiB more at 131,072 context with 4,096-token passes, and the plan subtracts the host buffer. No arithmetic changed. `--elastic` and `STRATA_METAL_INDEXED_QSA` are off unless asked for |
| Production config, pi | untouched: 32,768 context |
| Candidates | [pending/desktop-128k.json](pending/desktop-128k.json) (no `--elastic`), [pending/desktop-128k-elastic.json](pending/desktop-128k-elastic.json) (experimental). Neither has read a full 131,072-token prompt with this build |
| Swap | 1,066 MiB (68,248 pages) written by this session's `--elastic` tests; macOS had 0.92 GiB in swap before and 1.95 GiB after |
| Git | nothing committed |

Not run in this session, each because it takes the Mac for half an hour or risks swap, and so is the owner's call:
a full 130,944-token prompt with the new plan; `--elastic` with a larger reserve; `--prefill 8192` in the Long
shape (the memory is there in that shape); pi against a 128k server.

## 8. Evening: the full-context prompt with the new plan, and the install

The first of the runs listed above, done at the owner's request. Same binary as the afternoon's final build
(`build-metal/bin/strata`, SHA-256 `7835bd8c...`), no `--elastic`, nothing built or run on the GPU beside it. A VM,
Citrix, Brave, VS Code, Ghostty and the Claude app were open. Raw files: [full-128k/](full-128k/).

```sh
STRATA_METAL_INDEXED_QSA=1 .venv/bin/python -u metal/tools/bench.py --config strata-iq3_s.json \
  --context 131072 --prefill 130944 --prefill-new 120 --needle --prefill-repeat \
  --json build-metal/resume-128k/full-128k-newplan.json --log build-metal/resume-128k/full-128k-newplan.log \
  -- --kv f16 --prefill 4096 --reserve-gib 4
```

It ran under [run_full.py](full-128k/run_full.py), which read macOS's system-wide `Swapouts` counter every 5 seconds
from before the engine started until 63 seconds after it had exited, and would have killed the benchmark and the
engine the moment it moved. It never moved.

| | Morning (`strata-indexed-topk`, old plan) | Evening (`bin/strata`, new plan) |
| --- | ---: | ---: |
| Start: free / file cache / swap in use | 11.70 / 7.13 / 1.05 GiB | 3.70 / 15.99 / 1.93 GiB |
| Plan: context on the GPU, host memory | 5.58 GiB, not counted | 6.02 GiB (2.16 of it for the passes), 0.45 GiB |
| Expert cache | 2,575 experts, 5,038 MiB | 2,521 experts, 4,933 MiB |
| Prompt | 130,944 tokens, 35 passes | 130,944 tokens, 35 passes |
| Prefill | 64.92 tok/s | **63.68 tok/s** |
| First token after | 2,016.89 s | **2,056.18 s** (34 min 16 s) |
| Decode at full depth | 4.59 tok/s | **4.54 tok/s** (76 tokens) |
| Records retrieved | 5/5 | **5/5** |
| Repeat | 130,943 reused, 0.403 s, same ids | **130,943 reused**, 1 read, 0.309 s, 4.82 tok/s, **same ids** |
| New swap-outs, whole run | 3,948 pages (61.7 MiB) | **0** |
| Pressure level | 2 in the one sample taken (at about 94k) | **1 in every sample** |
| Compute buffer at exit | 2,201 MiB against 1,759 expected | no warning |
| Power, SSD bytes | 18.45 W, 1,396.8 GB | 18.13 W, 1,399.7 GB |

The two columns are different desktop states and temperatures, not a controlled comparison: the 2% in speed is
inside what one run shows. What changed is the swap column.

**Swap.** `Swapouts` was 1,586,191 before the start, in every 5-second reading during the 35 minutes, and 3, 13, 33
and 63 seconds after the engine had exited. `vm.swapusage` stayed at 1,975.69 MiB used. The benchmark's own count
over the whole run is 0 too. (Swap-ins rose by 236 pages: macOS reading 3.7 MiB of other programs' memory back.)

**Pressure.** `kern.memorystatus_vm_pressure_level` was 1 (normal) in all 70 half-minute rows of
[the trace](full-128k/full-128k-newplan-watch.jsonl), in the benchmark's 32 samples, and as the peak of the
5-second readings.

**How close it was.** While the prompt was read, free memory was 0.03-0.07 GiB, wired memory 16.6-17.5 GiB, and the
file cache fell from 16.0 to 5.5-7.2 GiB. The other programs' memory in the compressor grew from 8.2 GiB (held in
2.7 GiB) to 13.4-14.9 GiB (held in 5.2-6.3 GiB): as in the afternoon, macOS compressed about 6 GiB of other
programs' memory and kept about 7 GiB of file cache. It did not have to swap. This is one run on one evening's
desktop; with the other programs more active during the half hour the same plan could still swap. The plan still
does not count compressed memory.

**Time.** 2,056.14 s in the engine = GPU wait 1,807.35 + SSD not overlapped 52.12 + the rest 196.68 (GPU busy
1,920.81 s, overlapping). 1,279.0 GB read ahead, 1,233.1 GB of it used. The first pass took 38.5 s, passes 5 to 16
58-66 s, passes 17 to 29 68-75 s, passes 30 to 32 63-66 s (the last few tokens go in three short ones). Depth
reached: 32,768 at 414.0 s, 65,536 at 922.7 s, 98,304 at 1,496.0 s, 126,976 at 1,992.3 s. SoC 37.5 C at the start,
83.9 C at its hottest sample two minutes in, then a median of 65 C; battery sensor 30.8 -> 35.5 C.

**Answer.** 220.17 ms per token = GPU wait 129.52 + SSD 78.14 + CPU 12.51; 101.2 cache misses and 463.0 MB from the
SSD per token; 78.2% cache hits. Record positions 6,560 / 32,736 / 65,457 / 98,177 / 124,353.

**Engine log.** [All 20 lines](full-128k/full-128k-newplan-engine.txt); no `compute buffer size ... does not match
expectation` line at exit, so the 2,209 MiB reserved by `0004` was not exceeded at full depth.

### Installed

Done only because the run above was clean. Evidence: [installed/](installed/).

| Step | Result |
| --- | --- |
| `strata-iq3_s.json` | Re-read first: unchanged since 2026-10-11, no setting of the owner's differs from the candidate's base. With `pending/desktop-128k.json` merged in it is that file exactly: `--max-context 131072 --kv f16 --prefill 4096 --reserve-gib 4`, `STRATA_METAL_INDEXED_QSA=1`, alias `qwen3.8-27b-iq3_s`. Before and after: [strata-iq3_s.before.json](installed/strata-iq3_s.before.json), [strata-iq3_s.after.json](installed/strata-iq3_s.after.json) (the live file is not in git) |
| `~/.pi/agent/models.json` | `contextWindow` of `qwen-local` / `qwen3.8-27b-iq3_s`: 32768 -> 131072. One line; `maxTokens` stays 8192. Backup beside it: `models.json.backup-20261012-165047-32k` |
| `run-iq3_s.sh` | Engine started 16:51:16, `/health` said `loaded` at 16:51:25: context 6.02 GiB on the GPU and 0.45 GiB of host memory, 2,547 experts cached (4.87 GiB), 15.6 GiB resident |
| `/health` | `status ok`, `max_context 131072`, `loaded true` |
| `/v1/models` | `qwen3.8-flash-next-iq3_s` and the alias `qwen3.8-27b-iq3_s`, both `n_ctx 131072` |
| pi 1.0.2 smoke test (the command in OPTIMIZATION.md) | Exit code 0 after 35 s. Request 1: 891 tokens read in 17.1 s, 50 written at 5.64 tok/s: a `read` tool call for the fixture, which pi executed. Request 2: 940 of 973 tokens reused, 24 written at 4.67 tok/s: the answer contains `STRATA_PI_READ_CONFIRMED_7c4a91e2` |
| Swap, from before the server start until after its stop | 0 pages (same 5-second watch; pressure level 1) |

The server was stopped again after the test; nothing was left running. `./run-iq3_s.sh` starts it.

Not tested: a long prompt through the HTTP server or pi (the full-context prompt went through `bench.py`, which
talks to the engine directly, so the server's silence and stall limits have not met 75-second passes); a long pi
session; the 900-second idle unload, which gives the memory back and with it a 34-minute prefix.

To go back to 32,768:

```sh
cd /Users/rafal/Projects/strata-air/Strata
cp bench/results/2026-10-12-m4-air-128k-resume/installed/strata-iq3_s.before.json strata-iq3_s.json
cp ~/.pi/agent/models.json.backup-20261012-165047-32k ~/.pi/agent/models.json
```

and start the server again (if `models.json` was edited since, set that one `contextWindow` back by hand instead).

## 9. Later that evening: the engine's "direct" reads were not direct

Asked to go on, the work went back to why `--elastic` swapped. Holding the cache's idle memory across a change of
shape did not help (32 pages swapped), nor did taking every byte at once so that the engine's memory stayed level
(360 pages); the swap guard, now `metal/tools/swapwatch.py`, stopped both runs within seconds. A `vm_stat 1` trace
beside the second one showed what was going on: the engine's wired memory was steady, and **the system was paging
in about 1 GiB a second for the whole of every long pass**, growing the file cache by 1.6 GiB in four seconds and
compressing other programs' memory down to 0.5 GiB in RAM to make room.

**Cause.** The engine opens the model with `F_NOCACHE` and calls its reads direct. On this Mac such a read goes
past the file cache only if its offset in the file *and* its buffer are both multiples of 4,096 bytes (the length
does not matter). [direct-read-test.txt](aligned-reads/direct-read-test.txt): 256 MiB read in 8 MiB pieces leave
0 MiB in the file cache when both are aligned, and 224 MiB when either is off by 8, 32 or 512 bytes. Of the
model's 144 expert tensors, 6 start on a 4,096-byte boundary, and the places they are read to (GPU buffers, the
read-ahead buffer) are aligned to 32 bytes. So nearly every expert read went through the file cache and most of
it stayed there: about 40 GB per pass of a long prompt, 1.4 TB for a full context, which macOS then has to evict
as fast as it arrives.

**Fix** (`metal/src/expert_io.hpp`). Every read is now block-aligned. Where the destination can be laid out to
match the file (the read-ahead buffer: each tensor now sits as far off a block boundary as it does in its file),
only the partial blocks at the two ends go through a small aligned buffer and the rest is read in place. Reads
into GPU memory go through a 2 MiB aligned buffer per reader thread and are copied (the copy the kernel used to
make from its cache). `STRATA_ALIGNED_READS=0` gives the old reads.

One 8,192-token prompt each, 131,072 context, cache capped at 4,000 MiB, the old reads first
([reads-old](aligned-reads/reads-old.out), [reads-aligned](aligned-reads/reads-aligned.out), per-second traces
beside them, summed by [vmsum.py](aligned-reads/vmsum.py) over the seconds the engine was loaded):

| | Old reads | Aligned reads |
| --- | ---: | ---: |
| System page-ins | 41.5 GiB (511 MiB/s) | **1.6 GiB** (21 MiB/s) |
| Other programs' memory compressed meanwhile | 16.5 GiB | **2.6 GiB** |
| ... and decompressed again | 14.7 GiB | 1.4 GiB |
| Other programs' memory left in RAM | 1.1-3.0 GiB | 2.8-4.2 GiB |
| Prompt read | 101.7 tok/s | 104.4 tok/s |
| Written (24 tokens) | 5.28 tok/s | 5.59 tok/s |
| Read from the SSD | 94.6 GB | 90.4 GB |

**Output is unchanged.** Against `strata-reference`, as in section 5: identical ids and logits for chat, code and
the 7,900-token document on the default path, with indexed attention, and with `--elastic`
([gate2-*.txt](aligned-reads/)); the tiny-model tests pass, among them the protocol suite with
`STRATA_VERIFY_CACHE=1`, which compares every expert a layer computes from with the file.

**The old reads swapped once more, without `--elastic`.** An A B B A of old against aligned reads was stopped by
the guard 35 seconds in: the first leg, old reads on the plain engine, wrote 4 pages (64 KiB) to swap in its first
pass ([watch](aligned-reads/old-reads-swapped-4-pages-watch.txt)). The old reads were not run again, so there is
no interleaved speed comparison beyond the pair above.

**Speed.** The plain engine with aligned reads, the 12,288-token prompt of section 4, twice, each from below 45 C
([aligned-plain.out](aligned-reads/aligned-plain.out)): **95.5 and 101.7 tok/s**, the three passes taking 40, 43,
45 s and 36, 37, 47 s; 6.03 and 5.91 tok/s written (48 tokens, 4.9 and 5.7 GiB of cache); no swap-out. The six
runs of that prompt in the afternoon, with the old reads, gave 63.7 to 79.3 tok/s with a third pass of 70-93 s.
The afternoon was warmer (battery sensor 35 C against 33 C), so this is not a controlled comparison; the likely
link is that a kernel no longer compressing 16 GiB a run leaves the fanless SoC cooler.

### `--elastic` again, with aligned reads

The three evening runs of `--elastic` before the reads were aligned, for the record
([elastic/evening/](elastic/evening/)): with the cache's idle memory held, 32 pages swapped; a clean run of three
long prompts (the trace that showed other programs taking freed memory within three seconds); with all memory
taken at once, 360 pages. In every one the guard killed the engine within five seconds of the first swap-out.

With aligned reads, the same build:

| Run | `--elastic` cache | Swap-outs | |
| --- | ---: | ---: | --- |
| Five prompts of 2,000-2,400 tokens (the test that swapped 41 MiB in the afternoon) | 6.58 GiB | **0** | page-ins 0.8 GiB in 153 s; other programs kept 3.2-5.3 GiB in RAM |
| A B B A against the plain engine, prompts of 64-2,000 tokens, twice each | 7.67, 7.81 GiB | **0** | below |

A B B A, both engines with aligned reads, 131,072 context, 4 GiB reserve, 24 tokens written after each prompt,
order plain, elastic, elastic, plain, each from below 45 C ([elastic-ab2.json](elastic/evening/elastic-ab2.json)):

| | Plain | `--elastic` |
| --- | ---: | ---: |
| Expert cache (two runs) | 4,763 / 5,694 MiB | 7,667 / 7,812 MiB |
| Writing, mean of 12 requests | 6.38 tok/s | **7.71 tok/s** (+21%) |
| Per written token | 157 ms = GPU 82 + SSD 66 + CPU 9; 61 misses | 130 ms = GPU 81 + SSD 41 + CPU 8; 41 misses |
| First token, 64-token prompt | 7.26, 6.61 s | 6.52, 6.62 s |
| First token, 250 tokens | 9.78, 9.46 s | 8.75, 8.70 s |
| First token, 1,000 tokens | 13.35, 13.23 s | 13.11, 12.96 s |
| First token, 2,000 tokens (Long shape) | 20.14, 20.09 s | 24.63, 24.43 s (+4.4 s) |
| Swap-outs | 0, 0 | **0, 0** |

What `--elastic` is now (it changed during the evening; section 5 describes the afternoon's form): a change to or
from the Long shape takes its memory at once (the compute buffers and the read-ahead buffer are written to when
they are made, the read-ahead buffer is wired, the cache is cleared when it is made), and what the long passes do
not need of the cache's memory is held in an idle buffer, so that the engine's memory is the same in the Long
shape as in the Answer shape. The gentler allocation of section 5 (`LLAMA_MOE_CACHE_NO_CLEAR`) is gone: it made
the window in which other programs take the memory longer. Whether these measures matter now that the reads are
aligned was not tested separately.

Three clean runs are not a proof for every evening's desktop. `--elastic` stays off by default.

### Passes of 8,192 tokens

In the Long shape the cache's memory is free, so passes of 8,192 tokens fit there (3.28 GiB on the GPU and 0.92 GiB
of host memory, against 2.21 and 0.46 for 4,096); without `--elastic` they would take 1.5 GiB from the cache for
good. A B B A, `--elastic on`, a 16,384-token prompt, each run from below 45 C
([pass8k-ab.json](aligned-reads/pass8k-ab.json)):

| Passes of | Prompt read | Read from the SSD | Written after it (24 tokens) | Swap-outs |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 78.7, 79.1 tok/s | 197 GB | 5.97, 5.43 tok/s | 0, 0 |
| 8,192 | **87.1, 85.0 tok/s** (+9%) | 110-112 GB | 5.58, 5.26 tok/s | 0, 0 |

Both throttle: the first 8,192 tokens take 70-72 s and the second 113-119 s; with 4,096-token passes the four take
36-39, 40, 55-62 and 69-71 s. A pass of another size is not bit-identical to two of half the size (the kernels'
sums are ordered differently), so `--prefill 8192` was not put through the exact gate; it is the same kind of
change as `--prefill 2048` against 1,024, which this engine has always offered.

### The full-context prompt again: aligned reads, `--elastic`, 8,192-token passes

The 130,944-token prompt of section 8, the same evening, under the guard, from 42 C
([full-128k-night/](full-128k-night/)):

```sh
.venv/bin/python metal/tools/swapwatch.py full-128k-elastic8k --out build-metal/resume-128k -- \
  .venv/bin/python -u metal/tools/bench.py --config strata-iq3_s.json --context 131072 --prefill 130944 \
    --prefill-new 120 --needle --prefill-repeat --cool 42 --json ... --log ... -- --elastic on --prefill 8192 --verbose
```

| | Section 8: plain, 4,096, old reads | Now: `--elastic`, 8,192, aligned reads |
| --- | ---: | ---: |
| Expert cache while answering | 2,521 experts, 4,933 MiB | 3,730 experts, 7,293 MiB |
| Prefill | 63.68 tok/s | **70.14 tok/s** |
| First token after | 2,056.18 s (34 min 16 s) | **1,866.83 s (31 min 7 s)** |
| Decode at full depth (76 tokens) | 4.54 tok/s | **5.27 tok/s** |
| Per written token | 220 ms = GPU 130 + SSD 78 + CPU 13; 101 misses | 190 ms = GPU 135 + SSD 45 + CPU 10; 54 misses |
| Records retrieved | 5/5 | 5/5 |
| Repeat | 130,943 reused, 0.309 s, same ids | 130,943 reused, 0.251 s, 5.25 tok/s, same ids |
| Read from the SSD | 1,399.7 GB | 804.8 GB |
| Time not waiting for the GPU | SSD 52 s + rest 197 s | SSD 22 s + rest 94 s |
| System page-ins during the run | not traced | 15.7 GiB (9 MiB/s) |
| Other programs' memory left in RAM | not traced (free memory 0.03-0.07 GiB) | 2.3-4.3 GiB |
| Swap-outs, whole run | 0 | **0** |
| Pressure level | 1 | 1 |
| Power | 18.13 W | 16.95 W |

The cache came back from the Long shape at its planned size, 3,730 experts read back in 2.8 s; no compute-buffer
warning at exit. Different hours of one evening, not a controlled pair.

**What holds it at 70 tok/s is heat.** The first pass of 8,192 tokens took 72 s (114 tok/s). The second took
102 s, the third to the fourteenth 112-124 s, the last two 130 and 142 s. 1,751 of the 1,867 seconds were spent
waiting for the GPU. A cool Air reads this model at 114 tok/s; the fanless one settles at 58-70.

### State at the end of the night

- `build-metal/bin/strata` is the current source: aligned reads on by default, `--elastic` (off by default) with
  its long passes falling back to half and a quarter of `--prefill` when the memory has no room for the whole
  (the first attempt at the run above started with `--elastic` switched off for that reason and a 1.4 GiB cache;
  it was stopped after one pass).
- The production config is unchanged since section 8 (plain, 4,096-token passes, no `--elastic`); through the
  binary it now reads aligned.
- [pending/desktop-128k-elastic.json](pending/desktop-128k-elastic.json) is the configuration of the run above
  (`--elastic on --prefill 8192 --reserve-gib 4`). It has had one full-context run and nine shorter ones
  with aligned reads without a swap-out. It was not installed: whether to run the newer mode on the daily desktop
  is the owner's call. Its swap back-off has still only run on the tiny model.
- Swap written this evening by the experiments before the reads were aligned: 396 pages (6.2 MiB).
- Tests on this binary (SHA-256 `2f29cc14...`): [tests-night/](tests-night/) - kernel tests 28/28 and 533/533,
  parity (default, indexed, elastic, elastic-long), the shape test and its back-off variant, the protocol suite on
  both attention paths, with `STRATA_VERIFY_CACHE=1`, and with `STRATA_ALIGNED_READS=0`; the real-model exact gate
  with `--elastic` ([gate3-final-elastic.txt](aligned-reads/gate3-final-elastic.txt)). The five patch files were
  applied to a clean checkout of the pin once more and give the build tree byte for byte.
- Not done: `--prefill 8192` against a reference for near-ties; a long session through the HTTP server or pi;
  the n-gram table's rows, which are still read through the file cache (113 KiB per distinct token; the 15.7 GiB
  of page-ins above are probably mostly these, 0.9 GiB a pass).

## 10. Night: the model in the file cache, n-gram rows, and a top-k that was not reproducible

Asked to install the faster configuration and go on. `pending/desktop-128k-elastic.json` was installed as
`strata-iq3_s.json` at 20:35 (the plain one is kept as
[installed/strata-iq3_s.before-elastic.json](installed/strata-iq3_s.before-elastic.json)). Every run below is
under `metal/tools/swapwatch.py`; none wrote to swap (the system's counter stayed at 1,586,587 all night).

### 8 GiB of the Mac's memory was this model, in the file cache

With no engine running, `mincore` on a fresh mapping of the two shards
([resident.c](file-cache/resident.c), [measurements.txt](file-cache/measurements.txt)) found **8.03 GiB of them
in the OS file cache**, of 10.8 GiB of file cache in all:

- 4.03 GiB of shard 1: the **dense weights**. llama.cpp's loader copies them to the GPU out of its mapping of the
  file, and the pages it read stay behind: for as long as macOS keeps them, the dense weights are in memory twice.
- 4.00 GiB of shard 2: **rows of the n-gram table**, which are read through the mapping, 110 KiB per token, and
  stay as well.

The plan counts file cache as free, and it is: but macOS did not give these pages up first. In the full-context
run of section 9 it kept 6.1-7.0 GiB of file cache for half an hour with 0.06 GiB free, and compressed other
programs' memory instead. `msync(MS_INVALIDATE)` on a read-only mapping gives such pages back at once (0.8 ms for
a GiB of file with 74 MiB cached; `madvise(MADV_DONTNEED)`, `MS_DEACTIVATE` and `MS_KILLPAGES` leave them where
they are: [evict.c](file-cache/evict.c)). One call over shard 1 moved 3.8 GiB from "file cache" to "free".

The engine does this itself now (`metal/src/expert_io.hpp`, `ModelFiles::map`): after the load, every page of the
model leaves the file cache; after every long pass and every answer, the n-gram rows do. Because a load from a
file that is not cached is page faults one after another (5.8 s for the 3.9 GiB), the kernel is asked to read the
dense weights ahead of the loader (`madvise(MADV_WILLNEED)`, 1 MiB at a time): 2.1 s.
`STRATA_FILE_CACHE=keep` gives the old behaviour.

### The n-gram rows were read one page fault at a time

llama.cpp asks the OS to read a pass's rows ahead before it gathers them, on Linux and Windows; on macOS that call
was compiled out, so every row was a page fault of the one thread that copies it. [faults.c](file-cache/faults.c),
14,000 pages that were not cached:

| How | Time | Per page |
| --- | ---: | ---: |
| Page faults, one thread (as it was) | 1.36 s | 97 µs |
| `madvise(MADV_WILLNEED)` per page, then the same thread | 0.084 s | 6 µs |
| 16 threads faulting | 0.127 s | 9 µs |

`metal/patches/0006-apple-row-prefetch.patch` turns the read-ahead on for macOS
(`LLAMA_MMAP_NO_ROW_PREFETCH=1`: off).

### Old against new, A B B A

The binary of section 9 against the new one, each run from a file cache without the model in it, three prompts
(`--prefill 500,2000,10000 --prefill-new 24`), the installed configuration at `--reserve-gib 4`
([tidy_ab.sh](file-cache/tidy_ab.sh), [tidy-ab.txt](file-cache/tidy-ab.txt), per-second `vm_stat` beside each,
summed by [memsum.py](file-cache/memsum.py)):

| | Old (1st) | New | New | Old (4th) |
| --- | ---: | ---: | ---: | ---: |
| Ready after | 9.2 s | 5.7 s | 5.6 s | 9.3 s |
| ... of it loading the dense weights | 5.8 s | 2.2 s | 2.1 s | 5.8 s |
| Expert cache | 5,650 MiB | 6,900 MiB | 6,787 MiB | 6,762 MiB |
| 500 tokens: first token | 11.97 s | 10.76 s | 10.62 s | 11.79 s |
| 2,000 tokens: first token | 25.62 s | 24.74 s | 24.64 s | 26.54 s |
| 10,000 tokens: first token | 104.95 s | 110.37 s | 111.20 s | 119.91 s |
| CPU seconds in the three prompts | 2.1 / 2.7 / 9.6 | 1.3 / 1.3 / 4.2 | 1.3 / 1.3 / 4.2 | 2.2 / 2.9 / 10.0 |
| GPU wait, 10,000 tokens | 86.2 s | 98.3 s | 99.3 s | 102.3 s |
| Free memory while loaded, min / median | 0.01 / 0.09 GiB | 1.11 / 1.64 GiB | 0.98 / 1.54 GiB | 0.05 / 0.22 GiB |
| File cache, median | 6.40 GiB | 2.40 GiB | 2.47 GiB | 5.93 GiB |
| Other programs' memory compressed during the run | 4.53 GiB | 0.00 GiB | 0.00 GiB | 2.99 GiB |
| ... left in RAM, median | 4.83 GiB | 5.71 GiB | 5.96 GiB | 3.86 GiB |
| Model in the file cache after the run | 4.48 GiB | 0.20 GiB | 0.20 GiB | 4.47 GiB |
| Swap-outs | 0 | 0 | 0 | 0 |

- The CPU's share of a prompt is less than half of what it was (the rows), which is the whole of the gain in the
  first token of the 500- and 2,000-token prompts (10% and 5%). On the 10,000-token prompt the 5.6 s are there
  too, under a GPU wait that rose from run to run with the heat of four runs back to back.
- **macOS compressed nothing while the new engine ran**, where it compressed 3.0-4.5 GiB of other programs'
  memory in two and a half minutes of the old one, and 1-1.6 GiB stayed free where there had been 0.01-0.09.
- With the old binary the files are not cached only at the first start after a restart of the Mac; the 9.2 s are
  that case. Its usual start (dense weights cached) is 5-7 s, as before.
- The old runs' caches came back from long prompts at full size and the new runs' did not ("less memory is free
  than at the start": 6.06 of 6.75 GiB). The old engine saw the memory macOS had just made by compressing other
  programs as free; the new one sees what is there.

### `--reserve-gib`: 4, 3 or 2

With the model out of the file cache the reserve is real memory, so it was measured again, same three prompts
([reserve_runs.sh](file-cache/reserve_runs.sh)):

| `--reserve-gib` | 4 (the two runs above) | 3 | 2 |
| --- | ---: | ---: | ---: |
| Expert cache at the start | 6.8-6.9 GiB | 8.11 GiB | 8.42 GiB |
| Decode after 500 / 2,000 tokens | 6.4-6.5 / 7.5 tok/s | 6.76 / 8.36 tok/s | 6.89 / 8.23 tok/s |
| First token, 500 tokens | 10.6-10.8 s | 10.30 s | 13.38 s (no memory for the read-ahead buffer) |
| CPU seconds, 10,000 tokens | 4.2 | 4.6 | 9.9 |
| Free + file cache, minimum | 3.3 GiB | 2.2 GiB | 2.2 GiB |
| Compressed during the run | 0.00 GiB | 0.44 GiB | 0.65 GiB |
| Swap-outs | 0 | 0 | 0 |

One run each, on a Mac that was warm from the runs before (the 10,000-token prompts read at 76-78 tok/s).
A reserve of 3 buys 1.3 GiB of cache for 0.44 GiB of compression; a reserve of 2 buys nothing more and starts
to cost (a prompt without read-ahead, rows evicted before they were used).

A second sign was added to `--elastic`'s back-off, which so far only reacted to swap-outs: if macOS compressed
256 MiB or more since the last request and less than half the reserve is free, the cache gives back what is
missing (a quarter of itself at most) and may take it again after the next long prompt. Like the first sign it
has only run on the tiny model (`STRATA_ELASTIC_TEST_BACKOFF=tight`).

### The same long prompt did not give the same logits twice

The 10,000-token prompt's answers differed between runs of the same binary ("a software system", "a software
project"). `metal/tools/determinism.py` (new) starts the engine afresh several times, sends one prompt with
logits reported and compares:

| Prompt | Runs compared | First token's logit | Verdict |
| --- | --- | --- | --- |
| 10,000 tokens (8,192 + 1,805) | three starts of the new binary | 35.715, 35.644, 35.836 | different every time; the third wrote other tokens |
| ... the answer asked again in one engine | same context | | identical |
| 8,195 tokens (one pass of 8,192) | two starts | 37.078, 37.122 | different |
| 4,099 tokens (one pass of 4,096) | two starts | | identical |
| 8,195 tokens, the binary of section 9 | two starts | 37.138, 37.074 | different: not from this evening's changes |

The cause is llama.cpp's Metal top-k for long rows (`kernel_top_k_f32_i32`, used when a row has more than 2,048
candidates: for this model's block-sparse attention that is every context deeper than 8,192 tokens, and every
pass of 8,192). It finds the k-th value by radix selection and then hands out output positions with an atomic
counter, so which of several equal values at the threshold are taken depends on which GPU thread gets there
first. The shorter rows go through a bitonic sort, which is why the 7,900-token document of the gate never
showed it. **Every earlier result at more than 8,192 tokens, the full-context runs included, was one draw of
this**: valid (ties are ties), not repeatable.

`metal/patches/0007-metal-topk-deterministic.patch`: each thread counts its own contiguous slice of the row, one
thread turns the counts into offsets, and the result is written in index order, lowest indices first among ties.
After it ([determinism/after-fix.txt](determinism/after-fix.txt)): two starts with 8,195 tokens, identical IDs
and logits; 10,000 tokens with a 6,000 MiB and a 3,000 MiB expert cache, identical IDs and logits.

### Passes of 8,192 tokens against the reference

Open since section 9. The real-model gate on the final build ([gate-late/](gate-late/), `strata-reference`
re-recorded first, context 32,768, 3,000 MiB cache): default, indexed and `--elastic` paths **identical in IDs
and reported logits** on chat, code and the 7,900-token document; and `--elastic --prefill 8192`, which reads
that document in one pass where the reference took two, **identical too**.

### What the KV cache could give back (measured, not built)

The context reserves 3,072 MiB of KV cache for all 131,072 tokens from the first one. The plan of the installed
configuration at four context sizes ([plans.out](file-cache/plans.out), same minute, `--reserve-gib 3`):

| `--max-context` | Context without the passes' work memory | Expert cache while answering |
| ---: | ---: | ---: |
| 16,384 | 0.58 GiB | 10.57 GiB |
| 32,768 | 1.05 GiB | 10.05 GiB |
| 65,536 | 1.98 GiB | 9.03 GiB |
| 131,072 | 3.86 GiB | 7.09 GiB |

A conversation of 16,000 tokens in a 131,072 context could have a cache half as large again, if the KV cache
took its memory as it filled. It cannot simply be left untouched: [kvlazy.m](file-cache/kvlazy.m) shows that a
Metal buffer is resident as a whole from the first time the GPU uses it (and from `requestResidency` when it is
in a residency set). It would take one Metal buffer per stretch of cells over the same host memory. Not built.

### The final build, and what is installed now

`build-metal/bin/strata`, SHA-256 `42045d8f...`, is the source as committed: seven patch files (applied once more
to a clean checkout of the pin: 26 files, none differs from the build tree).

- **Tests** ([tests-late/](tests-late/)): kernel tests 2/2 and the top-k operator test, the protocol suite (29
  checks) as it is, with `STRATA_FILE_CACHE=keep LLAMA_MMAP_NO_ROW_PREFETCH=1` and with `STRATA_VERIFY_CACHE=1`,
  parity against the reference build on eight paths, the shape test and its two back-off variants.
- **Real-model gate** ([gate-late/](gate-late/)): identical IDs and logits on four paths, as above.
- **A longer prompt at `--reserve-gib 3`** ([final-late/final-r3.out](final-late/final-r3.out)): 24,000 tokens
  with five records hidden in them, 77.4 tok/s (310 s: GPU 293.6, CPU 9.7, SSD 6.8), then 5.58 tok/s written.
  The answer was limited to 64 tokens, which is four records and the start of the fifth; the four are exact.
  In the 340 s the engine was loaded macOS compressed nothing, free memory plus file cache never fell below
  3.25 GiB, no swap-out. The cache started at 7.52 GiB and came back from the long prompt with 6.05.
- **Through the server** ([final-late/server-smoke2.out](final-late/server-smoke2.out)), the way pi reaches it:
  a short chat (12.9 s, "The capital of France is Paris."), a 5,077-token document (66.3 s to 96 tokens: the
  long-prompt shape and back), a second turn on it (11.2 s; 5,072 of 5,088 prompt tokens reused). 6.5-7.0 tok/s
  written, no swap-out.

**Installed** (`strata-iq3_s.json`, a copy in
[installed/strata-iq3_s.after-elastic.json](installed/strata-iq3_s.after-elastic.json)):
`--max-context 131072 --kv f16 --prefill 8192 --reserve-gib 3 --elastic on`, `STRATA_METAL_INDEXED_QSA=1`.
To go back to the plain configuration of section 8, copy
[installed/strata-iq3_s.before-elastic.json](installed/strata-iq3_s.before-elastic.json) over it.

Not done, in order of likely value:

1. **The KV cache taking its memory as it fills** (table above: 7.1 -> 10.6 GiB of expert cache for a
   conversation of up to 16,000 tokens). The largest thing left for the speed of answers.
2. A full-context prompt on the final build. The top-k change makes every result beyond 8,192 tokens a new,
   repeatable one, so the 70.1 tok/s and 5.27 tok/s of section 9 stand as measured but were not repeated;
   the CPU's share (94 s of that run) should now be about half.
3. Both back-off signs have only ever fired on the tiny model.
4. The targets (120 tok/s read, 10 tok/s written at full depth) are not reached. A cool pass of 8,192 tokens
   reads at 114 tok/s; five minutes in, the fanless Air reads at 77.

## 11. Late night: the KV cache takes its memory as the conversation grows

The step that section 10 measured and left unbuilt. With `--elastic`, the KV cache (attention: 3,072 MiB at
131,072 context; the block-sparse attention's index: 768 MiB) is now mapped on the GPU 16,384 cells at a time
(0.47 GiB), as the conversation reaches them, and what it has not taken is the expert cache's. When the
conversation is about to pass the cells that are paid for, the cache gives the memory for the next ones first.

### How

- A Metal buffer is resident as a whole, so the parts are separate Metal buffers over the same host memory
  (`ggml_backend_buffer_set_views`; `metal/patches/0008-kv-cache-mapped-as-it-fills.patch`). A tensor is found by
  where it starts; the graph only ever reads the cache up to the cells in use (padded to 256), and
  `llama_kv_cache` maps more by itself if a slot beyond the mapped cells is taken, so the engine's bookkeeping
  decides who pays, not whether the result is right.
- The buffer is allocated without any Metal buffer (`ggml_backend_alloc_unmapped`) and gets its first part
  directly. Making it whole first and cutting it down does not work: Metal lets go of a released buffer's memory
  only after the GPU has done more work, 0.5-1.5 s later ([kvrel.m, kvviews2.m](kv-cache/),
  [metal-buffer-release.txt](kv-cache/metal-buffer-release.txt)), and for that long the whole 3.8 GiB would be
  wired beside a cache that had already been given them. A first version did exactly that (in a `--load-only`
  run; no swap-out), which is how it was found.
- The engine (`make_room_for_kv` in `metal/src/main.cpp`): at the first pass of a request, for the whole prompt.
  Before a long prompt the cache is about to be given up anyway: the buffer that holds its place is made that
  much smaller from the start, and the cache comes back smaller (0.03 s). Otherwise the cache is made smaller and
  read back (4 s, once per 16,384 tokens of a conversation that grows in short turns or in the middle of an
  answer).
- The KV cache never gives memory back while the engine runs (the server's idle unload starts it afresh).
  `STRATA_KV_STEP=N`: another step; `0`: all at the start, as before. Without `--elastic` nothing changes.

### The plan

[plan-kv2.out](kv-cache/plan-kv2.out), the installed configuration, same minute:

| | KV cache all at the start | as it fills |
| --- | ---: | ---: |
| Expert cache while answering, conversation up to 16,128 tokens | 6.72 GiB | **9.95 GiB** |
| ... at 131,072 tokens | 6.72 GiB | 6.67 GiB |
| Long passes | 8,192 tokens | 8,192 tokens |

### Same output

`metal/tools/determinism.py` on the real model, each run a fresh start ([det4.out](kv-cache/det4.out),
[det5.out](kv-cache/det5.out)); the reference is the KV cache mapped all at once:

| Prompt | KV step | What happened | Verdict |
| --- | ---: | --- | --- |
| 10,000 tokens (8,192 + 1,805) | 2,048 | the place-holder gave 0.29 GiB during the long prompt, cache 10.42 -> 10.13 GiB | identical IDs and logits |
| 10,000 tokens | 16,384 | nothing to give yet | identical |
| 900 tokens (one short pass) | 1,024 | the cache gave memory before the pass: 9.37 -> 9.34 GiB, read back in 3.8 s | identical |
| 700 tokens, 120 written | 1,024 | the cache gave memory at token 769 of the answer: 10.61 -> 10.58 GiB, 4.5 s | identical |

Tiny model, with `STRATA_KV_STEP=256` so that every request grows the cache several times: the shape test and its
two back-off variants, parity on eight paths, the protocol suite ([kv-cache/](kv-cache/)).

### Speed, A B B A

The installed configuration (`--reserve-gib 3`), `--prefill 500,2000 --prefill-new 64`, A = `STRATA_KV_STEP=0`
([kv-ab.out](kv-cache/kv-ab.out)):

| | A | B | B | A |
| --- | ---: | ---: | ---: | ---: |
| Expert cache | 7,189 MiB | 11,123 MiB | 11,061 MiB | 7,703 MiB |
| Written after 500 tokens | 7.04 tok/s | **8.55** | **8.48** | 7.13 |
| Written after 2,000 tokens | 7.34 tok/s | **8.71** | **8.66** | 7.56 |
| First token, 500 tokens | 10.50 s | 9.58 s | 9.59 s | 10.34 s |
| ... read from the SSD for it | 49.4 GB | 37.3 GB | 37.4 GB | 47.7 GB |
| First token, 2,000 tokens | 24.72 s | 26.75 s | 26.66 s | 25.04 s |

Answers are written **20% and 17% faster**, and a prompt that is read beside the cache starts 8% sooner. A
prompt of more than 1,024 new tokens starts 1.8 s later: the cache is given up for it and read back afterwards,
and a larger cache takes longer to read back (which about 95 written tokens make up for). In the 209 s the
engines were loaded macOS compressed 0.02 GiB and wrote nothing to swap.

### A long prompt past the first step

`--prefill 2000,24000 --prefill-new 96 --needle`, the installed configuration, per-second `vm_stat` beside it:

| | Before the KV change ([final-r3](final-late/final-r3.out)) | First version ([final-kv](kv-cache/final-kv.out)) | As committed ([final-kv2](kv-cache/final-kv2.out)) |
| --- | ---: | ---: | ---: |
| Expert cache at the start | 7.52 GiB | 10.21 GiB | 10.87 GiB |
| 24,000 tokens read | 77.4 tok/s (310 s) | 86.7 tok/s (277 s) | 85.4 tok/s (281 s) |
| Written at 2,000 / 24,000 tokens | 7.04 / 5.58 tok/s | 7.79 / 7.46 tok/s | 7.91 / 7.52 tok/s |
| Records found | (answer cut at 64 tokens) | 5/5, 5/5 | 5/5, 5/5 |
| The KV cache's second step | | 0.46 s | 0.03 s |
| Cache after the long prompt | 6.05 GiB | 9.75 GiB | 10.14 GiB |
| Compressed while loaded | 0.00 GiB | 1.38 GiB, all in one second | 0.32 GiB |
| Free + file cache, minimum | 3.25 GiB | 2.73 GiB | 3.71 GiB |
| Swap-outs | 0 | 0 | 0 |

The first version freed the buffer that holds the cache's place and made a smaller one when the KV cache took its
step. Metal returns freed memory late, both were there for a moment, free memory fell to 0.25 GiB and macOS
compressed 1.38 GiB of other programs' memory in that one second (23:03:44 in the trace). The committed version
makes that buffer the right size once. The read speeds are not a comparison (the three runs started at different
temperatures); the written speeds at 24,000 tokens are the larger cache.

### The final build

`build-metal/bin/strata`, SHA-256 `f3f16987...`: eight patch files (applied to a clean checkout of the pin: 32
files, none differs from the build tree).

- **Tests** ([tests-final/](tests-final/)): kernel tests 2/2 and the top-k operator test; the protocol suite,
  parity on eight paths, the shape test and both back-off variants, each as it is and with `STRATA_KV_STEP=256`;
  the protocol suite with `STRATA_VERIFY_CACHE=1`.
- **Real-model gate** ([kv-cache/gate/](kv-cache/gate/)): identical IDs and reported logits against
  `strata-reference` on the default, indexed and `--elastic` paths and with `--elastic --prefill 8192` (the two
  `--elastic` ones with the KV cache mapped as it fills).
- **Through the server** ([kv-cache/server-smoke3.out](kv-cache/server-smoke3.out)): the three requests of
  section 10 again: 11.5 s, 66.8 s, 9.4 s; 7.7-8.2 tok/s written where they were 6.5-7.0; expert cache hits
  88-92% where they were 84-89%; no swap-out.

Installed: unchanged file (`--max-context 131072 --kv f16 --prefill 8192 --reserve-gib 3 --elastic on`); the
engine it points at now does the above. `STRATA_KV_STEP=0` in the config's `env` turns the KV change off.

No run of the night wrote to swap: the system's counter is at 1,586,587, where it was at 20:35.

Not done, in order of likely value:

1. After a long prompt the whole cache is read back before the first token (3.6 s for 11 GiB): read back the
   most used part and let the rest come as it is asked for.
2. Passing a KV step outside a long prompt reads the whole cache back (4 s) where 0.47 GiB had to go.
3. A full-context prompt on this build. The last one (section 9: 70.1 tok/s read, 5.27 written) predates the
   top-k change, the row read-ahead and the KV change; at full depth the cache is the size it was then.
4. Both back-off signs have only fired on the tiny model.
5. The targets (120 tok/s read, 10 tok/s written at full depth) are not reached. Written: 8.5-8.7 tok/s up to
   16,000 tokens, 7.5 at 24,000. Read: 114 tok/s for a cool first pass, 77-87 once the Air is warm.

## 12. Morning of 2026-10-13: the expert cache is not read back

Points 1 and 2 of the list above. With `--elastic` the expert cache is given up for a long prompt and comes back
for the answer; it was then read back whole from the SSD (what it had held, most used first) before the first
token: 4.0 s for 10.5 GiB. It also was whenever the cache was made smaller: for a step of the KV cache outside a
long prompt, and for a back-off.

Now it comes back empty and fills as its experts are asked for. `--elastic-refill G` reads the most used G GiB
back first, `--elastic-refill all` everything, as before (`STRATA_ELASTIC_REFILL` in the environment does the same
where the option is not given). The default is 0, because nothing else measured better. Everything below is the
real model on the desktop as it was this morning (browser, VM, VS Code, the Claude app), the SoC cooled to 45 C
before every engine start, every run under `metal/tools/swapwatch.py`.

### How much to read back: nothing

The same 6,992 MiB cache in all eight engine starts (`--expert-cache-cap 7000`), a 2,000-token prompt, in the order
0, 1, 2, all, all, 2, 1, 0 ([refill-x2.out](refill/refill-x2.out), [refill-x2.json](refill/refill-x2.json)):

| Read back | First token | 50th token | The 49 tokens after the first |
| --- | ---: | ---: | ---: |
| nothing | 22.40, 23.08 s | 29.82, 30.56 s | 7.42, 7.48 s |
| 1 GiB | 22.54, 23.42 s | 29.80, 30.79 s | 7.26, 7.37 s |
| 2 GiB | 22.84, 22.92 s | 30.05, 30.12 s | 7.21, 7.20 s |
| all (6.83 GiB) | 24.26, 24.23 s | 31.32, 31.30 s | 7.06, 7.07 s |

Reading all of it back makes the next 49 tokens 0.4 s faster and the first one 1.5 s later. Reading the most used
1 or 2 GiB back gives 0.13 and 0.24 s over those tokens, for reads of 0.32 and 0.65 s at the SSD's 3.08 GiB/s. Two
starts with the same setting differ by up to 0.9 s at the first token (heat: the first and the last of eight),
which is more than the settings 0, 1 and 2 differ, there and at the 50th token: no amount came out ahead of none.
An earlier run without the cap ([refill-x1.out](refill/refill-x1.out): 0, 2, 5 GiB and all, 200 tokens written,
caches of 8.0-10.1 GiB) had the first tokens in that order (22.7-23.1, 23.2-23.6, 24.3 and 24.6-25.5 s) and the
200th of 0, 2 and 5 GiB within a second of each other (50.3-51.3 s), with "all" at 53.6 s.

Why it helps so little, from a conversation of three turns ([turns.py](refill/turns.py),
[turns1.out](refill/turns1.out); `--expert-cache-cap 7800`; the three starts that got the same 7,792 MiB):

| | Read back (one run) | Not read back (two runs) |
| --- | ---: | ---: |
| Turn 1, a 2,000-token document: first token | 24.55 s | 22.48, 22.48 s |
| ... 50th and 100th token | 31.35, 39.00 s | 29.68, 37.31 and 29.68, 37.33 s |
| ... experts missed by the prompt's last passes and by the 100 tokens | 260 + 5,471 | 594 + 5,720 and 596 + 5,719 |
| Turn 2, 525 more tokens (one pass beside the cache): first token | 11.77 s | 11.69, 11.71 s |
| ... 50th token; experts missed by the answer | 18.77 s; 6,080 | 18.62 s; 6,080 and 18.65 s; 6,079 |
| Turn 3, a 20-token question: first token | 2.36 s | 2.33, 2.34 s |
| ... 50th token; experts missed by the answer | 9.43 s; 6,426 | 9.29 s; 6,429 and 9.32 s; 6,428 |

- 3,986 experts (7.6 GiB) were read back (the context took 2.92 s to get its answering shape; 0.44 s without, at
  6.99 GiB), and it saved 583 reads in the passes that followed. A pass that finds the cache empty reads the
  experts it needs and no others.
- What is read back and not used does not stay: 100 tokens miss 5,500-5,700 times in a cache of 3,986 experts, so
  the cache is replaced once within 70 tokens. From the second turn on it holds the same experts either way (the
  same misses, the same bytes from the SSD: 63.0 and 35.4 GB).
- The written ids are the same in all four runs, in every turn. (The fourth run, read back, came first and got a
  7,199 MiB cache, below the cap: 24.28 s to the first token, 31.51 s to the 50th.)

### A B B A at the installed configuration

`trace_run.sh refill-ab --prefill 2000,10000 --prefill-new 64 --cool 45 --ab "STRATA_ELASTIC_REFILL=all
build-metal/bin/strata" "build-metal/bin/strata" --runs 2 -- --verbose`: A reads the cache back whole, B is the
default now; no cache cap ([refill-ab.out](refill/refill-ab.out), [refill-ab.json](refill/refill-ab.json)).

| 2,000-token prompt | A | B | B | A |
| --- | ---: | ---: | ---: | ---: |
| Expert cache | 11,305 MiB | 10,630 MiB | 10,966 MiB | 10,745 MiB |
| First token | 25.79 s | **22.52 s** | **22.59 s** | 25.52 s |
| 10th token | 26.83 s | 24.02 s | 24.06 s | 26.57 s |
| 50th token | 31.66 s | **29.40 s** | **29.36 s** | 31.48 s |
| 64th token | 33.23 s | 31.02 s | 30.94 s | 33.07 s |
| Written, the 63 tokens after the first | 8.47 tok/s | 7.41 tok/s | 7.54 tok/s | 8.34 tok/s |
| Experts missed per written token | 29.5 | 40.6 | 40.1 | 31.1 |

| 10,000-token prompt | A | B | B | A |
| --- | ---: | ---: | ---: | ---: |
| First token | 106.08 s | **104.33 s** | **104.31 s** | 106.43 s |
| 50th token | 114.14 s | **113.11 s** | **113.18 s** | 114.63 s |
| 64th token | 116.64 s | 115.56 s | 115.69 s | 117.17 s |
| Written, the 63 tokens after the first | 5.97 tok/s | 5.61 tok/s | 5.54 tok/s | 5.87 tok/s |

- After 2,000 tokens the first token comes **3.1 s sooner** (12%), the 50th and the 64th 2.2 s sooner. After
  10,000 tokens: 1.9 s and 1.2-1.3 s.
- The rate over the first 64 tokens is lower with B, 7.4-7.5 against 8.3-8.5 tok/s: the experts those tokens need
  are read while they are written (0.9 s more for the 63 tokens; 10 more misses a token, about 630 experts),
  where A had read them before the first, among the 5,499 it read back. No token of the answer comes later than
  with A. Once the cache is full again the rate is what it was: in the three-turn run above the second and third
  turn wrote at the same speed either way (6.60 and 6.59 tok/s read back, 6.68-6.69 and 6.61-6.65 not).
- After the 10,000-token prompt both wrote at 5.5-6.0 tok/s: the SoC was at 88 C after 100 s of reading.
- A's last engine took 4.01 s to give the context its answering shape after the first prompt (5,499 experts read
  back) and 3.69 s after the second (the cache came back with 9.63 GiB of its 10.50). Without the read-back that
  change took 0.44 s for a 6.99 GiB cache.
- No swap-out. What macOS compressed while each engine was loaded ([memory-per-engine.txt](refill/memory-per-engine.txt),
  from the per-second `vm_stat` traces): 0.85, 0.00, 0.11, 0.01 GiB in the order A B B A
  (`memsum.py`: 0.97 GiB in 578 s; [memsum.txt](refill/memsum.txt)). Nearly all of it is the first engine start
  of the series, during its 10,000-token prompt, where A and B do the same. refill-x1 had it the same way
  (1.52, 1.27 and 0.70 GiB in its first three starts, 0.27 GiB in the other five together); the capped run
  compressed nothing in eight starts.

### A step of the KV cache outside a long prompt, and a back-off

The cache is made anew, smaller, when the KV cache takes its next step while the cache is there. It is no longer
read back then either. [kvstep.sh](refill/kvstep.sh), as section 11's det5.sh: `STRATA_KV_STEP=1024`, so that a
short conversation crosses a step; no cache cap, caches of 9.8-11.1 GiB, every run a fresh start
([kvstep.out](refill/kvstep.out)):

| | Read back (as before) | Not read back |
| --- | ---: | ---: |
| 900 tokens in one pass, the cache gives memory before it: the step | 3.67 s (5,051 experts) | 0.62 s |
| ... first token; 50th token | 15.81 s; 22.11 s | 14.35 s; 21.30 s |
| 700 tokens and 120 written, the step before the 70th token: the step | 3.79 s (5,074 experts) | 0.93 s |
| ... longest wait between two tokens | 4.01 s | 1.37 s |
| ... 120th token | 30.77 s (9.95 GiB cache) | 30.24 s (11.05 GiB cache) |

Identical IDs and logits between the two, in both cases (`metal/tools/determinism.py`). The pass of 900 tokens
finds no experts in the cache to copy and reads them from the SSD, which is why its first token gains 1.5 s and
not 3. With a capped cache the step is paid from what the cap left unused and the cache is not touched at all
([kvstep-capped.out](refill/kvstep-capped.out): "6.84 -> 6.84 GiB, 0.00 s").

The back-off (the cache gives memory back when macOS swapped or ran short) goes the same way. Its two signs have
still only fired on the tiny model, where they are simulated.

### Same output

- **Tiny model**, 22 runs ([refill/tests/](refill/tests/), [run_tests.sh](refill/tests/run_tests.sh)): the shape
  test, also with `STRATA_KV_STEP=256`, with each back-off simulation, and with both; parity against
  `strata-reference` on eight paths, also with `STRATA_KV_STEP=256`; the protocol suite (29 checks), also with
  `STRATA_KV_STEP=256` and with `STRATA_VERIFY_CACHE=1`. Each as it is and with `STRATA_ELASTIC_REFILL=all`. The
  shape test has two more variants, the cache read back whole and in part (4 and 2 of the tiny model's 4 experts
  after every long prompt but the first), and fails if the part is not smaller than the whole.
- **Real-model exact gate** ([refill/gate/](refill/gate/)): identical IDs and reported logits against
  `strata-reference` on the default, indexed and `--elastic` paths and with `--elastic --prefill 8192`; no
  swap-out.
- **Real model, one setting against the other**: the two KV-step cases above (IDs and logits), the three-turn
  conversation (IDs).

`build-metal/bin/strata`, SHA-256 `b646b475...`. The patched llama.cpp tree was not touched: the eight patch files
are as they were.

### Tools

`metal/tools/bench.py` records when every token arrived (`token_s` in the JSON) and prints the 2nd, 10th and 50th;
`metal/tools/determinism.py` stores the same. In [refill/](refill/): `turns.py` (the three-turn conversation,
A B B A), `summ.py` (token times per engine start of an `--ab` run), `compr.py` and `per_engine.py` (the seconds
in which macOS compressed, and the sum per engine start, from a `vm_stat` trace), `trace_any.sh` (any command under
the swap watch with a trace beside it; `trace_run.sh` for bench.py is in `build-metal/resume-128k/`).

### Not done, and seen on the way

1. **The 0.4-0.9 s that are left.** The experts the first tokens after a long prompt ask for (0.4 s over 49
   tokens with a 7 GiB cache, 0.9 s over 63 with 11 GiB) could be read behind the tokens instead of in them, with
   the cache's read-ahead tickets (fills that are not dropped when their layer routes elsewhere). That is a change
   to the cache in patch 0001. Not built; those figures are all it could gain per long prompt.
2. **The engine's start fills the cache from the profile the same way** (3.5 s for 11 GiB; ready in 6.3-6.8 s
   with a 10-11 GiB cache, 5.2 s with 7 GiB). Whether a start without it answers its first request sooner was not
   measured (`--no-preload` exists). The server starts the engine afresh after 900 s without a request.
3. **The short-prompt shape takes memory that is not free.** It borrows 2.1 GiB for its one pass if that much is
   "free or file cache"; in the three-turn run 0.7-1.1 GiB were free and macOS compressed 0.97, 0.17 and 0.72 GiB
   of other programs' memory in that second (nothing in the fourth run). Both settings.
4. **A KV step in the middle of an answer frees the cache and makes it anew in one go**: 0.31 and 0.42 GiB
   compressed in that second ([kvstep-vmstat.txt](refill/kvstep-vmstat.txt), 07:09:32 and 07:10:10), with and
   without the read-back. It could be paid from memory kept aside for it, as it is with a capped cache.
5. Points 3-5 of section 11's list stand: a full-context prompt on this build, the back-off on the real model,
   the targets.

No run of the morning wrote to swap: the system's counter is at 1,586,587, where it was last night.
