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
