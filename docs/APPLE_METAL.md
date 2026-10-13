# Apple silicon (Metal): build from source on a Mac

Strata runs on Apple silicon Macs through a separate engine in `metal/`. **Status: experimental.** Everything here was
built and measured on one machine: a MacBook Air M4 (10-core GPU, 32 GB, fanless, macOS 15.7.9) with the IQ3_S model, on
2026-10-11. Nothing else has been run: no other Mac, no other size, no images. Back to the [README](../README.md).

The CUDA / HIP engine (`src/`) cannot be compiled for a Mac: its kernels are CUDA C++. This engine takes the model graph
and the Metal kernels from [llama.cpp](https://github.com/ggml-org/llama.cpp) (the `qwen4exp` architecture, `ggml-metal`,
at a pinned commit with a patch) and puts Strata around them: the memory plan, the expert cache's start-up fill from
Strata's expert profile, a read-only SSD tier that reads ahead of the GPU, and the `--serve` protocol. The server, the
web app and the APIs (`serve/`) are the same ones the other backends use.

## What a Mac changes

A PC has VRAM, RAM and an SSD. A Mac has one pool of memory that the CPU and the GPU share, and Metal lets one program
keep about two thirds of it resident on the GPU (21.3 of 32 GiB on the Air). So the tiers are:

- **GPU (wired shared memory):** the dense weights (3.9 GiB), the KV cache and work buffers (1.4 GiB at an 8K context,
  2.2 GiB at 32K), and an **expert cache** that takes the rest: at most what Metal allows, and never more than the
  memory that is free when the engine starts.
- **SSD, read-only:** every expert the cache does not hold. When a token routes to one, it is read from the GGUF file
  straight into its cache slot. On a Mac that slot is shared memory, so the buffer the SSD read lands in is the one the
  GPU computes from: no staging copy and no upload.
- **The 26.8 GiB n-gram table** stays on the SSD, a few rows read per token (llama.cpp's lazy rows).

IQ3_S has 46.8 GiB of routed experts (48 layers x 512, 1.95 MiB each on average). With every other program closed the
plan had room for a 13.4 GiB cache on the day of the measurements (29% of them; 2.2 GiB were still in swap from earlier
and count as owed, so the free memory was the limit and not Metal's 15.8 GiB); the rest is read from the SSD while the
model answers.

### The SSD works while the GPU works

The model is 48 layers, and each layer's ten experts are only known once the layer before has been computed. An
engine that waits for that has the GPU idle while the SSD reads and the SSD idle while the GPU computes. This one
reads ahead in both of its paths:

- **Writing an answer (one token per pass):** when a layer has routed, the engine applies the *next* layer's router to
  this layer's input and reads the experts it names while the GPU computes on (`metal/src/router_predict.hpp`). The
  two inputs are close because the layers add to one residual stream. A wrong guess costs a read and never changes
  what is computed: the layer still routes for itself, a guessed expert that it did not pick is the next to leave the
  cache, and a read that has not started by then is dropped. See "Measured" for what the guess is worth.
- **Reading a long prompt (passes of 1,024 tokens):** such a pass uses most of a layer's 512 experts, and llama.cpp
  asks for them one tensor at a time. The engine reads the next tensors into a staging buffer while the GPU computes
  with the current one (`metal/src/expert_io.hpp`): the same layer's other two tensors need the same experts, and the
  next layer's are guessed from what the last pass used there (before there was one: from Strata's expert profile).

### Nothing is written

The model files are opened read-only and the engine writes no pack, no cache file and no copy of the experts. Three
things that would otherwise put model data into swap are switched off or checked:

- **The cache is sized from free memory, not from the RAM size.** The engine counts what is free or file cache, takes
  off what macOS already holds in swap (memory other programs will want back) and 3 GiB more (`--reserve-gib`), and
  gives the expert cache what is left. A Mac with a browser, an editor and a VM open gave it 3.0 to 9.4 GiB over one
  afternoon; with them closed, 13.4. The log says how much closing programs would add. Wired memory is never swapped,
  so a cache that is too large pushes *other programs* into swap; that is what this rule avoids. (A cache size given as
  a number, `--expert-cache 7000mib`, is kept as asked: one benchmark run that did so while other programs had grown
  made macOS swap 1,772 pages. `--expert-cache-cap` keeps the rule and only limits what it may take.)
- **No repacked copies.** ggml's ARM kernels would copy 17.5 GiB of this model's experts (the Q2_0, Q4_K and IQ4_NL
  ones) into a repacked buffer. That is anonymous memory and went straight into swap on the 32 GB Air (18 GB of
  swap-outs in one start, before this was found). The engine loads with `use_extra_bufts = false`, and stops if its
  own memory after loading is more than the dense weights.
- **The GPU buffers stay wired while the engine runs.** ggml stops its residency requests after 3 idle minutes, after
  which macOS could compress or swap the expert cache. The engine keeps them going; to give the memory back, the model
  is unloaded instead (`"idle_unload_s"` in the config, 900 by default from `setup-mac.sh`; the next request loads it
  again in a few seconds).

Missed experts are read past the OS file cache (`--expert-io direct`, F_NOCACHE), so reading 40 GB during an answer does
not fill the memory with file pages. **Until 2026-10-12 that was only the intent:** such a read goes past the cache
only if its offset in the file and its buffer are both multiples of 4,096 bytes, and the experts' were not, so most
of what was read went through the file cache and stayed there (41.5 GiB of page-ins and 16.5 GiB of other
programs' memory compressed while one 8,192-token prompt was read; 1.6 and 2.6 GiB since the reads are aligned).
Every figure in this document from before that date was measured with the unaligned reads; see
[the report](../bench/results/2026-10-12-m4-air-128k-resume/README.md), section 9. `STRATA_ALIGNED_READS=0` reads
as before. `--expert-io cached` reads through the cache instead: experts that were read recently then
come from RAM, but with little free memory it made macOS swap (17,700 pages during three 120-token answers), so it is not the
default. The staging buffer for long prompts (1.1 GiB, or 0.45 GiB if that is all there is room for) is taken only for
the time of the prompt and only if that much memory is free beyond 1.5 GiB.

What llama.cpp reads through its mapping of the model files stays in the OS file cache: the dense weights after they
were copied to the GPU, and every row of the n-gram table that was looked up. On the M4 Air that was 8 GiB after an
evening of prompts, and macOS compressed other programs' memory before it let go of it. Since 2026-10-12 (night) the
engine gives those pages back itself (`msync(MS_INVALIDATE)` on a read-only mapping): all of the model after the
load, the table's rows after every long pass and every answer. The kernel is asked to read the dense weights ahead
of the loader, and a pass's n-gram rows ahead of the thread that gathers them (`madvise(MADV_WILLNEED)`; llama.cpp
had that call for Linux and Windows only, so on macOS every row was a page fault: 97 µs each against 6).
Measured, old against new, A B B A, three prompts: 3.0-4.5 GiB of other programs' memory compressed per run before,
none after; 1-1.6 GiB free where there were 0.01-0.09; the CPU's part of a 10,000-token prompt 4.2 s instead of
9.6-10.0; a start with the files not cached 5.6 s instead of 9.2
([report](../bench/results/2026-10-12-m4-air-128k-resume/README.md), section 10). `STRATA_FILE_CACHE=keep` and
`LLAMA_MMAP_NO_ROW_PREFETCH=1` give the old behaviour.

**Long prompts are reproducible since the same night, and were not before.** llama.cpp's Metal top-k for rows of
more than 2,048 candidates (for this model: every context deeper than 8,192 tokens) wrote its result through an
atomic counter, so which of several equal scores at the threshold were taken depended on the GPU's thread timing:
the same 10,000-token prompt gave a first logit of 35.72, 35.64 and 35.84 in three starts. With
`metal/patches/0007-metal-topk-deterministic.patch` it writes in index order and two starts give identical IDs and
logits. `metal/tools/determinism.py` is the check.

## What you need

| | |
| --- | --- |
| Mac | Apple silicon, 32 GB of memory or more. Measured on an M4 MacBook Air. |
| Disk | 84 GB for IQ3_S (two files: 54.8 GB of weights, 28.8 GB n-gram table). The internal SSD matters: missed experts are read from it for every token. |
| Tools | The Xcode command line tools (`xcode-select --install`), CMake 3.24+, git, Python 3.10+. Xcode itself is not needed: the Metal shaders are compiled when the engine starts. |

## Install

```sh
./setup-mac.sh                 # compiles the engine, downloads IQ3_S, writes run-iq3_s.sh and starts it
./setup-mac.sh --check         # only looks at this Mac
./setup-mac.sh --model IQ2_XS --context 8192 --no-start    # the other sizes install the same way; only IQ3_S was run
```

It compiles the engine into `build-metal/`, downloads the two shards into `Strata-data/models/<SIZE>/` next to the Strata
folder (curl, resumable, each file checked against the SHA-256 the Hub publishes), exports the tokenizer, and writes
`strata-<size>.json` and `run-<size>.sh`. `./run-iq3_s.sh` starts the server on `http://127.0.0.1:8080`.

**Close other programs before you start it.** The expert cache is sized once, at the start, from the memory that is free
then; the engine's log (`strata-iq3_s.log`) says what it got and what closing programs would add.

By hand:

```sh
cmake -S metal -B build-metal -DCMAKE_BUILD_TYPE=Release      # or -DSTRATA_ENABLE_METAL=ON on the top-level project
cmake --build build-metal -j
build-metal/bin/strata --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --plan
```

CMake takes llama.cpp at the pinned commit from a `llama.cpp` checkout next to the Strata folder (or
`-DSTRATA_LLAMA_DIR=...`), else from GitHub, into `build-metal/llama.cpp`, and applies `metal/patches/` there. The
checkout it comes from is not edited.

## Measured

MacBook Air M4, 32 GB, IQ3_S, context 8,192, greedy, 2026-10-11. **Before** is the engine as it was first written
(one token per pass, the SSD read after each layer has routed); **now** is the defaults. Every pair was run on the same
day, interleaved (before, now, now, before), with the same expert cache, each run started when the SoC had cooled to
42 C (`metal/tools/bench.py --ab ... --cool 42`); the figures are medians. Two states of the Mac:

- **as found:** a browser, an editor, a VM and the Claude app open. What the cache got moved between 3.0 and 9.4 GiB
  over the afternoon, so the runs cap it at one size for both engines (6.3 GiB = 3,320 experts, 13.5%).
- **closed:** only the Claude app open, 13.4 GiB of room, cache capped at 12.2 GiB (6,395 experts, 26%). The prefill
  rows are one run each. For the 32,000-token prompt the context is 40,960 and the caches differ: 11.2 GiB before
  (given as a number), 10.4 GiB now (what the plan had room for).

| | as found: before | now | closed: before | now |
| --- | ---: | ---: | ---: | ---: |
| Writes an answer, cool (2 requests of 120 tokens) | 5.50 tok/s | **6.60** (+20%) | 7.19 tok/s | **8.46** (+18%) |
| Writes for 5 minutes (7 requests of 200 tokens in turn) | 5.29 tok/s | **5.73** (+8%) | 6.88 tok/s | **7.32** (+6%) |
| ... its requests after the first 200 s | 4.81-4.96 | 4.93-5.31 | 6.31-6.54 | 6.23-6.76 |
| Reads a prompt of 32 tokens | 5.3 tok/s | 5.6 | 7.7 tok/s | 6.8 (-12%, one run each) |
| ... of 256 tokens | 22.9 tok/s | 26.3 (+15%) | 24.8 tok/s | 33.1 (+33%) |
| ... of 1,600 tokens | 48.0 tok/s | **64.4** (+34%) | 48.7 tok/s | **66.7** (+37%) |
| ... of 8,000 tokens (as found: a 5.0 GiB cache) | 47.9 tok/s | **62.0** (+29%) | 47.1 tok/s | **65.5** (+39%) |
| ... of 32,000 tokens (context 40,960) | not run | not run | 41.7 tok/s (768 s) | 54.5 (587 s, +31%) |
| Expert cache hits while writing (experts read ahead count as hits) | 71.5% | 85.6% | 85.4% | 92.2% |

- **The output is the same.** `bench.py --gate check`: nine requests (a chat, code, a calculation, Polish, Japanese,
  four turns of a chat, a code review, a 1,600-token and a 7,900-token document), 200 tokens each, greedy: all 1,800
  token ids equal those of the engine before. (The reference fixes `--chunk 7`: the engine before already gave other
  ids for one of the nine when the cache, and with it the pass size it reads a short prompt in, was a different size.)
  With `STRATA_VERIFY_CACHE=1` every expert a layer computed from was compared with the model file for 60 tokens of
  two requests: no difference.
- **Swap:** 0 pages swapped out in every run of the engine as it is now, in both states. The engine before swapped 4
  to 964 pages in seven of its runs, 1,876 in all, nearly all of them while reading long prompts (it read 512 bytes per
  run of experts through the mapped file; see below).
- **A 32-token prompt was read slower in the closed state** (6.8 against 7.7 tok/s, one run each; as found it was
  5.6 against 5.3, two runs each). It was seen in the last measurement and not followed up. Such a prompt is read in
  passes of a few tokens through the cache, and each token of a pass names experts to read ahead.
- **Start:** ready in 3.5-12 s; the cache fill reads 5-12 GiB at 3.1-3.5 GiB/s.

### Where a token's time goes, and how close that is to the hardware

One written token, as found (6.3 GiB cache, cool), from the engine's own counters (`bench.py --decode`). The engine
before has no counters (a token took it 182 ms); the middle column is the engine as it is now with the read-ahead off
(`--prefetch 0`):

| | read-ahead off | now | what bounds it |
| --- | ---: | ---: | --- |
| The GPU computes (the CPU waits for it) | 80 ms | 89 ms | It reads 3.9 GiB of dense weights and 0.94 GiB of experts per token. This GPU reads memory at 88.7 GiB/s (`metal/tools/membench.mm`): 54 ms. The matrix kernels run at 53-93 GiB/s; about 2,000 small operations per token take 15-20 ms more (`GGML_METAL_PROFILE=1`). |
| The GPU waits for the SSD | 93 ms (127 missed experts) | 53 ms (60) | The SSD reads 3.4-3.7 GB/s with 8 reads at a time, 3.0 GB/s in the small groups a layer asks for: 0.6-0.7 ms per missed expert (2.04 MB). 67 experts per token are now read ahead, while the GPU computes; the GPU's own part is 9 ms longer for it. |
| The CPU (graphs, the cache's bookkeeping, the prediction) | 5 ms | 9 ms | The prediction is 4 ms of it (48 products of 512 x 2,560 per token). |
| **A token** | **178 ms** | **152 ms** | |

- **Heat bounds a long session.** The Air has no fan. For about 200 s it holds the cool speed; then the GPU's part of a
  token goes from 89 to 130-140 ms and stays there (the SoC at 62 C). The whole Mac draws 16-24 W while writing; over
  five minutes the engine wrote 0.261 tokens per joule before and 0.309 now.
- **Reading a long prompt** is bound by the SSD at 1,024 tokens per pass: a pass used 68-94% of a layer's experts (56 GB
  read for 1,600 tokens), which the SSD delivers in 17 s, and the GPU computes those two passes in 12 s. Before, the two
  took turns (34 s); now they overlap (25 s). With `--prefill 2048` the SSD's share halves and the GPU is the bound:
  82 tok/s at 1,600 tokens, 81 at 8,000 (as found; the context takes 1.1 GiB more, which comes out of the expert cache).
- **A short prompt** (up to 32 tokens) is bound by the SSD too: its tokens route to about 5,000 different experts that
  the cache does not hold (9-11 GB), which is 3 s of reading however the passes are cut.

### What made the difference

Each line is one change, measured against the engine without it (as found, same cache, interleaved, cool start).

- **Experts read ahead of their layer** (`--prefetch`, on): 5.63 -> 6.28 tok/s (+12%), and 6.25 -> 6.60 (+6%) more
  when the prediction corrects itself. The next layer's router applied to this layer's input names ten experts; of
  the next layer's real ten it holds 67%, and of those the cache would miss, 60% (100 tokens of a chat). The engine
  reads the ones whose predicted score is above the predicted tenth score, corrected by how far off each expert's
  prediction has been so far: about 2 per layer, of which 70% are then routed to. Reads ahead start when the reads a
  layer is waiting for are done, use half of the read threads, and are dropped if their layer routes elsewhere before
  they start.
- **Fewer commands to the GPU per layer:** 5.35 -> 5.63 tok/s (+5%). llama.cpp's expert cache read the routed ids and
  wrote its slot map through two small GPU copy commands per layer; on shared memory, once the GPU is idle, both are
  plain memory copies.
- **Long prompts: the next tensors are read while the GPU computes** (`--prefill-ahead`, on): 49.6 -> 64.4 tok/s at
  1,600 tokens. One tensor ahead gave 55.7; a layer's three, with the next layer guessed, 59-64. The guess read 47 GB
  ahead for a 1,600-token prompt and used 40 of them.
- **512 bytes that blocked on the disk:** 46.5 -> 49.3 tok/s at 1,600 tokens, and no more swap-outs. llama.cpp pads
  every run of experts with the first 512 bytes of the next expert, read from the mapped file on the main thread: a
  page fault that waits for the SSD, 22% of the time of a long prompt in a profile. They are now read with the run.
- **The GPU must not see the waits** (from the first version). At a layer with a missed expert the GPU has nothing to
  do for about a millisecond, and an Apple GPU that idles that often is clocked down. The engine runs a small compute
  kernel in a loop during a request (`metal/src/gpu_keepalive.mm`). With the engine as it is now: 6.15 tok/s with it,
  3.56-3.70 without it or with it only during the reads (`STRATA_KEEPALIVE=off|fetch|always`).
- **Long prompts in big passes** (from the first version). More than 32 new tokens are read in passes of `--prefill`
  tokens (default 1,024), without the cache.

### What did not help

Measured, and left out or left off:

- **A different rule for what the cache keeps.** `metal/tools/cache_sim.py` replays a routing trace (1,826 passes of
  the gate's requests). At 7 GiB: least recently used (the engine's rule) 75.2% hits, the same with the room split
  between the layer groups by the profile 75.2%, a two-queue rule (S3-FIFO) 75.1%, least frequently used with the
  profile as a start 63.0%. Evicting the expert whose next use is furthest away, which needs the future, would give
  85.1%: that gap is what the read-ahead goes after instead.
- **The keep-busy kernel, tuned.** A smaller kernel (4,096 threads instead of 32,768): 6.22 against 6.15 tok/s. A
  pause of 60 microseconds between its runs: 5.44 against 5.33 tok/s over five minutes and 18.6 W against 21.5 W, a
  pause of 120: 5.11 tok/s. Left as it was (`STRATA_KEEPALIVE_SLEEP_US` sets the pause).
- **The MTP head** (`--mtp`, off): see below. As found, with the same cache cap, 6.45 against 6.60 tok/s.
- **Reading the next layer's tensor whole** instead of guessing its experts: 16.6 against 22.8 tok/s at 256 tokens
  (a pass of 256 tokens uses under half of a layer's experts) and 15 GB more read at 1,600.
- **Short prompts in one GPU pass** (`--prefill-from 9`, off): the first token of a 20-token question after 3.1 s
  instead of 3.4, of a 32-token one after 3.7 instead of 4.1 s; the answer itself is no faster (6.19 against 6.23
  tok/s), and a different pass size can change a token where two are nearly tied, so it is not the default.

### The MTP head and lookup: several tokens per pass (opt-in)

`--mtp FILE` loads the model's MTP head (ggml-org's `mtp-Qwen3.8-Flash-Next-Q4_0.gguf`, 2.2 GB): after every token the
head drafts up to three more, and one pass of the model reads the token and the drafts. That pass gives the model's own
choice after each of them; the first draft that is not the model's choice ends it there, and the context goes back
(llama.cpp's snapshots of the recurrent layers, `n_rs_seq`). Every token that is written is the model's own, from its
own logits. `--lookup N` drafts the same way from the context itself: where its last N tokens stood before, what
followed them.

- **What the head gets right** (`STRATA_MTP_PROBE=FILE`: the head drafts, the drafts are counted and not used; a chat
  and a code request, 80 tokens each): its first draft 74-82% of the time, and 96-100% when it gives it a probability
  of 0.75 or more. A draft of three tokens takes 19 ms; showing the head a token the model read, 11-13 ms.
- **What a pass of several tokens costs** (every expert cached): 1 token 93 ms, 2 tokens 118 ms, 3 tokens 152-170 ms,
  4 tokens 170-205 ms. A second token in a pass costs a third of a first one, not nothing: the MoE part is per token.
- **What it takes:** 2.04 GiB for the head on the GPU and 0.6 GiB more for the context (the head's own, and three
  snapshots of the recurrent state), all of it out of the expert cache.
- **What it gives:** as found, with the cache capped at the same 6.3 GiB: 6.45 tok/s against 6.60 without it, 88% of
  the drafts right. Closed, where the head left the cache 9.3-9.6 GiB of the 12.2: 8.74 against 8.46 tok/s cool (+3%), and
  8.17 against 7.32 over five minutes (+12%; its requests after the first 200 s 7.2-8.2 against 6.2-6.8 tok/s, 90% of
  1,327 drafts right, 0.389 against 0.348 tokens per joule). Once the Air is warm the GPU is what a token waits for,
  and a pass that yields 1.6 tokens is worth more. One of its four closed runs swapped 340 pages out.
- **It is not token-identical.** A pass of two to four tokens runs other Metal kernels than a pass of one, and the
  logits differ by a few hundredths. In the gate, four of six requests left the reference, each at one token where the
  two candidates were at most 0.074 logits apart (with the reference's tokens fed in, every other token was the same).
  Three of the four are within the 0.05 the gate calls a near-tie and one is not (0.057 and 0.074), so the gate fails
  and the head is off by default.

## Context size

**Since this section was written (2026-10-12):** a 130,944-token prompt was read at `--max-context 131072` with F16
KV (64.9 tok/s, 33.6 minutes to the first token, 4.59 tok/s written, all five test records found, 61.7 MiB of
swap-outs). The plan's double count of the context is fixed, the compute buffer no longer grows while the context
fills, and the host part of it is in the plan. With `STRATA_METAL_INDEXED_QSA=1` and `--prefill 4096` the context
takes 6.02 GiB on the GPU and 0.45 GiB of host memory at 131,072. Later the same day, with aligned reads,
`--elastic` and `--prefill 8192`: 70.1 tok/s (31.1 minutes), 5.27 tok/s written, five records, no swap-out; that
configuration, at `--reserve-gib 3`, is the installed one. The figures below predate all of this; the
reports are [the morning's](../bench/results/2026-10-12-m4-air-128k/README.md) and
[the afternoon's](../bench/results/2026-10-12-m4-air-128k-resume/README.md), the open points are in
[OPTIMIZATION.md](../OPTIMIZATION.md).

`--max-context` has no upper limit in the engine; the model's own is 262,144 tokens. What limits it on this Mac is
memory: the context takes about 31 KB per token (the KV cache and the work buffers), and all of it comes out of the
expert cache. Measured with `--load-only` (nothing is read or written by such a start; 0 swap-outs):

| `--max-context` | Context on the GPU | |
| ---: | ---: | --- |
| 8,192 | 1.38 GiB | run (every figure under "Measured") |
| 32,768 | 2.15 GiB | loaded; what `setup-mac.sh` writes |
| 40,960 | 2.40 GiB | run: a 32,000-token prompt, read in 587 s |
| 65,536 | 3.18 GiB | loaded, no prompt run |
| 131,072 | 5.21 GiB | loaded, no prompt run |
| 262,144 | about 9.4 GiB | not loaded: 31.4 KB per token carried on from 131,072 |

- **The largest context a prompt was run in is 40,960.** Nothing is known about speed or output above it.
- **Filling a context takes long.** A long prompt is read at 55-65 tok/s (closed), so 32,000 tokens took 9.8 minutes.
  At that rate 65,000 tokens would take 17-20 minutes and 131,000 tokens 35-40; neither was run, and the Air is warm
  for most of it.
- **At large contexts the plan gives the expert cache less than there is room for** (see "Left open"). At 131,072,
  with 20.8 GiB free, it gave 1.6 GiB. A cache that small serves one token per pass, so a prompt of up to 32 tokens
  is read a token at a time. Until that is looked at, 65,536 is the largest size that leaves a cache worth having
  with programs open (5.7 GiB with 21.1 GiB free).

## Left open

Where the work of 2026-10-11 stopped (the owner asked for no further optimisation once the closed-state runs were
done). None of this is fixed or decided; the numbers are under "Measured" unless they stand here.

- **The memory plan seems to count the context twice at large sizes.** `Engine::load` in `metal/src/main.cpp` makes
  a context without a cache to see what it takes, frees it, and reads the free memory right after
  (`host_memory()`), then takes the context off that figure again (`by_ram`). What the log said against what free
  memory minus swap, dense weights (3.9 GiB), context and reserve (3 GiB) comes to: at 40,960 7.2 GiB against 9.6,
  at 65,536 5.7 against 8.8, at 131,072 1.6 against 6.5. The differences are close to the context's own size, as if
  the freed context's memory were not back yet when the free memory is read. At 8,192 one start gave 6.75 against
  6.5, so it may only show with large buffers. It errs towards a smaller cache and never towards swap. Not
  investigated further: for the 40,960 and 65,536 figures the free memory was read by a separate `--plan` start
  seconds before.
- **A 32-token prompt, closed state:** 6.8 tok/s against 7.7 before (one run each). Not followed up. Short prompts go
  through the cache in passes of a few tokens, and every token of such a pass names up to ten experts to read ahead;
  whether that is the cause was not tested.
- **The MTP head (`--mtp`) is off.** Whether to turn it on is the owner's decision: it is not token-identical, it
  takes 2.6 GiB of the cache, and one closed run with it swapped 340 pages. A guess at the last point, not tested:
  with drafting the recurrent state has a snapshot per draft (`n_rs_seq`), so each of the up to six conversation
  checkpoints the engine keeps in host memory is that many times larger.
- **`--prefill 2048`** (long prompts a quarter faster, 1.1 GiB less cache) and **`--prefill-from 9`** (first token
  of a short prompt sooner, pass sizes change) are off for the reasons given with their numbers.
- **`--lookup` was never run on the real model.**
- **Not measured as found:** a 32,000-token prompt. **One run only:** every closed-state prefill row.
- **The goal's own stopping rule was not reached** (three attempts in a row that each gain under 3%): the last kept
  change before the stop, the prediction's correction, gained 6%.

## Options

`build-metal/bin/strata --help` lists them. The config's `args` take:

| Option | |
| --- | --- |
| `--max-context N` | context in tokens (default 8,192; `setup-mac.sh` writes 32,768). The KV cache is F16: 2.2 GiB with the work buffers at 32K; see "Context size". |
| `--expert-cache auto\|N\|Nmib` | all that fits (default), N experts, or N MiB. A number larger than the plan is kept as asked and may make macOS swap. |
| `--expert-cache-cap N` | with `auto`: at most N MiB (benchmarks: the same cache in every run, still never more than is free) |
| `--reserve-gib G` | of the memory free at start, leave G GiB free (default 3) |
| `--overcommit-gib G` | let the cache pass Metal's recommended working set by G GiB. Not measured: on this Mac the free memory was the limit in every run, never Metal's. |
| `--expert-profile FILE` | Strata's expert ranking (`data/expert-profile.bin`): the cache starts filled with its top entries. `--no-preload` starts empty. |
| `--expert-io direct\|cached` | read missed experts past the OS file cache (default) or through it |
| `--prefetch N` | read up to N experts each layer is predicted to route to before it is computed (default 10, 0 = off). `--prefetch-margin M`: only those predicted M above the tenth score (0); `--prefetch-learn R`: how fast the prediction corrects itself (0.1); `--prefetch-threads T`: read threads that may read ahead at once (4) |
| `--prefill N` | tokens per pass for long prompts (default 1,024; 0 = off). 2,048 needs 1.1 GiB more for the context and read 8,000 tokens at 81 instead of 62 tok/s. |
| `--prefill-ahead on\|off` | long prompts: read the next expert tensors while the GPU computes (default on) |
| `--prefill-from N` | stretches of N or more new tokens go in one such pass (default 33) |
| `--elastic on\|off` | **experimental, default off.** The long passes' work memory (2.67 GiB at 131,072 context with `--prefill 4096`) and the expert cache take turns in one budget: answers are written with the larger cache, a long prompt is read with the cache's memory, and the cache comes back empty afterwards and fills as its experts are asked for (`--elastic-refill all`: it is read back from the SSD first, as it was until 2026-10-13, 4.0 s for 10.5 GiB; `--elastic-refill G`: its most used G GiB. Not reading it back gave a first token after 22.5 instead of 25.7 s for a 2,000-token prompt and the 50th after 29.4 instead of 31.6 s, A B B A). Measured with aligned reads: 7.71 against 6.38 tok/s written; a full 130,944-token prompt with `--prefill 8192` at 70.1 tok/s and 5.27 tok/s written; no swap-out in ten runs. Before the reads were aligned it swapped in five. Read the [report](../bench/results/2026-10-12-m4-air-128k-resume/README.md) before turning it on. Not with `--mtp` or a set `--expert-cache` size. `--elastic-short N`: a stretch of up to N new tokens (1,024) is read in one pass beside the cache, in memory that is free at that moment; 0 = always give the cache up. With `--elastic`, `--prefill 8192` costs the answering cache nothing (9% faster on a 16,384-token prompt than 4,096); long passes fall back to half or a quarter of `--prefill` when the memory has no room. Before a request the cache gives memory back if macOS wrote to swap since the last one (a quarter of itself, for good), or if it compressed 256 MiB or more and less than half of `--reserve-gib` is free (what is missing; it may come back after the next long prompt): both measured on the tiny model only. With `--elastic` the KV cache also takes its GPU memory as the conversation grows, 16,384 tokens at a time (0.47 GiB at 131,072 context), and the expert cache has the rest until then: 9.95 GiB instead of 6.72 for a conversation of up to 16,000 tokens, 8.5-8.7 against 7.1-7.5 tok/s written (A B B A), identical logits. Passing a step costs nothing before a long prompt (the cache is given up for it anyway) and 0.6-0.9 s otherwise (the cache is made anew, smaller, and fills again as it is asked for; 3.7-3.8 s with `--elastic-refill all`); `STRATA_KV_STEP=0` takes all of it at the start |
| `--mtp FILE`, `--spec N`, `--spec-min-p P` | the MTP head: up to N drafts per pass (3) while the head gives them probability P (0.75) or more. Off unless `--mtp` is given. |
| `--lookup N` | draft what followed the last N tokens where they stood before in the context (0 = off) |
| `--io-threads N` | threads reading experts (default 8; 4 or more read the same on the Air's SSD) |
| `--plan` / `--load-only` | print the memory plan without loading / load, print it as it came out, exit |
| `--verbose` | llama.cpp's own log, and the engine's memory after every prompt pass and every change of shape |

While a long prompt is read the engine reports how far it is every 5 seconds, not only after each pass (a pass of
8,192 tokens takes a minute and a half): the tokens of the passes that are done, plus the pass under way times the
share of its layers that are computed. The server's line shows it with the share of the prompt:
`reading the prompt: 2,645 of 5,077 tokens (52%), 27 s so far`. `STRATA_PASS_PROGRESS_MS` sets another interval.

`metal/tools/swapwatch.py NAME -- <command>` runs a benchmark under a watch on macOS's swap-out counter and stops the
engine the moment it moves; use it for every run on the real model. `metal/tools/determinism.py` starts the engine
afresh several times with one long prompt and compares IDs and logits between the starts.

Flags of the CUDA engine that a config may carry (`--pack`, `--kv-resident`, ...) are accepted and not used; the log
names each one.

## What is not there

- **No images, no `logit_bias`, no parallel requests** (`"parallel"`), no `SAVE` / `RESTORE` of conversations, no
  `VRAM` command. The server answers such requests with the engine's error.
- **One KV format** (F16; `--kv q8_0` is passed to llama.cpp, not measured).
- **The cache does not shrink while it runs.** Programs opened after the start get what is left; unload the model
  (`idle_unload_s`, or stop the server) to give the memory back. (With `--elastic` it comes back from a long prompt
  only as large as the memory that is free then, and gives a quarter of itself back before a request if macOS has
  swapped since the last one. That back-off has run on the tiny test model only.)
- **Bits differ from the CUDA engine.** The kernels are ggml's, not Strata's; the output was not compared token by
  token with the CUDA engine or with the BF16 model.
- **`--lookup` was built and not measured** (the protocol test does not cover it either); the MTP head was measured
  on short requests only, and its gate was run on six of the nine requests.

## How it is built

- `metal/src/main.cpp`: the engine. The memory plan, the load (experts stay mapped from the GGUF, everything else of
  the layers goes into GPU buffers of its own), the cache fill from the profile, the conversation checkpoints (the
  recurrent layers' state at the last message boundary and before the last prompt token, so the next turn and a
  repeated prompt reuse the context), drafting and checking (`--mtp`, `--lookup`), and the serve loop.
- `metal/src/expert_io.hpp`: where each tensor sits in the shards; the threads that read experts into their slots
  (reads a layer waits for go before reads ahead, and a read that has not started can be dropped); the read-ahead of a
  long prompt's tensors. The first read of each tensor is compared with the mapped file; a mismatch turns the fetch
  off and llama.cpp's own upload takes over.
- `metal/src/router_predict.hpp`: the prediction of the next layer's experts. `metal/src/mtp_head.hpp`: the MTP head.
- `metal/src/gpu_keepalive.mm`: the kernel that keeps the GPU at its working clock.
- `metal/patches/0001-moe-cache-fetch-stats-preload.patch`, on llama.cpp `23b0202a1` (2026-10-10), 810 lines
  added in 8 files. The MoE cache offers its misses to a callback before uploading them (and the large-batch path its
  uploads and its padding); it can be filled with a list of (layer, expert) pairs, asks a callback which experts to
  bring in ahead and tracks their slots until their layer has routed; it reads the routed ids and writes its slot map
  in place when the buffers are host memory; it reports its hits, slots and times. ggml-metal counts the time the host
  waits for the GPU and can time every kernel (`GGML_METAL_PROFILE=1`). `LLAMA_MMAP_NO_PREFETCH` stops the read-ahead
  of a model file larger than the RAM.
- `0002`-`0005`: indexed block-sparse attention, the top-k workspace, the reserve for a pass appended to a filled
  cache, `llama_context_resize` (`--elastic`). `0006-apple-row-prefetch.patch`: the rows of a lazily read table are
  read ahead on macOS as on Linux and Windows. `0007-metal-topk-deterministic.patch`: the radix top-k writes its
  result in index order instead of in the order its threads arrive. `0008-kv-cache-mapped-as-it-fills.patch`: a
  Metal buffer can be mapped in parts (`ggml_backend_buffer_set_views`; a Metal buffer is resident as a whole, so
  the parts are separate buffers over the same host memory), and with `LLAMA_KV_LAZY_CELLS=N` the KV cache maps
  its cells N at a time as they come into use (`llama_kv_cells_mapped`, `llama_kv_map_cells`).

## Tests and tools

```sh
# a tiny random qwen4exp from llama.cpp's own test tool
cmake -S ../llama.cpp -B ../llama.cpp/build -DLLAMA_BUILD_TESTS=ON && cmake --build ../llama.cpp/build --target test-llama-archs
../llama.cpp/build/bin/test-llama-archs --arch '^qwen4exp$' --out /tmp/tiny
.venv/bin/python metal/tests/serve_protocol_test.py --model /tmp/tiny/qwen4exp-moe.gguf
```

29 checks: the protocol's lines, the same ids with the cache on, off, with direct and cached reads, every chunk
size, prefetch, `--elastic` and (a long prompt) read-ahead, context reuse against a fresh engine, STOP, and the
error answers. Two more tests on the same model: `metal/tests/elastic_shapes_test.py --model ...` (six requests
through `--elastic`'s three shapes against an engine without it) and `metal/tests/optimization_parity_test.py`
(ids and logits of the default, indexed and elastic paths against a reference build). The kernel tests are built
with `-DSTRATA_METAL_BUILD_TESTS=ON` and run by `ctest --test-dir build-metal`.
The tiny model has 4 experts, so it cannot miss: what the read-ahead does is checked on the real model, by the gate
and by `STRATA_VERIFY_CACHE=1`. `serve/` is not changed apart from the Monitor's CPU name and a note about GPU
readings on macOS (`serve/telemetry.py`).

On the real model (the server must not be running):

```sh
.venv/bin/python metal/tools/bench.py --config strata-iq3_s.json --context 8192 --gate record     # the reference ids (build-metal/gate-ref.json)
.venv/bin/python metal/tools/bench.py --config strata-iq3_s.json --context 8192 --gate check -- --chunk 7
.venv/bin/python metal/tools/bench.py --config strata-iq3_s.json --context 8192 --decode --sustained 300 --cool 42
.venv/bin/python metal/tools/bench.py --config strata-iq3_s.json --context 8192 --prefill 32,256,1600,8000 --decode \
    --cool 42 --runs 3 --ab build-metal/baseline/strata "build-metal/bin/strata --prefetch 0"
```

`bench.py` reports tokens/s for reading and writing, the time to the first token, the cache's hits, the GB read from
the SSD, where a token's time went, the Mac's watts (from the battery's telemetry, no sudo) and the swap-outs of every
request. `--gate check` compares ids with a reference; where one differs it runs the request again with the
reference's tokens fed in and accepts the difference only if both engines had the two tokens within 0.05 logits.
`--cool` waits for the SoC to cool first (it needs `metal/tools/temps.m` compiled to `build-metal/bench/temps`; the
command is in the file). `metal/tools/cache_sim.py` replays a routing trace (`STRATA_ROUTE_TRACE=FILE`) under other
cache rules; `metal/tools/membench.mm` measures how fast the GPU and the CPU read memory.
