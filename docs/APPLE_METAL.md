# Apple silicon (Metal): build from source on a Mac

Strata runs on Apple silicon Macs through a separate engine in `metal/`. **Status: experimental.** Everything here was
built and measured on one machine: a MacBook Air M4 (10-core GPU, 32 GB, fanless, macOS 15.7.9) with the IQ3_S model, on
2026-10-11. Nothing else has been run: no other Mac, no other size, no images. Back to the [README](../README.md).

The CUDA / HIP engine (`src/`) cannot be compiled for a Mac: its kernels are CUDA C++. This engine takes the model graph
and the Metal kernels from [llama.cpp](https://github.com/ggml-org/llama.cpp) (the `qwen4exp` architecture, `ggml-metal`,
at a pinned commit with a small patch) and puts Strata around them: the memory plan, the expert cache's start-up fill from
Strata's expert profile, a read-only SSD tier, and the `--serve` protocol. The server, the web app and the APIs
(`serve/`) are the same ones the other backends use.

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
plan has room for a 15-16 GiB cache, a third of them; the rest is read from the SSD while the model answers.

### Nothing is written

The model files are opened read-only and the engine writes no pack, no cache file and no copy of the experts. Three
things that would otherwise put model data into swap are switched off or checked:

- **The cache is sized from free memory, not from the RAM size.** The engine counts what is free or file cache, takes
  off what macOS already holds in swap (memory other programs will want back) and 3 GiB more (`--reserve-gib`), and
  gives the expert cache what is left. A Mac with a browser, an editor and a VM open gave it 7.5 GiB where an empty one
  would give 15.8. The log says how much closing programs would add. Wired memory is never swapped, so a cache that is
  too large pushes *other programs* into swap; that is what this rule avoids.
- **No repacked copies.** ggml's ARM kernels would copy 17.5 GiB of this model's experts (the Q2_0, Q4_K and IQ4_NL
  ones) into a repacked buffer. That is anonymous memory and went straight into swap on the 32 GB Air (18 GB of
  swap-outs in one start, before this was found). The engine loads with `use_extra_bufts = false`, and stops if its
  own memory after loading is more than the dense weights.
- **The GPU buffers stay wired while the engine runs.** ggml stops its residency requests after 3 idle minutes, after
  which macOS could compress or swap the expert cache. The engine keeps them going; to give the memory back, the model
  is unloaded instead (`"idle_unload_s"` in the config, 900 by default from `setup-mac.sh`; the next request loads it
  again in a few seconds).

Missed experts are read past the OS file cache (`--expert-io direct`, F_NOCACHE), so reading 40 GB during an answer does
not fill the memory with file pages. `--expert-io cached` reads through it instead: experts that were read recently then
come from RAM, but with little free memory it made macOS swap (17,700 pages during three 120-token answers), so it is not the
default.

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

MacBook Air M4, 32 GB, IQ3_S, context 8,192, the defaults. Other programs were open and held 7.5 GiB, and 4.5 GiB were
in swap from earlier, so the expert cache got **7.5 GiB: 3,911 of the 24,576 experts (16%)**. `metal/tools/bench.py`
runs these three requests; one run, greedy.

| Request | Reads the prompt | Writes the answer | Cache hits | Read from the SSD |
| --- | ---: | ---: | ---: | ---: |
| A one-line question, 200 tokens out | 20 tokens in 3.3 s | **6.0 tokens/s** | 76.5% | 46 GB |
| The next turn of that chat (219 of 240 prompt tokens reused), 41 tokens out | 21 tokens in 3.5 s | 6.0 tokens/s | 70.5% | 15 GB |
| A 1,610-token document to summarize, 200 tokens out | **48 tokens/s** | 6.3 tokens/s | 77.6% | 84 GB |

- **With every routed expert in the cache** a token takes 104 ms (9.6 tokens/s; a 16-token answer repeated, 99.8% hits).
  That is the ceiling of this engine on this GPU: a token reads 3.9 GiB of dense weights and about 0.9 GiB of experts.
- **A missed expert** costs its read: 0.6-0.8 ms each (2.8 GB/s when two experts are read at a time, 3.3-3.5 GB/s at
  eight; `strata --native MODEL --io-bench 1200` measures it on your SSD). At 76% hits a token misses about 113.
- **Hit rate by cache size** (short chats, different amounts of free memory): 6.5 GiB (14% of the experts) 69%,
  7.5 GiB 76.5%, 8.0 GiB 78%, 9.9 GiB 82%, 10.9 GiB (23%) 81-85%. The 15.8 GiB cache of an otherwise empty 32 GB Mac
  was not measured: it would have needed the open programs closed.
- **Start:** ready in 3.5-5.7 s once the model file's dense parts are in the OS file cache; the cache fill reads
  7.5-11 GiB in 2.2-3.6 s (3.0-3.5 GiB/s).
- **Swap:** 0 to 120 pages swapped out during these runs.

### What made the difference

- **The GPU must not see the waits.** At a layer with a missed expert the GPU has nothing to do for about a
  millisecond, and an Apple GPU that idles that often is clocked down. With every expert cached and a wait put in at
  each layer: no wait 101 ms per token, 0.5 ms 347 ms, 1.5 ms 409 ms (the waits themselves are 24 and 72 ms). The
  engine runs a small compute kernel in a loop during a request (`metal/src/gpu_keepalive.mm`): the chat above went
  from 2.93 to 5.94 tokens/s (4.19 with the kernel only during the reads). `STRATA_KEEPALIVE=off|fetch|always`.
- **Long prompts in big passes.** A prompt read in passes the expert cache serves (8-11 tokens here) ran at 5-6
  tokens/s. More than 32 new tokens are read in passes of `--prefill` tokens instead (default 1,024), without the
  cache: by then a layer routes to most of its 512 experts, and they are read from the SSD into a work buffer. 512 per
  pass read the document at 33 tokens/s, 1,024 at 45-48. The cache is left alone, so it stays what the answer needs.
- **Heat.** The Air has no fan. In one test (the all-cached answer repeated 24 times) the time per token went from 99 to
  190-240 ms after about 35 s of load and stayed there; the 200-token runs above did not show it. Expect long sessions
  to be slower than the table.

## Options

`build-metal/bin/strata --help` lists them. The config's `args` take:

| Option | |
| --- | --- |
| `--max-context N` | context in tokens (default 8,192; `setup-mac.sh` writes 32,768). The KV cache is F16: 2.2 GiB with the work buffers at 32K. |
| `--expert-cache auto\|N\|Nmib` | all that fits (default), N experts, or N MiB. A number larger than the plan is kept as asked and may make macOS swap. |
| `--reserve-gib G` | of the memory free at start, leave G GiB free (default 3) |
| `--overcommit-gib G` | let the cache pass Metal's recommended working set by G GiB. Not measured. |
| `--expert-profile FILE` | Strata's expert ranking (`data/expert-profile.bin`): the cache starts filled with its top entries. `--no-preload` starts empty. |
| `--expert-io direct\|cached` | read missed experts past the OS file cache (default) or through it |
| `--prefill N` | tokens per pass for long prompts (default 1,024; 0 = off). 2,048 needs 1.1 GiB more for the context. |
| `--io-threads N` | threads reading experts (default 8; 4 or more read the same on the Air's SSD) |
| `--plan` / `--load-only` | print the memory plan without loading / load, print it as it came out, exit |

Flags of the CUDA engine that a config may carry (`--pack`, `--mtp`, `--spec`, `--kv-resident`, ...) are accepted and
not used; the log names each one.

## What is not there

- **No speculation.** The CUDA engine drafts with the model's MTP layer and checks several tokens per pass (1.6-1.8x
  there). This engine writes one token per pass. llama.cpp has an MTP context for this model; it is not wired in.
- **No images, no `logit_bias`, no parallel requests** (`"parallel"`), no `SAVE` / `RESTORE` of conversations, no
  `VRAM` command. The server answers such requests with the engine's error.
- **One KV format** (F16; `--kv q8_0` is passed to llama.cpp, not measured).
- **The cache does not shrink while it runs.** Programs opened after the start get what is left; unload the model
  (`idle_unload_s`, or stop the server) to give the memory back.
- **Bits differ from the CUDA engine.** The kernels are ggml's, not Strata's; the output was not compared token by
  token with the CUDA engine or with the BF16 model.

## How it is built

- `metal/src/main.cpp`: the engine. The memory plan, the load (experts stay mapped from the GGUF, everything else of
  the layers goes into GPU buffers of its own), the cache fill from the profile, the conversation checkpoints (the
  recurrent layers' state at the last message boundary and before the last prompt token, so the next turn and a
  repeated prompt reuse the context), and the serve loop.
- `metal/src/expert_io.hpp`: where each tensor sits in the shards, and the thread pool that reads experts into their
  slots. The first read of each tensor is compared with the mapped file; a mismatch turns the fetch off and
  llama.cpp's own upload takes over.
- `metal/src/gpu_keepalive.mm`: the kernel that keeps the GPU at its working clock.
- `metal/patches/0001-moe-cache-fetch-stats-preload.patch`, on llama.cpp `23b0202a1` (2026-10-10): the MoE cache
  offers its misses to a callback before uploading them (and the large-batch path its uploads), reports its hits and
  slots, and can be filled with a list of (layer, expert) pairs; `LLAMA_MMAP_NO_PREFETCH` stops the read-ahead of a
  model file larger than the RAM. 249 lines added in 6 files.

## Tests

```sh
# a tiny random qwen4exp from llama.cpp's own test tool
cmake -S ../llama.cpp -B ../llama.cpp/build -DLLAMA_BUILD_TESTS=ON && cmake --build ../llama.cpp/build --target test-llama-archs
../llama.cpp/build/bin/test-llama-archs --arch '^qwen4exp$' --out /tmp/tiny
.venv/bin/python metal/tests/serve_protocol_test.py --model /tmp/tiny/qwen4exp-moe.gguf
```

20 checks: the protocol's lines, the same ids with the cache on, off, with direct and cached reads and every chunk
size, context reuse against a fresh engine, STOP, and the error answers. `serve/` is not changed apart from the
Monitor's CPU name and a note about GPU readings on macOS (`serve/telemetry.py`).
