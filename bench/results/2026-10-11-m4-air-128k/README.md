# M4 MacBook Air: 128k with other apps open

Machine: Apple M4 MacBook Air, 10 CPU cores (4 performance, 6 efficiency), 10 GPU cores, 32 GiB unified memory.
AC power, Low Power Mode off. Model: the local Qwen3.8-Flash-Next GSQ-RCO IQ3_S shards. Engine before this work:
Strata `3497a384`, Metal on llama.cpp `23b0202a189c44a54625aadcb37a946dd1d6278d` plus Strata's existing patch.
Measured on 2026-10-11. The target is normal use with a browser, Ghostty, and an agent harness open.

## Changes

The old memory planner measured a temporary context, freed it, sampled free host memory, then subtracted the context
again. Metal can retain the freed physical pages long enough for that sample to count them as occupied. The new
planner samples before and during the temporary context. It uses the smaller of free memory minus the measured
context, and free memory with the context alive, then subtracts the reserve. It still respects Metal's working set,
existing swap debt, and the cache cap.

Prompt read-ahead staging now releases after prefill, before the answer. Previously it stayed allocated until the
request ended. This returns up to about 1.1 GiB during decode without changing the model's math.

## Candidate settings

```text
--max-context 131072 --kv q8_0 --prefill 2048 --reserve-gib 4 --expert-cache auto
```

The reserve is part of the memory plan; prompt staging and conversation checkpoints also need memory while a request
runs. The cache is sized at startup and does not shrink when another app grows. No system memory limits are changed.

Q8 KV reduces the context allocation at a 1,024-token batch from 5.21 to 3.45 GiB. With a 2,048-token batch it uses
4.49 GiB. Quantizing KV can change output tokens; it is separate from the memory fix, which changes no model math.

## Measurements so far

The comparison uses the same 131,072-token capacity, Q8 KV, 2,048-token batch, 4 GiB reserve, and auto cache in both
engines. Order: old, new, new, old. Each run answers the same chat and code requests, 100 tokens each. Apps remain
open, so the amount of free memory changes. These are short prompts in a 128k allocation, not decode at 128k depth.

| Engine | Decode, each run | Median | Expert cache, each run | New swap-outs |
| --- | ---: | ---: | ---: | ---: |
| Before | 2.51, 4.66 tok/s | 3.58 tok/s | 577, 3,106 MiB | 0 |
| Updated | 6.12, 6.59 tok/s | 6.36 tok/s | 5,596, 6,896 MiB | 0 |

The median gain is 77%. The first old run had too little cache for two layers, which fell back to the CPU. This is
an auto-sizing comparison, not a comparison at a fixed cache size. See [planner-ab.json](planner-ab.json).

At 128k capacity, Q8 KV, 4 GiB reserve, a 4,096-token document followed by 48 output tokens:

| Maximum prefill batch | Prefill | Decode at 4k | Expert cache | New swap-outs |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 71.2 tok/s | 5.91 tok/s | 8,013 MiB | 0 |
| 4,096 | 67.0 tok/s | 5.17 tok/s | 6,030 MiB | 0 |
| 8,192 | 65.2 tok/s | 3.54 tok/s | 1,943 MiB | 0 |

These are one run each, in the order shown, without a cooldown. The 8,192 setting reserves more workspace, but this
prompt only exercises a roughly 4k pass. It does not establish the throughput of an 8k pass. The larger settings
lose enough expert cache to hurt decode. See [batch-sweep.json](batch-sweep.json).

## Validation

All 26 checks in `metal/tests/serve_protocol_test.py` pass on the generated tiny qwen4exp model. A real-model smoke
test retrieves all five exact values placed across 2,048 tokens. Repeating the request reuses 2,047 tokens, gives
the same 76 output token IDs, and reduces time to first token from 25.89 to 0.29 seconds.

Full-context retrieval and decode measurements are pending. This report does not yet claim that a filled 128k
context has passed.

## Reproduce

Keep a copy of the old executable before rebuilding. Use the same model and leave the usual apps open. Do not run
two real-model engines at once.

```sh
cmake --build build-metal -j 4
.venv/bin/python metal/tools/bench.py --config strata-iq3_s.json --context 131072 \
  --decode --only chat --only code --new 100 --runs 2 \
  --ab build-metal/tuning-128k/strata-before build-metal/bin/strata \
  --json build-metal/tuning-128k/planner-ab.json \
  -- --kv q8_0 --prefill 2048 --reserve-gib 4

.venv/bin/python metal/tools/bench.py --config strata-iq3_s.json --context 131072 \
  --prefill 130560 --prefill-new 128 --needle --prefill-repeat \
  --json build-metal/tuning-128k/full-context.json \
  -- --kv q8_0 --prefill 2048 --reserve-gib 4
```

The context limit includes prompt and output tokens. A 130,560-token prompt leaves room for the answer and the
engine's boundary margin. The retrieval test inserts five unique records at 5%, 25%, 50%, 75%, and 95% of the text.
It checks exact values, then repeats the request to check output identity and prefix reuse. This is a functional
test, not a broad model-quality evaluation.
