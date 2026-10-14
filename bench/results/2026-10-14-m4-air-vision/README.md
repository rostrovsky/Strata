# Pictures on the Metal engine — M4 MacBook Air, 2026-10-14

The owner asked whether Qwen3.8-Flash-Next is multimodal, and if so to read `~/Pictures/calipers.jpg`, give it to the
model, and fix what keeps the model from understanding it. It is multimodal ([Qwen's release
notes](https://qwen.ai/blog?id=qwen3.8-flash-next): "a multimodal MoE model"; vision benchmarks are published beside
the language ones). The installed model could not see: the vision encoder is a separate file that had not been
downloaded, the Mac had no `strata-vision`, and the Metal engine answered every picture with "this engine was started
without --vision".

Machine and state: MacBook Air M4 (10-core GPU, 32 GB, fanless), macOS 15.7.9, desktop programs open (other programs'
memory 6.7-8.7 GiB in RAM, 10-12 GiB stored compressed), the installed configuration (`--max-context 131072 --kv f16
--prefill 8192 --reserve-gib 3 --elastic on`, `STRATA_METAL_INDEXED_QSA=1`). Every real-model run under
`metal/tools/swapwatch.py`: **no swap-out in any run** (the counter stayed at 1,586,587 pages all day).

The picture: `calipers.jpg` is a WebP of 1,280 x 1,280 pixels (the name says JPEG; the server converts WebP to PNG with
Pillow, as it did before). A Mitutoyo dial caliper lying on its black case; the dial is yellow, graduated in 0.01 mm,
and its small print reads `505-682`, `SHOCK PROOF`, `MADE IN JAPAN`; the jaws are closed and the needle is on 0, so
it reads 0.00 mm. The beam's scale shows 60-120 where it comes out from under the slider.

## What was changed

1. **The vision projector** `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 bytes, SHA-256 `b1a82259...49bd0`, the
   same at setup's pinned revision and at `main`) was downloaded with the owner's permission into
   `Strata-data/models/mmproj/`. Its header: `clip.projector_type qwen3vl_merger`, 27 blocks of width 1,152, patch 16,
   spatial merge 2, projection 2,560 (the text model's input width), no deepstack layers.
2. **`strata-vision` is built with the engine** (`metal/CMakeLists.txt`, `STRATA_METAL_VISION`, on by default): llama.cpp's
   `mtmd` at the engine's pinned and patched commit, on Metal. `tools/vision/strata_vision.cpp` needed `F_RDADVISE`
   in place of `posix_fadvise`, which macOS does not have, and has `--no-warmup`.
3. **`GENI` in the Metal engine** (`--vision`, `metal/src/main.cpp`). A picture is its run of `<|image_pad|>`
   tokens, read as the rows `strata-vision` wrote, in llama.cpp's mixed token and embedding batches
   (`llama_batch_ext`, `llama_process`), at mtmd's M-RoPE positions: every row at the position p of its first token,
   height p + y and width p + x (`[t, y, x, 0]`, the order of mtmd's `set_position_mrope_2d`), the text after it from
   p + max(nx, ny). The conversation cache compares pictures by their grid and a hash of their rows as well as by
   tokens, and is cut at positions, not token counts. Text requests take the path they took before (`decode`,
   positions = indices). No drafts while the context holds a picture; a picture is refused beside the MTP head.
4. **The encoder on demand** (`serve/server.py`, config `"vision": {..., "on_demand": true}`): it starts for a request's
   pictures and ends once they are encoded, so its 1.1 GB never comes out of the expert cache (which the engine sizes
   from the memory that is free at its start). With one picture the encoder's own rows file is the one the engine
   reads, instead of a copy written for each request (~10 MB per turn: clients send the picture again every turn).
5. **`setup-mac.sh --vision yes|no`** (yes by default) downloads the projector and writes the `"vision"` section.
6. **Installed** (the owner's `strata-iq3_s.json`, git-ignored; before and after in `installed/`): `--vision` and the
   `"vision"` section with `"max_tokens": 2048`. pi's entry for `qwen3.8-flash-next-iq3_s` now has `"input": ["text",
   "image"]` (backup `~/.pi/agent/models.json.backup-20261014-184615-vision`); nothing else in pi was changed.

## The encoder (`encoder/`)

`strata-vision` by hand on the photo (converted to PNG for this test), the engine not running. These runs used the
first build of the encoder (17:28), the same encoding code as the one installed (17:50), which only adds
`--no-warmup`.

| | image tokens | time | peak memory |
| --- | ---: | ---: | ---: |
| GPU (Metal) | 1,024 (32 x 32) | 2.74 s, 2.87 s | 1.13 GB |
| GPU, the photo's full size | 1,600 (40 x 40) | 5.11 s | 1.16 GB |
| GPU, the warm-up's 2,048 x 2,048 grey square | 4,096 | 25.7 s | 1.29 GB |
| CPU (5 threads, BF16) | 1,024 | 206.4 s, 213.7 s | 1.22 GB |

The projector loads in 0.4-1.0 s. The CPU is not an option. The warm-up at 4,096 tokens (which the CUDA setup uses to
reserve VRAM before the engine starts) is 25.7 s and pointless on demand, hence `--no-warmup`.

## The model through the server (`server/`)

`scripts/vision_smoke.py` starts `serve/server.py` with a copy of the installed configuration plus the vision section
(port 8091) and sends OpenAI chat requests with the picture as a data URL. `temperature 0` unless said otherwise.

| run | image tokens | request | prompt | read in | answer |
| --- | ---: | --- | ---: | ---: | --- |
| smoke1 | 1,024 | describe it, no thinking | 1,071 | 18.2 s | 669 tokens, 6.78 tok/s |
| smoke1 | 1,024 | second turn (resolution, closed jaws) | 1,777 (1,739 reused) | 4.4 s | 371 tokens, 6.31 tok/s |
| smoke1 | 1,024 | describe it, thinking, 3,000 tokens at most | 1,111 | 19.9 s | 3,000 tokens of thinking, no answer (6.06 tok/s) |
| smoke2 | 1,600 | read the dial's text, no thinking | 1,635 | 20.9 s | 61 tokens, 8.11 tok/s |
| smoke2 | 1,600 | describe it, no thinking | 1,647 | 19.8 s | 700 tokens (cut), 7.40 tok/s |
| smoke3 | 1,600 | describe it, thinking, the model's sampling (t 1.0, top-p 0.95, top-k 20) | 1,687 | 21.5 s | 4,230 tokens, 6.96 tok/s, 635.5 s |

What it said (whole answers in `server/smoke*.json`):

- **smoke1, 1,024 tokens:** a Mitutoyo dial caliper in its case, "→|← 0.01mm", "MADE IN JAPAN", 0.00 mm because the
  jaws are closed. The model number it gave, "525-108C", is wrong; "SHOCK PROOF" is missing. The second turn explained
  the 0.01 mm graduation (ten divisions between the numbers, one turn = 1 mm) and how to see that the jaws are
  closed. The thinking run concluded 0.00 mm in its thinking and ran out of its 3,000 tokens before answering.
- **smoke2, 1,600 tokens (the photo's own size):** the dial's text exactly, line by line: `0 ... 90`, `Mitutoyo`,
  `→|←0.01mm`, `505-682`, `SHOCK PROOF`, `MADE IN JAPAN`. The description then has the model number right and calls
  the reading **70.00 mm** ("the jaws are open past the 70 mm mark"): wrong, from the first beam number to the right
  of the slider.
- **smoke3, 1,600 tokens with thinking:** right in every point: "a Mitutoyo 505-682 shock-proof dial caliper (0.01 mm
  resolution, made in Japan), shown closed and reading 0.00 mm", and why the beam's numbers are not the reading. One
  slip: it read the beam's figures (each with a graduation line through it) as 610, 710, ...

So the engine gives the model what it needs, the small print needs the photo's full size, and the reading needs
thinking (or luck). The 70.00 mm answer is the model's: it read the text in the right order and place, which a
position error would scramble.

The third smoke1 request reused nothing although its picture had just been read: with thinking on, the server puts the
reasoning-effort sentence at the top of the prompt, so the two prompts differ from the start. In smoke2 the second
request (another question about the same picture) reused nothing either: a conversation is taken up again only where
a message ends (the recurrent layers' state is kept there), and the two questions are in the same message as the
picture.

Memory (`server/*-vmstat.txt`, 1 s; `build-metal/resume-128k/memsum.py`): the 20-second windows around the encoder's
start and encoding have no compressions; the engine's start compressed 5.7 GiB of other programs' memory, as every
start does, and two blips of 190 and 167 MiB came during smoke1's long thinking answer.

## Through pi (`pi/`)

`scripts/pi_vision.py`: the installed `strata-iq3_s.json` (port 8080), and `pi --provider qwen-local --model
qwen3.8-flash-next-iq3_s --thinking off ... --print @calipers.jpg "What instrument ..., brand and model number ...,
what does it read right now? Three short lines."` (the picture's bytes are left out of `pi1-pi.jsonl`):

> **Instrument:** Dial caliper (Mitutoyo "Shock Proof" series, 0.01 mm resolution).
> **Markings:** Brand "Mitutoyo" on dial and case; dial shows model ref. "505-682" (partially obscured by glare).
> **Reading:** 0.00 mm — jaws fully closed, main scale at 0, dial needle at 0.

46.6 s for the whole request: the prompt of 2,502 tokens (pi's instructions and the picture's 1,600) read in 27.3 s,
89 tokens at 7.00 tok/s. No swap-out.

## Tests (`tests/`)

- `metal/tests/vision_test.py` (new, tiny qwen4exp model, random picture rows): six requests (a picture; a
  continuation; back to the checkpoint after the picture; another picture of the same size in its place; no picture;
  two pictures) give the same ids and logits in one engine in turn as in a fresh engine each; read in passes of 40
  (the pictures cut apart) and of 7 they give the same ids and logits within 1e-4 (the sixth decimal: a pass of text,
  of picture rows or of both builds its input rows in a different way in llama.cpp, and a picture that is not cut but
  has a pass to itself moves the logits as much; passes of 7 over text alone do the same). The cache was used past
  the picture (127 of 128) and not across a different one (0). The four refusals.
- `metal/tests/elastic_shapes_test.py`, `metal/tests/serve_protocol_test.py`: all passed (text unchanged).
- `serve/` tests: `test_server` (with the new `VisionOnDemand`), `test_vision_busy`, `test_frontend`,
  `test_exif_orientation`, `test_responses`: OK.

## Seen on the way, not changed

- The engine stops with `libc++abi: ... mutex lock failed: Invalid argument` when it is told to quit. Also in
  `strata-iq3_s.log` from text-only runs before this work; it happens after the last answer, at exit.
- The engine's "MB from the SSD" per request (63-853 GB here, 75-750 GB in the production log of text runs) cannot be
  what the SSD delivered in those seconds; the figure is not what it says. Also from before.
- The server's first progress line of an answer can say `349525.3 tok/s` (one token over no time).
