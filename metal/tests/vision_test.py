"""metal/tests/vision_test.py - pictures (GENI, --vision) on a tiny qwen4exp model: a prompt with pictures gives the
same ids and logits whether it is read whole, in passes that cut its pictures apart, or after a conversation the
engine continues from its cache or a checkpoint; a different picture of the same size is never taken from the cache.

    python metal/tests/vision_test.py --model tiny/qwen4exp-moe.gguf [--engine build-metal/bin/strata]

The tiny model has no tokenizer, so STRATA_METAL_TEST_IMAGE_PAD names its picture token (100), and the pictures are
random rows written as strata-vision writes them (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats).
"""
from __future__ import annotations

import argparse
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metal" / "tools"))
from bench import Engine

PAD = 100


def text(k, s):
    return [1 + (i * 7 + s) % 95 for i in range(k)]


def write_pictures(path, pics, width):
    """pics: (nx, ny, seed) per picture, in prompt order"""
    with open(path, "wb") as f:
        for nx, ny, seed in pics:
            rnd = random.Random(seed)
            n = nx * ny
            f.write(struct.pack("<5i", 0x31455653, n, nx, ny, width))
            f.write(struct.pack(f"<{n * width}f", *(rnd.gauss(0.0, 0.05) for _ in range(n * width))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", default=str(ROOT / "build-metal" / "bin" / "strata"))
    ap.add_argument("--width", type=int, default=256, help="the model's input width (n_embd_inp)")
    a = ap.parse_args()
    base = ["--native", a.model, "--max-context", "4096", "--prefill", "256", "--kv", "f16", "--verbose", "--vision"]
    env = {"STRATA_METAL_INDEXED_QSA": "1", "STRATA_METAL_TEST_IMAGE_PAD": str(PAD)}
    A, B, C = (8, 6, 1), (8, 6, 2), (4, 3, 3)          # B is A's size with other rows
    head = text(50, 3)
    p_a = head + [PAD] * 48 + text(30, 5)
    requests = [  # (name, ids, pictures)
        ("A", p_a, [A]),
        ("A, then more", p_a + text(40, 9), [A]),
        ("A, then other text", p_a + text(40, 11), [A]),   # back to the checkpoint after the picture
        ("B in A's place", head + [PAD] * 48 + text(30, 5), [B]),
        ("no picture", head + text(30, 5), []),
        ("two pictures", text(20, 1) + [PAD] * 12 + text(10, 2) + [PAD] * 48 + text(25, 4), [C, A]),
    ]
    ok = True
    with tempfile.TemporaryDirectory(prefix="strata-vision-test-") as temp:
        files = []
        for i, (_, _, pics) in enumerate(requests):
            files.append(None)
            if pics:
                files[i] = str(Path(temp) / f"r{i}.sve")
                write_pictures(files[i], pics, a.width)

        def run(name, extra, which):
            e = Engine(a.engine, base + extra, Path(temp) / f"{name}.log", env)
            try:
                return [e.gen(requests[i][1], 8, " probe=1", embeddings=files[i]) for i in which]
            finally:
                e.close()

        every = range(len(requests))
        one = run("one", [], every)                                  # one engine, the requests in turn
        # Other passes: of 40 tokens (they cut the pictures apart) and of 7 (other kernels).  These agree to within
        # the last digit printed (1e-6), not exactly: a pass of text, of picture rows or of both builds its input rows
        # in a different way (llama.cpp's build_inp_embd), and a picture that is not cut at all but has a pass to
        # itself moves the logits as much.  Passes of 7 over text alone do the same (matrix-vector kernels).
        cut = run("cut", ["--prefill", "40"], every)
        small = run("small", ["--prefill", "0", "--chunk", "7"], every)
        fresh = [run(f"fresh{i}", [], [i])[0] for i in every]       # each request alone

        def near(x, y):
            return x["ids"] == y["ids"] and all(abs(p[k] - q[k]) <= 1e-4 for p, q in zip(x["probes"], y["probes"])
                                                for k in (2, 4, 6, 8))
        for i, (name, ids, pics) in enumerate(requests):
            same = one[i]["ids"] == fresh[i]["ids"] and one[i]["probes"] == fresh[i]["probes"]
            close = near(cut[i], fresh[i]) and near(small[i], fresh[i])
            print(f"{name}: {'the same ids and logits' if same else 'DIFFERENT OUTPUT'} in turn as alone (read from "
                  f"{one[i]['resume']}); {'the same ids, logits within 1e-4' if close else 'DIFFERENT'} in passes of 40 "
                  "and of 7", flush=True)
            ok &= same and close
        # where the cache was used: past the picture for a continuation, never across a different picture
        r = [x["resume"] for x in one]
        reuse = r[1] >= len(p_a) - 1 and r[2] >= 98 and r[3] <= 50 and r[4] <= 50
        print(("the cache was used past the picture and not across another one: " if reuse else "WRONG REUSE: ")
              + str(r), flush=True)
        ok &= reuse
        # a picture is read: B in A's place answers differently
        differs = fresh[3]["probes"] != fresh[0]["probes"]
        print("another picture, other logits" if differs else "THE PICTURE CHANGED NOTHING", flush=True)
        ok &= differs

        # refusals: no --vision, a prompt that ends in a picture, more pictures than pad runs
        def answer(args, line):
            p = subprocess.run([a.engine, "--serve", *args], input=line + "\nQUIT\n", capture_output=True, text=True,
                               env={**__import__("os").environ, **env}, timeout=300)
            return [x for x in p.stdout.splitlines() if x.startswith("ERR")]
        plain = [x for x in base if x != "--vision"]
        ids = ",".join(map(str, p_a))
        twice = str(Path(temp) / "aa.sve")
        write_pictures(twice, [A, A], a.width)
        checks = [(plain, f"GENI 4 {files[0]} {ids}", "without --vision"),
                  (base, f"GENI 4 {files[0]} {','.join(map(str, head + [PAD] * 48))}", "cannot end in a picture"),
                  (base, f"GENI 4 {twice} {ids}", "more pictures than the prompt"),
                  (base, f"GENI 4 {files[0]} {','.join(map(str, head + [PAD] * 40 + text(30, 5)))}", "fewer <|image_pad|>")]
        for args, line, want in checks:
            got = answer(args, line)
            hit = any(want in x for x in got)
            print(f"refused ({want}): {'yes' if hit else 'NO'} {got}", flush=True)
            ok &= hit
    print("all passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
