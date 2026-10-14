"""metal/tests/serve_protocol_test.py - the Metal engine's `--serve` protocol, on a tiny random model (no GPU-hours,
no 55 GB file).

    python metal/tests/serve_protocol_test.py --model tiny/qwen4exp-moe.gguf [--engine build-metal/bin/strata]

The tiny model comes from llama.cpp's own test tool (any random qwen4exp works; its answers are noise, which is
fine: every check here compares the engine with itself):

    cmake -S llama.cpp -B llama.cpp/build -DLLAMA_BUILD_TESTS=ON && cmake --build llama.cpp/build --target test-llama-archs
    llama.cpp/build/bin/test-llama-archs --arch '^qwen4exp$' --out tiny

What is checked: READY and INFO; the same prompt gives the same ids with the expert cache on, off (llama.cpp's own
path over the mapped file), with direct reads, with every prompt chunk size, with prefetch and (a long prompt) with
read-ahead, and with --elastic (the context changing its shape for the long prompt and back); a prompt asked again
and a prompt that continues the last one reuse the context (RESUME) and give the ids a fresh engine gives; STOP ends a request;
bad requests are answered with ERR and the engine goes on; every engine ends with exit code 0 when it is ended as
serve/server.py ends it (QUIT, and its stdin closed at once: that aborted every time until 2026-10-14).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT = "1,2,3,4,5,6,7,8,9,10"
EXITS = []   # every engine's exit code (Engine.close)


class Engine:
    def __init__(self, exe, model, *extra):
        self.p = subprocess.Popen([exe, "--serve", "--native", model, "--max-context", "256", *extra],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                  bufsize=1)
        self.info = {}
        for line in self.p.stdout:
            if line.startswith("INFO "):
                self.info = dict(kv.split("=", 1) for kv in line.split()[1:])
            if line.startswith("READY"):
                self.ready = line.split()
                return
        raise SystemExit("the engine ended before READY")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def gen(self, max_new, ids, keys="", stop_after=None):
        """-> (tokens, resume, DONE fields or None, ERR text or None)"""
        self.send(f"GEN {max_new}{keys} {ids}")
        toks, resume = [], None
        for line in self.p.stdout:
            f = line.split()
            if f[0] == "T":
                toks.append(int(f[1]))
                if stop_after is not None and len(toks) == stop_after:
                    self.send("STOP")
            elif f[0] == "RESUME":
                resume = int(f[1])
            elif f[0] == "DONE":
                return toks, resume, f, None
            elif f[0] == "ERR":
                return toks, resume, None, line[4:].strip()
        raise SystemExit("the engine ended inside a request")

    def close(self):
        """QUIT and end of input, as serve/server.py ends the engine -> its exit code (None: killed)"""
        try:
            self.send("QUIT")
            self.p.stdin.close()
            code = self.p.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()
            code = None
        EXITS.append(code)
        return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", default=str(ROOT / "build-metal" / "bin" / "strata"))
    a = ap.parse_args()
    failed = []

    def check(name, ok, detail=""):
        print(("ok    " if ok else "FAIL  ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
        if not ok:
            failed.append(name)

    e = Engine(a.engine, a.model)
    check("READY <context> stop", e.ready[1] == "256" and "stop" in e.ready[2:], str(e.ready))
    check("INFO names the backend and the cache", e.info.get("backend") == "metal" and int(e.info.get("expert_slots", 0)) > 0,
          str(e.info))
    ref, resume, done, err = e.gen(10, PROMPT)
    check("a request answers", err is None and len(ref) == 10 and resume == 0 and done[5] == "length", str((ref, done, err)))
    check("DONE has the 15 fields serve/server.py reads", done is not None and len(done) >= 16, str(done))

    again, resume, done, _ = e.gen(10, PROMPT)
    check("the same prompt again: reused, same ids", again == ref and resume == 9, f"{again} resume {resume}")
    cont = PROMPT + "," + ",".join(str(t) for t in ref[:4]) + ",11,12"
    ext, resume, _, _ = e.gen(6, cont)
    fresh = Engine(a.engine, a.model)
    want, _, _, _ = fresh.gen(6, cont)
    fresh.close()
    check("a prompt that continues the last one: reused, same ids as a fresh engine", ext == want and resume and resume >= 9,
          f"{ext} vs {want}, resume {resume}")
    # (a prompt this far from the first one: a random tiny model answers near-identical prompts identically)
    other, resume, _, _ = e.gen(10, "11,12,13,14,15,16,17,18,19,20,21,22")
    back, resume2, _, _ = e.gen(10, PROMPT)
    check("a different prompt, then the first again: same ids", back == ref and other != ref, f"{back} resume {resume2}")

    toks, _, done, _ = e.gen(200, PROMPT, stop_after=3)
    check("STOP ends a request", done is not None and done[5] == "cancel" and len(toks) < 200, str(done))
    after, _, _, _ = e.gen(10, PROMPT)
    check("the request after a STOP is right", after == ref, str(after))

    _, _, _, err = e.gen(300, "1,2,3")
    check("too long for the context: ERR", err is not None and "exceeds the context" in err, str(err))
    _, _, _, err = e.gen(4, "1,2,99999999")
    check("an id outside the vocabulary: ERR", err is not None and "vocabulary" in err, str(err))
    e.send("NOPE 1")
    line = e.p.stdout.readline()
    check("an unknown command: ERR", line.startswith("ERR"), line)
    sampled, _, _, err = e.gen(10, PROMPT, " temperature=0.8 top_p=0.9 top_k=20 seed=7")
    sampled2, _, _, _ = e.gen(10, PROMPT, " temperature=0.8 top_p=0.9 top_k=20 seed=7")
    check("sampling keys: accepted, and a seed repeats", err is None and sampled == sampled2, f"{sampled} {sampled2}")
    last, _, _, _ = e.gen(10, PROMPT)
    check("still right after all of it", last == ref, str(last))
    e.close()

    # a prompt long enough for the long-prompt path (more than 32 new tokens), in passes of 48
    long_prompt = ",".join(str(1 + (i * 7) % 120) for i in range(150))
    x = Engine(a.engine, a.model, "--prefill", "48")
    long_ref, _, _, err = x.gen(10, long_prompt)
    x.close()
    check("a long prompt answers", err is None and len(long_ref) == 10, f"{long_ref} {err}")

    for extra in (["--expert-cache", "0"], ["--expert-io", "direct"], ["--chunk", "1"], ["--chunk", "7"],
                  ["--io-threads", "1"], ["--no-preload"], ["--prefetch", "2"], ["--prefetch", "4", "--prefetch-depth", "2"],
                  ["--elastic", "on"]):
        x = Engine(a.engine, a.model, *extra)
        got, _, _, err = x.gen(10, PROMPT)
        x.close()
        check(f"{' '.join(extra)}: same ids", got == ref, f"{got} {err}")
    # (--elastic: the long prompt is read beside the cache; --elastic-short 0: in the cache's place, which is read back)
    for extra in (["--prefill-ahead", "on"], ["--prefill-ahead", "on", "--io-threads", "2"], ["--prefetch", "2"],
                  ["--elastic", "on"], ["--elastic", "on", "--elastic-short", "0"]):
        x = Engine(a.engine, a.model, "--prefill", "48", *extra)
        got, _, _, err = x.gen(10, long_prompt)
        again, _, _, _ = x.gen(10, long_prompt.replace("1,", "2,", 1))
        back, _, _, _ = x.gen(10, long_prompt)
        x.close()
        check(f"long prompt, {' '.join(extra)}: same ids", got == long_ref and back == long_ref, f"{got} {back} {err}")

    check(f"QUIT and end of input: all {len(EXITS)} engines exit with 0", all(c == 0 for c in EXITS), str(EXITS))

    print(f"\n{'all passed' if not failed else str(len(failed)) + ' failed'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    watchdog = threading.Timer(600, lambda: (print("timed out"), sys.exit(2)))
    watchdog.daemon = True
    watchdog.start()
    main()
