"""metal/tests/preload_test.py - what the engine's start reads into the expert cache from the profile (--preload), on
the tiny random model of serve_protocol_test.py.

    python metal/tests/preload_test.py --model tiny/qwen4exp-moe.gguf [--engine build-metal/bin/strata]

A profile of every expert of the tiny model is written to a temporary file.  The start fills the cache with all of
it, with none of it, with part of it, and as STRATA_PRELOAD says where no option is given; each says so on stderr and
answers a short and a long prompt (whose first pass reads ahead by the profile) with the ids an engine without a
profile gives, also with --elastic and with the cache read back (--elastic-refill all), which takes its list from the
profile whether or not the start filled the cache.
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT = "1,2,3,4,5,6,7,8,9,10"
LONG = ",".join(str(1 + (i * 7) % 120) for i in range(150))


def run(exe, model, extra, prompts, env=None):
    """one engine, the prompts one after another -> (ids per prompt, stderr)"""
    with tempfile.TemporaryFile("w+") as err:
        p = subprocess.Popen([exe, "--serve", "--native", model, "--max-context", "512", *extra], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env={**os.environ, **(env or {})})
        out = []
        for line in p.stdout:
            if line.startswith("READY"):
                break
        for ids in prompts:
            p.stdin.write(f"GEN 8 {ids}\n")
            p.stdin.flush()
            got = []
            for line in p.stdout:
                f = line.split()
                if f[0] == "T":
                    got.append(int(f[1]))
                elif f[0] in ("DONE", "ERR"):
                    break
            out.append(got)
        p.stdin.write("QUIT\n")
        p.stdin.close()
        p.wait(timeout=60)
        err.seek(0)
        return out, err.read()


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

    def filled(err):
        """experts the start read from the profile (0: it said it starts empty, None: neither)"""
        m = re.search(r"filled with the profile's (\d+) most used experts", err)
        return int(m.group(1)) if m else 0 if "starts empty" in err else None

    def read_back(err):
        return sum(int(x) for x in re.findall(r"\((\d+) experts read back\)", err))

    with tempfile.TemporaryDirectory() as tmp:
        prof = str(Path(tmp) / "profile.bin")
        pairs = [(0, 1), (1, 0), (0, 0), (1, 1)]   # every expert of the tiny model (2 layers x 2), most used first
        Path(prof).write_bytes(b"STRP" + struct.pack("<5I", 1, 2, 2, 0, len(pairs))
                               + b"".join(struct.pack("<2H", l, x) for l, x in pairs))
        short_ref, _ = run(a.engine, a.model, [], [PROMPT, PROMPT])
        long_ref, _ = run(a.engine, a.model, ["--prefill", "48"], [LONG, PROMPT])
        cases = [("--preload all", ["--preload", "all"], None, 4), ("--preload 0", ["--preload", "0"], None, 0),
                 ("--preload 0.003 (part of it)", ["--preload", "0.003"], None, "part"), ("--no-preload", ["--no-preload"], None, 0),
                 ("STRATA_PRELOAD=0", [], {"STRATA_PRELOAD": "0"}, 0), ("STRATA_PRELOAD=all", [], {"STRATA_PRELOAD": "all"}, 4),
                 ("STRATA_PRELOAD=0 --preload all", ["--preload", "all"], {"STRATA_PRELOAD": "0"}, 4)]
        for name, extra, env, want in cases:
            got, err = run(a.engine, a.model, ["--expert-profile", prof, *extra], [PROMPT, PROMPT], env)
            n = filled(err)
            ok_n = (n is not None and 0 < n < len(pairs)) if want == "part" else n == want
            check(f"{name}: {want} experts filled at the start, same ids", ok_n and got == short_ref,
                  f"filled {n}, {got} vs {short_ref}")
        for name, extra in (("--preload 0", ["--preload", "0"]), ("--preload all", ["--preload", "all"]),
                            ("--preload 0 --elastic on", ["--preload", "0", "--elastic", "on"]),
                            ("--preload 0 --elastic on --elastic-short 0", ["--preload", "0", "--elastic", "on", "--elastic-short", "0"])):
            got, err = run(a.engine, a.model, ["--prefill", "48", "--expert-profile", prof, *extra], [LONG, PROMPT])
            check(f"long prompt, {name}: same ids", got == long_ref, f"{got} vs {long_ref}")
        # the read-back after a long prompt takes its list from the profile, also when the start did not fill
        for name, extra, want_back in (("--preload 0", ["--preload", "0"], False),
                                       ("--preload 0 --elastic-refill all", ["--preload", "0", "--elastic-refill", "all"], True),
                                       ("--preload all --elastic-refill all", ["--preload", "all", "--elastic-refill", "all"], True)):
            got, err = run(a.engine, a.model, ["--prefill", "48", "--elastic", "on", "--elastic-short", "0", "--verbose",
                                               "--expert-profile", prof, *extra], [LONG, PROMPT, LONG])
            n = read_back(err)
            check(f"long prompts, --elastic, {name}: {'experts read back' if want_back else 'nothing read back'}, same ids",
                  (n > 0) == want_back and got[0] == long_ref[0] and got[2] == long_ref[0], f"read back {n}, {got}")

    print(f"\n{'all passed' if not failed else str(len(failed)) + ' failed'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
