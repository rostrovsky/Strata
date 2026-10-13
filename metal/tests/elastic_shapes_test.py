"""metal/tests/elastic_shapes_test.py - `--elastic` on a tiny qwen4exp model: the context changes its shape between
requests and inside them, and every request answers with the ids and logits of an engine without `--elastic`.

    python metal/tests/elastic_shapes_test.py --model tiny/qwen4exp-moe.gguf [--engine build-metal/bin/strata]

Six requests in one engine: a long prompt, a short continuation (read through the cache), a long continuation, a
continuation that one short pass reads, the first prompt again (back to a checkpoint), an unrelated prompt.  Run
with the short shape (`--elastic-short 128`) and without it (`--elastic-short 0`: every long pass takes the cache's
memory); without it also with the cache read back when it returns, whole (`--elastic-refill all`) and in part
(0.003 GiB: two of the tiny model's four experts), where by default it fills as its experts are asked for.
STRATA_ELASTIC_REFILL=all|G in the environment sets that for every variant that does not say it.
One more variant reads with `--prefill 2048` under a plan that is made to say it only has room for passes of 1,024
(STRATA_ELASTIC_PLAN_LONG): the size of a long pass is chosen per prompt, and this one has the room.
STRATA_ELASTIC_TEST_BACKOFF=1 in the environment makes the cache give memory
back before every request, as it does when macOS has swapped; =tight, as it does when macOS has compressed other
programs' memory and little is free.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metal" / "tools"))
from bench import Engine


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", default=str(ROOT / "build-metal" / "bin" / "strata"))
    a = ap.parse_args()
    base = ["--native", a.model, "--max-context", "4096", "--prefill", "256", "--kv", "f16", "--verbose"]
    env = {"STRATA_METAL_INDEXED_QSA": "1"}
    p1 = [1 + (i * 7 + 3) % 120 for i in range(900)]
    p2 = p1 + [5, 9, 11] * 4
    p3 = p2 + [1 + (i * 5) % 100 for i in range(700)]
    p4 = p3 + [1 + (i * 3) % 90 for i in range(90)]
    requests = (p1, p2, p3, p4, p1, [7, 8, 9] * 30)
    variants = (("plain", []), ("elastic", ["--elastic", "on", "--elastic-short", "128"]),
                ("elastic-long", ["--elastic", "on", "--elastic-short", "0"]),
                ("elastic-long-refill", ["--elastic", "on", "--elastic-short", "0", "--elastic-refill", "all"]),
                ("elastic-long-refill-part", ["--elastic", "on", "--elastic-short", "0", "--elastic-refill", "0.003"]),
                # passes of up to 2,048 where the others take 256, with a plan that only had room for 1,024
                ("elastic-long-plan-small", ["--elastic", "on", "--elastic-short", "0", "--prefill", "2048"]))
    plan_small = {"STRATA_ELASTIC_PLAN_LONG": "1024"}
    out, shapes, read_back = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="strata-elastic-") as temp:
        for name, extra in variants:
            log = Path(temp) / (name + ".log")
            e = Engine(a.engine, base + extra, log, {**env, **(plan_small if name == "elastic-long-plan-small" else {})})
            try:
                out[name] = [(r["ids"], r["probes"], r["resume"], r["read"])
                             for r in (e.gen(ids, 10, " probe=1") for ids in requests)]
            finally:
                e.close()
            text = log.read_text(errors="replace")
            shapes[name] = [line.split("has its ")[1].split(" shape")[0] for line in text.splitlines()
                            if "the context has its" in line]
            read_back[name] = [int(line.split("(")[1].split(" experts read back")[0]) for line in text.splitlines()
                               if "the context has its answering shape" in line]
            if name == "elastic-long-plan-small":
                larger = "this prompt is read in passes of 2048 (the plan has room for 1024" in text
            print(f"{name}: shapes taken: {', '.join(shapes[name]) or 'none'}"
                  + (f"; experts read back: {read_back[name]}" if read_back[name] else ""), flush=True)
    ok = True
    for name in [v[0] for v in variants[1:]]:
        same = all(x[:2] == y[:2] for x, y in zip(out["plain"], out[name]))
        reuse = [x[2:] for x in out["plain"]] == [x[2:] for x in out[name]]
        print(f"{name}: {'identical ids and logits' if same else 'DIFFERENT OUTPUT'}, "
              f"{'the same reuse' if reuse else 'DIFFERENT REUSE'} over {len(requests)} requests", flush=True)
        ok &= same and reuse
    ok &= "short-prompt" in shapes["elastic"] and "long-prompt" in shapes["elastic"]
    ok &= "long-prompt" in shapes["elastic-long"] and "short-prompt" not in shapes["elastic-long"] and not shapes["plain"]
    # the read-back happened where it was asked for, and the partial one read less than the whole one
    # (the back-off tests give the cache up before requests too, with lines of their own: only the sums are compared)
    whole, part = sum(read_back["elastic-long-refill"]), sum(read_back["elastic-long-refill-part"])
    if 0 < part < whole:
        print(f"read back after long prompts: {whole} experts with --elastic-refill all, {part} with 0.003", flush=True)
    else:
        print(f"READ-BACK: {whole} experts with --elastic-refill all, {part} with 0.003 (expected 0 < part < whole)", flush=True)
        ok = False
    print("a plan with room for passes of 1,024: " + ("the prompt was read in passes of 2,048" if larger else
                                                      "THE LARGER PASSES WERE NOT TAKEN"), flush=True)
    ok &= larger
    print("all passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
