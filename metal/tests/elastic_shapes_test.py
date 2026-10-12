"""metal/tests/elastic_shapes_test.py - `--elastic` on a tiny qwen4exp model: the context changes its shape between
requests and inside them, and every request answers with the ids and logits of an engine without `--elastic`.

    python metal/tests/elastic_shapes_test.py --model tiny/qwen4exp-moe.gguf [--engine build-metal/bin/strata]

Six requests in one engine: a long prompt, a short continuation (read through the cache), a long continuation, a
continuation that one short pass reads, the first prompt again (back to a checkpoint), an unrelated prompt.  Run
with the short shape (`--elastic-short 128`) and without it (`--elastic-short 0`: every long pass takes the cache's
memory and the cache is read back).  STRATA_ELASTIC_TEST_BACKOFF=1 in the environment makes the cache give memory
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
                ("elastic-long", ["--elastic", "on", "--elastic-short", "0"]))
    out, shapes = {}, {}
    with tempfile.TemporaryDirectory(prefix="strata-elastic-") as temp:
        for name, extra in variants:
            log = Path(temp) / (name + ".log")
            e = Engine(a.engine, base + extra, log, env)
            try:
                out[name] = [(r["ids"], r["probes"], r["resume"], r["read"])
                             for r in (e.gen(ids, 10, " probe=1") for ids in requests)]
            finally:
                e.close()
            text = log.read_text(errors="replace")
            shapes[name] = [line.split("has its ")[1].split(" shape")[0] for line in text.splitlines()
                            if "the context has its" in line]
            print(f"{name}: shapes taken: {', '.join(shapes[name]) or 'none'}", flush=True)
    ok = True
    for name in ("elastic", "elastic-long"):
        same = all(x[:2] == y[:2] for x, y in zip(out["plain"], out[name]))
        reuse = [x[2:] for x in out["plain"]] == [x[2:] for x in out[name]]
        print(f"{name}: {'identical ids and logits' if same else 'DIFFERENT OUTPUT'}, "
              f"{'the same reuse' if reuse else 'DIFFERENT REUSE'} over {len(requests)} requests", flush=True)
        ok &= same and reuse
    ok &= "short-prompt" in shapes["elastic"] and "long-prompt" in shapes["elastic"]
    ok &= "long-prompt" in shapes["elastic-long"] and "short-prompt" not in shapes["elastic-long"] and not shapes["plain"]
    print("all passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
