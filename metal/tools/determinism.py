"""metal/tools/determinism.py - does the engine give the same logits for the same long prompt, run after run?

    python metal/tools/determinism.py --config strata-<model>.json --out DIR NAME[:engine options] ...

Every NAME starts the engine afresh (the config's options, then its own), sends the benchmark's long prompt
(--tokens of docs/DETAILS.md) with logits reported for --new tokens, then the same prompt again (only its last
token is read then: the answer alone is computed a second time).  The first run's first request is what the
others are compared with: identical token IDs and reported logits, or the first position where they differ and
how close the two best tokens were there.  Wrap real-model runs in metal/tools/swapwatch.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=10000)
    ap.add_argument("--new", type=int, default=24)
    ap.add_argument("--once", action="store_true", help="do not ask for the answer a second time")
    ap.add_argument("runs", nargs="+", help="NAME or 'NAME:--option value ...' (NAME=value words before the options: environment; EXE=path: another build)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    tok = bench.tokenizer(Path(cfg["tokenizer"]))
    body = tok.encode((bench.ROOT / "docs" / "DETAILS.md").read_text(encoding="utf-8"))
    head = tok.encode("<|im_start|>user\nSummarize the following document.\n\n", parse_special=True)
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n", parse_special=True)
    cut = body[2000:] * (1 + a.tokens // max(1, len(body) - 2000))   # the benchmark's third prompt
    ids = head + cut[:max(1, a.tokens - len(head) - len(tail))] + tail
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    first = None
    ok = True
    for spec in a.runs:
        name, _, rest = spec.partition(":")
        words = rest.split()
        env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
        while words and "=" in words[0] and not words[0].startswith("-"):
            k, v = words.pop(0).split("=", 1)
            env[k] = v
        exe = env.pop("EXE", cfg["exe"])   # EXE=path: another build of the engine
        e = bench.Engine(exe, list(cfg["args"]) + words, out / f"{name}-engine.txt", env)
        print(f"{name}: {e.describe()}", flush=True)
        try:
            got = [e.gen(ids, a.new, " probe=1") for _ in range(1 if a.once else 2)]
        finally:
            swapped = e.close()
        for i, r in enumerate(got):
            label = f"{name}, {'prompt read' if i == 0 else 'answer again'}"
            if first is None:
                first = r
                print(f"{label}: {len(ids)} tokens, {len(r['ids'])} written; this is what the others are compared with", flush=True)
                continue
            same_ids, same_probes = r["ids"] == first["ids"], r["probes"] == first["probes"]
            if same_ids and same_probes:
                print(f"{label}: identical IDs and logits", flush=True)
                continue
            ok = False
            at = next((j for j, (x, y) in enumerate(zip(r["probes"], first["probes"])) if x != y), min(len(r["probes"]), len(first["probes"])))
            x, y = (r["probes"][at], first["probes"][at]) if at < min(len(r["probes"]), len(first["probes"])) else (None, None)
            print(f"{label}: DIFFERENT from token {at} on ({'same' if same_ids else 'different'} IDs)\n    here  {x}\n    first {y}", flush=True)
        (out / f"{name}.json").write_text(json.dumps({"spec": spec, "swapouts": swapped, "requests": [
            {"ids": r["ids"], "probes": r["probes"], "resume": r["resume"], "token_s": r["token_s"]} for r in got]}),
            encoding="utf-8")
    print("all identical" if ok else "DIFFERENCES", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
