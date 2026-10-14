"""metal/tests/kv_hold_test.py - `--elastic` on a tiny qwen4exp model: steps of the KV cache taken while the expert
cache is there are paid by the memory held for them, and the answers are those of an engine without `--elastic`.

    python metal/tests/kv_hold_test.py --model tiny/qwen4exp-moe.gguf [--engine build-metal/bin/strata]

The KV cache takes 256 cells at a time (STRATA_KV_STEP=256).  Requests in one engine:
  1. a short question with a long answer: the question needs the second step at once (the first is mapped from the
     start), paid by the hold; the answer crosses more steps, which the cache pays (it is made anew, smaller);
  2. a long continuation (the context takes its long-prompt shape; when the cache comes back, the hold is taken again);
  3. a short continuation with an answer that goes past the 2,048 tokens request 2 made room for: that step is paid
     by the hold again.
Run with the hold (the default) and without it (STRATA_KV_HOLD=0); ids and logits must be the plain engine's.
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
    env = {"STRATA_METAL_INDEXED_QSA": "1", "STRATA_KV_STEP": "256"}
    q1 = [1 + (i * 7 + 3) % 120 for i in range(20)]
    variants = (("plain", [], {}), ("hold", ["--elastic", "on", "--elastic-short", "128"], {}),
                ("no-hold", ["--elastic", "on", "--elastic-short", "128"], {"STRATA_KV_HOLD": "0"}))
    out, lines = {}, {}
    with tempfile.TemporaryDirectory(prefix="strata-kv-hold-") as temp:
        for name, extra, more_env in variants:
            log = Path(temp) / (name + ".log")
            e = Engine(a.engine, base + extra, log, {**env, **more_env})
            try:
                got = []
                r = e.gen(q1, 600, " probe=1")
                got.append(r)
                ids = q1 + r["ids"] + [1 + (i * 5) % 100 for i in range(400)]
                r = e.gen(ids, 8, " probe=1")
                got.append(r)
                ids = ids + r["ids"] + [5, 9, 11] * 4
                r = e.gen(ids, 2100, " probe=1")   # (request 2 made room for 2,048 tokens past itself)
                got.append(r)
            finally:
                e.close()
            out[name] = [(r["ids"], r["probes"], r["resume"], r["read"]) for r in got]
            lines[name] = [ln.split("strata metal: ")[-1] for ln in log.read_text(errors="replace").splitlines()
                           if "the conversation reaches" in ln or "held for the KV" in ln or "the context has its" in ln]
            print(f"{name}:", flush=True)
            for ln in lines[name]:
                print(f"    {ln}", flush=True)
    ok = True
    for name in ("hold", "no-hold"):
        same = all(x[:2] == y[:2] for x, y in zip(out["plain"], out[name]))
        reuse = [x[2:] for x in out["plain"]] == [x[2:] for x in out[name]]
        print(f"{name}: {'identical ids and logits' if same else 'DIFFERENT OUTPUT'}, "
              f"{'the same reuse' if reuse else 'DIFFERENT REUSE'} over {len(out[name])} requests", flush=True)
        ok &= same and reuse
    steps = [ln for ln in lines["hold"] if "the conversation reaches" in ln]
    held = [ln for ln in steps if "were held for it" in ln]
    remade = [ln for ln in steps if "fills again" in ln or "read back" in ln]
    # the first step of request 1 and of request 3 are paid by the hold (taken again when the cache came back after
    # request 2); the ones in between by what the cache does not use (the tiny model's experts all fit in its cache,
    # so the cache itself is never made smaller for the KV cache here)
    print(f"with the hold: {len(steps)} steps, {len(held)} paid by the hold, {len(remade)} by the cache made anew", flush=True)
    ok &= len(held) >= 2 and len(steps) > len(held) and any("held for the KV" in ln for ln in lines["hold"])
    ok &= not any("were held for it" in ln or "held for the KV" in ln for ln in lines["no-hold"])
    print("all passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
