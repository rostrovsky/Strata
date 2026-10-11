"""metal/tools/bench.py - what the Metal engine does on this Mac, measured (docs/APPLE_METAL.md, "Measured").

    .venv/bin/python metal/tools/bench.py --config strata-iq3_s.json [--new 200] [--context 8192] [-- engine options]

Starts the engine of a model config in `--serve` (the server must not be running: two engines do not fit), then
runs three requests and prints what each one took:

  chat      a one-line question, answered with --new tokens
  again     the same conversation continued with a second question (the first turn is reused)
  document  a ~1,600-token text to summarize (the prompt path: passes of --prefill tokens)

For each: prompt tokens/s, output tokens/s over the first 50 tokens and over all of them (a fanless Mac slows down
once it is warm, so the two differ), the expert cache's hit rate and the megabytes read from the SSD.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def tokenizer(tdir: Path):
    import strata_tokenizer as ST
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    return ST.Tokenizer(tokens, (tdir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tdir / "token_type.json").read_text()))


def chat(turns):
    """ChatML, thinking left to the model; ends where the assistant starts."""
    out = ""
    for role, text in turns:
        out += f"<|im_start|>{role}\n{text}<|im_end|>\n"
    return out + "<|im_start|>assistant\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="strata-<model>.json")
    ap.add_argument("--new", type=int, default=200, help="tokens to generate per request")
    ap.add_argument("--context", type=int, default=None, help="override the config's --max-context")
    ap.add_argument("--only", choices=("chat", "again", "document"), action="append")
    ap.add_argument("extra", nargs="*", help="more engine options, after --")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    tok = tokenizer(Path(cfg["tokenizer"]))
    args = list(cfg["args"])
    if a.context:
        args[args.index("--max-context") + 1] = str(a.context)
    log = open(ROOT / "bench-metal.log", "w", encoding="utf-8")
    p = subprocess.Popen([cfg["exe"], "--serve", *args, *a.extra], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=log, text=True, bufsize=1)
    t0 = time.time()
    info = {}
    for line in p.stdout:
        if line.startswith("INFO "):
            info = dict(kv.split("=", 1) for kv in line.split()[1:])
        if line.startswith("READY"):
            break
    else:
        sys.exit("the engine ended before READY (see bench-metal.log)")
    print(f"ready in {time.time() - t0:.1f} s; context {info.get('context')}, expert cache {info.get('expert_slots')} "
          f"experts ({info.get('expert_cache_mib')} MiB), {info.get('expert_io')} reads", flush=True)

    def run(name, prompt):
        ids = tok.encode(prompt, parse_special=True)
        p.stdin.write(f"GEN {a.new} {','.join(map(str, ids))}\n")
        p.stdin.flush()
        start, stamps, out, resume = time.time(), [], [], 0
        for line in p.stdout:
            f = line.split()
            if f[0] == "T":
                stamps.append(time.time())
                out.append(int(f[1]))
            elif f[0] == "RESUME":
                resume = int(f[1])
            elif f[0] == "ERR":
                sys.exit(f"{name}: {line.strip()}")
            elif f[0] == "DONE":
                gen, n, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
                hits, look, fmb = int(f[9]), int(f[10]), float(f[13])
                read = int(f[14])
                first = min(50, len(stamps) - 1)
                r50 = first / (stamps[first] - stamps[0]) if first > 0 else 0.0
                rall = (gen - 1) / (dms / 1000) if gen > 1 and dms > 0 else 0.0
                print(f"{name:9s} prompt {n} tokens ({resume} reused, {read} read in {pms / 1000:.1f} s"
                      f"{f', {1000 * read / pms:.1f} tok/s' if read > 8 and pms > 0 else ''}); "
                      f"output {gen} tokens: {r50:.2f} tok/s over the first {first}, {rall:.2f} tok/s over all; "
                      f"cache hits {100 * hits / max(1, look):.1f}%, {fmb / 1000:.1f} GB read from the SSD; "
                      f"{time.time() - start:.1f} s in all", flush=True)
                return tok.decode(out)
        sys.exit("the engine ended inside a request (see bench-metal.log)")

    only = a.only or ["chat", "again", "document"]
    q1 = "Explain in a short paragraph why the sky is blue."
    answer = ""
    if "chat" in only:
        answer = run("chat", chat([("user", q1)]))
        print("  -> " + " ".join(answer.split())[:160], flush=True)
    if "again" in only and answer:
        # what the engine wrote stays in its context, so the second turn only reads the new question
        run("again", chat([("user", q1)]) + answer.replace("<|im_end|>", "") +
            "<|im_end|>\n<|im_start|>user\nNow say it in one sentence for a child.<|im_end|>\n<|im_start|>assistant\n")
    if "document" in only:
        doc = "\n".join((ROOT / "README.md").read_text(encoding="utf-8").split("\n")[53:140])
        text = run("document", chat([("user", "Summarize the following document in three bullet points.\n\n" + doc)]))
        print("  -> " + " ".join(text.split())[:160], flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=60)


if __name__ == "__main__":
    main()
