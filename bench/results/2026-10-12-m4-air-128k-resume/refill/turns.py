#!/usr/bin/env python3
"""turns.py - a conversation of three turns after a long prompt, per engine setting, interleaved (A B B A).

    .venv/bin/python turns.py --config strata-iq3_s.json --json OUT.json --log OUT.log
        [--first 2000] [--second 500] [--new 64] [--cool 45] [--runs 2] --ab "EXE --elastic-refill all" "EXE" -- engine options

Turn 1: a document of --first tokens to summarise (the context takes its long-prompt shape and the expert cache
is given up and comes back).  Turn 2: the answer and --second more tokens of the document (one pass beside the
cache).  Turn 3: a one-line question.  Each turn writes --new tokens.  What a turn finds in the cache is what the
turns before left there, which is what a read-back that is not whole changes.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "metal" / "tools" / "bench.py").exists())
sys.path.insert(0, str(ROOT / "metal" / "tools"))
from bench import Engine, cool_down, memory_state, per_token, temps, tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--first", type=int, default=2000)
    ap.add_argument("--second", type=int, default=500)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--cool", type=float, default=45.0)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--ab", nargs="+", required=True)
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    tok = tokenizer(Path(cfg["tokenizer"]))
    args = list(cfg["args"]) + a.extra
    body = tok.encode((ROOT / "docs" / "DETAILS.md").read_text(encoding="utf-8"))
    enc = lambda text: tok.encode(text, parse_special=True)
    head = enc("<|im_start|>user\nSummarize the following document.\n\n")
    tail = enc("<|im_end|>\n<|im_start|>assistant\n")
    turn1 = head + body[:a.first - len(head) - len(tail)] + tail
    more = (enc("<|im_end|>\n<|im_start|>user\nThe document goes on. Add what is new in this part.\n\n")
            + body[a.first:a.first + a.second] + tail)
    last = enc("<|im_end|>\n<|im_start|>user\nNow say all of it in one sentence.<|im_end|>\n<|im_start|>assistant\n")
    eos = set(enc("<|im_end|>") + enc("<|endoftext|>"))
    result = {"when": time.strftime("%Y-%m-%d %H:%M"), "args": args, "memory_before": memory_state(), "ab": {x: [] for x in a.ab}}
    marks = (1, 2, 10, 25, 50, 64, 100, 200)
    for i in range(a.runs):
        for spec in (a.ab if i % 2 == 0 else a.ab[::-1]):
            words = spec.split()
            env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
            while words and "=" in words[0] and not words[0].startswith(("/", ".")):
                k, v = words.pop(0).split("=", 1)
                env[k] = v
            t = cool_down(a.cool) if a.cool else None
            print(f"temperature before the run: SoC {t[0]} C, battery {t[1]} C" if t else "no temperature wait", flush=True)
            e = Engine(words[0], args + words[1:], a.log, env)
            print(f"{spec}: {e.describe()}", flush=True)
            one = {"info": e.info, "temps_before": temps(), "memory_before": e.mem0, "turns": []}
            ids = list(turn1)
            for n, extra in enumerate((None, more, last)):
                if extra is not None:
                    ids = ids + [x for x in one["turns"][-1]["ids"] if x not in eos] + extra
                r = e.gen(ids, a.new, " stats=1")
                at = r["token_s"]
                row = {"prompt": r["prompt"], "resume": r["resume"], "read": r["read"], "ids": r["ids"], "token_s": at,
                       "ttft_s": r["ttft_s"], "tps": r["tps"], "hit_rate": r["hit_rate"], "ssd_gb": r["ssd_gb"],
                       "swapouts": r["swapouts"], "per_token": per_token(r), "stats": r["stats"]}
                one["turns"].append(row)
                pt = row["per_token"] or {}
                print(f"  turn {n + 1}: {r['read']} tokens read ({r['resume']} reused), "
                      + " ".join(f"t{k}={at[k - 1]:.2f}" for k in marks if k <= len(at))
                      + f"; written {r['tps']:.2f} tok/s, misses/token {pt.get('misses', 0):.1f}, SSD "
                        f"{r['ssd_gb']:.1f} GB, swap-outs {r['swapouts']}", flush=True)
            one["swapouts"] = e.close()
            one["temps_after"] = temps()
            result["ab"][spec].append(one)
    print("\nsame written ids in every run, per turn:",
          [len({tuple(run["turns"][n]["ids"]) for runs in result["ab"].values() for run in runs}) == 1 for n in range(3)])
    for spec, runs in result["ab"].items():
        for n in range(3):
            first = statistics.mean(r["turns"][n]["ttft_s"] for r in runs)
            t50 = statistics.mean(r["turns"][n]["token_s"][min(50, len(r["turns"][n]["token_s"])) - 1] for r in runs)
            tps = statistics.mean(r["turns"][n]["tps"] for r in runs)
            print(f"  {spec}: turn {n + 1}: first token {first:.2f} s, token 50 {t50:.2f} s, {tps:.2f} tok/s "
                  f"(mean of {len(runs)}); swap-outs {sum(r['swapouts'] for r in runs)}")
    result["memory_after"] = memory_state()
    Path(a.json).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
