#!/usr/bin/env python3
"""runs.py - the runs of the borrow/KV-hold section, with wall-clock times to find each step in a vm_stat trace.

    .venv/bin/python runs.py turns  --config strata-iq3_s.json --out DIR [--cool 45] [--runs 2] --ab SPEC SPEC -- engine options
    .venv/bin/python runs.py answer --config strata-iq3_s.json --out DIR --tokens N --new M [--cool 45] SPEC ... -- engine options

SPEC: "[VAR=value ...] EXE [options]", as turns.py and bench.py take it.
turns:  the three-turn conversation of refill/turns.py (a 2,000-token document, 525 more tokens of it, a one-line
        question; 64 tokens written each), the specs interleaved A B B A.  Per turn: when it was sent and when its
        tokens came (time.time()), and the engine's lines about shapes and the KV cache during it.
answer: one request per spec, every spec its own engine start: the benchmark's document prompt of N tokens (N=30: a
        short question) and M tokens written with their logits (--force: fed in, M tokens whatever the model would
        write); IDs and logits compared with the first spec's.
        The engine's lines about the KV cache, and when the token after each step came.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "metal" / "tools" / "bench.py").exists())
sys.path.insert(0, str(ROOT / "metal" / "tools"))
from bench import Engine, cool_down, memory_state, per_token, temps, tokenizer  # noqa: E402

KEYS = ("the context has its", "the conversation reaches", "beside the cache", "is not free", "held for the KV",
        "answers are written with", "comes back with", "came back in", "macOS")


def parse(spec, cfg):
    words = spec.split()
    env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    while words and "=" in words[0] and not words[0].startswith(("/", ".", "-")):
        k, v = words.pop(0).split("=", 1)
        env[k] = v
    return words[0], words[1:], env


class Log:
    """the engine's log, read on from where it was last read"""
    def __init__(self, path):
        self.path, self.at = Path(path), 0

    def new(self):
        text = self.path.read_text(errors="replace")
        lines, self.at = text[self.at:].splitlines(), len(text)
        return [ln.replace("strata metal: ", "") for ln in lines if any(k in ln for k in KEYS)]


def hms(t):
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 10)}"


def turns(a, cfg, tok, args):
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
    k = 0
    for i in range(a.runs):
        for spec in (a.ab if i % 2 == 0 else a.ab[::-1]):
            exe, opts, env = parse(spec, cfg)
            t = cool_down(a.cool) if a.cool else None
            print(f"temperature before the run: SoC {t[0]} C, battery {t[1]} C" if t else "no temperature wait", flush=True)
            k += 1
            logp = Path(a.out) / f"turns-{k}-engine.txt"
            e = Engine(exe, args + opts, logp, env)
            log = Log(logp)
            print(f"{spec}: {e.describe()}", flush=True)
            one = {"info": e.info, "temps_before": temps(), "memory_before": e.mem0, "log": log.new(), "turns": []}
            ids = list(turn1)
            for n, extra in enumerate((None, more, last)):
                if extra is not None:
                    ids = ids + [x for x in one["turns"][-1]["ids"] if x not in eos] + extra
                sent = time.time()
                r = e.gen(ids, a.new, " stats=1")
                at = r["token_s"]
                row = {"prompt": r["prompt"], "resume": r["resume"], "read": r["read"], "ids": r["ids"], "token_s": at,
                       "sent": sent, "stamps": r["stamps"], "ttft_s": r["ttft_s"], "tps": r["tps"], "ssd_gb": r["ssd_gb"],
                       "swapouts": r["swapouts"], "per_token": per_token(r), "log": log.new()}
                one["turns"].append(row)
                pt = row["per_token"] or {}
                print(f"  turn {n + 1} sent {hms(sent)}, first token {hms(r['stamps'][0])}: {r['read']} tokens read "
                      f"({r['resume']} reused), t1={at[0]:.2f} t10={at[min(9, len(at) - 1)]:.2f} t50={at[min(49, len(at) - 1)]:.2f} "
                      f"t{len(at)}={at[-1]:.2f}; written {r['tps']:.2f} tok/s, misses/token {pt.get('misses', 0):.1f}, "
                      f"swap-outs {r['swapouts']}", flush=True)
                for ln in row["log"]:
                    print(f"    | {ln}", flush=True)
            one["swapouts"] = e.close()
            one["temps_after"] = temps()
            result["ab"][spec].append(one)
            Path(a.json).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print("\nsame written ids in every run, per turn:",
          [len({tuple(run["turns"][n]["ids"]) for runs in result["ab"].values() for run in runs}) == 1 for n in range(3)])
    result["memory_after"] = memory_state()
    Path(a.json).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")


def answer(a, cfg, tok, args):
    body = tok.encode((ROOT / "docs" / "DETAILS.md").read_text(encoding="utf-8"))
    head = tok.encode("<|im_start|>user\nSummarize the following document.\n\n", parse_special=True)
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n", parse_special=True)
    cut = body[2000:] * (1 + a.tokens // max(1, len(body) - 2000))
    ids = head + cut[:max(1, a.tokens - len(head) - len(tail))] + tail
    # --force: the answer is fed in (teacher forcing: every token still gets its logits), so that it is --new tokens
    # long whatever the model would write
    keys = " probe=1 stats=1" + (" force=" + ".".join(map(str, body[6000:6000 + a.new])) if a.force else "")
    result = {"when": time.strftime("%Y-%m-%d %H:%M"), "args": args, "tokens": len(ids), "forced": a.force, "runs": []}
    first, ok = None, True
    for k, spec in enumerate(a.specs, 1):
        exe, opts, env = parse(spec, cfg)
        t = cool_down(a.cool) if a.cool else None
        print(f"temperature before the run: SoC {t[0]} C, battery {t[1]} C" if t else "no temperature wait", flush=True)
        logp = Path(a.out) / f"answer-{k}-engine.txt"
        e = Engine(exe, args + opts, logp, env)
        log = Log(logp)
        print(f"{spec}: {e.describe()}", flush=True)
        start_log = log.new()
        sent = time.time()
        r = e.gen(ids, a.new, keys)
        lines = log.new()
        swapped = e.close()
        st = r["stamps"]
        gaps = sorted(((st[j + 1] - st[j], j + 2) for j in range(len(st) - 1)), reverse=True)[:3]
        print(f"  sent {hms(sent)}; first token {r['token_s'][0]:.2f} s, last ({len(st)}) {r['token_s'][-1]:.2f} s; "
              f"written {r['tps']:.2f} tok/s; longest waits between tokens: "
              + ", ".join(f"{g:.2f} s before token {j} ({hms(st[j - 1])})" for g, j in gaps) + f"; swap-outs {swapped}", flush=True)
        for ln in lines:
            print(f"    | {ln}", flush=True)
        if first is None:
            first = r
            print("  this is what the others are compared with", flush=True)
        else:
            same = r["ids"] == first["ids"] and r["probes"] == first["probes"]
            ok &= same
            print("  identical IDs and logits" if same else "  DIFFERENT IDs or logits", flush=True)
        result["runs"].append({"spec": spec, "info": e.info, "sent": sent, "stamps": st, "token_s": r["token_s"], "ids": r["ids"],
                               "probes": r["probes"], "tps": r["tps"], "per_token": per_token(r), "swapouts": swapped,
                               "log_start": start_log, "log": lines})
        Path(a.json).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print("all identical" if ok else "DIFFERENCES", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("turns", "answer"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--first", type=int, default=2000)
    ap.add_argument("--second", type=int, default=500)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="answer: feed the answer in (--new tokens of docs/DETAILS.md)")
    ap.add_argument("--cool", type=float, default=45.0)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--ab", nargs="+")
    ap.add_argument("specs", nargs="*")
    a, extra = ap.parse_known_args()
    if "--" in sys.argv:   # engine options after --
        extra = sys.argv[sys.argv.index("--") + 1:]
        a.specs = [s for s in a.specs if s not in extra]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    a.json = a.json or str(Path(a.out) / f"{a.mode}.json")
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    tok = tokenizer(Path(cfg["tokenizer"]))
    args = list(cfg["args"]) + extra
    if a.mode == "turns":
        turns(a, cfg, tok, args)
    else:
        sys.exit(0 if answer(a, cfg, tok, args) else 1)


if __name__ == "__main__":
    main()
