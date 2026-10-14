#!/usr/bin/env python3
"""starts.py - the first two requests after an engine start, timed from the launch, per engine setting (A B B A).

    .venv/bin/python starts.py --config strata-iq3_s.json --out DIR --name NAME [--cases chat,doc500,doc2000]
        [--new 100] [--new2 64] [--cool 45] [--runs 2] --ab "STRATA_PRELOAD=all EXE" "STRATA_PRELOAD=0 EXE" -- engine options

For each case, every setting in the order A B .. B A (the list, then the list reversed, --runs times): the SoC is
cooled to --cool, the engine started, and as soon as it says READY the case's prompt is sent (--new tokens written),
then a second request: the answer and a one-line question (--new2 tokens).  Every time is from the launch of the
engine (bench.py's Engine.t0), which is what a request that finds the server's engine stopped waits for.

  chat     a 33-token question (read in passes the cache serves: the engine cuts a prompt before its last message
           opens and before its last token, so no stretch reaches the 33 tokens of a long pass)
  doc500   a 500-token document to summarise (one pass beside the cache: it copies the experts the cache holds)
  doc2000  a 2,000-token document (a long prompt: the cache is given up for it and comes back empty)

Writes DIR/NAME.json and one engine log per start (DIR/NAME-<case>-<n>.txt).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "metal" / "tools" / "bench.py").exists())
sys.path.insert(0, str(ROOT / "metal" / "tools"))
from bench import Engine, chat, cool_down, memory_state, per_token, temps, tokenizer  # noqa: E402

MARKS = (1, 2, 10, 25, 50, 64, 100)


def from_log(path):
    """what the engine said about its start"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    if m := re.search(r"weights loaded in ([\d.]+) s", text):
        out["weights_s"] = float(m.group(1))
    if m := re.search(r"expert cache: (\d+) of (\d+) experts \(\d+%\), ([\d.]+) GiB", text):
        out["cache_experts"], out["cache_gib"] = int(m.group(1)), float(m.group(3))
    if m := re.search(r"filled with the profile's (\d+) most used experts in ([\d.]+) s \(([\d.]+) GiB", text):
        out["filled_experts"], out["fill_s"], out["filled_gib"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    elif "starts empty" in text:
        out["filled_experts"], out["fill_s"], out["filled_gib"] = 0, 0.0, 0.0
    return out


def busy(log):
    """waits while another engine runs on this Mac (other sessions share it): two do not fit"""
    while subprocess.run(["pgrep", "-x", "strata|strata-reference"], capture_output=True).returncode == 0:
        if log:
            print(f"[{time.strftime('%H:%M:%S')}] another engine is running; waiting", flush=True)
            log = False
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cases", default="chat,doc500,doc2000")
    ap.add_argument("--new", type=int, default=100)
    ap.add_argument("--new2", type=int, default=64)
    ap.add_argument("--cool", type=float, default=45.0)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--ab", nargs="+", required=True)
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    tok = tokenizer(Path(cfg["tokenizer"]))
    args = list(cfg["args"]) + a.extra
    enc = lambda text: tok.encode(text, parse_special=True)
    body = tok.encode((ROOT / "docs" / "DETAILS.md").read_text(encoding="utf-8"))
    head = enc("<|im_start|>user\nSummarize the following document.\n\n")
    tail = enc("<|im_end|>\n<|im_start|>assistant\n")
    doc = lambda n: head + body[:n - len(head) - len(tail)] + tail
    prompts = {"chat": enc(chat([("user", "Explain in a short paragraph why the sky is blue, and why the sun looks red "
                                          "when it sets in the evening.")])),
               "doc500": doc(500), "doc2000": doc(2000)}
    last = enc("<|im_end|>\n<|im_start|>user\nNow say all of it in one sentence.<|im_end|>\n<|im_start|>assistant\n")
    eos = set(enc("<|im_end|>") + enc("<|endoftext|>"))
    result = {"when": time.strftime("%Y-%m-%d %H:%M"), "args": args, "memory_before": memory_state(),
              "cases": {c: {x: [] for x in a.ab} for c in a.cases.split(",")}}
    path = out / f"{a.name}.json"
    n = 0
    for case in a.cases.split(","):
        for i in range(a.runs):
            for spec in (a.ab if i % 2 == 0 else a.ab[::-1]):
                words = spec.split()
                env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
                while words and "=" in words[0] and not words[0].startswith(("/", ".")):
                    k, v = words.pop(0).split("=", 1)
                    env[k] = v
                busy(log=True)
                t = cool_down(a.cool) if a.cool else None
                busy(log=False)
                n += 1
                log = out / f"{a.name}-{case}-{n}.txt"
                print(f"[{time.strftime('%H:%M:%S')}] {case}, {spec}: SoC {t[0] if t else '?'} C", flush=True)
                e = Engine(words[0], args + words[1:], str(log), env)
                one = {"n": n, "started": time.strftime("%H:%M:%S", time.localtime(e.t0)), "info": e.info,
                       "ready_s": round(e.ready_s, 3), "temps_before": t, "memory_before": e.mem0, "requests": []}
                ids = list(prompts[case])
                for r_i, new in enumerate((a.new, a.new2)):
                    if r_i:
                        ids = ids + [x for x in one["requests"][-1]["ids"] if x not in eos] + last
                    r = e.gen(ids, new, " stats=1")
                    at = [round(r["sent_s"] + x, 3) for x in r["token_s"]]   # from the launch
                    row = {"prompt": r["prompt"], "resume": r["resume"], "read": r["read"], "ids": r["ids"],
                           "sent_s": r["sent_s"], "token_s": r["token_s"], "from_launch_s": at, "ttft_s": r["ttft_s"],
                           "tps": r["tps"], "hit_rate": r["hit_rate"], "ssd_gb": r["ssd_gb"], "swapouts": r["swapouts"],
                           "per_token": per_token(r), "stats": r["stats"]}
                    one["requests"].append(row)
                    pt = row["per_token"] or {}
                    print(f"  request {r_i + 1}: {r['read']} read ({r['resume']} reused), sent at {r['sent_s']:.2f} s; "
                          "from the launch " + " ".join(f"t{k}={at[k - 1]:.2f}" for k in MARKS if k <= len(at))
                          + f"; written {r['tps']:.2f} tok/s, misses/token {pt.get('misses', 0):.1f}, SSD "
                            f"{r['ssd_gb']:.2f} GB, swap-outs {r['swapouts']}", flush=True)
                one["swapouts"] = e.close()
                one["temps_after"] = temps()
                one["log"] = from_log(log)
                lg = one["log"]
                print(f"  ready {one['ready_s']:.2f} s (weights {lg.get('weights_s')} s, fill {lg.get('fill_s')} s, "
                      f"{lg.get('filled_gib')} GiB); cache {one['info'].get('expert_cache_mib')} MiB; "
                      f"engine swap-outs {one['swapouts']}", flush=True)
                result["cases"][case][spec].append(one)
                path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    result["memory_after"] = memory_state()
    print("\nsame ids per case and request:",
          {c: [len({tuple(run["requests"][k]["ids"]) for runs in specs.values() for run in runs}) == 1 for k in range(2)]
           for c, specs in result["cases"].items()})
    for case, specs in result["cases"].items():
        for spec, runs in specs.items():
            for k in range(2):
                cells = []
                for m in (1, 50):
                    v = [run["requests"][k]["from_launch_s"][m - 1] for run in runs if len(run["requests"][k]["from_launch_s"]) >= m]
                    cells.append(f"t{m} " + ", ".join(f"{x:.2f}" for x in v))
                print(f"  {case:8s} {spec}: request {k + 1}: " + "; ".join(cells)
                      + (f"; ready {', '.join(str(r['ready_s']) for r in runs)}" if k == 0 else ""))
    path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
