"""metal/tools/bench.py - what the Metal engine does on this Mac, measured (docs/APPLE_METAL.md, "Measured").

    .venv/bin/python metal/tools/bench.py --config strata-iq3_s.json [--context 8192] [mode] [-- engine options]

Starts the engine of a model config in `--serve` (the server must not be running: two engines do not fit), runs one
mode and prints what it measured.  Every mode reports the swap-outs macOS did meanwhile (vm_stat): the engine is meant
to cause none.

  (no mode)            three requests: a one-line chat, its next turn, a ~1,600-token document to summarize
  --gate record        the gate suite (chat, code, documents, several turns, other languages), greedy, --new tokens
                       each: writes the ids and the three largest logits behind every token to --ref
  --gate check         the same suite against --ref: every id must match.  Where one differs the request is run
                       again with the reference's tokens fed in (teacher forcing), and the difference is accepted
                       only if both engines had the two tokens within --tie logits of each other at that place
  --prefill 32,256,..  reads prompts of N tokens and writes one token: tokens/s and the time to the first token
  --prefill-new N      also measures decode at each prompt depth
  --needle             retrieves five exact records spread across each prompt; requires --prefill-new >= 64
  --decode             the suite's short requests one after another: tokens/s, hit rate, and where a token's time goes
  --sustained S        --decode repeated for S seconds: tokens/s per 30 s (a fanless Mac slows down once it is warm)
  --ab EXE [EXE ...]   --decode (or --prefill) with each engine in turn, --runs times, medians

`--engine EXE` runs another build than the config's; `--json FILE` writes the numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
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


# ---- the Mac ---------------------------------------------------------------------------------------------------
def vm_stat():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    v = {m.group(1): int(m.group(2)) for m in re.finditer(r'"?([A-Za-z -]+)"?:\s+(\d+)\.', out)}
    return page, v


def swapouts():
    return vm_stat()[1].get("Swapouts", 0)


def memory_state():
    page, v = vm_stat()
    gib = page / 2**30
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage", "kern.memorystatus_vm_pressure_level"],
                        capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", sw)
    pressure = re.search(r"(?m)^(\d+)\s*$", sw)
    return {"free_gib": round(v.get("Pages free", 0) * gib, 2),
            "file_cache_gib": round(v.get("File-backed pages", 0) * gib, 2),
            "apps_gib": round((v.get("Anonymous pages", 0) + v.get("Pages occupied by compressor", 0)) * gib, 2),
            "wired_gib": round(v.get("Pages wired down", 0) * gib, 2),
            "swap_used_gib": round(float(m.group(1)) / 1024, 2) if m else None,
            "pressure_level": int(pressure.group(1)) if pressure else None}


def temps():
    """(hottest SoC sensor, battery) in Celsius from metal/tools/temps.m, or None if the helper is not built:
    clang -O2 -fobjc-arc -framework Foundation -framework IOKit metal/tools/temps.m -o build-metal/bench/temps"""
    exe = ROOT / "build-metal" / "bench" / "temps"
    if not exe.exists():
        return None
    die, bat = [], []
    for line in subprocess.run([str(exe)], capture_output=True, text=True).stdout.splitlines():
        name, _, c = line.rpartition(" ")
        (die if "tdie" in name else bat if "battery" in name else []).append(float(c))
    return (max(die) if die else None, max(bat) if bat else None)


def power_counters():
    """(accumulated mW, samples) of the whole Mac's power draw, from the battery's telemetry (no sudo): the average
    watts between two readings is the difference of the first over the difference of the second, / 1000"""
    out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True).stdout
    a = re.search(r'"AccumulatedSystemLoad"=(\d+)', out)
    n = re.search(r'"SystemLoadAccumulatorCount"=(\d+)', out)
    return (int(a.group(1)), int(n.group(1))) if a and n else None


def watts(p0, p1):
    if not p0 or not p1 or p1[1] <= p0[1]:
        return None
    return (p1[0] - p0[0]) / (p1[1] - p0[1]) / 1000.0


def cool_down(limit, timeout=300.0):
    """waits until the hottest SoC sensor is below `limit` Celsius (the same start for every run), or has stopped
    falling: with other programs at work the Mac does not get as cool as when it is left alone"""
    t0 = time.time()
    seen = []
    while True:
        t = temps()
        if t is None or t[0] is None or t[0] <= limit or time.time() - t0 > timeout:
            return t
        seen.append(t[0])
        if len(seen) >= 13 and min(seen[-13:-6]) - min(seen[-6:]) < 0.3:    # no cooler than half a minute ago
            return t
        time.sleep(5)


def thermal():
    """macOS's own word on heat, without sudo: a CPU speed limit below 100 means it is holding the chip back."""
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    return int(m.group(1)) if m else None


# ---- the engine ------------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, exe, args, log_path, env=None):
        self.exe = exe
        self.log = open(log_path, "w", encoding="utf-8")
        self.swap0 = swapouts()
        self.mem0 = memory_state()
        t0 = time.time()
        import os
        self.p = subprocess.Popen([exe, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, bufsize=1, env={**os.environ, **(env or {})})
        self.info = {}
        for line in self.p.stdout:
            if line.startswith("INFO "):
                self.info = dict(kv.split("=", 1) for kv in line.split()[1:])
            if line.startswith("READY"):
                break
        else:
            sys.exit(f"the engine ended before READY (see {log_path})")
        self.ready_s = time.time() - t0

    def describe(self):
        return (f"ready in {self.ready_s:.1f} s; context {self.info.get('context')}, expert cache "
                f"{self.info.get('expert_slots')} experts ({self.info.get('expert_cache_mib')} MiB), "
                f"{self.info.get('expert_io')} reads")

    def gen(self, ids, max_new, keys="", progress=False, embeddings=None):
        """One request.  -> dict: ids, probes, stats, the DONE figures, wall times.  `embeddings`: a file of
        strata-vision records for the prompt's pictures (a GENI request)."""
        s0 = swapouts()
        p0 = power_counters()
        head = f"GENI {max_new}{keys} {embeddings}" if embeddings else f"GEN {max_new}{keys}"
        self.p.stdin.write(f"{head} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        start, stamps, out, probes, stats, resume = time.time(), [], [], [], {}, 0
        last_progress = start
        progress_rows = []
        for line in self.p.stdout:
            f = line.split()
            if f[0] == "T":
                stamps.append(time.time())
                out.append(int(f[1]))
            elif f[0] == "P":
                # chosen, fed, logit of fed, then (id, logit) x3
                probes.append([int(f[1]), int(f[2]), float(f[3]), int(f[4]), float(f[5]), int(f[6]), float(f[7]),
                               int(f[8]), float(f[9])])
            elif f[0] == "STAT":
                stats[f[1]] = {k: float(v) for k, v in (kv.split("=", 1) for kv in f[2:])}
            elif f[0] == "RESUME":
                resume = int(f[1])
            elif f[0] == "PP":
                row = {"tokens": int(f[1]), "total": int(f[2]), "elapsed_s": float(f[3]) / 1000}
                progress_rows.append(row)
                if progress and time.time() - last_progress >= 30:
                    row["memory"] = memory_state()
                    row["swapouts"] = swapouts() - s0
                    print(f"  read {row['tokens']}/{row['total']} tokens in {row['elapsed_s']:.1f} s; "
                          f"swap-outs {row['swapouts']}", flush=True)
                    last_progress = time.time()
            elif f[0] == "ERR":
                sys.exit(f"the engine answered: {line.strip()}")
            elif f[0] == "DONE":
                gen, n, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
                hits, look, fmb, read = int(f[9]), int(f[10]), float(f[13]), int(f[14])
                accepted, offered = int(f[6]), int(f[7])
                first = min(50, len(stamps) - 1)
                return {"ids": out, "probes": probes, "stats": stats, "generated": gen, "prompt": n, "resume": resume,
                        "read": read, "prompt_ms": pms, "decode_ms": dms, "finish": f[5],
                        "drafts_accepted": accepted, "drafts_offered": offered,
                        "prefill_tps": 1000 * read / pms if read > 0 and pms > 0 else 0.0,
                        "ttft_s": (stamps[0] - start) if stamps else None,
                        "tps_first": first / (stamps[first] - stamps[0]) if first > 0 else 0.0,
                        "tps": (gen - 1) / (dms / 1000) if gen > 1 and dms > 0 else 0.0,
                        "hits": hits, "lookups": look, "hit_rate": hits / max(1, look), "ssd_gb": fmb / 1000,
                        "wall_s": time.time() - start, "stamps": stamps, "token_s": [round(t - start, 3) for t in stamps],
                        "progress": progress_rows, "swapouts": swapouts() - s0,
                        "watts": watts(p0, power_counters())}
        sys.exit("the engine ended inside a request (see the log)")

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()
        self.log.close()
        return swapouts() - self.swap0


def per_token(r):
    """Where the time of one written token went (ms), from the engine's STAT line of the answer."""
    a = r["stats"].get("answer")
    n = r["generated"] - 1
    if not a or n < 1:
        return None
    # the CPU's own share is what is left: building and encoding the graphs, the cache's bookkeeping, the sampler
    cpu = a["wall"] - a["gpu_wait"] - a["fetch"]
    return {"total_ms": a["wall"] / n, "gpu_wait_ms": a["gpu_wait"] / n, "ssd_ms": a["fetch"] / n, "cpu_ms": cpu / n,
            "encode_ms": a["encode"] / n, "gpu_busy_ms": a["gpu_busy"] / n, "sample_ms": a["sample"] / n,
            "misses": a["misses"] / n, "ssd_mb": a["fetch_mb"] / n, "waits": a["waits"] / n,
            "ahead": a.get("ahead", 0) / n, "ahead_used": a.get("ahead_used", 0) / n,
            "ahead_waits": a.get("ahead_waits", 0) / n, "predict_ms": a.get("predict", 0) / n}


# ---- the requests ----------------------------------------------------------------------------------------------
Q_SKY = "Explain in a short paragraph why the sky is blue."


def suite(tok):
    """name -> (kind, prompt ids).  The documents are cut from this repository's own files to a token count, and a
    reference file keeps the ids it was recorded with, so a later edit of those files does not move the gate."""
    def enc(text):
        return tok.encode(text, parse_special=True)

    def doc_ids(path, n_tokens, ask):
        head = enc("<|im_start|>user\n" + ask + "\n\n")
        tail = enc("<|im_end|>\n<|im_start|>assistant\n")
        body = tok.encode((ROOT / path).read_text(encoding="utf-8"))
        return head + body[:max(0, n_tokens - len(head) - len(tail))] + tail

    readme = "\n".join((ROOT / "README.md").read_text(encoding="utf-8").split("\n")[53:140])
    s = {
        "chat": ("short", enc(chat([("user", Q_SKY)]))),
        "code": ("short", enc(chat([("user", "Write a Python function that merges overlapping intervals in a list of "
                                             "(start, end) tuples. Add a short docstring and two test cases.")]))),
        "reason": ("short", enc(chat([("system", "You are a careful assistant. Answer concisely."),
                                      ("user", "A train leaves at 14:10 and arrives at 17:45 after two stops of 7 "
                                               "minutes each. Its average speed while moving is 96 km/h. How long "
                                               "is the route? Show the calculation.")]))),
        "polish": ("short", enc(chat([("user", "Wyjaśnij w kilku zdaniach, dlaczego zimą dni są krótsze niż latem, "
                                               "i podaj przykład dla Warszawy.")]))),
        "japanese": ("short", enc(chat([("user", "東京から京都まで新幹線で行く日帰り旅行の計画を、時間配分も含めて提案してください。")]))),
        "turns": ("short", enc(chat([
            ("user", "I want to learn to bake bread. Where do I start?"),
            ("assistant", "Start with a simple no-knead loaf: 500 g flour, 375 g water, 10 g salt and 2 g instant "
                          "yeast. Mix, rest 12 to 18 hours, shape, rest one more hour, then bake in a covered pot at "
                          "230 C for 30 minutes and uncovered for 15 more."),
            ("user", "I do not have a covered pot. What can I use instead?"),
            ("assistant", "A baking tray works. Put a second tray on the lowest rack and pour a cup of hot water into "
                          "it when the loaf goes in: the steam does what the lid would. The crust will be a little "
                          "thinner."),
            ("user", "And how do I know when the loaf is done? Give me three checks.")]))),
        "review": ("short", doc_ids("metal/src/gpu_keepalive.hpp", 500,
                                    "Explain what this C++ header is for and name one thing a caller could get wrong.")),
        "doc1600": ("doc", enc(chat([("user", "Summarize the following document in three bullet points.\n\n" + readme)]))),
        "doc8k": ("doc", doc_ids("docs/DETAILS.md", 7900, "Summarize the following document in five bullet points.")),
    }
    return s


def sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


def gate_record(e, tok, a):
    s = suite(tok)
    ref = {"engine": e.exe, "info": e.info, "new": a.new, "recorded": time.strftime("%Y-%m-%d %H:%M"), "prompts": {}}
    only = set(a.only or [])
    for name, (kind, ids) in s.items():
        if only and name not in only:
            continue
        r = e.gen(ids, a.new, " probe=1")
        ref["prompts"][name] = {"kind": kind, "prompt": ids, "prompt_sha": sha(ids), "ids": r["ids"], "probes": r["probes"]}
        margins = sorted(p[4] - p[6] for p in r["probes"])
        print(f"{name:9s} prompt {len(ids):5d} tokens, {len(r['ids'])} out ({r['finish']}), {r['tps']:.2f} tok/s; smallest "
              f"margins between the two best tokens: {', '.join(f'{m:.3f}' for m in margins[:3])}; "
              f"swap-outs {r['swapouts']}", flush=True)
    Path(a.ref).write_text(json.dumps(ref), encoding="utf-8")
    print(f"reference written to {a.ref}")


def gate_check(e, a):
    ref = json.loads(Path(a.ref).read_text(encoding="utf-8"))
    verdicts, ok_all = {}, True
    only = set(a.only or [])
    for name, p in ref["prompts"].items():
        if only and name not in only:
            continue
        want = p["ids"]
        r = e.gen(p["prompt"], ref["new"], " probe=1")
        if r["ids"] == want:
            same_probes = r["probes"] == p["probes"]
            ok = same_probes or not a.exact
            deltas = [abs(x[j] - y[j]) for x, y in zip(r["probes"], p["probes"]) for j in (2, 4, 6, 8)]
            max_delta = max(deltas, default=0.0)
            verdicts[name] = {"verdict": "identical" if ok else "logit mismatch", "probes_identical": same_probes,
                              "max_logit_delta": max_delta}
            ok_all &= ok
            print(f"{name:9s} identical ({len(want)} ids), {r['tps']:.2f} tok/s"
                  + f", reported logits {'identical' if same_probes else 'differ'}"
                  + (f" (max delta {max_delta:.6g})" if not same_probes else "")
                  + (f", drafts {r['drafts_accepted']}/{r['drafts_offered']} right" if r["drafts_offered"] else ""), flush=True)
            continue
        if a.exact:
            verdicts[name] = {"verdict": "token mismatch"}
            ok_all = False
            print(f"{name:9s} token mismatch (exact gate)", flush=True)
            continue
        # teacher forcing: the reference's tokens are fed in, so every place is compared on the same context
        f = e.gen(p["prompt"], len(want), " probe=1 force=" + ".".join(map(str, want)))
        ties, bad = [], []
        for i, (pc, pr) in enumerate(zip(f["probes"], p["probes"])):
            mine, ref_id = pc[0], want[i]
            if mine == ref_id:
                continue
            gap_here = pc[4] - pc[2]                      # this engine: its best token over the reference's
            ref_logits = {pr[3]: pr[4], pr[5]: pr[6], pr[7]: pr[8]}
            gap_ref = ref_logits[ref_id] - ref_logits[mine] if mine in ref_logits else float("inf")
            (ties if gap_here <= a.tie and gap_ref <= a.tie else bad).append(
                {"at": i, "reference": ref_id, "here": mine, "gap_here": round(gap_here, 4),
                 "gap_reference": None if gap_ref == float("inf") else round(gap_ref, 4)})
        first = next(i for i, (x, y) in enumerate(zip(r["ids"] + [None] * len(want), want)) if x != y)
        # a free run that differs where the forced one finds no difference was not explained by it
        ok = not bad and len(f["probes"]) == len(want) and bool(ties)
        ok_all &= ok
        verdicts[name] = {"verdict": "near-tie" if ok else "DIFFERENT", "first_difference": first, "ties": ties, "bad": bad}
        worst = max([max(t["gap_here"], t["gap_reference"] or 9.9) for t in ties + bad], default=0.0)
        print(f"{name:9s} {'near-tie' if ok else 'DIFFERENT'}: the free run leaves the reference at token {first}; with "
              f"the reference fed in, {len(ties)} places are near-ties (within {a.tie} logits in both engines; largest gap "
              f"{worst:.3f})"
              + (", " + str(len(bad)) + " are not: " + json.dumps(bad[:3]) if bad else ""), flush=True)
        if ties:
            print("          " + json.dumps(ties[:4]), flush=True)
    print("GATE " + ("PASSED" if ok_all else "FAILED") + ": " +
          ", ".join(f"{k} {v['verdict']}" for k, v in verdicts.items()))
    return ok_all, verdicts


def run_decode(e, tok, a, seconds=0.0):
    """The short requests of the suite, one after another (again and again for `seconds`)."""
    s = [(k, ids) for k, (kind, ids) in suite(tok).items() if (k in a.only if a.only else kind == "short")]
    rows, t0, stamps = [], time.time(), []
    while True:
        for name, ids in s:
            r = e.gen(ids, a.new, " stats=1")
            pt = per_token(r)
            rows.append({"name": name, "tps": r["tps"], "hit_rate": r["hit_rate"], "ssd_gb": r["ssd_gb"], "watts": r["watts"],
                         "ttft_s": r["ttft_s"], "generated": r["generated"], "decode_ms": r["decode_ms"],
                         "drafts_accepted": r["drafts_accepted"], "drafts_offered": r["drafts_offered"],
                         "per_token": pt, "swapouts": r["swapouts"], "at_s": time.time() - t0, "therm": thermal()})
            stamps += r["stamps"][1:]
            print(f"{name:9s} {r['tps']:.2f} tok/s ({r['generated']} tokens), first token after {r['ttft_s']:.2f} s, "
                  f"hits {100 * r['hit_rate']:.1f}%, {r['ssd_gb']:.1f} GB from the SSD, swap-outs {r['swapouts']}"
                  + (f", {r['watts']:.1f} W" if r["watts"] else "")
                  + (f", drafts {r['drafts_accepted']}/{r['drafts_offered']} right" if r["drafts_offered"] else "")
                  + (f"; per token {pt['total_ms']:.0f} ms = GPU {pt['gpu_wait_ms']:.0f} + SSD {pt['ssd_ms']:.0f} + "
                     f"CPU {pt['cpu_ms']:.0f} (GPU busy {pt['gpu_busy_ms']:.0f}; {pt['misses']:.0f} misses, "
                     f"{pt['ssd_mb']:.0f} MB" + (f"; read ahead {pt['ahead']:.0f}, {pt['ahead_used']:.0f} routed to, "
                                                  f"{pt['ahead_waits']:.0f} waited for, predict {pt['predict_ms']:.1f} ms" if pt["ahead"] else "") + ")"
                     if pt else ""), flush=True)
            if seconds and time.time() - t0 >= seconds:
                break
        if not seconds or time.time() - t0 >= seconds:
            break
    tokens = sum(r["generated"] - 1 for r in rows)
    ms = sum(r["decode_ms"] for r in rows)
    out = {"tps": 1000 * tokens / ms if ms else 0.0, "rows": rows,
           "hit_rate": statistics.mean(r["hit_rate"] for r in rows), "ssd_gb": sum(r["ssd_gb"] for r in rows),
           "swapouts": sum(r["swapouts"] for r in rows)}
    w = [r["watts"] for r in rows if r["watts"]]
    if w:
        out["watts"] = statistics.mean(w)
        out["tokens_per_joule"] = out["tps"] / out["watts"]
    pts = [r["per_token"] for r in rows if r["per_token"]]
    if pts:
        out["per_token"] = {k: statistics.mean(p[k] for p in pts) for k in pts[0]}
    if seconds:
        # tokens per second in windows of 30 s of the run (gaps between requests count: this is what a user gets)
        w = 30.0
        nwin = int((time.time() - t0) // w)
        out["windows"] = [sum(1 for x in stamps if t0 + i * w <= x < t0 + (i + 1) * w) / w for i in range(nwin)]
        print("tokens/s per 30 s: " + " ".join(f"{x:.2f}" for x in out["windows"]), flush=True)
    print(f"decode: {out['tps']:.2f} tok/s over {tokens} tokens, hits {100 * out['hit_rate']:.1f}%, "
          f"{out['ssd_gb']:.1f} GB from the SSD, swap-outs {out['swapouts']}"
          + (f", {out['watts']:.1f} W, {out['tokens_per_joule']:.3f} tokens per joule" if w else ""), flush=True)
    return out


def run_prefill(e, tok, a, sizes):
    body = tok.encode((ROOT / "docs" / "DETAILS.md").read_text(encoding="utf-8"))
    head = tok.encode("<|im_start|>user\nSummarize the following document.\n\n", parse_special=True)
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n", parse_special=True)
    needles = {"ALPHA": "cedar-58317", "BRAVO": "quartz-92406", "CHARLIE": "lilac-76183",
               "DELTA": "maple-40952", "ECHO": "coral-63821"} if a.needle else {}
    if needles:
        head = tok.encode("<|im_start|>user\nFind the five STRATA_TEST_RECORD entries in this document.\n\n", parse_special=True)
        tail = tok.encode("\n\nReturn the exact values of STRATA_TEST_RECORD ALPHA, BRAVO, CHARLIE, DELTA, and ECHO, "
                          "one per line. Do not summarize or explain.\n<|im_end|>\n<|im_start|>assistant\n"
                          "<think>\n\n</think>\n\n", parse_special=True)
    out = {}
    ctx = int(e.info.get("context", 0))
    for i, n in enumerate(sizes):
        if n + a.prefill_new + 8 > ctx:
            print(f"prefill {n}: skipped, the context is {ctx}", flush=True)
            continue
        # a different start of the text for every size, so no request reuses the one before it
        cut = body[1000 * i:] * (1 + n // max(1, len(body) - 1000 * i))
        content = cut[:max(1, n - len(head) - len(tail))]
        positions = {}
        for (name, value), fraction in zip(needles.items(), (0.05, 0.25, 0.5, 0.75, 0.95)):
            record = tok.encode(f"\n\nSTRATA_TEST_RECORD {name} = {value}\n\n")
            pos = int(len(content) * fraction)
            if pos + len(record) > len(content):
                raise ValueError("needle prompts must be large enough for all five records")
            content[pos:pos + len(record)] = record
            positions[name] = len(head) + pos
        ids = head + content + tail
        r = e.gen(ids, a.prefill_new, " stats=1", progress=True)
        st = r["stats"].get("prompt", {})
        out[n] = {"tokens": r["read"], "tps": r["prefill_tps"], "ttft_s": r["ttft_s"], "ssd_gb": r["ssd_gb"],
                  "swapouts": r["swapouts"], "gpu_wait_s": st.get("gpu_wait", 0) / 1000, "ssd_s": st.get("fetch", 0) / 1000,
                  "gpu_busy_s": st.get("gpu_busy", 0) / 1000, "wall_s": st.get("wall", 0) / 1000,
                  "passes": st.get("passes", 0), "read_ahead_gb": st.get("staged_mb", 0) / 1000,
                  "read_ahead_used_gb": st.get("staged_used_mb", 0) / 1000, "watts": r["watts"],
                  "decode_tps": r["tps"], "generated": r["generated"], "output": tok.decode(r["ids"]),
                  "progress": r["progress"], "decode_per_token": per_token(r), "token_s": r["token_s"]}
        if needles:
            out[n]["needle_positions"] = positions
            out[n]["needle_found"] = {name: value in out[n]["output"] for name, value in needles.items()}
        o = out[n]
        print(f"prefill {r['read']:6d} tokens: {r['prefill_tps']:7.1f} tok/s, first token after {r['ttft_s']:.2f} s, "
              f"{r['ssd_gb']:.1f} GB from the SSD, {int(o['passes'])} passes; {o['wall_s']:.1f} s = SSD {o['ssd_s']:.1f} + "
              f"GPU wait {o['gpu_wait_s']:.1f} + CPU {o['wall_s'] - o['ssd_s'] - o['gpu_wait_s']:.1f} (GPU busy "
              f"{o['gpu_busy_s']:.1f}"
              + (f"; {o['read_ahead_gb']:.1f} GB read ahead, {o['read_ahead_used_gb']:.1f} used" if o["read_ahead_gb"] else "")
              + f"); swap-outs {r['swapouts']}" + (f", {o['watts']:.1f} W" if o["watts"] else ""), flush=True)
        if r["generated"] > 1:
            at = r["token_s"]
            print(f"  decode at {n} tokens: {r['tps']:.2f} tok/s, {r['generated']} tokens"
                  + "".join(f", token {k} after {at[k - 1]:.2f} s" for k in (2, 10, 50) if k <= len(at))
                  + f"; {tok.decode(r['ids'])!r}", flush=True)
        if needles:
            print(f"  retrieval: {sum(out[n]['needle_found'].values())}/{len(needles)} exact values", flush=True)
        if a.prefill_repeat:
            again = e.gen(ids, a.prefill_new, " stats=1", progress=True)
            out[n]["repeat"] = {"read": again["read"], "resume": again["resume"], "ttft_s": again["ttft_s"],
                                "decode_tps": again["tps"], "same_ids": again["ids"] == r["ids"],
                                "swapouts": again["swapouts"]}
            print(f"  repeat: {again['resume']} tokens reused, first token after {again['ttft_s']:.2f} s, "
                  f"same output: {again['ids'] == r['ids']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="strata-<model>.json")
    ap.add_argument("--engine", help="another engine build than the config's")
    ap.add_argument("--new", type=int, default=200, help="tokens to generate per request")
    ap.add_argument("--context", type=int, default=None, help="override the config's --max-context")
    ap.add_argument("--only", action="append", help="requests to run (default mode: chat, again, document; gate: names)")
    ap.add_argument("--gate", choices=("record", "check"))
    ap.add_argument("--ref", default=str(ROOT / "build-metal" / "gate-ref.json"))
    ap.add_argument("--tie", type=float, default=0.05, help="gate: two tokens this close in logits are a near-tie")
    ap.add_argument("--exact", action="store_true", help="gate: require identical token IDs and all reported logits")
    ap.add_argument("--prefill", help="prompt sizes in tokens, e.g. 32,256,1600,8000,32000")
    ap.add_argument("--prefill-new", type=int, default=1, help="tokens to generate at each prefill depth (default 1)")
    ap.add_argument("--prefill-repeat", action="store_true", help="repeat each prefill request to check prefix reuse")
    ap.add_argument("--needle", action="store_true", help="prefill: retrieve five records spread across the prompt")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--sustained", type=float, default=0.0, help="seconds")
    ap.add_argument("--ab", nargs="+", help="engines to compare, interleaved; an entry may carry environment and options: "
                                            "'STRATA_KEEPALIVE=fetch EXE --prefetch 10'")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--pause", type=float, default=0.0, help="--ab: seconds of rest before every run (heat)")
    ap.add_argument("--cool", type=float, default=0.0, help="wait before every run until the SoC is below this many Celsius")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--log", default=str(ROOT / "bench-metal.log"))
    ap.add_argument("extra", nargs="*", help="more engine options, after --")
    a = ap.parse_args()
    if a.prefill_new < 1:
        ap.error("--prefill-new must be positive")
    if a.prefill_repeat and not a.prefill:
        ap.error("--prefill-repeat requires --prefill")
    if a.needle and (not a.prefill or min(map(int, a.prefill.split(","))) < 512 or a.prefill_new < 64):
        ap.error("--needle requires --prefill sizes of at least 512 and --prefill-new of at least 64")
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    tok = tokenizer(Path(cfg["tokenizer"]))
    args = list(cfg["args"])
    if a.context:
        args[args.index("--max-context") + 1] = str(a.context)
    args += a.extra
    result = {"when": time.strftime("%Y-%m-%d %H:%M"), "args": args, "memory_before": memory_state()}
    print(f"memory: {result['memory_before']}", flush=True)

    def start(spec):
        # "[NAME=value ...] EXE [engine options]"
        words = spec.split()
        # Match the server's engine environment; inline A/B settings override the config.
        env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
        while words and "=" in words[0] and not words[0].startswith(("/", ".")):
            k, v = words.pop(0).split("=", 1)
            env[k] = v
        exe, extra = words[0], words[1:]
        if a.cool:
            t = cool_down(a.cool)
            print(f"temperature before the run: SoC {t[0]} C, battery {t[1]} C" if t else "no temperature helper", flush=True)
        e = Engine(exe, args + list(extra), a.log, env)
        print(f"{Path(exe).parent.name}/{Path(exe).name}: {e.describe()}", flush=True)
        return e

    exe = a.engine or cfg["exe"]
    if a.ab:
        sizes = [int(x) for x in a.prefill.split(",")] if a.prefill else None
        runs = {x: [] for x in a.ab}
        for i in range(a.runs):
            for x in (a.ab if i % 2 == 0 else a.ab[::-1]):       # ABBA: neither is always the warm one
                if a.pause:
                    time.sleep(a.pause)
                e = start(x)
                one = {"info": e.info, "therm": thermal(), "temps_before": temps(), "memory_before": e.mem0,
                       "ready_s": e.ready_s}
                if sizes:
                    one["prefill"] = run_prefill(e, tok, a, sizes)
                if a.decode or not sizes:
                    one["decode"] = run_decode(e, tok, a, a.sustained)
                one["swapouts"] = e.close()
                one["temps_after"] = temps()
                runs[x].append(one)
        result["ab"] = runs
        print("\nmedians of", a.runs, "runs:")
        for x, rs in runs.items():
            line = f"  {x}: cache {rs[0]['info'].get('expert_cache_mib')} MiB"
            if "decode" in rs[0]:
                each = ", ".join("%.2f" % r["decode"]["tps"] for r in rs)
                line += (f", decode {statistics.median(r['decode']['tps'] for r in rs):.2f} tok/s ({each}), hits "
                         f"{100 * statistics.median(r['decode']['hit_rate'] for r in rs):.1f}%")
            if "prefill" in rs[0]:
                for n in rs[0]["prefill"]:
                    line += f", prefill {n}: {statistics.median(r['prefill'][n]['tps'] for r in rs):.1f} tok/s"
            print(line + f", swap-outs {sum(r['swapouts'] for r in rs)}")
    else:
        e = start(exe)
        result["info"] = e.info
        if a.gate == "record":
            gate_record(e, tok, a)
        elif a.gate == "check":
            ok, result["gate"] = gate_check(e, a)
            result["gate_passed"] = ok
        elif a.prefill or a.decode or a.sustained:
            if a.prefill:
                result["prefill"] = run_prefill(e, tok, a, [int(x) for x in a.prefill.split(",")])
            if a.decode or a.sustained:
                result["decode"] = run_decode(e, tok, a, a.sustained)
        else:
            only = a.only or ["chat", "again", "document"]

            def run(name, prompt):
                r = e.gen(tok.encode(prompt, parse_special=True), a.new)
                first = min(50, len(r["ids"]) - 1)
                rate = ", %.1f tok/s" % r["prefill_tps"] if r["read"] > 8 else ""
                print(f"{name:9s} prompt {r['prompt']} tokens ({r['resume']} reused, {r['read']} read in "
                      f"{r['prompt_ms'] / 1000:.1f} s{rate}); "
                      f"output {r['generated']} tokens: {r['tps_first']:.2f} tok/s over the first {first}, "
                      f"{r['tps']:.2f} tok/s over all; cache hits {100 * r['hit_rate']:.1f}%, {r['ssd_gb']:.1f} GB "
                      f"read from the SSD; {r['wall_s']:.1f} s in all; swap-outs {r['swapouts']}", flush=True)
                result[name] = {k: v for k, v in r.items() if k not in ("stamps", "probes")}
                return tok.decode(r["ids"])

            answer = ""
            if "chat" in only:
                answer = run("chat", chat([("user", Q_SKY)]))
                print("  -> " + " ".join(answer.split())[:160], flush=True)
            if "again" in only and answer:
                # what the engine wrote stays in its context, so the second turn only reads the new question
                run("again", chat([("user", Q_SKY)]) + answer.replace("<|im_end|>", "") +
                    "<|im_end|>\n<|im_start|>user\nNow say it in one sentence for a child.<|im_end|>\n<|im_start|>assistant\n")
            if "document" in only:
                doc = "\n".join((ROOT / "README.md").read_text(encoding="utf-8").split("\n")[53:140])
                text = run("document", chat([("user", "Summarize the following document in three bullet points.\n\n" + doc)]))
                print("  -> " + " ".join(text.split())[:160], flush=True)
        result["swapouts"] = e.close()
        print(f"swap-outs over the whole run: {result['swapouts']}", flush=True)
    result["memory_after"] = memory_state()
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    if a.gate == "check" and not result.get("gate_passed"):
        sys.exit(1)
    if a.needle:
        measured = [run["prefill"] for runs in result["ab"].values() for run in runs] if a.ab else [result["prefill"]]
        if any(not rows or not all(all(row["needle_found"].values()) for row in rows.values()) for rows in measured):
            sys.exit(1)
    if a.prefill_repeat:
        measured = [run["prefill"] for runs in result["ab"].values() for run in runs] if a.ab else [result["prefill"]]
        if any(not rows or not all(row["repeat"]["same_ids"] and row["repeat"]["read"] == 1
                                   for row in rows.values()) for rows in measured):
            sys.exit(1)


if __name__ == "__main__":
    main()
