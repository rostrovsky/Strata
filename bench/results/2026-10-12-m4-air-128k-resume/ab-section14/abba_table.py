"""abba_table.py NAME...: the 130,944-token runs of an A B B A side by side (bench.py JSON, the engine's log, the
1 Hz vm_stat trace beside it), run from build-metal/resume-128k"""
import json, re, sys

g = 16384 / 2**30


def vm_rows(name):
    hdr, rows = None, []
    for line in open(f"{name}-vmstat.txt"):
        t, rest = line.split(" ", 1); f = rest.split()
        if not f or f[0] == "Mach": continue
        if f[0] == "free": hdr = f; continue
        if hdr is None or len(f) != len(hdr): continue
        def num(x):
            m = re.match(r"([\d.]+)([KMG]?)", x); return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
        r = dict(zip(hdr, map(num, f))); r["t"] = t
        if r["swapouts"] > 1e6: continue          # vm_stat's first row: totals since boot
        rows.append(r)
    return rows


def one(name):
    d = json.load(open(f"{name}.json")); p = d["prefill"]["130944"]; pt = p["decode_per_token"]; t = p["token_s"]
    log = [l for l in open(f"{name}.log", errors="replace") if l.startswith("strata metal:")]
    pick = lambda pat: next((re.search(pat, l) for l in log if re.search(pat, l)), None)
    held = pick(r"memory now: ([\d.]+) GiB free or file cache, ([\d.]+) GiB held by other programs")
    ans = pick(r"answers are written with a ([\d.]+) GiB expert cache")
    full = pick(r"the expert cache gives it: ([\d.]+) -> ([\d.]+) GiB")
    passes = [int(m.group(1)) for l in log for m in [re.search(r"pass of (\d+) tokens to", l)] if m and int(m.group(1)) > 1024]
    rows = vm_rows(name)
    loaded = [i for i, r in enumerate(rows) if r["wired"] * g > 8]
    a, b = loaded[0], loaded[-1]
    comp_launch = sum(r["comprs"] for r in rows[:b + 1]) * g
    return {
        "free / file cache / other programs at start (bench)": (d["memory_before"]["free_gib"], d["memory_before"]["file_cache_gib"], d["memory_before"]["apps_gib"]),
        "engine: free or file cache / held by others": (held.group(1), held.group(2)) if held else None,
        "cache for answers GiB; experts": (ans.group(1) if ans else None, d["info"]["expert_slots"]),
        "cache at full depth GiB": full.group(2) if full else None,
        "long passes": f"{len(passes)} of {max(passes) if passes else 0}",
        "read tok/s": round(p["tps"], 2), "first token s": round(p["ttft_s"], 2),
        "wait GPU / SSD / CPU s": (round(p["gpu_wait_s"], 1), round(p["ssd_s"], 1), round(p["wall_s"] - p["gpu_wait_s"] - p["ssd_s"], 1)),
        "GPU busy s": round(p["gpu_busy_s"], 1), "SSD GB": round(p["ssd_gb"], 1), "W": round(p["watts"], 2),
        "written tok/s": round(p["decode_tps"], 2), "tokens": p["generated"],
        "per token ms GPU/SSD/CPU; misses; MB": (round(pt["total_ms"], 1), round(pt["gpu_wait_ms"], 1), round(pt["ssd_ms"], 1), round(pt["cpu_ms"], 1), round(pt["misses"], 1), round(pt["ssd_mb"], 1)),
        "token 2/10/50 after first, s": tuple(round(t[i] - t[0], 3) for i in (1, 9, 49)) if len(t) > 49 else None,
        "records": sum(p["needle_found"].values()), "output": p["output"][-40:].replace("\n", " "),
        "repeat s; same": (round(p["repeat"]["ttft_s"], 2), p["repeat"]["same_ids"]),
        "compressed from launch GiB": round(comp_launch, 2),
        "free min / median while loaded": (round(min(r["free"] for r in rows[a:b + 1]) * g, 2), round(sorted(r["free"] for r in rows[a:b + 1])[(b - a) // 2] * g, 2)),
        "swap-outs (trace; bench)": (int(sum(r["swapouts"] for r in rows)), d["swapouts"]),
    }


runs = {n: one(n) for n in sys.argv[1:]}
keys = next(iter(runs.values())).keys()
print("| | " + " | ".join(runs) + " |")
print("| --- |" + " ---: |" * len(runs))
for k in keys:
    print(f"| {k} | " + " | ".join(str(r[k]) for r in runs.values()) + " |")
