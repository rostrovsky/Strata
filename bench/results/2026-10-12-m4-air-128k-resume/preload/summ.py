#!/usr/bin/env python3
"""summ.py NAME.json [NAME-vmstat.txt]: starts.py's runs as tables (times from the launch of the engine), and what
macOS compressed while each engine was loaded (the stretches of the vm_stat trace with more than 8 GiB wired, as
../refill/per_engine.py, in the order of the starts)"""
import json
import re
import sys


def compressed(trace):
    g = 16384 / 2**30
    hdr, rows = None, []
    for line in open(trace):
        t, rest = line.split(" ", 1)
        f = rest.split()
        if not f or f[0] == "Mach":
            continue
        if f[0] == "free":
            hdr = f
            continue
        if hdr is None or len(f) != len(hdr):
            continue
        num = lambda x: (lambda m: float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)])(re.match(r"([\d.]+)([KMG]?)", x))
        r = dict(zip(hdr, map(num, f)))
        if r["swapouts"] > 1e6:   # the totals after a header
            continue
        r["t"] = t
        rows.append(r)
    runs, cur = [], []
    for r in rows:
        if r["wired"] * g > 8:
            cur.append(r)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [(run[0]["t"], sum(r["comprs"] for r in run) * g, sum(r["swapouts"] for r in run)) for run in runs if len(run) >= 10]


def main():
    res = json.load(open(sys.argv[1]))
    comp = compressed(sys.argv[2]) if len(sys.argv) > 2 else []
    starts = sorted((run["n"], case, spec, run) for case, specs in res["cases"].items() for spec, runs in specs.items() for run in runs)
    by_n = {}
    for i, (n, *_rest) in enumerate(starts):
        by_n[n] = comp[i] if i < len(comp) else None
    label = lambda spec: " ".join(w for w in spec.split() if "=" in w) or spec.split()[0]
    for case, specs in res["cases"].items():
        runs = sorted((run["n"], spec, run) for spec, rr in specs.items() for run in rr)
        if not runs:
            continue
        print(f"\n### {case}: {runs[0][2]['requests'][0]['prompt']} tokens, then {runs[0][2]['requests'][1]['read']} more "
              f"({runs[0][2]['requests'][1]['resume']} reused)\n")
        print("| " + " | ".join(["", *(label(s) for _, s, _ in runs)]) + " |")
        print("| --- |" + " ---: |" * len(runs))
        row = lambda name, f: print(f"| {name} | " + " | ".join(f(r) for _, _, r in runs) + " |")
        row("Start (n)", lambda r: f"{r['started']} ({r['n']})")
        row("SoC before", lambda r: f"{r['temps_before'][0]:.1f} C" if r.get("temps_before") else "?")
        row("Expert cache", lambda r: f"{r['info'].get('expert_cache_mib')} MiB")
        row("Filled at the start", lambda r: f"{r['log'].get('filled_gib', 0):.2f} GiB in {r['log'].get('fill_s', 0):.1f} s")
        row("READY", lambda r: f"{r['ready_s']:.2f} s")
        for k, name in ((0, "Request 1"), (1, "Request 2")):
            for m in (1, 2, 10, 50, 64, 100):
                if all(len(r["requests"][k]["from_launch_s"]) >= m for _, _, r in runs):
                    row(f"{name}: token {m}", lambda r, k=k, m=m: f"{r['requests'][k]['from_launch_s'][m - 1]:.2f} s")
            row(f"{name}: sent at", lambda r, k=k: f"{r['requests'][k]['sent_s']:.2f} s")
            row(f"{name}: misses per written token", lambda r, k=k: f"{(r['requests'][k]['per_token'] or {}).get('misses', 0):.1f}")
            row(f"{name}: written", lambda r, k=k: f"{r['requests'][k]['tps']:.2f} tok/s")
        row("Compressed while loaded", lambda r: f"{by_n[r['n']][1]:.2f} GiB" if by_n.get(r["n"]) else "?")
        row("Swap-outs", lambda r: str(r["swapouts"]))
        same = [len({tuple(r["requests"][k]["ids"]) for _, _, r in runs}) == 1 for k in range(2)]
        print(f"\nSame ids in every start: request 1 {same[0]}, request 2 {same[1]}.")


if __name__ == "__main__":
    main()
