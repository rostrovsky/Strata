"""passes.py NAME...: each long pass's time (from bench.py's progress lines) and how much of the engine's own memory
macOS held compressed at its end (the engine's --verbose pass lines), run from build-metal/resume-128k"""
import json, re, sys
runs = sys.argv[1:]
out = {}
for n in runs:
    d = json.load(open(f"{n}.json"))["prefill"]["130944"]
    ends = {x["tokens"]: x["elapsed_s"] for x in d["progress"]}
    comp = {}
    for l in open(f"{n}.log", errors="replace"):
        m = re.search(r"pass of (\d+) tokens to (\d+): ([\d.]+) MiB on the GPU, ([\d.]+) MiB in all, ([\d.]+) MiB of it compressed", l)
        if m and int(m.group(1)) > 1024: comp[int(m.group(2))] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
    rows, prev = [], 0.0
    for k in sorted(comp):
        if k not in ends: continue
        rows.append((k, ends[k] - prev) + comp[k]); prev = ends[k]
    out[n] = rows
print("| pass to | " + " | ".join(f"{n}: s, MiB compressed" for n in runs) + " |")
print("| ---: |" + " ---: |" * len(runs))
for i in range(max(len(r) for r in out.values())):
    print(f"| {out[runs[0]][i][0]:,} | " + " | ".join(f"{out[n][i][1]:.1f}, {out[n][i][4]:.0f}" if i < len(out[n]) else "-" for n in runs) + " |")
for n in runs:
    r = out[n][1:]
    print(f"{n}: passes 2-{len(out[n])} {sum(x[1] for x in r):.1f} s; engine's memory compressed, mean {sum(x[4] for x in r) / len(r):.0f} MiB; "
          f"on the GPU {max(x[2] for x in r):.1f} MiB, in all up to {max(x[3] for x in r):.1f} MiB")
