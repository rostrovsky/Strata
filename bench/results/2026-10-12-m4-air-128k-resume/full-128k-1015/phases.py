"""phases.py NAME: compressions (GiB) per stretch of NAME-vmstat.txt (trace_run.sh's 1 Hz vm_stat) while the engine is
loaded (wired > 8 GiB, as memsum.py), and in the seconds of the trace before and after that"""
import re, sys
g = 16384 / 2**30
rows = []
hdr = None
for line in open(f"{sys.argv[1]}-vmstat.txt"):
    t, rest = line.split(" ", 1); f = rest.split()
    if not f or f[0] == "Mach": continue
    if f[0] == "free": hdr = f; continue
    if hdr is None or len(f) != len(hdr): continue
    def num(x):
        m = re.match(r"([\d.]+)([KMG]?)", x); return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
    r = dict(zip(hdr, map(num, f))); r["t"] = t
    if r["swapouts"] > 1e6: continue
    rows.append(r)
loaded = [i for i, r in enumerate(rows) if r["wired"] * g > 8]
a, b = loaded[0], loaded[-1]
print("loaded", rows[a]["t"], "-", rows[b]["t"], f"({b - a + 1} rows)")
# first 30 s of the load, then 5-minute stretches
edges = [a, a + 30] + list(range(a + 30 + 300, b + 1, 300)) + [b + 1]
for s, e in zip(edges, edges[1:]):
    seg = rows[s:e]
    big = sorted(((r["comprs"] * g, r["t"]) for r in seg), reverse=True)[:3]
    print(f"{rows[s]['t']}-{rows[e - 1]['t']}: compressions {sum(r['comprs'] for r in seg) * g:.2f} GiB, free min "
          f"{min(r['free'] for r in seg) * g:.2f}; biggest seconds " + ", ".join(f"{t} {c:.2f}" for c, t in big))
before = rows[max(0, a - 60):a]; after = rows[b + 1:b + 61]
print(f"60 s before load: {sum(r['comprs'] for r in before) * g:.2f} GiB; 60 s after: {sum(r['comprs'] for r in after) * g:.2f} GiB")
