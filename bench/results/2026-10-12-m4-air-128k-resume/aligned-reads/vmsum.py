"""vmsum.py NAME: per-second page-ins, compressions and memory from NAME-vmstat.txt, summed over the engine's run."""
import re, sys
name = sys.argv[1]
hdr = None; rows = []
for line in open(f"{name}-vmstat.txt"):
    t, rest = line.split(" ", 1); f = rest.split()
    if not f or f[0] == "Mach": continue
    if f[0] == "free": hdr = f; continue
    if hdr is None or len(f) != len(hdr): continue
    def num(x):
        m = re.match(r"([\d.]+)([KMG]?)", x); return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
    r = dict(zip(hdr, map(num, f)))
    if r["swapouts"] > 1e6: continue
    rows.append((t, r))
g = 16384 / 2**30
run = [(t, r) for t, r in rows if r["wired"] * g > 8]      # while the engine holds its memory
if not run: sys.exit("no rows with the engine loaded")
tot = lambda k: sum(r[k] for _, r in run)
print(f"{name}: {len(run)} s with the engine loaded; page-ins {tot('pageins') * g:.1f} GiB ({tot('pageins') * g / len(run) * 1024:.0f} MiB/s), "
      f"compressions {tot('comprs') * g:.1f} GiB, decompressions {tot('dcomprs') * g:.1f} GiB, swap-outs {tot('swapouts'):.0f} pages")
print(f"   file cache {min(r['file-backed'] for _, r in run) * g:.2f}-{max(r['file-backed'] for _, r in run) * g:.2f} GiB, "
      f"other programs' memory in RAM {min(r['anonymous'] for _, r in run) * g:.2f}-{max(r['anonymous'] for _, r in run) * g:.2f} GiB, "
      f"in the compressor {min(r['cmprssed'] for _, r in run) * g:.1f}-{max(r['cmprssed'] for _, r in run) * g:.1f} GiB stored")
