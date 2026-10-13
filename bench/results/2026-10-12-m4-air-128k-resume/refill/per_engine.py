#!/usr/bin/env python3
"""per_engine.py NAME: what macOS compressed while each engine of the run was loaded (stretches of NAME-vmstat.txt with more than 8 GiB wired)"""
import re, sys
g = 16384 / 2**30
hdr = None; rows = []
for line in open(f"{sys.argv[1]}-vmstat.txt"):
    t, rest = line.split(" ", 1); f = rest.split()
    if not f or f[0] == "Mach": continue
    if f[0] == "free": hdr = f; continue
    if hdr is None or len(f) != len(hdr): continue
    def num(x):
        m = re.match(r"([\d.]+)([KMG]?)", x); return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
    r = dict(zip(hdr, map(num, f)))
    if r["swapouts"] > 1e6: continue
    r["t"] = t; rows.append(r)
runs, cur = [], []
for r in rows:
    if r["wired"] * g > 8: cur.append(r)
    elif cur: runs.append(cur); cur = []
if cur: runs.append(cur)
for i, run in enumerate(runs):
    if len(run) < 10: continue
    tot = lambda k: sum(r[k] for r in run) * g
    print(f"engine {i + 1}: {run[0]['t']}-{run[-1]['t']} ({len(run)} s): compressed {tot('comprs'):.2f} GiB, decompressed {tot('dcomprs'):.2f}, "
          f"free min {min(r['free'] for r in run) * g:.2f} median {sorted(r['free'] for r in run)[len(run) // 2] * g:.2f} GiB, "
          f"free + file cache min {min(r['free'] + r['file-backed'] for r in run) * g:.2f}, swap-outs {tot('swapouts') / g:.0f}")
