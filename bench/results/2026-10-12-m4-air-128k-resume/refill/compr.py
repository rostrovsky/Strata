#!/usr/bin/env python3
"""compr.py NAME [MIN_MIB]: the seconds of NAME-vmstat.txt in which macOS compressed MIN_MIB or more (default 16), with free memory and wired"""
import re, sys
g = 16384 / 2**30
name = sys.argv[1]; least = float(sys.argv[2]) if len(sys.argv) > 2 else 16
hdr = None
for line in open(f"{name}-vmstat.txt"):
    t, rest = line.split(" ", 1); f = rest.split()
    if not f or f[0] == "Mach": continue
    if f[0] == "free": hdr = f; continue
    if hdr is None or len(f) != len(hdr): continue
    def num(x):
        m = re.match(r"([\d.]+)([KMG]?)", x); return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
    r = dict(zip(hdr, map(num, f)))
    if r["swapouts"] > 1e6: continue
    if r["comprs"] * g * 1024 >= least:
        print(f"{t} compressed {r['comprs'] * g * 1024:7.0f} MiB, free {r['free'] * g:5.2f}, file cache {r['file-backed'] * g:5.2f}, wired {r['wired'] * g:5.2f} GiB")
