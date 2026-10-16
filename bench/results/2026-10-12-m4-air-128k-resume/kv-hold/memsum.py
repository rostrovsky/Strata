"""memsum.py NAME...: free memory, file cache and compression while the engine was loaded, from NAME-vmstat.txt"""
import re, sys
g = 16384 / 2**30
def load(name):
    hdr = None; rows = []
    for line in open(f"{name}-vmstat.txt"):
        t, rest = line.split(" ", 1); f = rest.split()
        if not f or f[0] == "Mach": continue
        if f[0] == "free": hdr = f; first = True; continue
        if hdr is None or len(f) != len(hdr): continue
        def num(x):
            m = re.match(r"([\d.]+)([KMG]?)", x); return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
        r = dict(zip(hdr, map(num, f)))
        if first: first = False; continue    # vm_stat's first row after a header: totals since boot
        rows.append(r)
    return rows
for n in sys.argv[1:]:
    run = [r for r in load(n) if r["wired"] * g > 8]      # while the engine holds its memory
    if not run: print(f"{n}: no rows with the engine loaded"); continue
    tot = lambda k: sum(r[k] for r in run) * g
    med = lambda k: sorted(r[k] for r in run)[len(run) // 2] * g
    print(f"{n}: {len(run)} s loaded; free min {min(r['free'] for r in run) * g:.2f}, median {med('free'):.2f} GiB; file cache median "
          f"{med('file-backed'):.2f}; free + file cache min {min(r['free'] + r['file-backed'] for r in run) * g:.2f}; page-ins {tot('pageins'):.1f} GiB, "
          f"compressions {tot('comprs'):.2f} GiB, decompressions {tot('dcomprs'):.2f}; other programs in RAM median {med('anonymous'):.2f}, "
          f"stored compressed max {max(r['cmprssed'] for r in run) * g:.1f}; swap-outs {tot('swapouts') / g:.0f} pages")
