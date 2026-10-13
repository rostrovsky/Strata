#!/usr/bin/env python3
"""summ.py RUN.json : per engine start of an --ab run, when the first token and later ones came (seconds from the request)."""
import json, sys
d = json.load(open(sys.argv[1]))
marks = (1, 2, 10, 25, 50, 64, 100, 200)
for spec, runs in d["ab"].items():
    name = spec.split("bin/strata")[-1].strip() or spec.split()[0]
    if "=" in spec.split()[0]:
        name = spec.split()[0] + " " + name
    for i, r in enumerate(runs):
        for n, p in r["prefill"].items():
            at = p.get("token_s") or []
            pt = p.get("decode_per_token") or {}
            row = " ".join(f"t{k}={at[k - 1]:.2f}" for k in marks if k <= len(at))
            print(f"{name:28s} run {i} prompt {n:>6}: cache {r['info'].get('expert_cache_mib')} MiB, read {p['tps']:.1f} tok/s, "
                  f"{row}; written {p['decode_tps']:.2f} tok/s, misses/token {pt.get('misses', 0):.1f}, "
                  f"SSD ms/token {pt.get('ssd_ms', 0):.1f}, swap-outs {r['swapouts']}")
