"""compare.py OLD.json NEW.json: two bench.py runs of the 130,944-token prompt side by side"""
import json, sys
def one(path):
    d = json.load(open(path)); p = d["prefill"]["130944"]; t = p["token_s"]; pt = p["decode_per_token"]
    return {
        "cache MiB (start)": d["info"]["expert_cache_mib"], "experts": d["info"]["expert_slots"],
        "read tok/s": round(p["tps"], 2), "first token s": round(p["ttft_s"], 2),
        "passes": p["passes"], "SSD GB": round(p["ssd_gb"], 1),
        "wait GPU/SSD/CPU s": (round(p["gpu_wait_s"]), round(p["ssd_s"], 1), round(p["wall_s"] - p["gpu_wait_s"] - p["ssd_s"], 1)),
        "GPU busy s": round(p["gpu_busy_s"], 1), "read ahead / used GB": (round(p["read_ahead_gb"], 1), round(p["read_ahead_used_gb"], 1)),
        "W": round(p["watts"], 2), "written tok/s": round(p["decode_tps"], 2), "tokens": p["generated"],
        "token 2/10/50 s": (t[1], t[9], t[49]) if len(t) > 49 else t[:3],
        "per token ms": {k: round(pt[k], 1) for k in ("total_ms", "gpu_wait_ms", "ssd_ms", "cpu_ms", "misses", "ssd_mb")},
        "records": sum(p["needle_found"].values()), "repeat": p["repeat"], "swapouts": d["swapouts"],
        "mem before": d["memory_before"], "mem after": d["memory_after"],
    }
a, b = one(sys.argv[1]), one(sys.argv[2])
for k in a: print(f"{k:22} | {a[k]} | {b[k]}")
