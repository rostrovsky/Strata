#!/usr/bin/env python3
"""probe_run.py - what macOS gives without compressing, with the engine loaded and idle (as when a short stretch
of a conversation arrives): starts the engine, then runs givable_probe (wired steps until macOS compresses) a few
times, half a minute apart.

    .venv/bin/python probe_run.py --config strata-iq3_s.json --probe build-metal/borrow/bin/givable_probe --log FILE
        [--exe build-metal/bin/strata] [STEP_MIB:MS ...] -- engine options
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "metal" / "tools" / "bench.py").exists())
sys.path.insert(0, str(ROOT / "metal" / "tools"))
from bench import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--exe")
    ap.add_argument("--pause", type=float, default=30.0)
    ap.add_argument("rates", nargs="*", default=["64:50", "64:50", "64:250"])
    a, _ = ap.parse_known_args()
    extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    rates = [r for r in a.rates if r not in extra]
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    e = Engine(a.exe or cfg["exe"], list(cfg["args"]) + extra, a.log, env)
    print(f"engine: {e.describe()}", flush=True)
    try:
        for rate in rates:
            time.sleep(a.pause)
            step, ms = rate.split(":")
            print(f"\n{time.strftime('%H:%M:%S')} probe: steps of {step} MiB every {ms} ms", flush=True)
            r = subprocess.run([a.probe, step, ms, "16", "8192"], capture_output=True, text=True)
            print(r.stdout, end="", flush=True)
            if r.returncode != 0:
                print(f"probe rc={r.returncode}", flush=True)
                break
    finally:
        time.sleep(5)
        print(f"engine swap-outs: {e.close()}", flush=True)


if __name__ == "__main__":
    main()
