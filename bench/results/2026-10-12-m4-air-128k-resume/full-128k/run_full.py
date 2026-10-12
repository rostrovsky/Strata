#!/usr/bin/env python3
"""Runs the full-context benchmark and watches macOS's system-wide Swapouts counter for the whole run (engine start
and shutdown included).  If the counter moves, bench.py is killed, then the engine, at once."""
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT = "/Users/rafal/Projects/strata-air/Strata"
OUTDIR = ROOT + "/build-metal/resume-128k"
NAME = "full-128k-newplan"
CONSOLE = f"{OUTDIR}/{NAME}.out"
TRACE = f"{OUTDIR}/{NAME}-watch.jsonl"
EVENTS = f"{OUTDIR}/{NAME}-watch.log"
CMD = [".venv/bin/python", "-u", "metal/tools/bench.py", "--config", "strata-iq3_s.json", "--context", "131072",
       "--prefill", "130944", "--prefill-new", "120", "--needle", "--prefill-repeat",
       "--json", f"build-metal/resume-128k/{NAME}.json", "--log", f"build-metal/resume-128k/{NAME}.log",
       "--", "--kv", "f16", "--prefill", "4096", "--reserve-gib", "4"]
POLL_S, TRACE_S, EVENT_S = 5, 30, 120


def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    return page, {m.group(1): int(m.group(2)) for m in re.finditer(r'"?([A-Za-z -]+)"?:\s+(\d+)\.', out)}


def state():
    page, v = vm()
    g = page / 2**30
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage", "kern.memorystatus_vm_pressure_level"],
                        capture_output=True, text=True).stdout
    used = re.search(r"used = ([\d.]+)M", sw)
    level = re.search(r"(?m)^(\d+)\s*$", sw)
    row = {"swapouts": v.get("Swapouts", 0), "swapins": v.get("Swapins", 0),
           "free_gib": round(v.get("Pages free", 0) * g, 2),
           "file_cache_gib": round(v.get("File-backed pages", 0) * g, 2),
           "anonymous_gib": round(v.get("Anonymous pages", 0) * g, 2),
           "wired_gib": round(v.get("Pages wired down", 0) * g, 2),
           "compressor_stored_gib": round(v.get("Pages stored in compressor", 0) * g, 2),
           "compressor_occupied_gib": round(v.get("Pages occupied by compressor", 0) * g, 2),
           "swap_used_mib": float(used.group(1)) if used else None,
           "pressure_level": int(level.group(1)) if level else None}
    return row


def temps():
    try:
        out = subprocess.run([ROOT + "/build-metal/bench/temps"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None, None
    die, bat = [], []
    for line in out.splitlines():
        name, _, c = line.rpartition(" ")
        (die if "tdie" in name else bat if "battery" in name else []).append(float(c))
    return (max(die) if die else None, max(bat) if bat else None)


def last_console():
    try:
        with open(CONSOLE, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2000))
            lines = [x for x in f.read().decode("utf-8", "replace").splitlines() if x.strip()]
        return lines[-1].strip()[:300] if lines else ""
    except OSError:
        return ""


def event(text):
    line = time.strftime("%H:%M:%S ") + text
    with open(EVENTS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def engine_pids():
    out = subprocess.run(["pgrep", "-x", "strata"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    os.chdir(ROOT)
    if engine_pids():
        sys.exit("an engine is already running")
    for p in (TRACE, EVENTS):
        open(p, "w").close()
    s0 = state()
    t = temps()
    base = s0["swapouts"]
    event(f"BASELINE swapouts {base}, swap used {s0['swap_used_mib']} MiB, pressure {s0['pressure_level']}, "
          f"free {s0['free_gib']} + file cache {s0['file_cache_gib']} GiB, compressor {s0['compressor_stored_gib']} in "
          f"{s0['compressor_occupied_gib']} GiB, SoC {t[0]} C, battery {t[1]} C")
    env = dict(os.environ, STRATA_METAL_INDEXED_QSA="1")
    t0 = time.time()
    with open(CONSOLE, "w") as console:
        bench = subprocess.Popen(CMD, stdout=console, stderr=subprocess.STDOUT, env=env)
    event(f"STARTED bench.py pid {bench.pid}")
    last_trace = last_event = 0.0
    peak_level, stopped, known = s0["pressure_level"] or 0, None, []
    while bench.poll() is None:
        time.sleep(POLL_S)
        s = state()
        now = time.time()
        known = engine_pids() or known
        moved = s["swapouts"] - base
        peak_level = max(peak_level, s["pressure_level"] or 0)
        if moved > 0:
            depth = last_console()
            bench.send_signal(signal.SIGTERM)
            time.sleep(0.2)
            for pid in known:
                if alive(pid):
                    os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            for pid in known:
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)
            if bench.poll() is None:
                time.sleep(1)
                if bench.poll() is None:
                    bench.kill()
            stopped = {"at_s": round(now - t0, 1), "pages": moved, "mib": moved * 16 / 1024, "console": depth, "state": s}
            event(f"SWAP MOVED: {moved} pages ({moved * 16 / 1024:.2f} MiB) at {now - t0:.0f} s; killed bench.py and "
                  f"engine {known}; last console line: {depth}")
            break
        if now - last_trace >= TRACE_S:
            tt = temps()
            row = {"t": time.strftime("%H:%M:%S"), "elapsed_s": round(now - t0, 1), "swapouts_new": moved, **s,
                   "soc_c": tt[0], "battery_c": tt[1], "engine_pids": known, "console": last_console()}
            with open(TRACE, "a") as f:
                f.write(json.dumps(row) + "\n")
            last_trace = now
            if now - last_event >= EVENT_S:
                event(f"ok +{now - t0:.0f} s: swap-outs new {moved}, pressure {s['pressure_level']}, free "
                      f"{s['free_gib']} + file {s['file_cache_gib']} GiB, compressor {s['compressor_stored_gib']} in "
                      f"{s['compressor_occupied_gib']} GiB, SoC {tt[0]} C | {row['console'][:150]}")
                last_event = now
    code = bench.wait()
    # the engine's shutdown and the minute after it count too
    tail = []
    for wait in (3, 10, 20, 30):
        time.sleep(wait)
        s = state()
        tail.append({"after_exit_s": sum((3, 10, 20, 30)[:len(tail) + 1]), "swapouts_new": s["swapouts"] - base, **s})
    final = {"baseline": s0, "exit_code": code, "wall_s": round(time.time() - t0, 1), "peak_pressure_level": peak_level,
             "stopped": stopped, "after_exit": tail, "engine_still_running": engine_pids()}
    with open(TRACE, "a") as f:
        f.write(json.dumps({"final": final}) + "\n")
    event(f"DONE exit code {code}, new swap-outs over the whole run {tail[-1]['swapouts_new']} pages, swap used "
          f"{tail[-1]['swap_used_mib']} MiB (was {s0['swap_used_mib']}), peak pressure level {peak_level}, "
          f"engine still running: {final['engine_still_running']}, stopped: {bool(stopped)}")


if __name__ == "__main__":
    main()
