"""metal/setup_metal.py - Strata on an Apple silicon Mac (EXPERIMENTAL, docs/APPLE_METAL.md).

    ./setup-mac.sh [--model IQ3_S] [--context 32768] [--data-dir DIR] [--port 8080] [--vision yes|no] [--yes]
                   [--no-start] [--check]

setup.py installs the CUDA / HIP engine on Windows and Linux; a Mac has neither, so this is its own, much smaller
installer for the Metal engine (metal/).  It keeps setup.py's layout, so the two agree on where things are:

  - the engine is compiled here (build-metal/bin/strata; CMake and the Xcode command line tools);
  - the model's two GGUF shards go to <data>/models/<SIZE>/ (default data folder: Strata-data next to this one),
    downloaded with curl, resumable, and checked against the SHA-256 the Hub publishes for them;
  - the vision projector (--vision yes, the default: pictures) goes to <data>/models/mmproj/, checked the same way,
    and strata-vision, the image encoder, is compiled with the engine;
  - the tokenizer is exported to <data>/packs/<size>/tokenizer (tools/strata_tokenizer.py);
  - strata-<size>.json and run-<size>.sh are written here, and run-<size>.sh starts serve/server.py.  The config
    unloads the model after 15 idle minutes ("idle_unload_s", --idle-unload): the next request loads it again.

Nothing is written next to the model files while Strata runs: the engine opens them read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"   # the revision setup.py pins (2026-09-29)
FILE = "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf"
MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"   # the vision encoder and projector, one for every size (0.91 GB)
# download GB, GiB of routed experts (the part the memory plan divides between the expert cache and the SSD)
MODELS = {"Q2_0": (66.4, 31.7), "IQ2_XS": (68.0, 33.1), "IQ3_XXS": (75.8, 40.0), "IQ3_S": (83.6, 46.8)}
CONTEXTS = (8192, 32768, 65536, 131072)


def say(msg=""):
    print(msg, flush=True)


def fail(msg):
    say("\n  x " + msg)
    sys.exit(1)


def sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def machine():
    ram = int(sysctl("hw.memsize") or 0) / 2**30
    return {"chip": sysctl("machdep.cpu.brand_string"), "ram_gib": ram,
            "pcores": int(sysctl("hw.perflevel0.physicalcpu") or 0), "macos": platform.mac_ver()[0]}


def check(data: Path, model: str):
    """What this Mac is, and whether the engine can run on it.  Returns the machine facts."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        fail("the Metal engine needs an Apple silicon Mac (this is %s %s). On Windows and Linux use START-HERE.bat / "
             "setup.sh." % (sys.platform, platform.machine()))
    m = machine()
    free = shutil.disk_usage(data if data.exists() else data.parent).free / 1e9
    say(f"  {m['chip']}, {m['ram_gib']:.0f} GB of unified memory, macOS {m['macos']}, {free:.0f} GB free in {data}")
    if m["ram_gib"] < 24:
        fail("the Metal engine was written for a Mac with 32 GB of memory or more; with less, too little of the model "
             "stays in memory.")
    dl, experts = MODELS[model]
    # Metal keeps about two thirds of the memory resident for one process (measured: 21.3 of 32 GiB on an M4 Air)
    cache = max(0.0, m["ram_gib"] * 2 / 3 - 5.5)
    if cache < experts:
        say(f"  {model}: about {cache:.0f} of its {experts:.0f} GiB of experts fit the expert cache "
            f"({100 * cache / experts:.0f}%); the others are read from the SSD while it answers (read-only).")
    else:
        say(f"  {model}: all {experts:.0f} GiB of its experts fit the expert cache.")
    for tool in ("cmake", "git", "curl", "cc"):
        if shutil.which(tool) is None:
            fail(f"`{tool}` was not found. Install the Xcode command line tools (xcode-select --install)"
                 + (" and CMake (brew install cmake, or pip install cmake)" if tool == "cmake" else "") + ".")
    return m, free


def build_engine():
    exe = ROOT / "build-metal" / "bin" / "strata"
    say("  compiling the Metal engine (a few minutes the first time) ...")
    cfg = ["cmake", "-S", str(ROOT / "metal"), "-B", str(ROOT / "build-metal"), "-DCMAKE_BUILD_TYPE=Release"]
    for cmd in (cfg, ["cmake", "--build", str(ROOT / "build-metal"), "-j", str(os.cpu_count() or 8)]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            say((r.stdout + r.stderr)[-4000:])
            fail("the engine did not compile (the lines above say where).")
    if not exe.exists():
        fail(f"the build finished without {exe}")
    return exe


def hub_files(model):
    """name -> (bytes, sha256) of the model's shards (model "": the files at the top, the projector), from the Hub's
    file listing at the pinned revision."""
    url = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}" + (f"/{model}" if model else "")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            listing = json.load(r)
    except (OSError, ValueError) as e:
        fail(f"could not ask Hugging Face for the file list ({e}).")
    out = {}
    for f in listing:
        name = f.get("path", "").split("/")[-1]
        if name.endswith(".gguf") and f.get("lfs"):
            out[name] = (int(f["size"]), f["lfs"]["oid"])
    return out


def sha256_of(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def download(model, dest: Path):
    """The two shards into dest, resumable (a .part file continues where it stopped), each checked once."""
    dest.mkdir(parents=True, exist_ok=True)
    names = [FILE.format(q=model, i=i) for i in (1, 2)]
    if all((dest / n).exists() for n in names):
        say("  the model files are here")
        return [dest / n for n in names]
    want = hub_files(model)
    for n in names:
        if n not in want:
            fail(f"{n} is not in {REPO} at the pinned revision.")
    base = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{model}/"
    for n in names:
        fetch(base, n, dest, *want[n])
    return [dest / n for n in names]


def download_mmproj(dest: Path):
    """The vision projector into dest, the same way as the shards."""
    f = dest / MMPROJ
    if f.exists():
        say("  the vision projector is here")
        return f
    want = hub_files("")
    if MMPROJ not in want:
        fail(f"{MMPROJ} is not in {REPO} at the pinned revision.")
    dest.mkdir(parents=True, exist_ok=True)
    fetch(f"https://huggingface.co/{REPO}/resolve/{REVISION}/", MMPROJ, dest, *want[MMPROJ])
    return f


def fetch(base, n, dest: Path, size, sha):
    """base + n into dest/n with curl, resumable (a .part file continues where it stopped), checked once."""
    f, part = dest / n, dest / (n + ".part")
    if not f.exists():
        say(f"  downloading {n} ({size / 1e9:.1f} GB) ...")
        for attempt in range(200):
            r = subprocess.run(["curl", "-L", "-C", "-", "--connect-timeout", "30", "--speed-limit", "20000",
                                "--speed-time", "60", "--progress-bar", "-o", str(part), base + n])
            if r.returncode == 0 or (part.exists() and part.stat().st_size >= size):
                break
            say(f"  the download stopped (curl {r.returncode}); continuing ...")
        if not part.exists() or part.stat().st_size != size:
            fail(f"{n}: {part.stat().st_size if part.exists() else 0} bytes of {size}. Run this again to continue.")
        say(f"  checking {n} (SHA-256) ...")
        if sha256_of(part) != sha:
            part.unlink()
            fail(f"{n} does not match the hash Hugging Face publishes; it was deleted. Run this again.")
        part.rename(f)


def main():
    ap = argparse.ArgumentParser(description="Set up Strata's Metal engine on an Apple silicon Mac (experimental).")
    ap.add_argument("--model", choices=list(MODELS), default="IQ3_S", help="the size (default IQ3_S)")
    ap.add_argument("--context", type=int, default=32768, help="context in tokens (default 32768)")
    ap.add_argument("--data-dir", default=None, help="where the model files go (default: Strata-data next to Strata)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--reserve-gib", type=float, default=None,
                    help="of the memory that is free when the engine starts, how much it leaves free (the engine's "
                         "default: 3)")
    ap.add_argument("--expert-io", choices=("direct", "cached"), default=None,
                    help="read missed experts past the OS file cache (default) or through it")
    ap.add_argument("--idle-unload", type=int, default=900, metavar="SECONDS",
                    help="give the memory back after this long without a request; the next request loads the model "
                         "again in a few seconds (default 900, 0 = keep it loaded)")
    ap.add_argument("--vision", choices=("yes", "no"), default="yes",
                    help="pictures: download the vision projector (0.91 GB) and give the server the image encoder, "
                         "which starts for each request with pictures (default yes)")
    ap.add_argument("--yes", action="store_true", help="no questions")
    ap.add_argument("--no-start", action="store_true", help="install only")
    ap.add_argument("--no-browser", action="store_true", help="do not open the web app when the model is ready")
    ap.add_argument("--check", action="store_true", help="only check this Mac")
    a = ap.parse_args()

    data = Path(a.data_dir).expanduser().resolve() if a.data_dir else ROOT.parent / "Strata-data"
    tag = a.model.lower()
    say("Strata for Apple silicon (Metal engine, experimental)\n\n[1] this Mac")
    m, free = check(data, a.model)
    if a.check:
        return
    shards = [data / "models" / a.model / FILE.format(q=a.model, i=i) for i in (1, 2)]
    vision = a.vision == "yes"
    get_model = not all(s.exists() for s in shards)
    get_mmproj = vision and not (data / "models" / "mmproj" / MMPROJ).exists()
    if get_model or get_mmproj:
        need = (MODELS[a.model][0] if get_model else 0.0) + (0.91 if get_mmproj else 0.0)
        if free < need + 2:
            fail(f"{need:.1f} GB to download and {data} has {free:.0f} GB free.")
        if not a.yes:
            what = " and ".join(([f"{a.model} ({MODELS[a.model][0]:.0f} GB)"] if get_model else []) +
                                (["the vision projector (0.91 GB)"] if get_mmproj else []))
            ans = input(f"  Download {what} from huggingface.co/{REPO} into {data}? [Y/n] ").strip().lower()
            if ans not in ("", "y", "yes"):
                fail("nothing was downloaded.")

    say("\n[2] the engine")
    exe = build_engine()
    say(f"  {exe}")

    say("\n[3] the model")
    shards = download(a.model, data / "models" / a.model)
    mmproj = download_mmproj(data / "models" / "mmproj") if vision else None

    say("\n[4] the tokenizer")
    pack = data / "packs" / tag
    if not (pack / "tokenizer" / "vocab.json").exists() or not (pack / "tokenizer" / "chat_template.jinja").exists():
        pack.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]),
                            "--out", str(pack)], capture_output=True, text=True)
        if r.returncode != 0:
            say(r.stdout + r.stderr)
            fail("the tokenizer could not be exported from the model file.")
    say(f"  {pack / 'tokenizer'}")

    say("\n[5] the start script")
    args = ["--native", str(shards[0]), "--expert-profile", str(ROOT / "data" / "expert-profile.bin"),
            "--expert-cache", "auto", "--max-context", str(a.context)]
    if a.reserve_gib is not None:
        args += ["--reserve-gib", f"{a.reserve_gib:g}"]
    if a.expert_io:
        args += ["--expert-io", a.expert_io]
    if mmproj:
        args.append("--vision")
    cfg = {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"qwen3.8-flash-next-{tag}", "log": str(ROOT / f"strata-{tag}.log"), "port": a.port,
           "backend": "metal"}
    if mmproj:
        # Pictures: the encoder (Metal, 1.1 GB while it runs) starts for each request with pictures and ends after
        # it, so it never takes the expert cache's memory.  Up to 2,048 tokens a picture: a 1280x1280 photo keeps its
        # full size (1,600 tokens; at 1,024 the model misread the small print on it).
        cfg["vision"] = {"exe": str(exe.parent / "strata-vision"), "mmproj": str(mmproj), "model": str(shards[0]),
                         "gpu": True, "max_tokens": 2048, "on_demand": True}
    if a.idle_unload > 0:
        # a Mac's memory is the engine's and every other program's: the model holds 12-21 GB of it only while in use
        cfg["idle_unload_s"] = a.idle_unload
    cfg_path = ROOT / f"strata-{tag}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    serve = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
             "--port", str(a.port)] + ([] if a.no_browser else ["--open"])
    script = ROOT / f"run-{tag}.sh"
    script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n",
                      encoding="utf-8")
    script.chmod(0o755)
    say(f"  config: {cfg_path}\n  start script: {script}")
    if a.no_start:
        say(f"\nReady. Start it with ./{script.name} ; the web app and the API are at http://127.0.0.1:{a.port}")
        return
    say("\nStarting the model. The first minute fills the expert cache from the SSD; the Mac stays usable.\n")
    os.execv(str(script), [str(script)])


if __name__ == "__main__":
    main()
