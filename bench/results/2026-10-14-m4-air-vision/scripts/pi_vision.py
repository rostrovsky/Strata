"""pi_vision.py OUT IMAGE: the installed configuration (strata-iq3_s.json, port 8080, as run-iq3_s.sh starts it but
without the browser) and pi with a picture attached (@IMAGE), thinking off.  Writes OUT-console.txt, OUT-pi.jsonl,
OUT-pi.err; prints what pi got back."""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/rafal/Projects/strata-air/Strata")
out, image = sys.argv[1], sys.argv[2]
console = open(out + "-console.txt", "w")
srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "serve/server.py"), "--engine", "strata", "--config",
                        str(ROOT / "strata-iq3_s.json"), "--port", "8080"], cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        if "ready: http" in Path(out + "-console.txt").read_text(errors="replace"):
            break
        if srv.poll() is not None:
            sys.exit("the server ended before it was ready")
        time.sleep(1)
    else:
        sys.exit("the server was not ready after 300 s")
    name = json.load(open(ROOT / "strata-iq3_s.json"))["model_name"]
    t0 = time.time()
    pi = subprocess.run(["/Users/rafal/.bun/bin/pi", "--provider", "qwen-local", "--model", name, "--offline", "--no-session",
                         "--thinking", "off", "--no-extensions", "--no-skills", "--no-prompt-templates",
                         "--no-context-files", "--tools", "read", "--mode", "json", "--print", f"@{image}",
                         "What instrument is in this picture, what brand and model number are printed on it, and what does "
                         "it read right now? Answer in three short lines."],
                        capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    Path(out + "-pi.jsonl").write_text(pi.stdout)
    Path(out + "-pi.err").write_text(pi.stderr[-4000:])
    text = ""
    for line in pi.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "agent_end":
            for m in ev.get("messages", []):
                if m.get("role") == "assistant":
                    text = "".join(c.get("text", "") for c in m.get("content", []) if c.get("type") == "text")
    print(f"pi (exit {pi.returncode}) in {time.time() - t0:.1f} s:\n{text or pi.stdout[-2000:]}", flush=True)
finally:
    srv.send_signal(2)
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        srv.kill()
