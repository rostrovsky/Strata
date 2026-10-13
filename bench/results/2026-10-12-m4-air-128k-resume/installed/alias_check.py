"""alias_check.py OUT: the server under the model's right name: /v1/models, a chat request under the config's
model_name, one under the old alias (must still be answered), and pi's own smoke test (a real `read` tool call)."""
import json, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path("/Users/rafal/Projects/strata-air/Strata")
out = sys.argv[1]
name = json.load(open(ROOT / "strata-iq3_s.json"))["model_name"]
console = open(out + "-console.txt", "w")
srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "serve/server.py"), "--engine", "strata", "--config",
                        str(ROOT / "strata-iq3_s.json"), "--port", "8080"], cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
def get(path):
    with urllib.request.urlopen("http://127.0.0.1:8080" + path, timeout=60) as r: return json.load(r)
def ask(model, text, max_tokens=48):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: d = json.load(r)
    return d.get("model"), (d["choices"][0]["message"].get("content") or "")[:120]
ok = False
try:
    for _ in range(180):
        if "ready: http" in Path(out + "-console.txt").read_text(errors="replace"): break
        if srv.poll() is not None: sys.exit("the server ended before it was ready")
        time.sleep(1)
    models = [m["id"] for m in get("/v1/models")["data"]]
    print("/v1/models:", models)
    a = ask(name, "Reply with one short sentence: what is the capital of France?")
    print(f"asked as {name}: answered as {a[0]}: {a[1]!r}")
    b = ask("qwen3.8-27b-iq3_s", "Reply with the single word: ready")
    print(f"asked as qwen3.8-27b-iq3_s (the old name): answered as {b[0]}: {b[1]!r}")
    pi = subprocess.run(["/Users/rafal/.bun/bin/pi", "--provider", "qwen-local", "--model", name, "--offline", "--no-session", "--thinking", "off",
                         "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files", "--tools", "read", "--mode", "json", "--print",
                         "Use read to open /Users/rafal/Projects/strata-air/Strata/build-metal/lowlevel-128k/pi-smoke.txt, then return its exact content."],
                        capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    Path(out + "-pi.jsonl").write_text(pi.stdout)
    Path(out + "-pi.err").write_text(pi.stderr[-4000:])
    called = '"toolName":"read"' in pi.stdout.replace(" ", "") or '"name":"read"' in pi.stdout.replace(" ", "")
    marker = "STRATA_PI_READ_CONFIRMED_7c4a91e2" in pi.stdout
    print(f"pi (exit {pi.returncode}): read tool called: {called}; marker returned: {marker}")
    ok = name in models and "qwen3.8-27b-iq3_s" not in models and a[0] == name and bool(a[1]) and called and marker
finally:
    srv.send_signal(signal.SIGINT)
    try: srv.wait(timeout=60)
    except subprocess.TimeoutExpired: srv.kill()
print("all as expected" if ok else "NOT AS EXPECTED")
sys.exit(0 if ok else 1)
