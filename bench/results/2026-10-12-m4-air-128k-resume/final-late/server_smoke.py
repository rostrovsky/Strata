"""server_smoke.py OUT: the installed configuration through serve/server.py, as pi reaches it: a short chat, a
long document (a long-prompt shape and back), a second turn on the same conversation.  Writes OUT-console.txt and
OUT.json; exit 0 if all three answered."""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path("/Users/rafal/Projects/strata-air/Strata")
out = sys.argv[1]
console = open(out + "-console.txt", "w")
srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "serve/server.py"), "--engine", "strata", "--config",
                        str(ROOT / "strata-iq3_s.json"), "--port", "8080"], cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
def ask(messages, max_tokens):
    body = json.dumps({"model": "qwen3.8-27b-iq3_s", "messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return {"seconds": round(time.time() - t0, 2), "usage": d.get("usage"), "finish": d["choices"][0].get("finish_reason"),
            "content": (m.get("content") or "")[:400], "reasoning": (m.get("reasoning_content") or "")[:200]}
results, ok = [], False
try:
    for _ in range(180):
        if "ready: http" in Path(out + "-console.txt").read_text(errors="replace"): break
        if srv.poll() is not None: sys.exit("the server ended before it was ready")
        time.sleep(1)
    else:
        sys.exit("the server was not ready after 180 s")
    doc = (ROOT / "docs/DETAILS.md").read_text(encoding="utf-8")[:14000]
    first = [{"role": "user", "content": "Reply with one short sentence: what is the capital of France?"}]
    results.append(("short chat", ask(first, 64)))
    long = [{"role": "user", "content": "Here is a document.\n\n" + doc + "\n\nIn one sentence: what is this document about?"}]
    results.append(("long document", ask(long, 96)))
    again = long + [{"role": "assistant", "content": results[-1][1]["content"]}, {"role": "user", "content": "Name one number it mentions."}]
    results.append(("second turn", ask(again, 64)))
    ok = all(r["content"] or r["reasoning"] for _, r in results)
finally:
    srv.send_signal(signal.SIGINT)
    try: srv.wait(timeout=60)
    except subprocess.TimeoutExpired: srv.kill()
    json.dump(results, open(out + ".json", "w"), indent=1)
    for name, r in results: print(f"{name}: {r['seconds']} s, {r['usage']}, finish {r['finish']}: {r['content'][:110]!r}")
sys.exit(0 if ok else 1)
