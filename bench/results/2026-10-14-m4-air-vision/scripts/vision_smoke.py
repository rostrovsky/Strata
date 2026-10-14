"""vision_smoke.py OUT IMAGE: the production engine with a vision section (strata-iq3_s-vision.json, port 8091)
through serve/server.py: a picture and a question without thinking, a second turn on the same picture, and the
same question with thinking.  Writes OUT-console.txt and OUT.json."""
import base64, json, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path("/Users/rafal/Projects/strata-air/Strata")
out, image = sys.argv[1], sys.argv[2]
CFG = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "build-metal/vision/strata-iq3_s-vision.json"
TEXT_ONLY = len(sys.argv) > 4 and sys.argv[4] == "text"
THINK = len(sys.argv) > 4 and sys.argv[4] == "think"
console = open(out + "-console.txt", "w")
srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "serve/server.py"), "--engine", "strata", "--config",
                        str(CFG), "--port", "8091"], cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
model = json.load(open(CFG))["model_name"]
url = "data:image/webp;base64," + base64.b64encode(Path(image).read_bytes()).decode()


def ask(messages, max_tokens, think, greedy=True):
    req = {"model": model, "messages": messages, "max_tokens": max_tokens,
           "chat_template_kwargs": {"enable_thinking": think}}
    if greedy:
        req["temperature"] = 0
    body = json.dumps(req).encode()
    req = urllib.request.Request("http://127.0.0.1:8091/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    res = {"seconds": round(time.time() - t0, 2), "usage": d.get("usage"), "finish": d["choices"][0].get("finish_reason"),
           "content": m.get("content") or "", "reasoning": m.get("reasoning_content") or ""}
    print(json.dumps({k: v for k, v in res.items() if k != "reasoning"}, indent=1), flush=True)
    return res


results = []
try:
    for _ in range(300):
        if "ready: http" in Path(out + "-console.txt").read_text(errors="replace"):
            break
        if srv.poll() is not None:
            sys.exit("the server ended before it was ready")
        time.sleep(1)
    else:
        sys.exit("the server was not ready after 300 s")
    question = ("Describe this picture in detail: what is the object, its brand and any text or model number you can "
                "read on it, and what measurement it shows right now.")
    first = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                          {"type": "text", "text": question}]}]
    if THINK:       # the model's own sampling (temperature 1, top-p 0.95, top-k 20), room to finish
        results.append(ask(first, 8000, True, greedy=False))
        sys.exit(0)
    if TEXT_ONLY:   # the small print, then the same description question as before
        results.append(ask([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                         {"type": "text", "text": "Read out every piece of text printed "
                                                          "on the yellow dial face, exactly as written, line by line."}]}],
                           200, False))
        results.append(ask(first, 700, False))
        sys.exit(0)
    r1 = ask(first, 700, False)
    results.append(r1)
    second = first + [{"role": "assistant", "content": r1["content"]},
                      {"role": "user", "content": "What is the resolution (smallest division) of the dial, and how "
                                                  "would you know the jaws are fully closed?"}]
    results.append(ask(second, 400, False))
    results.append(ask(first, 3000, True))
finally:
    json.dump(results, open(out + ".json", "w"), indent=1)
    srv.send_signal(2)
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        srv.kill()
