"""DeepSeek (Ollama cloud) text-only helper. The key is read from OLLAMA_API_KEY or a .env file; never printed."""
import json, os, time, urllib.request

ENV_CANDIDATES = [os.environ.get("DV_ENV", ""), ".env", "~/Developer/decision-vison/.env"]
MODEL = os.environ.get("DV_LLM", "deepseek-v4.1-flash")


def key():
    k = os.environ.get("OLLAMA_API_KEY")
    if k:
        return k
    for p in ENV_CANDIDATES:
        if p and os.path.exists(p):
            for l in open(p):
                if l.strip().startswith("OLLAMA_API_KEY="):
                    return l.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OLLAMA_API_KEY not found")


def chat(system, user, model=MODEL, temperature=0.7, timeout=240, retries=4):
    body = json.dumps(dict(model=model, messages=[dict(role="system", content=system), dict(role="user", content=user)],
                           stream=False, format="json", think=False, options=dict(temperature=temperature))).encode()
    err = None
    for i in range(retries):
        try:
            req = urllib.request.Request("https://ollama.com/api/chat", data=body,
                                         headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(json.loads(r.read())["message"]["content"])
        except Exception as e:  # the message never contains the key
            err = type(e).__name__ + ": " + str(e)[:150]
            time.sleep(10 * (i + 1))
    raise RuntimeError(err)


if __name__ == "__main__":
    print(chat("Return JSON.", 'Return {"zh": "<colloquial Chinese for: red jacket on the left>", '
                               '"fn": "<what a drinking glass is used for, 3 words>"}'))
