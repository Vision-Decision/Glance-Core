"""Functional paraphrases of element names with DeepSeek via Ollama cloud (TEXT ONLY: app, role, name, context).

Run locally (where .env holds OLLAMA_API_KEY; the key is read into this process's environment only, never printed or
written anywhere):
  python scripts/guisyn/paraphrase.py keys  --host gen-host --out /tmp/para_keys.json     # unique element keys from shards
  python scripts/guisyn/paraphrase.py run   --keys /tmp/para_keys.json --cache /tmp/paraphrases.json [--n 4000]
  scp /tmp/paraphrases.json gen-host:/data/guisyn/paraphrases.json                          # build_shards picks it up
Output: {"app\\trole\\tname": "the button that ...", ...}; null = model judged the label too generic.
Icon-only elements without names are never sent (they are marked `uncaptioned` for later GPU captioning).
"""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys, time, urllib.request

MODEL = "deepseek-v4.1-flash"
URL = "https://ollama.com/api/chat"
PROMPT = """You write functional referring expressions for GUI elements (for a visual grounding dataset).
For each item you get the application, the accessibility role, the element's label, and where it is.
Write ONE short noun phrase (max 12 words) that identifies the element by what it does or is for, e.g.
"the button that saves the current document", "the menu with options for adjusting image colors",
"the field for typing a search query". Do not just repeat the label in quotes; you may reuse its words.
If the purpose cannot be inferred with confidence (numbers, file names, user content, ambiguous labels), use null.
Return ONLY a JSON array: [{"id": <id>, "expr": <string or null>}, ...]

Items:
"""


def load_env(path):
    for line in open(path):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*$", line)
        if m and m.group(1) == "OLLAMA_API_KEY":
            os.environ["OLLAMA_API_KEY"] = m.group(2).strip().strip("'\"")


def keys_cmd(a):
    code = r'''
import glob, json, collections
c = collections.Counter(); ctx = {}
for f in glob.glob("/data/guisyn/stage/shards/desktop/*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        for q in r["questions"]:
            if q["present"] and q["kind"] in ("name", "icon") and q["meta"].get("name"):
                k = "\t".join([r["app"], q["meta"]["role"], q["meta"]["name"]])
                c[k] += 1
                if q["meta"].get("tmpl") == "context": ctx[k] = q["text"]
print(json.dumps([[k, n, ctx.get(k, "")] for k, n in c.most_common()]))
'''
    out = subprocess.run(["ssh", a.host, "/data/guisyn/venv/bin/python -"], input=code, capture_output=True, text=True, check=True).stdout
    json.dump(json.loads(out), open(a.out, "w"))
    print("keys", len(json.loads(out)))


def call(batch):
    items = "\n".join(json.dumps(dict(id=i, app=k.split("\t")[0], role=k.split("\t")[1], label=k.split("\t")[2],
                                      context=ctx)) for i, (k, ctx) in batch)
    body = json.dumps(dict(model=MODEL, stream=False, options=dict(temperature=0.3),
                           messages=[dict(role="user", content=PROMPT + items)])).encode()
    req = urllib.request.Request(URL, data=body, headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"],
                                                          "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                txt = json.load(r)["message"]["content"]
            m = re.search(r"\[.*\]", txt, re.S)
            return {int(x["id"]): x.get("expr") for x in json.loads(m.group(0))}
        except Exception as e:
            err = type(e).__name__
            time.sleep(5 * (attempt + 1))
    print("batch failed:", err, file=sys.stderr)
    return {}


def run_cmd(a):
    load_env(a.env)
    keys = json.load(open(a.keys))[: a.n]
    cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    todo = [(k, ctx) for k, n, ctx in keys if k not in cache]
    batches = [list(enumerate(todo))[i:i + 40] for i in range(0, len(todo), 40)]
    print("todo", len(todo), "batches", len(batches))
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for b, res in zip(batches, ex.map(call, batches)):
            for i, (k, ctx) in b:
                if i in res:
                    e = res[i]
                    cache[k] = e.strip() if isinstance(e, str) and 3 <= len(e) <= 120 else None
            json.dump(cache, open(a.cache, "w"))
    print("cached", len(cache), "non-null", sum(v is not None for v in cache.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["keys", "run"])
    ap.add_argument("--host", default="gen-host"); ap.add_argument("--out", default="/tmp/para_keys.json")
    ap.add_argument("--keys", default="/tmp/para_keys.json"); ap.add_argument("--cache", default="/tmp/paraphrases.json")
    ap.add_argument("--env", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    keys_cmd(a) if a.cmd == "keys" else run_cmd(a)
