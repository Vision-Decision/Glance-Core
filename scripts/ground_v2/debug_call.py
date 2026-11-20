"""Debug one generation call: print done_reason, token counts and the raw content head/tail (never the key)."""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_llm import GEN_SYS, b64, fetch_image, gen_payload, marked

scenes = {json.loads(l)["image_id"]: json.loads(l) for l in open(sys.argv[1])}
sc = scenes[int(sys.argv[2])]
img = fetch_image(sc, sys.argv[3])
ims = [b64(marked(img, [(o["num"], o["box"]) for o in sc["objects"] if o.get("num")])), b64(img)]
fmt = sys.argv[4] if len(sys.argv) > 4 else "json"
body = dict(model="deepseek-v4.1-flash", stream=False, think=False, options=dict(temperature=0.7, num_predict=6000),
            messages=[dict(role="system", content=GEN_SYS), dict(role="user", content=json.dumps(gen_payload(sc)), images=ims)])
if fmt == "json":
    body["format"] = "json"
req = urllib.request.Request("https://ollama.com/api/chat", data=json.dumps(body).encode(),
                             headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"], "Content-Type": "application/json"})
t = time.time()
try:
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read()[:300]); sys.exit()
c = r["message"]["content"]
print(f"{time.time() - t:.0f}s", r.get("done_reason"), r.get("prompt_eval_count"), r.get("eval_count"), len(c))
print(repr(c[:300])); print(repr(c[-300:]))
try:
    json.loads(c); print("JSON OK")
except Exception as e:
    print("JSON FAIL", e)
