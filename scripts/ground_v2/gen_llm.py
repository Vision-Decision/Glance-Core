"""Module 2 (LLM): DeepSeek writes phrasings per numbered object / group, then a second call verifies every sentence.

The LLM only writes sentences; answers (boxes, presence) come from Module 1/3 rules. The model sees the photo
(deepseek-v4.1-flash on the Ollama cloud API has vision) twice: with numbered boxes drawn, and clean.

  1. gen    : per object short / medium / long / colloquial / zh_short / zh / function / zh_function phrasings,
              observed colour + action, colour_neg / action_neg (medium with only the colour / action changed);
              per multi-target group one EN and one ZH phrase naming every target.
  2. verify : a fresh call gets all generated sentences shuffled, without their intended targets, and says which
              numbered objects each one refers to ([] = matches nothing, "other" = unnumbered object / ambiguous).
              Module 3 keeps a positive only if the verified set == intended set, a negative only if verified [].

Reads OLLAMA_API_KEY from the environment (never printed or written). Resumable: image ids already in --out are skipped.
  python scripts/ground_v2/gen_llm.py --scenes scenes.jsonl --img_root ~/gv2/img --out gen.jsonl --workers 12
"""
import argparse, base64, io, json, os, random, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

GEN_SYS = """You write training data for a visual-grounding model. You see a COCO photo twice: first with numbered boxes drawn on some objects, then the clean photo. The JSON input lists the numbered objects (category, box as percent of width/height [x1,y1,x2,y2], how many objects of that category the photo has, human-written referring expressions when available), the photo captions, and every annotated category with its count.

For EACH numbered object return:
- "colour": main colour(s) of the object as actually seen in the photo, or null if unclear (dark, backlit, tiny, mixed).
- "action": what it is doing or its state if it is a person, animal or vehicle (e.g. "sitting", "eating grass", "parked"), only if clearly visible; else null.
- "short": 1-4 words that pick out THIS object and nothing else in the photo ("the red bus", "left zebra", "blue mug"). If it is the only object of its kind, the plain noun is fine ("the giraffe").
- "medium": 5-9 words, natural, identifies it uniquely (attribute and/or location or relation to another object).
- "long": 12-25 words, one descriptive phrase or sentence with several visible details that still identifies it uniquely.
- "colloquial": how someone would casually say it out loud in English, may use slang or a vague word ("that little pup on the couch", "the dude in the red cap"); still unique.
- "zh_short": Chinese, 2-8 characters, uniquely identifying (like short).
- "zh": natural colloquial Chinese, medium length, uniquely identifying.
- "function": refer to it ONLY by what it is used for / what you can do with it, WITHOUT naming the object, its category, a synonym or a hypernym ("something to sit on", "the thing you drink from on the left", "what you'd use to cut the cake"). It must not also fit any other object in the photo; add colour or location if needed. null for people/animals or if impossible.
- "zh_function": the same idea in Chinese without the category name ("能坐的东西", "左边那个喝水用的"), or null.
- "colour_neg": copy "medium" but change ONLY the colour word to a clearly different colour that this object does NOT have (if medium has no colour, insert a wrong colour before the noun). The result must not describe any object in the photo. null if "colour" is null.
- "action_neg": only for a person or animal: copy "medium" and REPLACE its action/pose words with one the object is clearly NOT doing and that is visibly checkable in a still photo (standing -> lying down, sitting -> jumping, sleeping -> eating). If medium has no action, insert one wrong action. Do not keep the old action next to the new one; the phrase must stay grammatical and not contradict itself. null for objects, vehicles, or if "action" is null.

For EACH group return "en" and "zh": one natural phrase that refers to exactly those numbered objects together, to nothing else, and names every one of them ("the two zebras on the left", "the man and his surfboard", "左边那两匹斑马", "那个男的和他的冲浪板"). Use a number word ("two", "both", "两") for groups of the same kind.

Rules: only describe what is visible; never mention the drawn boxes, numbers, labels or "box"; left/right are from the viewer (x grows to the right) and must be clearly true; do not guess identities, and use gender/age words only when obvious; every positive phrase must refer to exactly one object (the numbered one), judged against ALL objects in the photo including unnumbered ones; all phrases of one object must be worded differently.
Return ONLY JSON: {"objects": [{"id": 1, "colour": ..., ...}], "groups": [{"gid": "g1", "en": ..., "zh": ...}]}"""

VER_SYS = """You check referring expressions against a photo. The photo is shown first with numbered boxes on some objects, then clean. For EACH expression decide what it refers to in the photo and answer:
- a list of the numbers of ALL numbered objects it refers to (one number for a single object, several if it clearly means several objects together);
- [] if it does not truly describe anything in the photo (wrong colour, wrong action or state, wrong number of objects, wrong side, object not there);
- "other" if it refers to something that has no number, or if it is ambiguous (fits several objects but does not mean all of them), or if you cannot tell.
Be strict about colours, actions, counts and left/right (from the viewer). Chinese and English expressions are judged the same way. Do not reward vagueness: if a phrase like "something to sit on" fits two objects, answer "other".
Return ONLY JSON: {"results": [{"sid": "s1", "refers": [2]}, {"sid": "s2", "refers": []}, {"sid": "s3", "refers": "other"}]}"""

POS_FIELDS = ["short", "medium", "long", "colloquial", "zh_short", "zh", "function", "zh_function"]
NEG_FIELDS = ["colour_neg", "action_neg"]
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60)]


def font(sz):
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


def b64(img, q=90):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def marked(img, nums):
    """nums: [(num, box xyxy px)] -> copy with numbered boxes"""
    im = img.copy(); d = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(2, round(max(W, H) / 250)); f = font(max(14, round(max(W, H) / 28)))
    for n, b in nums:
        c = COLORS[(n - 1) % len(COLORS)]
        d.rectangle(b, outline=c, width=lw)
        t = str(n); tw, th = d.textbbox((0, 0), t, font=f)[2:]
        x, y = max(0, b[0]), max(0, b[1] - th - 6) if b[1] - th - 6 >= 0 else max(0, b[1])
        d.rectangle([x, y, x + tw + 8, y + th + 6], fill=c)
        d.text((x + 4, y + 2), t, fill=(255, 255, 255), font=f)
    return im


def pbox(b, W, H):
    return [round(100 * b[0] / W), round(100 * b[1] / H), round(100 * b[2] / W), round(100 * b[3] / H)]


def gen_payload(sc):
    W, H = sc["W"], sc["H"]
    cnt = {}
    for o in sc["objects"]:
        cnt[o["cat"]] = cnt.get(o["cat"], 0) + 1
    objs = [dict(id=o["num"], category=o["cat"], box=pbox(o["box"], W, H), count_of_category_in_photo=cnt[o["cat"]],
                 human_refs=o["refs"][:3]) for o in sc["objects"] if o.get("describe")]
    groups = [dict(gid=f"g{i + 1}", ids=g["nums"]) for i, g in enumerate(sc["groups"])]
    others = [dict(id=o["num"], category=o["cat"], box=pbox(o["box"], W, H)) for o in sc["objects"] if o.get("num") and not o.get("describe")]
    d = dict(numbered_objects=objs, other_numbered_objects_only_for_groups=others, groups=groups, captions=sc["captions"],
             all_categories_in_photo=cnt)
    return d


def parse_json(s):
    """the model sometimes wraps the object in ```json fences despite format=json"""
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j <= i:
            raise
        return json.loads(s[i:j + 1])


class Client:
    def __init__(self, model, key):
        self.model, self.key = model, key
        self.lock = threading.Lock(); self.usage = dict(prompt=0, out=0, calls=0, retries=0)

    def chat(self, sys_msg, user, images, temperature):
        body = json.dumps(dict(model=self.model, stream=False, format="json", think=False,
                               options=dict(temperature=temperature),
                               messages=[dict(role="system", content=sys_msg), dict(role="user", content=user, images=images)])).encode()
        err = None
        for attempt in range(40):  # the endpoint caps concurrent requests per key (429): retry soon, don't idle
            req = urllib.request.Request("https://ollama.com/api/chat", data=body,
                                         headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    res = json.loads(r.read())
                with self.lock:
                    self.usage["prompt"] += res.get("prompt_eval_count") or 0
                    self.usage["out"] += res.get("eval_count") or 0
                    self.usage["calls"] += 1
                return parse_json(res["message"]["content"]), (res.get("prompt_eval_count") or 0, res.get("eval_count") or 0)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ConnectionError) as e:
                err = f"{type(e).__name__}: {getattr(e, 'code', '')} {str(e)[:100]}"  # never contains the key
                with self.lock:
                    self.usage["retries"] += 1
                    self.usage[f"err_{getattr(e, 'code', type(e).__name__)}"] = self.usage.get(f"err_{getattr(e, 'code', type(e).__name__)}", 0) + 1
                busy = getattr(e, "code", None) == 429
                time.sleep((2 + 6 * random.random()) if busy else min(60, 5 * 2 ** min(attempt, 4)) * (0.5 + random.random()))
        raise RuntimeError(err)


def fetch_image(sc, root):
    p = f"{root}/{sc['year']}/{sc['file']}"
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        for attempt in range(4):
            try:
                data = urllib.request.urlopen(sc["url"].replace("https://", "http://"), timeout=60).read()
                with open(p + ".tmp", "wb") as f:
                    f.write(data)
                os.replace(p + ".tmp", p)
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
    return Image.open(p).convert("RGB")


def candidates(sc, gen):
    """flatten generated sentences: [(sid, text, kind, field, intended_nums or [])]"""
    out = []
    objs = {o.get("id"): o for o in gen.get("objects", []) if isinstance(o, dict)}
    for o in sc["objects"]:
        if not o.get("describe"):
            continue
        g = objs.get(o["num"])
        if not g:
            continue
        for f in POS_FIELDS:
            if isinstance(g.get(f), str) and g[f].strip():
                out.append((f, g[f].strip(), "pos", [o["num"]], o["num"]))
        for f in NEG_FIELDS:
            if isinstance(g.get(f), str) and g[f].strip():
                out.append((f, g[f].strip(), "neg", [], o["num"]))
    gg = {x.get("gid"): x for x in gen.get("groups", []) if isinstance(x, dict)}
    for i, grp in enumerate(sc["groups"]):
        g = gg.get(f"g{i + 1}")
        if not g:
            continue
        for f in ("en", "zh"):
            if isinstance(g.get(f), str) and g[f].strip():
                out.append((f"group_{f}", g[f].strip(), "pos", sorted(grp["nums"]), f"g{i + 1}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_verify", action="store_true")
    a = ap.parse_args()
    key = os.environ.get("OLLAMA_API_KEY")
    assert key, "OLLAMA_API_KEY not set"
    cl = Client(a.model, key)
    scenes = [json.loads(l) for l in open(a.scenes)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["image_id"] for l in open(a.out)}
    todo = [s for s in scenes if s["image_id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(todo)} scenes to do ({len(done)} done)", flush=True)
    lock = threading.Lock(); fout = open(a.out, "a"); n_ok = [0, 0]
    t0 = time.time()

    def work(sc):
        try:
            img = fetch_image(sc, a.img_root)
            nums = [(o["num"], o["box"]) for o in sc["objects"] if o.get("num")]
            ims = [b64(marked(img, nums)), b64(img)]
            gen, u1 = cl.chat(GEN_SYS, json.dumps(gen_payload(sc), ensure_ascii=False), ims, 0.7)
            rec = dict(image_id=sc["image_id"], model=a.model, gen=gen, usage=[u1])
            if not a.no_verify:
                cands = candidates(sc, gen)
                order = list(range(len(cands))); random.Random(sc["image_id"]).shuffle(order)
                sids = {f"s{k + 1}": i for k, i in enumerate(order)}
                exprs = [dict(sid=s, expression=cands[i][1]) for s, i in sids.items()]
                ver, u2 = cl.chat(VER_SYS, json.dumps(dict(expressions=exprs), ensure_ascii=False), ims, 0.0)
                res = {r.get("sid"): r.get("refers") for r in ver.get("results", []) if isinstance(r, dict)}
                rec["cands"] = [dict(field=c[0], text=c[1], kind=c[2], intended=c[3], owner=c[4],
                                     verified=res.get(next(s for s, i in sids.items() if i == j))) for j, c in enumerate(cands)]
                rec["usage"].append(u2)
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush(); n_ok[0] += 1
        except Exception as e:
            with lock:
                n_ok[1] += 1
                print("scene failed", sc["image_id"], type(e).__name__, str(e)[:150], flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for k, _ in enumerate(ex.map(work, todo)):
            if k % 10 == 0:
                print(f"{k}/{len(todo)} ok={n_ok[0]} fail={n_ok[1]} usage={cl.usage} {time.time() - t0:.0f}s", flush=True)
    print(f"done ok={n_ok[0]} fail={n_ok[1]} usage={cl.usage} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
