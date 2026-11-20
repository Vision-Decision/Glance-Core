"""v1.2 category-free grounding data, step 2 (LLM, self-hosted vLLM on a Colab GPU): write the wording, then verify blind.

Per image, one *writer* call and one *verifier* call to the same open VLM (default Qwen/Qwen3.8-27B-FP8, thinking off):
  writer   sees the image with every annotated object numbered, the object list (name + 0-1000 box) and the rule-made
           tasks of catfree_scenes.py; returns n phrasings per task. It never decides the answer.
  verifier sees the same numbered image and object list, plus every phrasing of the image shuffled, without any task or
           target. For each phrasing it answers the object number(s) it refers to, "none" (nothing visible in the image
           fits, numbered or not) or "other" (an unnumbered thing / ambiguous). catfree_build.py keeps a positive only if
           the answer is exactly its target, a negative only if the answer is "none".

  python scripts/v12/catfree_gen.py --scenes scenes.jsonl --img_root /content/data/coco/train2014 --out gen.jsonl
Resumable (image ids already in --out are skipped). Runs in its own venv (vllm pins torch); no API key involved.
"""
import argparse, base64, io, json, os, random, re, sys, time

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catfree_scenes import CLASSES, REL_EN, REL_ZH  # noqa: E402

LANG = {"en": "English", "zh": "natural, colloquial Simplified Chinese (as a native speaker would say it)"}

WRITER = """You write search queries for a visual-grounding model. The user asks the model to find ONE thing in a photo
(or to say it is not there). Objects in the photo are numbered with boxes; the list gives each number, its category and its
box (x1,y1,x2,y2 on a 0-1000 scale; x grows to the right, y downwards). For every task write `n` DIFFERENT queries in the
requested language, following the task kind exactly:

- "function": refer to object `target` ONLY by what it is used for / what you can do with it (its function or affordance),
  like someone who does not know its name: "something to drink from", "what I could sit on", "能装水的东西", "可以用来切东西的".
  No name of its category, no synonym, no more specific name (breed, brand, model), no colour, no position. The query must
  NOT also fit any other object in the photo (look at the image, not only the list). 2-9 words (Chinese: 3-14 characters).
- "generic": refer to object `target` with a broad class word only, e.g. one of `class_words` or an equally broad everyday
  word ("the animal", "that vehicle", "那个家具", "吃的"); no category name, synonym, attribute or position. 1-5 words.
- "relation": refer to object `target` with a broad class word (from `class_words` or equally broad, e.g. "thing", "东西"),
  NOT its category name or a synonym, located by `relation` relative to the anchor object, which you name with
  `anchor_name` or a plain synonym of it. Directions are as seen in the photo (left = left side of the image).
  Vary the syntax: "the animal to the left of the fridge", "冰箱左边的那只动物", "what's sitting on the table, that thing".
- "absent_function": the category `absent` is NOT in the photo. Describe what it is used for (without naming it, a
  synonym or a hypernym) so that the query fits NOTHING visible in the photo, numbered or not. E.g. "something to cut paper
  with", "能用来打电话的东西". If every function of `absent` is served by something visible, return no query.
- "absent_generic": no object of the class in `class_words` is in the photo. Write a natural query for such a thing with a
  class word only ("the vehicle", "那只动物"); only if truly nothing visible (numbered or not) belongs to the class.
- "absent_relation": a query in the same style as "relation" (class word + `relation` + `anchor_name`) that fits NOTHING
  in the photo: there are members of the class, but none is in that relation to the anchor (for the class "thing": nothing
  at all is there).
- "absent_relation_noclass": the same style, but nothing of the class in `class_words` is anywhere in the photo
  ("the animal to the left of the bench" in a photo without animals).
- "absent_position": the category `absent` is NOT in the photo. Write a natural query that names it (its name or a plain
  synonym) together with the position word / superlative `position` ("the suitcase on the left side", "the biggest giraffe",
  "最左边的长颈鹿"). Do not add other attributes.

General rules: never mention numbers, boxes, labels or the list; never use a category name for the thing being looked for
(you may name the anchor in relation tasks); the n queries of one task must differ in wording, not just articles; write each
query in the task's language only. Return ONLY JSON: {"results": [{"tid": "t0", "texts": ["...", "..."]}, ...]} with one
entry per task (an empty "texts" list if a task cannot be done under the rules)."""

VERIFIER = """You check search queries for a visual-grounding dataset. Objects in the photo are numbered with boxes; the list
gives each number, its category and its box (x1,y1,x2,y2 on a 0-1000 scale; x grows to the right). For EACH query decide what
a careful person would point at in the photo:
- a list with the number(s) of the object(s) that the query fits (every numbered object it fits, not just the best one);
- "none" if nothing visible in the photo fits the query at all, numbered or not;
- "other" if it fits something visible that is not numbered, or it is too ambiguous to decide.
Directions are as seen in the photo. Judge each query on its own. Return ONLY JSON:
{"answers": [{"qid": "q0", "ids": [3]}, {"qid": "q1", "ids": "none"}, {"qid": "q2", "ids": "other"}, ...]}"""


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default(size=sz)


PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)]


def numbered_image(sc, root, long_side=896):
    img = Image.open(f"{root}/{sc['file']}").convert("RGB")
    s = long_side / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    d = ImageDraw.Draw(img)
    f = font(max(14, long_side // 40))
    for o in sc["objs"]:
        c = PALETTE[(o["id"] - 1) % len(PALETTE)]
        b = [v * s for v in o["box"]]
        d.rectangle(b, outline=c, width=3)
        t = str(o["id"])
        tw, th = d.textbbox((0, 0), t, font=f)[2:]
        x, y = b[0] + 2, max(0, b[1] - th - 4) if b[1] > th + 6 else b[1] + 2
        d.rectangle([x - 2, y - 1, x + tw + 2, y + th + 3], fill=c)
        d.text((x, y), t, fill=(0, 0, 0) if sum(c) > 400 else (255, 255, 255), font=f)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def obj_list(sc):
    W, H = sc["W"], sc["H"]
    nb = lambda b: [round(1000 * b[0] / W), round(1000 * b[1] / H), round(1000 * b[2] / W), round(1000 * b[3] / H)]
    return [dict(id=o["id"], category=o["cat"] + (" (crowd)" if o["crowd"] else ""), box=nb(o["box"])) for o in sc["objs"]]


def task_payload(sc, t, n_pos, n_neg):
    obj = {o["id"]: o for o in sc["objs"]}
    k = t["kind"]
    p = dict(tid=t["tid"], language=LANG[t["lang"]])
    cw = lambda c: CLASSES[c][1] + CLASSES[c][2] if t["lang"] == "zh" else CLASSES[c][1]
    if k == "func_pos":
        p.update(kind="function", target=t["target"], n=n_pos)
    elif k == "generic_pos":
        p.update(kind="generic", target=t["target"], class_words=cw(t["cls"]), n=n_pos)
    elif k == "rel_pos":
        p.update(kind="relation", target=t["target"], anchor=t["anchor"], anchor_name=obj[t["anchor"]]["cat"],
                 relation=REL_EN[t["rel"]], class_words=cw(t["cls"]), n=n_pos)
    elif k == "func_neg":
        p.update(kind="absent_function", absent=t["absent_cat"], n=n_neg)
    elif k == "generic_neg":
        p.update(kind="absent_generic", class_words=cw(t["cls"]), n=n_neg)
    elif k == "rel_neg":
        p.update(kind="absent_relation_noclass" if t.get("sub") == "rel_absent" else "absent_relation", anchor=t["anchor"],
                 anchor_name=obj[t["anchor"]]["cat"], relation=REL_EN[t["rel"]], class_words=cw(t["cls"]), n=n_neg)
    elif k == "pos_absent":
        p.update(kind="absent_position", absent=t["absent_cat"], position=t["position"], n=n_neg)
    return p


def parse_json(s):
    s = re.sub(r"^```(json)?|```$", "", s.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def msgs(system, url, text):
    return [dict(role="system", content=system),
            dict(role="user", content=[dict(type="image_url", image_url=dict(url=url)), dict(type="text", text=text)])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--n_pos", type=int, default=3)
    ap.add_argument("--n_neg", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/k: this process handles scenes with index %% k == i")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    si, sk = map(int, a.shard.split("/"))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["image_id"] for l in open(a.out)}
    scenes = [json.loads(l) for i, l in enumerate(open(a.scenes)) if i % sk == si]
    scenes = [s for s in scenes if s["image_id"] not in done]
    if a.limit:
        scenes = scenes[: a.limit]
    print(f"[gen] {len(scenes)} scenes to do ({len(done)} done)", flush=True)
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, limit_mm_per_prompt={"image": 1},
              enable_prefix_caching=True, max_num_seqs=a.max_num_seqs)
    sp_w = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1500)
    sp_v = SamplingParams(temperature=0.0, max_tokens=1200)
    ck = dict(enable_thinking=False)
    fout = open(a.out, "a")
    t0, n_done = time.time(), 0
    for c in range(0, len(scenes), a.chunk):
        chunk = scenes[c:c + a.chunk]
        urls = [numbered_image(sc, a.img_root) for sc in chunk]
        convs = []
        for sc, u in zip(chunk, urls):
            body = json.dumps(dict(objects=obj_list(sc), tasks=[task_payload(sc, t, a.n_pos, a.n_neg) for t in sc["tasks"]]),
                              ensure_ascii=False)
            convs.append(msgs(WRITER, u, body))
        outs = llm.chat(convs, sp_w, chat_template_kwargs=ck, use_tqdm=False)
        gens, vconvs, qmaps = [], [], []
        for sc, u, o in zip(chunk, urls, outs):
            j = parse_json(o.outputs[0].text) or {}
            res = j.get("results", []) if isinstance(j, dict) else []
            g = {}
            tids = {t["tid"] for t in sc["tasks"]}
            for r in res if isinstance(res, list) else []:
                if isinstance(r, dict) and r.get("tid") in tids and isinstance(r.get("texts"), list):
                    g[r["tid"]] = [str(x).strip() for x in r["texts"] if str(x).strip()][:4]
            gens.append(g)
            items = [(tid, i, t) for tid, ts in g.items() for i, t in enumerate(ts)]
            random.Random(sc["image_id"]).shuffle(items)
            qmap = {f"q{n}": [tid, i] for n, (tid, i, _) in enumerate(items)}
            qmaps.append(qmap)
            body = json.dumps(dict(objects=obj_list(sc), queries=[dict(qid=f"q{n}", text=t) for n, (_, _, t) in enumerate(items)]),
                              ensure_ascii=False)
            vconvs.append(msgs(VERIFIER, u, body) if items else None)
        idx = [i for i, v in enumerate(vconvs) if v is not None]
        vouts = llm.chat([vconvs[i] for i in idx], sp_v, chat_template_kwargs=ck, use_tqdm=False) if idx else []
        ver = [dict() for _ in chunk]
        for i, o in zip(idx, vouts):
            j = parse_json(o.outputs[0].text) or {}
            for r in (j.get("answers", []) if isinstance(j, dict) else []):
                if isinstance(r, dict) and r.get("qid") in qmaps[i]:
                    ver[i][r["qid"]] = r.get("ids")
        for sc, g, qm, v, o in zip(chunk, gens, qmaps, ver, outs):
            fout.write(json.dumps(dict(image_id=sc["image_id"], gen=g, qmap=qm, verify=v,
                                       raw_ok=bool(g), model=a.model), ensure_ascii=False) + "\n")
        fout.flush()
        n_done += len(chunk)
        el = time.time() - t0
        print(f"[gen] {n_done}/{len(scenes)} images, {el/60:.1f} min, {n_done/el*60:.0f} img/min, "
              f"texts {sum(len(x) for g in gens for x in g.values())}, verified {sum(len(x) for x in ver)}", flush=True)
    print("[gen] DONE", flush=True)


if __name__ == "__main__":
    main()
