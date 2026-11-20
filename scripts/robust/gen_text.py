"""Generate the query text of glance-robust-ref items with DeepSeek (Ollama cloud API; text only).

Reads OLLAMA_API_KEY from the environment (source .env; the key is never printed or written).
Usage: python3 scripts/robust/gen_text.py --src src.jsonl --out gen.jsonl [--model deepseek-v4.1-flash]
Resumable: ids already in --out are skipped. Each output line: {"id", "text", "ok", "model"}.
"""
import argparse, collections, json, os, random, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HEAD = ("You write test queries for a visual-grounding model that must box one object in a COCO photo (or say it is "
        "absent). For each input item follow the task below and return ONLY JSON of the form "
        '{"results": [{"id": "<id>", "text": "<query>", "ok": true}]} with exactly one entry per input id. '
        'Set "ok": false (text may be empty) if the item cannot be done under the rules. ')
LANG = {"en": "Write the query in English.", "zh": "Write the query in natural, colloquial Simplified Chinese (native speaker)."}
TASK = {
    ("zh", "translate"):
        "TASK: translate the English referring expression `orig` into natural, colloquial Simplified Chinese, the way a "
        "native speaker would ask someone to point at the object. Keep every attribute and location cue and add nothing. "
        "Fix typos in the source. Use everyday Chinese words (guy -> 男的/那个人, kid -> 小孩, donut -> 甜甜圈).",
    ("para", "paraphrase"):
        "TASK: paraphrase the English referring expression `orig` so that it means exactly the same but uses different "
        "words: replace the head noun with a natural synonym or near-equivalent when one exists (man -> guy/gentleman, "
        "couch -> sofa, 'left' -> 'on the left-hand side'), change the syntax (relative clause vs. prepositional phrase, "
        "reorder), fix typos. Keep every distinguishing attribute and location cue, add no new information. It must "
        "differ from the original by more than word order. Write in English.",
    ("func", "function_only"):
        "TASK: refer to the TARGET object (category `cat`) ONLY by what it is used for or what you can do with it (its "
        "function / affordance), like a person who does not know or care about its name: e.g. 'something to drink from', "
        "'what I could sit on', 'the thing that tells the time', '能喝的东西', '可以坐的地方'. Do NOT use its category "
        "name, a synonym or a close hypernym (for a cup: no cup/mug/glass/container/dish; for a teddy bear: no "
        "'stuffed toy'/'毛绒玩具'). The query must NOT also fit any "
        "of the `others` objects in the photo (if a bottle and a cup are both present, 'something to drink' is not "
        "allowed; use a more specific function or set ok=false). 2-10 words.",
    ("func", "function_attr"):
        "TASK: rewrite `orig` (it refers to an object of category `cat`) so that the object's name is replaced by a "
        "description of its function / affordance, keeping the original attribute and location cues so the same object "
        "is meant: e.g. 'red chair on left' -> 'the red thing to sit on, on the left' / '左边那个红色的能坐的东西'; "
        "'far right sandwich' -> 'the food all the way on the right you can eat with your hands'. Do NOT use the "
        "category name or a synonym.",
    ("func", "function"):  # absent
        "TASK: the category `absent_cat` does NOT appear in the photo; `others` lists what IS in it. Write a functional "
        "/ affordance query (what it is used for, without naming it, e.g. 'something to cut paper with', '能骑的东西') "
        "WITHOUT naming `absent_cat`, a synonym or a hypernym (no 'teddy bear'/'毛绒玩具', no 'sink'/'水池'), that fits `absent_cat` but would NOT plausibly be answered by any object in `others` (e.g. no 'something to "
        "sit on' if a bench/couch/bed/chair or even a large flat surface people sit on is present). If impossible, ok=false. "
        "2-10 words.",
    ("vague", "generic"):
        "TASK: refer to the target only with a vague generic noun phrase at the level of its COCO supercategory "
        "`supercat` or a broader everyday word, e.g. 'the animal', 'the vehicle', 'something to eat', 'the furniture', "
        "'that gadget', '那只动物', '吃的', '那辆车', without naming the specific category `cat` (if present). For absent "
        "items (`absent_supercat`) the phrase must not plausibly refer to any object in `others`; if it might, ok=false. "
        "If `relation` and `anchor` are given, append the relation to the anchor (e.g. 'the animal on the couch'). 1-7 words.",
    ("vague", "relational"):
        "TASK: refer to the target with a generic word (thing / object / animal / vehicle / food / item / gadget or its "
        "supercategory word), NOT its category name `cat`, and locate it relative to the ANCHOR object `anchor` (which "
        "you may name) using `relation`: the target is `relation` the anchor in image coordinates ('left of' = further "
        "left in the photo). e.g. 'the animal left of the bench', '电视右边的那个东西'. For absent items "
        "(`absent_supercat`) the generic word must not plausibly refer to anything in `others`; if it might, ok=false.",
    ("vague", "generic_attr"):
        "TASK: rewrite `orig` the way a casual user would say it: shorter and vaguer, replacing the specific noun with a "
        "loose everyday word ('that one', 'the thing', 'the guy', 'the kid', 'the lady', '那个', '那个小孩', '那个东西'), "
        "but keep enough of the location / attribute cues that it still clearly picks out the same object. If `orig` has no "
        "location / attribute cue besides the noun (e.g. 'a laptop'), set ok=false. Only use an "
        "age / gender word (kid, lady, guy, 小孩, 女的) if `orig` implies it. e.g. 'man in blue shirt standing left' -> "
        "'the guy on the left'; 'second donut from the right' -> 'that one second from the right'.",
}
FIELDS = ("id", "orig", "cat", "supercat", "absent_cat", "absent_supercat", "anchor", "relation", "others")


def payload(it):
    d = {k: it[k] for k in FIELDS if it.get(k) is not None}
    if "others" in d:
        d["others"] = sorted({o["cat"] for o in d["others"]})
    return d


def call(model, key, msgs, timeout=180):
    body = json.dumps(dict(model=model, messages=msgs, stream=False, format="json", think=False,
                           options=dict(temperature=0.7))).encode()
    req = urllib.request.Request("https://ollama.com/api/chat", data=body,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    key = os.environ.get("OLLAMA_API_KEY")
    assert key, "OLLAMA_API_KEY not set"
    items = [json.loads(l) for l in open(a.src)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [it for it in items if it["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    groups = collections.defaultdict(list)
    for it in todo:
        groups[(it["type"], it["sub"], it["lang"])].append(it)
    batches = [(g, v[i:i + a.batch]) for g, v in groups.items() for i in range(0, len(v), a.batch)]
    random.Random(0).shuffle(batches)
    print(f"{len(todo)} items, {len(batches)} batches", flush=True)
    lock = threading.Lock(); fout = open(a.out, "a"); stats = collections.Counter()

    def work(job):
        (typ, sub, lang), bat = job
        sysmsg = HEAD + TASK[(typ, sub)] + " " + LANG[lang]
        user = json.dumps([payload(it) for it in bat], ensure_ascii=False)
        ids = {it["id"] for it in bat}
        for attempt in range(4):
            try:
                res = json.loads(call(a.model, key, [dict(role="system", content=sysmsg), dict(role="user", content=user)]))
                res = res.get("results", res) if isinstance(res, dict) else res
                got = {r["id"]: r for r in res if isinstance(r, dict) and r.get("id") in ids}
                if len(got) < len(ids) * 0.8 and attempt < 3:
                    raise ValueError(f"only {len(got)}/{len(ids)} ids")
                with lock:
                    for i, r in got.items():
                        fout.write(json.dumps(dict(id=i, text=str(r.get("text", "")).strip(), ok=bool(r.get("ok", True)),
                                                   model=a.model), ensure_ascii=False) + "\n")
                    fout.flush(); stats["ok"] += len(got); stats["missing"] += len(ids) - len(got)
                return
            except Exception as e:  # the message never contains the key
                time.sleep(20 * (attempt + 1))
                err = type(e).__name__ + ": " + str(e)[:120]
        with lock:
            stats["failed_batches"] += 1
            print("batch failed", typ, sub, lang, err, flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for n, _ in enumerate(ex.map(work, batches)):
            if n % 25 == 0:
                print(n, len(batches), dict(stats), f"{time.time() - t0:.0f}s", flush=True)
    print("done", dict(stats), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
