"""Build Jev request files (one request per image; all questions about that image in one request).

    python evals/jev/build.py <bench> --root ~/jev        writes <root>/runs/<bench>/requests.jsonl
benches: grefcoco | refcoco_val ... refcocog_test (8 splits) | ss2 | sspro | venus | attr | ambig | robust | od
         fetch_coco (downloads the COCO images every bench needs into <root>/imgs/coco/{train2014,val2017})

Fixed question templates (chosen before any run, identical across items; no per-benchmark tuning beyond the
image's name: `photo` for natural images, `screen` for screenshots). Every grounding item becomes two questions,
following TypeSafe's guidance that absence is asked as a parallel noul question:
    <k>_box : Locate the thing in `photo` that matches the description "<expr>".
    <k>_yes : Is there something in `photo` that matches the description "<expr>"?
GUI:
    <k>_box : Locate the UI element in `screen` that matches the instruction "<text>".
    <k>_yes : Is there a UI element in `screen` that matches the instruction "<text>"?
Each request also carries `items` (gold + ids) for the scorer; the client ignores that field.
"""
import argparse, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA", "refcocoplus_testB",
          "refcocog_val", "refcocog_test"]
NAT_STATE = "An image is attached as `photo`."
GUI_STATE = "A screenshot is attached as `screen`."


def nat_q(expr):
    d = expr.replace('"', "'")
    return {"_box": dict(type="box", instructions=f'Locate the thing in `photo` that matches the description "{d}".'),
            "_yes": dict(type="noul", instructions=f'Is there something in `photo` that matches the description "{d}"?')}


def gui_q(text):
    d = text.replace('"', "'")
    return {"_box": dict(type="box", instructions=f'Locate the UI element in `screen` that matches the instruction "{d}".'),
            "_yes": dict(type="noul", instructions=f'Is there a UI element in `screen` that matches the instruction "{d}"?')}


def write(path, reqs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(path, "requests", len(reqs), "items", sum(len(r.get("items", [1])) for r in reqs),
          "questions", sum(len(r["questions"]) for r in reqs))


def coco_req(rid, path, exprs):
    """exprs: list of (expr_text, item_dict)"""
    qs, items = {}, []
    for k, (e, it) in enumerate(exprs):
        for suf, q in nat_q(e).items():
            qs[f"q{k}{suf}"] = q
        items.append(dict(it, k=f"q{k}", text=e))
    return dict(id=rid, state=NAT_STATE, questions=qs, images={"photo": path}, items=items)


def b_prep(name, root):
    """data/prep/eval_<name>.jsonl (gRefCOCO merged or a RefCOCO split) -> one request per image"""
    reqs = []
    for l in open(f"{REPO}/data/prep/eval_{name}.jsonl"):
        r = json.loads(l)
        ex = [(q["text"], dict(sent_id=q.get("sent_id"), present=q["present"], box=q.get("box"),
                               boxes=q.get("boxes") or ([q["box"]] if q.get("box") else []),
                               n_targets=q.get("n_targets", 1 if q["present"] else 0), W=r["W"], H=r["H"]))
              for q in r["questions"]]
        reqs.append(coco_req(f"{name}/{r['image_id']}", f"{root}/imgs/coco/train2014/{r['file']}", ex))
    return reqs


def b_solo(root, n_images=150):
    """Multi-question control: the first n_images RefCOCO-val images, but one expression (box + noul) per request.
    Same items and templates as the grouped refcoco_val run, so the difference isolates request grouping."""
    reqs = []
    for i, l in enumerate(open(f"{REPO}/data/prep/eval_refcoco_val.jsonl")):
        if i >= n_images:
            break
        r = json.loads(l)
        for j, q in enumerate(r["questions"]):
            it = dict(sent_id=q.get("sent_id"), present=True, box=q["box"], boxes=[q["box"]], n_targets=1,
                      W=r["W"], H=r["H"], grouped_id=f"refcoco_val/{r['image_id']}", grouped_k=f"q{j}")
            reqs.append(coco_req(f"solo/{r['image_id']}_{j}", f"{root}/imgs/coco/train2014/{r['file']}", [(q["text"], it)]))
    return reqs


def b_robust(root):
    by = {}
    for fn in sorted(glob.glob(f"{root}/data/robust/hf/data/*.jsonl")):
        for l in open(fn):
            it = json.loads(l)
            by.setdefault(it["image_id"], (it["file"], []))[1].append(
                (it["text"], dict(item_id=it["id"], type=it["type"], sub=it.get("sub"), lang=it.get("lang"),
                                  present=it["present"], box=it.get("box"), W=it["W"], H=it["H"])))
    return [coco_req(f"robust/{iid}", f"{root}/imgs/coco/train2014/{f}", ex) for iid, (f, ex) in sorted(by.items())]


def b_attr(root):
    d = json.load(open(f"{REPO}/outputs/probes/ambiguity_attr/B_attribute_glance_v1.json"))
    reqs = []
    for it in d["items"]:
        ex = [(q["text"], dict(idx=it["idx"], image_id=it["image_id"], qi=j, type=q["type"], present=q["present"],
                               swap=q.get("swap"), box=it["gt_box"], W=it["W"], H=it["H"], salience=it.get("salience")))
              for j, q in enumerate(it["questions"])]
        reqs.append(coco_req(f"attr/{it['idx']}_{it['image_id']}",
                             f"{root}/imgs/coco/val2017/{it['image_id']:012d}.jpg", ex))
    return reqs


def b_ambig(root):
    d = json.load(open(f"{REPO}/outputs/probes/ambiguity_attr/A_ambiguity_glance_v1.json"))
    reqs = []
    for it in d["items"]:
        qs, items = {}, []
        for j, q in enumerate(it["questions"]):
            k = f"q{j}"
            base = dict(image_id=it["image_id"], kind=it["kind"], tag=q["tag"], text=q["text"], k=k, qi=j,
                        gt=q.get("gt"), gt_boxes=it["gt_boxes"], W=it["W"], H=it["H"], n_inst=it["n_inst"])
            if q["tag"] == "count":
                opts = list(q["probs"].keys())
                qs[k] = dict(type="choice", instructions=q["text"].replace("?", " in `photo`?"),
                             criteria={o: None for o in opts})
            elif q["tag"] == "more_than_one":
                qs[k] = dict(type="noul", instructions=q["text"].replace("?", " in `photo`?"))
            else:
                for suf, qq in nat_q(q["text"]).items():
                    qs[k + suf] = qq
            items.append(base)
        reqs.append(dict(id=f"ambig/{it['image_id']}", state=NAT_STATE, questions=qs,
                         images={"photo": f"{root}/imgs/coco/val2017/{it['image_id']:012d}.jpg"}, items=items))
    return reqs


def b_gui(name, root):
    reqs = []
    for i, l in enumerate(open(f"{root}/gui/{name}.jsonl")):
        r = json.loads(l)
        qs, items = {}, []
        for k, q in enumerate(r["questions"]):
            for suf, qq in gui_q(q["text"]).items():
                qs[f"q{k}{suf}"] = qq
            items.append(dict(k=f"q{k}", text=q["text"], present=q["present"], box=q.get("box"), meta=q.get("meta"),
                              W=r["wh"][0], H=r["wh"][1]))
        reqs.append(dict(id=f"{name}/{i}", state=GUI_STATE, questions=qs,
                         images={"screen": os.path.join(r["image_root"], r["file"])}, items=items))
    return reqs


def b_od(root):
    """OpenDecisions: the item's own state and question, unchanged (it is already in the decisions format)."""
    reqs = []
    for l in open(f"{root}/data/od/opendecisions.jsonl"):
        it = json.loads(l)
        q = {k: v for k, v in it["question"].items() if k in ("type", "instructions", "criteria")}
        reqs.append(dict(id=f"od/{it['id']}", state=it["state"], questions={"answer": q},
                         items=[dict(od_id=it["id"], type=q["type"], labels=it["labels"])]))
    return reqs


def fetch_coco(root):
    import requests
    need = set()
    for s in SPLITS + ["grefcoco_val_merged"]:
        for l in open(f"{REPO}/data/prep/eval_{s}.jsonl"):
            need.add(("train2014", json.loads(l)["file"]))
    for fn in glob.glob(f"{root}/data/robust/hf/data/*.jsonl"):
        for l in open(fn):
            need.add(("train2014", json.loads(l)["file"]))
    for f in ("A_ambiguity_glance_v1", "B_attribute_glance_v1"):
        for it in json.load(open(f"{REPO}/outputs/probes/ambiguity_attr/{f}.json"))["items"]:
            need.add(("val2017", f"{it['image_id']:012d}.jpg"))
    todo = [(d, f) for d, f in sorted(need) if not os.path.exists(f"{root}/imgs/coco/{d}/{f}")]
    print("coco images needed", len(need), "missing", len(todo), flush=True)
    for d in ("train2014", "val2017"):
        os.makedirs(f"{root}/imgs/coco/{d}", exist_ok=True)
    sess = requests.Session()

    def get(df):
        d, f = df
        for _ in range(5):
            try:
                r = sess.get(f"http://images.cocodataset.org/{d}/{f}", timeout=60)
                if r.status_code == 200 and len(r.content) > 1000:
                    open(f"{root}/imgs/coco/{d}/{f}.tmp", "wb").write(r.content)
                    os.replace(f"{root}/imgs/coco/{d}/{f}.tmp", f"{root}/imgs/coco/{d}/{f}")
                    return True
            except Exception:
                pass
        print("FAILED", d, f, flush=True)
        return False

    with ThreadPoolExecutor(32) as ex:
        ok = sum(ex.map(get, todo))
    print("downloaded", ok, "of", len(todo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--root", default=os.path.expanduser("~/jev"))
    a = ap.parse_args()
    b, root = a.bench, a.root
    if b == "fetch_coco":
        return fetch_coco(root)
    if b == "grefcoco":
        reqs = b_prep("grefcoco_val_merged", root)
    elif b in SPLITS:
        reqs = b_prep(b, root)
    elif b in ("ss2", "sspro", "venus"):
        reqs = b_gui(b, root)
    elif b in ("robust", "attr", "ambig", "od", "solo"):
        reqs = globals()[f"b_{b}"](root)
    else:
        sys.exit(f"unknown bench {b}")
    write(f"{root}/runs/{b}/requests.jsonl", reqs)


if __name__ == "__main__":
    main()
