"""Turn closed raw jobs into append-only release shards (run periodically; idempotent).

  python build_shards.py --raw /data/guisyn/raw --web /data/guisyn/rawweb --stage /data/guisyn/stage

For every raw job file not yet built:
  - drop screenshots whose dHash is within Hamming 3 of any ScreenSpot-v2 / ScreenSpot-Pro / VenusBench-GD image
  - positives: template referring expressions (name | function | position | icon) over the filtered elements
  - hard negatives: UI-chrome descriptions seen in OTHER screenshots of the SAME app, verified absent from this
    screenshot's FULL AT-SPI tree (exact + fuzzy name match, text content, doc vocabulary); easy negatives: from other apps
  - write  stage/shards/{source}/{ts}-{NNNN}.jsonl, .images.tar (images/{source}/{id}.webp), .a11y.tar
    via .tmp + rename into stage (the uploader only ever sees complete files)
"""
import argparse, glob, gzip, io, json, multiprocessing as mp, os, random, sys, tarfile, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/guisyn/repo")
from exprs import element_exprs, ordinal_exprs, Absence, CHROME_ROLES, norm, clean, good_desc

MAX_POS = 64
# AT-SPI hit-test trust policy. "self"/"desc": the hit lands on the element; "anc1": toolkit stopped at the parent;
# "ancK": stopped K levels up (GTK2 / Chromium / darktable do not hit-test deep children); "other": the point belongs to
# a different widget -> stale extents (e.g. LibreOffice GTK menubar) or hidden/covered widget -> always dropped.
LENIENT_APPS = {"gimp", "codium", "darktable"}
GENERIC = set("ok cancel close apply help yes no back next finish reset defaults default delete remove add new edit open save "
              "more options settings preferences menu search find clear done continue skip previous up down left right "
              "about view file tools window insert format refresh reload stop start run browse select all none".split())
PARA = {}  # DeepSeek functional paraphrases (text-only), loaded from /data/guisyn/paraphrases.json if present


def hit_ok(e, app):
    r = e.get("hit")
    if r in (None, "self", "desc", "anc1", "popup"):
        return True
    if r == "other":
        return False
    return app in LENIENT_APPS
NEG_FRAC = 0.2


def load_state(p):
    return json.load(open(p)) if os.path.exists(p) else dict(done=[], shard_no=0)


def thumb32(path):
    try:
        from PIL import Image
        import numpy as np
        return np.asarray(Image.open(path).convert("L").resize((32, 32), Image.BILINEAR), dtype=np.uint8)
    except Exception:
        return None


class EvalDedup:
    """Stage 1: the repo's 64-bit dHash index (Hamming <= 3; dv/data/prep_v1.HashIndex), as in v1 prep.
    Stage 2 (only on a stage-1 hit): 32x32 grayscale thumbnail mean-abs-diff < 2.5/255 against any eval image ->
    duplicate. (64-bit dHash alone fires on low-detail screens: e.g. a FreeCAD frame vs a Hacker News screenshot.)"""

    def __init__(self, path):
        import numpy as np
        self.idx, self.n = eval_index(path)
        tp = path.replace(".json", "_thumbs.npz")
        z = np.load(tp) if os.path.exists(tp) else None
        self.th = z["thumbs"].astype(np.int16) if z is not None else None
        self.names = z["names"] if z is not None else None

    def dup(self, path, h):
        if not self.idx.hit(h):
            return False, "none"
        if self.th is None:
            return True, "coarse"
        t = thumb32(path)
        if t is None:
            return True, "coarse"
        d = abs(self.th - t.astype("int16")[None]).mean(axis=(1, 2))
        j = int(d.argmin())
        return bool(d[j] < 2.5), f"mad{d[j]:.1f}:{self.names[j]}"


def eval_index(path):
    from dv.data.prep_v1 import HashIndex
    d = json.load(open(path))
    return HashIndex([h for k, v in d.items() for h in v]), sum(len(v) for v in d.values())


def pool_add(pool, rec):
    """UI-chrome descriptions of a screenshot -> per-app negative pool."""
    W, H = rec["W"], rec["H"]
    r = random.Random(rec["id"])
    for e in rec["elements"]:
        if not hit_ok(e, rec["app"]):
            continue
        if e.get("uncaptioned") or e["role"] not in CHROME_ROLES or not e["name"] or len(e["name"]) > 60:
            continue
        if e.get("ctx", {}).get("top", {}).get("role") == "document web" or e.get("web_content"):
            pass
        for t, k, m in element_exprs(e, [e], W, H, r):
            if k in ("name", "icon", "function"):
                nm = clean(e["name"]) if k != "function" else good_desc(e.get("desc", ""), e["name"])
                if not nm or sum(ord(ch) < 128 for ch in t) < 0.9 * len(t):
                    continue
                pool[rec["app"]][norm(nm)].append(dict(text=t, name=nm, role=e["role"], src=rec["id"], kind=k))


INTER_SAME = [{"push button", "toggle button", "button", "split button", "push button menu", "tool bar item", "icon",
               "radio button", "check box"}, {"menu item", "check menu item", "radio menu item", "menu"},
              {"page tab"}, {"list item", "tree item", "table cell"}]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def tree_checks(rec, nodes):
    """Using the full AT-SPI tree: (1) drop elements whose box (IoU > 0.8) is shared with another SHOWING widget of the
    same role family but a different name that is not its ancestor/descendant (hidden toolbar items reported at a
    visible item's position -> name/box swaps, seen in LibreOffice); (2) return a toolbar-completeness counter."""
    if not nodes:
        return rec["elements"], None
    s = (rec.get("fstats") or {}).get("scale", 1.0) or 1.0
    by_i = {n["i"]: n for n in nodes}
    fam = {r: k for k, rs in enumerate(INTER_SAME) for r in rs}
    cand = []
    for n in nodes:
        if "showing" in n["states"] and n.get("box") and n["role"] in fam and n["box"][2] > 0 and n["box"][3] > 0:
            x, y, w, h = n["box"]
            cand.append((n["i"], fam[n["role"]], norm(clean(n["name"])), [x * s, y * s, (x + w) * s, (y + h) * s]))
    def anc(i):
        out, p = set(), by_i.get(i, {}).get("p", -1)
        while p >= 0 and p in by_i:
            out.add(p); p = by_i[p]["p"]
        return out
    keep = []
    for e in rec["elements"]:
        f = fam.get(e["role"])
        bad = False
        if e["role"] in ("list item", "menu item", "table cell") and not e.get("popup"):
            # items of a collapsed combo box popup (Qt reports them SHOWING at the combo's own box)
            p, k = by_i.get(e["i"], {}).get("p", -1), 0
            while p >= 0 and p in by_i and k < 5:
                if by_i[p]["role"] == "combo box" and "expanded" not in by_i[p]["states"]:
                    bad = True; break
                p, k = by_i[p]["p"], k + 1
            if bad:
                continue
        if f is not None:
            ea = None
            for i, f2, nm, b in cand:
                if i == e["i"] or f2 != f or nm == norm(clean(e["name"])) or _iou(b, e["box"]) <= 0.8:
                    continue
                ea = ea if ea is not None else anc(e["i"])
                if i in ea or e["i"] in anc(i):
                    continue
                bad = True; break
        if not bad:
            keep.append(e)
    tbs = {tuple(n["box"]): n["i"] for n in nodes if n["role"] == "tool bar" and n.get("box")}
    SKIP = {"filler", "separator", "panel", "tool bar"}
    def tb_items(tb):
        """node indices of the toolbar's top-most showing widgets (any role except fillers/separators/panels)"""
        ti = tbs.get(tuple(tb))
        if ti is None:
            return None
        out, stack = set(), [ti]
        kids = {}
        for n in nodes:
            kids.setdefault(n["p"], []).append(n)
        while stack:
            for c in kids.get(stack.pop(), []):
                if "showing" not in c["states"] or not c.get("box") or c["box"][2] <= 0:
                    continue
                if c["role"] in SKIP:
                    stack.append(c["i"])
                else:
                    out.add(c["i"])
        return out
    tb_count = tb_items
    return keep, tb_count


def build_record(rec, raw_root, pools, apps, rng, absence_texts, nodes=None):
    W, H = rec["W"], rec["H"]
    rec["elements"] = [e for e in rec["elements"] if hit_ok(e, rec["app"])]
    n0 = len(rec["elements"])
    rec["elements"], tb_count = tree_checks(rec, nodes)
    conflicts = n0 - len(rec["elements"])
    els = [e for e in rec["elements"] if not e.get("uncaptioned")]
    qs = []
    for e in els:
        for t, k, m in element_exprs(e, els, W, H, rng):
            qs.append(dict(text=t, present=True, box=e["box"], kind=k,
                           meta=dict(role=e["role"], name=e["name"], tmpl=m["t"], el=e["i"])))
    para = PARA.get("d") or {}
    if para:
        cnt = Counter(norm(clean(e["name"])) for e in els)
        for e in els:
            p = para.get("\t".join([rec["app"], e["role"], clean(e["name"])]))
            if p and cnt[norm(clean(e["name"]))] == 1:
                qs.append(dict(text=p, present=True, box=e["box"], kind="function",
                               meta=dict(role=e["role"], name=e["name"], tmpl="deepseek_paraphrase", el=e["i"])))
    for e, t, k, m in ordinal_exprs(els, W, H, tb_count):
        qs.append(dict(text=t, present=True, box=e["box"], kind=k, meta=dict(role=e["role"], name=e["name"], tmpl=m["t"], el=e["i"])))
    # cap positives: <= 2 expressions per element, <= 35% text-heavy roles, per-kind quotas then leftover fill
    rng.shuffle(qs)
    QUOTA = dict(name=0.45, function=0.20, position=0.20, icon=0.15)
    per_el, out, nlab, nk = Counter(), [], 0, Counter()
    def take(x):
        nonlocal nlab
        el = x["meta"]["el"]
        if per_el[el] >= 2 or len(out) >= MAX_POS or x.get("_used"):
            return
        if x["meta"]["role"] in ("label", "paragraph", "table cell", "static", "heading", "list item"):
            if nlab >= MAX_POS * 0.35:
                return
            nlab += 1
        per_el[el] += 1; nk[x["kind"]] += 1; x["_used"] = True
        out.append(x)
    for kind, f in QUOTA.items():
        for x in qs:
            if x["kind"] == kind and nk[kind] < f * MAX_POS:
                take(x)
    for x in qs:
        take(x)
    for x in out:
        x.pop("_used", None)
    if len(out) < 3:
        return None
    # negatives
    ab = Absence([n for _, n in rec["all_names"]], absence_texts)
    n_neg = max(1, round(len(out) * NEG_FRAC / (1 - NEG_FRAC)))
    n_hard = round(n_neg * 0.8)
    negs, seen = [], set()
    pool = pools.get(rec["app"], {})
    others = [x for x in apps if x != rec["app"] and pools.get(x)]
    for kind, want in (("negative_hard", n_hard), ("negative_easy", n_neg - n_hard)):
        if kind == "negative_hard":
            keys = [k for k in pool if k not in ab.nset]   # names never seen in this screenshot's full tree
            src = [(pool, k) for k in keys]
        else:
            src = []
            for oa in rng.sample(others, min(len(others), 12)):
                ks = [k for k in pools[oa] if k not in ab.nset]
                src += [(pools[oa], k) for k in rng.sample(ks, min(len(ks), 20))]
        rng.shuffle(src)
        got = 0
        for P, k in src:
            if got >= want:
                break
            c = rng.choice(P[k])
            if c["src"] == rec["id"] or k in seen or not ab.absent(c["name"]):
                continue
            seen.add(k)
            negs.append(dict(text=c["text"], present=False, box=None, kind=kind,
                             meta=dict(role=c["role"], name=c["name"], from_id=c["src"], verified="full_tree" if rec.get("source", "desktop") == "desktop" else "viewport_dom")))
            got += 1
    unc = [dict(box=e["box"], role=e["role"], desc=e.get("desc", "")) for e in rec["elements"] if e.get("uncaptioned")]
    return dict(image=None, W=W, H=H, source=rec.get("source", "desktop"), app=rec["app"], url=rec.get("url"),
                questions=out + negs, uncaptioned=unc,
                meta=dict(id=rec["id"], scale=rec.get("scale", 1), theme=rec.get("theme"), icons=rec.get("icons"),
                          qtstyle=rec.get("qtstyle"), doc=rec.get("doc"), dpr=rec.get("dpr"), actions=rec.get("actions"),
                          n_elements=len(rec["elements"]), conflicts_dropped=conflicts, fstats=rec.get("fstats")))


def texts_of(a11y_path):
    """-> (texts for absence proofs, full node list or None)"""
    try:
        d = json.load(gzip.open(a11y_path, "rt"))
    except Exception:
        return [], None
    if "nodes" in d:
        return ([n.get("text", "") for n in d["nodes"] if n.get("text")] + [n.get("desc", "") for n in d["nodes"] if n.get("desc")],
                d["nodes"])
    return d.get("texts", []), None


G = {}


def _one(args):
    """per-record work (runs in a forked pool; pools / dedup index come from the parent via G)"""
    from dv.data.prep_v1 import phash_file
    l, src, root, name = args
    st = Counter()
    rec = json.loads(l)
    rec.setdefault("source", src)
    ip = f"{root}/{rec['image']}"
    if not os.path.exists(ip):
        st["missing_image"] += 1; return dict(stats=st)
    h, _ = phash_file(ip)
    isdup, why = G["ded"].dup(ip, h)
    if why != "none":
        st[f"eval_coarse_hit_{src}"] += 1
    if isdup:
        st[f"eval_dup_{src}"] += 1; return dict(stats=st)
    rng = random.Random(rec["id"])
    texts, nodes = texts_of(f"{root}/{rec['a11y']}")
    out = build_record(rec, root, G["pools"], G["apps"], rng, texts, nodes)
    if out is None:
        st[f"too_few_{src}"] += 1; return dict(stats=st)
    ext = os.path.splitext(ip)[1]
    out["image"] = f"images/{src}/{rec['id']}{ext}"
    out["a11y"] = f"{name}.a11y.tar:{rec['id']}.json.gz"
    out["meta"]["dhash"] = h
    return dict(stats=st, out=out, ip=ip, ap=f"{root}/{rec['a11y']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/data/guisyn/raw")
    ap.add_argument("--web", default="/data/guisyn/rawweb")
    ap.add_argument("--stage", default="/data/guisyn/stage")
    ap.add_argument("--work", default="/data/guisyn/build")
    ap.add_argument("--evalhash", default="/data/guisyn/eval_hashes.json")
    ap.add_argument("--min_jobs", type=int, default=1)
    ap.add_argument("--max_recs_per_shard", type=int, default=2500)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    st = load_state(f"{a.work}/state.json")
    done = set(st["done"])
    from dv.data.prep_v1 import phash_file
    pp = "/data/guisyn/paraphrases.json"
    if os.path.exists(pp):  # generic labels get context-specific paraphrases ("closes the hyperlink dialog"): skip them
        PARA["d"] = {k: v for k, v in json.load(open(pp)).items()
                     if v and norm(clean(k.split("\t")[-1])) not in GENERIC and len(norm(k.split("\t")[-1])) > 2}
    ded = EvalDedup(a.evalhash)
    n_eval = ded.n
    # pools are rebuilt from every closed job (cheap: raw records only)
    jobs = {"desktop": sorted(glob.glob(f"{a.raw}/jobs/*.jsonl")), "web": sorted(glob.glob(f"{a.web}/jobs/*.jsonl"))}
    pools = defaultdict(lambda: defaultdict(list))
    for src, fs in jobs.items():
        for f in fs:
            for l in open(f):
                pool_add(pools, json.loads(l))
    apps = sorted(pools)
    stats = Counter()
    for src, fs in jobs.items():
        root = a.raw if src == "desktop" else a.web
        new = [f for f in fs if f not in done]
        if len(new) < a.min_jobs:
            continue
        # group new jobs into shards of <= max_recs_per_shard records
        batch, nrec = [], 0
        groups = []
        for f in new:
            n = sum(1 for _ in open(f))
            batch.append(f); nrec += n
            if nrec >= a.max_recs_per_shard:
                groups.append(batch); batch, nrec = [], 0
        if batch:
            groups.append(batch)
        for g in groups:
            st["shard_no"] += 1
            name = f"{time.strftime('%Y%m%dT%H%M%S')}-{st['shard_no']:04d}"
            d = f"{a.stage}/shards/{src}"
            os.makedirs(d, exist_ok=True)
            tmpd = f"{a.work}/tmp"; os.makedirs(tmpd, exist_ok=True)
            fj, fi, fa = (f"{tmpd}/{name}{x}.tmp" for x in (".jsonl", ".images.tar", ".a11y.tar"))
            rng = random.Random(name)
            lines = [(l, src, root, name) for f in g for l in open(f)]
            G.update(pools=pools, apps=apps, ded=ded)
            with open(fj, "w") as oj, tarfile.open(fi, "w") as ti, tarfile.open(fa, "w") as ta, \
                    mp.get_context("fork").Pool(a.procs) as pool:
                for res in pool.imap(_one, lines, chunksize=4):
                    for k, v in res["stats"].items():
                        stats[k] += v
                    out = res.get("out")
                    if out is None:
                        continue
                    ti.add(res["ip"], arcname=out["image"])
                    ta.add(res["ap"], arcname=f"{out['meta']['id']}.json.gz")
                    oj.write(json.dumps(out, ensure_ascii=False) + "\n")
                    stats[f"recs_{src}"] += 1
                    for qq in out["questions"]:
                        stats[f"q_{src}_{qq['kind']}"] += 1
            for x, t in ((".images.tar", fi), (".a11y.tar", fa), (".jsonl", fj)):  # jsonl last = shard complete
                os.rename(t, f"{d}/{name}{x}")
            done.update(g)
            st["done"] = sorted(done)
            json.dump(st, open(f"{a.work}/state.json", "w"))
            print("shard", src, name, "jobs", len(g), flush=True)
    stats["eval_hashes"] = n_eval
    print(json.dumps(stats))
    with open(f"{a.work}/build_log.jsonl", "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%F %T"), **stats)) + "\n")


if __name__ == "__main__":
    main()
