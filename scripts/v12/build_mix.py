"""v1.2 training mixture = v1.1 base pools (replay) + a qtype-stratified sample of Glance-Data (`single` view) + the
category-free set, with consistency groups; asserts decontamination; writes stats.

  python scripts/v12/build_mix.py --prep /content/data/prep_v1 --gd /content/gd/single --catfree /content/v12/out/catfree.jsonl \
      --eval_ids /content/v12/eval_image_ids.json --out /content/data/prep_v12

Base: dv.data.prep_v1.mix over the v1.1 pools with --base_total questions (v1.1 used 1M at 40/25/22/13 text/GUI/natural/
pixmo); its per-domain GUI caps and resolutions are unchanged.

Glance-Data keep-rates per qtype (--gd_keep) aim the sample at v1.1's measured failures (neg_count, cat_only, sole-object
positional words) while keeping the phrasing ladders (short/medium/long/casual en+zh) as consistency groups:
every positive of one image with the same target box joins one group; the group's original is the first of
medium/en, short/en, long/en, casual/en present (the plainest English phrasing), every other phrasing is a variant.
"""
import argparse, collections, glob, json, os, random, subprocess, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

GD_KEEP = dict(short=0.09, medium=0.09, long=0.09, casual=0.09, cat_only=0.65, function=0.5, pos_pos_sole=0.5,
               neg_position=0.5, neg_count=0.25, hneg_attribute=0.12, hneg_relation=0.15, neg_near=0.12)
ORIG_PRI = [("medium", "en"), ("short", "en"), ("long", "en"), ("casual", "en")]


def pack(base, qs, maxq, rng):
    groups = collections.OrderedDict()
    for q in qs:
        groups.setdefault(q.get("grp") or id(q), []).append(q)
    gl = list(groups.values()); rng.shuffle(gl)
    out, cur = [], []
    for g in gl:
        if cur and len(cur) + len(g) > maxq:
            out.append(dict(base, questions=cur)); cur = []
        cur += g[:maxq]
    if cur:
        out.append(dict(base, questions=cur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--gd", required=True)
    ap.add_argument("--catfree", required=True)
    ap.add_argument("--eval_ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_total", type=int, default=820000)
    ap.add_argument("--base_frac", default="text=0.4634,gui=0.2927,natural=0.1220,pixmo=0.1220")
    ap.add_argument("--gd_keep", default=",".join(f"{k}={v}" for k, v in GD_KEEP.items()))
    ap.add_argument("--max_q", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    st = collections.defaultdict(dict)
    # ---- 1. base replay mixture (v1.1 pools, v1.1 mix code)
    if not os.path.exists(f"{a.out}/train_base.jsonl"):
        fr = dict(kv.split("=") for kv in a.base_frac.split(","))
        cmd = [sys.executable, "-m", "dv.data.prep_v1", "mix", "--out", a.prep, "--total", str(a.base_total)] + \
              sum([[f"--f_{k}", v] for k, v in fr.items()], [])
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, stdout=open(f"{a.out}/log_base_mix.txt", "w"), stderr=subprocess.STDOUT)
        os.replace(f"{a.prep}/train.jsonl", f"{a.out}/train_base.jsonl")
        os.replace(f"{a.prep}/stats.json", f"{a.out}/stats_base.json")
    base = [json.loads(l) for l in open(f"{a.out}/train_base.jsonl")]
    # ---- 2. Glance-Data sample with consistency groups
    keep = {k: float(v) for k, v in (kv.split("=") for kv in a.gd_keep.split(","))}
    by_img, meta = collections.defaultdict(list), {}
    for fn in sorted(glob.glob(f"{a.gd}/*.jsonl")):
        for l in open(fn):
            r = json.loads(l)
            meta[r["image_id"]] = {k: r[k] for k in ("src", "img", "wh", "max_vis", "plen", "image_id")}
            for q in r["questions"]:
                if rng.random() < keep.get(q["qtype"], 0.0):
                    by_img[r["image_id"]].append(q)
    gd_recs, cons = [], collections.Counter()
    for iid, qs in by_img.items():
        tgt = collections.defaultdict(list)
        for q in qs:
            if q["present"] and q.get("box") is not None:
                tgt[tuple(round(v) for v in q["box"])].append(q)
        for g in tgt.values():
            if len(g) < 2:
                continue
            orig = next((q for qt, lg in ORIG_PRI for q in g if q["qtype"] == qt and q["lang"] == lg), None)
            if orig is None:
                continue
            cg = f"gd{iid}:{id(orig)}"
            for q in g:
                q["grp"] = cg
                if q is orig:
                    q["anchor"] = True
            cons["groups"] += 1; cons["variants"] += len(g) - 1
        gd_recs += pack(meta[iid], qs, a.max_q, rng)
    st["glance_data"] = dict(images=len(by_img), records=len(gd_recs), consistency=dict(cons),
                             qtype=dict(collections.Counter(q["qtype"] for r in gd_recs for q in r["questions"])),
                             lang=dict(collections.Counter(q["lang"] for r in gd_recs for q in r["questions"])),
                             absent=sum(not q["present"] for r in gd_recs for q in r["questions"]))
    # ---- 3. category-free set
    cf = [json.loads(l) for l in open(a.catfree)]
    st["catfree"] = dict(records=len(cf), qtype=dict(collections.Counter(q["qtype"] for r in cf for q in r["questions"])),
                         lang=dict(collections.Counter(q["lang"] for r in cf for q in r["questions"])),
                         consistency_variants=sum(bool(q.get("grp")) and not q.get("anchor") for r in cf for q in r["questions"]))
    # ---- 4. decontamination: no COCO image of any eval set (ids from scripts/v12/eval_ids.py)
    ev = json.load(open(a.eval_ids))
    bad = {}
    for name, recs in (("glance_data", gd_recs), ("catfree", cf), ("base_natural", [r for r in base if r.get("src") == "natural"])):
        ids = {r["image_id"] for r in recs}
        bad[name] = {k: len(ids & set(v)) for k, v in ev.items()}
    st["decontam_overlap"] = bad
    assert not any(n for d in bad.values() for n in d.values()), f"CONTAMINATED {bad}"
    # ---- 5. mixture
    allr = base + gd_recs + cf
    rng.shuffle(allr)
    with open(f"{a.out}/train.jsonl", "w") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    part = lambda r: ("catfree" if r["src"] == "catfree" else "glance_data" if r["src"] in ("glance_data", "gap_rule")
                      else {"natural": "natural_v1"}.get(r["src"], None) or ("text" if not r.get("img") else
                      "pixmo" if "pixmo" in r["img"] else "gui"))
    tot = collections.defaultdict(collections.Counter)
    for r in allr:
        p = part(r)
        tot[p]["records"] += 1; tot[p]["questions"] += len(r["questions"]); tot[p]["prefix_tokens"] += r.get("plen", 0)
        tot[p]["absent_box"] += sum(q.get("type", "box") == "box" and not q["present"] for q in r["questions"])
        tot[p]["box"] += sum(q.get("type", "box") == "box" for q in r["questions"])
    T = {k: sum(v[k] for v in tot.values()) for k in ("records", "questions", "prefix_tokens")}
    st["mix"] = {p: dict(v, **{f"share_{k}": round(v[k] / T[k], 4) for k in T}) for p, v in tot.items()}
    st["mix"]["total"] = T
    json.dump(st, open(f"{a.out}/stats_v12.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(st, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
