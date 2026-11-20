"""AR IoU-head experiment, step 0 (CPU): build the held-out item sets the frozen controlled-study heads were never
trained on.  python ablation/ar_iou_prep.py [--prep data/prep] [--out DIR]

Why not grefcalib / calib_natural: they were held out of *v1* training only; all of their images (600 / 378) are v0
training images and their positive expressions are v0 training expressions, so for the controlled study (trained on
data/prep/train.jsonl) they are in-sample.  The only in-distribution images v0 never saw are the RefCOCO-family
eval splits (data/prep/stats.json held_out_images).  Test = gRefCOCO val merged (1,501 images).  We use
  H  eval_arhead_rcg.jsonl   RefCOCOg-umd val+test expressions on the 3,549 images that are in neither gRefCOCO val
                             nor gRefCOCO testA/B (all present, single target): trains the AR IoU head.
  C  eval_arcal_gtest.jsonl  gRefCOCO testA+testB, single-target (with box) + no-target expressions, minus the one
                             image shared with val: label-free-protocol calibration split (Platt fitted here only).
  T  eval_arhead_train.jsonl 1,000 random v0 *training* images, present expressions only: in-sample head variant
                             (the gate's IoU head was trained in-sample, jointly with the model).
H, C and the test set are pairwise image-disjoint; T is disjoint from all three.
"""
import argparse, json, os, random


def rd(p):
    return [json.loads(l) for l in open(p)]


def wr(p, recs):
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="data/prep")
    ap.add_argument("--out", default="data/prep")
    ap.add_argument("--n_train_images", type=int, default=1000)
    a = ap.parse_args()
    P = lambda n: f"{a.prep}/eval_{n}.jsonl"
    val = {r["image_id"] for r in rd(P("grefcoco_val_merged"))}
    gt = rd(P("grefcoco_testA")) + rd(P("grefcoco_testB"))
    gti = {r["image_id"] for r in gt}
    # C: gRefCOCO testA+B, single-target with box + no-target
    C = []
    for r in gt:
        if r["image_id"] in val:
            continue
        qs = [q for q in r["questions"] if (not q["present"]) or (q.get("n_targets") == 1 and q.get("box") is not None)]
        if qs:
            C.append(dict(r, questions=qs))
    # H: RefCOCOg val+test on images outside gRefCOCO val/testA/testB
    H = [r for n in ("refcocog_val", "refcocog_test") for r in rd(P(n)) if r["image_id"] not in val | gti]
    # T: v0 training images (in-sample), present only
    tr = rd(f"{a.prep}/train.jsonl")
    rng = random.Random(0)
    ids = sorted({r["image_id"] for r in tr})
    keep = set(rng.sample(ids, a.n_train_images))
    T = []
    for r in tr:
        if r["image_id"] in keep:
            qs = [dict(q, n_targets=1) for q in r["questions"] if q["present"] and q.get("box") is not None]
            if qs:
                T.append(dict(r, questions=qs))
    assert not ({r["image_id"] for r in H} & {r["image_id"] for r in C})
    assert not ({r["image_id"] for r in T} & (val | gti | {r["image_id"] for r in H}))
    os.makedirs(a.out, exist_ok=True)
    st = {}
    for n, R in (("arhead_rcg", H), ("arcal_gtest", C), ("arhead_train", T)):
        wr(P(n).replace(a.prep, a.out), R)
        st[n] = dict(images=len({r["image_id"] for r in R}), records=len(R), q=sum(len(r["questions"]) for r in R),
                     absent=sum(not q["present"] for r in R for q in r["questions"]))
    print(json.dumps(st))


if __name__ == "__main__":
    main()
