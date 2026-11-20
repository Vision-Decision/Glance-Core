"""Local smoke test of di.engine.GlanceEngine on a few public Decision Index items (CPU is fine, just slow).

    PYTHONPATH=. python -m di.smoke_test --suite <suite-0.3 dir> --vision <work/vision> --per_bench 2 --device mps \
        --option ckpt=checkpoints/v1/ckpt_final.pt --option temps=outputs/v1/eval_results.json

Checks:
  1. parity: the engine's B=1 batch builder == dv.data.mixed.collate_mixed on a text and an image record;
  2. every response passes the kit's `validate` (all options get a probability, sums to 1, chosen key valid);
  3. determinism: the first K requests are run twice and must give identical answers (and near-identical probabilities);
  4. the >26-option fallback (chunk tournament) runs on real BANKING77/CLINC150/... rows and on a synthetic 77-option row;
  5. prints per-benchmark accuracy on the sampled rows (tiny n: a sanity signal, not a score) and wall time.
"""
import argparse
import collections
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from di.engine import GlanceEngine, Unsupported  # noqa: E402


def pick_text(suite_dir, edition, per_bench, max_chars, must=()):
    from decision_index.suite.io import Suite, read_jsonl
    suite = Suite(suite_dir, edition)
    keep = suite.in_edition
    got = collections.defaultdict(list)
    for path in suite.row_paths:
        for row in read_jsonl(path):
            e = row["_evaluation"]
            if not keep(e):
                continue
            b = e.get("dataset")
            want = per_bench + (2 if b in must else 0)
            if len(got[b]) >= want:
                continue
            size = len(json.dumps(row["state"])) + len(json.dumps(row["questions"]))
            if size > max_chars:
                continue
            got[b].append(row)
    return got


def pick_vision(vdir, per_bench):
    got = collections.defaultdict(list)
    for line in open(os.path.join(vdir, "vision-rows.jsonl")):
        row = json.loads(line)
        if len(got[row["benchmark"]]) < per_bench:
            got[row["benchmark"]].append(row)
    return got


def parity(eng, image_path=None):
    from dv.data.mixed import MixedDataset, collate_mixed
    q = {"type": "choice", "instr": "Which colour is named?", "options": ["red", "blue", "green"], "target": [1, 0, 0]}
    recs = [dict(src="t", text="The colour is red.", questions=[q, dict(type="noul", instr="Is it red?", target=1.0)])]
    if image_path:
        recs.append(dict(src="i", img=os.path.abspath(image_path), max_vis=eng.max_vis, upscale=1.0, text="A photo.",
                         questions=[q]))
    ok = True
    for rec in recs:
        ds = MixedDataset(None, "", eng.proc, eng.cfg, train=False, recs=[rec])
        it = ds[0]
        ref = collate_mixed([it], ds.tok)
        imgs = []
        if rec.get("img"):
            imgs, _ = eng._load_images([rec["img"]])
        segs, pv = eng._prefix(rec["text"], imgs)
        mine, _ = eng._batch(segs, it["q_ids"])
        for k in ("p_ids", "p_valid", "p_img_mask", "p_pos", "q_ids", "q_valid", "q_last", "q_pos", "q_img"):
            if not torch.equal(ref[k], mine[k]):
                print("PARITY MISMATCH", rec["src"], k, ref[k].shape, mine[k].shape)
                ok = False
        if pv is not None and not torch.equal(ref["pixel_values"], pv[0]):
            print("PARITY MISMATCH pixel_values"); ok = False
    print(json.dumps({"parity_with_collate_mixed": ok, "records": [r["src"] for r in recs]}))
    return ok


def correct(row, resp):
    exp = row.get("expected") or {}
    hits = [resp["answers"][k]["choice"] == v for k, v in exp.items()
            if v is not None and k in resp["answers"] and resp["answers"][k]["type"] == "choice"]
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite")
    ap.add_argument("--edition", default="0.3")
    ap.add_argument("--vision")
    ap.add_argument("--per_bench", type=int, default=1)
    ap.add_argument("--max_chars", type=int, default=5000)
    ap.add_argument("--repeat", type=int, default=4, help="first K requests run twice (determinism)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--chunk_mode", default="tournament")
    ap.add_argument("--out", default=os.path.join(REPO, "di/smoke_results.jsonl"))
    ap.add_argument("--option", action="append", default=[],
                    help="extra GlanceEngine option key=value, e.g. ckpt=checkpoints/v1/ckpt_final.pt "
                         "temps=outputs/v1/eval_results.json (default weights: the HF release)")
    a = ap.parse_args()
    from decision_index.engines import validate

    t = time.perf_counter()
    eng = GlanceEngine(device=a.device, dtype=a.dtype, chunk_mode=a.chunk_mode, image_root=a.vision,
                       **dict(o.split("=", 1) for o in a.option))
    print(json.dumps({"loaded_s": round(time.perf_counter() - t, 1), "device": eng.device, "dtype": eng.dtype}), flush=True)

    jobs = []
    if a.suite:
        must = {"BANKING77", "CLINC150+OOS", "ChessBench", "POP909", "API-Bank"}
        for b, rows in sorted(pick_text(a.suite, a.edition, a.per_bench, a.max_chars, must).items()):
            jobs += [("text", b, r) for r in rows]
    vis = pick_vision(a.vision, a.per_bench) if a.vision else {}
    for b, rows in sorted(vis.items()):
        jobs += [("vision", b, r) for r in rows]
    syn = {"intent": {"type": "choice", "instructions": "Classify the intent of the customer message in state.",
                      "criteria": {f"intent_{i:02d}": f"label {i}" for i in range(77)}}}
    syn["intent"]["criteria"]["intent_42"] = "the customer wants to know where their card is (card arrival)"
    jobs.append(("synthetic", "synthetic-77", {"state": "I ordered my card two weeks ago and it still has not arrived.",
                                                "questions": syn, "expected": {"intent": "intent_42"}}))

    first_img = next((os.path.join(a.vision, r["images"][0]) for rows in vis.values() for r in rows), None)
    ok_parity = parity(eng, first_img)

    per = collections.defaultdict(lambda: dict(n=0, ok=0, unsupported=0, error=0, hits=[], ms=[], chunked=0))
    det = []
    with open(a.out, "w") as f:
        for i, (kind, b, row) in enumerate(jobs):
            kw = dict(images=row.get("images", ()), prompt=row.get("prompt")) if kind == "vision" else {}
            s = per[b]
            s["n"] += 1
            t = time.perf_counter()
            try:
                resp, raw = eng(row["state"], row["questions"], **kw)
                validate(row["questions"], resp)
                s["ok"] += 1
                s["chunked"] += bool(raw["chunked"])
                s["hits"] += correct(row, resp)
                if i < a.repeat or raw["chunked"]:
                    resp2, _ = eng(row["state"], row["questions"], **kw)
                    same = all(resp["answers"][k].get("choice") == resp2["answers"][k].get("choice") for k in resp["answers"])
                    dmax = max(abs(p - resp2["answers"][k]["probabilities"][o]) if resp["answers"][k]["type"] == "choice"
                               else abs(resp["answers"][k]["noul"] - resp2["answers"][k]["noul"])
                               for k in resp["answers"] for o, p in (resp["answers"][k].get("probabilities") or {"_": 0}).items())
                    det.append(dict(bench=b, same_answers=same, max_prob_diff=dmax))
                rec = dict(kind=kind, bench=b, status="ok", raw=raw,
                           answers={k: (v.get("choice"), round(max((v.get("probabilities") or {"": v.get("noul", 0)}).values()), 4))
                                    for k, v in resp["answers"].items()}, expected=row.get("expected"))
            except Unsupported as e:
                s["unsupported"] += 1
                rec = dict(kind=kind, bench=b, status="unsupported", error=str(e))
            except Exception as e:  # noqa: BLE001
                s["error"] += 1
                rec = dict(kind=kind, bench=b, status="error", error=f"{type(e).__name__}: {e}")
            ms = (time.perf_counter() - t) * 1000
            s["ms"].append(ms)
            rec["ms"] = round(ms, 1)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(json.dumps({"i": i, "bench": b, "status": rec["status"], "ms": round(ms), "chunked": rec.get("raw", {}).get("chunked"),
                              "prefix": rec.get("raw", {}).get("prefix_tokens")}), flush=True)
    summary = {b: dict(n=s["n"], ok=s["ok"], unsupported=s["unsupported"], error=s["error"], chunked=s["chunked"],
                       acc=(sum(s["hits"]) / len(s["hits"]) if s["hits"] else None), n_scored_fields=len(s["hits"]),
                       ms_mean=round(sum(s["ms"]) / len(s["ms"]))) for b, s in sorted(per.items())}
    out = dict(parity=ok_parity, determinism=dict(n=len(det), all_same=all(d["same_answers"] for d in det),
                                                   max_prob_diff=max([d["max_prob_diff"] for d in det] or [0])),
               totals=dict(requests=sum(s["n"] for s in per.values()), ok=sum(s["ok"] for s in per.values()),
                           errors=sum(s["error"] for s in per.values()), unsupported=sum(s["unsupported"] for s in per.values())),
               per_bench=summary)
    json.dump(out, open(a.out.replace(".jsonl", "_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
