"""Build release/v1.1/results.json (+ the benchmark tables used in README.md) from the eval JSONs.

python3 scripts/v1/make_card.py --v11 outputs/v1.1/eval_results.json --v1 outputs/v1/eval_results.json \
    --v0 outputs/v0/eval_results.json --base outputs/v1/eval_base_qwen35_4b.json --out release/v1.1
Prints the markdown tables; writes results.json (HF model-index style) and tables.md.
"""
import argparse, json, os

REC = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcocoplus_val", "refcocoplus_testA", "refcocoplus_testB",
       "refcocog_val", "refcocog_test"]
REC_NAMES = {"refcoco_val": ("RefCOCO", "val"), "refcoco_testA": ("RefCOCO", "testA"), "refcoco_testB": ("RefCOCO", "testB"),
             "refcocoplus_val": ("RefCOCO+", "val"), "refcocoplus_testA": ("RefCOCO+", "testA"),
             "refcocoplus_testB": ("RefCOCO+", "testB"), "refcocog_val": ("RefCOCOg", "val"), "refcocog_test": ("RefCOCOg", "test")}


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}"


def num(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def get(d, *ks):
    for k in ks:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v11", required=True); ap.add_argument("--v1", required=True); ap.add_argument("--v0", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base_gui", default="", help="json {ss2: acc, sspro: acc, ...} for the generative base (other agent)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = {k: json.load(open(getattr(a, k))) for k in ["v11", "v1", "v0", "base"]}
    bg = json.load(open(a.base_gui)) if a.base_gui and os.path.exists(a.base_gui) else {}
    L = []
    # ---- REC
    L.append("| Benchmark | split | v1.1 | v1 | v0 |\n|---|---|---|---|---|")
    for k in REC:
        n, sp = REC_NAMES[k]
        L.append(f"| {n} | {sp} | {pct(get(E['v11'], 'sets', k, 'acc50'))} | {pct(get(E['v1'], 'sets', k, 'acc50'))} | "
                 f"{pct(get(E['v0'], 'sets', k, 'acc50'))} |")
    L.append("")
    L.append("| RefCOCO-family box-confidence ECE (P(present)·P(IoU≥0.5) vs. hit) | v1.1 | v1 | v0 |\n|---|---|---|---|")
    for k in REC:
        n, sp = REC_NAMES[k]
        L.append(f"| {n} {sp} | {num(get(E['v11'], 'sets', k, 'ece_box_conf'))} | {num(get(E['v1'], 'sets', k, 'ece_box_conf'))} | "
                 f"{num(get(E['v0'], 'sets', k, 'ece_box_conf'))} |")
    L.append("")
    g = "grefcoco_val_merged"
    L.append("| gRefCOCO val (+RefCOCO val single-target) | v1.1 | v1 | v0 |\n|---|---|---|---|")
    for key, lab, f in [("N_acc", "N-acc (no-target, P(present)<0.5)", pct), ("T_acc_single", "T-acc single-target (P≥0.5 & IoU≥0.5)", pct),
                        ("auroc_presence", "AUROC of P(present)", num), ("ece_presence", "ECE of P(present)", num),
                        ("ece_box_conf", "ECE of box confidence", num)]:
        L.append(f"| {lab} | {f(get(E['v11'], 'sets', g, 'calibrated', key) or get(E['v11'], 'sets', g, key))} | "
                 f"{f(get(E['v1'], 'sets', g, 'calibrated', key) or get(E['v1'], 'sets', g, key))} | {f(get(E['v0'], 'sets', g, key))} |")
    L.append("")
    # ---- GUI
    L.append("| GUI grounding (click = predicted box center in GT box) | v1.1 single pass | v1.1 + crop-zoom | v1 | base Qwen3.5-4B (generative) |\n|---|---|---|---|---|")
    for s, lab in [("ss2", "ScreenSpot-v2"), ("sspro", "ScreenSpot-Pro")]:
        L.append(f"| {lab} | {pct(get(E['v11'], 'sets', s, 'raw', 'click_acc'))} | {pct(get(E['v11'], 'sets', s, 'zoom', 'click_acc'))} | "
                 f"{pct(get(E['v1'], 'sets', s, 'raw', 'click_acc'))} | {pct(bg.get(s))} |")
    vo = get(E["v11"], "sets", "venus", "official_protocol") or {}
    L.append(f"| VenusBench-GD grounding (positive items) | {pct(get(E['v11'], 'sets', 'venus', 'raw', 'click_acc'))} | "
             f"{pct(get(E['v11'], 'sets', 'venus', 'zoom', 'click_acc'))} | {pct(get(E['v1'], 'sets', 'venus', 'raw', 'click_acc'))} | {pct(bg.get('venus'))} |")
    L.append(f"| VenusBench-GD refusal (P(present)<0.5) | {pct(get(E['v11'], 'sets', 'venus', 'raw', 'refusal_acc'))} | – | "
             f"{pct(get(E['v1'], 'sets', 'venus', 'raw', 'refusal_acc'))} | – |")
    L.append("")
    if vo:
        L.append("VenusBench-GD with the official scorer (inclusionAI/UI-Venus@VenusBench-GD `eval.py`: point-in-box for "
                 "positives, `result==\"negative\"` for refusals, micro-average):\n")
        L.append("| subtask | v1.1 single pass | v1.1 + crop-zoom |\n|---|---|---|")
        zo = get(E["v11"], "sets", "venus", "zoom_official_protocol") or {}
        for k, v in vo.items():
            if isinstance(v, dict):
                L.append(f"| {k} (n={v['n']}) | {pct(v['acc'])} | {pct(get(zo, k, 'acc'))} |")
        L.append("")
    # ---- OpenDecisions
    def od(e, m):
        return get(e, "sets", "opendecisions", m) or {}
    L.append("| OpenDecisions (official score.py) | Capability | Intelligence | Calibration | acc choice / noul / score | TVD (129 calib. items) |\n|---|---|---|---|---|---|")
    rows = [("v1.1 (choice T from difficulty-matched held-out split)", od(E["v11"], "cal_choice_hard")),
            ("v1.1 (per-type T, general held-out split)", od(E["v11"], "cal")), ("v1.1 (no temperature)", od(E["v11"], "raw")),
            ("v1 (per-type T)", od(E["v1"], "cal")), ("base Qwen3.5-4B, same prompt, per-type T", od(E["base"], "cal")),
            ("base Qwen3.5-4B, same prompt, no T", od(E["base"], "raw"))]
    for lab, r in rows:
        if not r:
            continue
        ac = r.get("accuracy_per_type", {})
        L.append(f"| {lab} | {num(r.get('capability'), 1)} | {num(r.get('intelligence'), 1)} | {num(r.get('calibration'), 1)} | "
                 f"{num(ac.get('choice'), 1)} / {num(ac.get('noul'), 1)} / {num(ac.get('score'), 1)} | {num(r.get('calibration_items_tvd'), 3)} |")
    L.append("")
    lat = E["v11"].get("latency", {})
    L.append(f"Latency ({lat.get('gpu', '?')}, batch 1, bf16, CUDA-synchronized model time, median / p90 ms): "
             f"image + 1 box question {num(get(lat, 'box_1q', 'ms_median'), 1)} / {num(get(lat, 'box_1q', 'ms_p90'), 1)}; "
             f"image + 8 questions {num(get(lat, 'box_8q', 'ms_median'), 1)} / {num(get(lat, 'box_8q', 'ms_p90'), 1)}; "
             f"text state + 1 typed question {num(get(lat, 'text_1q', 'ms_median'), 1)} / {num(get(lat, 'text_1q', 'ms_p90'), 1)} "
             f"(COCO images at ≤640 visual tokens; OpenDecisions states).")
    md = "\n".join(L)
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/tables.md", "w").write(md + "\n")
    # ---- model-index style results
    res = []
    def add(task, ds, split, metric, val):
        if val is not None:
            res.append(dict(task=dict(type=task), dataset=dict(name=ds, split=split), metrics=[dict(type=metric, value=round(100 * val, 2) if metric != "ece" and metric != "auroc" else round(val, 4))]))
    for k in REC:
        n, sp = REC_NAMES[k]
        add("object-detection", n, sp, "acc@0.5", get(E["v11"], "sets", k, "acc50"))
    gg = get(E["v11"], "sets", g, "calibrated") or get(E["v11"], "sets", g) or {}
    add("object-detection", "gRefCOCO", "val", "N-acc", gg.get("N_acc"))
    add("object-detection", "gRefCOCO", "val", "auroc", gg.get("auroc_presence"))
    add("object-detection", "gRefCOCO", "val", "ece", gg.get("ece_presence"))
    for s, n in [("ss2", "ScreenSpot-v2"), ("sspro", "ScreenSpot-Pro")]:
        add("gui-grounding", n, "test", "accuracy", get(E["v11"], "sets", s, "raw", "click_acc"))
        add("gui-grounding", n + " (crop-zoom 2nd pass)", "test", "accuracy", get(E["v11"], "sets", s, "zoom", "click_acc"))
    add("gui-grounding", "VenusBench-GD (positive items)", "test", "accuracy", get(E["v11"], "sets", "venus", "raw", "click_acc"))
    add("gui-grounding", "VenusBench-GD refusal", "test", "accuracy", get(E["v11"], "sets", "venus", "raw", "refusal_acc"))
    add("gui-grounding", "VenusBench-GD (official scorer, overall)", "test", "accuracy", get(vo, "overall", "acc"))
    best = od(E["v11"], "cal_choice_hard") or od(E["v11"], "cal")
    for m in ["capability", "intelligence", "calibration"]:
        if best.get(m) is not None:
            res.append(dict(task=dict(type="multiple-choice"), dataset=dict(name="OpenDecisions", split="test"),
                            metrics=[dict(type=m, value=round(best[m], 2))]))
    json.dump(dict(model_index=[dict(name="decision-vision-v1.1-4b", results=res)], latency_ms=lat,
                   raw=dict(v11=E["v11"]["sets"], temps=E["v11"].get("temps"), temps_choice_hard=E["v11"].get("temps_choice_hard"))),
              open(f"{a.out}/results.json", "w"), indent=1, default=str)
    print(md)


if __name__ == "__main__":
    main()
