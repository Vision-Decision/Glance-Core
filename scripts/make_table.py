"""Print a markdown comparison table from eval_results.json (+ baseline_results.json).
python scripts/make_table.py outputs/v0/eval_results.json outputs/v0/baseline_results.json"""
import json, sys

ev = json.load(open(sys.argv[1]))
bl = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
f = lambda x: "-" if x is None else (f"{100*x:.1f}" if isinstance(x, float) and x <= 1.0 else f"{x:.1f}")
print("| set | n | Acc@0.5 | gated Acc@0.5 | mIoU | N-acc | T-acc(single) | AUROC P(present) | ECE P(present) | ECE box-conf |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, s in ev["sets"].items():
    print(f"| {k} | {s['n']} | {f(s.get('acc50'))} | {f(s.get('gated_acc50'))} | {f(s.get('mean_iou'))} | {f(s.get('N_acc'))} | "
          f"{f(s.get('T_acc_single'))} | {f(s.get('auroc_presence'))} | {f(s.get('ece_presence'))} | {f(s.get('ece_box_conf'))} |")
if "latency" in ev:
    L = ev["latency"]
    print(f"\nLatency ({L.get('gpu')}), batch 1, model-only median ms: 1q {L['1q']['model_ms_median']:.1f}, "
          f"8q {L['8q']['model_ms_median']:.1f} (preprocess {L['1q']['preprocess_ms_median']:.1f} ms)")
if bl:
    print("\nBaseline", bl["model"], "(generative, 0-1000 bbox or 'none')")
    for k, s in bl["sets"].items():
        print(f"- {k}: n={s['n']} acc50={f(s.get('acc50'))} N_acc={f(s.get('N_acc'))} none_rate={f(s['none_rate'])} "
              f"latency_1q_median={s['latency_ms_median_1q']:.0f} ms")
    print(f"- 8 questions on one image (8 sequential decodes): {bl['latency_ms_median_8q_same_image']:.0f} ms")
