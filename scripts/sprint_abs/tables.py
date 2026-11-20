"""results.json (analyze.py) -> markdown tables.  python scripts/sprint_abs/tables.py outputs/sprint_abs/v1/results.json"""
import json, sys


def f(x, k=1, pct=True):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{100 * x:.{k}f}" if pct else f"{x:.{k}f}"


def ci(d, k=1):
    return f"{f(d['est'], k)} [{f(d['lo'], k)}, {f(d['hi'], k)}]"


def main(p):
    R = json.load(open(p))
    L = []
    L.append(f"# sprint_abs tables ({R['tag']})\n")
    # GREC
    if R.get("grec"):
        L.append("## GREC official protocol (gRefCOCO; Pr@(F1=1, GIoU>=0.5), N-acc, T-acc)\n")
        L.append("| split | variant | Pr@F1=1 | 95% CI | N-acc | T-acc | Pr single | Pr multi | Pr (plain IoU) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for sp, d in R["grec"].items():
            for v, m in d.items():
                if not isinstance(m, dict) or "pr_f1" not in m:
                    continue
                L.append(f"| {sp} | {v} | {f(m['pr_f1'])} | {ci(m['pr_f1_ci']) if 'pr_f1_ci' in m else '-'} | {f(m['N_acc'])} | "
                         f"{f(m['T_acc'])} | {f(m['pr_single'])} | {f(m['pr_multi'])} | {f(m.get('pr_f1_plainIoU'))} |")
            if "paired_multi_minus_single_official" in d:
                L.append(f"| {sp} | paired diff multi-single (official) | {ci(d['paired_multi_minus_single_official'])} | | | | | | |")
        L.append(f"\nTuned on grefcalib: {json.dumps(R.get('grec_tuned_on_grefcalib'))}; extra-box model: {json.dumps(R.get('grec_extra_model'))}\n")
    # Venus official
    if R.get("venus_official"):
        L.append("## VenusBench-GD, official UI-Venus evaluator\n")
        tasks = ["element_grounding", "visual_grounding", "spatial_grounding", "reason_grounding", "functional_grounding", "refusal_grounding"]
        L.append("| config | Element | Visual | Spatial | Basic | Reasoning | Functional | Refusal | Advanced | Overall | n |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for cfg, t in R["venus_official"].items():
            o = t["overall"]
            c = [f(t[k]["acc"], 2) if k in t else "-" for k in tasks]
            L.append(f"| {cfg} | {c[0]} | {c[1]} | {c[2]} | {f(o['basic'], 2)} | {c[3]} | {c[4]} | {c[5]} | {f(o['advanced'], 2)} | {f(o['acc'], 2)} | {o['n']} |")
        L.append("")
    # presence
    L.append("## Presence ĝ calibration (fit on held-out training-distribution splits)\n")
    L.append("| set | subset | method | AUROC | N-acc@.5 | recall@.5 | ECE15 | ECE15-mass | ECE10 | Brier | NLL | +EM prior: ECE15 / N-acc / π̂ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, d in R["presence"].items():
        for sub, t in d.items():
            for m, x in t.items():
                em = x.get("em_prior")
                ems = f"{f(em['ece15'])} / {f(em.get('N_acc'))} / {em['pi_hat']:.3f}" if em else "-"
                L.append(f"| {name} | {sub} | {m} | {f(x['auroc'])} | {f(x.get('N_acc'))} | {f(x.get('recall_present'))} | {f(x['ece15'])} | "
                         f"{f(x['ece15_mass'])} | {f(x['ece10'])} | {f(x['brier'], 2)} | {x['nll']:.3f} | {ems} |")
    # kappa
    L.append("\n## κ̂ calibration (natural: event IoU>=0.5; GUI: click in box)\n")
    L.append("| set | fit | method | AUROC | AURC | ECE15 | ECE15-mass | Brier | mean κ̂ | rate |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, d in R["kappa"].items():
        items = d.items() if "raw" not in d else [("natural_fit", d)]
        for src, t in items:
            for m, x in t.items():
                L.append(f"| {name} | {src} | {m} | {f(x['auroc'])} | {f(x['aurc'], 2)} | {f(x['ece15'])} | {f(x['ece15_mass'])} | "
                         f"{f(x['brier'], 2)} | {f(x['mean_p'])} | {f(x['rate'])} |")
    # box conf
    L.append("\n## Box confidence ĝ·κ̂ vs (present ∧ IoU>=0.5)  [the v0→v1 regression metric]\n")
    L.append("| set | method | ECE15 | ECE15-mass | Brier | π̂ (EM) |")
    L.append("|---|---|---|---|---|---|")
    for name, d in R["box_conf"].items():
        for m, x in d.items():
            L.append(f"| {name} | {m} | {f(x['ece15'], 2)} | {f(x['ece15_mass'], 2)} | {f(x['brier'], 2)} | {x.get('pi_hat', '-') if 'pi_hat' not in x else round(x['pi_hat'], 3)} |")
    if R.get("boot_box_conf"):
        L.append("\nBootstrap (image clusters): ")
        for k, v in R["boot_box_conf"].items():
            L.append(f"- {k}: raw ECE {ci(v['raw'], 2)}; fixed {ci(v['fixed'], 2)}; paired diff {ci(v['paired'], 2)}; κ̂ AUROC {ci(v['kappa_auroc'])}")
    if R.get("boot_presence"):
        L.append("\n## Presence bootstrap CIs (image clusters, raw ĝ unless noted)\n")
        for k, v in R["boot_presence"].items():
            L.append(f"- {k}: AUROC {ci(v['auroc'])}; N-acc {ci(v['N_acc'])}; ECE raw {ci(v['ece_raw'], 2)} → Platt {ci(v['ece_platt'], 2)} "
                     f"(paired {ci(v['ece_paired_platt_minus_raw'], 2)}); subset ECE raw {ci(v['ece_sub_raw'], 2)} → Platt {ci(v['ece_sub_platt'], 2)}")
    # risk control
    L.append("\n## Risk-controlled accept/abstain (δ=0.1; 50/50 image splits ×reps; coverage = fraction answered)\n")
    for name, d in R["risk_control"].items():
        L.append(f"\n**{name}** (n={d['n']}, absent {f(d['absent_rate'])}%, correct-if-answered-all {f(d['base_correct'])}%); default ĝ≥.5: "
                 f"coverage {f(d['default_rule']['coverage'])}, selective risk {f(d['default_rule']['selective_risk'])}, "
                 f"false-accept on absent {f(d['default_rule']['false_accept_absent'])}\n")
        L.append("| rule | α | coverage | realized risk | violation rate |")
        L.append("|---|---|---|---|---|")
        for k, v in d.items():
            if k.startswith(("selective", "joint")) and isinstance(v, dict):
                for a, x in v.items():
                    viol = x.get("violation_rate", x.get("violated"))
                    L.append(f"| {k} | {a} | {f(x['coverage'])} | {f(x['risk'], 2)} | {viol} |")
            if k.startswith("absent_conformal"):
                for a, x in v.items():
                    L.append(f"| {k} | {a} | present accepted {f(x['accept_present'])} | false-accept absent {f(x.get('false_accept_absent'), 2)} | |")
        L.append(f"\nAURC ĝ {f(d.get('aurc[g]'), 2)} vs ĝκ̂ {f(d.get('aurc[g*k]'), 2)}; paired diff {ci(d['aurc_paired_gk_minus_g'], 2) if 'aurc_paired_gk_minus_g' in d else '-'}")
    # prop 1
    L.append("\n## Prop. 1 two-knob plug-in rule vs one-knob thresholds (expected cost per query, lower is better)\n")
    L.append("| set | costs | two-knob plug-in | ĝ≥.5 | ĝ thr tuned on calib | ĝκ̂ thr tuned | ĝ oracle thr | paired two−one(tuned) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, d in R["prop1"].items():
        for c, e in d.items():
            pd = e.get("paired_two_minus_one_tuned")
            ps = f"{pd['est']:.4f} [{pd['lo']:.4f}, {pd['hi']:.4f}]" if pd else "-"
            L.append(f"| {name} | {c} | {e['two_knob_plugin']:.4f} | {e['one_knob_default']:.4f} | {e['one_knob_tuned']:.4f} | "
                     f"{e['product_tuned']:.4f} | {e['one_knob_oracle']:.4f} | {ps} |")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
