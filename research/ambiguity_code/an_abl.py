"""Dataset-level multi-target stats from existing controlled-ablation rows (gate head, v0 data) on gRefCOCO val merged."""
import json, sys
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    U = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I; return I / U if U > 0 else 0.0
def auroc(pos, neg):
    import bisect
    neg = sorted(neg); s = 0
    for p in pos:
        lo = bisect.bisect_left(neg, p); hi = bisect.bisect_right(neg, p); s += lo + 0.5 * (hi - lo)
    return round(s / (len(pos) * len(neg)), 4)
mean = lambda x: round(sum(x) / len(x), 4)
qs = []
for l in open('data/prep/eval_grefcoco_val_merged.jsonl'):
    r = json.loads(l)
    for q in r['questions']: qs.append((r, q))
out = {}
for run in sys.argv[1:]:
    rows = json.load(open(f'outputs/ablation/{run}/rows.json'))['grefcoco_val_merged']
    assert len(rows) == len(qs) and all(a['n_targets'] == q['n_targets'] for a, (_, q) in zip(rows, qs))
    S, M = [], []
    for a, (r, q) in zip(rows, qs):
        W, H = r['W'], r['H']; b = [a['box'][0]*W, a['box'][1]*H, a['box'][2]*W, a['box'][3]*H] if a['box'] else [0, 0, 0, 0]
        p = a.get('p_present', 1 - a.get('p_none', 0))
        if q['n_targets'] == 1:
            S.append(dict(p=p, k=a.get('p_iou50'), H=a.get('H_center'), hit=iou(b, q['box']) >= 0.5))
        elif q['n_targets'] >= 2:
            G = q['boxes']; U = [min(g[0] for g in G), min(g[1] for g in G), max(g[2] for g in G), max(g[3] for g in G)]
            ious = [iou(b, g) for g in G]
            M.append(dict(p=p, k=a.get('p_iou50'), H=a.get('H_center'), hit_any=max(ious) >= 0.5, hit_union=iou(b, U) >= 0.5))
    o = dict(n_single=len(S), n_multi=len(M), p_ge_0_5_single=mean([x['p'] >= 0.5 for x in S]), p_ge_0_5_multi=mean([x['p'] >= 0.5 for x in M]),
             acc50_single=mean([x['hit'] for x in S]), multi_hit_any_instance=mean([x['hit_any'] for x in M]), multi_hit_union=mean([x['hit_union'] for x in M]),
             multi_neither=mean([not x['hit_any'] and not x['hit_union'] for x in M]))
    if S[0]['k'] is not None:
        o.update(kappa_mean_single=mean([x['k'] for x in S]), kappa_mean_multi=mean([x['k'] for x in M]),
                 auroc_multi_vs_single_1mkappa=auroc([1 - x['k'] for x in M], [1 - x['k'] for x in S]),
                 auroc_multi_vs_single_Hcenter=auroc([x['H'] for x in M], [x['H'] for x in S]))
    out[run] = o; print(run, o)
json.dump(out, open('outputs/probes/ambiguity_attr/grefcoco_multitarget_ablation_rows_summary.json', 'w'), indent=1)
