import json, sys, collections, math
O = sys.argv[1]
b = json.load(open(O + '/base_qwen35_4b_generative.json'))
g = json.load(open(O + '/B_attribute_glance_v1.json'))
ga = json.load(open(O + '/A_ambiguity_glance_v1.json'))
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * p, 1), max(0.0, round(100 * (c - h), 1)), round(100 * (c + h), 1)]
gl = {(it['image_id'], q['text']): q for it in g['items'] for q in it['questions']}
out = {'B': {}, 'A': {}}
for t in ('color', 'size', 'position', 'count', 'action', 'bare'):
    R = [r for r in b['B'] if r['type'] == t]
    pos = [r for r in R if r['present']]; neg = [r for r in R if not r['present']]
    G = [gl[(r['image_id'], r['text'])] for r in R]
    gpos = [q for q in G if q['present']]; gneg = [q for q in G if not q['present']]
    out['B'][t] = dict(n_pos=len(pos), n_neg=len(neg),
        base_FA=wilson(sum(not r['none'] for r in neg), len(neg)), glance_FA_0_5=wilson(sum(q['p'] >= 0.5 for q in gneg), len(gneg)),
        base_FR=wilson(sum(r['none'] for r in pos), len(pos)), glance_FR_0_5=wilson(sum(q['p'] < 0.5 for q in gpos), len(gpos)),
        base_neg_boxed_on_object=wilson(sum((not r['none']) and (r['iou_gt'] or 0) >= 0.5 for r in neg), len(neg)),
        base_pos_acc50=wilson(sum((r['iou_gt'] or 0) >= 0.5 for r in pos), len(pos)), glance_pos_acc50_gated=wilson(sum(q['p'] >= 0.5 and q['iou_gt'] >= 0.5 for q in gpos), len(gpos)))
    print('B', t, out['B'][t])
gaa = {(i['image_id'], q['text']): (i, q) for i in ga['items'] for q in i['questions'] if 'iou_inst' in q}
for kind in ('single', 'multi'):
    for tag in ('bare_the', 'plural'):
        R = [r for r in b['A'] if r['kind'] == kind and r['tag'] == tag]
        G = [gaa[(r['image_id'], r['text'])][1] for r in R]
        o = dict(n=len(R), base_none=wilson(sum(r['none'] for r in R), len(R)),
                 base_hit_any=wilson(sum(bool(r['iou_inst']) and max(r['iou_inst']) >= 0.5 for r in R), len(R)),
                 base_iou_union_ge_0_5=wilson(sum((r['iou_union'] or 0) >= 0.5 for r in R), len(R)),
                 glance_hit_any=wilson(sum(max(q['iou_inst']) >= 0.5 for q in G), len(G)),
                 glance_iou_union_ge_0_5=wilson(sum(q['iou_union'] >= 0.5 for q in G), len(G)),
                 glance_p_lt_0_5=wilson(sum(q['p'] < 0.5 for q in G), len(G)))
        out['A'][f'{kind}/{tag}'] = o; print('A', kind, tag, o)
json.dump(out, open(O + '/base_vs_glance_summary.json', 'w'), indent=1)
