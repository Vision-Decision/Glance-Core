import json, sys, math, collections
d = json.load(open(sys.argv[1])); tau = float(sys.argv[3])
def auroc(pos, neg):
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return round(s / (len(pos) * len(neg)), 3) if pos and neg else None
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * p, 1), max(0.0, round(100 * (c - h), 1)), round(100 * (c + h), 1)]
med = lambda x: round(sorted(x)[len(x) // 2], 3) if x else None
rows = [dict(q, salience=it['salience'], cat=it['cat'], image_id=it['image_id']) for it in d['items'] for q in it['questions']]
out = dict(tau_dial=tau)
def block(R, label):
    pos = [r for r in R if r['present']]; neg = [r for r in R if not r['present']]
    o = dict(n_pos=len(pos), n_neg=len(neg), p_pos_med=med([r['p'] for r in pos]), p_neg_med=med([r['p'] for r in neg]),
             FA_0_5=wilson(sum(r['p'] >= 0.5 for r in neg), len(neg)), FA_dial=wilson(sum(r['p'] >= tau for r in neg), len(neg)),
             FR_0_5_pos=wilson(sum(r['p'] < 0.5 for r in pos), len(pos)),
             false_abs_lt_0_1_pos=wilson(sum(r['p'] < 0.1 for r in pos), len(pos)),
             neg_boxed_on_object=wilson(sum(r['p'] >= 0.5 and r['iou_gt'] >= 0.5 for r in neg), len(neg)),
             conf_neg_mean=round(sum(r['conf'] for r in neg) / max(1, len(neg)), 3),
             auroc_p=auroc([r['p'] for r in pos], [r['p'] for r in neg]),
             auroc_conf=auroc([r['conf'] for r in pos], [r['conf'] for r in neg]))
    return o
bare = [r for r in rows if r['type'] == 'bare']
for t in ('color', 'size', 'position', 'count', 'action'):
    R = [r for r in rows if r['type'] == t]
    if t == 'count': R = R + bare  # correct = the bare category name (single instance)
    out[t] = block(R, t)
    # paired: per image, is max P(swapped) > P(correct)?
    by = collections.defaultdict(lambda: dict(pos=[], neg=[]))
    for r in R: by[r['image_id']]['pos' if r['present'] else 'neg'].append(r['p'])
    pr = [max(v['neg']) >= min(v['pos']) for v in by.values() if v['pos'] and v['neg']]
    out[t]['paired_swap_ge_correct'] = round(sum(pr) / len(pr), 3)
out['bare'] = dict(n=len(bare), p_med=med([r['p'] for r in bare]), FR_0_5=wilson(sum(r['p'] < 0.5 for r in bare), len(bare)))
out['color_by_swap'] = {s: block([r for r in rows if r['type'] == 'color' and (r['present'] or r.get('swap') == s)], s) for s in ('contrast', 'scene_color')}
out['by_salience'] = {}
for sal in ('small', 'medium', 'large'):
    out['by_salience'][sal] = {t: block([r for r in rows if r['salience'] == sal and (r['type'] == t or (t == 'count' and r['type'] == 'bare'))], t)
                               for t in ('color', 'count', 'position', 'action', 'size')}
# vertical vs horizontal position
for ax, words in (('horizontal', ('left', 'right')), ('vertical', ('top', 'bottom'))):
    out[f'position_{ax}'] = block([r for r in rows if r['type'] == 'position' and r['text'].split()[-1] in words], ax)
cc = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r['type'] == 'count': cc[r['cat']][0] += r['p'] >= 0.5; cc[r['cat']][1] += 1
out['count_FA_by_category'] = {k: f'{v[0]}/{v[1]}' for k, v in sorted(cc.items(), key=lambda kv: -kv[1][0] / kv[1][1])}
out['all_swaps'] = block([r for r in rows if r['type'] != 'bare' or True], 'all')
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for k, v in out.items():
    if k in ('by_salience', 'color_by_swap'):
        for kk, vv in v.items(): print(k, kk, vv)
    else: print(k, v)
