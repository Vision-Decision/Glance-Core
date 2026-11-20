import json, statistics as st, sys
d = json.load(open(sys.argv[1]))
def med(x): return round(st.median(x), 3) if x else None
def mean(x): return round(sum(x) / len(x), 3) if x else None
def auroc(pos, neg):
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return round(s / (len(pos) * len(neg)), 3)
out = {}
for kind in ('single', 'multi'):
    its = [i for i in d['items'] if i['kind'] == kind]
    for tag in ('bare_the', 'bare_a', 'plural', 'leftmost', 'rightmost', 'biggest'):
        rows = [(i, q) for i in its for q in i['questions'] if q['tag'] == tag]
        if not rows: continue
        P = [q['p'] for _, q in rows]; K = [q['kappa'] for _, q in rows]
        best = [max(q['iou_inst']) for _, q in rows]
        hit_any = mean([b >= 0.5 for b in best])
        hit_gt = mean([q['iou_inst'][q['gt']] >= 0.5 for _, q in rows]) if isinstance(rows[0][1]['gt'], int) else None
        un = mean([q['iou_union'] >= 0.5 for _, q in rows])
        H = [q['center']['entropy_norm'] for _, q in rows]
        nm = [q['center']['n_inst_mass_ge_0.1'] for _, q in rows]
        split = [q['center']['split_ratio'] for _, q in rows]
        # box spans >=2 instances: box covers >=50% of the area of 2+ GT boxes
        def covers(b, g):
            ix = max(0, min(b[2], g[2]) - max(b[0], g[0])); iy = max(0, min(b[3], g[3]) - max(b[1], g[1]))
            return ix * iy / ((g[2] - g[0]) * (g[3] - g[1]))
        span = mean([sum(covers(q['box'], g) >= 0.5 for g in i['gt_boxes']) >= 2 for i, q in rows]) if kind == 'multi' else None
        # which instance: largest? leftmost?
        largest = mean([max(range(i['n_inst']), key=lambda k: q['iou_inst'][k]) == max(range(i['n_inst']), key=lambda k: i['area_frac'][k]) for i, q in rows if max(q['iou_inst']) >= 0.5]) if kind == 'multi' else None
        out[f'{kind}/{tag}'] = dict(n=len(rows), p_med=med(P), p_min=round(min(P), 3), frac_p_lt_0_5=mean([p < 0.5 for p in P]),
                                   kappa_mean=mean(K), conf_mean=mean([q['conf'] for _, q in rows]),
                                   hit_any_instance=hit_any, hit_named_gt=hit_gt, iou_union_ge_0_5=un, box_spans_2plus=span,
                                   picks_largest_when_hit=largest,
                                   H_med=med(H), n_inst_mass_ge_0_1_mean=mean(nm), split_ratio_med=med(split),
                                   frac_mass_split=mean([s >= 0.3 for s in split]) if kind == 'multi' else None)
for k, v in out.items(): print(k, v)
# kappa calibration on bare_the: kappa vs hit any
for kind in ('single', 'multi'):
    rows = [q for i in d['items'] if i['kind'] == kind for q in i['questions'] if q['tag'] == 'bare_the']
    print(kind, 'kappa | hit', [(round(q['kappa'], 2), max(q['iou_inst']) >= 0.5) for q in rows][:50])
# count/more-than-one
for kind in ('single', 'multi'):
    its = [i for i in d['items'] if i['kind'] == kind]
    cnt = [(i['n_inst'], [q for q in i['questions'] if q['tag'] == 'count'][0]) for i in its]
    acc = mean([q['answer'] == str(n) for n, q in cnt])
    acc2 = mean([q['answer'] == str(n) for n, q in cnt if n <= 3])
    mto = [[q for q in i['questions'] if q['tag'] == 'more_than_one'][0]['p_yes'] for i in its]
    print(kind, 'count acc', acc, 'acc(n<=3)', acc2, 'P(more than one) med', med(mto), 'frac>0.5', mean([x > 0.5 for x in mto]))
    out[f'{kind}/count'] = dict(acc=acc, acc_n_le_3=acc2, p_more_than_one_med=med(mto), frac_more_than_one_gt_0_5=mean([x > 0.5 for x in mto]))
# ambiguity detectors: single vs multi on bare_the
S1 = [i for i in d['items'] if i['kind'] == 'single']; M = [i for i in d['items'] if i['kind'] == 'multi']
g = lambda i, tag: [q for q in i['questions'] if q['tag'] == tag][0]
dets = {'center_entropy_norm': lambda i: g(i, 'bare_the')['center']['entropy_norm'],
        'neg_top1': lambda i: -g(i, 'bare_the')['center']['top1'],
        'one_minus_kappa': lambda i: 1 - g(i, 'bare_the')['kappa'],
        'one_minus_p': lambda i: 1 - g(i, 'bare_the')['p'],
        'center_mass_out_pred_box_1.5x': lambda i: g(i, 'bare_the')['center']['mass_out_pred15'],
        'center_n_peaks': lambda i: g(i, 'bare_the')['center']['n_peaks'],
        'center_peak2_over_peak1': lambda i: g(i, 'bare_the')['center']['peak2_over_peak1'],
        'P_more_than_one': lambda i: g(i, 'more_than_one')['p_yes'],
        'P_count_ge_2': lambda i: 1 - sum(v for k, v in g(i, 'count')['probs'].items() if k in ('0', '1'))}
out['ambiguity_auroc_multi_vs_single'] = {k: auroc([f(i) for i in M], [f(i) for i in S1]) for k, f in dets.items()}
print(out['ambiguity_auroc_multi_vs_single'])
# same detectors on 'a X' and plural phrasing
for tag in ('bare_a','plural'):
    out[f'ambiguity_auroc_{tag}'] = {k: auroc([f2(i) for i in M], [f2(i) for i in S1]) for k, f2 in {
        '1-kappa': lambda i: 1 - g(i, tag)['kappa'], 'mass_out_pred15': lambda i: g(i, tag)['center']['mass_out_pred15'],
        'n_peaks': lambda i: g(i, tag)['center']['n_peaks'], 'entropy_norm': lambda i: g(i, tag)['center']['entropy_norm']}.items()}
    print(tag, out[f'ambiguity_auroc_{tag}'])
# multi split by n
for lo, hi in ((2, 2), (3, 6)):
    R = [i for i in M if lo <= i['n_inst'] <= hi]
    q = [g(i, 'bare_the') for i in R]
    out[f'multi_n{lo}-{hi}/bare_the'] = dict(n=len(R), kappa_mean=mean([x['kappa'] for x in q]), hit_any=mean([max(x['iou_inst']) >= 0.5 for x in q]),
        iou_union_ge_0_5=mean([x['iou_union'] >= 0.5 for x in q]), mass_out_pred15_med=med([x['center']['mass_out_pred15'] for x in q]),
        n_peaks_mean=mean([x['center']['n_peaks'] for x in q]), split_med=med([x['center']['split_ratio'] for x in q]))
    print(lo, hi, out[f'multi_n{lo}-{hi}/bare_the'])
print('single mop15 med', med([g(i,'bare_the')['center']['mass_out_pred15'] for i in S1]), 'npk', mean([g(i,'bare_the')['center']['n_peaks'] for i in S1]))
print('single count answers', [(g(i,'count')['answer']) for i in S1])
print('multi count answers', [(i['n_inst'], g(i,'count')['answer']) for i in M])
json.dump(out, open(sys.argv[2], 'w'), indent=1)
